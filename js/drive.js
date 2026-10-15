// Keeps clips in the user's Google Drive so every device sees the same clips.
// Files live in a "직캠 스테이지" folder: one MP4 per clip (its details in the file's
// description) plus a PNG thumbnail. The app only asks for the drive.file scope, so it can
// see the files it created and nothing else in the Drive.
import { GOOGLE_CLIENT_ID } from './config.js';

const BASE = globalThis.__FANCAM_DRIVE_BASE__ || 'https://www.googleapis.com'; // overridable for tests
const API = BASE + '/drive/v3';
const UPLOAD = BASE + '/upload/drive/v3';
const SCOPE = 'https://www.googleapis.com/auth/drive.file';
const FOLDER_NAME = '직캠 스테이지';
const CHUNK = 16 * 1024 * 1024; // resumable upload chunk (multiple of 256 KiB)
const LS = { clientId: 'fancam-drive-client', token: 'fancam-drive-token', on: 'fancam-drive-on', folder: 'fancam-drive-folder' };

const ls = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch {} },
};

// ---------------------------------------------------------------------------
// Setup and sign-in (Google Identity Services, token model)

export const clientId = () => GOOGLE_CLIENT_ID || ls.get(LS.clientId) || '';
export function setClientId(id) { ls.set(LS.clientId, id ? id.trim() : null); }
export const isEnabled = () => !!clientId() && ls.get(LS.on) === '1';
export function disable() { ls.set(LS.on, null); ls.set(LS.token, null); ls.set(LS.folder, null); }

export function looksLikeClientId(id) {
  return /^[\w-]+\.apps\.googleusercontent\.com$/.test((id || '').trim());
}

function savedToken() {
  try {
    const t = JSON.parse(ls.get(LS.token));
    return t && t.expiresAt > Date.now() + 60000 ? t.accessToken : null;
  } catch { return null; }
}
export const hasToken = () => !!savedToken();

let gisPromise = null;
// Load the Google sign-in script ahead of time: the sign-in popup must open straight from a tap.
export function preload() {
  gisPromise ??= new Promise((resolve, reject) => {
    if (window.google?.accounts?.oauth2) return resolve();
    const s = document.createElement('script');
    s.src = 'https://accounts.google.com/gsi/client';
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => { gisPromise = null; reject(new Error('구글 로그인을 불러오지 못했어요. 인터넷 연결을 확인해 주세요.')); };
    document.head.append(s);
  });
  return gisPromise;
}

// Call from a click/tap. Opens Google's popup (it closes by itself once access was granted before).
export function signIn() {
  return new Promise((resolve, reject) => {
    if (!window.google?.accounts?.oauth2) {
      preload().catch(() => {});
      reject(new Error('구글 로그인을 준비 중이에요. 잠시 후 다시 눌러 주세요.'));
      return;
    }
    const client = google.accounts.oauth2.initTokenClient({
      client_id: clientId(),
      scope: SCOPE,
      callback: res => {
        if (res.error || !res.access_token) { reject(new Error(res.error_description || res.error || '로그인하지 못했어요.')); return; }
        if (!google.accounts.oauth2.hasGrantedAllScopes(res, SCOPE)) {
          reject(new Error('드라이브 접근을 허용해야 동기화할 수 있어요. 체크 상자를 켜고 다시 시도해 주세요.'));
          return;
        }
        ls.set(LS.token, JSON.stringify({ accessToken: res.access_token, expiresAt: Date.now() + (res.expires_in || 3600) * 1000 }));
        ls.set(LS.on, '1');
        resolve();
      },
      error_callback: err => reject(new Error(err?.type === 'popup_closed' ? '로그인 창이 닫혔어요.'
        : err?.type === 'popup_failed_to_open' ? '팝업이 막혔어요. 이 사이트의 팝업을 허용해 주세요.' : '로그인하지 못했어요.')),
    });
    client.requestAccessToken({ prompt: ls.get(LS.on) === '1' ? '' : 'consent' });
  });
}

// ---------------------------------------------------------------------------
// HTTP

class NeedsSignIn extends Error {}
export const isSignInError = err => err instanceof NeedsSignIn;

function authHeader() {
  const t = savedToken();
  if (!t) throw new NeedsSignIn('구글 로그인이 필요해요.');
  return { Authorization: `Bearer ${t}` };
}

async function request(url, opts = {}) {
  let res;
  try {
    res = await fetch(url, { ...opts, headers: { ...authHeader(), ...opts.headers } });
  } catch (err) {
    if (err instanceof NeedsSignIn) throw err;
    throw Object.assign(new Error('인터넷에 연결할 수 없어요.'), { offline: true });
  }
  if (res.status === 401) { ls.set(LS.token, null); throw new NeedsSignIn('구글 로그인이 만료됐어요.'); }
  if (!res.ok) {
    let msg = '';
    try { msg = (await res.json()).error?.message || ''; } catch {}
    if (res.status === 403 && /storage|quota/i.test(msg)) throw new Error('구글 드라이브 용량이 가득 찼어요.');
    if (res.status === 403 && /has not been used|disabled/i.test(msg)) throw new Error('구글 클라우드 콘솔에서 Google Drive API를 "사용"으로 켜 주세요.');
    throw new Error(`구글 드라이브 오류 (${res.status}) ${msg}`.trim());
  }
  return res;
}
const json = async (url, opts) => (await request(url, opts)).json();
const q = s => encodeURIComponent(s);

// ---------------------------------------------------------------------------
// Folder and listing

async function folderId() {
  const cached = ls.get(LS.folder);
  if (cached) return cached;
  const found = await json(`${API}/files?q=${q(
    `mimeType='application/vnd.google-apps.folder' and trashed=false and appProperties has { key='fancamRoot' and value='1' }`,
  )}&fields=files(id)&spaces=drive`);
  let id = found.files[0]?.id;
  if (!id) {
    const made = await json(`${API}/files?fields=id`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: FOLDER_NAME, mimeType: 'application/vnd.google-apps.folder', appProperties: { fancamRoot: '1' } }),
    });
    id = made.id;
  }
  ls.set(LS.folder, id);
  return id;
}

export async function listRemote() {
  let folder = await folderId();
  const files = [];
  let page = '';
  for (;;) {
    let res;
    try {
      res = await json(`${API}/files?q=${q(`'${folder}' in parents and trashed=false`)}`
        + `&fields=${q('nextPageToken,files(id,name,size,description,appProperties)')}&pageSize=1000&spaces=drive${page ? '&pageToken=' + page : ''}`);
    } catch (err) {
      // The folder may have been deleted by hand: forget it and start over once.
      if (ls.get(LS.folder) && /\(404\)/.test(err.message)) { ls.set(LS.folder, null); folder = await folderId(); continue; }
      throw err;
    }
    files.push(...res.files);
    if (!res.nextPageToken) break;
    page = res.nextPageToken;
  }
  const byId = new Map();
  for (const f of files) {
    const id = f.appProperties?.fancamId;
    if (!id) continue;
    if (!byId.has(id)) byId.set(id, { id, video: null, thumb: null });
    const e = byId.get(id);
    if (f.appProperties.kind === 'video') e.video = f;
    if (f.appProperties.kind === 'thumb') e.thumb = f;
  }
  const out = [];
  for (const e of byId.values()) {
    if (!e.video) continue; // the video goes up last; without it the clip isn't finished
    try { e.meta = JSON.parse(e.video.description || '{}'); } catch { e.meta = {}; }
    e.meta.id = e.id;
    e.size = Number(e.video.size) || 0;
    out.push(e);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Transfers

export function metaOf(clip) {
  const { blob, thumbnail, uploaded, inDrive, dirtyMeta, ...meta } = clip;
  return { ...meta, size: blob.size };
}
const fileName = clip => `${clip.name.replace(/[\\/:*?"<>|]/g, '_')}.stage-${clip.layout}.mp4`;

function xhr(method, url, { headers = {}, body, onUpload } = {}) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open(method, url);
    for (const [k, v] of Object.entries(headers)) x.setRequestHeader(k, v);
    if (onUpload) x.upload.onprogress = e => { if (e.lengthComputable) onUpload(e.loaded); };
    x.onload = () => resolve(x);
    x.onerror = () => reject(Object.assign(new Error('인터넷에 연결할 수 없어요.'), { offline: true }));
    x.send(body);
  });
}

async function resumableUpload(metadata, blob, onProgress) {
  const init = await request(`${UPLOAD}/files?uploadType=resumable&fields=id`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json; charset=UTF-8', 'X-Upload-Content-Type': blob.type || 'application/octet-stream', 'X-Upload-Content-Length': String(blob.size) },
    body: JSON.stringify(metadata),
  });
  // The session URL comes in Location; if the browser can't read it, rebuild it from the upload ID.
  const uploadId = init.headers.get('X-GUploader-UploadID');
  const session = init.headers.get('Location')
    || (uploadId && `${UPLOAD}/files?uploadType=resumable&upload_id=${encodeURIComponent(uploadId)}`);
  if (!session) throw new Error('구글 드라이브 업로드를 시작하지 못했어요.');
  const total = blob.size;
  let offset = 0, failures = 0;
  while (offset < total || total === 0) {
    const end = Math.min(offset + CHUNK, total);
    let x;
    try {
      x = await xhr('PUT', session, {
        headers: { 'Content-Range': total ? `bytes ${offset}-${end - 1}/${total}` : 'bytes */0' },
        body: blob.slice(offset, end),
        onUpload: n => onProgress?.((offset + n) / Math.max(total, 1)),
      });
    } catch (err) {
      if (++failures > 4) throw err;
      await new Promise(r => setTimeout(r, 1000 * 2 ** failures));
      // Ask how much arrived, then carry on from there.
      const probe = await xhr('PUT', session, { headers: { 'Content-Range': `bytes */${total}` } }).catch(() => null);
      const range = probe?.getResponseHeader('Range');
      if (probe && (probe.status === 200 || probe.status === 201)) return JSON.parse(probe.responseText);
      offset = range ? Number(range.split('-')[1]) + 1 : 0;
      continue;
    }
    if (x.status === 200 || x.status === 201) return JSON.parse(x.responseText);
    if (x.status === 308) {
      const range = x.getResponseHeader('Range');
      offset = range ? Number(range.split('-')[1]) + 1 : end;
      failures = 0;
      continue;
    }
    if (x.status === 401) { ls.set(LS.token, null); throw new NeedsSignIn('구글 로그인이 만료됐어요.'); }
    if (x.status >= 500 && ++failures <= 4) { await new Promise(r => setTimeout(r, 1000 * 2 ** failures)); continue; }
    throw new Error(`구글 드라이브 업로드 오류 (${x.status})`);
  }
}

export async function uploadClip(clip, onProgress) {
  const parent = await folderId();
  if (clip.thumbnail) {
    await resumableUpload({ name: `${clip.id}.png`, parents: [parent], appProperties: { fancamId: clip.id, kind: 'thumb' } }, clip.thumbnail);
  }
  // Video last: listRemote only counts a clip once its video exists.
  await resumableUpload({
    name: fileName(clip), parents: [parent], mimeType: 'video/mp4',
    description: JSON.stringify(metaOf(clip)),
    appProperties: { fancamId: clip.id, kind: 'video' },
  }, clip.blob, onProgress);
  onProgress?.(1);
}

export async function updateMeta(clip, entry) {
  await request(`${API}/files/${entry.video.id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: fileName(clip), description: JSON.stringify(metaOf(clip)) }),
  });
}

// Moves the clip to the Drive trash (recoverable for 30 days).
export async function deleteClip(entry) {
  for (const f of [entry.video, entry.thumb]) {
    if (!f) continue;
    await request(`${API}/files/${f.id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ trashed: true }),
    });
  }
}

async function fetchMedia(fileId, onBytes) {
  const res = await request(`${API}/files/${fileId}?alt=media`);
  if (!res.body || !onBytes) return res.blob();
  const reader = res.body.getReader();
  const chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    onBytes(value.byteLength);
  }
  return new Blob(chunks);
}

// Her face photos (a small JSON file next to the clips): { updated, refs: [...] }.
export async function getFaces() {
  const folder = await folderId();
  const res = await json(`${API}/files?q=${q(
    `'${folder}' in parents and trashed=false and appProperties has { key='fancamFaces' and value='1' }`,
  )}&fields=files(id)&spaces=drive`);
  const f = res.files[0];
  if (!f) return null;
  return { fileId: f.id, data: JSON.parse(await (await fetchMedia(f.id)).text()) };
}

export async function putFaces(data, fileId) {
  const body = new Blob([JSON.stringify(data)], { type: 'application/json' });
  if (fileId) {
    await request(`${UPLOAD}/files/${fileId}?uploadType=media`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body });
  } else {
    await resumableUpload({ name: '설윤 얼굴 사진.json', parents: [await folderId()], mimeType: 'application/json', appProperties: { fancamFaces: '1' } }, body);
  }
}

export async function downloadClip(entry, onProgress) {
  let got = 0;
  const video = await fetchMedia(entry.video.id, n => { got += n; onProgress?.(Math.min(1, got / Math.max(entry.size, 1))); });
  const thumbnail = entry.thumb ? new Blob([await fetchMedia(entry.thumb.id)], { type: 'image/png' }) : null;
  return { ...entry.meta, blob: new Blob([video], { type: 'video/mp4' }), thumbnail };
}
