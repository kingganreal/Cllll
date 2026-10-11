// Keeps clips in a private GitHub repository so the same clips show up on every device.
// Each clip is stored as   clips/<id>/meta.json, thumb.png, part-000.bin, part-001.bin, …
// (parts keep every file well under GitHub's 100 MB blob limit). Changes go in as one commit
// per clip through the Git Data API, which works on any size and from any browser.
const API = globalThis.__FANCAM_GITHUB_API__ || 'https://api.github.com'; // overridable for tests
const CONFIG_KEY = 'fancam-stage-sync';
const PART_SIZE = 20 * 1024 * 1024;

export function getConfig() {
  try { return JSON.parse(localStorage.getItem(CONFIG_KEY)) || null; } catch { return null; }
}
export function setConfig(cfg) {
  try {
    if (cfg) localStorage.setItem(CONFIG_KEY, JSON.stringify(cfg));
    else localStorage.removeItem(CONFIG_KEY);
  } catch {}
}

// "Connect another device" link: the settings ride in the #hash, which never leaves the browser.
export function makeConnectLink(cfg) {
  const code = btoa(unescape(encodeURIComponent(JSON.stringify({ r: cfg.repo, t: cfg.token }))))
    .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  return `${location.origin}${location.pathname}#connect=${code}`;
}
export function parseConnectCode(code) {
  const b64 = code.replace(/-/g, '+').replace(/_/g, '/');
  const { r, t } = JSON.parse(decodeURIComponent(escape(atob(b64))));
  if (!r || !t) throw new Error('연결 링크가 잘못됐어요.');
  return { repo: r, token: t };
}

function explain(status, message = '') {
  if (status === 401) return '토큰이 틀렸거나 만료됐어요. 새 토큰을 만들어 주세요.';
  if (status === 403 && /rate limit/i.test(message)) return 'GitHub 요청 한도에 걸렸어요. 잠시 후 다시 시도해 주세요.';
  if (status === 403) return '토큰 권한이 부족해요. 저장소의 Contents 권한을 "Read and write"로 줘야 해요.';
  if (status === 404) return '저장소를 찾을 수 없어요. 이름과, 토큰이 그 저장소에 접근할 수 있는지 확인해 주세요.';
  if (status === 413 || status === 422 && /too large/i.test(message)) return '파일이 너무 커서 올릴 수 없어요.';
  return `GitHub 오류 (${status}) ${message}`.trim();
}

async function gh(cfg, path, { method = 'GET', body, accept, signal } = {}) {
  let res;
  try {
    res = await fetch(API + path, {
      method, signal,
      headers: {
        Authorization: `Bearer ${cfg.token}`,
        Accept: accept || 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
      cache: 'no-store',
    });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw Object.assign(new Error('인터넷에 연결할 수 없어요.'), { offline: true });
  }
  if (!res.ok) {
    let msg = '';
    try { msg = (await res.json()).message || ''; } catch {}
    throw Object.assign(new Error(explain(res.status, msg)), { status: res.status, ghMessage: msg });
  }
  return res;
}
const json = async (cfg, path, opts) => (await gh(cfg, path, opts)).json();

const repoPath = cfg => `/repos/${cfg.repo}`;

// Checks the token and repository, and makes sure the repository has a first commit.
export async function connect(repoInput, token) {
  token = token.trim();
  let repo = repoInput.trim().replace(/^https?:\/\/github\.com\//, '').replace(/\.git$/, '').replace(/\/+$/, '');
  const cfg = { repo, token };
  if (!repo.includes('/')) {
    const me = await json(cfg, '/user');
    cfg.repo = repo = `${me.login}/${repo || 'fancam-clips'}`;
  }
  const info = await json(cfg, repoPath(cfg));
  if (!info.private) {
    throw new Error('이 저장소는 공개 저장소예요. 영상이 누구에게나 보이게 되니, 비공개(Private) 저장소를 써 주세요.');
  }
  if (info.permissions && !info.permissions.push) throw new Error(explain(403));
  cfg.branch = info.default_branch || 'main';
  try {
    await json(cfg, `${repoPath(cfg)}/git/ref/heads/${encodeURIComponent(cfg.branch)}`);
  } catch (err) {
    if (err.status !== 404 && err.status !== 409) throw err;
    // Empty repository: the Git Data API needs one commit to build on.
    await json(cfg, `${repoPath(cfg)}/contents/README.md`, {
      method: 'PUT',
      body: {
        message: '직캠 스테이지 클립 저장소',
        content: btoa(unescape(encodeURIComponent('# 직캠 스테이지 클립\n\n직캠 스테이지 앱이 기기 간 동기화에 쓰는 저장소예요. 직접 고치지 마세요.\n'))),
        branch: cfg.branch,
      },
    });
  }
  return cfg;
}

// ---------------------------------------------------------------------------
// Reading

const metaCache = new Map(); // blob sha -> parsed meta (blobs never change)

export async function listRemote(cfg) {
  const ref = await json(cfg, `${repoPath(cfg)}/git/ref/heads/${encodeURIComponent(cfg.branch)}`);
  const head = ref.object.sha;
  const commit = await json(cfg, `${repoPath(cfg)}/git/commits/${head}`);
  const tree = await json(cfg, `${repoPath(cfg)}/git/trees/${commit.tree.sha}?recursive=1`);
  const byId = new Map();
  for (const e of tree.tree) {
    const m = e.type === 'blob' && e.path.match(/^clips\/([\w-]+)\/(.+)$/);
    if (!m) continue;
    const [, id, file] = m;
    if (!byId.has(id)) byId.set(id, { id, files: {}, parts: [] });
    const entry = byId.get(id);
    entry.files[file] = e;
    if (/^part-\d+\.bin$/.test(file)) entry.parts.push(e);
  }
  const out = [];
  for (const entry of byId.values()) {
    const metaFile = entry.files['meta.json'];
    if (!metaFile) continue; // half-written clip, ignore
    if (!metaCache.has(metaFile.sha)) {
      const res = await gh(cfg, `${repoPath(cfg)}/git/blobs/${metaFile.sha}`, { accept: 'application/vnd.github.raw+json' });
      metaCache.set(metaFile.sha, JSON.parse(await res.text()));
    }
    entry.meta = metaCache.get(metaFile.sha);
    entry.parts.sort((a, b) => a.path.localeCompare(b.path));
    entry.size = entry.parts.reduce((s, p) => s + (p.size || 0), 0);
    out.push(entry);
  }
  return { clips: out, truncated: !!tree.truncated };
}

async function fetchBlob(cfg, sha, onBytes, signal) {
  const res = await gh(cfg, `${repoPath(cfg)}/git/blobs/${sha}`, { accept: 'application/vnd.github.raw+json', signal });
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

export async function downloadClip(cfg, entry, onProgress, signal) {
  const total = entry.size || 1;
  let got = 0;
  const parts = [];
  for (const p of entry.parts) {
    parts.push(await fetchBlob(cfg, p.sha, n => { got += n; onProgress?.(Math.min(1, got / total)); }, signal));
  }
  const thumbnail = entry.files['thumb.png']
    ? await fetchBlob(cfg, entry.files['thumb.png'].sha, null, signal).then(b => new Blob([b], { type: 'image/png' }))
    : null;
  return { ...entry.meta, blob: new Blob(parts, { type: 'video/mp4' }), thumbnail };
}

// ---------------------------------------------------------------------------
// Writing

function toBase64(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).slice(String(r.result).indexOf(',') + 1));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

// XHR rather than fetch so we get upload progress.
function createBlob(cfg, base64, onSent) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}${repoPath(cfg)}/git/blobs`);
    xhr.setRequestHeader('Authorization', `Bearer ${cfg.token}`);
    xhr.setRequestHeader('Accept', 'application/vnd.github+json');
    xhr.setRequestHeader('X-GitHub-Api-Version', '2022-11-28');
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.upload.onprogress = e => { if (e.lengthComputable) onSent?.(e.loaded / e.total); };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body.sha);
      else reject(Object.assign(new Error(explain(xhr.status, body.message)), { status: xhr.status }));
    };
    xhr.onerror = () => reject(Object.assign(new Error('인터넷에 연결할 수 없어요.'), { offline: true }));
    xhr.send(JSON.stringify({ content: base64, encoding: 'base64' }));
  });
}

const textBase64 = s => btoa(unescape(encodeURIComponent(s)));

// Applies tree changes as one commit; retries if another device committed in between.
async function commit(cfg, changes, message) {
  for (let attempt = 0; ; attempt++) {
    const ref = await json(cfg, `${repoPath(cfg)}/git/ref/heads/${encodeURIComponent(cfg.branch)}`);
    const head = ref.object.sha;
    const base = await json(cfg, `${repoPath(cfg)}/git/commits/${head}`);
    const tree = await json(cfg, `${repoPath(cfg)}/git/trees`, {
      method: 'POST', body: { base_tree: base.tree.sha, tree: changes },
    });
    const c = await json(cfg, `${repoPath(cfg)}/git/commits`, {
      method: 'POST', body: { message, tree: tree.sha, parents: [head] },
    });
    try {
      await json(cfg, `${repoPath(cfg)}/git/refs/heads/${encodeURIComponent(cfg.branch)}`, {
        method: 'PATCH', body: { sha: c.sha },
      });
      return;
    } catch (err) {
      if (err.status !== 422 || attempt >= 4) throw err;
    }
  }
}

export function metaOf(clip) {
  const { blob, thumbnail, uploaded, dirtyMeta, ...meta } = clip;
  return { ...meta, size: blob.size };
}

export async function uploadClip(cfg, clip, onProgress) {
  const dir = `clips/${clip.id}`;
  const blob = clip.blob;
  const count = Math.max(1, Math.ceil(blob.size / PART_SIZE));
  const changes = [];
  for (let i = 0; i < count; i++) {
    const part = blob.slice(i * PART_SIZE, (i + 1) * PART_SIZE);
    const sha = await createBlob(cfg, await toBase64(part), f => onProgress?.((i + f) / count));
    changes.push({ path: `${dir}/part-${String(i).padStart(3, '0')}.bin`, mode: '100644', type: 'blob', sha });
  }
  if (clip.thumbnail) {
    changes.push({ path: `${dir}/thumb.png`, mode: '100644', type: 'blob', sha: await createBlob(cfg, await toBase64(clip.thumbnail)) });
  }
  // meta.json last: a clip without it is treated as not there yet.
  changes.push({ path: `${dir}/meta.json`, mode: '100644', type: 'blob',
    sha: await createBlob(cfg, textBase64(JSON.stringify(metaOf(clip), null, 1))) });
  await commit(cfg, changes, `클립 추가: ${clip.name}`);
  onProgress?.(1);
}

export async function updateMeta(cfg, clip) {
  const sha = await createBlob(cfg, textBase64(JSON.stringify(metaOf(clip), null, 1)));
  await commit(cfg, [{ path: `clips/${clip.id}/meta.json`, mode: '100644', type: 'blob', sha }], `이름 변경: ${clip.name}`);
}

export async function deleteClip(cfg, entry, name) {
  const changes = Object.values(entry.files).map(f => ({ path: f.path, mode: '100644', type: 'blob', sha: null }));
  if (changes.length) await commit(cfg, changes, `클립 삭제: ${name || entry.id}`);
}
