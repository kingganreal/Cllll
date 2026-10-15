import { clips, kv, requestPersistence } from './store.js';
import { AlphaClip } from './player.js';
import * as sync from './sync.js';
import * as drive from './drive.js';
import { GOOGLE_CLIENT_ID } from './config.js';
import * as pcHelper from './helper.js';

const $ = sel => document.querySelector(sel);
const fmtTime = s => {
  if (!isFinite(s)) return '0:00';
  s = Math.max(0, s);
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
};
// SAM 2 tracking on PCs with WebGPU (each GPU is checked against the CPU once, see sam2.js).
const SAM_ENABLED = false;
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

let toastTimer;
function toast(msg, ms = 2600) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

// ---------------------------------------------------------------------------
// Routing

const views = { library: $('#view-library'), create: $('#view-create'), stage: $('#view-stage') };
let current = null;

function go(name) {
  if (!views[name]) name = 'library';
  if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
  if (current === name) return;
  if (current === 'stage') stage.leave();
  if (current === 'create') create.leave();
  current = name;
  for (const [k, el] of Object.entries(views)) el.hidden = k !== name;
  if (name === 'library') library.render();
  if (name === 'create') create.enter();
  if (name === 'stage') stage.enter();
}
document.addEventListener('click', e => {
  const nav = e.target.closest('[data-nav]');
  if (nav) go(nav.dataset.nav);
});
window.addEventListener('hashchange', () => go(location.hash.slice(1)));

// ---------------------------------------------------------------------------
// Wake lock (keep the screen on while processing or showing the stage)

let wakeLock = null, wantWake = false;
async function keepAwake(on) {
  wantWake = on;
  try {
    if (on && !wakeLock && navigator.wakeLock) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => { wakeLock = null; });
    } else if (!on && wakeLock) {
      await wakeLock.release();
      wakeLock = null;
    }
  } catch {}
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && wantWake) keepAwake(true);
});

// ---------------------------------------------------------------------------
// Library

const library = {
  refresh() { if (current === 'library') this.render(); },

  async render() {
    const grid = $('#clip-grid');
    let list = [];
    try { list = await clips.all(); } catch (err) { toast('저장소를 열 수 없어요: ' + err.message); }
    for (const img of grid.querySelectorAll('img')) URL.revokeObjectURL(img.src);
    grid.replaceChildren();
    $('#clip-count').textContent = list.length ? String(list.length) : '';
    $('#empty-library').hidden = list.length > 0;
    for (const c of list) grid.append(this.card(c));
    // With the Windows app running, clips can go straight onto the desktop.
    if (await pcHelper.desktopRunning()) {
      for (const b of grid.querySelectorAll('[data-a=desktop]')) b.hidden = false;
    }
  },

  card(c) {
    const el = document.createElement('article');
    el.className = 'card';
    const thumb = document.createElement('div');
    thumb.className = 'card-thumb checker';
    if (c.thumbnail) {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(c.thumbnail);
      img.alt = c.name;
      thumb.append(img);
    }
    thumb.addEventListener('click', () => stage.addAndOpen(c.id));
    const body = document.createElement('div');
    body.className = 'card-body';
    const sizeMb = (c.blob.size / 1048576).toFixed(c.blob.size > 1e7 ? 0 : 1);
    body.innerHTML = `
      <div><div class="card-name"></div>
      <div class="card-meta mono">${fmtTime(c.duration)} · ${c.hasAudio ? '소리 있음' : '소리 없음'} · ${sizeMb}MB</div>
      <div class="card-meta sync-badge" data-sync="${c.id}">${cloud.on ? (c.inDrive ? '모든 기기에 있음' : '올리기 대기') : ''}</div></div>
      <div class="card-actions">
        <button class="btn primary" type="button" data-a="stage">무대로</button>
        <button class="btn ghost" type="button" data-a="desktop" hidden>바탕화면에</button>
        <button class="btn ghost" type="button" data-a="rename">이름</button>
        <button class="btn ghost" type="button" data-a="save">저장</button>
        <button class="btn ghost danger" type="button" data-a="delete">삭제</button>
      </div>`;
    const nameEl = body.querySelector('.card-name');
    nameEl.textContent = c.name;
    body.addEventListener('click', async e => {
      const b = e.target.closest('[data-a]');
      if (!b) return;
      const a = b.dataset.a;
      if (a === 'stage') stage.addAndOpen(c.id);
      if (a === 'desktop') sendToDesktop(c, b);
      if (a === 'save') downloadClip(c);
      if (a === 'rename') {
        const input = document.createElement('input');
        input.type = 'text'; input.value = c.name; input.maxLength = 40;
        nameEl.replaceChildren(input);
        input.focus(); input.select();
        const commit = async () => {
          const v = input.value.trim() || c.name;
          nameEl.textContent = v;
          if (v === c.name) return;
          // Re-read: sync may have changed the record since this card was drawn.
          const fresh = (await clips.get(c.id)) || c;
          fresh.name = c.name = v;
          fresh.updated = c.updated = Date.now();
          if (fresh.inDrive) fresh.dirtyMeta = true;
          await clips.put(fresh);
          cloud.kick();
        };
        input.addEventListener('keydown', ev => { if (ev.key === 'Enter') input.blur(); });
        input.addEventListener('blur', commit, { once: true });
      }
      if (a === 'delete') {
        if (b.dataset.confirm) {
          const fresh = (await clips.get(c.id)) || c;
          await clips.delete(c.id);
          stage.forget(c.id);
          if (fresh.inDrive || cloud.running) {
            const pending = (await kv.get('pendingDeletes')) || [];
            await kv.set('pendingDeletes', [...new Set([...pending, c.id])]);
          }
          this.render();
          toast(cloud.on ? '삭제했어요. 다른 기기에서도 지워져요.' : '삭제했어요');
          cloud.kick();
        } else {
          b.dataset.confirm = '1';
          b.textContent = '한 번 더 누르면 삭제';
          setTimeout(() => { delete b.dataset.confirm; b.textContent = '삭제'; }, 3000);
        }
      }
    });
    el.append(thumb, body);
    return el;
  },
};

async function sendToDesktop(c, button) {
  const label = button?.textContent;
  try {
    const full = (await clips.get(c.id)) || c;
    await pcHelper.sendToDesktop(full, f => { if (button) button.textContent = `보내는 중 ${Math.round(f * 100)}%`; });
    toast('바탕화면에 띄웠어요. 끌어서 옮기고, 오른쪽 클릭으로 설정을 바꿀 수 있어요.', 4200);
  } catch (err) {
    toast(err.message, 4500);
  } finally {
    if (button) button.textContent = label;
  }
}

function downloadClip(c) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(c.blob);
  // The layout is kept in the name so 가져오기 can read it back.
  a.download = `${c.name.replace(/[\\/:*?"<>|]/g, '_')}.stage-${c.layout}.mp4`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
  toast('기기에 저장했어요. 다른 기기에서 "파일 가져오기"로 열 수 있어요.', 3600);
}

$('#btn-new').addEventListener('click', () => go('create'));
$('#btn-open-stage').addEventListener('click', () => go('stage'));
$('#btn-import').addEventListener('click', () => $('#import-input').click());
$('#import-input').addEventListener('change', async e => {
  const files = [...e.target.files];
  e.target.value = '';
  for (const f of files) {
    try {
      await importClip(f);
      toast(`"${f.name}" 가져왔어요`);
    } catch (err) {
      toast('가져오지 못했어요: ' + err.message, 4000);
    }
  }
  library.render();
  cloud.kick();
});

// Reads back a clip saved with "저장" (or any colour|alpha packed MP4).
async function importClip(file) {
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto';
  v.src = URL.createObjectURL(file);
  try {
    await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = () => rej(new Error('영상을 열 수 없어요')); });
    const m = file.name.match(/\.stage-(sbs|tb)\.mp4$/i);
    const layout = m ? m[1].toLowerCase() : (v.videoWidth >= v.videoHeight ? 'sbs' : 'tb');
    const width = layout === 'sbs' ? v.videoWidth / 2 : v.videoWidth;
    const height = layout === 'sbs' ? v.videoHeight : v.videoHeight / 2;
    v.currentTime = Math.min(1, v.duration / 3);
    await new Promise(r => { v.onseeked = r; });
    const thumbnail = await packedThumbnail(v, layout, width, height);
    const name = file.name.replace(/(\.stage-(sbs|tb))?\.mp4$/i, '');
    await clips.put({
      id: uid(), name, created: Date.now(), blob: file, thumbnail, layout,
      crop: { x: 0, y: 0, w: 1, h: 1 }, width, height, duration: v.duration, hasAudio: true,
    });
  } finally {
    URL.revokeObjectURL(v.src);
  }
}

function packedThumbnail(v, layout, width, height) {
  const s = Math.min(1, 360 / Math.max(width, height));
  const w = Math.round(width * s), h = Math.round(height * s);
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const ax = layout === 'sbs' ? width : 0, ay = layout === 'sbs' ? 0 : height;
  ctx.drawImage(v, ax, ay, width, height, 0, 0, w, h);
  const a = ctx.getImageData(0, 0, w, h);
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(v, 0, 0, width, height, 0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h);
  for (let j = 0; j < img.data.length; j += 4) img.data[j + 3] = a.data[j];
  ctx.putImageData(img, 0, 0);
  return new Promise(r => c.toBlob(r, 'image/png'));
}

// ---------------------------------------------------------------------------
// Cloud sync across phone / tablet / computer through Google Drive (see js/drive.js).
// Clips synced earlier through GitHub (js/sync.js) are moved over once.

const cloud = {
  running: false,
  again: false,
  lastRun: 0,

  get on() { return drive.isEnabled(); },

  kick() {
    if (!this.on) return;
    if (!drive.hasToken()) { this.askSignIn(); return; }
    if (this.running) { this.again = true; return; }
    this.run();
  },

  status(text, kind = '') {
    const el = $('#sync-status');
    el.hidden = !text;
    el.textContent = text;
    el.className = 'sync-status ' + kind;
  },

  // Google sign-ins last about an hour; renewing one needs a tap (it opens a popup).
  askSignIn() {
    const el = $('#sync-status');
    el.hidden = false;
    el.className = 'sync-status';
    el.textContent = '';
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'btn primary';
    b.textContent = '구글 드라이브와 동기화하기';
    b.onclick = () => this.signInAndSync();
    const note = document.createElement('span');
    note.textContent = '로그인 확인이 필요해요';
    el.append(b, note);
    drive.preload().catch(() => {});
  },

  async signInAndSync() {
    try {
      await drive.signIn();
      this.renderPanel();
      library.refresh();
      this.kick();
    } catch (err) {
      toast(err.message, 4500);
    }
  },

  badge(id, text) {
    const el = document.querySelector(`[data-sync="${CSS.escape(id)}"]`);
    if (el) el.textContent = text;
  },

  async run() {
    this.running = true;
    try {
      do { this.again = false; await this.pass(); } while (this.again);
      const t = new Date();
      this.status(`드라이브 동기화 완료 · ${t.getHours()}:${String(t.getMinutes()).padStart(2, '0')}`);
    } catch (err) {
      console.error(err);
      if (drive.isSignInError(err)) this.askSignIn();
      else this.status('동기화 실패: ' + err.message + (err.offline ? ' 연결되면 다시 시도해요.' : ''), 'error');
    } finally {
      this.running = false;
      this.lastRun = Date.now();
    }
  },

  async pass() {
    this.status('구글 드라이브 확인 중…', 'busy');
    let remote = await drive.listRemote();

    // 0. One-time move of clips that only exist in the old GitHub sync.
    const github = sync.getConfig();
    if (github) {
      try {
        const { clips: old } = await sync.listRemote(github);
        const have = new Set([...(await clips.all()).map(c => c.id), ...remote.map(r => r.id)]);
        for (const [i, r] of old.filter(r => !have.has(r.id)).entries()) {
          const label = `GitHub에서 옮기는 중 (${i + 1}) ${r.meta.name}`;
          this.status(label, 'busy');
          const rec = await sync.downloadClip(github, r, f => this.status(`${label} ${Math.round(f * 100)}%`, 'busy'));
          delete rec.uploaded;
          await clips.put(rec);
        }
        sync.setConfig(null);
        toast('GitHub에 있던 클립을 구글 드라이브로 옮기고 있어요. 다 끝나면 GitHub 저장소는 지워도 돼요.', 5000);
      } catch (err) {
        console.warn('GitHub migration skipped', err);
      }
    }

    // 1. Deletions made on this device.
    const pending = (await kv.get('pendingDeletes')) || [];
    for (const id of pending) {
      const entry = remote.find(r => r.id === id);
      if (entry) await drive.deleteClip(entry);
    }
    if (pending.length) {
      await kv.set('pendingDeletes', []);
      remote = remote.filter(r => !pending.includes(r.id));
    }

    // 2. Changes made on other devices: deletions and renames.
    const remoteById = new Map(remote.map(r => [r.id, r]));
    const local = await clips.all();
    let changed = false;
    for (const c of local) {
      const r = remoteById.get(c.id);
      if (!r) {
        if (c.inDrive) { await clips.delete(c.id); stage.forget(c.id); changed = true; }
        continue;
      }
      const remoteNewer = (r.meta.updated || 0) > (c.updated || 0);
      const localNewer = (c.updated || 0) > (r.meta.updated || 0);
      if (c.dirtyMeta || (localNewer && c.name !== r.meta.name)) {
        await drive.updateMeta(c, r);
        c.dirtyMeta = false;
        c.inDrive = true;
        await clips.put(c);
      } else if (remoteNewer || !c.inDrive) {
        c.name = r.meta.name || c.name;
        c.updated = r.meta.updated;
        c.inDrive = true;
        await clips.put(c);
        changed = true;
      }
    }
    if (changed) library.refresh();

    // 3. New clips from this device.
    const toUpload = (await clips.all()).filter(c => !c.inDrive && !remoteById.has(c.id));
    for (const [i, c] of toUpload.entries()) {
      const label = `드라이브에 올리는 중 (${i + 1}/${toUpload.length}) ${c.name}`;
      this.status(label, 'busy');
      await drive.uploadClip(c, f => {
        this.status(`${label} ${Math.round(f * 100)}%`, 'busy');
        this.badge(c.id, `올리는 중 ${Math.round(f * 100)}%`);
      });
      const still = await clips.get(c.id);
      if (still) {
        still.inDrive = true;
        await clips.put(still);
        this.badge(c.id, '모든 기기에 있음');
      } else {
        // Deleted while uploading: remove it from Drive on the next pass.
        const p = (await kv.get('pendingDeletes')) || [];
        await kv.set('pendingDeletes', [...p, c.id]);
        this.again = true;
      }
    }

    // 4. New clips from other devices.
    const localIds = new Set((await clips.all()).map(c => c.id));
    const toDownload = remote.filter(r => !localIds.has(r.id));
    for (const [i, r] of toDownload.entries()) {
      const label = `받는 중 (${i + 1}/${toDownload.length}) ${r.meta.name || ''}`;
      this.status(label, 'busy');
      const rec = await drive.downloadClip(r, f => this.status(`${label} ${Math.round(f * 100)}%`, 'busy'));
      rec.inDrive = true;
      delete rec.uploaded;
      await clips.put(rec);
      requestPersistence();
      library.refresh();
    }
  },

  renderPanel() {
    const hasId = !!drive.clientId();
    $('#drive-setup').hidden = hasId;
    $('#drive-signin').hidden = !hasId || (this.on && drive.hasToken());
    $('#drive-connected').hidden = !(this.on && drive.hasToken());
    $('#drive-reset').hidden = !!GOOGLE_CLIENT_ID;
    $('#sync-link').hidden = !!GOOGLE_CLIENT_ID;
    $('#drive-origin').textContent = location.origin;
    $('#sync-link-text').hidden = true;
    drive.preload().catch(() => {}); // the sign-in popup must open straight from a tap
  },
};

$('#btn-sync').addEventListener('click', () => {
  const panel = $('#sync-panel');
  panel.hidden = !panel.hidden;
  cloud.renderPanel();
});
$('#sync-close').addEventListener('click', () => { $('#sync-panel').hidden = true; });
$('#drive-origin-copy').addEventListener('click', () => copyText(location.origin, '주소를 복사했어요.'));
$('#drive-form').addEventListener('submit', async e => {
  e.preventDefault();
  const id = $('#drive-client').value.trim();
  if (!drive.looksLikeClientId(id)) {
    toast('클라이언트 ID는 ".apps.googleusercontent.com"으로 끝나요. 다시 확인해 주세요.', 4500);
    return;
  }
  drive.setClientId(id);
  cloud.renderPanel();
  await cloud.signInAndSync();
});
$('#drive-login').addEventListener('click', () => cloud.signInAndSync());
$('#drive-reset').addEventListener('click', () => {
  drive.setClientId(null);
  drive.disable();
  cloud.renderPanel();
});
$('#sync-now').addEventListener('click', () => cloud.kick());
$('#sync-link').addEventListener('click', () => {
  const link = `${location.origin}${location.pathname}#gdrive=${encodeURIComponent(drive.clientId())}`;
  copyText(link, '복사했어요. 다른 기기로 보내서 열어 주세요.', $('#sync-link-text'));
});
$('#sync-off').addEventListener('click', e => {
  const b = e.currentTarget;
  if (!b.dataset.confirm) {
    b.dataset.confirm = '1';
    b.textContent = '한 번 더 누르면 끊기';
    setTimeout(() => { delete b.dataset.confirm; b.textContent = '이 기기 연결 끊기'; }, 3000);
    return;
  }
  delete b.dataset.confirm;
  b.textContent = '이 기기 연결 끊기';
  drive.disable();
  cloud.status('');
  cloud.renderPanel();
  library.refresh();
  toast('이 기기만 연결을 끊었어요. 클립은 그대로 남아 있어요.');
});

async function copyText(text, done, fallbackField) {
  try {
    await navigator.clipboard.writeText(text);
    toast(done);
  } catch {
    if (fallbackField) {
      fallbackField.hidden = false;
      fallbackField.value = text;
      fallbackField.select();
    }
    toast('길게 눌러 복사해 주세요: ' + text, 5000);
  }
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && Date.now() - cloud.lastRun > 60000) cloud.kick();
});

// ---------------------------------------------------------------------------
// Create (extract a member from a fancam)

const create = {
  file: null, url: null, start: 0, end: 0, persons: null, target: null, abort: null, busy: false,
  detectedRect: null, runId: 0,

  // The area of the frame we work on: the picture without black bars, unless switched off.
  rect() {
    const v = $('#src-video');
    return this.detectedRect && $('#auto-crop').checked
      ? this.detectedRect
      : { x: 0, y: 0, w: v.videoWidth, h: v.videoHeight };
  },

  enter() {
    const warn = [];
    if (typeof VideoEncoder === 'undefined') warn.push('이 브라우저는 영상 인코딩을 지원하지 않아요. 최신 Chrome, Edge, Safari(iOS 16.4 이상)에서 열어 주세요.');
    $('#create-error').hidden = !warn.length;
    $('#create-error').textContent = warn.join(' ');
    // Load the models in the background so they're ready by the time she's picked.
    import('./processor.js').then(p => p.getLandmarker()).catch(err => console.warn(err));
    import('./matting.js').then(m => m.preload()).catch(() => {});
  },

  leave() {
    if (this.busy) this.cancel();
    keepAwake(false);
  },

  reset() {
    this.persons = this.target = null;
    $('#pick-canvas').hidden = true;
    $('#src-video').hidden = false;
    $('#pick-hint').hidden = true;
    $('#btn-start').hidden = true;
    $('#btn-pick').hidden = false;
    $('#btn-pick').textContent = '멤버 고르기';
  },

  async load(file) {
    this.file = file;
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = URL.createObjectURL(file);
    const v = $('#src-video');
    v.src = this.url;
    v.controls = true;
    await new Promise((res, rej) => {
      v.onloadedmetadata = res;
      v.onerror = () => rej(new Error('이 영상 형식은 브라우저에서 열 수 없어요. MP4로 바꿔서 다시 시도해 주세요.'));
    });
    await ensureDuration(v);
    this.detectedRect = null;
    $('#crop-row').hidden = true;
    try {
      const p = await import('./processor.js');
      this.detectedRect = await p.detectContentRect(v, 0, v.duration);
      await p.seek(v, 0);
    } catch (err) { console.warn('border detection failed', err); }
    if (this.detectedRect) {
      $('#crop-row').hidden = false;
      $('#auto-crop').checked = true;
      $('#crop-label').textContent = `검은 테두리 잘라내기 (${this.detectedRect.w}×${this.detectedRect.h}만 사용)`;
    }
    this.start = 0;
    this.end = v.duration;
    $('#scrub').max = String(v.duration);
    $('#step-edit').hidden = false;
    $('#drop').classList.add('compact');
    $('#drop strong').textContent = '다른 영상 고르기';
    $('#create-error').hidden = true;
    const base = file.name.replace(/\.[^.]+$/, '');
    if (!$('#clip-name').dataset.touched) $('#clip-name').value = /설윤|sullyoon|seolyoon/i.test(base) ? '설윤' : $('#clip-name').value;
    this.reset();
    this.updateTrim();
  },

  updateTrim() {
    const v = $('#src-video');
    $('#start-out').textContent = fmtTime(this.start);
    $('#end-out').textContent = fmtTime(this.end);
    $('#len-out').textContent = fmtTime(this.end - this.start);
    $('#now-out').textContent = fmtTime(v.currentTime);
  },

  async pick() {
    const btn = $('#btn-pick');
    btn.disabled = true;
    btn.textContent = 'AI 모델 불러오는 중…';
    try {
      const p = await import('./processor.js');
      await p.getLandmarker();
      const v = $('#src-video');
      v.pause();
      btn.textContent = '사람 찾는 중…';
      await p.seek(v, this.start + 0.02);
      const r = this.rect();
      const [aw, ah] = p.analysisSize(r, Number($('#quality').value));
      const c = document.createElement('canvas'); c.width = aw; c.height = ah;
      c.getContext('2d').drawImage(v, r.x, r.y, r.w, r.h, 0, 0, aw, ah);
      this.persons = await p.detect(c);
      if (!this.persons.length) {
        throw new Error('시작 장면에서 사람을 찾지 못했어요. 시작 지점을 멤버가 잘 보이는 곳으로 옮겨 주세요.');
      }
      this.target = p.pickDefault(this.persons);
      this.tap = null;
      this.pickApi = p;
      // Start fetching the SAM 2 tracker now (PC with WebGPU), so it's ready when 추출 starts.
      if (SAM_ENABLED) import('./sam2.js').then(m => m.load()).catch(() => {});
      this.drawPick();
      $('#pick-canvas').hidden = false;
      v.hidden = true;
      $('#pick-hint').hidden = false;
      $('#btn-start').hidden = false;
      btn.textContent = '다시 고르기';
    } catch (err) {
      $('#create-error').textContent = err.message;
      $('#create-error').hidden = false;
      btn.textContent = '멤버 고르기';
    } finally {
      btn.disabled = false;
    }
  },

  drawPick() {
    const v = $('#src-video');
    const c = $('#pick-canvas');
    const rect = this.rect();
    const s = Math.min(1, 1280 / Math.max(rect.w, rect.h));
    c.width = Math.round(rect.w * s); c.height = Math.round(rect.h * s);
    const ctx = c.getContext('2d');
    ctx.drawImage(v, rect.x, rect.y, rect.w, rect.h, 0, 0, c.width, c.height);
    // Dim everyone, then tint each person: pink for the chosen one, blue for the rest.
    ctx.fillStyle = 'rgba(8,6,12,.45)';
    ctx.fillRect(0, 0, c.width, c.height);
    for (const p of this.persons) {
      const chosen = p === this.target;
      const m = document.createElement('canvas'); m.width = p.w; m.height = p.h;
      const mctx = m.getContext('2d');
      const img = mctx.createImageData(p.w, p.h);
      const [r, g, b] = chosen ? [255, 159, 182] : [159, 211, 255];
      for (let k = 0, j = 0; k < p.mask.length; k++, j += 4) {
        img.data[j] = r; img.data[j + 1] = g; img.data[j + 2] = b;
        img.data[j + 3] = p.mask[k] > 0.5 ? (chosen ? 110 : 70) : 0;
      }
      mctx.putImageData(img, 0, 0);
      ctx.drawImage(m, 0, 0, c.width, c.height);
      if (chosen) {
        // Re-draw her un-dimmed so she pops.
        mctx.globalCompositeOperation = 'source-in';
        mctx.drawImage(v, rect.x, rect.y, rect.w, rect.h, 0, 0, p.w, p.h);
        ctx.globalAlpha = 0.75;
        ctx.drawImage(m, 0, 0, c.width, c.height);
        ctx.globalAlpha = 1;
      }
      if (p.bbox) {
        const [x0, y0] = p.bbox;
        ctx.font = `600 ${Math.round(c.height / 32)}px ${getComputedStyle(document.body).fontFamily}`;
        ctx.fillStyle = chosen ? '#ff9fb6' : '#9fd3ff';
        ctx.fillText(chosen ? '✓ 선택됨' : '탭해서 선택', x0 * c.width + 6, Math.max(c.height / 28, y0 * c.height - 8));
      }
    }
  },

  async run() {
    const v = $('#src-video');
    const p = this.pickApi;
    const opts = {
      file: this.file, video: v, start: this.start, end: this.end, target: this.target, rect: this.rect(),
      tap: this.tap, others: this.persons, stop: (this.stopFlag = { requested: false }),
      useSam: SAM_ENABLED,
      fps: Number($('#fps').value), maxSide: Number($('#quality').value),
      withAudio: $('#with-audio').checked,
    };
    // ~4 Mbps at 960, ~7 at 1280, ~12 at 1920 (30fps)
    opts.bitrate = Math.round(opts.maxSide * opts.maxSide * (opts.maxSide >= 1920 ? 3.2 : 2.2) * opts.fps / 30);
    const name = $('#clip-name').value.trim() || '설윤';

    this.busy = true;
    const runId = ++this.runId;
    this.abort = new AbortController();
    opts.signal = this.abort.signal;
    $('#step-edit').hidden = true;
    $('#drop').hidden = true;
    $('#step-progress').hidden = false;
    keepAwake(true);
    const pc = $('#progress-canvas'), pctx = pc.getContext('2d');
    let lastDraw = 0;
    $('#progress-text').textContent = 'AI 준비 중…';
    opts.onProgress = info => {
      if (runId !== this.runId) return;
      if (info.stage) { $('#progress-text').textContent = info.stage + '…'; return; }
      const pct = info.done / info.total * 100;
      $('#progress-bar').style.width = pct.toFixed(1) + '%';
      const mode = info.sam ? 'SAM 2 추적 + GPU 누끼' : info.matting ? 'GPU 고급 누끼' : '기본 누끼';
      $('#progress-text').textContent =
        `${info.done} / ${info.total} 프레임 · ${pct.toFixed(0)}% · 남은 시간 약 ${fmtTime(info.eta)}\n${mode}${info.found ? '' : ' · 멤버를 놓쳐서 다시 찾는 중'}`;
      const now = performance.now();
      if (now - lastDraw > 250 && info.preview) {
        lastDraw = now;
        if (pc.width !== info.preview.width) { pc.width = info.preview.width; pc.height = info.preview.height; }
        pctx.drawImage(info.preview, 0, 0);
      }
    };
    try {
      const result = await p.extract(opts);
      if (runId !== this.runId) return; // cancelled meanwhile
      const clip = { id: uid(), name, created: Date.now(), ...result };
      delete clip.partial;
      await clips.put(clip);
      requestPersistence();
      cloud.kick();
      toast(result.partial ? `${fmtTime(result.duration)}까지 저장했어요. 무대에 올렸어요` : '완성! 무대에 올렸어요');
      this.done();
      await stage.addAndOpen(clip.id);
    } catch (err) {
      if (runId !== this.runId) return; // cancelled: the screen was already reset
      this.done();
      if (err.name !== 'AbortError') {
        console.error(err);
        $('#create-error').textContent = '추출에 실패했어요: ' + err.message;
        $('#create-error').hidden = false;
      }
    }
  },

  // Finish early: stop at the next frame and save what's done so far as a clip.
  stopAndSave() {
    if (!this.busy || !this.stopFlag) return;
    this.stopFlag.requested = true;
    const b = $('#btn-stop');
    b.disabled = true;
    b.textContent = '저장하는 중…';
  },

  // Stop right away: reset the screen now; the work still in flight notices the abort and quits.
  cancel() {
    if (!this.busy) return;
    this.abort?.abort();
    this.runId++;
    this.done();
    toast('취소했어요');
  },

  done() {
    this.busy = false;
    $('#btn-stop').disabled = false;
    $('#btn-stop').textContent = '여기까지 저장';
    keepAwake(false);
    $('#step-progress').hidden = true;
    $('#step-edit').hidden = false;
    $('#drop').hidden = false;
    $('#progress-bar').style.width = '0';
    this.reset();
  },
};

$('#file-input').addEventListener('change', async e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  try { await create.load(f); }
  catch (err) { $('#create-error').textContent = err.message; $('#create-error').hidden = false; }
});
$('#clip-name').addEventListener('input', e => { e.target.dataset.touched = '1'; });
$('#src-video').addEventListener('timeupdate', () => {
  $('#scrub').value = String($('#src-video').currentTime);
  create.updateTrim();
});
$('#scrub').addEventListener('input', e => {
  const v = $('#src-video');
  v.currentTime = Number(e.target.value);
  create.reset();
});
$('#set-start').addEventListener('click', () => {
  const t = $('#src-video').currentTime;
  create.start = Math.min(t, create.end - 0.5);
  create.reset();
  create.updateTrim();
});
$('#set-end').addEventListener('click', () => {
  const t = $('#src-video').currentTime;
  create.end = Math.max(t, create.start + 0.5);
  create.updateTrim();
});
$('#btn-pick').addEventListener('click', () => {
  if (!create.file) return;
  if (create.persons) { create.reset(); $('#src-video').hidden = false; return; }
  create.pick();
});
$('#pick-canvas').addEventListener('click', e => {
  const c = e.currentTarget, r = c.getBoundingClientRect();
  const nx = (e.clientX - r.left) / r.width, ny = (e.clientY - r.top) / r.height;
  create.target = create.pickApi.pickAt(create.persons, nx, ny);
  create.tap = [nx, ny];
  create.drawPick();
});
$('#btn-start').addEventListener('click', () => create.run());
$('#auto-crop').addEventListener('change', () => create.reset());
// The pick is analysed at a size that depends on quality, so re-pick after changing it.
$('#quality').addEventListener('change', () => { if (create.persons) create.reset(); });
$('#btn-cancel').addEventListener('click', () => create.cancel());
$('#btn-stop').addEventListener('click', () => create.stopAndSave());

// ---------------------------------------------------------------------------
// YouTube: the browser can't download from YouTube, so we record it while it plays.
// Desktop Chrome/Edge records the YouTube tab directly; phones use built-in screen recording.

// MediaRecorder files often have no duration until the browser has scanned to the end.
async function ensureDuration(v) {
  if (isFinite(v.duration) && v.duration > 0) return;
  await new Promise(resolve => {
    const done = () => { v.removeEventListener('durationchange', check); clearTimeout(timer); resolve(); };
    const check = () => { if (isFinite(v.duration)) done(); };
    const timer = setTimeout(done, 8000);
    v.addEventListener('durationchange', check);
    v.currentTime = 1e7;
  });
  v.currentTime = 0;
  if (!isFinite(v.duration)) throw new Error('영상 길이를 알 수 없어요. 다시 녹화해 주세요.');
}

function parseYouTube(text) {
  const m = text.trim().match(
    /(?:youtube\.com\/(?:watch\?(?:.*&)?v=|shorts\/|live\/|embed\/|v\/)|youtu\.be\/)([\w-]{11})/i);
  return m ? m[1] : null;
}

const isTouchDevice = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent)
  || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
const canCaptureTab = !!navigator.mediaDevices?.getDisplayMedia && !isTouchDevice
  && typeof MediaRecorder !== 'undefined';

const capture = {
  recorder: null, stream: null, chunks: [], startedAt: 0, timer: null, videoId: null,

  showHelp(id) {
    this.videoId = id;
    const watch = `https://www.youtube.com/watch?v=${id}`;
    const help = $('#yt-help');
    help.hidden = false;
    if (canCaptureTab) {
      const record = `
        <ol>
          <li><b>유튜브 탭 열기</b> <a id="yt-open" href="${watch}" target="_blank" rel="noopener">이 링크</a>를 새 탭에서 열고, 화질을 가장 높게 맞춘 다음 일시정지해 두세요.</li>
          <li><b>녹화 시작</b> 아래 버튼을 누르고 방금 연 유튜브 탭을 고르세요. <b>"탭 오디오도 공유"</b>를 켜야 노래가 같이 녹음돼요.</li>
          <li><b>재생</b> 유튜브 탭에서 <b>전체화면(F 키)</b>으로 처음부터 재생하세요. 전체화면이어야 화면 속 영상만 깔끔하게 녹화돼요.</li>
          <li><b>녹화 끝</b> 다 보면 이 탭으로 돌아와서 "녹화 끝"을 누르세요. 앞뒤 남는 부분은 다음 화면에서 잘라 내면 돼요.</li>
        </ol>
        <div class="row" id="rec-controls">
          <button id="rec-start" class="btn ${pcHelper.isWindowsDesktop ? 'ghost' : 'primary'}" type="button">녹화 시작</button>
        </div>`;
      if (pcHelper.isWindowsDesktop) {
        // The PC helper gets the original file (up to 4K); recording stays as a fallback.
        help.innerHTML = `
          <div id="helper-box" class="helper-box"></div>
          <details class="alt">
            <summary>또는 화면 녹화로 가져오기 (화질이 화면 크기로 제한돼요)</summary>
            ${record}
          </details>`;
        youtubeHelper.show(watch);
      } else {
        help.innerHTML = record;
      }
      $('#rec-start').onclick = () => this.start();
    } else {
      help.innerHTML = `
        <ol>
          <li><b>유튜브에서 열기</b> <a href="${watch}" target="_blank" rel="noopener">이 링크</a>를 눌러 유튜브 앱에서 여세요.</li>
          <li><b>화면 녹화 켜기</b> iPhone·iPad: 제어 센터 → 화면 기록 버튼. 갤럭시 등 안드로이드: 빠른 설정 → 화면 녹화(소리는 "미디어 소리" 선택).</li>
          <li><b>전체화면으로 재생</b> 영상을 전체화면으로 끝까지 재생하고 녹화를 멈추세요.</li>
          <li><b>불러오기</b> 이 화면으로 돌아와 위의 "직캠 영상 고르기"에서 방금 녹화한 영상을 고르세요. 검은 테두리는 자동으로 잘라 내요.</li>
        </ol>
        <p class="hint">휴대폰 브라우저는 다른 앱 화면을 직접 녹화할 수 없어서, 기기의 화면 녹화 기능을 써야 해요. PC의 Chrome이나 Edge에서는 이 앱이 유튜브 탭을 바로 녹화해요.</p>`;
    }
  },

  async start() {
    const ask = audio => navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { ideal: 30 } },
      audio: audio && { suppressLocalAudioPlayback: false },
      preferCurrentTab: false,
      selfBrowserSurface: 'exclude',
      surfaceSwitching: 'exclude',
    });
    let stream;
    try {
      try {
        stream = await ask(true);
      } catch (err) {
        if (err.name === 'NotAllowedError') throw err;
        stream = await ask(false); // some systems can't capture audio at all
      }
    } catch (err) {
      if (err.name !== 'NotAllowedError') toast('녹화를 시작할 수 없어요: ' + err.message, 4000);
      return;
    }
    if (!stream.getAudioTracks().length) toast('소리 없이 녹화돼요. 노래도 담으려면 "탭 오디오도 공유"를 켜고 다시 시작하세요.', 5000);
    const type = ['video/mp4;codecs=avc1,mp4a.40.2', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm']
      .find(t => MediaRecorder.isTypeSupported(t)) || '';
    this.stream = stream;
    this.chunks = [];
    this.recorder = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 10_000_000, audioBitsPerSecond: 192_000 });
    this.recorder.ondataavailable = e => { if (e.data.size) this.chunks.push(e.data); };
    this.recorder.onstop = () => this.finish();
    // Chrome's "공유 중지" bar ends the track: treat it like pressing 녹화 끝.
    stream.getVideoTracks()[0].addEventListener('ended', () => this.stop());
    this.recorder.start(1000);
    this.startedAt = Date.now();
    keepAwake(true);
    const ctl = $('#rec-controls');
    ctl.innerHTML = `<span class="mono"><span class="rec-dot"></span><span id="rec-time">0:00</span> 녹화 중</span>
      <button id="rec-stop" class="btn primary" type="button">녹화 끝</button>`;
    $('#rec-stop').onclick = () => this.stop();
    this.timer = setInterval(() => { $('#rec-time').textContent = fmtTime((Date.now() - this.startedAt) / 1000); }, 500);
  },

  stop() {
    if (this.recorder?.state === 'recording') this.recorder.stop();
    this.stream?.getTracks().forEach(t => t.stop());
  },

  async finish() {
    clearInterval(this.timer);
    keepAwake(false);
    const type = this.recorder.mimeType || 'video/webm';
    const blob = new Blob(this.chunks, { type });
    this.recorder = this.stream = null;
    this.chunks = [];
    this.showHelp(this.videoId);
    if (blob.size < 1000) { toast('녹화된 내용이 없어요.'); return; }
    const ext = type.includes('mp4') ? 'mp4' : 'webm';
    const file = new File([blob], `youtube-${this.videoId}.${ext}`, { type });
    try {
      await create.load(file);
      toast('녹화 완료! 구간을 다듬고 멤버를 골라 주세요.', 3600);
    } catch (err) {
      $('#create-error').textContent = err.message;
      $('#create-error').hidden = false;
    }
  },
};

// Windows PC helper: downloads the original YouTube file (up to 4K) and hands it to the editor.
const youtubeHelper = {
  url: null, checkTimer: null, busy: false,

  async show(url) {
    this.url = url;
    clearInterval(this.checkTimer);
    this.render('checking');
    if (await pcHelper.isRunning()) { this.render('ready'); return; }
    this.render('off');
    // Notice the helper as soon as it's started, without a click.
    this.checkTimer = setInterval(async () => {
      if (!document.body.contains($('#helper-box'))) { clearInterval(this.checkTimer); return; }
      if (await pcHelper.isRunning()) { clearInterval(this.checkTimer); this.render('ready'); }
    }, 3000);
  },

  render(state, info = {}) {
    const box = $('#helper-box');
    if (!box) return;
    const head = '<p class="helper-title"><b>4K 원본으로 받기</b> <span class="pill">추천</span></p>';
    if (state === 'checking') {
      box.innerHTML = head + '<p class="hint">PC 도우미를 찾는 중…</p>';
    } else if (state === 'off') {
      box.innerHTML = head + `
        <p class="hint">직캠 스테이지 윈도우 앱이 유튜브 원본 파일을 받아 줘요. 클립을 바탕화면에 띄우는 것도 이 앱이 해요.</p>
        <ol>
          <li><b>윈도우 앱 설치</b> <a href="${pcHelper.DESKTOP_APP_URL}" target="_blank" rel="noopener">여기</a>에서
            <code>FancamStage-Setup-….exe</code>를 받아 실행하세요. "Windows의 PC 보호" 창이 뜨면 <b>추가 정보 → 실행</b>.</li>
          <li><b>앱이 켜지면</b> 여기가 자동으로 바뀌어요. 처음 받을 때 필요한 프로그램(200MB 정도)을 받느라 몇 분 걸려요.</li>
        </ol>
        <p class="hint">켰는데도 그대로라면, 주소창 왼쪽 아이콘을 눌러 <b>로컬 네트워크 접근</b>을 허용해 주세요.
          앱을 설치하기 싫으면 <button id="helper-get" class="btn small" type="button">도우미 스크립트(.bat)</button>만 써도 돼요.</p>`;
      $('#helper-get').onclick = () => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(pcHelper.launcherFile());
        a.download = 'FancamStage-Helper.bat';
        document.body.append(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 60000);
      };
    } else if (state === 'ready') {
      box.innerHTML = head + `
        <p class="hint"><span class="ok-dot"></span>PC 도우미가 켜져 있어요. 유튜브에 있는 가장 좋은 화질(최대 4K)로 받아요.</p>
        <div class="row"><button id="helper-start" class="btn primary" type="button">원본 받기</button></div>`;
      $('#helper-start').onclick = () => this.download();
    } else if (state === 'working') {
      box.innerHTML = head + `
        <div class="bar"><span style="width:${(info.progress * 100).toFixed(1)}%"></span></div>
        <p class="mono helper-progress"></p>`;
      box.querySelector('.helper-progress').textContent = info.text;
    } else if (state === 'error') {
      box.innerHTML = head + '<p class="error"></p><div class="row"><button id="helper-retry" class="btn" type="button">다시 시도</button></div>';
      box.querySelector('.error').textContent = info.text;
      $('#helper-retry').onclick = () => this.show(this.url);
    }
  },

  async download() {
    if (this.busy) return;
    this.busy = true;
    keepAwake(true);
    let id = null;
    try {
      id = await pcHelper.startDownload(this.url);
      let st;
      for (;;) {
        await new Promise(r => setTimeout(r, 1000));
        st = await pcHelper.jobState(id);
        if (st.state === 'done') break;
        if (st.state === 'error') throw new Error(st.error.replace(/^ERROR:\s*/, ''));
        if (st.state === 'missing') throw new Error('도우미가 다시 시작돼서 작업이 사라졌어요.');
        const what = st.state === 'merging' ? '영상과 소리 합치는 중…'
          : `${st.part > 1 ? '소리' : '영상'} 받는 중 ${Math.round(st.percent)}%`;
        // Video is most of the bytes: count it as 0–85%, audio 85–95%, merging the rest.
        const progress = st.state === 'merging' ? 0.97 : (st.part > 1 ? 0.85 + st.percent / 1000 : st.percent / 100 * 0.85);
        this.render('working', { progress, text: (st.title ? st.title + '\n' : '') + what });
      }
      const file = await pcHelper.fetchFile(id, st.title, st.ext, f =>
        this.render('working', { progress: 1, text: `앱으로 옮기는 중 ${Math.round(f * 100)}%` }));
      pcHelper.finish(id);
      id = null;
      await create.load(file);
      const v = $('#src-video');
      if (Math.max(v.videoWidth, v.videoHeight) >= 2000) {
        $('#quality').value = '1920';
        toast('4K 원본이라 화질을 "FHD · 4K 원본용"으로 맞췄어요. 구간을 다듬고 멤버를 골라 주세요.', 4200);
      } else {
        toast('받았어요! 구간을 다듬고 멤버를 골라 주세요.', 3600);
      }
    } catch (err) {
      if (id) pcHelper.finish(id);
      this.render('error', { text: '받지 못했어요: ' + err.message });
    } finally {
      this.busy = false;
      keepAwake(false);
    }
  },
};

$('#yt-form').addEventListener('submit', e => {
  e.preventDefault();
  const id = parseYouTube($('#yt-url').value);
  if (!id) { toast('유튜브 영상 링크가 아니에요. 공유 버튼으로 복사한 링크를 붙여 넣어 주세요.', 4000); return; }
  capture.showHelp(id);
});

// ---------------------------------------------------------------------------
// Stage: the "wallpaper" where clips dance

const BACKGROUNDS = [
  { id: 'night', name: '밤 무대', css: 'radial-gradient(80% 60% at 50% 100%, #3a2a4a 0%, #16121f 60%, #0b0910 100%)' },
  { id: 'spot', name: '스포트라이트', css: 'radial-gradient(40% 70% at 50% 35%, rgba(255,236,214,.35), transparent 70%), linear-gradient(#07060a, #1c1726)' },
  { id: 'dawn', name: '새벽', css: 'linear-gradient(160deg, #ffd6e0 0%, #c9d8ff 55%, #a6c8ff 100%)' },
  { id: 'mint', name: '민트', css: 'linear-gradient(180deg, #d9fff3, #a8e8dc)' },
  { id: 'black', name: '검정', css: '#000' },
  { id: 'white', name: '흰색', css: '#fff' },
  { id: 'green', name: '그린스크린', css: '#00ff00' },
  { id: 'blue', name: '블루스크린', css: '#0047ff' },
];

const stage = {
  state: { bg: 'night', color: '#222222', sprites: [] },
  sprites: [],          // { s: state entry, el, clip: AlphaClip, data }
  selected: null,
  loaded: false,
  playing: true,
  bgUrl: null,

  async load() {
    if (this.loaded) return;
    this.loaded = true;
    try {
      const saved = await kv.get('stage');
      if (saved) this.state = { ...this.state, ...saved };
    } catch {}
  },

  save() {
    clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(() => kv.set('stage', this.state).catch(() => {}), 300);
  },

  async enter() {
    await this.load();
    await this.applyBackground();
    await this.mount();
    this.playing = true;
    this.updatePlayButton();
    keepAwake(true);
    this.poke();
    if (!this.resizeObserver) {
      this.resizeObserver = new ResizeObserver(() => this.layoutAll());
      this.resizeObserver.observe($('#stage'));
    }
  },

  leave() {
    keepAwake(false);
    this.closeSheet();
    this.select(null);
    for (const sp of this.sprites) { sp.clip.destroy(); sp.el.remove(); }
    this.sprites = [];
    if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
  },

  async mount() {
    for (const sp of this.sprites) { sp.clip.destroy(); sp.el.remove(); }
    this.sprites = [];
    const keep = [];
    for (const s of this.state.sprites) {
      const data = await clips.get(s.clipId).catch(() => null);
      if (!data) continue;
      keep.push(s);
      this.spawn(s, data);
    }
    this.state.sprites = keep;
    $('#stage-empty').hidden = keep.length > 0;
  },

  spawn(s, data) {
    const clip = new AlphaClip(data);
    const el = document.createElement('div');
    el.className = 'sprite';
    el.append(clip.canvas);
    $('#stage').append(el);
    clip.setFlip(!!s.flip);
    clip.video.playbackRate = s.speed || 1;
    clip.video.muted = true; // autoplay needs muted; the user turns sound on with a tap
    const sp = { s, el, clip, data };
    this.sprites.push(sp);
    this.layout(sp);
    this.bindGestures(sp);
    if (this.playing) clip.play();
    return sp;
  },

  async addAndOpen(clipId) {
    await this.load();
    const n = this.state.sprites.length;
    this.state.sprites.push({ clipId, x: 0.5 + ((n % 3) - 1) * 0.22 * Math.min(1, n), y: 0.55, h: 0.85, flip: false, speed: 1 });
    this.save();
    if (current === 'stage') {
      const data = await clips.get(clipId);
      const sp = this.spawn(this.state.sprites.at(-1), data);
      $('#stage-empty').hidden = true;
      this.select(sp);
    } else {
      go('stage');
    }
  },

  forget(clipId) {
    this.state.sprites = this.state.sprites.filter(s => s.clipId !== clipId);
    this.save();
  },

  layout(sp) {
    const st = $('#stage').getBoundingClientRect();
    const h = sp.s.h * st.height;
    const w = h * sp.clip.aspect;
    Object.assign(sp.el.style, {
      width: w + 'px', height: h + 'px',
      left: (sp.s.x * st.width - w / 2) + 'px', top: (sp.s.y * st.height - h / 2) + 'px',
    });
    sp.clip.resize(w, h);
  },
  layoutAll() { for (const sp of this.sprites) this.layout(sp); },

  bindGestures(sp) {
    const pts = new Map();
    let startDist = 0, startH = 0, last = null;
    const stageRect = () => $('#stage').getBoundingClientRect();
    const mid = () => {
      const a = [...pts.values()];
      return { x: a.reduce((s, p) => s + p.x, 0) / a.length, y: a.reduce((s, p) => s + p.y, 0) / a.length };
    };
    const dist = () => { const [a, b] = [...pts.values()]; return Math.hypot(a.x - b.x, a.y - b.y); };

    sp.el.addEventListener('pointerdown', e => {
      e.preventDefault();
      e.stopPropagation();
      sp.el.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      this.select(sp);
      this.poke();
      last = mid();
      if (pts.size === 2) { startDist = dist(); startH = sp.s.h; }
    });
    sp.el.addEventListener('pointermove', e => {
      if (!pts.has(e.pointerId)) return;
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      const r = stageRect();
      const m = mid();
      sp.s.x += (m.x - last.x) / r.width;
      sp.s.y += (m.y - last.y) / r.height;
      last = m;
      if (pts.size === 2 && startDist > 0) sp.s.h = Math.min(3, Math.max(0.1, startH * dist() / startDist));
      this.layout(sp);
    });
    const end = e => {
      if (!pts.delete(e.pointerId)) return;
      last = pts.size ? mid() : null;
      if (pts.size === 2) { startDist = dist(); startH = sp.s.h; } else startDist = 0;
      this.save();
    };
    sp.el.addEventListener('pointerup', end);
    sp.el.addEventListener('pointercancel', end);
    sp.el.addEventListener('wheel', e => {
      e.preventDefault();
      sp.s.h = Math.min(3, Math.max(0.1, sp.s.h * Math.exp(-e.deltaY * 0.0015)));
      this.layout(sp);
      this.save();
    }, { passive: false });
  },

  select(sp) {
    for (const o of this.sprites) o.el.classList.toggle('selected', o === sp);
    this.selected = sp;
    $('#sprite-bar').hidden = !sp;
    if (sp) {
      pcHelper.desktopRunning().then(on => { $('[data-sprite=desktop]').hidden = !on; });
      $('#sprite-speed').value = String(sp.s.speed || 1);
      const snd = $('#sprite-sound');
      snd.hidden = !sp.data.hasAudio;
      snd.textContent = sp.clip.video.muted ? '소리 켜기' : '소리 끄기';
    }
  },

  // Show the controls for a moment; in fullscreen they fade away on their own.
  poke() {
    const view = views.stage;
    view.classList.remove('ui-hidden');
    $('#stage').classList.remove('immersive');
    clearTimeout(this.hideTimer);
    this.hideTimer = setTimeout(() => {
      if (!$('#sheet').hidden) return;
      view.classList.add('ui-hidden');
      $('#stage').classList.add('immersive');
    }, 3500);
  },

  async applyBackground() {
    const st = $('#stage');
    if (this.bgUrl) { URL.revokeObjectURL(this.bgUrl); this.bgUrl = null; }
    const { bg } = this.state;
    if (bg === 'image') {
      const blob = await kv.get('bgImage').catch(() => null);
      if (blob) {
        this.bgUrl = URL.createObjectURL(blob);
        st.style.background = `center / cover no-repeat url("${this.bgUrl}"), #000`;
        return;
      }
    }
    if (bg === 'color') { st.style.background = this.state.color; return; }
    st.style.background = (BACKGROUNDS.find(b => b.id === bg) || BACKGROUNDS[0]).css;
  },

  openSheet(title, content) {
    $('#sheet-title').textContent = title;
    $('#sheet-body').replaceChildren(content);
    $('#sheet').hidden = false;
    this.poke();
  },
  closeSheet() { $('#sheet').hidden = true; },

  async showAddSheet() {
    const list = await clips.all();
    const wrap = document.createElement('div');
    if (!list.length) {
      wrap.innerHTML = '<p class="hint">아직 클립이 없어요. 보관함에서 직캠 영상으로 먼저 만들어 주세요.</p>';
      const b = document.createElement('button');
      b.className = 'btn primary'; b.type = 'button'; b.textContent = '클립 만들러 가기';
      b.onclick = () => go('create');
      wrap.append(b);
    } else {
      wrap.className = 'pick-list';
      for (const c of list) {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'pick-item';
        const img = document.createElement('img');
        img.className = 'checker';
        if (c.thumbnail) img.src = URL.createObjectURL(c.thumbnail);
        const span = document.createElement('span');
        span.textContent = c.name;
        b.append(img, span);
        b.onclick = () => { this.closeSheet(); this.addAndOpen(c.id); };
        wrap.append(b);
      }
    }
    this.openSheet('클립 올리기', wrap);
  },

  showBgSheet() {
    const wrap = document.createElement('div');
    wrap.style.display = 'grid'; wrap.style.gap = '14px';
    const sw = document.createElement('div');
    sw.className = 'swatches';
    for (const b of BACKGROUNDS) {
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'swatch' + (this.state.bg === b.id ? ' on' : '');
      el.style.background = b.css;
      el.textContent = b.name;
      el.onclick = () => { this.state.bg = b.id; this.save(); this.applyBackground(); this.showBgSheet(); };
      sw.append(el);
    }
    const img = document.createElement('button');
    img.type = 'button';
    img.className = 'swatch' + (this.state.bg === 'image' ? ' on' : '');
    img.style.background = 'repeating-linear-gradient(45deg, #2e2a3b 0 8px, #1b1825 8px 16px)';
    img.textContent = '내 사진…';
    img.onclick = () => $('#bg-input').click();
    sw.append(img);

    const row = document.createElement('label');
    row.className = 'swatch-color';
    row.innerHTML = '<input type="color" id="bg-color"> 단색 직접 고르기';
    const input = row.querySelector('input');
    input.value = this.state.color;
    input.oninput = () => { this.state.bg = 'color'; this.state.color = input.value; this.save(); this.applyBackground(); };

    const tip = document.createElement('p');
    tip.className = 'hint';
    tip.textContent = '그린스크린/블루스크린은 화면 녹화 후 다른 편집 앱에서 크로마키로 배경을 지울 때 써요.';
    wrap.append(sw, row, tip);
    this.openSheet('배경', wrap);
  },

  updatePlayButton() { $('#btn-play').textContent = this.playing ? '일시정지' : '재생'; },

  togglePlay() {
    this.playing = !this.playing;
    for (const sp of this.sprites) this.playing ? sp.clip.play() : sp.clip.video.pause();
    this.updatePlayButton();
  },

  async fullscreen() {
    const el = document.documentElement;
    try {
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        await (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      } else if (el.requestFullscreen || el.webkitRequestFullscreen) {
        await (el.requestFullscreen || el.webkitRequestFullscreen).call(el, { navigationUI: 'hide' });
      } else {
        toast('iPhone은 공유 → "홈 화면에 추가"로 설치하면 전체화면으로 열려요.', 4200);
      }
    } catch {
      toast('전체화면을 열 수 없어요.');
    }
  },

  // Desktop Chrome/Edge: pop the stage into an always-on-top mini window.
  async pip() {
    if (this.pipWindow) { this.pipWindow.close(); return; }
    try {
      const st = $('#stage');
      const r = st.getBoundingClientRect();
      const pw = await documentPictureInPicture.requestWindow({
        width: Math.round(Math.min(480, r.width / 2)), height: Math.round(Math.min(720, r.height / 1.5)),
      });
      this.pipWindow = pw;
      for (const link of document.querySelectorAll('link[rel=stylesheet], style')) pw.document.head.append(link.cloneNode(true));
      pw.document.body.style.cssText = 'margin:0;height:100vh;overflow:hidden;background:#000';
      const placeholder = document.createElement('div');
      placeholder.className = 'stage-empty';
      placeholder.innerHTML = '<p>작은 창에서 재생 중이에요.</p>';
      st.before(placeholder);
      pw.document.body.append(st);
      const ro = new pw.ResizeObserver(() => this.layoutAll());
      ro.observe(st);
      pw.addEventListener('pagehide', () => {
        ro.disconnect();
        placeholder.replaceWith(st);
        this.pipWindow = null;
        this.layoutAll();
      });
      this.layoutAll();
    } catch (err) {
      toast('작은 창을 열 수 없어요: ' + err.message);
    }
  },
};

$('#stage').addEventListener('pointerdown', e => {
  if (e.target.closest('.sprite')) return;
  stage.select(null);
  if (views.stage.classList.contains('ui-hidden')) stage.poke();
  else if (!$('#sheet').hidden) stage.closeSheet();
});
views.stage.addEventListener('pointermove', e => { if (e.pointerType === 'mouse') stage.poke(); });
views.stage.addEventListener('click', e => {
  const a = e.target.closest('[data-action]');
  if (!a) return;
  stage.poke();
  ({
    add: () => stage.showAddSheet(),
    bg: () => stage.showBgSheet(),
    play: () => stage.togglePlay(),
    fullscreen: () => stage.fullscreen(),
    pip: () => stage.pip(),
    'close-sheet': () => stage.closeSheet(),
  })[a.dataset.action]?.();
});
$('#sprite-bar').addEventListener('click', e => {
  const b = e.target.closest('[data-sprite]');
  const sp = stage.selected;
  if (!b || !sp) return;
  stage.poke();
  const act = b.dataset.sprite;
  if (act === 'flip') { sp.s.flip = !sp.s.flip; sp.clip.setFlip(sp.s.flip); }
  if (act === 'sound') {
    sp.clip.video.muted = !sp.clip.video.muted;
    if (!sp.clip.video.muted) sp.clip.play();
    b.textContent = sp.clip.video.muted ? '소리 켜기' : '소리 끄기';
  }
  if (act === 'desktop') sendToDesktop(sp.data, b);
  if (act === 'front') {
    $('#stage').append(sp.el);
    stage.state.sprites = [...stage.state.sprites.filter(s => s !== sp.s), sp.s];
  }
  if (act === 'remove') {
    sp.clip.destroy(); sp.el.remove();
    stage.sprites = stage.sprites.filter(o => o !== sp);
    stage.state.sprites = stage.state.sprites.filter(s => s !== sp.s);
    stage.select(null);
    $('#stage-empty').hidden = stage.sprites.length > 0;
  }
  stage.save();
});
$('#sprite-speed').addEventListener('change', e => {
  const sp = stage.selected;
  if (!sp) return;
  sp.s.speed = Number(e.target.value);
  sp.clip.video.playbackRate = sp.s.speed;
  stage.save();
});
$('#bg-input').addEventListener('change', async e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  await kv.set('bgImage', f);
  stage.state.bg = 'image';
  stage.save();
  await stage.applyBackground();
  stage.showBgSheet();
});
document.addEventListener('keydown', e => {
  if (current !== 'stage' || e.target.matches('input, select, textarea')) return;
  if (e.key === ' ') { e.preventDefault(); stage.togglePlay(); }
  if (e.key === 'f') stage.fullscreen();
  if ((e.key === 'Delete' || e.key === 'Backspace') && stage.selected) $('[data-sprite=remove]').click();
});

// ---------------------------------------------------------------------------
// Boot

if ('documentPictureInPicture' in window && matchMedia('(pointer: fine)').matches) $('#btn-pip').hidden = false;
if (typeof VideoEncoder === 'undefined') {
  const w = $('#support-warning');
  w.hidden = false;
  w.textContent = '이 브라우저에선 새 클립을 만들 수 없어요(보기만 가능). 최신 Chrome, Edge, Safari(iOS 16.4 이상)를 써 주세요.';
}
if ('serviceWorker' in navigator && location.protocol !== 'file:') {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
// "Connect another device" link: #gdrive=<client id>
if (location.hash.startsWith('#gdrive=')) {
  const id = decodeURIComponent(location.hash.slice('#gdrive='.length));
  history.replaceState(null, '', '#library');
  if (drive.looksLikeClientId(id)) {
    drive.setClientId(id);
    setTimeout(() => {
      $('#sync-panel').hidden = false;
      cloud.renderPanel();
      toast('"구글로 로그인"을 누르면 이 기기도 연결돼요.', 4000);
    });
  }
}
go(location.hash.slice(1) || 'library');
cloud.kick();
