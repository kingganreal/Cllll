// Clips sent from the web app, kept as files in 문서\직캠 스테이지 (the web app backs every clip up
// here and takes them back if the browser's copy is gone), and the characters on the desktop
// (kept in the app-data folder).
const fs = require('fs');
const path = require('path');
const { app } = require('electron');

const root = () => app.getPath('userData');
const clipsDir = () => path.join(app.getPath('documents'), '직캠 스테이지');
const oldClipsDir = () => path.join(root(), 'clips'); // before 0.1.2
const stateFile = () => path.join(root(), 'state.json');

let moved = false;
function ensureDirs() {
  fs.mkdirSync(clipsDir(), { recursive: true });
  if (moved) return;
  moved = true;
  // Bring clips over from where older versions kept them.
  try {
    for (const f of fs.readdirSync(oldClipsDir())) {
      const to = path.join(clipsDir(), f);
      if (!fs.existsSync(to)) fs.copyFileSync(path.join(oldClipsDir(), f), to);
    }
    fs.rmSync(oldClipsDir(), { recursive: true, force: true });
  } catch {}
}

const clipPath = (id, ext) => path.join(clipsDir(), `${id}.${ext}`);
const safeId = id => /^[\w-]{1,64}$/.test(id || '');

function listClips() {
  ensureDirs();
  const out = [];
  for (const f of fs.readdirSync(clipsDir())) {
    if (!f.endsWith('.json')) continue;
    try {
      const meta = JSON.parse(fs.readFileSync(path.join(clipsDir(), f), 'utf8'));
      if (fs.existsSync(clipPath(meta.id, 'mp4'))) out.push(meta);
    } catch {}
  }
  return out.sort((a, b) => (b.received || 0) - (a.received || 0));
}

function getClip(id) {
  if (!safeId(id)) return null;
  try { return JSON.parse(fs.readFileSync(clipPath(id, 'json'), 'utf8')); } catch { return null; }
}

function saveMeta(meta) {
  ensureDirs();
  fs.writeFileSync(clipPath(meta.id, 'json'), JSON.stringify(meta, null, 1));
}

function deleteClip(id) {
  if (!safeId(id)) return;
  for (const ext of ['mp4', 'png', 'json', 'part']) fs.rmSync(clipPath(id, ext), { force: true });
}

let state = null, saveTimer = null;
function loadState() {
  if (state) return state;
  try { state = JSON.parse(fs.readFileSync(stateFile(), 'utf8')); } catch { state = {}; }
  state.overlays ??= [];
  return state;
}
function saveState() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { fs.writeFileSync(stateFile(), JSON.stringify(loadState(), null, 1)); } catch {}
  }, 300);
}

module.exports = { clipsDir, clipPath, safeId, listClips, getClip, saveMeta, deleteClip, loadState, saveState, ensureDirs };
