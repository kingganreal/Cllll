// Colab through Google Drive, no tunnel (colab/server.py watch_drive, notebook 5-C): jobs go into
// 내 드라이브\직캠 스테이지\코랩\jobs\<id>\ in the folder Google Drive for desktop keeps on this PC,
// Colab picks them up from there and puts the finished clip back next to them.
//   GET    /drive/status              { folder, colab: { alive, age, gpu } }
//   PUT    /drive/jobs/<id>/source    the video (unless Colab gets it from YouTube itself)
//   POST   /drive/jobs/<id>           the job (JSON), written last: Colab starts on it
//   GET    /drive/jobs/<id>           its status.json ({ state: 'waiting' } until Colab answers)
//   POST   /drive/jobs/<id>/stop      finish early, keeping what's made
//   DELETE /drive/jobs/<id>           cancel (or clean up after taking the clip)
//   GET    /drive/jobs/<id>/result    the finished clip
const fs = require('fs');
const os = require('os');
const path = require('path');
const { pipeline } = require('stream/promises');

const SUB = path.join('직캠 스테이지', '코랩');

/** The "My Drive" folder of Google Drive for desktop on this PC, or null. */
function driveRoot() {
  const names = ['My Drive', '내 드라이브'];
  const tries = [];
  if (process.env.FANCAM_DRIVE) tries.push(process.env.FANCAM_DRIVE);
  for (const letter of 'GHIJKLMNOPQRSTUVWXYZDEF') for (const n of names) tries.push(`${letter}:\\${n}`);
  const home = os.homedir();
  for (const n of names) tries.push(path.join(home, n), path.join(home, 'Google Drive', n));
  tries.push(path.join(home, 'Google Drive'));
  return tries.find(p => { try { return fs.statSync(p).isDirectory(); } catch { return false; } }) || null;
}

const jobsDir = () => { const r = driveRoot(); return r && path.join(r, SUB, 'jobs'); };
const jobDir = id => { const j = jobsDir(); if (!j) throw new Error('PC에서 구글 드라이브 폴더를 찾지 못했어요. 구글 드라이브 앱을 켜고 로그인해 주세요.'); return path.join(j, id); };
const readJson = file => { try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch { return null; } };

function status() {
  const root = driveRoot();
  if (!root) return { folder: null, colab: null };
  const alive = readJson(path.join(root, SUB, 'alive.json'));
  const age = alive ? Date.now() / 1000 - alive.time : null;
  // Google Drive brings the file down with some delay, so a few minutes still counts as on.
  return { folder: path.join(root, SUB), colab: alive ? { alive: age < 300, age: Math.round(age), gpu: alive.gpu } : null };
}

const cleanups = new Map();
function later(id, ms) {
  clearTimeout(cleanups.get(id));
  cleanups.set(id, setTimeout(() => { cleanups.delete(id); try { fs.rmSync(jobDir(id), { recursive: true, force: true }); } catch {} }, ms));
}

/** Handles /drive/… requests; returns false for other paths. */
async function route(req, res, p, send, readJsonBody) {
  if (!p.startsWith('/drive/')) return false;
  if (p === '/drive/status') { send(200, status()); return true; }
  const m = p.match(/^\/drive\/jobs\/([\w-]{1,64})(\/source|\/stop|\/result)?$/);
  if (!m) { send(404, { error: 'not found' }); return true; }
  const [, id, sub] = m, d = jobDir(id);
  if (sub === '/source' && req.method === 'PUT') {
    fs.mkdirSync(d, { recursive: true });
    const tmp = path.join(d, 'source.uploading');
    await pipeline(req, fs.createWriteStream(tmp));
    fs.renameSync(tmp, path.join(d, 'source.mp4'));
    send(200, { ok: true, size: fs.statSync(path.join(d, 'source.mp4')).size });
  } else if (!sub && req.method === 'POST') {
    const job = await readJsonBody(req);
    fs.mkdirSync(d, { recursive: true });
    const src = path.join(d, 'source.mp4');
    if (!job.url) job.sourceSize = fs.statSync(src).size;
    job.created = Date.now();
    fs.writeFileSync(path.join(d, 'job.json.tmp'), JSON.stringify(job));
    fs.renameSync(path.join(d, 'job.json.tmp'), path.join(d, 'job.json'));
    send(200, { ok: true });
  } else if (!sub && req.method === 'GET') {
    send(200, readJson(path.join(d, 'status.json')) || { state: 'waiting' });
  } else if (sub === '/stop' && req.method === 'POST') {
    fs.writeFileSync(path.join(d, 'stop'), '');
    send(200, { ok: true });
  } else if (!sub && req.method === 'DELETE') {
    const st = readJson(path.join(d, 'status.json'));
    if (st && ['done', 'error', 'cancelled'].includes(st.state)) later(id, 1000);
    else {
      try { fs.writeFileSync(path.join(d, 'cancel'), ''); } catch {}
      later(id, 5 * 60 * 1000); // after Colab has seen it
    }
    send(200, { ok: true });
  } else if (sub === '/result' && req.method === 'GET') {
    const st = readJson(path.join(d, 'status.json'));
    const file = st?.file && path.join(d, path.basename(st.file));
    if (!file || !fs.existsSync(file)) { send(404, { error: '아직 클립이 PC로 안 내려왔어요.' }); return true; }
    // (reading it is what makes Google Drive bring it down, when it streams files)
    res.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Content-Length': fs.statSync(file).size });
    await pipeline(fs.createReadStream(file), res).catch(() => {});
  } else {
    send(405, { error: 'method' });
  }
  return true;
}

module.exports = { route, status, driveRoot };
