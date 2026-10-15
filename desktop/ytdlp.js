// YouTube downloads in original quality (up to 4K) with yt-dlp. The tools are fetched on first use
// into the app-data folder: yt-dlp, ffmpeg (joins 4K video and audio) and deno (yt-dlp needs it
// for YouTube). Same job API as the old PowerShell helper, so the web app works with either.
const fs = require('fs');
const path = require('path');
const { spawn, execFile } = require('child_process');
const { Readable } = require('stream');
const { pipeline } = require('stream/promises');
const { app } = require('electron');

const isWin = process.platform === 'win32';
const exe = name => name + (isWin ? '.exe' : '');
const binDir = () => process.env.FANCAM_BIN_DIR || path.join(app.getPath('userData'), 'bin');
const dlDir = () => path.join(app.getPath('userData'), 'downloads');

const TOOLS = [
  { file: 'yt-dlp', url: 'https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe' },
  { file: 'ffmpeg', zip: 'https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip', also: ['ffprobe'] },
  { file: 'deno', zip: 'https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip' },
];

let toolsPromise = null;
let toolsStatus = '';
let updated = false;

async function download(url, dest) {
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`받기 실패 (${res.status}) ${url}`);
  const tmp = dest + '.part';
  await pipeline(Readable.fromWeb(res.body), fs.createWriteStream(tmp));
  fs.renameSync(tmp, dest);
}

function findFile(dir, name) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) { const f = findFile(p, name); if (f) return f; }
    else if (e.name.toLowerCase() === name.toLowerCase()) return p;
  }
  return null;
}

function unzip(zip, dir) {
  return new Promise((resolve, reject) => {
    execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
      `Expand-Archive -LiteralPath '${zip.replace(/'/g, "''")}' -DestinationPath '${dir.replace(/'/g, "''")}' -Force`],
    { windowsHide: true }, err => (err ? reject(err) : resolve()));
  });
}

function ensureTools() {
  toolsPromise ??= (async () => {
    fs.mkdirSync(binDir(), { recursive: true });
    fs.mkdirSync(dlDir(), { recursive: true });
    for (const t of TOOLS) {
      const target = path.join(binDir(), exe(t.file));
      if (fs.existsSync(target)) continue;
      if (!isWin) throw new Error('유튜브 받기는 윈도우에서만 돼요.');
      toolsStatus = `${t.file} 준비 중 (처음 한 번)`;
      if (t.url) {
        await download(t.url, target);
      } else {
        const tmp = fs.mkdtempSync(path.join(binDir(), 'tmp-'));
        try {
          const zip = path.join(tmp, 'a.zip');
          await download(t.zip, zip);
          await unzip(zip, tmp);
          for (const name of [t.file, ...(t.also || [])]) {
            const f = findFile(tmp, exe(name));
            if (!f) throw new Error(`${name}을(를) 찾지 못했어요.`);
            fs.copyFileSync(f, path.join(binDir(), exe(name)));
          }
        } finally {
          fs.rmSync(tmp, { recursive: true, force: true });
        }
      }
    }
    toolsStatus = '';
    // YouTube changes often; keep yt-dlp current (once per app start, in the background).
    if (!updated) {
      updated = true;
      spawn(path.join(binDir(), exe('yt-dlp')), ['-U'], { windowsHide: true, stdio: 'ignore' }).on('error', () => {});
    }
  })().catch(err => { toolsPromise = null; toolsStatus = ''; throw err; });
  return toolsPromise;
}

// Old downloads that were never picked up.
function cleanOld() {
  try {
    for (const f of fs.readdirSync(dlDir())) {
      const p = path.join(dlDir(), f);
      if (Date.now() - fs.statSync(p).mtimeMs > 86400000) fs.rmSync(p, { force: true });
    }
  } catch {}
}

const jobs = new Map();

function start(url) {
  const id = Date.now().toString() + Math.floor(Math.random() * 1000);
  const job = { id, lines: [], part: 0, percent: 0, merging: false, exit: null, error: '', preparing: true, stderr: '' };
  jobs.set(id, job);
  (async () => {
    try {
      await ensureTools();
    } catch (err) {
      job.exit = -1;
      job.error = '도구를 준비하지 못했어요: ' + err.message;
      return;
    }
    job.preparing = false;
    cleanOld();
    const titleFile = path.join(dlDir(), `${id}.title.txt`);
    job.titleFile = titleFile;
    const args = [
      '--newline', '--no-playlist', '--no-mtime', '--encoding', 'utf-8',
      // Best video up to 4K plus best audio. Prefer VP9 over AV1: browsers decode it faster.
      '-f', 'bv*+ba/b', '-S', 'res:2160,fps,vcodec:vp9,acodec:opus',
      '--merge-output-format', 'mp4',
      '--print-to-file', '%(title)s', titleFile,
      '-o', path.join(dlDir(), `${id}.%(ext)s`),
      url,
    ];
    const env = { ...process.env, PATH: binDir() + path.delimiter + (process.env.PATH || '') };
    const p = spawn(path.join(binDir(), exe('yt-dlp')), args, { env, windowsHide: true });
    job.proc = p;
    let buf = '';
    p.stdout.on('data', d => {
      buf += d.toString('utf8');
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (/^\[download\] Destination:/.test(line)) { job.part++; job.percent = 0; }
        const m = line.match(/^\[download\]\s+([\d.]+)%/);
        if (m) job.percent = Number(m[1]);
        if (/^\[Merger\]/.test(line)) job.merging = true;
      }
    });
    p.stderr.on('data', d => { job.stderr = (job.stderr + d.toString('utf8')).slice(-4000); });
    p.on('error', err => { job.exit = -1; job.error = err.message; });
    p.on('close', code => {
      job.exit = code;
      if (code !== 0) {
        const errLine = job.stderr.split(/\r?\n/).filter(l => /ERROR/.test(l)).pop();
        job.error = errLine || `yt-dlp가 실패했어요 (코드 ${code})`;
      }
    });
  })();
  return id;
}

function outputFile(id) {
  try {
    const f = fs.readdirSync(dlDir()).find(n => ['mp4', 'webm', 'mkv'].some(ext => n === `${id}.${ext}`));
    return f ? path.join(dlDir(), f) : null;
  } catch { return null; }
}

function title(job) {
  try { return fs.readFileSync(job.titleFile, 'utf8').split(/\r?\n/).find(Boolean) || ''; } catch { return ''; }
}

function state(id) {
  const job = jobs.get(id);
  if (!job) return { state: 'missing' };
  if (job.preparing && job.exit === null) return { state: 'downloading', part: 1, percent: 0, title: toolsStatus || '준비 중…' };
  if (job.exit !== null) {
    const file = job.exit === 0 && outputFile(id);
    if (file) return { state: 'done', title: title(job), size: fs.statSync(file).size, ext: path.extname(file).slice(1) };
    return { state: 'error', error: job.error || '받지 못했어요.', title: title(job) };
  }
  if (job.merging) return { state: 'merging', title: title(job) };
  return { state: 'downloading', part: Math.max(1, job.part), percent: job.percent, title: title(job) };
}

function finish(id) {
  const job = jobs.get(id);
  if (job?.proc && job.exit === null) { try { job.proc.kill(); } catch {} }
  try { for (const f of fs.readdirSync(dlDir())) if (f.startsWith(id + '.')) fs.rmSync(path.join(dlDir(), f), { force: true }); } catch {}
  jobs.delete(id);
}

module.exports = { start, state, outputFile, finish, ensureTools };
