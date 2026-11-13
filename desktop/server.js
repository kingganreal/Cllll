// Local server on http://localhost:8790 for the web app (open in Chrome on the same PC):
//   /status                         is the app running?
//   /download /job /file /done      YouTube originals through yt-dlp (see ytdlp.js)
//   PUT /clips/<id>?meta=<json>     receive a clip (body: packed MP4), kept as a file
//   PUT /clips/<id>/thumb           receive its thumbnail (body: PNG)
//   POST /clips/<id>/show           put it on the desktop
//   GET /clips                      every clip kept here (details)
//   GET /clips/<id>/video, /thumb   a clip's files, to restore it into the web app
//   /drive/…                        Colab through Google Drive (see drivejobs.js)
const http = require('http');
const fs = require('fs');
const { pipeline } = require('stream/promises');
const store = require('./store');
const ytdlp = require('./ytdlp');
const drivejobs = require('./drivejobs');
const { ALLOWED_ORIGINS, PORT } = require('./config');

const API_VERSION = 2;

function allowed(origin) {
  if (!origin) return true; // not from a web page
  return ALLOWED_ORIGINS.includes(origin) || /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin);
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.setEncoding('utf8');
    req.on('data', d => { body += d; if (body.length > 1e5) req.destroy(); });
    req.on('end', () => { try { resolve(body ? JSON.parse(body) : {}); } catch (e) { reject(e); } });
    req.on('error', reject);
  });
}

function start({ appVersion, onClipsChanged, onShow }) {
  const server = http.createServer(async (req, res) => {
    const origin = req.headers.origin;
    const send = (code, obj) => {
      const body = Buffer.from(JSON.stringify(obj));
      res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': body.length });
      res.end(body);
    };
    if (!allowed(origin)) return send(403, { error: '허용되지 않은 사이트예요.' });
    if (origin) {
      res.setHeader('Access-Control-Allow-Origin', origin);
      res.setHeader('Vary', 'Origin');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
      res.setHeader('Access-Control-Allow-Private-Network', 'true');
    }
    if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }

    const url = new URL(req.url, 'http://localhost');
    const p = url.pathname;
    const id = url.searchParams.get('id');
    let m;
    try {
      if (p === '/status') return send(200, { ok: true, version: API_VERSION, desktop: true, appVersion });

      // --- YouTube ---
      if (p === '/download' && req.method === 'POST') {
        const { url: link } = await readJson(req);
        if (!/^https:\/\/(www\.|m\.|music\.)?(youtube\.com|youtu\.be)\//.test(link || '')) return send(400, { error: '유튜브 링크만 받을 수 있어요.' });
        return send(200, { id: ytdlp.start(link) });
      }
      if (p === '/job') return send(200, ytdlp.state(id));
      if (p === '/file') {
        const file = id && ytdlp.outputFile(id);
        if (!file) return send(404, { error: '파일이 없어요.' });
        const size = fs.statSync(file).size;
        res.writeHead(200, { 'Content-Type': file.endsWith('.webm') ? 'video/webm' : 'video/mp4', 'Content-Length': size });
        return pipeline(fs.createReadStream(file), res).catch(() => {});
      }
      if (p === '/done') { ytdlp.finish(id); return send(200, { ok: true }); }

      // --- Clips for the desktop ---
      if ((m = p.match(/^\/clips\/([\w-]{1,64})$/)) && req.method === 'PUT') {
        const clipId = m[1];
        const meta = JSON.parse(url.searchParams.get('meta') || '{}');
        const tmp = store.clipPath(clipId, 'part');
        store.ensureDirs();
        await pipeline(req, fs.createWriteStream(tmp));
        fs.renameSync(tmp, store.clipPath(clipId, 'mp4'));
        store.saveMeta({
          id: clipId, name: String(meta.name || '클립'), layout: ['tb', 'circle'].includes(meta.layout) ? meta.layout : 'sbs',
          crop: meta.crop || { x: 0, y: 0, w: 1, h: 1 }, width: Number(meta.width) || 0, height: Number(meta.height) || 0,
          duration: Number(meta.duration) || 0, hasAudio: !!meta.hasAudio, received: Date.now(),
          created: Number(meta.created) || Date.now(), fps: Number(meta.fps) || 0, codec: String(meta.codec || ''),
        });
        onClipsChanged();
        return send(200, { ok: true });
      }
      if ((m = p.match(/^\/clips\/([\w-]{1,64})\/thumb$/)) && req.method === 'PUT') {
        await pipeline(req, fs.createWriteStream(store.clipPath(m[1], 'png')));
        onClipsChanged();
        return send(200, { ok: true });
      }
      if ((m = p.match(/^\/clips\/([\w-]{1,64})\/show$/)) && req.method === 'POST') {
        if (!store.getClip(m[1])) return send(404, { error: '앱에 없는 클립이에요.' });
        onShow(m[1]);
        return send(200, { ok: true });
      }
      if (p === '/clips' && req.method === 'GET') {
        return send(200, { clips: store.listClips() });
      }
      if ((m = p.match(/^\/clips\/([\w-]{1,64})\/(video|thumb)$/)) && req.method === 'GET') {
        const file = store.clipPath(m[1], m[2] === 'video' ? 'mp4' : 'png');
        if (!store.getClip(m[1]) || !fs.existsSync(file)) return send(404, { error: '파일이 없어요.' });
        res.writeHead(200, { 'Content-Type': m[2] === 'video' ? 'video/mp4' : 'image/png', 'Content-Length': fs.statSync(file).size });
        return pipeline(fs.createReadStream(file), res).catch(() => {});
      }
      if (await drivejobs.route(req, res, p, send, readJson)) return;
      return send(404, { error: 'not found' });
    } catch (err) {
      console.error(err);
      if (!res.headersSent) send(500, { error: err.message });
    }
  });
  server.requestTimeout = 0; // big uploads take a while
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(PORT, '127.0.0.1', () => {
      // Chrome may try ::1 for "localhost" first; listen there too when possible.
      const v6 = http.createServer(server.listeners('request')[0]);
      v6.on('error', () => {});
      v6.requestTimeout = 0;
      v6.listen(PORT, '::1');
      resolve(server);
    });
  });
}

module.exports = { start };
