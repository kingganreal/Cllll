// Making clips on a Colab GPU instead of this device: the notebook (colab/server.py) prints an
// address like https://….trycloudflare.com/#key=…; with it saved here, 추출 sends the video and
// the pick there and brings the finished clip back. Picking, the stage and everything else stay
// in the app.

const LS = 'fancam-colab';
const PART = 32 * 1024 * 1024; // a Cloudflare quick tunnel takes at most 100 MB per request

export function saved() {
  try { return localStorage.getItem(LS) || ''; } catch { return ''; }
}
export function save(text) {
  try { text ? localStorage.setItem(LS, text.trim()) : localStorage.removeItem(LS); } catch {}
}

/** { base, bases, key } from the address the notebook printed, or null. Newer notebooks open
 * several tunnels (one is slow) and list the others after &more=; pieces are spread over them. */
export function parse(text = saved()) {
  const m = (text || '').trim().match(/^(https?:\/\/[^/#?\s]+)[^#]*#key=([\w-]+)(?:&more=([\w.,-]+))?$/);
  if (!m) return null;
  const bases = [m[1], ...(m[3] ? m[3].split(',').filter(Boolean).map(h => 'https://' + h) : [])];
  return { base: m[1], bases, key: m[2] };
}

// opts.via picks the tunnel (any number; it wraps around).
async function call(path, opts = {}, server = parse()) {
  if (!server) throw new Error('코랩 주소가 없어요.');
  const { via = 0, ...rest } = opts;
  const res = await fetch(server.bases[via % server.bases.length] + path, { ...rest, headers: { 'X-Key': server.key, ...(rest.headers || {}) } });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `코랩 서버 오류 (${res.status})`);
  }
  return res;
}

/** The server's status ({ ok, gpu }) or null when it can't be reached. */
export async function status(text) {
  const server = parse(text);
  if (!server) return null;
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 6000);
    const r = await (await call('/status', { signal: ctrl.signal }, server)).json();
    clearTimeout(timer);
    return r.ok ? r : null;
  } catch { return null; }
}

/**
 * Makes a clip on the server. `params`: { name, start, end, fps, maxSide, shape, faceRef, pick }.
 * Resolves to { blob, meta: { layout, width, height, duration, partial } }.
 * `stop.requested` finishes early keeping what's made; `signal` cancels.
 */
export async function extract(file, params, options) {
  try {
    return await run(file, params, options);
  } catch (err) {
    // The server couldn't get it from YouTube (it often blocks Colab): send the file instead.
    if (params.url && /^download:/.test(err.message)) {
      console.warn('Colab could not download it, sending the file:', err.message);
      return run(file, { ...params, url: null }, options);
    }
    throw err;
  }
}

// The whole clip in one request (servers from older notebooks), counting what has arrived.
async function fetchWhole(id, onProgress) {
  const res = await call(`/jobs/${id}/result`);
  const meta = JSON.parse(res.headers.get('X-Clip') || '{}');
  const size = Number(res.headers.get('Content-Length')) || 0, reader = res.body.getReader(), chunks = [];
  let sum = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    sum += value.byteLength;
    onProgress?.({ stage: `완성된 클립 받는 중 ${(sum / 1048576).toFixed(1)}${size ? ` / ${(size / 1048576).toFixed(1)}` : ''}MB` });
  }
  return { blob: new Blob(chunks, { type: 'video/mp4' }), meta };
}

// Runs task(0..count-1) with at most `width` going at once.
async function parallel(count, width, task) {
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(width, count) }, async () => {
    while (next < count) await task(next++);
  }));
}

// A piece that fails over a flaky tunnel is tried again a couple of times.
async function retry(fn, tries = 3) {
  for (let t = 1; ; t++) {
    try { return await fn(t - 1); } catch (err) {
      if (t >= tries || err.name === 'AbortError') throw err;
      await new Promise(r => setTimeout(r, 1000 * t));
    }
  }
}

async function run(file, params, { onProgress, stop, signal }) {
  const id = (await (await call('/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params) })).json()).id;
  const cancel = () => call(`/jobs/${id}`, { method: 'DELETE' }).catch(() => {});
  signal?.addEventListener('abort', cancel, { once: true });
  const aborted = () => { if (signal?.aborted) throw new DOMException('취소됨', 'AbortError'); };
  let stopLive = async () => {};
  try {
    // From YouTube: the server downloads it itself (faster than sending a big original from here).
    // A quick tunnel is slow per connection, so the pieces go several at a time.
    const parts = params.url ? 0 : Math.max(1, Math.ceil(file.size / PART));
    let sent = 0;
    const lanes = parse()?.bases.length || 1;
    await parallel(parts, Math.min(8, 3 * lanes), async i => {
      aborted();
      await retry(t => call(`/jobs/${id}/part?i=${i}`, { method: 'PUT', body: file.slice(i * PART, (i + 1) * PART), via: i + t }));
      onProgress?.({ stage: `코랩으로 영상 보내는 중 ${Math.round(++sent / parts * 100)}%` });
    });
    await call(`/jobs/${id}/start`, { method: 'POST' });
    const t0 = performance.now();
    // Circle clips can be fetched while they're being made (the server's /live: a file that only
    // grows), so little is left to fetch at the end. Whole 4 MB pieces are taken as soon as they
    // are written, a few at once over all the tunnels; whatever is missing is fetched at the end.
    const CH = 4 << 20;
    const have = new Map(); // offset -> Blob of CH bytes
    const live = { on: true, avail: 0, next: 0, ctls: new Set(), workers: [] };
    const received = () => have.size * CH;
    const wait = ms => new Promise(r => setTimeout(r, ms));
    // Bytes [a, b) of the clip, counting them as they arrive.
    const getRange = async (path, via, signal, onBytes, headers) => {
      const res = await call(path, { via, signal, headers });
      const reader = res.body.getReader(), chunks = [];
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        onBytes?.(value.byteLength);
      }
      return { blob: new Blob(chunks), res };
    };
    const liveWorker = async w => {
      while (live.on) {
        if (live.next + CH > live.avail) { // nothing whole to take yet: see how far it's written
          if (w === 0) {
            try {
              const { res } = await getRange(`/jobs/${id}/live?from=0&to=0`, 0);
              live.avail = Number(res.headers.get('X-Size')) || 0;
            } catch {}
          }
          if (live.next + CH > live.avail) await wait(2000);
          continue;
        }
        const a = live.next;
        live.next += CH;
        const ctl = new AbortController();
        live.ctls.add(ctl);
        try {
          const { blob } = await getRange(`/jobs/${id}/live?from=${a}&to=${a + CH}`, w, ctl.signal);
          if (blob.size === CH) have.set(a, blob);
        } catch { /* left as a gap: fetched at the end */ }
        live.ctls.delete(ctl);
      }
    };
    live.workers = Array.from({ length: Math.min(8, 2 * lanes) }, (_, w) => liveWorker(w));
    stopLive = async () => { live.on = false; for (const c of live.ctls) c.abort(); await Promise.all(live.workers); };
    let stopSent = false, s;
    for (;;) {
      aborted();
      await new Promise(r => setTimeout(r, 1500));
      if (stop?.requested && !stopSent) { stopSent = true; call(`/jobs/${id}/stop`, { method: 'POST' }).catch(() => {}); }
      s = await (await call(`/jobs/${id}`)).json();
      if (s.state === 'error') { stopLive(); throw new Error(s.error || '코랩에서 처리하지 못했어요.'); }
      if (s.state === 'cancelled') { stopLive(); throw new DOMException('취소됨', 'AbortError'); }
      if (s.state === 'done') break;
      if (s.state === 'queued') onProgress?.({ stage: '코랩에서 앞 작업이 끝나길 기다리는 중' });
      else if (s.state === 'downloading') onProgress?.({ stage: '코랩이 유튜브에서 영상 받는 중' });
      else if (s.stage || !s.total) onProgress?.({ stage: s.stage || '코랩에서 준비 중' });
      else {
        const elapsed = (performance.now() - t0) / 1000;
        onProgress?.({ done: s.done, total: s.total, elapsed, eta: s.done ? elapsed / s.done * (s.total - s.done) : 0, fixes: s.fixes, received: received() });
      }
    }
    await stopLive();
    onProgress?.({ stage: '완성된 클립 받는 중' });
    // Several ranges at once, for the same reason as the upload.
    const info = await call(`/jobs/${id}/info`).then(r => r.json(), () => null);
    if (!info) { // a notebook from before ranges
      const whole = await fetchWhole(id, onProgress);
      call(`/jobs/${id}`, { method: 'DELETE' }).catch(() => {});
      return whole;
    }
    const { size, meta } = info;
    // What's still missing (all of it for clips that can't be fetched early), in pieces spread
    // over the tunnels, several at once.
    const missing = [];
    for (let a = 0; a < size; a += CH) {
      const b = Math.min(size, a + CH);
      if (!(b - a === CH && have.has(a))) missing.push([a, b]);
    }
    const already = size - missing.reduce((x, [a, b]) => x + b - a, 0);
    const got = new Array(missing.length).fill(0);
    const MB = v => (v / 1048576).toFixed(1);
    const t1 = performance.now();
    const show = () => {
      const now = got.reduce((x, y) => x + y, 0), sum = already + now;
      const secs = (performance.now() - t1) / 1000, speed = secs > 1 ? ` · ${MB(now / secs)}MB/s` : '';
      onProgress?.({ stage: `완성된 클립 받는 중 ${Math.round(sum / size * 100)}% (${MB(sum)} / ${MB(size)}MB${speed})` });
    };
    show();
    await parallel(missing.length, Math.min(16, 4 * lanes), async i => {
      aborted();
      const [a, b] = missing[i];
      const { blob } = await retry(t => {
        got[i] = 0;
        return getRange(`/jobs/${id}/result`, i + t, signal, n => { got[i] += n; show(); }, { Range: `bytes=${a}-${b - 1}` });
      });
      have.set(a, blob);
    });
    const blob = new Blob([...have.keys()].sort((x, y) => x - y).map(k => have.get(k)), { type: 'video/mp4' });
    call(`/jobs/${id}`, { method: 'DELETE' }).catch(() => {}); // free the server's disk
    return { blob, meta };
  } finally {
    stopLive();
    signal?.removeEventListener('abort', cancel);
  }
}
