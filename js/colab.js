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

/** { base, key } from the address the notebook printed, or null. */
export function parse(text = saved()) {
  const m = (text || '').trim().match(/^(https?:\/\/[^/#?\s]+)[^#]*#key=([\w-]+)$/);
  return m ? { base: m[1], key: m[2] } : null;
}

async function call(path, opts = {}, server = parse()) {
  if (!server) throw new Error('코랩 주소가 없어요.');
  const res = await fetch(server.base + path, { ...opts, headers: { 'X-Key': server.key, ...(opts.headers || {}) } });
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
    try { return await fn(); } catch (err) {
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
  try {
    // From YouTube: the server downloads it itself (faster than sending a big original from here).
    // A quick tunnel is slow per connection, so the pieces go several at a time.
    const parts = params.url ? 0 : Math.max(1, Math.ceil(file.size / PART));
    let sent = 0;
    await parallel(parts, 3, async i => {
      aborted();
      await retry(() => call(`/jobs/${id}/part?i=${i}`, { method: 'PUT', body: file.slice(i * PART, (i + 1) * PART) }));
      onProgress?.({ stage: `코랩으로 영상 보내는 중 ${Math.round(++sent / parts * 100)}%` });
    });
    await call(`/jobs/${id}/start`, { method: 'POST' });
    const t0 = performance.now();
    let stopSent = false, s;
    for (;;) {
      aborted();
      await new Promise(r => setTimeout(r, 1500));
      if (stop?.requested && !stopSent) { stopSent = true; call(`/jobs/${id}/stop`, { method: 'POST' }).catch(() => {}); }
      s = await (await call(`/jobs/${id}`)).json();
      if (s.state === 'error') throw new Error(s.error || '코랩에서 처리하지 못했어요.');
      if (s.state === 'cancelled') throw new DOMException('취소됨', 'AbortError');
      if (s.state === 'done') break;
      if (s.state === 'queued') onProgress?.({ stage: '코랩에서 앞 작업이 끝나길 기다리는 중' });
      else if (s.state === 'downloading') onProgress?.({ stage: '코랩이 유튜브에서 영상 받는 중' });
      else if (s.stage || !s.total) onProgress?.({ stage: s.stage || '코랩에서 준비 중' });
      else {
        const elapsed = (performance.now() - t0) / 1000;
        onProgress?.({ done: s.done, total: s.total, elapsed, eta: s.done ? elapsed / s.done * (s.total - s.done) : 0, fixes: s.fixes });
      }
    }
    onProgress?.({ stage: '완성된 클립 받는 중' });
    // Several ranges at once, for the same reason as the upload.
    const info = await call(`/jobs/${id}/info`).then(r => r.json(), () => null);
    if (!info) { // a notebook from before ranges
      const whole = await fetchWhole(id, onProgress);
      call(`/jobs/${id}`, { method: 'DELETE' }).catch(() => {});
      return whole;
    }
    const { size, meta } = info;
    const RANGE = 8 * 1024 * 1024, n = Math.max(1, Math.ceil(size / RANGE)), pieces = new Array(n);
    const MB = v => (v / 1048576).toFixed(1);
    const got = new Array(n).fill(0); // bytes so far of each piece (counted as they arrive)
    const show = () => {
      const sum = got.reduce((x, y) => x + y, 0);
      onProgress?.({ stage: `완성된 클립 받는 중 ${Math.round(sum / size * 100)}% (${MB(sum)} / ${MB(size)}MB)` });
    };
    show();
    await parallel(n, 6, async i => {
      aborted();
      const a = i * RANGE, b = Math.min(size, a + RANGE) - 1;
      pieces[i] = await retry(async () => {
        got[i] = 0;
        const res = await call(`/jobs/${id}/result`, { headers: { Range: `bytes=${a}-${b}` } });
        const reader = res.body.getReader(), chunks = [];
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          got[i] += value.byteLength;
          show();
        }
        return new Blob(chunks);
      });
    });
    const blob = new Blob(pieces, { type: 'video/mp4' });
    call(`/jobs/${id}`, { method: 'DELETE' }).catch(() => {}); // free the server's disk
    return { blob, meta };
  } finally {
    signal?.removeEventListener('abort', cancel);
  }
}
