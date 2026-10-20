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

async function run(file, params, { onProgress, stop, signal }) {
  const id = (await (await call('/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params) })).json()).id;
  const cancel = () => call(`/jobs/${id}`, { method: 'DELETE' }).catch(() => {});
  signal?.addEventListener('abort', cancel, { once: true });
  const aborted = () => { if (signal?.aborted) throw new DOMException('취소됨', 'AbortError'); };
  try {
    // From YouTube: the server downloads it itself (faster than sending a big original from here).
    const parts = params.url ? 0 : Math.max(1, Math.ceil(file.size / PART));
    for (let i = 0; i < parts; i++) {
      aborted();
      onProgress?.({ stage: `코랩으로 영상 보내는 중 ${Math.round(i / parts * 100)}%` });
      await call(`/jobs/${id}/part?i=${i}`, { method: 'PUT', body: file.slice(i * PART, (i + 1) * PART) });
    }
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
    const res = await call(`/jobs/${id}/result`);
    const meta = JSON.parse(res.headers.get('X-Clip') || '{}');
    const blob = new Blob([await res.arrayBuffer()], { type: 'video/mp4' });
    call(`/jobs/${id}`, { method: 'DELETE' }).catch(() => {}); // free the server's disk
    return { blob, meta };
  } finally {
    signal?.removeEventListener('abort', cancel);
  }
}
