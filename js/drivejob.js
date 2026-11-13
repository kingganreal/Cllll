// Colab through Google Drive instead of a tunnel (the Windows app's /drive/… routes, see
// desktop/drivejobs.js, and colab/server.py watch_drive): the job and the video go into the
// Google Drive folder on this PC, Colab takes them from Drive and puts the clip back there, and
// Google Drive brings it down at full speed. Same results as js/colab.js extract().
const BASE = 'http://localhost:8790';
const LS = 'fancam-colab-drive';

export function enabled() { try { return localStorage.getItem(LS) === '1'; } catch { return false; } }
export function setEnabled(on) { try { on ? localStorage.setItem(LS, '1') : localStorage.removeItem(LS); } catch {} }

async function call(path, opts = {}) {
  const res = await fetch(BASE + path, { cache: 'no-store', ...opts });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw Object.assign(new Error(body.error || `윈도우 앱 오류 (${res.status})`), { status: res.status });
  }
  return res;
}

/** { ok, why, gpu }: whether a job can go through Drive now, and if not, what's missing. */
export async function ready() {
  let s;
  try {
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 3000);
    s = await (await call('/drive/status', { signal: ctrl.signal })).json();
  } catch {
    return { ok: false, why: '직캠 스테이지 윈도우 앱이 켜져 있지 않아요 (0.1.4 이상).' };
  }
  if (!s.folder) return { ok: false, why: 'PC에서 구글 드라이브 폴더를 찾지 못했어요. 구글 드라이브 앱을 켜고 로그인해 주세요.' };
  if (!s.colab?.alive) {
    return { ok: false, why: s.colab ? `코랩이 ${Math.round(s.colab.age / 60)}분째 소식이 없어요. 코랩의 5-C 칸이 실행 중인지 확인해 주세요.`
      : '코랩의 5-C 칸(드라이브로 주고받기)을 실행해 주세요.' };
  }
  return { ok: true, gpu: s.colab.gpu };
}

const wait = ms => new Promise(r => setTimeout(r, ms));

export async function extract(file, params, { onProgress, stop, signal }) {
  const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  const aborted = () => { if (signal?.aborted) throw new DOMException('취소됨', 'AbortError'); };
  const cancel = () => call(`/drive/jobs/${id}`, { method: 'DELETE' }).catch(() => {});
  signal?.addEventListener('abort', cancel, { once: true });
  try {
    if (!params.url) {
      onProgress?.({ stage: `구글 드라이브에 영상 넣는 중 (${(file.size / 1048576).toFixed(0)}MB)` });
      await call(`/drive/jobs/${id}/source`, { method: 'PUT', body: file });
    }
    aborted();
    await call(`/drive/jobs/${id}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params) });
    const t0 = performance.now();
    let stopSent = false, s, startedAt = null;
    for (;;) {
      aborted();
      await wait(3000);
      if (stop?.requested && !stopSent) { stopSent = true; call(`/drive/jobs/${id}/stop`, { method: 'POST' }).catch(() => {}); }
      s = await (await call(`/drive/jobs/${id}`)).json();
      if (s.state === 'error') {
        // YouTube refused Colab: send the video itself instead
        if (params.url && /^download:/.test(s.error || '')) {
          signal?.removeEventListener('abort', cancel);
          cancel();
          return extract(file, { ...params, url: null }, { onProgress, stop, signal });
        }
        throw new Error(s.error || '코랩에서 처리하지 못했어요.');
      }
      if (s.state === 'cancelled') throw new DOMException('취소됨', 'AbortError');
      if (s.state === 'done' && s.file) break;
      if (s.state === 'waiting') {
        const secs = (performance.now() - t0) / 1000;
        onProgress?.({ stage: secs < 20 ? '구글 드라이브로 코랩에 보내는 중' : `코랩이 드라이브에서 가져가길 기다리는 중 (${Math.round(secs)}초)` });
      } else if (s.state === 'queued') onProgress?.({ stage: '코랩에서 앞 작업이 끝나길 기다리는 중' });
      else if (s.state === 'downloading') onProgress?.({ stage: '코랩이 유튜브에서 영상 받는 중' });
      else if (s.stage || !s.total) onProgress?.({ stage: s.stage || '코랩에서 준비 중' });
      else {
        startedAt ??= performance.now();
        const elapsed = (performance.now() - startedAt) / 1000;
        onProgress?.({ done: s.done, total: s.total, elapsed, eta: s.done ? elapsed / s.done * (s.total - s.done) : 0, fixes: s.fixes });
      }
    }
    // The clip comes down through Google Drive; until it's on this PC the app answers 404.
    const MB = v => (v / 1048576).toFixed(1);
    const t1 = performance.now();
    for (;;) {
      aborted();
      try {
        const res = await call(`/drive/jobs/${id}/result`);
        const size = Number(res.headers.get('Content-Length')) || s.size || 0;
        const reader = res.body.getReader(), chunks = [];
        let got = 0;
        const t2 = performance.now();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          got += value.byteLength;
          const secs = (performance.now() - t2) / 1000;
          onProgress?.({ stage: `구글 드라이브에서 클립 받는 중 ${size ? Math.round(got / size * 100) : 0}% (${MB(got)} / ${MB(size)}MB${secs > 1 ? ` · ${MB(got / secs)}MB/s` : ''})` });
        }
        if (s.size && got !== s.size) throw new Error('클립을 끝까지 받지 못했어요.');
        call(`/drive/jobs/${id}`, { method: 'DELETE' }).catch(() => {}); // clean up the Drive folder
        return { blob: new Blob(chunks, { type: s.meta?.kind === '3d' ? 'application/octet-stream' : 'video/mp4' }), meta: s.meta || {} };
      } catch (err) {
        if (err.name === 'AbortError' || err.status !== 404) throw err;
        const secs = Math.round((performance.now() - t1) / 1000);
        if (secs > 600) throw new Error('10분이 지나도 클립이 PC의 구글 드라이브로 안 내려왔어요. 구글 드라이브 앱이 동기화 중인지 확인해 주세요.');
        onProgress?.({ stage: `완성! 구글 드라이브가 PC로 내려받길 기다리는 중 (${secs}초)` });
        await wait(3000);
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') cancel();
    throw err;
  } finally {
    signal?.removeEventListener('abort', cancel);
  }
}
