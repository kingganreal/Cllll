// The GPU models (SAM 2, matting, face recognition) run in a worker (js/gpu-worker.js).
// onnxruntime's WebGPU backend can hang for good: on an RTX 3050 laptop we saw GPU faults and
// runs that never returned while the page spun at 100% CPU, freezing it mid-clip. In a worker that
// can't freeze the page: every call has a deadline, and a worker that misses one is shut down, so
// the clip carries on with the CPU tools (and the next clip starts a fresh worker).

import { gpuAdapter } from './ort.js';

const TRUST_KEY = 'fancam-sam2-gpu-trust';

class GpuWorker {
  constructor() {
    this.worker = new Worker(new URL('./gpu-worker.js', import.meta.url), { type: 'module' });
    this.calls = new Map();
    this.nextId = 1;
    this.dead = null;
    this.onProgress = null;
    this.worker.onmessage = ({ data }) => {
      if (data.type === 'trust') { try { localStorage.setItem(TRUST_KEY, JSON.stringify(data.value)); } catch {} return; }
      if (data.type === 'progress') {
        // Still downloading: that's not hanging, so the deadlines start over.
        for (const call of this.calls.values()) if (call.timeout) { clearTimeout(call.timer); call.timer = this.deadline(call); }
        this.onProgress?.(data.value);
        return;
      }
      const call = this.calls.get(data.id);
      if (!call) return;
      this.calls.delete(data.id);
      clearTimeout(call.timer);
      if (data.ok) call.resolve(data.value);
      else call.reject(new Error(data.error));
    };
    this.worker.onerror = e => this.kill('GPU 작업이 멈췄어요: ' + (e.message || 'worker error'));
    let trust = {};
    try { trust = JSON.parse(localStorage.getItem(TRUST_KEY)) || {}; } catch {}
    this.call('config', [{ trust }]).catch(() => {});
  }

  /**
   * Calls `method` in the worker. With a `timeout` (ms), rejects and shuts the worker down if the
   * answer takes longer. Loading has none: creating a session can keep the worker busy for a while.
   */
  call(method, args = [], { timeout = 0, transfer = [] } = {}) {
    if (this.dead) return Promise.reject(new Error(this.dead));
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      const call = { resolve, reject, method, timeout };
      call.timer = timeout ? this.deadline(call) : 0;
      this.calls.set(id, call);
      this.worker.postMessage({ id, method, args }, transfer);
    });
  }

  deadline(call) {
    return setTimeout(() => this.kill(`GPU가 응답하지 않아요 (${call.method})`), call.timeout);
  }

  kill(reason) {
    if (this.dead) return;
    this.dead = reason;
    console.warn('GPU worker stopped:', reason);
    this.worker.terminate();
    for (const call of this.calls.values()) { clearTimeout(call.timer); call.reject(new Error(reason)); }
    this.calls.clear();
  }
}

let current = null;

/** The GPU worker, started if needed (or again, after one was shut down). */
export function gpuWorker() {
  if (!current || current.dead) current = new GpuWorker();
  return current;
}

export async function webgpuAvailable() { return !!(await gpuAdapter()); }
