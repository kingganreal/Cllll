// Finds people for the circle's fast tracking: YOLO11n (Ultralytics, AGPL-3.0, see
// models/YOLO-NOTICE.txt) on the GPU in the worker. Much faster than SAM 2, but it only finds
// boxes each frame; which box is her is decided in processor.js (motion + her face).

import { loadOrt, createSession, gpuAdapter } from './ort.js';

const MODEL = new URL('../models/yolo11n.onnx', import.meta.url).href;
const SIZE = 640, N = 8400; // input side, candidate boxes
const MIN_SCORE = 0.3;

let loadPromise = null;

export function load() {
  loadPromise ??= (async () => {
    const ort = await loadOrt();
    const gpu = typeof document === 'undefined' && await gpuAdapter();
    const s = await createSession(MODEL, { executionProviders: [gpu ? 'webgpu' : 'wasm'], graphOptimizationLevel: 'all' });
    return { ort, s };
  })().catch(err => { loadPromise = null; throw err; });
  return loadPromise;
}

const input = new Float32Array(3 * SIZE * SIZE);

/** Everyone in an RGBA picture (w x h): [{ box: [x0, y0, x1, y1] (0..1), score }]. */
export async function people({ ort, s }, rgba, w, h) {
  // Letterboxed into the top-left of 640x640, grey elsewhere (as it was trained).
  const k = SIZE / Math.max(w, h), nw = Math.round(w * k), nh = Math.round(h * k), m = SIZE * SIZE;
  input.fill(114 / 255);
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(h - 1, Math.floor(y / k)) * w;
    for (let x = 0; x < nw; x++) {
      const j = (sy + Math.min(w - 1, Math.floor(x / k))) * 4, o = y * SIZE + x;
      input[o] = rgba[j] / 255; input[m + o] = rgba[j + 1] / 255; input[2 * m + o] = rgba[j + 2] / 255;
    }
  }
  const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) });
  const d = out[s.outputNames[0]].data; // (84, 8400): cx, cy, w, h, then 80 class scores (0 = person)
  const found = [];
  for (let a = 0; a < N; a++) {
    const score = d[4 * N + a];
    if (score < MIN_SCORE) continue;
    const cx = d[a], cy = d[N + a], bw = d[2 * N + a], bh = d[3 * N + a];
    found.push({ score, box: [(cx - bw / 2) / k / w, (cy - bh / 2) / k / h, (cx + bw / 2) / k / w, (cy + bh / 2) / k / h].map(v => Math.min(1, Math.max(0, v))) });
  }
  found.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const f of found) if (kept.every(q => iou(q.box, f.box) < 0.5)) kept.push(f);
  return kept;
}

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = ix * iy;
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i || 1);
}
