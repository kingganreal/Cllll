// Face recognition: tells which person in a frame is her, from a few photos registered once.
// SCRFD-500M finds faces (with eyes, nose and mouth corners); each face is straightened to the
// standard 112x112 crop and MobileFaceNet (ArcFace, WebFace600K) turns it into 512 numbers.
// Two crops of the same person point the same way (cosine similarity around 0.4 and up);
// different people come out near 0. Models: InsightFace "buffalo_s" (see models/face/NOTICE.txt).
// While making a clip it runs in the GPU worker (js/gpu-worker.js) next to SAM 2 and the matting
// model, on the GPU. On the page (registering photos, picking) it runs on the CPU: the models are
// small, and keeping the page's onnxruntime CPU-only avoids mixing CPU and GPU sessions in one
// runtime, which sometimes hung.

import { loadOrt, createSession, gpuAdapter } from './ort.js';

const DIR = new URL('../models/face/', import.meta.url).href;
const DET = 640;
// Where the eyes, nose and mouth corners sit in the 112x112 crop the recogniser was trained on.
const TEMPLATE = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];

// Measured on a group fancam where every member wears the same white outfit: her faces scored
// 0.37-0.75 against her photos, the other members mostly 0.0-0.42 (up to 0.51). So "her" needs a
// clear score and a lead over the next face, and "someone else" only a low one.
export const SAME = 0.42;  // at least this similar, and MARGIN ahead of the next face: her
export const MARGIN = 0.08;
export const OTHER = 0.25; // at most this similar (on a clear face): someone else
// Faces this sure (against the photos alone) also teach the matcher how she looks in this video
// (stage lights, make-up, camera): members then separate better than with the photos alone.
const LEARN = 0.4, LEARN_MARGIN = 0.12;
// Once it knows her look in this video, her faces score 0.43 and up (1st percentile) against it,
// so a clear face under this is someone else.
const OTHER_LEARNED = 0.35;

let loadPromise = null;

export function load() {
  loadPromise ??= (async () => {
    const ort = await loadOrt();
    const inWorker = typeof document === 'undefined';
    const gpu = inWorker && await gpuAdapter();
    const ep = globalThis.__FANCAM_FACE_EP__ || (gpu ? 'webgpu' : 'wasm'); // tests can force one
    const opts = { executionProviders: [ep], graphOptimizationLevel: 'all' };
    const det = await createSession(DIR + 'det_500m.onnx', opts);
    const rec = await createSession(DIR + 'w600k_mbf.onnx', opts);
    return { ort, det, rec };
  })().catch(err => { loadPromise = null; throw err; });
  return loadPromise;
}

const canvas = (w, h) => {
  const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
  return [c, c.getContext('2d', { willReadFrequently: true })];
};
let detCanvas = null, cropCanvas = null;

/**
 * Every face in a picture.
 * @param {(ctx: CanvasRenderingContext2D) => void} draw draws the picture at its own size (w x h)
 *   with the context's current transform, e.g. ctx => ctx.drawImage(img, 0, 0)
 * @returns {Promise<{box:number[], kps:number[][], score:number, size:number, emb:Float32Array}[]>}
 *   box and kps in the picture's pixels; size = eye-to-mouth height in pixels
 */
export async function faces(models, draw, w, h, { minSize = 14 } = {}) {
  const { ort, det, rec } = models;
  detCanvas ??= canvas(DET, DET);
  const [, dctx] = detCanvas;
  // Fit into the top-left of a 640x640 square, as the detector was trained.
  const s = DET / Math.max(w, h);
  dctx.setTransform(1, 0, 0, 1, 0, 0);
  dctx.fillStyle = '#000';
  dctx.fillRect(0, 0, DET, DET);
  dctx.imageSmoothingQuality = 'high';
  dctx.setTransform(s, 0, 0, s, 0, 0);
  draw(dctx);
  const px = dctx.getImageData(0, 0, DET, DET).data;
  const n = DET * DET, input = new Float32Array(3 * n);
  for (let k = 0, j = 0; k < n; k++, j += 4) {
    input[k] = (px[j] - 127.5) / 128;
    input[n + k] = (px[j + 1] - 127.5) / 128;
    input[2 * n + k] = (px[j + 2] - 127.5) / 128;
  }
  const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, DET, DET]) });
  const o = det.outputNames.map(name => out[name].data);

  let found = [];
  [8, 16, 32].forEach((stride, si) => {
    const scores = o[si], boxes = o[si + 3], kps = o[si + 6];
    const gw = DET / stride;
    for (let a = 0; a < scores.length; a++) {
      if (scores[a] < 0.5) continue;
      const cell = a >> 1, cx = (cell % gw) * stride, cy = Math.floor(cell / gw) * stride;
      const b = [cx - boxes[a * 4] * stride, cy - boxes[a * 4 + 1] * stride, cx + boxes[a * 4 + 2] * stride, cy + boxes[a * 4 + 3] * stride].map(v => v / s);
      const k5 = [];
      for (let p = 0; p < 5; p++) k5.push([(cx + kps[a * 10 + p * 2] * stride) / s, (cy + kps[a * 10 + p * 2 + 1] * stride) / s]);
      found.push({ box: b, kps: k5, score: scores[a] });
    }
  });
  found.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const f of found) if (kept.every(k => iou(k.box, f.box) < 0.4)) kept.push(f);
  found = kept.map(f => ({ ...f, size: (f.kps[3][1] + f.kps[4][1]) / 2 - (f.kps[0][1] + f.kps[1][1]) / 2 }))
    .filter(f => Math.min(f.box[2] - f.box[0], f.box[3] - f.box[1]) >= minSize);
  if (!found.length) return found;

  // Straighten each face into the 112x112 crop, drawn from the full-size picture.
  cropCanvas ??= canvas(112, 112);
  const [, cctx] = cropCanvas;
  cctx.imageSmoothingQuality = 'high';
  const m = 112 * 112, batch = new Float32Array(found.length * 3 * m);
  found.forEach((f, i) => {
    const [ar, ai, bx, by] = similarity(f.kps, TEMPLATE);
    cctx.setTransform(1, 0, 0, 1, 0, 0);
    cctx.fillStyle = '#000';
    cctx.fillRect(0, 0, 112, 112);
    cctx.setTransform(ar, ai, -ai, ar, bx, by);
    draw(cctx);
    const c = cctx.getImageData(0, 0, 112, 112).data, off = i * 3 * m;
    for (let k = 0, j = 0; k < m; k++, j += 4) {
      batch[off + k] = (c[j] - 127.5) / 127.5;
      batch[off + m + k] = (c[j + 1] - 127.5) / 127.5;
      batch[off + 2 * m + k] = (c[j + 2] - 127.5) / 127.5;
    }
  });
  const r = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', batch, [found.length, 3, 112, 112]) });
  const e = r[rec.outputNames[0]].data;
  found.forEach((f, i) => { f.emb = normalise(e.slice(i * 512, (i + 1) * 512)); });
  return found;
}

/** Her reference: the average direction of the registered faces. */
export function reference(embeddings) {
  if (!embeddings?.length) return null;
  const avg = new Float32Array(512);
  for (const e of embeddings) for (let k = 0; k < 512; k++) avg[k] += e[k];
  return normalise(avg);
}

export function similarity2(a, b) {
  let d = 0;
  for (let k = 0; k < 512; k++) d += a[k] * b[k];
  return d;
}

/** Which face (if any) is her: the best match, clearly ahead of the next one. */
export function findHer(found, ref) {
  return matcher(ref).findHer(found);
}

/**
 * Recognises her through one video. Scores each face against her photos and, once it has seen
 * her for sure a couple of times, against how she looks in this video too (whichever is higher).
 * @param {Float32Array} ref  her photos (reference())
 * @param {Float32Array} [seed]  a face known to be hers in this video (e.g. the one picked)
 */
export function matcher(ref, seed) {
  const sum = new Float32Array(512);
  let seen = 0, video = null;
  const learn = emb => {
    for (let k = 0; k < 512; k++) sum[k] += emb[k];
    seen++;
    video = normalise(sum);
  };
  if (seed) learn(seed);
  return {
    get learned() { return seen; },
    learn,
    /** A face (already scored by findHer) clearly not hers. */
    other(f) { return f.sim < (seen >= 5 ? OTHER_LEARNED : OTHER); },
    /**
     * Sets f.sim on every face; returns hers or null.
     * @param {(f) => boolean} [plausible] a last check on the best match (e.g. where it is)
     */
    findHer(found, plausible = () => true) {
      let best = null, second = -1;
      for (const f of found) {
        f.photoSim = similarity2(f.emb, ref);
        f.sim = seen >= 2 ? Math.max(f.photoSim, similarity2(f.emb, video)) : f.photoSim;
        if (!best || f.sim > best.sim) { if (best) second = best.sim; best = f; }
        else if (f.sim > second) second = f.sim;
      }
      let her = best && best.sim >= SAME && best.sim - second >= MARGIN ? best : null;
      if (her && !plausible(her)) her = null;
      if (her && her.photoSim >= LEARN && her.sim - second >= LEARN_MARGIN) learn(her.emb);
      return her;
    },
  };
}

function normalise(v) {
  let s = 0;
  for (let k = 0; k < v.length; k++) s += v[k] * v[k];
  s = 1 / Math.sqrt(s || 1);
  const out = new Float32Array(v.length);
  for (let k = 0; k < v.length; k++) out[k] = v[k] * s;
  return out;
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = ix * iy;
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i);
}

// Least-squares rotation + scale + shift taking points `src` onto `dst` (as complex numbers:
// dst = a * src + b). Returns [Re a, Im a, Re b, Im b].
function similarity(src, dst) {
  const N = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < N; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
  sx /= N; sy /= N; dx /= N; dy /= N;
  let nr = 0, ni = 0, den = 0;
  for (let i = 0; i < N; i++) {
    const zx = src[i][0] - sx, zy = src[i][1] - sy, wx = dst[i][0] - dx, wy = dst[i][1] - dy;
    nr += wx * zx + wy * zy;   // Re(w * conj z)
    ni += wy * zx - wx * zy;   // Im(w * conj z)
    den += zx * zx + zy * zy;
  }
  const ar = nr / den, ai = ni / den;
  return [ar, ai, dx - (ar * sx - ai * sy), dy - (ai * sx + ar * sy)];
}
