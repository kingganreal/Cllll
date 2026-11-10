// Extracts one person from a video and encodes the result as an "alpha-packed" MP4:
// colour on one half of each frame, the alpha matte (greyscale) on the other half.
// Packed MP4 plays on every phone/tablet/desktop browser; js/player.js recombines
// the halves into a transparent image with a tiny WebGL shader.
import { PoseLandmarker } from '../vendor/mediapipe/vision_bundle.mjs';
import { Muxer, StreamTarget } from '../vendor/mp4-muxer/mp4-muxer.mjs';
import { openSource, seekVideo } from './source.js';
import { gpuWorker, webgpuAvailable } from './gpu.js';
import * as sam2 from './sam2.js';
import * as face from './face.js';
import { iou as boxIou } from './yolo.js';

export { seekVideo as seek };

// Longest side of the frame fed to the segmentation model. Full-HD output gets a finer
// matte so edges stay crisp after upscaling.
const analysisSide = maxSide => (maxSide >= 1920 ? 960 : 640);
const GRID = 4;            // stride for the coarse masks used to track a person

// ---------------------------------------------------------------------------
// Model

let landmarkerPromise = null;
let delegate = 'GPU';

export function getLandmarker() {
  landmarkerPromise ??= (async () => {
    const base = new URL('../vendor/mediapipe/wasm/', import.meta.url).href;
    const fileset = {
      wasmLoaderPath: base + 'vision_wasm_internal.js',
      wasmBinaryPath: base + 'vision_wasm_internal.wasm',
    };
    const options = {
      baseOptions: {
        modelAssetPath: new URL('../models/pose_landmarker_full.task', import.meta.url).href,
        delegate,
      },
      // IMAGE mode looks for everyone in every frame. VIDEO mode only keeps following the
      // person it already has and never notices a second one, so it can slide onto another
      // member when two cross. Tracker below does the following instead.
      runningMode: 'IMAGE',
      numPoses: 8, // a whole group (NMIXX is six) plus a dancer or two
      outputSegmentationMasks: true,
      minPoseDetectionConfidence: 0.35,
      minPosePresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
    };
    try {
      return await PoseLandmarker.createFromOptions(fileset, options);
    } catch (err) {
      if (delegate === 'CPU') throw err;
      console.warn('GPU delegate failed, falling back to CPU', err);
      delegate = options.baseOptions.delegate = 'CPU';
      return await PoseLandmarker.createFromOptions(fileset, options);
    }
  })().catch(err => { landmarkerPromise = null; throw err; });
  return landmarkerPromise;
}

// Some GPUs find people but hand back blank masks. Detect that and switch to the CPU.
export async function detect(source) {
  let persons = detectPersons(await getLandmarker(), source);
  if (delegate === 'GPU' && persons.length && persons.every(p => p.area === 0)) {
    console.warn('GPU returned empty masks, switching to CPU');
    const old = landmarkerPromise;
    delegate = 'CPU';
    landmarkerPromise = null;
    (await old).close();
    persons = detectPersons(await getLandmarker(), source);
  }
  return persons;
}

// Runs the model on `source` and returns every person found, each with its own mask.
function detectPersons(landmarker, source) {
  const persons = [];
  landmarker.detect(source, result => {
    const masks = result.segmentationMasks || [];
    result.landmarks.forEach((landmarks, i) => {
      const m = masks[i];
      if (!m) return;
      // Mask memory is only valid inside this callback, so copy it out.
      persons.push(describe(m.getAsFloat32Array().slice(), m.width, m.height, landmarks));
    });
  });
  return persons;
}

function describe(mask, w, h, landmarks) {
  const gw = Math.ceil(w / GRID), gh = Math.ceil(h / GRID);
  const grid = new Uint8Array(gw * gh);
  let area = 0, sx = 0, sy = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let gy = 0; gy < gh; gy++) {
    const y = gy * GRID;
    for (let gx = 0; gx < gw; gx++) {
      const x = gx * GRID;
      if (mask[y * w + x] > 0.5) {
        grid[gy * gw + gx] = 1;
        area++; sx += x; sy += y;
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
  }
  return {
    mask, w, h, landmarks, grid, area: area / (gw * gh),
    cx: area ? sx / area / w : 0.5,
    cy: area ? sy / area / h : 0.5,
    bbox: area ? [x0 / w, y0 / h, Math.min(1, (x1 + GRID) / w), Math.min(1, (y1 + GRID) / h)] : null,
  };
}

function iou(a, b) {
  if (!a || !b || a.grid.length !== b.grid.length) return 0;
  let inter = 0, union = 0;
  const ga = a.grid, gb = b.grid;
  for (let i = 0; i < ga.length; i++) { inter += ga[i] & gb[i]; union += ga[i] | gb[i]; }
  return union ? inter / union : 0;
}

// The person under a tapped point (normalised coords), or the closest one.
export function pickAt(persons, nx, ny) {
  let best = null, bestValue = 0.3;
  for (const p of persons) {
    const x = Math.min(p.w - 1, Math.round(nx * p.w)), y = Math.min(p.h - 1, Math.round(ny * p.h));
    const v = p.mask[y * p.w + x];
    if (v > bestValue) { bestValue = v; best = p; }
  }
  if (best) return best;
  let bestDist = Infinity;
  for (const p of persons) {
    const d = Math.hypot(p.cx - nx, p.cy - ny);
    if (d < bestDist) { bestDist = d; best = p; }
  }
  return best;
}

// Fancams keep their member big and centred, so that's the default guess.
export function pickDefault(persons) {
  let best = null, bestScore = -Infinity;
  for (const p of persons) {
    const score = p.area * (1 - 1.5 * Math.hypot(p.cx - 0.5, p.cy - 0.5));
    if (score > bestScore) { bestScore = score; best = p; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Following her through the video.
// Each candidate is scored on: overlap with where she was, her clothing colours, where she should
// be given her recent motion, closeness to the centre (fancams keep her centred), and size.

const HIST_BINS = 12 * 3 + 4;

// Colour histogram of a person's pixels (hue x saturation for colours, 4 grey levels otherwise).
function colourHist(p, rgba) {
  const h = new Float32Array(HIST_BINS);
  const gw = Math.ceil(p.w / GRID);
  let n = 0;
  for (let i = 0; i < p.grid.length; i++) {
    if (!p.grid[i]) continue;
    const x = (i % gw) * GRID, y = Math.floor(i / gw) * GRID;
    const j = (y * p.w + x) * 4;
    const r = rgba[j] / 255, g = rgba[j + 1] / 255, b = rgba[j + 2] / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
    const sat = max ? d / max : 0;
    if (sat > 0.22 && max > 0.18) {
      let hue = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
      hue = (hue + 6) % 6 / 6;
      h[Math.min(11, Math.floor(hue * 12)) * 3 + Math.min(2, Math.floor((sat - 0.22) / 0.26))]++;
    } else {
      h[36 + Math.min(3, Math.floor(max * 4))]++;
    }
    n++;
  }
  if (n) for (let k = 0; k < HIST_BINS; k++) h[k] /= n;
  return n > 20 ? h : null;
}

function histSimilarity(a, b) {
  if (!a || !b) return 0.5;
  let s = 0;
  for (let k = 0; k < HIST_BINS; k++) s += Math.sqrt(a[k] * b[k]);
  return s; // Bhattacharyya coefficient, 1 = identical
}

class Tracker {
  constructor(target, fps) {
    this.last = target;      // last confident detection
    this.ref = null;         // her colour histogram, learnt from the first frames
    this.vx = 0; this.vy = 0;
    this.lost = 0;
    this.fps = fps;
  }

  update(persons, rgba) {
    const last = this.last;
    const lostLong = this.lost > this.fps; // over a second: look again everywhere
    const px = last.cx + this.vx * Math.min(this.lost + 1, 10), py = last.cy + this.vy * Math.min(this.lost + 1, 10);
    let best = null, bestScore = -Infinity, second = -Infinity, bestParts = null;
    for (const p of persons) {
      if (!p.area) continue;
      const hist = colourHist(p, rgba);
      p.hist = hist;
      const overlap = lostLong ? 0 : iou(p, last);
      const looks = histSimilarity(hist, this.ref);
      const motion = Math.max(0, 1 - Math.hypot(p.cx - px, p.cy - py) / 0.25);
      const centre = Math.max(0, 1 - Math.hypot(p.cx - 0.5, p.cy - 0.5) / 0.5);
      const size = Math.max(0, 1 - Math.abs(Math.log(p.area / Math.max(last.area, 1e-4))) / Math.log(3));
      const score = lostLong
        ? 1.6 * looks + 1.0 * centre + 0.5 * size
        : 1.8 * overlap + 1.2 * looks + 0.8 * motion + 0.5 * centre + 0.4 * size;
      if (score > bestScore) { second = bestScore; bestScore = score; best = p; bestParts = { overlap, looks, motion }; }
      else if (score > second) second = score;
    }
    const ok = best && (lostLong
      ? bestParts.looks > 0.62 && bestScore > 1.6
      : (bestParts.overlap > 0.2 || (bestParts.looks > 0.6 && bestParts.motion > 0.3)) && bestScore > 1.2);
    if (!ok) { this.lost++; return null; }

    // Learn how she moves and what she wears (slowly, and only when sure it's her).
    const dx = best.cx - last.cx, dy = best.cy - last.cy, steps = this.lost + 1;
    this.vx = this.vx * 0.6 + (dx / steps) * 0.4;
    this.vy = this.vy * 0.6 + (dy / steps) * 0.4;
    if (best.hist && (bestParts.overlap > 0.4 || !this.ref)) {
      if (!this.ref) this.ref = best.hist.slice();
      else for (let k = 0; k < HIST_BINS; k++) this.ref[k] = this.ref[k] * 0.97 + best.hist[k] * 0.03;
    }
    this.last = best;
    this.lost = 0;
    return best;
  }
}

// ---------------------------------------------------------------------------
// Her region. When two people overlap the model often returns one merged outline for both,
// but its skeleton (joint positions) still belongs to one person. So her region is the
// outline limited to a zone around her skeleton, minus anyone else, softened a little.

const LIMBS = [[11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
  [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32], [15, 19], [16, 20], [27, 29], [28, 30]];

function makeRegionBuilder(w, h) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const n = w * h;
  const skel = new Float32Array(n), zone = new Float32Array(n), tmp = new Float32Array(n), out = new Float32Array(n);

  return (target, others) => {
    const lm = target.landmarks || [];
    const ok = i => lm[i] && (lm[i].visibility ?? 1) > 0.2;
    const pt = i => [lm[i].x * w, lm[i].y * h];
    const vis = lm.filter((l, i) => ok(i));
    if (vis.length < 6) return target.mask.slice(); // too little skeleton to go on
    const ys = vis.map(l => l.y * h);
    const bodyH = Math.max(h * 0.15, Math.max(...ys) - Math.min(...ys));

    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = ctx.fillStyle = '#fff';
    ctx.lineCap = ctx.lineJoin = 'round';
    ctx.lineWidth = Math.max(4, bodyH * 0.11);
    ctx.beginPath();
    for (const [a, b] of LIMBS) {
      if (!ok(a) || !ok(b)) continue;
      ctx.moveTo(...pt(a)); ctx.lineTo(...pt(b));
    }
    ctx.stroke();
    if ([11, 12, 24, 23].every(ok)) { // torso
      ctx.beginPath();
      [11, 12, 24, 23].forEach((i, k) => (k ? ctx.lineTo(...pt(i)) : ctx.moveTo(...pt(i))));
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    const head = [0, 7, 8].filter(ok);
    if (head.length) { // head and hair
      const hx = head.reduce((s, i) => s + lm[i].x, 0) / head.length * w;
      const hy = head.reduce((s, i) => s + lm[i].y, 0) / head.length * h;
      ctx.beginPath(); ctx.arc(hx, hy, bodyH * 0.13, 0, Math.PI * 2); ctx.fill();
      if (ok(11) && ok(12)) { // neck
        const [sx, sy] = [(lm[11].x + lm[12].x) / 2 * w, (lm[11].y + lm[12].y) / 2 * h];
        ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(sx, sy); ctx.stroke();
      }
    }
    const px = ctx.getImageData(0, 0, w, h).data;
    for (let k = 0, j = 0; k < n; k++, j += 4) skel[k] = px[j] / 255;

    // A generous zone around the skeleton for clothes and hair.
    boxMean(skel, w, h, Math.max(2, Math.round(bodyH * 0.06)), zone, tmp);
    const m = target.mask;
    for (let k = 0; k < n; k++) {
      const z = Math.min(1, zone[k] * 4);
      let v = Math.min(z, Math.max(m[k], skel[k]));
      for (const o of others) { // pixels another member clearly owns
        const ov = o.mask[k];
        if (ov > 0.5 && ov > m[k] + 0.15) { v = 0; break; }
      }
      skel[k] = v;
    }
    // Soften and grow slightly so the matting keeps fine edges.
    boxMean(skel, w, h, Math.max(1, Math.round(Math.max(w, h) * 0.008)), out, tmp);
    for (let k = 0; k < n; k++) out[k] = Math.min(1, out[k] * 1.6);
    return out.slice();
  };
}

// ---------------------------------------------------------------------------
// Edge refinement: a guided filter snaps the model's soft mask to colour edges (hair!).

// Pieces of the matte floating away from her (a neighbour's raised arm inside her area): keep her
// body - the biggest piece - and whatever touches the ground just around it; clear the rest.
function makeStrayRemover(w, h) {
  const n = w * h, label = new Int32Array(n), stack = new Int32Array(n), near = new Float32Array(n), tmp = new Float32Array(n);
  const body = new Float32Array(n);
  const reach = Math.max(2, Math.round(Math.max(w, h) * 0.015));
  return alpha => {
    label.fill(0);
    let next = 0, best = 0, bestArea = 0;
    const areas = [0];
    for (let k = 0; k < n; k++) {
      if (label[k] || alpha[k] < 0.3) continue;
      const id = ++next;
      let top = 0, area = 0;
      stack[top++] = k; label[k] = id;
      while (top) {
        const q = stack[--top]; area++;
        const x = q % w;
        if (x > 0 && !label[q - 1] && alpha[q - 1] >= 0.3) { label[q - 1] = id; stack[top++] = q - 1; }
        if (x < w - 1 && !label[q + 1] && alpha[q + 1] >= 0.3) { label[q + 1] = id; stack[top++] = q + 1; }
        if (q >= w && !label[q - w] && alpha[q - w] >= 0.3) { label[q - w] = id; stack[top++] = q - w; }
        if (q < n - w && !label[q + w] && alpha[q + w] >= 0.3) { label[q + w] = id; stack[top++] = q + w; }
      }
      areas.push(area);
      if (area > bestArea) { bestArea = area; best = id; }
    }
    if (next < 2) return;
    for (let k = 0; k < n; k++) body[k] = label[k] === best ? 1 : 0;
    boxMean(body, w, h, reach, near, tmp);
    const keep = new Uint8Array(next + 1);
    keep[best] = 1;
    for (let k = 0; k < n; k++) if (label[k] && near[k] > 0) keep[label[k]] = 1;
    // (big pieces stay too: two halves of her split by a microphone stand)
    for (let id = 1; id <= next; id++) if (areas[id] > bestArea * 0.25) keep[id] = 1;
    // (faint edge pixels - hair - belong to no piece: they stay where they're near her body)
    for (let k = 0; k < n; k++) if (alpha[k] > 0 && (label[k] ? !keep[label[k]] : near[k] === 0)) alpha[k] = 0;
  };
}

function boxMean(src, w, h, r, out, tmp) {
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let acc = 0;
    for (let x = 0; x <= Math.min(r, w - 1); x++) acc += src[row + x];
    for (let x = 0; x < w; x++) {
      const lo = x - r, hi = x + r;
      tmp[row + x] = acc / (Math.min(hi, w - 1) - Math.max(lo, 0) + 1);
      if (hi + 1 < w) acc += src[row + hi + 1];
      if (lo >= 0) acc -= src[row + lo];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = 0; y <= Math.min(r, h - 1); y++) acc += tmp[y * w + x];
    for (let y = 0; y < h; y++) {
      const lo = y - r, hi = y + r;
      out[y * w + x] = acc / (Math.min(hi, h - 1) - Math.max(lo, 0) + 1);
      if (hi + 1 < h) acc += tmp[(hi + 1) * w + x];
      if (lo >= 0) acc -= tmp[lo * w + x];
    }
  }
}

function makeGuidedFilter(w, h) {
  const n = w * h;
  const buf = () => new Float32Array(n);
  const tmp = buf(), II = buf(), Ip = buf(), mI = buf(), mp = buf(), mII = buf(), mIp = buf(), a = buf(), b = buf();
  return (I, p, r, eps, out) => {
    for (let i = 0; i < n; i++) { II[i] = I[i] * I[i]; Ip[i] = I[i] * p[i]; }
    boxMean(I, w, h, r, mI, tmp);
    boxMean(p, w, h, r, mp, tmp);
    boxMean(II, w, h, r, mII, tmp);
    boxMean(Ip, w, h, r, mIp, tmp);
    for (let i = 0; i < n; i++) {
      const varI = mII[i] - mI[i] * mI[i];
      const cov = mIp[i] - mI[i] * mp[i];
      a[i] = cov / (varI + eps);
      b[i] = mp[i] - a[i] * mI[i];
    }
    boxMean(a, w, h, r, II, tmp); // reuse buffers for mean(a), mean(b)
    boxMean(b, w, h, r, Ip, tmp);
    for (let i = 0; i < n; i++) out[i] = II[i] * I[i] + Ip[i];
  };
}

// ---------------------------------------------------------------------------
// Video helpers

function fitSize(w, h, maxSide) {
  const s = Math.min(1, maxSide / Math.max(w, h));
  return [Math.max(2, Math.round(w * s / 2) * 2), Math.max(2, Math.round(h * s / 2) * 2)];
}

// The part of the frame to use. Screen recordings carry black bars we crop away.
export function fullRect(video) {
  return { x: 0, y: 0, w: video.videoWidth, h: video.videoHeight };
}

// Multiples of 32 so the matting model's stages divide evenly (the slight stretch is undone
// when the matte is scaled back onto the frame).
export function analysisSize(rect, maxSide) {
  const side = analysisSide(maxSide);
  const s = Math.min(1, side / Math.max(rect.w, rect.h));
  const to32 = v => Math.max(64, Math.round(v * s / 32) * 32);
  return [to32(rect.w), to32(rect.h)];
}

// Finds black bars (letterboxing, screen-recording borders) by sampling a few frames.
// Returns the picture area in video pixels, or null when there's nothing worth cropping.
export async function detectContentRect(video, start, end) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.min(1, 240 / Math.max(vw, vh));
  const w = Math.max(8, Math.round(vw * s)), h = Math.max(8, Math.round(vh * s));
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const rowLit = new Uint8Array(h), colLit = new Uint8Array(w);
  const samples = 6;
  for (let i = 0; i < samples; i++) {
    await seekVideo(video, start + (end - start) * (i + 0.5) / samples);
    ctx.drawImage(video, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h).data;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const j = (y * w + x) * 4;
        if (Math.max(d[j], d[j + 1], d[j + 2]) > 40) { rowLit[y] = 1; colLit[x] = 1; }
      }
    }
  }
  const first = a => a.indexOf(1), last = a => a.lastIndexOf(1);
  const y0 = first(rowLit), y1 = last(rowLit), x0 = first(colLit), x1 = last(colLit);
  if (y0 < 0 || x0 < 0) return null;
  const cw = x1 - x0 + 1, ch = y1 - y0 + 1;
  // Ignore slivers (<3%) and anything that would throw away most of the frame.
  if (cw * ch < w * h * 0.15 || (cw > w * 0.97 && ch > h * 0.97)) return null;
  const inset = 1; // stay clear of the soft edge between bar and picture
  const rx = Math.round((x0 + inset) / s), ry = Math.round((y0 + inset) / s);
  return {
    x: rx, y: ry,
    w: Math.min(vw - rx, Math.round((cw - 2 * inset) / s)),
    h: Math.min(vh - ry, Math.round((ch - 2 * inset) / s)),
  };
}

async function pickVideoEncoder(width, height, fps, bitrate) {
  if (typeof VideoEncoder === 'undefined') {
    throw new Error('이 브라우저는 영상 인코딩(WebCodecs)을 지원하지 않아요. 최신 Chrome/Safari/Edge/Firefox를 써 주세요.');
  }
  const candidates = [
    ['avc', 'avc1.640033'], ['avc', 'avc1.4d0033'], ['avc', 'avc1.42e033'],
    ['vp9', 'vp09.00.51.08'], ['av1', 'av01.0.12M.08'],
  ];
  for (const [mux, codec] of candidates) {
    const config = { codec, width, height, bitrate, framerate: fps, latencyMode: 'quality' };
    if (mux === 'avc') config.avc = { format: 'avc' };
    try {
      const res = await VideoEncoder.isConfigSupported(config);
      if (res.supported) return { mux, config };
    } catch {}
  }
  throw new Error(`이 기기에서 ${width}×${height} 영상을 인코딩할 수 없어요. 화질을 낮춰 보세요.`);
}

async function prepareAudio(file, start, end) {
  if (typeof AudioEncoder === 'undefined') return null;
  const sampleRate = 48000;
  let decoded;
  try {
    const ctx = new OfflineAudioContext(2, sampleRate, sampleRate);
    decoded = await ctx.decodeAudioData(await file.arrayBuffer());
  } catch (err) {
    console.warn('audio decode failed', err);
    return null;
  }
  const channels = Math.min(2, decoded.numberOfChannels);
  for (const [mux, codec] of [['aac', 'mp4a.40.2'], ['opus', 'opus']]) {
    const config = { codec, sampleRate, numberOfChannels: channels, bitrate: 160000 };
    try {
      const res = await AudioEncoder.isConfigSupported(config);
      if (res.supported) {
        return {
          mux, config, decoded, channels,
          from: Math.floor(start * decoded.sampleRate),
          to: Math.min(decoded.length, Math.floor(end * decoded.sampleRate)),
        };
      }
    } catch {}
  }
  return null;
}

async function encodeAudio(audio, out) {
  let failure = null;
  const encoder = new AudioEncoder({
    output: (chunk, meta) => out.push({ chunk, meta }),
    error: e => { failure = e; },
  });
  encoder.configure(audio.config);
  const { decoded, channels, from, to } = audio;
  const sr = decoded.sampleRate;
  const block = 4800;
  const planes = Array.from({ length: channels }, (_, c) => decoded.getChannelData(c));
  for (let f = from; f < to; f += block) {
    const n = Math.min(block, to - f);
    const data = new Float32Array(n * channels);
    for (let c = 0; c < channels; c++) data.set(planes[c].subarray(f, f + n), c * n);
    const ad = new AudioData({
      format: 'f32-planar', sampleRate: sr, numberOfFrames: n, numberOfChannels: channels,
      timestamp: Math.round((f - from) / sr * 1e6), data,
    });
    encoder.encode(ad);
    ad.close();
    if (failure) throw failure;
  }
  await encoder.flush();
  encoder.close();
  if (failure) throw failure;
}

// ---------------------------------------------------------------------------
// SAM 2 helpers

// The first frame's prompt. Points on her own joints (head, chest, hips, knees) say "this whole
// person"; a box around her helps too, but only when nobody else overlaps it (otherwise the box
// would cover them as well). Other people found next to her get "not her" points.
function samPrompt(o) {
  const S = sam2.SIZE, t = o.target;
  const pts = [], labels = [];
  const add = (x, y, label) => { if (x > 0 && x < 1 && y > 0 && y < 1) { pts.push([x * S, y * S]); labels.push(label); } };
  if (o.tap) add(o.tap[0], o.tap[1], 1);
  const lm = t.landmarks || [];
  const vis = i => lm[i] && (lm[i].visibility ?? 1) > 0.5;
  const mid = (a, b) => [(lm[a].x + lm[b].x) / 2, (lm[a].y + lm[b].y) / 2];
  if (vis(0)) add(lm[0].x, lm[0].y, 1);
  if (vis(11) && vis(12)) add(...mid(11, 12).map((v, k) => (k ? v + 0.02 : v)), 1); // chest
  if (vis(23) && vis(24)) add(...mid(23, 24), 1);
  for (const k of [25, 26]) if (vis(k)) add(lm[k].x, lm[k].y, 1);
  if (!pts.length) add(t.cx, t.cy, 1);

  const others = (o.others || []).filter(p => p !== t && p.area && p.bbox);
  let box = null;
  if (t.bbox) {
    const [x0, y0, x1, y1] = t.bbox;
    const area = (x1 - x0) * (y1 - y0);
    const overlaps = others.some(p => {
      const ix = Math.max(0, Math.min(x1, p.bbox[2]) - Math.max(x0, p.bbox[0]));
      const iy = Math.max(0, Math.min(y1, p.bbox[3]) - Math.max(y0, p.bbox[1]));
      return ix * iy > area * 0.08;
    });
    const pad = 0.02;
    if (!overlaps) box = [Math.max(0, x0 - pad) * S, Math.max(0, y0 - pad) * S, Math.min(1, x1 + pad) * S, Math.min(1, y1 + pad) * S];
  }
  for (const p of others) {
    const inside = t.bbox && p.cx > t.bbox[0] && p.cx < t.bbox[2] && p.cy > t.bbox[1] && p.cy < t.bbox[3];
    if (!inside) add(p.cx, p.cy, 0);
  }
  return [pts, labels, box];
}

// SAM's 256x256 mask (of the square-squashed frame) -> a soft region at analysis size, grown a
// little (more when SAM only runs every few frames, to cover her movement in between).
// region.inner is her body shrunk away from the outline: solid for sure, whatever the matting
// model thinks (it sometimes turns a dress or an arm see-through against a similar background).
function makeSamRegion(w, h) {
  const n = w * h, core = new Float32Array(n), tmp = new Float32Array(n), grown = new Float32Array(n);
  const hard = new Float32Array(n), shrunk = new Float32Array(n);
  return (low, every) => {
    for (let y = 0; y < h; y++) {
      const fy = Math.max(0, (y + 0.5) / h * 256 - 0.5), y0 = Math.floor(fy), y1 = Math.min(255, y0 + 1), wy = fy - y0;
      for (let x = 0; x < w; x++) {
        const fx = Math.max(0, (x + 0.5) / w * 256 - 0.5), x0 = Math.floor(fx), x1 = Math.min(255, x0 + 1), wx = fx - x0;
        const v = (low[y0 * 256 + x0] * (1 - wx) + low[y0 * 256 + x1] * wx) * (1 - wy)
          + (low[y1 * 256 + x0] * (1 - wx) + low[y1 * 256 + x1] * wx) * wy;
        core[y * w + x] = v === v ? 1 / (1 + Math.exp(-v)) : 0; // NaN (a bad GPU step): not her
      }
    }
    const side = Math.max(w, h);
    const r = Math.max(2, Math.round(side * (0.012 + 0.006 * (every - 1))));
    boxMean(core, w, h, r, grown, tmp);
    const region = new Float32Array(n);
    for (let k = 0; k < n; k++) region[k] = Math.min(1, Math.max(core[k], grown[k] * 2.5));
    for (let k = 0; k < n; k++) hard[k] = core[k] > 0.5 ? 1 : 0;
    boxMean(hard, w, h, Math.max(2, Math.round(side * (0.01 + 0.005 * (every - 1)))), shrunk, tmp);
    const inner = new Float32Array(n);
    for (let k = 0; k < n; k++) { const v = (shrunk[k] - 0.92) / 0.08; inner[k] = v <= 0 ? 0 : v >= 1 ? 1 : v; }
    // Her area for the matting model: firm out to most of the margin, so the matte isn't cut part
    // way into her edges (that cut moved with every SAM step and made the edges shimmer).
    const area = new Float32Array(n);
    for (let k = 0; k < n; k++) { const v = (grown[k] - 0.1) / 0.25; area[k] = v <= 0 ? 0 : v >= 1 ? 1 : v * v * (3 - 2 * v); }
    region.core = core.slice();
    region.inner = inner;
    region.area = area;
    return region;
  };
}

// Which pose-model person a face belongs to: the one whose nose/eyes fall inside the face box.
export function personForFace(persons, fc, rw, rh) {
  const [x0, y0, x1, y1] = [fc.box[0] / rw, fc.box[1] / rh, fc.box[2] / rw, fc.box[3] / rh];
  const padX = (x1 - x0) * 0.25, padY = (y1 - y0) * 0.25;
  let best = null, bestHits = 0;
  for (const p of persons) {
    let hits = 0;
    for (const i of [0, 2, 5]) {
      const l = p.landmarks?.[i];
      if (l && l.x > x0 - padX && l.x < x1 + padX && l.y > y0 - padY && l.y < y1 + padY) hits++;
    }
    if (hits > bestHits) { bestHits = hits; best = p; }
  }
  if (best) return best;
  // No skeleton matched: the person whose outline covers the middle of the face.
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  for (const p of persons) {
    const x = Math.min(p.w - 1, Math.round(cx * p.w)), y = Math.min(p.h - 1, Math.round(cy * p.h));
    if (p.mask[y * p.w + x] > 0.5) return p;
  }
  return null;
}

// The pose model misses people in a crowd (it finds a handful per frame). If nobody matches
// her face, look again at a crop around it (her body hangs below her face), and map that person
// back onto the whole frame. `src` is the frame at analysis size, like `persons`' masks.
export async function personAtFace(src, persons, fc, rw, rh) {
  const hit = personForFace(persons, fc, rw, rh);
  if (hit) return hit;
  const W = src.width, H = src.height;
  const fx0 = fc.box[0] / rw * W, fy0 = fc.box[1] / rh * H, fx1 = fc.box[2] / rw * W, fy1 = fc.box[3] / rh * H;
  const fw = fx1 - fx0, fh = fy1 - fy0;
  const x0 = Math.max(0, Math.floor(fx0 - 2.5 * fw)), x1 = Math.min(W, Math.ceil(fx1 + 2.5 * fw));
  const y0 = Math.max(0, Math.floor(fy0 - fh)), y1 = Math.min(H, Math.ceil(fy1 + 8 * fh));
  const cw = x1 - x0, ch = y1 - y0;
  if (cw < 16 || ch < 16) return null;
  // MediaPipe's CPU path aborts (and stays broken) when it finds someone in a picture whose width
  // isn't a multiple of 4, so scale the crop to one.
  const c = document.createElement('canvas');
  c.width = Math.ceil(cw / 4) * 4; c.height = Math.ceil(ch / 4) * 4;
  c.getContext('2d').drawImage(src, x0, y0, cw, ch, 0, 0, c.width, c.height);
  const inCrop = { ...fc, box: [fx0 - x0, fy0 - y0, fx1 - x0, fy1 - y0] };
  const p = personForFace(await detect(c), inCrop, cw, ch);
  if (!p) return null;
  const mask = new Float32Array(W * H);
  for (let y = 0; y < ch; y++) {
    const sy = Math.min(p.h - 1, Math.floor(y * p.h / ch));
    for (let x = 0; x < cw; x++) mask[(y0 + y) * W + x0 + x] = p.mask[sy * p.w + Math.min(p.w - 1, Math.floor(x * p.w / cw))];
  }
  const landmarks = p.landmarks.map(l => ({ ...l, x: (l.x * cw + x0) / W, y: (l.y * ch + y0) / H }));
  return describe(mask, W, H, landmarks);
}

// A face clear enough to say "that's not her": big enough and turned mostly towards the camera
// (a profile can fool the recogniser even for the same person).
function clearFace(fc, minSize) {
  const [le, re, nose] = fc.kps;
  const eyes = Math.hypot(re[0] - le[0], re[1] - le[1]);
  const t = ((nose[0] - le[0]) * (re[0] - le[0]) + (nose[1] - le[1]) * (re[1] - le[1])) / (eyes * eyes || 1);
  return fc.size >= minSize && eyes > 0.3 * (fc.box[2] - fc.box[0]) && t > 0.25 && t < 0.75;
}

// ---------------------------------------------------------------------------
// Main pipeline

const abortError = () => new DOMException('취소됨', 'AbortError');
// Let the page handle taps (like 취소) between the heavy steps of a frame.
const breathe = () => (globalThis.scheduler?.yield ? scheduler.yield() : new Promise(r => setTimeout(r, 0)));

/**
 * @param {object} o
 * @param {File} o.file          source video file
 * @param {HTMLVideoElement} o.video  a loaded <video> for `file`
 * @param {number} o.start, o.end     trim range in seconds
 * @param {{x,y,w,h}} [o.rect] part of the frame to use (video pixels); whole frame if omitted
 * @param {object} o.target       person picked on the first frame (from detect)
 * @param {number} o.fps, o.maxSide, o.bitrate
 * @param {boolean} o.withAudio
 * @param {boolean} [o.useMatting]  use the GPU matting model when available (default true)
 * @param {boolean} [o.useSam]      follow her with SAM 2 when WebGPU is available (default true)
 * @param {[number, number]} [o.tap] where she was tapped on the pick frame (0..1 coords), if tapped
 * @param {object[]} [o.others]      the other people found on the pick frame
 * @param {{requested: boolean}} [o.stop] set requested=true to finish early and keep what's done
 * @param {Float32Array} [o.faceRef]  her face (face.reference of the registered photos): checked
 *   every half second, and the tracker is put back on her if it has drifted to someone else
 * @param {Float32Array} [o.faceSeed]  her face on the pick frame, when she was picked by face
 * @param {(p:object)=>void} o.onProgress
 * @param {AbortSignal} o.signal
 */
export async function extract(o) {
  const { video, start, end, fps, signal } = o;
  const check = () => { if (signal?.aborted) throw abortError(); };

  const r = o.rect ?? fullRect(video);
  const [aw, ah] = analysisSize(r, o.maxSide);
  // Circle mode: a round window that follows her, so the clip is a square picture with a round
  // alpha instead of her outline. (Her cut-out is still made, to know where she is.)
  const circle = o.shape === 'circle';
  const side = Math.max(64, Math.round(Math.min(o.maxSide, 1080, r.w, r.h) / 16) * 16);
  const [hw, hh] = circle ? [side, side] : fitSize(r.w, r.h, o.maxSide);
  // Portrait fancams pack side-by-side, landscape ones top/bottom, keeping frames squarish.
  const layout = hh >= hw ? 'sbs' : 'tb';
  const W = layout === 'sbs' ? hw * 2 : hw;
  const H = layout === 'sbs' ? hh : hh * 2;
  const alphaX = layout === 'sbs' ? hw : 0, alphaY = layout === 'sbs' ? 0 : hh;

  o.onProgress?.({ done: 0, total: 1, elapsed: 0, eta: 0, stage: '준비 중' });
  // The GPU models run in a worker that is shut down if the GPU hangs (see js/gpu.js); whatever
  // fails falls back to the CPU tools below.
  // The circle only follows where she is, so it doesn't need the matting model; with the fast
  // tracker (o.tracker 'yolo', the default for it) YOLO boxes stand in for SAM 2 as well.
  const wantYolo = o.shape === 'circle' && o.tracker !== 'sam';
  if (o.shape === 'circle') o = { ...o, useMatting: false, ...(wantYolo ? { useSam: false } : {}) };
  let gpu = (o.useMatting !== false || o.useSam !== false || o.faceRef || wantYolo) && await webgpuAvailable() ? gpuWorker() : null;
  const configure = () => gpu.call('config', [{ samEp: globalThis.__FANCAM_SAM_EP__, recent: globalThis.__FANCAM_SAM_RECENT__ }]).catch(() => {});
  if (gpu) {
    configure();
    gpu.onProgress = f => o.onProgress?.({ done: 0, total: 1, elapsed: 0, eta: 0, stage: `SAM 2 추적 모델 받는 중 ${Math.round(f * 100)}%` });
  }
  const off = what => err => { console.warn(what + ' off:', err); return null; };
  // Loading deadlines. Each model loads in a few seconds once downloaded, but now and then the
  // worker's GPU never gets going; then a fresh worker usually does. (SAM 2's ~100 MB first download
  // reports progress, which restarts the deadline.)
  const LOAD = 45000, SAM_LOAD = 60000;
  // The matting model in the worker (whichever worker is current).
  const workerMatter = () => ({
    run(px) { const copy = px.slice(); return gpu.call('matte', [copy], { timeout: 20000, transfer: [copy.buffer] }); },
  });
  // One after another, each with its own deadline: in the worker they'd queue behind each other
  // anyway, and a deadline shouldn't count the time spent waiting for the others.
  const loadGpuModels = async () => {
    const load = LOAD, samLoad = SAM_LOAD;
    const m = o.useMatting === false ? null
      : await gpu.call('mattingCreate', [aw, ah], { timeout: load }).then(workerMatter).catch(off('matting'));
    const samOk = o.useSam !== false && !gpu.dead && await gpu.call('samLoad', [], { timeout: samLoad }).then(() => true, off('SAM 2'));
    const faceOk = !!o.faceRef && !gpu.dead && await gpu.call('faceLoad', [], { timeout: load }).then(() => true, () => false);
    const yoloOk = wantYolo && !gpu.dead && await gpu.call('yoloLoad', [], { timeout: load }).then(() => true, off('YOLO'));
    return [m, samOk, faceOk, yoloOk];
  };
  let [source, [matter, samOk, faceOk, yoloOk]] = await Promise.all([
    openSource(o.file, video),
    gpu ? loadGpuModels() : [null, false, false, false],
  ]);
  let yoloTrack = !!yoloOk; // (without it the pose model follows her instead)
  let samModels = samOk ? gpu : null;
  let faceModels = !o.faceRef ? null
    : faceOk ? { gpu } : await face.load().then(local => ({ local })).catch(off('face check'));
  const cleanup = [() => source.close()];
  try {
    check();
    const enc = await pickVideoEncoder(W, H, fps, o.bitrate);

    // Audio: copy it straight from the file when possible, else decode and re-encode.
    let audioOut = null;
    if (o.withAudio && source.audio) {
      const c = source.audio.config;
      audioOut = { codec: source.audio.mux, numberOfChannels: c.numberOfChannels, sampleRate: c.sampleRate };
    }
    const reencode = o.withAudio && !audioOut ? await prepareAudio(o.file, start, end) : null;
    if (reencode) audioOut = { codec: reencode.mux, numberOfChannels: reencode.channels, sampleRate: reencode.config.sampleRate };
    check();

    const parts = [];
    const muxer = new Muxer({
      target: new StreamTarget({ onData: (data, position) => parts.push({ data: data.slice(), position }), chunked: true }),
      video: { codec: enc.mux, width: W, height: H, frameRate: fps },
      audio: audioOut || undefined,
      fastStart: 'fragmented',
      firstTimestampBehavior: 'offset',
    });

    // Audio goes into the file alongside the video (so stopping early can cut it at the same point).
    const audioChunks = [];
    if (audioOut && source.audio && !reencode) {
      for await (const chunk of source.audioChunks(start, end, signal)) {
        audioChunks.push({ chunk, meta: audioChunks.length ? undefined : { decoderConfig: source.audio.config } });
      }
    } else if (reencode) {
      await encodeAudio(reencode, audioChunks);
    }
    const hasAudio = audioChunks.length > 0;
    let audioNext = 0;
    const addAudioUntil = us => {
      while (audioNext < audioChunks.length && audioChunks[audioNext].chunk.timestamp < us) {
        const { chunk, meta } = audioChunks[audioNext++];
        muxer.addAudioChunk(chunk, meta);
      }
    };
    check();

    let encodeError = null;
    const encoder = new VideoEncoder({
      output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
      error: e => { encodeError = e; },
    });
    encoder.configure(enc.config);
    cleanup.push(() => { if (encoder.state !== 'closed') encoder.close(); });

    // Canvases
    const mk = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
    const aCanvas = mk(aw, ah), aCtx = aCanvas.getContext('2d', { willReadFrequently: true });
    const mCanvas = mk(aw, ah), mCtx = mCanvas.getContext('2d');
    const out = mk(W, H), oCtx = out.getContext('2d', { alpha: false });
    oCtx.imageSmoothingEnabled = true;
    oCtx.imageSmoothingQuality = 'high';
    aCtx.imageSmoothingQuality = 'high';
    const maskImage = mCtx.createImageData(aw, ah);

    const n = aw * ah;
    const guide = new Float32Array(n), raw = new Float32Array(n), refined = new Float32Array(n);
    let alpha = new Float32Array(n), prevAlpha = null;
    let prevGray = null; // last frame's brightness, for smoothing still edges
    const guided = makeGuidedFilter(aw, ah);
    const buildRegion = makeRegionBuilder(aw, ah);
    const dropStrays = makeStrayRemover(aw, ah);
    const radius = Math.max(2, Math.round(Math.max(aw, ah) / 110));
    const tracker = new Tracker(o.target, fps);
    let lastRegion = null, badMatte = 0;

    // SAM 2 follows her every `samEvery` frames; the matting model draws the edges every frame.
    // SAM 2 runs in the GPU worker, which keeps the tracker's memory; start() begins a new one.
    const samTracker = {
      start: (input, ...prompt) => gpu.call('samStart', [input, ...prompt], { timeout: 30000 }),
      next: input => gpu.call('samNext', [input], { timeout: 30000 }),
    };
    let sam = samModels ? samTracker : null;
    const samEvery = Math.max(1, Math.round(fps / 15));
    const sCanvas = gpu ? mk(sam2.SIZE, sam2.SIZE) : null;
    const sCtx = sCanvas?.getContext('2d', { willReadFrequently: true });
    const toRegion = gpu ? makeSamRegion(aw, ah) : null;
    let samRegion = null, samSeen = true, samSince = 0;

    // Face check: every half second, find her face among everyone's; if the tracker's mask
    // isn't on her face (or also covers someone else's), start it again on her.
    // (every quarter second: when she slips behind someone, the tracker can latch onto the member
    // next to her, and this is what notices)
    const faceEvery = samEvery * Math.max(1, Math.round(fps * 0.25 / samEvery));
    const faceMin = Math.max(r.w, r.h) * 0.012;       // smallest face worth recognising
    let faceHidden = false, faceFixes = 0;
    const herFace = o.faceRef && face.matcher(o.faceRef, o.faceSeed);
    let faceCanvas = null;
    const lookForFaces = async f => {
      try {
        let found;
        if (faceModels.gpu && !faceModels.gpu.dead) {
          faceCanvas ??= new OffscreenCanvas(r.w, r.h);
          f.draw(faceCanvas.getContext('2d'), r, r.w, r.h);
          const bmp = faceCanvas.transferToImageBitmap();
          found = await faceModels.gpu.call('faces', [bmp, r.w, r.h, faceMin], { timeout: 20000, transfer: [bmp] });
        } else {
          faceModels.local ??= await face.load(); // the GPU worker was stopped: carry on on the CPU
          found = await face.faces(faceModels.local, ctx => f.draw(ctx, r, r.w, r.h), r.w, r.h, { minSize: faceMin });
        }
        // A fancam keeps her near the middle. Someone at the edge needs a much surer match to count
        // as her (another member there scored 0.51 while her half-hidden face scored 0.40).
        const nearMiddle = fc => { const x = (fc.box[0] + fc.box[2]) / 2 / r.w; return x > 0.2 && x < 0.8; };
        return { found, her: herFace.findHer(found, fc => nearMiddle(fc) || fc.sim >= 0.55) };
      } catch (err) {
        console.warn('face check failed:', err);
        restartGpu(err);
        return { found: [], her: null };
      }
    };
    // The GPU can be reset under us (a GPU fault shows as "device is lost", or the worker stops
    // answering). Then start a fresh worker, wait for its models (twice at most) and start SAM 2
    // again on her; if that fails, the CPU tools carry on.
    let gpuRestarts = 0, gpuRestarting = null, samRestart = false, lastFace = null;
    const gpuBroke = err => /OrtRun|lost|GPU가|worker/i.test(err?.message || '');
    const restartGpu = err => {
      if (!gpu || gpuRestarting || gpuRestarts >= 2 || !gpuBroke(err)) return;
      gpuRestarts++;
      console.warn('GPU models failed, restarting them:', err);
      gpu.kill('restarting: ' + err.message);
      gpu = gpuWorker();
      configure();
      matter = null;
      gpuRestarting = loadGpuModels().then(([m, samBack, faceBack, yoloBack]) => {
        matter = m;
        if (wantYolo) yoloTrack = !!yoloBack;
        samRestart = !!samBack;
        if (faceBack && faceModels) faceModels = { ...faceModels, gpu };
        globalThis.__fancamDebug?.push({ restarted: gpuRestarts, matting: !!m, sam: !!samBack, face: !!faceBack });
      }).finally(() => { gpuRestarting = null; });
    };
    if (gpu?.dead) restartGpu(new Error(gpu.dead)); // stopped while loading: try a fresh worker
    // How much of a face (eyes and nose) lies inside a mask at analysis size.
    const faceIn = (fc, m) => {
      if (!m) return 0;
      let hit = 0;
      for (const [x, y] of fc.kps.slice(0, 3)) {
        const ax = Math.min(aw - 1, Math.max(0, Math.round(x / r.w * aw))), ay = Math.min(ah - 1, Math.max(0, Math.round(y / r.h * ah)));
        if (m[ay * aw + ax] > 0.5) hit++;
      }
      return hit / 3;
    };
    // Hiding on someone else's face: at once if it's somewhere else, but where she was a moment ago a
    // poor match is more likely her on a bad frame (blur, hair across her face), so look twice there.
    let lastHer = null, doubted = false;
    const sawHer = (fc, i) => {
      lastHer = { x: (fc.box[0] + fc.box[2]) / 2, y: (fc.box[1] + fc.box[3]) / 2, w: fc.box[2] - fc.box[0], i };
      doubted = false;
    };
    const hideFor = (strangers, i) => {
      const nearHer = fc => lastHer && i - lastHer.i <= fps
        && Math.hypot((fc.box[0] + fc.box[2]) / 2 - lastHer.x, (fc.box[1] + fc.box[3]) / 2 - lastHer.y) < 1.5 * lastHer.w;
      if (strangers.every(nearHer) && !doubted) { doubted = true; return false; }
      return true;
    };
    // SAM prompt for her, found by face on this frame.
    const promptFromFace = async (her, found) => {
      const persons = await detect(aCanvas);
      const target = await personAtFace(aCanvas, persons, her, r.w, r.h);
      const cx = (her.box[0] + her.box[2]) / 2 / r.w, cy = (her.box[1] + her.box[3]) / 2 / r.h;
      const [pts, labels, box] = samPrompt({
        target: target || { cx, cy, landmarks: [], bbox: null }, tap: [cx, cy],
        others: persons.filter(p => p !== target),
      });
      if (!target?.landmarks?.length) {
        // No skeleton for her: a face alone makes SAM pick just her head and shoulders (and then
        // remember that), so also point at where her body is, below her face.
        const fh = (her.box[3] - her.box[1]) / r.h;
        for (const dy of [2.5, 5, 7.5]) if (cy + dy * fh < 0.98) { pts.push([cx * sam2.SIZE, (cy + dy * fh) * sam2.SIZE]); labels.push(1); }
      }
      for (const fc of found) {
        if (fc === her) continue;
        const x = (fc.box[0] + fc.box[2]) / 2 / r.w, y = (fc.box[1] + fc.box[3]) / 2 / r.h;
        pts.push([x * sam2.SIZE, y * sam2.SIZE]); labels.push(0);
      }
      return [pts, labels, box];
    };
    let offFace = 0; // face checks in a row where the tracker's mask missed her face

    const bounds = [1, 1, 0, 0];

    // Circle mode's fast tracker: YOLO finds everyone each frame; her box is the one her face is
    // in (checked every quarter second), else the one that best continues her last box (where it
    // was, moved as it was moving) and looks like her (colours: YOLO also finds the cameraman in
    // black when she's behind him). Her box goes into `alpha`, and the circle centres on it as it
    // would on a cut-out. A clear stranger's face in her box doesn't blank the circle: it goes back
    // to where her face last was and waits there for her face.
    let yoloBox = o.target?.bbox ? [...o.target.bbox] : null, yoloVel = [0, 0], yoloLost = 0;
    let herBox = yoloBox, herHist = null, frozen = false;
    // Colours in the middle of a box (her dress and arms, not the background at its edges).
    const histOf = (b, px) => {
      const h = new Float32Array(64);
      const x0 = Math.floor((b[0] + (b[2] - b[0]) * 0.25) * aw), x1 = Math.ceil((b[2] - (b[2] - b[0]) * 0.25) * aw);
      const y0 = Math.floor((b[1] + (b[3] - b[1]) * 0.2) * ah), y1 = Math.ceil((b[3] - (b[3] - b[1]) * 0.2) * ah);
      let sum = 0;
      for (let y = Math.max(0, y0); y < Math.min(ah, y1); y += 3) {
        for (let x = Math.max(0, x0); x < Math.min(aw, x1); x += 3) {
          const j = (y * aw + x) * 4;
          h[(px[j] >> 6) * 16 + (px[j + 1] >> 6) * 4 + (px[j + 2] >> 6)]++; sum++;
        }
      }
      if (sum) for (let k = 0; k < 64; k++) h[k] /= sum;
      return h;
    };
    const alike = (a, b) => { let s = 0; for (let k = 0; k < 64; k++) s += Math.sqrt(a[k] * b[k]); return s; };
    const learnHist = h => { if (!herHist) herHist = h; else for (let k = 0; k < 64; k++) herHist[k] = herHist[k] * 0.7 + h[k] * 0.3; };
    const trackWithYolo = async (f, i, px) => {
      let boxes;
      try {
        const copy = px.slice();
        boxes = (await gpu.call('people', [copy, aw, ah], { timeout: 20000, transfer: [copy.buffer] })).map(p => p.box);
      } catch (err) {
        if (err.name === 'AbortError') throw err;
        console.warn('YOLO failed, following with the pose model:', err);
        yoloTrack = false;
        restartGpu(err);
        return false;
      }
      const area = b => (b[2] - b[0]) * (b[3] - b[1]);
      let pick = null;
      if (!yoloBox && o.tap) {
        const [tx, ty] = o.tap;
        pick = boxes.filter(b => tx > b[0] && tx < b[2] && ty > b[1] && ty < b[3]).sort((a, b) => area(a) - area(b))[0] || null;
      }
      if (faceModels && i % faceEvery === 0) {
        const { found, her } = await lookForFaces(f);
        lastFace = { i, found, her };
        check();
        // a face belongs to a box when it sits in the box's upper part
        const inBox = (fc, b) => {
          const x = (fc.box[0] + fc.box[2]) / 2 / r.w, y = (fc.box[1] + fc.box[3]) / 2 / r.h;
          return x > b[0] && x < b[2] && y > b[1] && y < b[1] + (b[3] - b[1]) * 0.45;
        };
        if (her) {
          sawHer(her, i);
          const mine = boxes.filter(b => inBox(her, b)).sort((a, b) => area(a) - area(b))[0];
          if (mine) {
            if (yoloBox && boxIou(mine, yoloBox) < 0.3) faceFixes++;
            pick = mine;
            herBox = mine;
            learnHist(histOf(mine, px));
            frozen = false;
          }
        } else if (yoloBox) {
          const strangers = found.filter(fc => herFace.other(fc) && clearFace(fc, faceMin * 2) && inBox(fc, yoloBox));
          if (strangers.length && hideFor(strangers, i) && herBox) {
            // following someone else: back to where her face last was, and wait there
            yoloBox = [...herBox];
            yoloVel = [0, 0];
            frozen = true;
            faceFixes++;
          } else if (!strangers.length) doubted = false;
        }
      }
      if (!herHist && (pick || yoloBox)) learnHist(histOf(pick || yoloBox, px)); // her colours from the pick
      if (!pick && yoloBox && !frozen) {
        const guess = [yoloBox[0] + yoloVel[0], yoloBox[1] + yoloVel[1], yoloBox[2] + yoloVel[0], yoloBox[3] + yoloVel[1]];
        let bestScore = 0.25;
        for (const b of boxes) {
          const sc = boxIou(b, guess);
          if (sc > bestScore && (!herHist || alike(histOf(b, px), herHist) > 0.6)) { pick = b; bestScore = sc; }
        }
      }
      if (pick) {
        if (yoloBox) {
          const dx = (pick[0] + pick[2] - yoloBox[0] - yoloBox[2]) / 2, dy = (pick[1] + pick[3] - yoloBox[1] - yoloBox[3]) / 2;
          yoloVel = [yoloVel[0] * 0.5 + dx * 0.5, yoloVel[1] * 0.5 + dy * 0.5];
        }
        // ease the box itself (it jumps when an arm goes up or a leg is hidden)
        yoloBox = yoloBox && boxIou(pick, yoloBox) > 0.3 ? pick.map((v, k) => yoloBox[k] + (v - yoloBox[k]) * 0.6) : [...pick];
        yoloLost = 0;
      } else {
        yoloLost++;
        yoloVel = [yoloVel[0] * 0.8, yoloVel[1] * 0.8];
      }
      alpha.fill(0);
      // Lost or waiting: the circle stays where she was rather than going blank.
      const shown = !!yoloBox;
      if (shown) {
        const x0 = Math.floor(yoloBox[0] * aw), x1 = Math.ceil(yoloBox[2] * aw), y0 = Math.floor(yoloBox[1] * ah), y1 = Math.ceil(yoloBox[3] * ah);
        for (let y = Math.max(0, y0); y < Math.min(ah, y1); y++) alpha.fill(1, y * aw + Math.max(0, x0), y * aw + Math.min(aw, x1));
      }
      return shown;
    };

    // Circle mode: follow the middle of her cut-out (upper body weighs more than legs, so the
    // circle sits around her torso) with a radius fitted to her size, both eased so the window
    // glides instead of jittering. When she's not found for a moment it stays put, then fades.
    let cam = null, camAlpha = 0, lostFor = 0;
    let disk = null;
    const composeCircle = (f, found) => {
      let x0 = aw, y0 = ah, x1 = -1, y1 = -1, sx = 0, sy = 0, sw = 0;
      for (let y = 0; y < ah; y += 2) {
        for (let x = 0; x < aw; x += 2) {
          const a = alpha[y * aw + x];
          if (a > 0.5) {
            if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
            sx += x * a; sy += y * a; sw += a;
          }
        }
      }
      const seen = found && sw > 20;
      if (seen) {
        // In source pixels: the box's middle across, a bit above the middle down (her face and
        // torso, not her feet), and a radius that fits most of her.
        const bw = (x1 - x0) / aw * r.w, bh = (y1 - y0) / ah * r.h;
        const want = {
          x: (sx / sw) / aw * r.w,
          y: (y0 / ah + (y1 - y0) / ah * 0.42) * r.h,
          rad: Math.max(bw * 0.7, bh * 0.55, Math.min(r.w, r.h) * 0.12),
        };
        want.rad = Math.min(want.rad, Math.min(r.w, r.h) / 2);
        if (!cam) cam = { ...want };
        else {
          cam.x += (want.x - cam.x) * 0.2;
          cam.y += (want.y - cam.y) * 0.2;
          cam.rad += (want.rad - cam.rad) * 0.08;
        }
        lostFor = 0;
      } else lostFor++;
      camAlpha += ((cam && lostFor <= fps ? 1 : 0) - camAlpha) * 0.15;

      oCtx.fillStyle = '#000';
      oCtx.fillRect(0, 0, W, H);
      if (cam) {
        // Keep the window inside the picture (no black edge where it would run off it).
        const rad = cam.rad, d = rad * 2;
        const x = Math.min(Math.max(cam.x, rad), r.w - rad), y = Math.min(Math.max(cam.y, rad), r.h - rad);
        f.draw(oCtx, { x: r.x + x - rad, y: r.y + y - rad, w: d, h: d }, hw, hh);
      }
      if (!disk) {
        disk = document.createElement('canvas');
        disk.width = hw; disk.height = hh;
        const g = disk.getContext('2d');
        const grad = g.createRadialGradient(hw / 2, hh / 2, hw / 2 - Math.max(2, hw / 160), hw / 2, hh / 2, hw / 2);
        grad.addColorStop(0, '#fff');
        grad.addColorStop(1, '#000');
        g.fillStyle = '#000';
        g.fillRect(0, 0, hw, hh);
        g.fillStyle = grad;
        g.beginPath(); g.arc(hw / 2, hh / 2, hw / 2, 0, Math.PI * 2); g.fill();
        g.fillStyle = '#fff';
        g.beginPath(); g.arc(hw / 2, hh / 2, hw / 2 - Math.max(2, hw / 160), 0, Math.PI * 2); g.fill();
      }
      oCtx.globalAlpha = camAlpha;
      oCtx.drawImage(disk, alphaX, alphaY);
      oCtx.globalAlpha = 1;
    };
    let thumbnail = null, found = true;
    const t0 = performance.now();
    let total = Math.max(1, Math.floor((end - start) * fps));

    let encoded = 0;
    // Where the time goes (tests set globalThis.__fancamProf = {} to read it).
    const prof = globalThis.__fancamProf;
    let lap = performance.now();
    const mark = name => { if (!prof) return; const now = performance.now(); prof[name] = (prof[name] || 0) + now - lap; lap = now; };
    let failure = null;
    try {
      for await (const f of source.frames(start, end, fps, signal)) {
        mark('decode');
        check();
        if (encodeError) throw encodeError;
        const i = f.index;
        total = f.total;

        if (o.stop?.requested && encoded > 0) break; // 여기까지 저장
        if (gpuRestarting) {
          o.onProgress?.({ done: i, total, elapsed: 0, eta: 0, stage: 'GPU가 초기화돼서 다시 준비하는 중' });
          await gpuRestarting;
          check();
        }
        f.draw(aCtx, r, aw, ah);
        const px = aCtx.getImageData(0, 0, aw, ah).data;
        await breathe(); check();
        mark('read');

        if (yoloTrack) {
          found = await trackWithYolo(f, i, px);
          mark('yolo');
        }
        let samStarted = false;
        if (!sam && samRestart && i % samEvery === 0) {
          // The GPU models are back after a restart: start tracking her again, from her face if it
          // was seen in the last half second, else from the person the pose tracker follows.
          samRestart = false;
          const seen = lastFace?.her && i - lastFace.i <= fps / 2 ? lastFace : null;
          const prompt = seen ? await promptFromFace(seen.her, seen.found).catch(() => null)
            : tracker.last ? samPrompt({ target: tracker.last, others: [] }) : null;
          if (prompt) {
            try {
              f.draw(sCtx, r, sam2.SIZE, sam2.SIZE);
              const out = await samTracker.start(sam2.toInput(sCtx.getImageData(0, 0, sam2.SIZE, sam2.SIZE).data), ...prompt);
              samSeen = out.score > 0;
              samRegion = samSeen ? toRegion(out.low, samEvery) : null;
              samSince = 0;
              sam = samTracker;
              samStarted = true;
              faceHidden = false;
            } catch (err) {
              if (err.name === 'AbortError') throw err;
              console.warn('SAM 2 restart failed:', err);
              restartGpu(err);
            }
          } else samRestart = true; // nobody to start from yet: try again on the next SAM frame
        }
        if (sam && !samStarted) {
          try {
            if (i % samEvery === 0) {
              f.draw(sCtx, r, sam2.SIZE, sam2.SIZE);
              const input = sam2.toInput(sCtx.getImageData(0, 0, sam2.SIZE, sam2.SIZE).data);
              if (i === 0) {
                o.onProgress?.({ done: 0, total, elapsed: 0, eta: 0, stage: 'GPU 확인 중 (처음 한 번, 30초쯤)' });
                if (!(await gpu.call('samTrusted', [input, ...samPrompt(o)], { timeout: 180000 }))) throw new Error('GPU result differs from CPU');
              }
              mark('sam-input');
              const out = i === 0 ? await sam.start(input, ...samPrompt(o)) : await sam.next(input);
              mark('sam');
              if (i === 0 && out.score <= 0 && !faceModels) throw new Error('SAM 2 could not find her on the first frame');
              samSeen = out.score > 0;
              samRegion = samSeen ? toRegion(out.low, samEvery) : null;
              samSince = 0;

              if (faceModels && i % faceEvery === 0) {
                const { found, her } = await lookForFaces(f);
                lastFace = { i, found, her };
                check();
                mark('face');
                const mask = samRegion?.core;
                const strangers = found.filter(fc => fc !== her && herFace.other(fc) && clearFace(fc, faceMin * 2) && faceIn(fc, mask) > 0.6);
                if (her) {
                  faceHidden = false;
                  sawHer(her, i);
                  // A face right next to hers (someone just behind her) can't be cut away cleanly; ignore it.
                  const apart = fc => Math.hypot((fc.box[0] + fc.box[2] - her.box[0] - her.box[2]) / 2,
                    (fc.box[1] + fc.box[3] - her.box[1] - her.box[3]) / 2) > 1.5 * (her.box[2] - her.box[0]);
                  // Off her face once can be a turn of the head; twice in a row, or someone else's
                  // face in the mask, and the tracker has drifted.
                  offFace = faceIn(her, mask) < 0.6 ? offFace + 1 : 0;
                  if (offFace >= 2 || strangers.some(apart)) {
                    offFace = 0;
                    // Drifted (or spread onto someone else): start again from her face.
                    sam = samTracker;
                    const again = await sam.start(input, ...(await promptFromFace(her, found)));
                    samSeen = again.score > 0;
                    samRegion = samSeen ? toRegion(again.low, samEvery) : null;
                    faceFixes++;
                  }
                } else if (strangers.length) {
                  // Someone else's face, clearly, and no sign of hers: don't show them.
                  if (hideFor(strangers, i)) faceHidden = true;
                } else doubted = false;
                if (globalThis.__fancamDebug) {
                  const core = samRegion?.core, box = [1, 1, 0, 0];
                  if (core) for (let k = 0; k < n; k += 3) if (core[k] > 0.5) { const x = (k % aw) / aw, y = Math.floor(k / aw) / ah; box[0] = Math.min(box[0], x); box[1] = Math.min(box[1], y); box[2] = Math.max(box[2], x); box[3] = Math.max(box[3], y); }
                  globalThis.__fancamDebug.push({ i, faces: found.map(fc => ({ sim: +fc.sim.toFixed(3), size: Math.round(fc.size), x: +((fc.box[0] + fc.box[2]) / 2 / r.w).toFixed(2), in: faceIn(fc, samRegion?.core), her: fc === her })), core: box.map(v => +v.toFixed(2)), fixes: faceFixes, hidden: faceHidden });
                }
              }
            } else samSince++;
          } catch (err) {
            if (err.name === 'AbortError') throw err;
            console.warn('SAM 2 failed, continuing with the pose tracker:', err);
            sam = null;
            restartGpu(err);
          }
        }

        if (sam) {
          found = samSeen && !faceHidden;
          let pha = null;
          if (matter) {
          try { pha = await matter.run(px); } catch (err) { console.warn('GPU matting failed:', err); matter = null; restartGpu(err); }
          }
          mark('matting');
          if (faceHidden || !samRegion) {
            alpha.fill(0); // not her (or SAM says she isn't in view)
          } else if (pha) {
            const inner = samRegion.inner;
            if (globalThis.__FANCAM_ALPHA_OLD__) {
              for (let k = 0; k < n; k++) { const v = Math.max(pha[k] * samRegion[k], inner[k]); alpha[k] = v < 0.02 ? 0 : v; }
            } else {
              // The matte inside her area, with the faint haze around her (stage light, blur the
              // model half-keeps) cut. Where the picture didn't change since
              // the last frame, lean on the last matte so still edges don't shimmer; where it did,
              // take the new one (no trails behind a moving arm).
              const area = samRegion.area;
              for (let k = 0, j = 0; k < n; k++, j += 4) {
                let v = Math.max(pha[k] * area[k], inner[k]);
                // Below 10% is haze; hair strands (30-70%) stay nearly as they were.
                v = (v - 0.1) / 0.85;
                // (also catches NaN: a bad value would otherwise be carried into every later frame)
                v = v > 0 ? (v >= 1 ? 1 : v) : 0;
                const g = 0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2];
                if (prevAlpha && prevGray) {
                  // Moved if the picture changed, or if the matte changed a lot (dark hair moving
                  // over a dark stage barely changes the picture).
                  const pa = prevAlpha[k];
                  const d = Math.max(Math.abs(g - prevGray[k]) / 12, Math.abs(v - pa) / 0.25);
                  const w = d < 0.4 ? 0.4 : d > 1 ? 1 : d;
                  v = pa + w * (v - pa);
                }
                alpha[k] = v < 0.02 ? 0 : v;
              }
              dropStrays(alpha);
            }
          } else {
            // No matting model: snap SAM's soft mask to colour edges instead.
            for (let k = 0, j = 0; k < n; k++, j += 4) guide[k] = (0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2]) / 255;
            guided(guide, samRegion.core, radius, 0.004, refined);
            for (let k = 0; k < n; k++) { const v = (refined[k] - 0.3) / 0.4; alpha[k] = v < 0 ? 0 : v > 1 ? 1 : v; }
          }
        } else if (!yoloTrack) {
          const persons = await detect(aCanvas);
          await breathe(); check();
          let person = persons.length && persons[0].w === aw ? tracker.update(persons, px) : (tracker.lost++, null);
          if (faceModels && i % faceEvery === 0 && persons.length && persons[0].w === aw) {
            // Her face says who she is: follow that person, and drop someone who clearly isn't her.
            const { found, her } = await lookForFaces(f);
            lastFace = { i, found, her };
            check();
            const byFace = her && await personAtFace(aCanvas, persons, her, r.w, r.h);
            if (byFace) {
              faceHidden = false;
              sawHer(her, i);
              if (byFace !== person) {
                person = byFace;
                tracker.last = byFace; tracker.lost = 0; tracker.vx = tracker.vy = 0;
                tracker.ref = colourHist(byFace, px);
                faceFixes++;
              }
            } else {
              const strangers = person ? found.filter(fc => herFace.other(fc) && clearFace(fc, faceMin * 2) && personForFace([person], fc, r.w, r.h)) : [];
              if (strangers.length && hideFor(strangers, i)) faceHidden = true;
              else if (!strangers.length) doubted = false;
            }
            globalThis.__fancamDebug?.push({ i, faces: found.map(fc => ({ sim: +fc.sim.toFixed(3), size: Math.round(fc.size), her: fc === her })), byFace: !!byFace, fixes: faceFixes, hidden: faceHidden });
          }
          if (faceHidden) person = null;
          const others = person ? persons.filter(q => q !== person && q.w === aw && q.h === ah) : [];
          found = !!person;
          globalThis.__fancamDebug?.push({ i, n: persons.length, lost: tracker.lost,
            people: persons.map(q => ({ cx: +q.cx.toFixed(3), cy: +q.cy.toFixed(3), area: +q.area.toFixed(3), me: q === person,
              lm: q.landmarks.map(l => [+l.x.toFixed(3), +l.y.toFixed(3), +(l.visibility ?? 1).toFixed(2)]) })) });

          let pha = null;
          if (matter) {
            try { pha = await matter.run(px); } catch (err) {
              console.warn('GPU matting failed, continuing with the basic cut-out:', err);
              matter = null;
              restartGpu(err);
            }
          }
          if (pha && person) {
            // Safety net: if the GPU's matte keeps coming out empty where the pose model clearly sees
            // her, the GPU isn't computing it right. Switch to the basic cut-out for the rest.
            let sum = 0, cnt = 0;
            for (let k = 0; k < n; k += 7) if (person.mask[k] > 0.8) { sum += pha[k]; cnt++; }
            badMatte = cnt > 50 && sum / cnt < 0.2 ? badMatte + 1 : 0;
            if (badMatte >= 8) { console.warn('GPU matte looks wrong, switching to the basic cut-out'); matter = null; pha = null; }
          }
          if (pha) {
            // GPU matting for the edges, limited to her region.
            let region = null;
            if (person) region = lastRegion = buildRegion(person, others);
            else if (lastRegion && tracker.lost <= fps) region = lastRegion; // briefly missed: keep her area
            for (let k = 0; k < n; k++) {
              const v = region ? pha[k] * region[k] : 0;
              alpha[k] = v < 0.02 ? 0 : v;
            }
          } else if (person) {
            for (let k = 0, j = 0; k < n; k++, j += 4) {
              guide[k] = (0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2]) / 255;
              // Where another member overlaps her, give the pixel to whoever the model is surer of.
              let other = 0;
              for (const q of others) if (q.mask[k] > other) other = q.mask[k];
              raw[k] = other > person.mask[k] ? Math.max(0, person.mask[k] - other) : person.mask[k];
            }
            const region = buildRegion(person, others);
            for (let k = 0; k < n; k++) raw[k] *= region[k];
            guided(guide, raw, radius, 0.004, refined);
            for (let k = 0; k < n; k++) {
              // Tighten the soft ramp: the model's masks run a little wide, which shows as a halo.
              let v = (refined[k] - 0.32) / 0.42;
              v = v < 0 ? 0 : v > 1 ? 1 : v;
              alpha[k] = prevAlpha ? v * 0.85 + prevAlpha[k] * 0.15 : v; // calm frame-to-frame flicker
            }
          } else {
            // Missed her this frame: hold the last cut-out briefly, then fade, rather than show someone else.
            const keep = tracker.lost <= Math.round(fps / 3) ? 1 : 0.6;
            for (let k = 0; k < n; k++) alpha[k] = prevAlpha ? prevAlpha[k] * keep : 0;
          }
        }

        if (circle) {
          composeCircle(f, found);
          mark('compose');
        } else {
        // Track the overall bounding box so the player can crop empty space.
        for (let y = 0; y < ah; y += 2) {
          for (let x = 0; x < aw; x += 2) {
            if (alpha[y * aw + x] > 0.5) {
              const nx = x / aw, ny = y / ah;
              if (nx < bounds[0]) bounds[0] = nx; if (ny < bounds[1]) bounds[1] = ny;
              if (nx > bounds[2]) bounds[2] = nx; if (ny > bounds[3]) bounds[3] = ny;
            }
          }
        }

        mark('alpha');
        const md = maskImage.data;
        for (let k = 0, j = 0; k < n; k++, j += 4) {
          const v = alpha[k] * 255;
          md[j] = md[j + 1] = md[j + 2] = v; md[j + 3] = 255;
        }
        mCtx.putImageData(maskImage, 0, 0);
        f.draw(oCtx, r, hw, hh);
        oCtx.drawImage(mCanvas, alphaX, alphaY, hw, hh);

        mark('compose');
        }
        const frame = new VideoFrame(out, { timestamp: Math.round(i * 1e6 / fps), duration: Math.round(1e6 / fps) });
        encoder.encode(frame, { keyFrame: i % fps === 0 });
        encoded++;
        addAudioUntil((i + 1) * 1e6 / fps);
        frame.close();
        while (encoder.encodeQueueSize > 3) { check(); await new Promise(res => setTimeout(res, 5)); }
        mark('encode');

        if (!thumbnail && i >= Math.min(total - 1, Math.floor(total * 0.15))) {
          thumbnail = await makeThumbnail(out, layout, hw, hh);
        }

        [alpha, prevAlpha] = [prevAlpha ?? new Float32Array(n), alpha];
        prevGray ??= new Float32Array(n);
        for (let k = 0, j = 0; k < n; k++, j += 4) prevGray[k] = 0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2];
        mark('rest');
        const elapsed = (performance.now() - t0) / 1000;
        o.onProgress?.({
          done: i + 1, total, elapsed, eta: elapsed / (i + 1) * (total - i - 1),
          preview: out, layout, found, matting: !!matter, sam: !!sam, yolo: yoloTrack, circle: o.shape === 'circle', face: !!faceModels, fixes: faceFixes, decoder: source.kind,
        });
      }
    } catch (err) {
      // The GPU was reset, a model crashed...: keep what's made so far rather than lose it all.
      if (err.name === 'AbortError' || !encoded) throw err;
      console.warn('stopped early, keeping what was made:', err);
      failure = err;
    }
    check();
    if (!encoded) throw new Error('영상에서 화면을 읽지 못했어요. 구간을 다시 정하거나 다른 영상으로 해 주세요.');

    await encoder.flush();
    encoder.close();
    if (encodeError) throw encodeError;
    addAudioUntil(encoded * 1e6 / fps);
    muxer.finalize();

    parts.sort((a, b) => a.position - b.position);
    const blob = new Blob(parts.map(p => p.data), { type: 'video/mp4' });

    // Pad the crop a little and fall back to the full frame if she was never found.
    const pad = 0.03;
    const crop = circle ? { x: 0, y: 0, w: 1, h: 1 } : bounds[2] > bounds[0]
      ? { x: Math.max(0, bounds[0] - pad), y: Math.max(0, bounds[1] - pad),
          w: Math.min(1, bounds[2] + pad + 2 / aw) - Math.max(0, bounds[0] - pad),
          h: Math.min(1, bounds[3] + pad + 2 / ah) - Math.max(0, bounds[1] - pad) }
      : { x: 0, y: 0, w: 1, h: 1 };

    return {
      blob, thumbnail, layout, crop, width: hw, height: hh, fps,
      duration: encoded / fps, hasAudio, codec: enc.config.codec, partial: encoded < total,
      error: failure?.message,
    };
  } finally {
    for (const c of cleanup.reverse()) { try { c(); } catch {} }
  }
}

// A transparent PNG of the current packed frame (colour half + alpha half).
async function makeThumbnail(packed, layout, hw, hh) {
  const s = Math.min(1, 360 / Math.max(hw, hh));
  const w = Math.round(hw * s), h = Math.round(hh * s);
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const ax = layout === 'sbs' ? hw : 0, ay = layout === 'sbs' ? 0 : hh;
  ctx.drawImage(packed, ax, ay, hw, hh, 0, 0, w, h);
  const a = ctx.getImageData(0, 0, w, h);
  ctx.drawImage(packed, 0, 0, hw, hh, 0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h);
  for (let j = 0; j < img.data.length; j += 4) img.data[j + 3] = a.data[j];
  ctx.putImageData(img, 0, 0);
  return new Promise(res => c.toBlob(res, 'image/png'));
}
