// Extracts one person from a video and encodes the result as an "alpha-packed" MP4:
// colour on one half of each frame, the alpha matte (greyscale) on the other half.
// Packed MP4 plays on every phone/tablet/desktop browser; js/player.js recombines
// the halves into a transparent image with a tiny WebGL shader.
import { PoseLandmarker } from '../vendor/mediapipe/vision_bundle.mjs';
import { Muxer, StreamTarget } from '../vendor/mp4-muxer/mp4-muxer.mjs';
import { openSource, seekVideo } from './source.js';
import * as matting from './matting.js';
import * as sam2 from './sam2.js';

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
      numPoses: 5,
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

// The first frame's prompt: where she was tapped (or her middle), a box around her whole body,
// and "not her" points on the other people found next to her.
function samPrompt(o) {
  const S = sam2.SIZE, t = o.target;
  const pts = [], labels = [];
  const [tx, ty] = o.tap ?? [t.cx, t.cy];
  pts.push([tx * S, ty * S]); labels.push(1);
  let box = null;
  if (t.bbox) {
    const [x0, y0, x1, y1] = t.bbox, pad = 0.02;
    box = [Math.max(0, x0 - pad) * S, Math.max(0, y0 - pad) * S, Math.min(1, x1 + pad) * S, Math.min(1, y1 + pad) * S];
  }
  for (const p of o.others || []) {
    if (p === t || !p.area) continue;
    const inside = t.bbox && p.cx > t.bbox[0] && p.cx < t.bbox[2] && p.cy > t.bbox[1] && p.cy < t.bbox[3];
    if (!inside) { pts.push([p.cx * S, p.cy * S]); labels.push(0); }
  }
  return [pts, labels, box];
}

// SAM's 256x256 mask (of the square-squashed frame) -> a soft region at analysis size, grown a
// little (more when SAM only runs every few frames, to cover her movement in between).
function makeSamRegion(w, h) {
  const n = w * h, core = new Float32Array(n), tmp = new Float32Array(n), grown = new Float32Array(n);
  return (low, every) => {
    for (let y = 0; y < h; y++) {
      const fy = Math.max(0, (y + 0.5) / h * 256 - 0.5), y0 = Math.floor(fy), y1 = Math.min(255, y0 + 1), wy = fy - y0;
      for (let x = 0; x < w; x++) {
        const fx = Math.max(0, (x + 0.5) / w * 256 - 0.5), x0 = Math.floor(fx), x1 = Math.min(255, x0 + 1), wx = fx - x0;
        const v = (low[y0 * 256 + x0] * (1 - wx) + low[y0 * 256 + x1] * wx) * (1 - wy)
          + (low[y1 * 256 + x0] * (1 - wx) + low[y1 * 256 + x1] * wx) * wy;
        core[y * w + x] = 1 / (1 + Math.exp(-v));
      }
    }
    const r = Math.max(2, Math.round(Math.max(w, h) * (0.012 + 0.006 * (every - 1))));
    boxMean(core, w, h, r, grown, tmp);
    const region = new Float32Array(n);
    for (let k = 0; k < n; k++) region[k] = Math.min(1, Math.max(core[k], grown[k] * 2.5));
    region.core = core.slice();
    return region;
  };
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
 * @param {(p:object)=>void} o.onProgress
 * @param {AbortSignal} o.signal
 */
export async function extract(o) {
  const { video, start, end, fps, signal } = o;
  const check = () => { if (signal?.aborted) throw abortError(); };

  const r = o.rect ?? fullRect(video);
  const [aw, ah] = analysisSize(r, o.maxSide);
  const [hw, hh] = fitSize(r.w, r.h, o.maxSide);
  // Portrait fancams pack side-by-side, landscape ones top/bottom, keeping frames squarish.
  const layout = hh >= hw ? 'sbs' : 'tb';
  const W = layout === 'sbs' ? hw * 2 : hw;
  const H = layout === 'sbs' ? hh : hh * 2;
  const alphaX = layout === 'sbs' ? hw : 0, alphaY = layout === 'sbs' ? 0 : hh;

  o.onProgress?.({ done: 0, total: 1, elapsed: 0, eta: 0, stage: '준비 중' });
  let [source, matter, samModels] = await Promise.all([
    openSource(o.file, video),
    o.useMatting === false ? null : matting.createMatter(aw, ah).catch(err => { console.warn('matting off:', err); return null; }),
    o.useSam === false ? null : sam2.load(f => o.onProgress?.({ done: 0, total: 1, elapsed: 0, eta: 0, stage: `SAM 2 추적 모델 받는 중 ${Math.round(f * 100)}%` }))
      .catch(err => { console.warn('SAM 2 off:', err); return null; }),
  ]);
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
    const guided = makeGuidedFilter(aw, ah);
    const buildRegion = makeRegionBuilder(aw, ah);
    const radius = Math.max(2, Math.round(Math.max(aw, ah) / 110));
    const tracker = new Tracker(o.target, fps);
    let lastRegion = null, badMatte = 0;

    // SAM 2 follows her every `samEvery` frames; the matting model draws the edges every frame.
    let sam = samModels ? new sam2.Tracker(samModels) : null;
    const samEvery = Math.max(1, Math.round(fps / 15));
    const sCanvas = sam ? mk(sam2.SIZE, sam2.SIZE) : null;
    const sCtx = sCanvas?.getContext('2d', { willReadFrequently: true });
    const toRegion = sam ? makeSamRegion(aw, ah) : null;
    let samRegion = null, samSeen = true, samSince = 0;

    const bounds = [1, 1, 0, 0];
    let thumbnail = null, found = true;
    const t0 = performance.now();
    let total = Math.max(1, Math.floor((end - start) * fps));

    let encoded = 0;
    for await (const f of source.frames(start, end, fps, signal)) {
      check();
      if (encodeError) throw encodeError;
      const i = f.index;
      total = f.total;

      if (o.stop?.requested && encoded > 0) break; // 여기까지 저장
      f.draw(aCtx, r, aw, ah);
      const px = aCtx.getImageData(0, 0, aw, ah).data;
      await breathe(); check();

      if (sam) {
        try {
          if (i % samEvery === 0) {
            f.draw(sCtx, r, sam2.SIZE, sam2.SIZE);
            const input = sam2.toInput(sCtx.getImageData(0, 0, sam2.SIZE, sam2.SIZE).data);
            if (i === 0) {
              o.onProgress?.({ done: 0, total, elapsed: 0, eta: 0, stage: 'GPU 확인 중 (처음 한 번, 30초쯤)' });
              if (!(await sam2.gpuTrusted(samModels, input, ...samPrompt(o)))) throw new Error('GPU result differs from CPU');
            }
            const out = i === 0 ? await sam.start(input, ...samPrompt(o)) : await sam.next(input);
            if (i === 0 && out.score <= 0) throw new Error('SAM 2 could not find her on the first frame');
            samSeen = out.score > 0;
            samRegion = samSeen ? toRegion(out.low, samEvery) : null;
            samSince = 0;
          } else samSince++;
        } catch (err) {
          if (err.name === 'AbortError') throw err;
          console.warn('SAM 2 failed, continuing with the pose tracker:', err);
          sam = null;
        }
      }

      if (sam) {
        found = samSeen;
        let pha = null;
        if (matter) {
          try { pha = await matter.run(px); } catch (err) { console.warn('GPU matting failed:', err); matter = null; }
        }
        if (pha && samRegion) {
          for (let k = 0; k < n; k++) { const v = pha[k] * samRegion[k]; alpha[k] = v < 0.02 ? 0 : v; }
        } else if (samRegion) {
          // No matting model: snap SAM's soft mask to colour edges instead.
          for (let k = 0, j = 0; k < n; k++, j += 4) guide[k] = (0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2]) / 255;
          guided(guide, samRegion.core, radius, 0.004, refined);
          for (let k = 0; k < n; k++) { const v = (refined[k] - 0.3) / 0.4; alpha[k] = v < 0 ? 0 : v > 1 ? 1 : v; }
        } else {
          alpha.fill(0); // SAM says she isn't in view
        }
      } else {
        const persons = await detect(aCanvas);
        await breathe(); check();
        const person = persons.length && persons[0].w === aw ? tracker.update(persons, px) : (tracker.lost++, null);
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

      const md = maskImage.data;
      for (let k = 0, j = 0; k < n; k++, j += 4) {
        const v = alpha[k] * 255;
        md[j] = md[j + 1] = md[j + 2] = v; md[j + 3] = 255;
      }
      mCtx.putImageData(maskImage, 0, 0);
      f.draw(oCtx, r, hw, hh);
      oCtx.drawImage(mCanvas, alphaX, alphaY, hw, hh);

      const frame = new VideoFrame(out, { timestamp: Math.round(i * 1e6 / fps), duration: Math.round(1e6 / fps) });
      encoder.encode(frame, { keyFrame: i % fps === 0 });
      encoded++;
      addAudioUntil((i + 1) * 1e6 / fps);
      frame.close();
      while (encoder.encodeQueueSize > 3) { check(); await new Promise(res => setTimeout(res, 5)); }

      if (!thumbnail && i >= Math.min(total - 1, Math.floor(total * 0.15))) {
        thumbnail = await makeThumbnail(out, layout, hw, hh);
      }

      [alpha, prevAlpha] = [prevAlpha ?? new Float32Array(n), alpha];
      const elapsed = (performance.now() - t0) / 1000;
      o.onProgress?.({
        done: i + 1, total, elapsed, eta: elapsed / (i + 1) * (total - i - 1),
        preview: out, layout, found, matting: !!matter, sam: !!sam, decoder: source.kind,
      });
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
    const crop = bounds[2] > bounds[0]
      ? { x: Math.max(0, bounds[0] - pad), y: Math.max(0, bounds[1] - pad),
          w: Math.min(1, bounds[2] + pad + 2 / aw) - Math.max(0, bounds[0] - pad),
          h: Math.min(1, bounds[3] + pad + 2 / ah) - Math.max(0, bounds[1] - pad) }
      : { x: 0, y: 0, w: 1, h: 1 };

    return {
      blob, thumbnail, layout, crop, width: hw, height: hh, fps,
      duration: encoded / fps, hasAudio, codec: enc.config.codec, partial: encoded < total,
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
