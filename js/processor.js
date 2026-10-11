// Extracts one person from a video and encodes the result as an "alpha-packed" MP4:
// colour on one half of each frame, the alpha matte (greyscale) on the other half.
// Packed MP4 plays on every phone/tablet/desktop browser; js/player.js recombines
// the halves into a transparent image with a tiny WebGL shader.
import { PoseLandmarker } from '../vendor/mediapipe/vision_bundle.mjs';
import { Muxer, StreamTarget } from '../vendor/mp4-muxer/mp4-muxer.mjs';

const ANALYSIS_SIDE = 640; // longest side of the frame fed to the segmentation model
const GRID = 4;            // stride for the coarse masks used to track a person

// ---------------------------------------------------------------------------
// Model

let landmarkerPromise = null;
let delegate = 'GPU';
let lastTimestamp = 0;

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
      runningMode: 'VIDEO',
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

// VIDEO mode needs strictly increasing timestamps across every call.
function nextTimestamp() {
  lastTimestamp = Math.max(lastTimestamp + 1, Math.round(performance.now()));
  return lastTimestamp;
}

// Runs the model on `source` and returns every person found, each with its own mask.
function detectPersons(landmarker, source) {
  const persons = [];
  landmarker.detectForVideo(source, nextTimestamp(), result => {
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

// Follows the chosen person from one frame to the next.
function track(persons, prev) {
  let best = null, bestScore = 0.12;
  for (const p of persons) {
    const s = iou(p, prev);
    if (s > bestScore) { bestScore = s; best = p; }
  }
  if (best) return best;
  // Fast moves can break overlap: fall back to a nearby person of similar size.
  let bestDist = 0.18;
  for (const p of persons) {
    const d = Math.hypot(p.cx - prev.cx, p.cy - prev.cy);
    const sizeRatio = p.area / Math.max(prev.area, 1e-6);
    if (d < bestDist && sizeRatio > 0.4 && sizeRatio < 2.5) { bestDist = d; best = p; }
  }
  return best;
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

export function seek(video, time) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { cleanup(); reject(new Error('영상 탐색 시간이 초과됐어요.')); }, 10000);
    const onSeeked = () => { cleanup(); resolve(); };
    const cleanup = () => { clearTimeout(timer); video.removeEventListener('seeked', onSeeked); };
    video.addEventListener('seeked', onSeeked);
    video.currentTime = time;
  });
}

function fitSize(w, h, maxSide) {
  const s = Math.min(1, maxSide / Math.max(w, h));
  return [Math.max(2, Math.round(w * s / 2) * 2), Math.max(2, Math.round(h * s / 2) * 2)];
}

// The part of the frame to use. Screen recordings carry black bars we crop away.
export function fullRect(video) {
  return { x: 0, y: 0, w: video.videoWidth, h: video.videoHeight };
}

export function analysisSize(rect) {
  return fitSize(rect.w, rect.h, ANALYSIS_SIDE);
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
    await seek(video, start + (end - start) * (i + 0.5) / samples);
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

async function encodeAudio(audio, muxer) {
  let failure = null;
  const encoder = new AudioEncoder({
    output: (chunk, meta) => muxer.addAudioChunk(chunk, meta),
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
// Main pipeline

/**
 * @param {object} o
 * @param {File} o.file          source video file
 * @param {HTMLVideoElement} o.video  a loaded <video> for `file`
 * @param {number} o.start, o.end     trim range in seconds
 * @param {{x,y,w,h}} [o.rect] part of the frame to use (video pixels); whole frame if omitted
 * @param {object} o.target       person picked on the first frame (from detectPersons)
 * @param {number} o.fps, o.maxSide, o.bitrate
 * @param {boolean} o.withAudio
 * @param {(p:object)=>void} o.onProgress
 * @param {AbortSignal} o.signal
 */
export async function extract(o) {
  const { video, start, end, fps, signal } = o;
  const r = o.rect ?? fullRect(video);
  const [aw, ah] = analysisSize(r);
  const [hw, hh] = fitSize(r.w, r.h, o.maxSide);
  // Portrait fancams pack side-by-side, landscape ones top/bottom, keeping frames squarish.
  const layout = hh >= hw ? 'sbs' : 'tb';
  const W = layout === 'sbs' ? hw * 2 : hw;
  const H = layout === 'sbs' ? hh : hh * 2;
  const alphaX = layout === 'sbs' ? hw : 0, alphaY = layout === 'sbs' ? 0 : hh;

  const enc = await pickVideoEncoder(W, H, fps, o.bitrate);
  const audio = o.withAudio ? await prepareAudio(o.file, start, end) : null;
  if (signal?.aborted) throw new DOMException('취소됨', 'AbortError');

  const parts = [];
  const muxer = new Muxer({
    target: new StreamTarget({ onData: (data, position) => parts.push({ data: data.slice(), position }), chunked: true }),
    video: { codec: enc.mux, width: W, height: H, frameRate: fps },
    audio: audio ? { codec: audio.mux, numberOfChannels: audio.channels, sampleRate: audio.config.sampleRate } : undefined,
    fastStart: 'fragmented',
    firstTimestampBehavior: 'offset',
  });

  if (audio) await encodeAudio(audio, muxer);

  let encodeError = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: e => { encodeError = e; },
  });
  encoder.configure(enc.config);

  // Canvases
  const mk = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
  const aCanvas = mk(aw, ah), aCtx = aCanvas.getContext('2d', { willReadFrequently: true });
  const mCanvas = mk(aw, ah), mCtx = mCanvas.getContext('2d');
  const out = mk(W, H), oCtx = out.getContext('2d', { alpha: false });
  oCtx.imageSmoothingEnabled = true;
  oCtx.imageSmoothingQuality = 'high';
  const maskImage = mCtx.createImageData(aw, ah);

  const n = aw * ah;
  const guide = new Float32Array(n), raw = new Float32Array(n), refined = new Float32Array(n);
  let alpha = new Float32Array(n), prevAlpha = null;
  const guided = makeGuidedFilter(aw, ah);
  const radius = Math.max(2, Math.round(Math.max(aw, ah) / 110));

  const total = Math.max(1, Math.floor((end - start) * fps));
  const bounds = [1, 1, 0, 0];
  let prev = o.target, lostFrames = 0, thumbnail = null;
  const t0 = performance.now();

  for (let i = 0; i < total; i++) {
    if (signal?.aborted) { encoder.close(); throw new DOMException('취소됨', 'AbortError'); }
    if (encodeError) throw encodeError;

    // Aim for the middle of each output frame's time slot.
    await seek(video, Math.min(start + (i + 0.5) / fps, video.duration - 0.001));
    aCtx.drawImage(video, r.x, r.y, r.w, r.h, 0, 0, aw, ah);
    const persons = await detect(aCanvas);
    const person = prev ? track(persons, prev) : pickDefault(persons);

    if (person && person.w === aw && person.h === ah) {
      prev = person; lostFrames = 0;
      const px = aCtx.getImageData(0, 0, aw, ah).data;
      const others = persons.filter(q => q !== person && q.w === aw && q.h === ah).map(q => q.mask);
      for (let k = 0, j = 0; k < n; k++, j += 4) {
        guide[k] = (0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2]) / 255;
        // Where another member overlaps her, give the pixel to whoever the model is surer of.
        let other = 0;
        for (const m of others) if (m[k] > other) other = m[k];
        raw[k] = other > person.mask[k] ? Math.max(0, person.mask[k] - other) : person.mask[k];
      }
      guided(guide, raw, radius, 0.004, refined);
      for (let k = 0; k < n; k++) {
        // Tighten the soft ramp: the model's masks run a little wide, which shows as a halo.
        let v = (refined[k] - 0.32) / 0.42;
        v = v < 0 ? 0 : v > 1 ? 1 : v;
        alpha[k] = prevAlpha ? v * 0.85 + prevAlpha[k] * 0.15 : v; // calm frame-to-frame flicker
      }
    } else {
      // Lost her this frame: fade out rather than risk showing someone else.
      if (++lostFrames > fps) prev = null; // after ~1s, re-acquire the most central person
      for (let k = 0; k < n; k++) alpha[k] = prevAlpha ? prevAlpha[k] * 0.5 : 0;
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
    oCtx.drawImage(video, r.x, r.y, r.w, r.h, 0, 0, hw, hh);
    oCtx.drawImage(mCanvas, alphaX, alphaY, hw, hh);

    const frame = new VideoFrame(out, { timestamp: Math.round(i * 1e6 / fps), duration: Math.round(1e6 / fps) });
    encoder.encode(frame, { keyFrame: i % fps === 0 });
    frame.close();
    while (encoder.encodeQueueSize > 3) await new Promise(r => setTimeout(r, 5));

    if (!thumbnail && i >= Math.min(total - 1, Math.floor(total * 0.15))) {
      thumbnail = await makeThumbnail(video, r, mCanvas, hw, hh);
    }

    [alpha, prevAlpha] = [prevAlpha ?? new Float32Array(n), alpha];
    const elapsed = (performance.now() - t0) / 1000;
    o.onProgress?.({
      done: i + 1, total, elapsed, eta: elapsed / (i + 1) * (total - i - 1),
      preview: out, layout, found: lostFrames === 0,
    });
  }

  await encoder.flush();
  encoder.close();
  if (encodeError) throw encodeError;
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
    duration: total / fps, hasAudio: !!audio, codec: enc.config.codec,
  };
}

async function makeThumbnail(video, r, maskCanvas, hw, hh) {
  const s = Math.min(1, 360 / Math.max(hw, hh));
  const w = Math.round(hw * s), h = Math.round(hh * s);
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(maskCanvas, 0, 0, w, h);
  const a = ctx.getImageData(0, 0, w, h);
  ctx.drawImage(video, r.x, r.y, r.w, r.h, 0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h);
  for (let j = 0; j < img.data.length; j += 4) img.data[j + 3] = a.data[j];
  ctx.putImageData(img, 0, 0);
  return new Promise(r => c.toBlob(r, 'image/png'));
}
