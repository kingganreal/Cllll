// SAM 2.1 (hiera-tiny) video object tracking on the GPU via onnxruntime-web (WebGPU).
// You tap her once; SAM 2 keeps a memory of what she looks like (her last few masks and an
// "object pointer" per frame) and finds that same person in every following frame, even
// when other members stand next to her or cross in front.
//
// This is a port of the streaming video predictor in axinc-ai/ailia-models
// (image_segmentation/segment-anything-2, MIT) which runs Meta's SAM 2 (Apache-2.0) as ONNX.
// Only what single-object, forward-only tracking needs is kept.
// Models: models/sam2/*.onnx — simplified (onnxsim) so onnxruntime-web accepts them, with
// float16-stored weights that are cast back to float32 when the model loads.

const ORT = new URL('../vendor/onnxruntime/ort.webgpu.bundle.min.mjs', import.meta.url).href;
const DIR = new URL('../models/sam2/', import.meta.url).href;
const FILES = {
  encoder: 'image_encoder.onnx', prompt: 'prompt_encoder.onnx', decoder: 'mask_decoder.onnx',
  memAttn: 'memory_attention.onnx', memEnc: 'memory_encoder.onnx', mlp: 'mlp.onnx', tpos: 'obj_ptr_tpos_proj.onnx',
};

export const SIZE = 1024;           // SAM 2 input size
const NUM_MASKMEM = 7;              // 1 conditioning (tapped) frame + 6 previous frames
const MAX_OBJ_PTRS = 16;
const HW = 64 * 64;                 // tokens in the 64x64 feature map
const C = 256, MEM = 64;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

let loadPromise = null;

/** Loads all SAM 2 parts (about 100 MB the first time; the browser caches them after). */
export function load(onProgress) {
  loadPromise ??= (async () => {
    const ep = globalThis.__FANCAM_SAM_EP__ || 'webgpu'; // tests can force 'wasm'
    if (ep === 'webgpu' && (!navigator.gpu || !(await navigator.gpu.requestAdapter()))) throw new Error('WebGPU를 쓸 수 없어요.');
    const ort = await import(ORT);
    ort.env.wasm.wasmPaths = new URL('../vendor/onnxruntime/', import.meta.url).href;
    ort.env.wasm.numThreads = 1;
    const consts = await (await fetch(DIR + 'constants.json')).json();
    const sessions = {}, buffers = {};
    const names = Object.keys(FILES);
    let done = 0;
    for (const k of names) {
      buffers[k] = await fetchWithProgress(DIR + FILES[k], f => onProgress?.((done + f) / names.length));
      sessions[k] = await ort.InferenceSession.create(buffers[k], { executionProviders: [ep], graphOptimizationLevel: 'all' });
      done++;
      onProgress?.(done / names.length);
    }
    return { ort, sessions, consts, buffers, ep };
  })().catch(err => { loadPromise = null; throw err; });
  return loadPromise;
}

async function fetchWithProgress(url, onFrac) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`SAM 2 모델을 받지 못했어요 (${res.status})`);
  const total = Number(res.headers.get('Content-Length')) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const out = new Uint8Array(total);
  const reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got);
    got += value.byteLength;
    onFrac(got / total);
  }
  return out;
}

// GPUs differ, and onnxruntime's WebGPU kernels have been wrong before on some shapes (see
// matting.js). So the first time on a given GPU, run two steps of tracking on both the GPU and
// the CPU and only trust the GPU if they agree. The answer is remembered per GPU.
const TRUST_KEY = 'fancam-sam2-gpu-trust';

async function gpuId() {
  try {
    const a = await navigator.gpu.requestAdapter();
    const i = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {});
    return [i.vendor, i.architecture, i.device, i.description, 'ort1.30'].join('|');
  } catch { return 'unknown'; }
}

/** True when the GPU computes SAM 2 like the CPU does. `input` is one frame from toInput(). */
export async function gpuTrusted(models, input, points, labels, box) {
  if (models.ep !== 'webgpu') return true;
  const id = await gpuId();
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(TRUST_KEY)) || {}; } catch {}
  if (id in saved) return saved[id];
  const { ort, consts, buffers } = models;
  const cpu = {};
  for (const k of Object.keys(buffers)) cpu[k] = await ort.InferenceSession.create(buffers[k], { executionProviders: ['wasm'] });
  const run = async sessions => {
    const t = new Tracker({ ort, sessions, consts });
    const a = await t.start(input, points, labels, box);
    const b = await t.next(input); // same picture again: exercises the memory path
    return [a.low, b.low];
  };
  const [g, c] = [await run(models.sessions), await run(cpu)];
  let ok = true;
  for (let f = 0; f < 2; f++) {
    let agree = 0, pos = 0;
    for (let k = 0; k < 65536; k++) { if ((g[f][k] > 0) === (c[f][k] > 0)) agree++; if (c[f][k] > 0) pos++; }
    if (agree / 65536 < 0.98 || pos < 20) ok = false;
  }
  for (const sess of Object.values(cpu)) sess.release?.();
  saved[id] = ok;
  try { localStorage.setItem(TRUST_KEY, JSON.stringify(saved)); } catch {}
  if (!ok) console.warn('SAM 2 on this GPU does not match the CPU result; not using it.');
  return ok;
}

const data = async t => (t.location && t.location !== 'cpu' ? await t.getData() : t.data);

// (1, c, 64, 64) channel-first map -> (4096, 1, c) tokens
function toTokens(src, c) {
  const out = new Float32Array(HW * c);
  for (let ch = 0; ch < c; ch++) {
    const base = ch * HW;
    for (let p = 0; p < HW; p++) out[p * c + ch] = src[base + p];
  }
  return out;
}
// (4096, 1, c) tokens -> (1, c, 64, 64)
function fromTokens(src, c) {
  const out = new Float32Array(HW * c);
  for (let p = 0; p < HW; p++) {
    const row = p * c;
    for (let ch = 0; ch < c; ch++) out[ch * HW + p] = src[row + ch];
  }
  return out;
}

function sinePE(pos, dim) { // get_1d_sine_pe for one position
  const half = dim / 2, out = new Float32Array(dim);
  for (let i = 0; i < half; i++) {
    const t = Math.pow(10000, (2 * Math.floor(i / 2)) / half);
    out[i] = Math.sin(pos / t);
    out[half + i] = Math.cos(pos / t);
  }
  return out;
}

// Bilinear 256x256 -> 1024x1024 (align_corners=false), then shaped for the memory encoder.
function maskForMemory(low, binarize) {
  const out = new Float32Array(SIZE * SIZE);
  const s = 256 / SIZE;
  for (let y = 0; y < SIZE; y++) {
    const fy = Math.max(0, (y + 0.5) * s - 0.5), y0 = Math.floor(fy), y1 = Math.min(255, y0 + 1), wy = fy - y0;
    for (let x = 0; x < SIZE; x++) {
      const fx = Math.max(0, (x + 0.5) * s - 0.5), x0 = Math.floor(fx), x1 = Math.min(255, x0 + 1), wx = fx - x0;
      const v = (low[y0 * 256 + x0] * (1 - wx) + low[y0 * 256 + x1] * wx) * (1 - wy)
        + (low[y1 * 256 + x0] * (1 - wx) + low[y1 * 256 + x1] * wx) * wy;
      const m = binarize ? (v > 0 ? 1 : 0) : 1 / (1 + Math.exp(-v));
      out[y * SIZE + x] = m * 20 - 10; // sigmoid_scale_for_mem_enc / sigmoid_bias_for_mem_enc
    }
  }
  return out;
}

/** Turns a 1024x1024 RGBA frame into the encoder's normalised CHW input. */
export function toInput(rgba) {
  const n = SIZE * SIZE, x = new Float32Array(3 * n);
  for (let k = 0, j = 0; k < n; k++, j += 4) {
    x[k] = (rgba[j] / 255 - MEAN[0]) / STD[0];
    x[n + k] = (rgba[j + 1] / 255 - MEAN[1]) / STD[1];
    x[2 * n + k] = (rgba[j + 2] / 255 - MEAN[2]) / STD[2];
  }
  return x;
}

export class Tracker {
  constructor({ ort, sessions, consts }) {
    this.ort = ort;
    this.s = sessions;
    this.k = consts; // maskmem_tpos_enc [7][64], no_mem_embed [256], no_obj_ptr [256]
    this.cond = null;           // the tapped frame's memory
    this.recent = new Map();    // frame index -> { mem, memPos, ptr }
    this.frame = -1;
    this.currPos = null;        // positional encoding tokens (same for every frame)
    this.trackPrompt = null;    // prompt embeddings with no clicks (same for every frame)
  }

  T(type, arr, dims) { return new this.ort.Tensor(type, arr, dims); }

  async encode(input) {
    const o = await this.s.encoder.run({ input_image: this.T('float32', input, [1, 3, SIZE, SIZE]) });
    const f = {
      fpn0: o.backbone_fpn_0, fpn1: o.backbone_fpn_1,
      fpn2: await data(o.backbone_fpn_2),
    };
    if (!this.currPos) this.currPos = toTokens(await data(o.vision_pos_enc_2), C);
    return f;
  }

  async prompt(points, labels) {
    const n = Math.max(1, points.length);
    const coords = new Float32Array(n * 2), lab = new Int32Array(n).fill(-1);
    points.forEach(([x, y], i) => { coords[2 * i] = x; coords[2 * i + 1] = y; lab[i] = labels[i]; });
    const o = await this.s.prompt.run({
      coords: this.T('float32', coords, [1, n, 2]), labels: this.T('int32', lab, [1, n]),
      masks: this.T('float32', new Float32Array(256 * 256), [1, 256, 256]), masks_enable: this.T('int32', new Int32Array([0]), [1]),
    });
    return { sparse: o.sparse_embeddings, dense: o.dense_embeddings, pe: o.dense_pe };
  }

  // SAM mask head. With one click (or none, while tracking) it picks the best of the 3 candidate
  // masks; with more prompts (a box) it uses the single-mask output unless that one is unstable.
  async heads(pixFeat, f, prompt, multimask) {
    const o = await this.s.decoder.run({
      image_embeddings: this.T('float32', pixFeat, [1, C, 64, 64]), image_pe: prompt.pe,
      sparse_prompt_embeddings: prompt.sparse, dense_prompt_embeddings: prompt.dense,
      high_res_features1: f.fpn0, high_res_features2: f.fpn1,
    });
    const masks = await data(o.masks), iou = await data(o.iou_pred), tokens = await data(o.sam_tokens_out);
    const score = (await data(o.object_score_logits))[0];
    const bestOfThree = () => { let b = 1; for (let m = 2; m < 4; m++) if (iou[m] > iou[b]) b = m; return b; };
    let best, tokIdx;
    if (multimask) {
      best = bestOfThree();
      tokIdx = best;
    } else {
      // dynamic_multimask_via_stability
      let inner = 0, outer = 0;
      for (let k = 0; k < 65536; k++) { const v = masks[k]; if (v > 0.05) inner++; if (v > -0.05) outer++; }
      best = !outer || inner / outer >= 0.98 ? 0 : bestOfThree();
      tokIdx = 0;
    }
    const visible = score > 0;
    const low = visible ? masks.slice(best * 65536, (best + 1) * 65536) : new Float32Array(65536).fill(-1024);
    if (o.sam_tokens_out.dims[1] === 1) tokIdx = 0;
    const token = tokens.slice(tokIdx * C, (tokIdx + 1) * C);
    let ptr = await data((await this.s.mlp.run({ x: this.T('float32', token, [1, C]) })).x_out);
    if (!visible) ptr = Float32Array.from(this.k.no_obj_ptr);
    return { low, ptr: Float32Array.from(ptr), score, iou: iou[best] };
  }

  async memorise(f, low, fromClick) {
    const o = await this.s.memEnc.run({
      pix_feat: this.T('float32', f.fpn2, [1, C, 64, 64]),
      masks: this.T('float32', maskForMemory(low, fromClick), [1, 1, SIZE, SIZE]),
    });
    return { mem: toTokens(await data(o.vision_features), MEM), memPos: toTokens(await data(o.vision_pos_enc), MEM) };
  }

  /**
   * First frame. `points` in 1024x1024 space with `labels` (1 = her, 0 = not her); `box` =
   * [x0, y0, x1, y1] around her whole body, so the mask is the whole person, not just her top.
   * Returns 256x256 mask logits.
   */
  async start(input, points, labels = points.map(() => 1), box = null) {
    if (box) {
      points = [[box[0], box[1]], [box[2], box[3]], ...points];
      labels = [2, 3, ...labels];
    }
    const f = await this.encode(input);
    const pix = Float32Array.from(f.fpn2);
    const nm = this.k.no_mem_embed;
    for (let ch = 0; ch < C; ch++) { const b = ch * HW; for (let p = 0; p < HW; p++) pix[b + p] += nm[ch]; }
    const out = await this.heads(pix, f, await this.prompt(points, labels), points.length <= 1);
    const mem = await this.memorise(f, out.low, true);
    this.frame = 0;
    this.cond = { ...mem, ptr: out.ptr, frame: 0 };
    this.recent.clear();
    return out;
  }

  /** Next frame. Returns 256x256 mask logits (of the 1024x1024 square view) and a score. */
  async next(input) {
    const frame = ++this.frame;
    const f = await this.encode(input);
    this.trackPrompt ??= await this.prompt([], []);

    // Spatial memories: the tapped frame, then up to 6 previous frames (oldest first).
    const mems = [[0, this.cond]];
    for (let tpos = 1; tpos < NUM_MASKMEM; tpos++) {
      const prev = this.recent.get(frame - (NUM_MASKMEM - tpos));
      if (prev) mems.push([tpos, prev]);
    }
    const n1 = mems.length * HW;
    const m1 = new Float32Array(n1 * MEM), p1 = new Float32Array(n1 * MEM);
    mems.forEach(([tpos, m], i) => {
      m1.set(m.mem, i * HW * MEM);
      const t = this.k.maskmem_tpos_enc[NUM_MASKMEM - tpos - 1];
      const off = i * HW * MEM;
      for (let p = 0; p < HW; p++) for (let c = 0; c < MEM; c++) p1[off + p * MEM + c] = m.memPos[p * MEM + c] + t[c];
    });

    // Object pointers: the tapped frame, then up to 15 previous frames, with their distance.
    const maxPtrs = Math.min(frame + 1, MAX_OBJ_PTRS);
    const ptrs = [[frame - this.cond.frame, this.cond.ptr]];
    for (let d = 1; d < maxPtrs; d++) {
      const prev = this.recent.get(frame - d);
      if (prev) ptrs.push([d, prev.ptr]);
    }
    const tdMax = Math.max(1, maxPtrs - 1);
    const pe = new Float32Array(ptrs.length * C);
    ptrs.forEach(([d], i) => pe.set(sinePE(d / tdMax, C), i * C));
    const tpos = await data((await this.s.tpos.run({ x: this.T('float32', pe, [ptrs.length, C]) })).x_out);
    const split = C / MEM, n2 = ptrs.length * split;
    const m2 = new Float32Array(n2 * MEM), p2 = new Float32Array(n2 * MEM);
    ptrs.forEach(([, ptr], i) => {
      for (let s = 0; s < split; s++) {
        m2.set(ptr.subarray(s * MEM, (s + 1) * MEM), (i * split + s) * MEM);
        p2.set(tpos.subarray(i * MEM, (i + 1) * MEM), (i * split + s) * MEM);
      }
    });

    const o = await this.s.memAttn.run({
      curr: this.T('float32', toTokens(f.fpn2, C), [HW, 1, C]),
      curr_pos: this.T('float32', this.currPos, [HW, 1, C]),
      memory_1: this.T('float32', m1, [n1, 1, MEM]), memory_pos_1: this.T('float32', p1, [n1, 1, MEM]),
      memory_2: this.T('float32', m2, [n2, 1, MEM]), memory_pos_2: this.T('float32', p2, [n2, 1, MEM]),
      attention_mask_1: this.T('bool', new Uint8Array(n1).fill(1), [n1, 1]),
      attention_mask_2: this.T('bool', new Uint8Array(n2).fill(1), [n2, 1]),
    });
    const pix = fromTokens(await data(o.pix_feat), C);
    const out = await this.heads(pix, f, this.trackPrompt, true);
    const mem = await this.memorise(f, out.low, false);
    this.recent.set(frame, { ...mem, ptr: out.ptr });
    for (const k of this.recent.keys()) if (k < frame - MAX_OBJ_PTRS) this.recent.delete(k);
    return out;
  }
}
