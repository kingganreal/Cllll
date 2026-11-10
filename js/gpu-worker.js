// Runs the GPU models off the page: SAM 2, the matting model and face recognition (see js/gpu.js).
// Messages: { id, method, args } -> { id, ok, value } or { id, ok: false, error }.
import * as matting from './matting.js';
import * as sam2 from './sam2.js';
import * as face from './face.js';
import * as yolo from './yolo.js';

let trust = {};
sam2.setTrustStore({
  get: () => ({ ...trust }),
  set: v => { trust = v; postMessage({ type: 'trust', value: v }); },
});

let matter = null, samModels = null, tracker = null, faceModels = null, yoloModel = null;

// SAM output for the page: the mask logits (transferred) and the score.
const samOut = out => ({ value: { low: out.low, score: out.score, iou: out.iou }, transfer: [out.low.buffer] });

const methods = {
  // Test switches and the saved GPU check answers from the page.
  config({ trust: saved, samEp, recent }) {
    if (saved) trust = saved;
    if (samEp) globalThis.__FANCAM_SAM_EP__ = samEp;
    if (recent) globalThis.__FANCAM_SAM_RECENT__ = recent;
    return true;
  },
  // Warm up while the user is still choosing, so 추출 starts sooner.
  async preload() {
    // One after another: they all start the GPU, and doing that at once has hung.
    await matting.preload();
    await sam2.load().catch(() => {});
    await face.load().catch(() => {});
    return true;
  },

  async mattingCreate(w, h) { matter = await matting.createMatter(w, h); return true; },
  async matte(rgba) {
    const pha = Float32Array.from(await matter.run(rgba));
    return { value: pha, transfer: [pha.buffer] };
  },
  mattingReset() { matter?.reset(); return true; },

  async samLoad() {
    samModels = await sam2.load(f => postMessage({ type: 'progress', value: f }));
    return true;
  },
  samTrusted(input, points, labels, box) { return sam2.gpuTrusted(samModels, input, points, labels, box); },
  async samStart(input, points, labels, box) {
    tracker = new sam2.Tracker(samModels);
    return samOut(await tracker.start(input, points, labels, box));
  },
  async samNext(input) { return samOut(await tracker.next(input)); },

  async faceLoad() { faceModels = await face.load(); return true; },
  async yoloLoad() { yoloModel = await yolo.load(); return true; },
  people(rgba, w, h) { return yolo.people(yoloModel, rgba, w, h); },
  async faces(bitmap, w, h, minSize) {
    try {
      const found = await face.faces(faceModels, ctx => ctx.drawImage(bitmap, 0, 0), w, h, { minSize });
      return found.map(({ box, kps, score, size, emb }) => ({ box, kps, score, size, emb }));
    } finally {
      bitmap.close();
    }
  },
};

onmessage = async ({ data: { id, method, args } }) => {
  try {
    const r = await methods[method](...args);
    if (r && typeof r === 'object' && 'transfer' in r) postMessage({ id, ok: true, value: r.value }, r.transfer);
    else postMessage({ id, ok: true, value: r });
  } catch (err) {
    postMessage({ id, ok: false, error: err?.message || String(err) });
  }
};
