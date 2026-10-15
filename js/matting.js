// Robust Video Matting (RVM, MobileNetV3) on the GPU through onnxruntime-web's WebGPU backend.
// It gives clean, temporally steady edges (hair, fingers, skirts) for every person in the frame;
// processor.js multiplies it by the chosen member's region, so other members still drop out.
// Model: https://github.com/PeterL1n/RobustVideoMatting (GPL-3.0, see models/RVM-LICENSE.txt),
// modified for WebGPU: onnxruntime's WebGPU convolution gives wrong results when a 3x3 conv has an
// input-channel count that isn't a multiple of 4 (decoder convs with 171/107/59/35 channels), so
// those inputs are padded with zero channels (same maths); and AveragePool ceil_mode is turned off,
// which is equivalent because frames are always sized in multiples of 32 (see analysisSize).

const ORT = new URL('../vendor/onnxruntime/ort.webgpu.bundle.min.mjs', import.meta.url).href;
const MODEL = new URL('../models/rvm_mobilenetv3_webgpu.onnx', import.meta.url).href;

let sessionPromise = null;

export async function webgpuAvailable() {
  try { return !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { return false; }
}

async function session() {
  sessionPromise ??= (async () => {
    if (!(await webgpuAvailable())) throw new Error('이 기기에서는 WebGPU를 쓸 수 없어요.');
    const ort = await import(ORT);
    ort.env.wasm.wasmPaths = new URL('../vendor/onnxruntime/', import.meta.url).href;
    ort.env.wasm.numThreads = 1;
    const s = await ort.InferenceSession.create(MODEL, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });
    return { ort, s };
  })().catch(err => { sessionPromise = null; throw err; });
  return sessionPromise;
}

export function preload() { return session().then(() => true, () => false); }

/** A matting run over one video: keeps RVM's recurrent state between frames. */
export async function createMatter(width, height) {
  const { ort, s } = await session();
  const zero = () => new ort.Tensor('float32', new Float32Array(1), [1, 1, 1, 1]);
  let rec = [zero(), zero(), zero(), zero()];
  // RVM works internally at half size (sizes are multiples of 32, so every stage divides evenly)
  // and refines edges at full input size.
  if (width % 32 || height % 32) throw new Error('matting needs sizes in multiples of 32');
  const ratio = new ort.Tensor('float32', new Float32Array([0.5]), [1]);
  const n = width * height;
  const input = new Float32Array(3 * n);

  return {
    /** @param {Uint8ClampedArray} rgba frame pixels (width*height*4); returns alpha 0..1 */
    async run(rgba) {
      for (let k = 0, j = 0; k < n; k++, j += 4) {
        input[k] = rgba[j] / 255;
        input[n + k] = rgba[j + 1] / 255;
        input[2 * n + k] = rgba[j + 2] / 255;
      }
      const out = await s.run({
        src: new ort.Tensor('float32', input, [1, 3, height, width]),
        r1i: rec[0], r2i: rec[1], r3i: rec[2], r4i: rec[3], downsample_ratio: ratio,
      });
      for (const t of rec) if (t.dispose && t.location !== 'cpu') t.dispose();
      rec = [out.r1o, out.r2o, out.r3o, out.r4o];
      const pha = out.pha.data instanceof Float32Array ? out.pha.data : await out.pha.getData();
      out.fgr.dispose?.();
      return pha;
    },
    // Forget the motion history (after a cut to a very different shot).
    reset() { rec = [zero(), zero(), zero(), zero()]; },
  };
}
