// One copy of onnxruntime-web for every model (matting, SAM 2, face). When two modules create
// sessions at the same time, onnxruntime starts its wasm twice and the second one fails with
// "multiple calls to 'initWasm()'". So it's set up once here and sessions are made one at a time.

const ORT = new URL('../vendor/onnxruntime/ort.webgpu.bundle.min.mjs', import.meta.url).href;

let ortPromise = null;
let queue = Promise.resolve();

export function loadOrt() {
  ortPromise ??= import(ORT).then(ort => {
    ort.env.wasm.wasmPaths = new URL('../vendor/onnxruntime/', import.meta.url).href;
    ort.env.wasm.numThreads = 1;
    return ort;
  }).catch(err => { ortPromise = null; throw err; });
  return ortPromise;
}

export async function createSession(model, options) {
  const ort = await loadOrt();
  const made = queue.then(() => ort.InferenceSession.create(model, options));
  queue = made.catch(() => {});
  return made;
}
