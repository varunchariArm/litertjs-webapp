// public/js/segmenter_selfie.js
// Note: Some Selfie Segmentation TFLite builds include a custom op `Convolution2DTransposeBias`.
// If WebGPU compilation fails due to this unresolved custom op, we automatically fall back to WASM.
import { loadAndCompile, setWebGpuDevice } from "@litertjs/core";
import { ensureLiteRtOnce } from "./runtime.js";
import { runWithTfjsTensors } from "@litertjs/tfjs-interop";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";

export class SegmenterSelfie {
  constructor({
    modelUrl = "/models/selfie_general_256x256.tflite",
    wasmPath = "/wasm/",
    accelerator = "wasm",            // CPU default (fastest for this tiny model)
    inputSize = 256,                 // 256x256 (or 144x256 for landscape)
    threshold = 0.5,                 // probability threshold
    color = [26, 211, 106],          // RGB color for 'person'
  }) {
    this.modelUrl = modelUrl;
    this.wasmPath = wasmPath;
    this.accelerator = accelerator;  // 'wasm' | 'webgpu'
    this.inputSize = inputSize;
    this.threshold = threshold;
    this.color = color;
    this.model = null;
    this.backendName = "";
    this.type = "segmenter";         // keep same type so main.js path works
    this._warmedUp = false;
  }

  async init() {
    // Pick backend: prefer WebGPU, fallback to CPU if unavailable
    if (this.accelerator === "webgpu" && ("gpu" in navigator)) {
      await tf.setBackend("webgpu");
    } else {
      this.accelerator = "wasm";
      await tf.setBackend("cpu");
    }
    await tf.ready();

    await ensureLiteRtOnce(this.wasmPath);

    // Try to compile the model with requested accelerator
    try {
      if (this.accelerator === "webgpu") {
        const backend = tf.backend();
        if (!backend || !backend.device) throw new Error("No TFJS WebGPU device");
        // IMPORTANT: set device BEFORE compiling the LiteRT model
        setWebGpuDevice(backend.device);
        this.model = await loadAndCompile(this.modelUrl, { accelerator: "webgpu" });
      } else {
        // WASM path (TFJS stays on CPU)
        this.model = await loadAndCompile(this.modelUrl, { accelerator: "wasm" });
      }
    } catch (e) {
      const msg = String(e?.message || e);
      console.warn('[SegmenterSelfie] compile error:', msg);
      // Treat unresolved custom ops AND missing GPU buffer handles as GPU compile failures
      const isGpuCompileFailure = /custom op|Convolution2DTransposeBias|unresolved custom op|does not have a buffer handle|buffer handle/i.test(msg);

      if (this.accelerator === 'webgpu' && isGpuCompileFailure) {
        console.warn('[SegmenterSelfie] WebGPU compile failed; falling back to WASM.', e);
        this.accelerator = 'wasm';
        await tf.setBackend('cpu');
        await tf.ready();
        this.model = await loadAndCompile(this.modelUrl, { accelerator: 'wasm' });
      } else {
        throw e;
      }
    }

    // Warmup once (NHWC, [1,S,S,3])
    if (!this._warmedUp) {
      const S = this.inputSize | 0;
      const warm = tf.tidy(() => tf.zeros([1, S, S, 3], "float32"));
      try { runWithTfjsTensors(this.model, warm); } finally { warm.dispose(); }
      this._warmedUp = true;
    }

    this.backendName = this.accelerator.toUpperCase();
  }

  /**
   * Returns { rgba, width, height } where rgba is Uint8ClampedArray of upsampled mask.
   * Model output: [1, S, S, 1] probability for "person".
   */
  async run(sourceEl) {
    if (!this.model) return { rgba: new Uint8ClampedArray(0), width: 0, height: 0 };

    const S = this.inputSize;
    const vidW = sourceEl.videoWidth || sourceEl.width || 0;
    const vidH = sourceEl.videoHeight || sourceEl.height || 0;

    // 1) Forward pass: NHWC, range [0,1]. (Selfie models expect 256x256x3 / 144x256x3)
    const prob = tf.tidy(() => {
      const img = tf.browser.fromPixels(sourceEl).toFloat();       // [H0,W0,3]
      const resized = tf.image.resizeBilinear(img, [S, S]);        // [S,S,3]
      const nhwc = resized.div(255).reshape([1, S, S, 3]);         // [1,S,S,3]
      const out = runWithTfjsTensors(this.model, nhwc)[0];         // -> [1,S,S,1]
      return out; // keep tensor for arg
    });

    // 2) Threshold to binary labels and upsample to video size
    let mask; // [S,S] int
    try {
      mask = prob.greater(this.threshold).toInt().squeeze();       // [S,S]
    } finally {
      // keep prob until after squeeze complete
    }

    let up = mask;
    if (vidW && vidH) {
      up = tf.image
        .resizeNearestNeighbor(mask.expandDims(-1), [vidH, vidW], false)
        .squeeze([-1])                                             // [vidH,vidW]
        .toInt();
    }
    if (up !== mask) mask.dispose();
    prob.dispose();

    // 3) Build RGBA on CPU (background transparent)
    const H = up.shape[0], W = up.shape[1];
    const idx = await up.data(); // int32 0/1
    up.dispose();

    const [r, g, b] = this.color;
    const rgba = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      const j = i * 4;
      if (idx[i]) {  // person
        rgba[j] = r; rgba[j + 1] = g; rgba[j + 2] = b; rgba[j + 3] = 220;
      } else {
        rgba[j] = 0; rgba[j + 1] = 0; rgba[j + 2] = 0; rgba[j + 3] = 0; // transparent
      }
    }
    return { rgba, width: W, height: H };
  }
}