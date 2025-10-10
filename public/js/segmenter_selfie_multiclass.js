// public/js/segmenter_selfie_multiclass.js
// MediaPipe Image Segmenter – Selfie Multiclass variant (lightweight, CPU & GPU friendly)
// Produces multi-class masks (background + several person-related classes).
// We keep class 0 transparent by default and color classes 1..N.

import { loadAndCompile, setWebGpuDevice } from "@litertjs/core";
import { ensureLiteRtOnce } from "./runtime.js";
import { runWithTfjsTensors } from "@litertjs/tfjs-interop";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";

function defaultPalette(n) {
  // simple deterministic palette
  const out = [];
  for (let k = 0; k < n; k++) {
    let x = (k * 1103515245 + 12345) >>> 0;
    const r = (x >>> 16) & 0xff; x = (x * 1103515245 + 12345) >>> 0;
    const g = (x >>> 16) & 0xff; x = (x * 1103515245 + 12345) >>> 0;
    const b = (x >>> 16) & 0xff;
    out.push([r, g, b]);
  }
  return out;
}


export class SegmenterSelfieMulticlass {
  constructor({
    modelUrl = "/models/selfie_multiclass_256x256.tflite",
    wasmPath = "/wasm/",
    accelerator = "webgpu", // both webgpu and wasm generally supported
    inputSize = 256,
    classColorsUrl = "/models/selfie_multiclass_colors.json", // optional palette [[r,g,b],...]
    classLabelsUrl = "/models/selfie_multiclass_labels.json",  // optional labels ["bg","hair",...]
    overlayAlpha = 0.75,
  }) {
    this.modelUrl = modelUrl;
    this.wasmPath = wasmPath;
    this.accelerator = accelerator;
    this.inputSize = inputSize;
    this.overlayAlpha = overlayAlpha;
    this.model = null;
    this.backendName = "";
    this.type = "segmenter";
    this._warmedUp = false;
    this.classColorsUrl = classColorsUrl;
    this.classLabelsUrl = classLabelsUrl;
    this.classColors = null; // [[r,g,b], ...]
    this.classLabels = null; // optional
  }

  async init() {
    // Pick backend: try WebGPU if requested & available, else CPU
    if (this.accelerator === 'webgpu' && ('gpu' in navigator)) {
      await tf.setBackend('webgpu');
    } else {
      this.accelerator = 'wasm';
      await tf.setBackend('cpu');
    }
    await tf.ready();

    await ensureLiteRtOnce(this.wasmPath);

    // Try compile with requested accel; set device before WebGPU compile
    try {
      if (this.accelerator === 'webgpu') {
        const backend = tf.backend();
        if (!backend || !backend.device) throw new Error('No TFJS WebGPU device');
        setWebGpuDevice(backend.device);
        this.model = await loadAndCompile(this.modelUrl, { accelerator: 'webgpu' });
      } else {
        this.model = await loadAndCompile(this.modelUrl, { accelerator: 'wasm' });
      }
    } catch (e) {
      const msg = String(e?.message || e);
      const isGpuCompileFailure = /custom op|unresolved custom op|does not have a buffer handle|Convolution2DTransposeBias/i.test(msg);
      if (this.accelerator === 'webgpu' && isGpuCompileFailure) {
        console.warn('[SegmenterSelfieMC] WebGPU compile failed; falling back to WASM.', e);
        this.accelerator = 'wasm';
        await tf.setBackend('cpu');
        await tf.ready();
        this.model = await loadAndCompile(this.modelUrl, { accelerator: 'wasm' });
      } else {
        throw e;
      }
    }

    // Load optional colors & labels
    try {
      const res = await fetch(this.classColorsUrl);
      if (res.ok) {
        const json = await res.json();
        this.classColors = Array.isArray(json) ? json : (Array.isArray(json.colors) ? json.colors : null);
      }
    } catch {}
    try {
      const res = await fetch(this.classLabelsUrl);
      if (res.ok) {
        const arr = await res.json();
        this.classLabels = Array.isArray(arr) ? arr : null;
      }
    } catch {}

    // Warmup
    if (!this._warmedUp) {
      const S = this.inputSize | 0;
      const warm = tf.tidy(() => tf.zeros([1, S, S, 3], 'float32'));
      try { runWithTfjsTensors(this.model, warm); } finally { warm.dispose(); }
      this._warmedUp = true;
    }

    this.backendName = this.accelerator.toUpperCase();
  }

  // Returns { rgba, width, height } – class 0 transparent, others colored
  async run(sourceEl) {
    if (!this.model) return { rgba: new Uint8ClampedArray(0), width: 0, height: 0 };

    const S = this.inputSize;
    const vidW = sourceEl.videoWidth || sourceEl.width || 0;
    const vidH = sourceEl.videoHeight || sourceEl.height || 0;

    // 1) Preprocess NHWC [1,S,S,3], [0,1]
    const logits = tf.tidy(() => {
      const img = tf.browser.fromPixels(sourceEl).toFloat();
      const resized = tf.image.resizeBilinear(img, [S, S]);
      const nhwc = resized.div(255).reshape([1, S, S, 3]);
      const out = runWithTfjsTensors(this.model, nhwc)[0]; // expect [1,S,S,C] or [1,C,S,S]
      return out;
    });

    // 2) Get labels [S,S]
    const shp = logits.shape; // [1,S,S,C] or [1,C,S,S]
    let labels;
    if (shp.length === 4 && shp[3] > 1) {
      labels = logits.argMax(3).squeeze([0]);
    } else if (shp.length === 4 && shp[1] > 1) {
      const nhwcLogits = logits.transpose([0,2,3,1]);
      labels = nhwcLogits.argMax(3).squeeze([0]);
      nhwcLogits.dispose();
    } else {
      // binary fallback
      labels = logits.squeeze().greater(0.5).toInt();
    }
    logits.dispose();

    // 3) Upsample to video size
    let up = labels;
    if (vidW && vidH) {
      up = tf.image.resizeNearestNeighbor(labels.expandDims(-1), [vidH, vidW], false)
                   .squeeze([-1]).toInt();
    }
    if (up !== labels) labels.dispose();

    // 4) CPU colorization
    const H = up.shape[0], W = up.shape[1];
    const idx = await up.data();
    up.dispose();

    // choose palette length from channel dim if known; default 7 classes
    const numC = (shp.length === 4) ? Math.max(shp[1], shp[3]) : 2;
    const palette = (Array.isArray(this.classColors) && this.classColors.length >= numC)
      ? this.classColors
      : defaultPalette(numC);

    const rgba = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      const cls = idx[i] | 0;
      const j = i * 4;
      if (cls === 0) {
        rgba[j] = 0; rgba[j+1] = 0; rgba[j+2] = 0; rgba[j+3] = 0;
      } else {
        const c = palette[cls] || palette[cls % palette.length] || [255,0,0];
        rgba[j] = c[0] | 0; rgba[j+1] = c[1] | 0; rgba[j+2] = c[2] | 0; rgba[j+3] = 220;
      }
    }

    return { rgba, width: W, height: H };
  }
}