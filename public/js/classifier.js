import { loadAndCompile, setWebGpuDevice } from "@litertjs/core";
import { runWithTfjsTensors } from "@litertjs/tfjs-interop";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import '@tensorflow/tfjs-backend-wasm';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { ensureLiteRtOnce } from "./runtime.js";

// The referenced torchvision MobileNetV2 .tflite expects NCHW inputs and TorchVision normalization.
async function assertAsset(url) {
  const res = await fetch(url, { method: 'GET' });
  if (!res.ok) throw new Error(`Asset not found or not served: ${url} (HTTP ${res.status})`);
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('text/html')) throw new Error(`Unexpected HTML at ${url}. Did you place the file under public/models/?`);
  return url;
}

export class Classifier {
  constructor({ modelUrl, labelsUrl, wasmPath, accelerator = 'webgpu', topK = 1 }) {
    this.modelUrl = modelUrl;
    this.labelsUrl = labelsUrl;
    this.wasmPath = wasmPath;
    this.accelerator = accelerator; // 'webgpu' | 'wasm'
    this.topK = topK;
    this.model = null;
    this.labels = [];
    this.backendName = "";
    this.type = 'classifier';
  }

  async init() {
    // Init TFJS backend based on requested accelerator
    setWasmPaths('/tfwasm/')
    if (this.accelerator === 'webgpu') {
      if (!('gpu' in navigator)) {
        throw new Error("WebGPU not supported in this browser.");
      }
      await tf.setBackend('webgpu');
    } else {
      await tf.setBackend('wasm');
    }
    await tf.ready();

    await ensureLiteRtOnce(this.wasmPath);

    if (this.accelerator === 'webgpu') {
      const backend = tf.backend();
      if (!backend || !backend.device) throw new Error("Failed to access TFJS WebGPU device.");
      setWebGpuDevice(backend.device);
    }

    await assertAsset(this.modelUrl);

    // imagenet_labels.txt is one label per line
    const labelsRes = await fetch(this.labelsUrl);
    if (!labelsRes.ok) throw new Error(`Labels file missing: ${this.labelsUrl} (HTTP ${labelsRes.status})`);
    const txt = await labelsRes.text();
    this.labels = txt.split(/\r?\n/).map(s => s.trim()).filter(Boolean);

    this.model = await loadAndCompile(this.modelUrl, { accelerator: this.accelerator });

    // Warm up once to compile pipelines/shaders and allocate persistent buffers
    if (!this._warmedUp) {
      const inputH = 224, inputW = 224; // MobileNetV2 default
      const warm = tf.tidy(() => tf.zeros([1, 3, inputH, inputW], 'float32'));
      try {
        runWithTfjsTensors(this.model, warm);
      } finally {
        warm.dispose();
      }
      this._warmedUp = true;
    }

    this.backendName = this.accelerator.toUpperCase();
  }

  async run(sourceEl) {
    if (!this.model) return [];

    const inputH = 224, inputW = 224;

    // Helper: preprocess using TorchVision normalization (expects NCHW)
    const prepTorchVision = () => tf.tidy(() => {
      const img = tf.browser.fromPixels(sourceEl).toFloat();
      const resized = tf.image.resizeBilinear(img, [inputH, inputW]);
      const float01 = resized.div(255).reshape([1, inputH, inputW, 3]); // NHWC
      const mean = tf.tensor1d([0.485, 0.456, 0.406]).reshape([1,1,1,3]);
      const std  = tf.tensor1d([0.229, 0.224, 0.225]).reshape([1,1,1,3]);
      const nhwc = float01.sub(mean).div(std);
      return nhwc.transpose([0,3,1,2]); // NCHW [1,3,H,W]
    });

    // Helper: preprocess for classic TFLite MobileNet scaling to [-1,1] (also NCHW)
    const prepTflite = () => tf.tidy(() => {
      const img = tf.browser.fromPixels(sourceEl).toFloat();
      const resized = tf.image.resizeBilinear(img, [inputH, inputW]);
      // Scale [0,255] -> [0,1] -> [-1,1]
      const nhwc = resized.div(255).mul(2).sub(1).reshape([1, inputH, inputW, 3]);
      return nhwc.transpose([0,3,1,2]); // NCHW
    });

    // Helper: run once and return topK or null if non-finite
    const tryRun = async (inputTensor) => {
        const cpuData = await inputTensor.data();
        const cpuTensor = tf.tensor(cpuData, inputTensor.shape, inputTensor.dtype);

        const outputs = runWithTfjsTensors(this.model, cpuTensor);
        const logits = outputs[0]; // [1,1000]
        // Numerically-stable softmax is used by tf.softmax internally
        const probs = tf.softmax(logits);

        // Check for non-finite values
        const finite = tf.all(tf.isFinite(probs));
        const ok = finite.arraySync();
        if (!ok) return null;

        const squeezed = probs.squeeze(); // [1000]
        const { values, indices } = tf.topk(squeezed, this.topK);
        return { values, indices };
    };

    // Try TorchVision normalization first
    let inp = prepTorchVision();
    let result = await tryRun(inp);
    inp.dispose();

    // Fallback to TFLite Mobilenet scaling if non-finite
    if (result === null) {
      const alt = prepTflite();
      result = await tryRun(alt);
      alt.dispose();
    }

    // If still null, bail gracefully
    if (result === null) {
      return [ { label: 'Model produced NaNs', prob: 0 } ];
    }

    const vals = await result.values.data();
    const idxs = await result.indices.data();
    result.values.dispose();
    result.indices.dispose();

    const out = [];
    for (let j = 0; j < idxs.length; j++) {
      const i = idxs[j] | 0;
      const p = vals[j];
      // Guard label bounds
      const label = (i >= 0 && i < this.labels.length) ? this.labels[i] : `class_${i}`;
      // Guard prob sanity
      const prob = Number.isFinite(p) ? p : 0;
      out.push({ label, prob });
    }
    return out;
  }
}