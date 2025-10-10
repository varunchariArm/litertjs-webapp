import { loadLiteRt, loadAndCompile, setWebGpuDevice } from "@litertjs/core";
import { runWithTfjsTensors } from "@litertjs/tfjs-interop";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";

async function assertAsset(url) {
  const res = await fetch(url, { method: 'GET' });
  if (!res.ok) throw new Error(`Asset not found or not served: ${url} (HTTP ${res.status})`);
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('text/html')) throw new Error(`Unexpected HTML at ${url}. Did you place the file under public/models/?`);
  return url;
}

const NUM_DETECTIONS = 100; // common for SSD MobileNet V2

export class Detector {
  constructor({ modelUrl, labelsUrl, wasmPath, accelerator = 'webgpu' }) {
    this.modelUrl = modelUrl;
    this.labelsUrl = labelsUrl;
    this.wasmPath = wasmPath;
    this.accelerator = accelerator; // 'webgpu' | 'wasm'
    this.model = null;
    this.labels = [];
    this.backendName = "";
    this.type = 'detector';
  }

  async init() {
    // Init TFJS backend based on requested accelerator
    if (this.accelerator === 'webgpu') {
      if (!('gpu' in navigator)) {
        throw new Error("WebGPU not supported in this browser. Try Chrome/Edge 121+.");
      }
      await tf.setBackend('webgpu');
    } else {
      await tf.setBackend('cpu');
    }
    await tf.ready();

    // Init LiteRT runtime (loads Wasm assets regardless)
    await loadLiteRt(this.wasmPath);

    if (this.accelerator === 'webgpu') {
      const backend = tf.backend();
      if (!backend || !backend.device) {
        throw new Error("Failed to access TFJS WebGPU device.");
      }
      setWebGpuDevice(backend.device);
    }

    // Ensure model asset exists and is not HTML
    await assertAsset(this.modelUrl);

    // Load model + labels
    this.model = await loadAndCompile(this.modelUrl, { accelerator: this.accelerator });
    const labelsRes = await fetch(this.labelsUrl);
    if (!labelsRes.ok) throw new Error(`Labels file missing: ${this.labelsUrl} (HTTP ${labelsRes.status})`);
    this.labels = await labelsRes.json();

    this.backendName = this.accelerator.toUpperCase();
  }

  /**
   * Run detection on a HTMLVideoElement (or canvas/image)
   * Returns [{bbox:[x,y,w,h], classId, label, score}]
   */
  async run(sourceEl) {
    if (!this.model) return [];

    // Wrap pre/post with tf.tidy for GC of intermediate tensors
    const { boxes, classes, scores, count } = tf.tidy(() => {
      // Preprocess: from video -> normalized tensor [1,H,W,3]
      const inputH = 300; // typical SSD size; adjust to your model
      const inputW = 300;
      const img = tf.browser.fromPixels(sourceEl).toFloat();
      const resized = tf.image.resizeBilinear(img, [inputH, inputW]);
      const normalized = resized.div(255).reshape([1, inputH, inputW, 3]).transpose([0,3,1,2]); // NHWC->NCHW

      // Run model
      const outputs = runWithTfjsTensors(this.model, normalized);
      // Expect: [boxes, classes, scores, num]
      // boxes: [1, N, 4] (ymin,xmin,ymax,xmax) in normalized coords
      // classes: [1, N] (int)
      // scores: [1, N]
      // num: [1]
      const b = outputs[0];
      const c = outputs[1];
      const s = outputs[2];
      const n = outputs[3];
      return { boxes: b, classes: c, scores: s, count: n };
    });

    // Read back small outputs to CPU
    const n = (await count.data())[0] | 0;
    const num = Math.min(n || NUM_DETECTIONS, NUM_DETECTIONS);
    const boxesArr = await boxes.data();
    const classesArr = await classes.data();
    const scoresArr = await scores.data();

    boxes.dispose(); classes.dispose(); scores.dispose(); count.dispose();

    // Map to objects with pixel coords relative to sourceEl
    const W = sourceEl.videoWidth || sourceEl.width;
    const H = sourceEl.videoHeight || sourceEl.height;

    const dets = [];
    for (let i = 0; i < num; i++) {
      const yMin = boxesArr[i*4 + 0];
      const xMin = boxesArr[i*4 + 1];
      const yMax = boxesArr[i*4 + 2];
      const xMax = boxesArr[i*4 + 3];
      const cls = classesArr[i] | 0;
      const score = scoresArr[i];

      const x = xMin * W;
      const y = yMin * H;
      const w = (xMax - xMin) * W;
      const h = (yMax - yMin) * H;

      dets.push({ bbox: [x, y, w, h], classId: cls, label: this.labels[cls] || String(cls), score });
    }
    return dets;
  }
}