import { setupWebcam } from "./webcam.js";
import { Classifier } from "./classifier.js";
import { drawTopK, resizeCanvasToVideo, drawSegmentationOverlay } from "./draw.js";

const els = {
  video: document.getElementById("webcam"),
  canvas: document.getElementById("overlay"),
  status: document.getElementById("status"),
  gpuInfo: document.getElementById("gpuInfo"),
  benchBtn: document.getElementById("benchBtn"),
  compareClsBtn: document.getElementById("compareClsBtn"),
  compareSegBtn: document.getElementById("compareSegBtn"),
  compareStatus: document.getElementById("compareStatus"),
  benchTfjs: document.getElementById("benchTfjs"),
  toggleBackendBtn: document.getElementById("toggleBackendBtn"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  backendSelect: document.getElementById("backendSelect"),
  modelPath: document.getElementById("modelPath"),
  taskSelect: document.getElementById("taskSelect"),
  // perf + segmentation tuning controls (optional in DOM)
  segInterval: document.getElementById("segInterval"),
  segIntervalOut: document.getElementById("segIntervalOut"),
  overlayAlpha: document.getElementById("overlayAlpha"),
  overlayAlphaOut: document.getElementById("overlayAlphaOut"),
};

const perfEl = document.getElementById('perf');
let fpsEMA = 0;
let lastPerfUpdate = 0;
const PERF_UPDATE_MS = 500; // update perf pill ~2x/sec
let lastSegTs = 0;          // segmentation throttle timestamp

// GPU info overlay (WebGPU → Vulkan → Mali on Arm)
(async () => {
  try {
    if ('gpu' in navigator) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter?.requestAdapterInfo) {
        const info = await adapter.requestAdapterInfo();
        if (els.gpuInfo) {
          els.gpuInfo.textContent = `GPU: ${info.vendor || '—'} (${info.architecture || '—'}) • Backend: ${info.backend || '—'}`;
        }
      } else if (els.gpuInfo) {
        els.gpuInfo.textContent = 'GPU: WebGPU available';
      }
    } else if (els.gpuInfo) {
      els.gpuInfo.textContent = 'GPU: WebGPU not available';
    }
  } catch (e) {
    console.warn(e);
    if (els.gpuInfo) els.gpuInfo.textContent = 'GPU: —';
  }
})();

if (els.taskSelect) {
  const onTaskChange = () => {
    const task = els.taskSelect.value;
    // Update model path hint
    els.modelPath.textContent =
      task === 'classification' ? 'models/torchvision_mobilenet_v2.tflite' :
      task === 'selfie'         ? 'models/selfie_general_256x256.tflite' :
      task === 'selfie-mc'      ? 'models/selfie_multiclass_256x256.tflite' :
                                  'models/torchvision_mobilenet_v2.tflite';
    // Show seg-only controls for selfie segmentation tasks
    const segCtl = document.getElementById('segCtl');
    const alphaCtl = document.getElementById('alphaCtl');
    const showSeg = ['selfie', 'selfie-mc'].includes(task);
    if (segCtl) segCtl.style.display = showSeg ? '' : 'none';
    if (alphaCtl) alphaCtl.style.display = showSeg ? '' : 'none';
  };
  els.taskSelect.addEventListener('change', onTaskChange);
  onTaskChange();
}

if (els.segInterval) {
  els.segInterval.addEventListener('input', () => {
    if (els.segIntervalOut) els.segIntervalOut.textContent = String(els.segInterval.value);
  });
}
if (els.overlayAlpha) {
  els.overlayAlpha.addEventListener('input', () => {
    if (els.overlayAlphaOut) els.overlayAlphaOut.textContent = Number(els.overlayAlpha.value).toFixed(2);
  });
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden && running) {
    stopEverything();
    els.status.textContent = 'paused (tab hidden)';
    els.stopBtn.disabled = true;
    els.startBtn.disabled = false;
    if (els.taskSelect) els.taskSelect.disabled = false;
    if (els.backendSelect) els.backendSelect.disabled = false;
  }
});

let runner;                 // Classifier or Segmenter
let running = false;
let rafId = null;

// --- Classification throttling state ---
const CLASSIFY_INTERVAL_MS = 200;  // ~5 FPS
let classifyInFlight = false;
let lastClassifyTs = 0;

els.startBtn.addEventListener("click", async () => {
  if (running) return; // prevent double-starts
  try {
    els.startBtn.disabled = true;

    // lock selectors while running
    if (els.taskSelect) els.taskSelect.disabled = true;
    if (els.backendSelect) els.backendSelect.disabled = true;

    els.status.textContent = "initializing…";

    await setupWebcam(els.video, { width: 640, height: 480 });
    resizeCanvasToVideo(els.canvas, els.video);

    const task = els.taskSelect ? els.taskSelect.value : 'classification';
    els.modelPath.textContent =
      task === 'classification' ? 'models/torchvision_mobilenet_v2.tflite' :
      task === 'selfie'         ? 'models/selfie_general_256x256.tflite' :
      task === 'selfie-mc'      ? 'models/selfie_multiclass_256x256.tflite' :
                                  'models/torchvision_mobilenet_v2.tflite';

    const backend = els.backendSelect.value;

    if (task === 'classification') {
      runner = new Classifier({
        modelUrl: "/models/torchvision_mobilenet_v2.tflite",
        labelsUrl: "/models/imagenet_labels.txt",
        wasmPath: "/wasm/",
        accelerator: backend,
      });
    } else if (task === 'selfie') {
      const { SegmenterSelfie } = await import('./segmenter_selfie.js');
      runner = new SegmenterSelfie({
        modelUrl: "/models/selfie_general_256x256.tflite", // or the landscape one
        wasmPath: "/wasm/",
        accelerator: backend,  // try 'wasm' first for CPU; 'webgpu' also works
        inputSize: 256,        // 256 for general; 144x256 requires a small code tweak
        threshold: 0.5,
        color: [26,211,106],
      });
    } else if (task === 'selfie-mc') {
      const { SegmenterSelfieMulticlass } = await import('./segmenter_selfie_multiclass.js');
      runner = new SegmenterSelfieMulticlass({
        modelUrl: "/models/selfie_multiclass_256x256.tflite",
        wasmPath: "/wasm/",
        accelerator: backend,
        inputSize: 256,
      });
    }

    await runner.init();
    els.status.textContent = `ready • ${runner.backendName} • ${runner.type}`;

    // reset perf UI
    fpsEMA = 0; lastPerfUpdate = 0; lastSegTs = 0;
    if (perfEl) perfEl.textContent = '—';

    // toggle buttons
    els.stopBtn.disabled = false;
    els.startBtn.disabled = true;

    classifyInFlight = false;
    lastClassifyTs = 0;

    running = true;
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      els.video.requestVideoFrameCallback(loop);
    } else {
      rafId = requestAnimationFrame(loop);
    }
  } catch (err) {
    console.error(err);
    els.status.textContent = err?.message || String(err);
    if (perfEl) perfEl.textContent = '—';
    els.stopBtn.disabled = true;
    els.startBtn.disabled = false;
    if (els.taskSelect) els.taskSelect.disabled = false;
    if (els.backendSelect) els.backendSelect.disabled = false;
  }
});

function stopEverything() {
  // stop render loop
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;

  // Best-effort cancel for rVFC; subsequent loop() will no-op due to running=false
  // (No explicit cancel API for rVFC; guard suffices.)

  // stop webcam tracks
  const stream = els.video && els.video.srcObject;
  if (stream && typeof stream.getTracks === 'function') {
    stream.getTracks().forEach(t => {
      try { t.stop(); } catch (_) {}
    });
  }
  if (els.video) {
    els.video.srcObject = null;
  }

  // clear canvas
  if (els.canvas) {
    const ctx = els.canvas.getContext('2d');
    ctx && ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  }

  // reset perf and drop runner for clean restart
  if (perfEl) perfEl.textContent = '—';
  runner = null;
}

async function restartWithCurrentSettings() {
  // Stop if running, then click Start to reuse existing flow
  if (running) {
    stopEverything();
    await new Promise(r => setTimeout(r, 0));
  }
  // allow UI to re-enable
  els.stopBtn.disabled = true;
  els.startBtn.disabled = false;
  if (els.startBtn) els.startBtn.click();
}

els.stopBtn.addEventListener("click", () => {
  stopEverything();
  els.status.textContent = "stopped";
  els.stopBtn.disabled = true;
  els.startBtn.disabled = false;
  if (els.taskSelect) els.taskSelect.disabled = false;
  if (els.backendSelect) els.backendSelect.disabled = false;
});

// Toggle backend and restart if running
if (els.toggleBackendBtn) {
  els.toggleBackendBtn.addEventListener('click', async () => {
    const cur = els.backendSelect?.value || 'webgpu';
    if (els.backendSelect) els.backendSelect.value = (cur === 'webgpu') ? 'wasm' : 'webgpu';
    await restartWithCurrentSettings();
  });
}

// ---- Benchmark (restored): WebGPU vs WASM for the CURRENT task ----
if (els.benchBtn) {
  els.benchBtn.addEventListener('click', async () => {
    try {
      const task = els.taskSelect ? els.taskSelect.value : 'classification';
      const backends = ['webgpu', 'wasm'];
      const results = [];

      // Ensure webcam frame
      await setupWebcam(els.video, { width: 640, height: 480 });

      // Helper to instantiate the correct runner per task/backend
      const makeRunner = async (task, backend) => {
        if (task === 'classification') {
          return new Classifier({
            modelUrl: '/models/torchvision_mobilenet_v2.tflite',
            labelsUrl: '/models/imagenet_labels.txt',
            wasmPath: '/wasm/',
            accelerator: backend,
          });
        } else if (task === 'selfie') {
          const { SegmenterSelfie } = await import('./segmenter_selfie.js');
          return new SegmenterSelfie({
            modelUrl: '/models/selfie_general_256x256.tflite',
            wasmPath: '/wasm/',
            accelerator: backend,
            inputSize: 256,
            threshold: 0.5,
            color: [26,211,106],
          });
        } else if (task === 'selfie-mc') {
          const { SegmenterSelfieMulticlass } = await import('./segmenter_selfie_multiclass.js');
          return new SegmenterSelfieMulticlass({
            modelUrl: '/models/selfie_multiclass_256x256.tflite',
            wasmPath: '/wasm/',
            accelerator: backend,
            inputSize: 256,
          });
        }
      };

      // Warmup/Run counts per task
      const WARM = task === 'classification' ? 10 : 5;
      const RUNS = task === 'classification' ? 50 : 20;

      for (const b of backends) {
        if (els.status) els.status.textContent = `benchmark • init ${b}…`;

        const r = await makeRunner(task, b);
        const tInit0 = performance.now();
        await r.init();
        const tInit1 = performance.now();

        // Warmup
        for (let i = 0; i < WARM; i++) {
          const out = await r.run(els.video);
          (out && out.forEach && out.forEach(t => t?.dispose?.()));
        }

        // Time only run()
        const t0 = performance.now();
        for (let i = 0; i < RUNS; i++) {
          const out = await r.run(els.video);
          (out && out.forEach && out.forEach(t => t?.dispose?.()));
        }
        const t1 = performance.now();

        const initMs = (tInit1 - tInit0).toFixed(1);
        const avgMs = ((t1 - t0) / RUNS).toFixed(1);
        const fps = (1000 / ((t1 - t0) / RUNS)).toFixed(1);

        results.push({ backend: b.toUpperCase(), initMs, avgMs, fps });
      }

      if (els.status) {
        els.status.textContent =
          `benchmark → ` +
          results.map(r => `${r.backend}: init ${r.initMs} ms, ${r.avgMs} ms (${r.fps} FPS)`).join(' • ');
      }
    } catch (err) {
      console.error(err);
      if (els.status) els.status.textContent = `benchmark error: ${err?.message || err}`;
    }
  });
}

function loop(ts = performance.now()) {
  if (!running) return;

  resizeCanvasToVideo(els.canvas, els.video);
  const t0 = performance.now();

  if (!runner) {
    // schedule next frame and bail early
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      els.video.requestVideoFrameCallback(loop);
    } else {
      rafId = requestAnimationFrame(loop);
    }
    return;
  }

  if (runner.type === 'segmenter') {
    // Throttle segmentation according to UI slider (ms between runs)
    const minInterval = Math.max(0, Number(els.segInterval?.value || 0));
    if ((t0 - lastSegTs) >= minInterval) {
      lastSegTs = t0;
      runner
        .run(els.video)
        .then(({ rgba, width, height }) => {
          const alpha = Math.min(1, Math.max(0, Number(els.overlayAlpha?.value || 0.5)));
          drawSegmentationOverlay(els.canvas, els.video, rgba, width, height, alpha);

          // perf
          const dt = performance.now() - t0;
          const fps = dt > 0 ? 1000 / dt : 0;
          fpsEMA = fpsEMA ? (fpsEMA * 0.9 + fps * 0.1) : fps;
          if (perfEl && (t0 - lastPerfUpdate) > PERF_UPDATE_MS) {
            perfEl.textContent = `${runner.backendName} • ${runner.type} • ${fpsEMA.toFixed(1)} FPS`;
            lastPerfUpdate = t0;
            // Append live FPS to GPU info pill
            if (els.gpuInfo) {
              const base = els.gpuInfo.textContent || '';
              els.gpuInfo.textContent = base.replace(/\s•\s[0-9.]+\sFPS$/, '') + ` • ${fpsEMA.toFixed(1)} FPS`;
            }
          }
        })
        .catch(console.error);
    }
  } else {
    // Classifier: throttle to ~5 FPS
    if (!classifyInFlight && ts - lastClassifyTs >= CLASSIFY_INTERVAL_MS) {
      classifyInFlight = true;
      runner
        .run(els.video)
        .then((result) => {
          drawTopK(els.canvas, result);
          // perf
          const dt = performance.now() - t0;
          const fps = dt > 0 ? 1000 / dt : 0;
          fpsEMA = fpsEMA ? (fpsEMA * 0.9 + fps * 0.1) : fps;
          if (perfEl && (t0 - lastPerfUpdate) > PERF_UPDATE_MS) {
            perfEl.textContent = `${runner.backendName} • ${runner.type} • ${fpsEMA.toFixed(1)} FPS`;
            lastPerfUpdate = t0;
            // Append live FPS to GPU info pill
            if (els.gpuInfo) {
              const base = els.gpuInfo.textContent || '';
              els.gpuInfo.textContent = base.replace(/\s•\s[0-9.]+\sFPS$/, '') + ` • ${fpsEMA.toFixed(1)} FPS`;
            }
          }
        })
        .catch(console.error)
        .finally(() => {
          classifyInFlight = false;
          lastClassifyTs = ts;
        });
    }
  }

  // schedule next frame once at the end
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    els.video.requestVideoFrameCallback(loop);
  } else {
    rafId = requestAnimationFrame(loop);
  }
}
// ---- CPU Comparison: LiteRT WASM (XNNPACK) vs TFJS-CPU (classification) ----
if (els.compareClsBtn) {
  els.compareClsBtn.addEventListener('click', async () => {
    try {
      const tfmod = await import('@tensorflow/tfjs');
      const tf = tfmod.default || tfmod;
      await setupWebcam(els.video, { width: 640, height: 480 });
      const INPUT = 224;

      // Preprocess once on CPU
      await tf.setBackend('cpu');
      await tf.ready();
      const inputNCHW = tf.tidy(() => {
        const img = tf.browser.fromPixels(els.video).toFloat();
        const resized = tf.image.resizeBilinear(img, [INPUT, INPUT]).div(255);
        return resized.reshape([1, INPUT, INPUT, 3]).transpose([0,3,1,2]); // NCHW
      });
      const inputNHWC = tf.tidy(() => inputNCHW.transpose([0,2,3,1]));      // NHWC for TFJS model

      const results = [];

      // TFJS-CPU baseline
      if (els.compareStatus) els.compareStatus.textContent = 'comparison • TFJS-CPU…';
      const mobilenetMod = await import('@tensorflow-models/mobilenet');
      const mobilenet = mobilenetMod.default || mobilenetMod;
      const model = await mobilenet.load({ version: 2, alpha: 1.0 });

      const inputNHWC_on = inputNHWC.clone();
      const WARM_T = 5, RUNS_T = 20;
      for (let i = 0; i < WARM_T; i++) { model.infer(inputNHWC_on, 'conv_preds').dispose?.(); }
      const t0 = performance.now();
      for (let i = 0; i < RUNS_T; i++) { model.infer(inputNHWC_on, 'conv_preds').dispose?.(); }
      const t1 = performance.now();
      inputNHWC_on.dispose();
      model.dispose?.();

      const avgT = (t1 - t0) / RUNS_T;
      results.push({ backend: 'TFJS-CPU', avgMs: avgT.toFixed(1), fps: (1000/avgT).toFixed(1) });

      // LiteRT WASM
      if (els.compareStatus) els.compareStatus.textContent = 'comparison • LiteRT WASM…';
      const r = new Classifier({
        modelUrl: '/models/torchvision_mobilenet_v2.tflite',
        labelsUrl: '/models/imagenet_labels.txt',
        wasmPath: '/wasm/',
        accelerator: 'wasm',
      });
      await r.init();

      const { runWithTfjsTensors } = await import('@litertjs/tfjs-interop');
      const inputNCHW_on = inputNCHW.clone();
      const WARM_L = 10, RUNS_L = 50;
      for (let i = 0; i < WARM_L; i++) {
        runWithTfjsTensors(r.model, inputNCHW_on).forEach(t => t.dispose?.());
      }
      const l0 = performance.now();
      for (let i = 0; i < RUNS_L; i++) {
        runWithTfjsTensors(r.model, inputNCHW_on).forEach(t => t.dispose?.());
      }
      const l1 = performance.now();
      inputNCHW_on.dispose();

      const avgL = (l1 - l0) / RUNS_L;
      results.push({ backend: 'LiteRT WASM (XNNPACK)', avgMs: avgL.toFixed(1), fps: (1000/avgL).toFixed(1) });

      // Cleanup shared tensors
      inputNCHW.dispose(); inputNHWC.dispose();

      if (els.compareStatus) {
        els.compareStatus.textContent =
          'CPU comparison (classification) → ' +
          results.map(r => `${r.backend}: ${r.avgMs} ms (${r.fps} FPS)`).join(' • ');
      }
    } catch (err) {
      console.error(err);
      if (els.compareStatus) els.compareStatus.textContent = `comparison error: ${err?.message || err}`;
    }
  });
}

// ---- CPU Comparison: Segmentation (Selfie Multiclass or Selfie) ----
if (els.compareSegBtn) {
  els.compareSegBtn.addEventListener('click', async () => {
    try {
      await setupWebcam(els.video, { width: 640, height: 480 });

      // --- TFJS-CPU baseline via @tensorflow-models/body-segmentation ---
      if (els.compareStatus) els.compareStatus.textContent = `comparison • Segmentation • TFJS-CPU…`;
      const tfmod = await import('@tensorflow/tfjs');
      const tf = tfmod.default || tfmod;
      await tf.setBackend('cpu');
      await tf.ready();

      const bsMod = await import('@tensorflow-models/body-segmentation');
      const bodySeg = bsMod.default || bsMod;

      let tfModelEnum, tfOptions;
      tfModelEnum = bodySeg.SupportedModels?.MediaPipeSelfieSegmentation;
      if (!tfModelEnum) throw new Error('MediaPipeSelfieSegmentation not available in this body-segmentation version.');
      tfOptions = { runtime: 'tfjs', modelType: 'general' };

      const tfSeg = await bodySeg.createSegmenter(tfModelEnum, tfOptions);
      const WARM_T = 3, RUNS_T = 10;
      for (let i = 0; i < WARM_T; i++) { await tfSeg.segmentPeople(els.video); }
      const t0 = performance.now();
      for (let i = 0; i < RUNS_T; i++) { await tfSeg.segmentPeople(els.video); }
      const t1 = performance.now();
      const tfjsAvgMs = (t1 - t0) / RUNS_T;
      const tfjsFps = 1000 / tfjsAvgMs;

      // --- LiteRT WASM (XNNPACK) ---
      if (els.compareStatus) els.compareStatus.textContent += ' • LiteRT WASM…';
      let seg;
      const { SegmenterSelfie } = await import('./segmenter_selfie.js');
      seg = new SegmenterSelfie({
        modelUrl: '/models/selfie_general_256x256.tflite',
        wasmPath: '/wasm/',
        accelerator: 'wasm',
        inputSize: 256,
        threshold: 0.5,
        color: [26,211,106],
      });

      const tInit0 = performance.now();
      await seg.init();
      const tInit1 = performance.now();

      const WARM = 5, RUNS = 20;
      for (let i = 0; i < WARM; i++) { await seg.run(els.video); }
      const l0 = performance.now();
      for (let i = 0; i < RUNS; i++) { await seg.run(els.video); }
      const l1 = performance.now();

      const liteAvg = (l1 - l0) / RUNS;
      const liteFps = 1000 / liteAvg;

      if (els.compareStatus) {
        els.compareStatus.textContent =
          `CPU comparison (segmentation: ) → ` +
          `TFJS-CPU: ${isFinite(tfjsAvgMs) ? tfjsAvgMs.toFixed(1) + ' ms (' + tfjsFps.toFixed(1) + ' FPS)' : 'N/A'} • ` +
          `LiteRT WASM: init ${(tInit1 - tInit0).toFixed(1)} ms, ${liteAvg.toFixed(1)} ms (${liteFps.toFixed(1)} FPS)`;
      }
    } catch (err) {
      console.error(err);
      if (els.compareStatus) els.compareStatus.textContent = `comparison error: ${err?.message || err}`;
    }
  });
}