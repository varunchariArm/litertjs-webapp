# LiteRT.js Webcam AI (WebGPU + WASM/XNNPACK)

Real-time **classification** and **segmentation** directly in the browser using **LiteRT.js**:
- **WebGPU** on Arm Mali GPUs (via Vulkan) for high-performance GPU inference.
- **WASM + XNNPACK** on Arm CPUs (NEON SIMD) for optimized CPU execution.
- 100% **on-device**; webcam frames never leave the local machine.

## ✨ Overview
LiteRT.js enables TensorFlow Lite models to run fully in-browser with native-like performance. This demo showcases:
- Live webcam inference.
- **Classification** (MobileNetV2).
- **Segmentation** (Selfie / Selfie Multiclass).
- Built-in benchmarking and CPU vs GPU comparisons.

## 🧠 Supported Models

| Task | Model | Dataset | Path |
|------|--------|----------|------|
| **Classification** | MobileNetV2 | ImageNet | `public/models/torchvision_mobilenet_v2.tflite` |
| **Selfie Segmentation (Person)** | MediaPipe Selfie Segmentation | Human mask | `public/models/selfie_general_256x256.tflite` |
| **Selfie Multiclass Segmentation** | ADE20K / LiteRT Multiclass | Scene parsing | `public/models/selfie_multiclass_256x256.tflite` |

## 🧩 Architecture
```
WebApp (JS / HTML / CSS)
        ↓
     LiteRT.js
    ↙        ↘
 WebGPU      WASM (SIMD)
  ↓              ↓
Vulkan API   XNNPACK
  ↓              ↓
Mali GPU     Arm CPU (NEON SIMD)
```
All inference happens locally on the Arm SoC — GPU or CPU — with zero cloud dependencies.

## ⚙️ Tech Stack
- **LiteRT.js Runtime** — TensorFlow Lite execution in JS.
- **WebGPU** — Vulkan-based shader compute (Arm Mali GPU).
- **WASM + XNNPACK** — SIMD-optimized CPU kernels.
- **getUserMedia** — Webcam stream capture.
- **HTML, CSS, JS (Vite)** — Responsive and modular frontend.

## 🚀 Setup

### Prerequisites
- Node.js 18+.
- Chrome/Edge (v113+ with WebGPU enabled).

### Install
```bash
npm install
# Copy LiteRT wasm runtime files (not checked into Git)
mkdir -p public/wasm && cp -r node_modules/@litertjs/core/wasm/* public/wasm/
```

### Run
```bash
npm run dev
# open localhost URL and allow camera permissions
```

### Build / Preview
```bash
npm run build
npm run preview
```

## 📊 Benchmarks

### Classification (MobileNetV2)
- **Benchmark:**  
  - WEBGPU: init 313.2 ms, 12.4 ms (80.8 FPS)
  - WASM: init 133.4 ms, 49.4 ms (20.3 FPS)
- **CPU Comparison:**  
  - TFJS-CPU: 592.6 ms (1.7 FPS)
  - LiteRT WASM (XNNPACK): 13.9 ms (72.2 FPS)

### Segmentation (Selfie Multiclass)
- **Benchmark:**  
  - WEBGPU: init 296.8 ms, 16.3 ms (61.2 FPS)
  - WASM: init 254.8 ms, 184.3 ms (5.4 FPS)
- **CPU Comparison (Selfie):**  
  - TFJS-CPU: 294.0 ms (3.4 FPS)
  - LiteRT WASM: init 66.7 ms, 19.4 ms (51.4 FPS)

## 🧾 Notes
- Uses WebGPU when available, otherwise falls back to WASM (XNNPACK).
- CPU comparison buttons measure **inference-only** latency on identical inputs.
- All model data and inference remain local to the device.
- EfficientViT segmentation path and model have been removed for simplicity.

## 📚 Credits
- **LiteRT.js** by Google AI Edge.
- **Arm Mali / Cortex Optimizations** via WebGPU + XNNPACK.
- **Demo by:** Varun Chari.

---
© 2025 LiteRT.js Edge AI Demo — Powered by Arm