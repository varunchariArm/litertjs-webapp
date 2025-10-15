// public/js/runtime.js
import { loadLiteRt } from "@litertjs/core";
import { simd, threads } from 'wasm-feature-detect';

const FLAG = "__litert_loaded__";

/** Ensure LiteRT WASM assets are loaded exactly once per page. */
export async function ensureLiteRtOnce(wasmPath = "/wasm/") {
  const g = (globalThis[FLAG] ||= { p: null, ok: false });
  if (g.ok) return;
  if (!g.p) {
    g.p = loadLiteRt(wasmPath)
      .catch((e) => {
        const msg = String(e?.message || e);
        if (/already loading|already loaded/i.test(msg)) return; // benign
        throw e;
      })
      .then(async () => { 
        const [hasSIMD, hasThreads] = await Promise.all([simd(), threads()]);
        console.log({ hasSIMD, hasThreads, crossOriginIsolated });
        g.ok = true; 
      });
  }
  await g.p;
}