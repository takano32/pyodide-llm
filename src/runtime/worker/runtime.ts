// worker/runtime.ts (T378.3): what this visit's worker finds once and every load reads: the runtime (Pyodide, the
// engine's Python, the kernels, forward.ts), what the page's URL fixes for the visit, WebGPU and its adapter. This module alone
// sets these (init(), and the adapter as it comes): the others read the names, and ask for the runtime (runtimeAsked)
// or wait for it (runtimeReady).
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T367.3).

import { built } from "../built.ts";
import type { Built } from "../built.ts";
import { resolvePyodideVersion, pyodideSteps, placePython } from "./pyodide.ts";
import { since, took } from "./clock.ts";
import { wentWithout } from "./outdated.ts";
import { told } from "../failure.ts";
import type { EngineWindow, Pyodide, PyodideModule } from "../contracts/python.ts";
import type { Force } from "../contracts/gpu/protocol.ts";
import type * as Forward from "../forward.ts";
import type { Adapter, Compiled } from "../forward.ts";

// (T368.3: these two are there once the runtime is ready, which every load waits for: their types say that state.
// forward.ts may not come, no WebAssembly SIMD or its file gone: forwardModule is undefined then, and jsKernels null)
export let pyodide: Pyodide, llama2_numpy: EngineWindow;
// the WASM SIMD kernels that llama2_numpy.py loads with ctypes (the sampling, a conversion's quantizer and readers):
// their path in Pyodide's file system, or undefined
export let pythonKernels: string | undefined;
// T93: the forward pass in JavaScript (forward.ts) and its kernels, compiled once
export let forwardModule: typeof Forward | undefined, jsKernels: Compiled | null | undefined;
// T93 stage 2: the kernels for a shared memory (only where the page is cross-origin isolated)
export let sharedKernels: Compiled | null | undefined;
// T101: the kernels for a 64-bit memory, for a model past 4 GiB ({ plain, shared }), where the browser has Memory64
export let wideKernels: { plain: Compiled | null; shared: Compiled | null } | undefined;
// ?wide=on: a 64-bit memory for every model, to try that path on a small one (measuring, tests), as ?offline=on says
export let forceWide = false;
// ?gpuTest= (T148, T156; the tests only): what a fallback adapter is taken for (hasWebGpu's comment below)
export let gpuForce: Force = {};
// ?bench= (T45): the page measures the CPU's combinations, and no GPU starts beside them
export let benchPage = false;
// T107: ?hfParts=<MiB>&hfConnections=<N> fix the two, to measure; the page offers no way to them
const HF_CONNECTIONS = 6;
export let hfPartBytes = 0, hfConnections = HF_CONNECTIONS;  // 0: not fixed, decided per file from its first part
// when Pyodide became usable, to tell that first model from the ones chosen afterwards
export let pyodideAt = 0;
// T156: the adapter, asked for before a model is loaded ({ fallback, limits } or null): whether a model the device
// cannot hold twice goes on the GPU alone is decided before its bytes come (forward.ts's weightsPlace, gpuOnlyUnfit)
export let gpuAdapter: Adapter | null = null;
// init() as a promise: every load waits for it, also the one that replaces the first
let initialized: Promise<void> | undefined;

// T148: a prompt's tokens through the layers on the GPU (gpu.ts) by default, wherever this worker has WebGPU and
// forward.ts can put the model there, and the GPU is faster than the CPU here (AGENTS.md's policy 9: no option).
// ?gpuTest=on, for the tests only: a fallback adapter (SwiftShader, the only WebGPU of CI) taken as a GPU, and every
// block of a prompt it can take given to it (its speed is no GPU's: the tests look at its numbers, not at its time),
// and the first right shader of the matrices taken untimed (SwiftShader timed Llama 3.2 1B's past gpu.ts's 180 s)
export const hasWebGpu = Boolean(self.navigator?.gpu);
// (and its key, T148's: what the page kept of the model on this device holds only for the same, shaders.ts's deviceKey)
export const adapterAsked = !hasWebGpu ? Promise.resolve() : navigator.gpu.requestAdapter().then(async (adapter) => {
  if (!adapter) return;
  const { maxStorageBufferBindingSize, maxBufferSize } = adapter.limits;
  // (T232, packed: WGSL's packed int8 dot, which ternary weights are multiplied with on the GPU)
  gpuAdapter = { fallback: Boolean(adapter.info?.isFallbackAdapter ?? (adapter as GPUAdapter & { isFallbackAdapter?: boolean }).isFallbackAdapter),
    limits: { maxStorageBufferBindingSize, maxBufferSize },
    packed: Boolean(navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product")) };
  try {
    const wgsl = await import("../shaders/key.ts");
    gpuAdapter.key = wgsl.deviceKey(adapter);
    // (T232: a model of ternary weights has a key of its own, with its shaders: gpu.ts says that one)
    gpuAdapter.ternaryKey = wgsl.deviceKey(adapter, adapter, true);
  } catch (error) {
    // no key: nothing the page kept of this device holds (a model on the GPU alone is weighed again)
    console.warn(`gpu: no key of this device (${told(error)}): what the page kept of it is not used`);
  }
}).catch((error) => {
  // (T380.2: said. What was found of the adapter before it failed stays; with none, no model goes on the GPU)
  console.warn(`gpu: the adapter could not be asked for (${told(error)}): no GPU on this visit`);
});

async function init(search: string) {
  const started = performance.now();
  const asked = new URLSearchParams(search);
  const parts = Number(asked.get("hfParts")), connections = Number(asked.get("hfConnections"));
  if (parts >= 1 && parts <= 64) hfPartBytes = Math.round(parts * 1024 * 1024);
  if (connections >= 1 && connections <= 32) hfConnections = Math.floor(connections);
  forceWide = asked.get("wide") === "on";
  // (T156: ?gpuTest=only, the tests' too: a model the GPU can take on the GPU alone, whatever its size)
  gpuForce = (["on", "only"] as (string | null)[]).includes(asked.get("gpuTest")) ? { fallback: true, always: true, quick: true, only: asked.get("gpuTest") === "only" } : {};
  benchPage = asked.has("bench");
  // T367.2: the engine's Python (a window and its parts, T348) is one archive, engine.zip, of this worker's
  // build (built.ts). It is small and of this site, so it is asked for now, beside Pyodide, and is there when that is ready.
  const engine = built("engine.zip").then((res) => res.arrayBuffer());
  engine.catch(() => {});  // (it is awaited below: a load that ends before that leaves no unhandled rejection)
  const version = await resolvePyodideVersion(search);
  pyodide = await pyodideSteps(version, (url: string): Promise<PyodideModule> => import(/* @vite-ignore */ url));

  await placePython(pyodide, engine, "engine.zip");
  llama2_numpy = pyodide.pyimport("llama2_numpy");

  // The WASM SIMD kernels (kernels/*.ts), which llama2_numpy.py loads with ctypes. They are optional: without
  // them, or with ?kernel=off, NumPy does the math, several times slower. They are read even with ?kernel=off: the
  // switches say what is used (request.ts's switchesNow()), and the benchmark's rounds with the kernels need them there (T119: its
  // "everything" round sampled on NumPy after the page's switch had turned them off, and said nothing).
  for (const name of ["simdkernel.so", "simdkernel_relaxed.wasmlib"] satisfies Built[]) {
    // (also where it is gone, T367.4: NumPy answers, and T439: the page is told why it is slower)
    const kernel = await built(name).catch((error) => void wentWithout(error));
    if (kernel) {
      pyodide.FS.writeFile(`/home/pyodide/${name}`, new Uint8Array(await kernel.arrayBuffer()));
      pythonKernels = "/home/pyodide/simdkernel.so";
    }
  }
  // T93: the forward pass runs in forward.ts, on the plain build of the same kernels (simdkernel.so stays for the
  // sampling, which works on Python's logits). Without them (no WebAssembly SIMD) the engine runs NumPy.
  try {
    // (T439: a module's failure has no status: this module's own address is asked for, and nothing waits for it)
    forwardModule = await import("../forward.ts").catch((error) => {
      void wentWithout(error, import.meta.url);
      throw error;
    });
    // one build of the kernels: simdkernel_<kind>.wasm and simdkernel_relaxed_<kind>.wasm, or null
    const build = async (kind: "plain" | "shared" | "plain64" | "shared64", wide = false) => {
      const [plain, relaxed] = await Promise.all(([`simdkernel_${kind}.wasm`, `simdkernel_relaxed_${kind}.wasm`] satisfies Built[]).map((name) =>
        built(name).then((res) => res.arrayBuffer()).catch((error) => {
          // (T439: the kernels of a 64-bit memory are no slower path: without them a model past 4 GB is refused)
          if (!wide) void wentWithout(error);
          return null;
        })));
      return plain ? forwardModule!.compileKernels(plain, relaxed, wide) : null;
    };
    jsKernels = await build("plain");
    if (jsKernels && self.crossOriginIsolated) sharedKernels = await build("shared");
    // T101: on their own, so that a browser that says it makes 64-bit memories and then cannot compile their kernels
    // (Playwright's WebKit did, 2026-09-25) keeps the 32-bit ones: in the same try it lost every kernel and ran NumPy
    if (jsKernels && forwardModule.memory64()) {
      try {
        wideKernels = { plain: await build("plain64", true), shared: self.crossOriginIsolated ? await build("shared64", true) : null };
      } catch {
        wideKernels = undefined;
      }
    }
  } catch (error) {
    // (T380.2: said, whatever it was: no WebAssembly SIMD here, a kernel that did not compile, forward.ts that did not come)
    console.warn(`kernels: forward.js and its kernels did not start (${told(error)}): the engine runs NumPy`);
    jsKernels = null;
  }
  took("pyodide", since(started));
  pyodideAt = performance.now();
}

/** A load was asked for: the runtime starts with the first one (the model downloads while Pyodide loads), from what
 * the page's URL said, and is the same promise for every later one */
export function runtimeAsked(search: string) {
  initialized ??= init(search);
  return initialized;
}
/** The runtime as a promise, which a load waits for (undefined before any was asked for) */
export const runtimeReady = () => initialized;
