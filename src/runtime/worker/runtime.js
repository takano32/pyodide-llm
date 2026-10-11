// worker/runtime.js (T378.3): the runtime this visit's worker loads once: Pyodide, the engine's Python, the kernels,
// forward.js, and what the page's URL switches. (Moved here from worker.js as it was; the next commit makes this
// module the owner of what init() sets.)
// A module of worker.js; it takes its neighbours' names by static imports, and a bundler links them (T367.3).

import { built } from "../built.js";
import { state, loadSeconds } from "./state.js";
import { resolvePyodideVersion, pyodideSteps, placePython } from "./pyodide.js";
import { since } from "./clock.js";

// Only what the engine has a fallback for. A name it does not know is refused there, and the page says so.
function switchesOf(search) {
  const parameters = new URLSearchParams(search);
  const names = (parameters.get("without") ?? "").split(",").map((name) => name.trim()).filter(Boolean);
  if (parameters.get("kernel") === "off" && !names.includes("kernels")) {
    names.push("kernels");
  }
  return names;
}

export async function init(search) {
  const started = performance.now();
  const asked = new URLSearchParams(search);
  const parts = Number(asked.get("hfParts")), connections = Number(asked.get("hfConnections"));
  if (parts >= 1 && parts <= 64) state.hfPartBytes = Math.round(parts * 1024 * 1024);
  if (connections >= 1 && connections <= 32) state.hfConnections = Math.floor(connections);
  state.forceWide = asked.get("wide") === "on";
  // (T156: ?gpuTest=only, the tests' too: a model the GPU can take on the GPU alone, whatever its size)
  state.gpuForce = ["on", "only"].includes(asked.get("gpuTest")) ? { fallback: true, always: true, quick: true, only: asked.get("gpuTest") === "only" } : {};
  state.benchPage = asked.has("bench");
  // T367.2: the engine's Python (a window and its parts, T348) is one archive, engine.zip, of this worker's
  // build (built.js). It is small and of this site, so it is asked for now, beside Pyodide, and is there when that is ready.
  const engine = built("engine.zip").then((res) => res.arrayBuffer());
  engine.catch(() => {});  // (it is awaited below: a load that ends before that leaves no unhandled rejection)
  const version = await resolvePyodideVersion(search);
  state.pyodide = await pyodideSteps(version, (url) => import(/* @vite-ignore */ url));

  await placePython(state.pyodide, engine, "engine.zip");
  state.llama2_numpy = state.pyodide.pyimport("llama2_numpy");

  // The WASM SIMD kernels (kernels/*.ts), which llama2_numpy.py loads with ctypes. They are optional: without
  // them, or with ?kernel=off, NumPy does the math, several times slower. They are read even with ?kernel=off: the
  // switches say what is used (disabled), and the benchmark's rounds with the kernels need them there (T119: its
  // "everything" round sampled on NumPy after the page's switch had turned them off, and said nothing).
  state.disabled = state.pageSwitches = switchesOf(search);
  for (const name of ["simdkernel.so", "simdkernel_relaxed.wasmlib"]) {
    const kernel = await built(name).catch(() => undefined);  // (also where it is gone, T367.4: NumPy answers)
    if (kernel) {
      state.pyodide.FS.writeFile(`/home/pyodide/${name}`, new Uint8Array(await kernel.arrayBuffer()));
      state.kernels = "/home/pyodide/simdkernel.so";
    }
  }
  // T93: the forward pass runs in forward.js, on the plain build of the same kernels (simdkernel.so stays for the
  // sampling, which works on Python's logits). Without them (no WebAssembly SIMD) the engine runs NumPy.
  try {
    state.forwardModule = await import("../forward.js");
    // one build of the kernels: simdkernel_<kind>.wasm and simdkernel_relaxed_<kind>.wasm, or null
    const build = async (kind, wide = false) => {
      const [plain, relaxed] = await Promise.all([`simdkernel_${kind}.wasm`, `simdkernel_relaxed_${kind}.wasm`].map((name) =>
        built(name).then((res) => res.arrayBuffer()).catch(() => null)));
      return plain ? state.forwardModule.compileKernels(plain, relaxed, wide) : null;
    };
    state.jsKernels = await build("plain");
    if (state.jsKernels && self.crossOriginIsolated) state.sharedKernels = await build("shared");
    // T101: on their own, so that a browser that says it makes 64-bit memories and then cannot compile their kernels
    // (Playwright's WebKit did, 2026-09-25) keeps the 32-bit ones: in the same try it lost every kernel and ran NumPy
    if (state.jsKernels && state.forwardModule.memory64()) {
      try {
        state.wideKernels = { plain: await build("plain64", true), shared: self.crossOriginIsolated ? await build("shared64", true) : null };
      } catch {
        state.wideKernels = undefined;
      }
    }
  } catch {
    state.jsKernels = null;
  }
  loadSeconds.pyodide = since(started);
  state.pyodideAt = performance.now();
}
