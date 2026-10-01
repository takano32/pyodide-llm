// What the fake workers answer: the fixtures of tests/bench.mjs, and the failures to try
import { scenario, F } from "./page-harness.mjs";

const { aCpu, aCheck, aBandwidths, aTokens, aLayer, aGenerate, aCeilings, aPrompt, stepsStep, aStorage, aLine, real } = F;

export const base = () => ({
  device: { cores: 16, memoryGB: 8, simd: true, relaxedSimd: true, memory64: true, crossOriginIsolated: true, sharedMemory: true, webgpu: true, opfs: true, syncHandle: true, usage: 1e8, quota: 1e10 },
  cpu: structuredClone(aCpu),
  info: { worker: true, adapter: "nvidia · turing", fallback: false, packed: true, features: ["shader-f16", "subgroups"], subgroupSizes: [32, 32],
    maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 32, maxComputeWorkgroupStorageSize: 49152, maxComputeInvocationsPerWorkgroup: 1024 },
  check: structuredClone(aCheck),
  steps: {},          // a step's name -> an error (the step answers {error})
  lostAfter: undefined, // the device is lost when this step is asked
  storage: { mib: 64, pieces: 8, openSeconds: 0.0004, secondHandle: "refused (NoModificationAllowedError)", reopen: "opened", sequential: { seconds: 0.33, flushSeconds: 0.31 },
    scattered: { seconds: 0.24, flushSeconds: 0.23 }, scatteredFlushEach: { seconds: 0.24, flushSeconds: 0.23 }, read: { seconds: 0.01, wrong: 0 } },
  line: structuredClone(aLine),
  paths: { threads: 4, how: { remembered: true }, gpu: { seconds: 4.2, matrices: "TF.js 64×64", attention: "llama.cpp flash" }, status: "prompts and answers on WebGPU", rows: structuredClone(real.rows) },
  rounds: [{ name: "everything", without: [], tokens: 64, speed: 334.6, backend: "SIMD kernels, int8, relaxed SIMD", seconds: 8.4 },
    { name: "without the kernels", without: ["kernels"], tokens: 64, speed: 44.8, backend: "NumPy (without kernels)", seconds: 8.1 }],
  stall: undefined,   // a section name that never answers
});

export function use(s) {
  scenario.answer = async (worker, message) => {
    const file = worker.url.replace(/\?.*$/, "").split("/pyodide-llm/")[1];
    const fail = (name) => (s.steps[name] !== undefined ? { error: s.steps[name] } : undefined);
    if (file === "worker.js") {
      if (message.type === "init") {
        worker.emit({ type: "status", text: "Downloading" });
        worker.emit({ type: "ready", load: message.load });
      } else if (message.type === "paths") worker.emit({ type: "paths", load: message.load, ...s.paths });
      else if (message.type === "bench") worker.emit({ type: "bench", load: message.load, rows: s.rounds, pyodide: "314.0.7" });
      return;
    }
    if (file === "benchmark/waiter.js") return worker.emit({ microseconds: 24.3 });
    if (file === "benchmark/storage.js") return worker.emit(message.step === "info" ? { result: { syncHandle: true } } : fail("storage") ?? { result: s.storage });
    if (file === "benchmark/sections.js") {
      if (message.step === "device") return worker.emit(fail("device") ?? { result: s.device });
      if (message.step === "cpu") return worker.emit(fail("cpu") ?? { result: s.cpu });
      if (message.step === "line") return worker.emit(fail("line") ?? { result: s.line });
    }
    if (file === "benchmark/gpu.js") {
      if (s.stall === "gpu" && message.step === "check") return;  // never answers
      const step = message.step === "bandwidth" ? `bandwidth: ${Object.keys({ "llm-jp-3 150M w1": 0, "Llama 3.2 1B w1": 0, "Llama 3.2 1B classifier": 0 }).find((name) => JSON.stringify(message.shape) === JSON.stringify({ "llm-jp-3 150M w1": [2048, 512], "Llama 3.2 1B w1": [8192, 2048], "Llama 3.2 1B classifier": [128256, 2048] }[name]))}`
        : message.step === "token" ? `a token of ${message.model}${message.kind === "packed" ? ", packed int8" : ""}${message.sample ? ", chosen on the GPU" : ""}`
        : { info: "the adapter", check: "the shaders against JavaScript", layer: "a layer of a token", "layer steps": "the steps of a layer", generate: "tokens generated on the GPU",
          overhead: "what a token costs besides the weights", ceilings: "the device's ceilings", prompt: "a prompt all at once", bridge: "bridge" }[message.step];
      if (s.lostAfter === step) worker.emit({ lost: "destroyed: Device was destroyed." });
      const failed = s.steps[step] !== undefined ? { error: s.steps[step] } : undefined;
      if (failed) return worker.emit(failed);
      switch (message.step) {
        case "info": return worker.emit({ result: s.info });
        case "check": return worker.emit({ result: s.check });
        case "bandwidth": return worker.emit({ result: aBandwidths.find((b) => b.name === step)?.result ?? aBandwidths[0].result });
        case "token": {
          const hit = aTokens.find((t) => t.name === step) ?? aTokens[0];
          return worker.emit({ result: hit.result });
        }
        case "layer": return worker.emit({ result: aLayer.result });
        case "layer steps": return worker.emit({ result: stepsStep.result });
        case "generate": return worker.emit({ result: aGenerate.result });
        case "overhead": return worker.emit({ result: { dispatches: 100, emptyDispatches: 1.2, submitOnly: 0.1, submitAndWait: 0.5, readToken: 0.3, readLogits: 0.9, vocab: 128256 } });
        case "ceilings": return worker.emit({ result: aCeilings });
        case "prompt": return worker.emit({ result: aPrompt.result });
        case "bridge": return worker.emit({ result: { waitAsync: true } });
      }
    }
    throw new Error(`no answer for ${file} ${JSON.stringify(message).slice(0, 80)}`);
  };
}
