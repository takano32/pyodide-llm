// tests/bench-info-check.ts (T301's review): what /benchmark/'s GPU worker answers its first step, "info" (src/runtime/benchmark/gpu/
// device.ts), where there is no GPU or a device is refused, on a WebGPU that writes down what it is asked and computes nothing
// (tests/fakes/webgpu.ts), in Node:
//
//   node tests/bench-info-check.ts
//
// T301 made the section tell "there is none here" from "it failed": no navigator.gpu in the worker and no adapter are answered
// (the section is "not in this browser"), and what fails past them (an adapter request that throws, a device that is refused)
// is the step's error (the section is "failed", and its warning says why). No CI machine has a device that refuses, and the
// worker runs nowhere else, so this is the one place the branches are walked. Each case is a process of its own (the steps
// share a module's state, made once).
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { runtimeUrl } from "./tree.ts";
import { standIn } from "./fakes/webgpu.ts";

const CASES = ["a device", "no adapter", "a device refused", "an adapter request that throws", "no navigator.gpu"];
if (process.argv[2] !== "case") {
  for (const one of CASES) {
    const ran = spawnSync(process.execPath, [import.meta.filename, "case", one], { encoding: "utf8", timeout: 120000 });
    assert.equal(ran.status, 0, `${one}:\n${ran.stdout}${ran.stderr}`);
  }
  console.log(`bench-info-check: the GPU step "info" and the section's state on ${CASES.length} cases: ok`);
  process.exit(0);
}
const kind = process.argv[3];

const { adapter } = standIn({ fallback: true });
const gpu = navigator.gpu;
if (kind === "no adapter") gpu.requestAdapter = async () => null;
if (kind === "an adapter request that throws") gpu.requestAdapter = async () => { throw new Error("NotAllowedError: no adapter for this page"); };
if (kind === "a device refused") adapter.requestDevice = async () => { throw new Error("OperationError: the device was refused"); };
if (kind === "no navigator.gpu") Object.defineProperty(globalThis, "navigator", { configurable: true, value: { userAgent: "bench-info-check" } });
globalThis.self = globalThis;
const posted = [];
globalThis.postMessage = (message) => posted.push(message);
let receiver;
Object.defineProperty(globalThis, "onmessage", { configurable: true, get: () => receiver, set: (handler) => { receiver = handler; } });
await import(runtimeUrl("benchmark/gpu.js"));
await receiver({ data: { step: "info" } });
const answer = posted.find((message) => message.step === "info");
assert.ok(answer, `${kind}: the step answered nothing`);

const { gpuNone, warnings } = await import("../src/bench.ts");
const step = { name: "the adapter", ...(answer.error ? { error: answer.error } : { result: answer.result }) };
const section = gpuNone(step);
if (kind === "a device") {
  assert.ok(!answer.error && answer.result.worker && answer.result.adapter, JSON.stringify(answer));
  assert.equal(section, undefined, "a device: the section runs");
} else if (kind === "no adapter") {
  assert.deepEqual(answer, { step: "info", result: { worker: true } });
  assert.deepEqual(section, { status: "none", markdown: "navigator.gpu gave no adapter" });
} else if (kind === "no navigator.gpu") {
  assert.deepEqual(answer, { step: "info", result: { worker: false } });
  assert.deepEqual(section, { status: "none", markdown: "no WebGPU in a worker here" });
} else {
  // what failed past the adapter's answer is the step's error, the section's failure and a warning of the report
  assert.ok(answer.error?.includes(kind === "a device refused" ? "the device was refused" : "no adapter for this page"), JSON.stringify(answer));
  assert.deepEqual(section, { status: "error", markdown: answer.error });
  assert.deepEqual(warnings([{ title: "GPU", ...section! }]), [`GPU: failed: ${answer.error}`]);
}
console.log(`bench-info-check: ${kind}: ok`);
