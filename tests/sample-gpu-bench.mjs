// T219: the GPU's sampling alone, the working tree's shaders against a git ref's (main), in turn in one process, in
// Node on Dawn (the npm package webgpu: tests/bench-dawn.mjs) with Mesa's lavapipe: whether a change to SAMPLE (T151)
// or to the sampling in chunks (T191's SAMPLER_STAGES) costs time, as /benchmark/'s "the sampling alone" measures
// them (Llama 3's vocabulary, logits as a model's and flat ones, the penalty 1: T191's review), before and after. The
// ms of one sampling is T168's difference of a submission of n and one of 2n (n from the working tree's shaders,
// the same for both), the median of `rounds` rounds, the versions in a random order each round. On lavapipe the numbers
// are a CPU's, not any GPU's.
// T219's review: read a ratio against its noise. The versions were first timed in a fixed order, and lavapipe gave the
// same shader one speed in one place of the round and another in the next (a store under a condition no thread meets
// took 0.67 to 0.77 of the time of the shader it was made from; the flag of T219 read 1.59 on one EPYC 7763 and 0.73 on
// another run's, the same shaders), so the order is random now, and the working tree's text is timed twice: its
// ratio to itself is what the method cannot tell apart, and a ratio of the tree to the ref means something only far
// from it.
//   VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1) \
//   node tests/sample-gpu-bench.mjs <the webgpu package's directory> [--against <ref>] [--rounds 9] [--seed 1]
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { otherTree } from "./other-tree.mjs";

const args = process.argv.slice(2), OPTIONS = ["--against", "--rounds", "--seed"];
const webgpu = args.find((a, i) => !a.startsWith("--") && !OPTIONS.includes(args[i - 1]));
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const against = option("--against", "main"), rounds = Number(option("--rounds", 9));
// a seeded shuffle (the same orders again for the same seed)
let seed = Number(option("--seed", 1)) >>> 0;
const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
const shuffled = (list) => {
  const out = [...list];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
};
if (!webgpu) {
  console.error("node tests/sample-gpu-bench.mjs <the webgpu package's directory> [--against <ref>] [--rounds 9] [--seed 1]");
  process.exit(2);
}
const root = new URL("../", import.meta.url).pathname;
// the two versions of the shaders: the working tree's and the ref's. (T351: shaders.js is a window over public/shaders/, so
// a version is a tree's public/shaders.js with what it imports beside it, never the one file: the ref's whole tree is
// tests/other-tree.mjs's, which holds either form)
const scratch = path.join(root, ".tmp", "sample-gpu-bench");
fs.mkdirSync(scratch, { recursive: true });
const old = path.join(otherTree(against).folder, "public", "shaders.js");
// (the working tree's text again, a copy: a module is imported once by its address, and so are the modules it asks for)
const again = path.join(scratch, "working-tree-again");
fs.rmSync(again, { recursive: true, force: true });
fs.mkdirSync(again, { recursive: true });
fs.copyFileSync(path.join(root, "public", "shaders.js"), path.join(again, "shaders.js"));
fs.cpSync(path.join(root, "public", "shaders"), path.join(again, "shaders"), { recursive: true });
const versions = [{ name: against, wgsl: await import(pathToFileURL(old).href) }, { name: "working tree", wgsl: await import(pathToFileURL(path.join(root, "public", "shaders.js")).href) },
  { name: "working tree again", wgsl: await import(pathToFileURL(path.join(again, "shaders.js")).href) }];
// (three versions of the text, each its own module: the ref's is not the tree's, and the copy is not the tree's object)
if (versions[0].wgsl === versions[1].wgsl || versions[1].wgsl === versions[2].wgsl || versions[1].wgsl.SAMPLE !== versions[2].wgsl.SAMPLE) {
  throw new Error("the versions of the shaders are not three modules, or the copy of the working tree's is another text");
}

const { create, globals } = await import(pathToFileURL(path.resolve(webgpu, "index.js")).href);
Object.assign(globalThis, globals);
// The Dawn instance is kept (navigator.gpu, as the benchmark's worker finds it): one made for the one call
// (create([]).requestAdapter()) was collected while its device lived, and Dawn ended in a bus error or a system_error
// at the first asynchronous call after (a pipeline made asynchronously, an error scope popped), at random
Object.defineProperty(globalThis, "navigator", { value: { gpu: create([]) }, configurable: true });
const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice({ requiredFeatures: ["shader-f16", "subgroups"].filter((name) => adapter.features.has(name)),
  requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
console.log(`adapter: ${adapter.info?.description ?? adapter.info?.vendor ?? "?"}${adapter.info?.isFallbackAdapter ? " (a fallback adapter: the numbers are a CPU's)" : ""}` +
  `, subgroups ${device.features.has("subgroups") ? "yes" : "no"}, against ${against}, ${rounds} rounds`);

const VOCAB = 128256, MOST = 128, SETTINGS = { temperature: 0.7, topp: 0.9, penalty: 1 }, SUBMISSION_MS = 40, N_MOST = 256;
const { STORAGE, COPY_DST, UNIFORM } = GPUBufferUsage;
const buffer = (size, usage = STORAGE | COPY_DST) => device.createBuffer({ size, usage });
// the sampler's buffers by SAMPLE's binding numbers, and 7 the chunks' partial results (the larger of the versions')
const partsBytes = Math.max(...versions.map(({ wgsl }) => wgsl.samplePartsBytes(VOCAB)));
const b = { 0: buffer(VOCAB * 4), 1: buffer(VOCAB * 4), 2: buffer(VOCAB * 4), 3: buffer(versions[1].wgsl.STATE_BYTES), 4: buffer(MOST * 4),
  5: buffer(MOST * 4), 6: buffer(versions[1].wgsl.SAMPLING_BYTES, UNIFORM | COPY_DST), 7: buffer(partsBytes) };
device.queue.writeBuffer(b[5], 0, Float32Array.from({ length: MOST }, () => Math.random()));
device.queue.writeBuffer(b[6], 0, versions[1].wgsl.samplingSettings({ vocab: VOCAB, ...SETTINGS }));
const history = [...Array(64)].map(() => (Math.random() * VOCAB) | 0), start = versions[1].wgsl.samplingState({ token: history[63], pos: 0, history });

const pipeline = async (code) => {
  // (synchronously, under a validation scope: createComputePipelineAsync ended Dawn in Node here with a system_error,
  // and a child process run after Dawn started ended it with a bus error: the ref is read above, before)
  device.pushErrorScope("validation");
  const made = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const invalid = await device.popErrorScope();
  if (invalid) throw new Error(invalid.message);
  return made;
};
const group = (pipe, bindings) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: bindings.map((binding) => ({ binding, resource: { buffer: b[binding] } })) });
// each version's two forms as lists of [pipeline, bind group, workgroups]
for (const version of versions) {
  const { wgsl } = version, one = await pipeline(wgsl.SAMPLE), chunks = wgsl.sampleChunks(VOCAB);
  version.forms = { one: [[one, group(one, [0, 1, 2, 3, 4, 5, 6]), 1]], chunks: [] };
  for (const stage of wgsl.SAMPLER_STAGES) {
    const pipe = await pipeline(stage.code);
    version.forms.chunks.push([pipe, group(pipe, stage.bindings), stage.chunks ? chunks : 1]);
  }
}
const submission = async (list, n) => {
  device.queue.writeBuffer(b[3], 0, start);
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  for (let i = 0; i < n; i++) {
    for (const [pipe, bindGroup, workgroups] of list) {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
    }
  }
  pass.end();
  const began = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - began;
};
// logits as a model's (a spread of 2 and 20 peaks 8 to 14 over it) and flat ones (a spread of 1, every token over the floor)
const madeUp = (spread, peaks) => {
  const logits = new Float32Array(VOCAB);
  for (let i = 0; i < VOCAB; i++) logits[i] = spread * Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  for (let i = 0; i < peaks; i++) logits[(Math.random() * VOCAB) | 0] += 8 + 6 * Math.random();
  return logits;
};
const middle = (values) => [...values].sort((x, y) => x - y)[values.length >> 1];
const rows = [];
for (const [name, logits] of [["as a model's", madeUp(2, 20)], ["flat", madeUp(1, 0)]]) {
  device.queue.writeBuffer(b[0], 0, logits);
  for (const form of ["one", "chunks"]) {
    // each version warmed (a fallback adapter compiles on the first submission), then n from the working tree's
    // shaders: a submission of 40 ms or more, up to N_MOST; the same n for both versions
    for (const version of versions) await submission(version.forms[form], 2);
    const fresh = versions[1].forms[form];
    let n = 1;
    while ((await submission(fresh, n)) < SUBMISSION_MS && n < N_MOST) n *= 2;
    const differences = versions.map(() => []);
    for (let round = 0; round < rounds; round++) {
      for (const v of shuffled(versions.map((_, i) => i))) {
        const once = await submission(versions[v].forms[form], n), twice = await submission(versions[v].forms[form], 2 * n);
        differences[v].push((twice - once) / n);
      }
    }
    const ms = differences.map(middle);
    rows.push({ logits: name, form, n, ms, ratio: ms[1] / ms[0], noise: ms[2] / ms[1] });
  }
}
console.log(`ms a sampling (Llama 3's vocabulary ${VOCAB}, the difference of 2n and n over n, the median of ${rounds} rounds, the versions in a random order each round):`);
console.log(`| logits | form | n | ${versions[0].name} | working tree | working tree again | working tree / ${versions[0].name} | again / working tree (the noise) |`);
console.log("|---|---|---:|---:|---:|---:|---:|---:|");
for (const row of rows) console.log(`| ${row.logits} | ${row.form} | ${row.n} | ${row.ms.map((x) => x.toFixed(3)).join(" | ")} | ${row.ratio.toFixed(3)} | ${row.noise.toFixed(3)} |`);
process.exit(0);
