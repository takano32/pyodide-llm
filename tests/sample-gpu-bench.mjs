// T219: the GPU's sampling alone, the working tree's shaders against a git ref's (main), in turn in one process, in
// Node on Dawn (the npm package webgpu: tests/bench-dawn.mjs) with Mesa's lavapipe: whether a change to SAMPLE (T151)
// or to the sampling in chunks (T191's SAMPLER_STAGES) costs time, as /benchmark/'s "the sampling alone" measures
// them (Llama 3's vocabulary, logits as a model's and flat ones, the penalty 1: T191's review), before and after. The
// ms of one sampling is T168's difference of a submission of n and one of 2n (n from the working tree's shaders,
// the same for both), the median of `rounds` rounds taken in turn: old, new, old, new, ... On lavapipe the numbers are a
// CPU's, not any GPU's: read the ratio only (a device's numbers come from /benchmark/ on the device).
//   VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1) \
//   node tests/sample-gpu-bench.mjs <the webgpu package's directory> [--against <ref>] [--rounds 5]
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const args = process.argv.slice(2), OPTIONS = ["--against", "--rounds"];
const webgpu = args.find((a, i) => !a.startsWith("--") && !OPTIONS.includes(args[i - 1]));
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const against = option("--against", "main"), rounds = Number(option("--rounds", 5));
if (!webgpu) {
  console.error("node tests/sample-gpu-bench.mjs <the webgpu package's directory> [--against <ref>] [--rounds 5]");
  process.exit(2);
}
const root = new URL("../", import.meta.url).pathname;
const { create, globals } = await import(pathToFileURL(path.resolve(webgpu, "index.js")).href);
Object.assign(globalThis, globals);
const adapter = await create([]).requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice();
console.log(`adapter: ${adapter.info?.description ?? adapter.info?.vendor ?? "?"}${adapter.info?.isFallbackAdapter ? " (a fallback adapter: the numbers are a CPU's)" : ""}` +
  `, subgroups ${device.features.has("subgroups") ? "yes" : "no"}, against ${against}, ${rounds} rounds`);

// the two versions of shaders.js: the working tree's and the ref's, written under .tmp (shaders.js imports nothing)
const scratch = path.join(root, ".tmp", "sample-gpu-bench");
fs.mkdirSync(scratch, { recursive: true });
const old = path.join(scratch, `shaders-${against.replace(/[^\w.-]/g, "_")}.js`);
fs.writeFileSync(old, execFileSync("git", ["show", `${against}:public/shaders.js`], { cwd: root, maxBuffer: 1 << 26 }));
const versions = [{ name: against, wgsl: await import(pathToFileURL(old).href) }, { name: "working tree", wgsl: await import(pathToFileURL(path.join(root, "public", "shaders.js")).href) }];

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
  const module = device.createShaderModule({ code }), info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length) throw new Error(errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
  return device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
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
    // n from the working tree's shaders: a submission of 40 ms or more, up to N_MOST; the same n for both versions
    const fresh = versions[1].forms[form];
    await submission(fresh, 2);
    let n = 1;
    while ((await submission(fresh, n)) < SUBMISSION_MS && n < N_MOST) n *= 2;
    const differences = versions.map(() => []);
    for (let round = 0; round < rounds; round++) {
      for (let v = 0; v < versions.length; v++) {
        const once = await submission(versions[v].forms[form], n), twice = await submission(versions[v].forms[form], 2 * n);
        differences[v].push((twice - once) / n);
      }
    }
    const ms = differences.map(middle);
    rows.push({ logits: name, form, n, [versions[0].name]: ms[0], "working tree": ms[1], ratio: ms[1] / ms[0] });
  }
}
console.log(`ms a sampling (Llama 3's vocabulary ${VOCAB}, the difference of 2n and n over n, the median of ${rounds} rounds in turn):`);
console.log(`| logits | form | n | ${versions[0].name} | working tree | working tree / ${versions[0].name} |`);
console.log("|---|---|---:|---:|---:|---:|");
for (const row of rows) console.log(`| ${row.logits} | ${row.form} | ${row.n} | ${row[versions[0].name].toFixed(3)} | ${row["working tree"].toFixed(3)} | ${row.ratio.toFixed(3)} |`);
process.exit(0);
