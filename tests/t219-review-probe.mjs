// The review of T219 (2), a throwaway (branch t219-review-probe, never on main): where lavapipe's 1.6 times of SAMPLE
// comes from. Variants of the working tree's shaders.js, made here by replacing text, timed in turn with main's in one
// process on Dawn and lavapipe, as tests/sample-gpu-bench.mjs times two:
//   main      origin/main's shaders.js (no flag)
//   branch    the working tree's (the flag: the magnitude's max a logit, any_of, a uniform early return)
//   noMag     the branch without the magnitude's max in the loops (the ops a logit gone; the flag of +inf and all -inf stays)
//   noReturn  the branch with the flag set by thread 0 and no return: finish() writes nothing where not_finite is set
//   noAny     noReturn without any_of: every thread's magnitude by atomicMax before best_of's barriers, thread 0 decides
// --write <dir>: only write the variants' files there (for the check to run on each).
//   node tests/t219-review-probe.mjs <the webgpu package's directory> [--rounds 9] [--write <dir>]
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const args = process.argv.slice(2), OPTIONS = ["--rounds", "--write"];
const webgpu = args.find((a, i) => !a.startsWith("--") && !OPTIONS.includes(args[i - 1]));
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const rounds = Number(option("--rounds", 9)), writeTo = option("--write", null);
const root = new URL("../", import.meta.url).pathname;
const scratch = writeTo ? path.resolve(writeTo) : path.join(root, ".tmp", "t219-review-probe");
fs.mkdirSync(scratch, { recursive: true });

const branch = fs.readFileSync(path.join(root, "public", "shaders.js"), "utf8");
let mainText;
try {
  mainText = execFileSync("git", ["show", "origin/main:public/shaders.js"], { cwd: root, maxBuffer: 1 << 26, stdio: ["ignore", "pipe", "ignore"] }).toString();
} catch {
  execFileSync("git", ["fetch", "--depth", "1", "origin", "main"], { cwd: root, stdio: "ignore" });
  mainText = execFileSync("git", ["show", "FETCH_HEAD:public/shaders.js"], { cwd: root, maxBuffer: 1 << 26 }).toString();
}

// a replacement that must hit exactly `times` times within the text between `from` and the next export after it
const within = (text, from, find, put, times = 1) => {
  const start = text.indexOf(from);
  if (start < 0) throw new Error(`no ${from}`);
  const next = text.indexOf("\nexport ", start + from.length);
  const end = next < 0 ? text.length : next;
  const part = text.slice(start, end), count = part.split(find).length - 1;
  if (count !== times) throw new Error(`${JSON.stringify(find.slice(0, 60))} in ${from}: ${count} times, not ${times}`);
  return text.slice(0, start) + part.split(find).join(put) + text.slice(end);
};
const MAG = "        magnitude = max(magnitude, bitcast<u32>(v) & FLOAT_MAGNITUDE);\n";
const FINISH = "fn finish(token: u32) {\n";
const REFUSING = "fn finish(token: u32) {\n    if (state.not_finite != 0u) {\n        return;\n    }\n";
const SAMPLE_BLOCK = `    if (any_of(is_nan_magnitude(magnitude) || bitcast<u32>(best) == INFINITY_BITS || argmax == NONE, t)) {
        if (t == 0u) {
            state.not_finite = 1u;
            state.stopped = 1u;
        }
        return;
    }
`;
const PICK_BLOCK = `    if (any_of(bad || bitcast<u32>(top.value) == INFINITY_BITS || argmax == NONE, t)) {
        if (t == 0u) {
            state.not_finite = 1u;
            state.stopped = 1u;
        }
        return;
    }
`;
const noReturn = (text) => {
  let s = within(text, "const SAMPLER_DRAW", FINISH, REFUSING);
  s = within(s, "export const SAMPLE = ", SAMPLE_BLOCK, SAMPLE_BLOCK.replace("        return;\n", ""));
  return within(s, "export const SAMPLE_PICK", PICK_BLOCK, PICK_BLOCK.replace("        return;\n", ""));
};
const noAny = (text) => {
  let s = within(text, "const SAMPLER_DRAW", FINISH, REFUSING);
  // SAMPLE: reset with found, every thread's magnitude before best_of, thread 0 decides
  s = within(s, "export const SAMPLE = ", "        atomicStore(&found, NONE);\n", "        atomicStore(&found, NONE);\n        atomicStore(&any_word, 0u);\n");
  s = within(s, "export const SAMPLE = ", "    let largest = best_of(value, at, t);\n", "    atomicMax(&any_word, magnitude);\n    let largest = best_of(value, at, t);\n");
  s = within(s, "export const SAMPLE = ", SAMPLE_BLOCK,
    "    if (t == 0u && (is_nan_magnitude(atomicLoad(&any_word)) || bitcast<u32>(best) == INFINITY_BITS || argmax == NONE)) {\n" +
    "        state.not_finite = 1u;\n        state.stopped = 1u;\n    }\n");
  // SAMPLE_MAX
  s = within(s, "export const SAMPLE_MAX", "    let t = lid.x;\n    if (stopped(t)) {\n", "    let t = lid.x;\n    if (t == 0u) {\n        atomicStore(&any_word, 0u);\n    }\n    if (stopped(t)) {\n");
  s = within(s, "export const SAMPLE_MAX", "    let best = best_of(value, at, t);\n", "    atomicMax(&any_word, magnitude);\n    let best = best_of(value, at, t);\n");
  s = within(s, "export const SAMPLE_MAX", "    let chunk_bad = any_of(is_nan_magnitude(magnitude), t);\n", "    let chunk_bad = is_nan_magnitude(atomicLoad(&any_word));\n");
  // SAMPLE_PICK: the chunks' flags folded before largest()'s barriers, thread 0 decides
  s = within(s, "export const SAMPLE_PICK", "        atomicStore(&found_chunk, NONE);\n", "        atomicStore(&found_chunk, NONE);\n        atomicStore(&any_word, 0u);\n");
  s = within(s, "export const SAMPLE_PICK", "    let top = largest(t);\n    let argmax = top.at;\n", "");
  s = within(s, "export const SAMPLE_PICK", PICK_BLOCK,
    "    atomicMax(&any_word, u32(bad));\n    let top = largest(t);\n    let argmax = top.at;\n" +
    "    if (t == 0u && (atomicLoad(&any_word) != 0u || bitcast<u32>(top.value) == INFINITY_BITS || argmax == NONE)) {\n" +
    "        state.not_finite = 1u;\n        state.stopped = 1u;\n    }\n");
  return s;
};
const texts = {
  main: mainText,
  branch,
  noMag: within(within(branch, "export const SAMPLE = ", MAG, ""), "export const SAMPLE_MAX", MAG, ""),
  noReturn: noReturn(branch),
  noAny: noAny(branch),
  // (checked, not timed) a NaN seen by the signed bits: one with the sign set (x86's default NaN) goes unseen
  signedMag: within(within(branch, "export const SAMPLE = ", MAG, MAG.replace("bitcast<u32>(v) & FLOAT_MAGNITUDE", "bitcast<u32>(max(bitcast<i32>(v), 0))")),
    "export const SAMPLE_MAX", MAG, MAG.replace("bitcast<u32>(v) & FLOAT_MAGNITUDE", "bitcast<u32>(max(bitcast<i32>(v), 0))")),
};
const TIMED = ["main", "branch", "noMag", "noReturn", "noAny"];
for (const [name, text] of Object.entries(texts)) fs.writeFileSync(path.join(scratch, `${name}.js`), text);
if (writeTo) {
  console.log(`wrote ${Object.keys(texts).join(", ")} into ${scratch}`);
  process.exit(0);
}
if (!webgpu) {
  console.error("node tests/t219-review-probe.mjs <the webgpu package's directory> [--rounds 9] [--write <dir>]");
  process.exit(2);
}
const versions = [];
for (const name of TIMED) versions.push({ name, wgsl: await import(pathToFileURL(path.join(scratch, `${name}.js`)).href) });

const { create, globals } = await import(pathToFileURL(path.resolve(webgpu, "index.js")).href);
Object.assign(globalThis, globals);
Object.defineProperty(globalThis, "navigator", { value: { gpu: create([]) }, configurable: true });
const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice({ requiredFeatures: ["shader-f16", "subgroups"].filter((name) => adapter.features.has(name)),
  requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
console.log(`adapter: ${adapter.info?.description ?? "?"}, ${rounds} rounds, LP_NATIVE_VECTOR_WIDTH ${process.env.LP_NATIVE_VECTOR_WIDTH ?? "(default)"}`);

const VOCAB = 128256, MOST = 128, SETTINGS = { temperature: 0.7, topp: 0.9, penalty: 1 }, SUBMISSION_MS = 40, N_MOST = 256;
const { STORAGE, COPY_DST, UNIFORM } = GPUBufferUsage;
const buffer = (size, usage = STORAGE | COPY_DST) => device.createBuffer({ size, usage });
const newest = versions[1].wgsl;
const partsBytes = Math.max(...versions.map(({ wgsl }) => wgsl.samplePartsBytes(VOCAB)));
const b = { 0: buffer(VOCAB * 4), 1: buffer(VOCAB * 4), 2: buffer(VOCAB * 4), 3: buffer(newest.STATE_BYTES), 4: buffer(MOST * 4),
  5: buffer(MOST * 4), 6: buffer(newest.SAMPLING_BYTES, UNIFORM | COPY_DST), 7: buffer(partsBytes) };
device.queue.writeBuffer(b[5], 0, Float32Array.from({ length: MOST }, () => Math.random()));
device.queue.writeBuffer(b[6], 0, newest.samplingSettings({ vocab: VOCAB, ...SETTINGS }));
const history = [...Array(64)].map(() => (Math.random() * VOCAB) | 0), start = newest.samplingState({ token: history[63], pos: 0, history });
const pipeline = async (code) => {
  device.pushErrorScope("validation");
  const made = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const invalid = await device.popErrorScope();
  if (invalid) throw new Error(invalid.message);
  return made;
};
const group = (pipe, bindings) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: bindings.map((binding) => ({ binding, resource: { buffer: b[binding] } })) });
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
const madeUp = (spread, peaks) => {
  const logits = new Float32Array(VOCAB);
  for (let i = 0; i < VOCAB; i++) logits[i] = spread * Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  for (let i = 0; i < peaks; i++) logits[(Math.random() * VOCAB) | 0] += 8 + 6 * Math.random();
  return logits;
};
const middle = (values) => [...values].sort((x, y) => x - y)[values.length >> 1];
console.log(`ms a sampling (vocabulary ${VOCAB}, (2n - n) / n, the median of ${rounds} rounds, the versions in turn), and / main:`);
console.log(`| logits | form | n | ${versions.map((v) => v.name).join(" | ")} | ${versions.slice(1).map((v) => `${v.name} / main`).join(" | ")} |`);
console.log(`|---|---|---:|${versions.map(() => "---:").join("|")}|${versions.slice(1).map(() => "---:").join("|")}|`);
for (const [name, logits] of [["as a model's", madeUp(2, 20)], ["flat", madeUp(1, 0)]]) {
  device.queue.writeBuffer(b[0], 0, logits);
  for (const form of ["one", "chunks"]) {
    for (const version of versions) await submission(version.forms[form], 2);
    let n = 2;
    while ((await submission(versions[1].forms[form], n)) < SUBMISSION_MS && n < N_MOST) n *= 2;
    const differences = versions.map(() => []);
    for (let round = 0; round < rounds; round++) {
      for (let v = 0; v < versions.length; v++) {
        const once = await submission(versions[v].forms[form], n), twice = await submission(versions[v].forms[form], 2 * n);
        differences[v].push((twice - once) / n);
      }
    }
    const ms = differences.map(middle);
    console.log(`| ${name} | ${form} | ${n} | ${ms.map((x) => x.toFixed(3)).join(" | ")} | ${ms.slice(1).map((x) => (x / ms[0]).toFixed(3)).join(" | ")} |`);
  }
}
process.exit(0);
