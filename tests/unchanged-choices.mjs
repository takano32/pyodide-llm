// unchanged-choices.mjs (T346 review)
// What forward.js decides without a browser, over a grid of made-up numbers, as one JSON object: the GPU or the CPU for a
// block and a step (promptTimes, tokenTimes: BETTER, KEEP, TIMED), where the weights go (weightsPlace, gpuOnlyUnfit,
// aloneVerdict), what the memory must hold (footprint, keysInHalf, gpuBytes, needsWide, automaticDtype, weightsMemory),
// the status line (gpuLine). calls only runs the kernels for 12 shapes at one thread; the constants that choose a device
// are in none of it (the review: BETTER 0.95 -> 0.96 and KEEP 5 -> 6 passed every other check).
//   node tests/unchanged-choices.mjs <the root of a tree>
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = path.resolve(process.argv[2] ?? ".");
const f = await import(pathToFileURL(path.join(root, "public/forward.js")));
const found = {};
const attempt = (name, make) => { try { found[name] = JSON.stringify(make()) ?? "undefined"; } catch (error) { found[name] = `throws ${error.message}`; } };

// a block: the CPU's ms a token over a few blocks, the GPU's line and its blocks, the ratio of the last
for (const cpuMs of [0.5, 1, 2, 4]) for (const ratio of [0.5, 0.94, 0.95, 0.96, 1, 1.5]) for (const blocks of [1, 2, 3, 5, 6, 8]) {
  const times = f.promptTimes();
  times.started([{ count: 16, ms: 20 }, { count: 64, ms: 40 }]);
  for (let i = 0; i < blocks; i++) { times.cpu(4, cpuMs * (1 + i / 7)); times.gpu(64, ratio * (40) * (1 + (i % 3) / 9)); }
  attempt(`prompt ${cpuMs} ${ratio} ${blocks}`, () => [times.of(16, 4), times.of(64, 4), times.of(64, 2), times.threshold(64, 4), times.threshold(16, 4)]);
}
for (const cpuMs of [4, 10, 20]) for (const gpuMs of [3, 9.5, 10, 10.5, 19, 20, 25]) for (const count of [1, 2, 3, 5, 6, 9]) {
  const times = f.tokenTimes();
  for (let i = 0; i < count; i++) { times.cpu(2, cpuMs * (1 + i / 11)); times.gpu(gpuMs * (1 + (i % 2) / 13)); }
  attempt(`token ${cpuMs} ${gpuMs} ${count}`, () => [times.of(2), times.of(1)]);
}
const GiB = 2 ** 30;
for (const deviceMemory of [2, 4, 8, 16]) for (const cpu of [0.5, 1.5, 3, 6.4, 6.6].map((n) => n * GiB)) for (const gpu of [0.5, 2, 3.5].map((n) => n * GiB)) for (const eligible of [false, true]) for (const forced of [false, true]) {
  attempt(`place ${deviceMemory} ${cpu} ${gpu} ${eligible} ${forced}`, () => f.weightsPlace({ cpu, gpuOnly: gpu - 0.25 * GiB, gpu, deviceMemory, eligible, forced }));
}
for (const cpu of [{}, { GBps: 12 }, { GBps: 12, promptGMACs: 30 }, { promptGMACs: 30 }]) for (const gpu of [{}, { stepMs: 20 }, { stepMs: 20, promptMs: 3 }, { promptMs: 3 }])
  for (const usage of [undefined, { prompt: 10, written: 1 }, { prompt: 1, written: 10 }]) for (const size of [0.5 * GiB, 4 * GiB])
    attempt(`alone ${JSON.stringify([cpu, gpu, usage, size])}`, () => f.aloneVerdict({ size, layerWeights: size * 0.9, cpu, gpu, usage }));
const headers = [[256, 512, 4, 8, 8, -2000, 300], [2048, 8192, 16, 32, 8, 128256, 4096], [3072, 8192, 28, 24, 8, 128256, 4096], [4096, 14336, 32, 32, 8, 128256, 4096],
  [1024, 3072, 28, 16, 8, 151936, 4096], [768, 3072, 12, 12, 12, 50257, 1024], [5120, 17408, 64, 24, 4, 248320, 4096]];
const adapters = [null, { fallback: true, packed: true, limits: { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 32 } },
  { packed: true, limits: { maxStorageBufferBindingSize: 2 ** 27, maxBufferSize: 2 ** 28 } }, { packed: false, limits: { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 32 } },
  { packed: true, limits: { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 32 } }];
for (const header of headers) {
  for (const dtype of ["int8", "int6", "float32", "float16", "ternary"]) for (const arch of ["llama", "gpt2", "neox", "qwen35", "lfm2"]) for (const halfKV of [false, true]) {
    for (const relaxed of [false, true]) for (const shared of [false, true]) {
      const options = { dtype, arch, halfKV, relaxed, shared, int8: dtype === "int8", head_dim: arch === "llama" && header[0] === 1024 ? 128 : 0 };
      attempt(`footprint ${JSON.stringify([header, options])}`, () => [f.footprint(header, 1e9, options), f.keysInHalf(header, 1e9, options)]);
    }
    attempt(`gpuBytes ${JSON.stringify([header, dtype, arch])}`, () => f.gpuBytes(header, { dtype, arch }));
    for (const head_dim of [0, 20, 36]) adapters.forEach((adapter, i) => attempt(`gpuOnlyUnfit ${JSON.stringify([header, dtype, arch, i, head_dim])}`, () => f.gpuOnlyUnfit(header, dtype, { arch, head_dim }, adapter)));
  }
}
for (const size of [1e9, 2.5e9, 3.9e9, 4.1e9, 8e9, 17e9]) for (const after of [1e8, 3e8, 2e9, 1e10]) {
  attempt(`wide ${size} ${after}`, () => [f.needsWide(size, after), f.pastWide(size, after), f.automaticDtype(size, after, false), f.automaticDtype(size, after, true)]);
}
for (const size of [3e7, 3e8, 4e9]) for (const shared of [false, true]) for (const spare of [0, 2 ** 30]) attempt(`memory ${size} ${shared} ${spare}`, () => {
  const made = f.weightsMemory(size < 4e9 ? size : 1e6, { shared, spare }); const memory = made.memory ?? made; return [Object.keys(made), memory.buffer?.byteLength, typeof made.maximum === "bigint" ? String(made.maximum) : made.maximum];
});
for (const prompts of [undefined, f.PROMPTS_UNTIMED, f.PROMPTS_GPU, f.PROMPTS_CPU, "x"]) for (const answers of [undefined, "untimed", "gpu", "cpu", "why"])
  attempt(`line ${prompts} ${answers}`, () => f.gpuLine(prompts, answers));
attempt("holes", () => f.gpuHoles({ wq: { offset: 10, scales: 90, shape: [2, 4, 4], group: 2, kind: "int8" } }));
attempt("constants", () => [f.BATCH, f.GPU_END_MS, f.GPU_BLOCK, f.GPU_TOKENS, f.PATH_ROUNDS, f.SEARCH_SECONDS, f.BOTH_ON_8, f.GPU_WEIGHT_BYTES, f.GPU_TERNARY_BYTES]);
// (a pipe is read after exit() otherwise: the output is long, AGENTS.md's gpu-check lesson)
process.stdout.write(`${JSON.stringify(found)}\n`, () => process.exit(0));
