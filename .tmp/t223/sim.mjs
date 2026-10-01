// T223's review: forward.js's real thread search (createForward on kernels that do nothing, a shared memory, no helpers
// started: spawn resolves at once) on a made-up clock with noise, over many visits that remember a count between them
// as the page does (localStorage), on the owner's Android's llm-jp-3 150M (1 thread 110, 2 123, 4 38.6, 8 32.3 tok/s).
//   node .tmp/t223/sim.mjs [visits]
import fs from "node:fs";
import { createForward, footprint } from "../../public/forward.js";
import { CONTROL_BYTES } from "../../public/jobs.js";

const plans = JSON.parse(fs.readFileSync(new URL("../t130/plans.json", import.meta.url), "utf8"));
const p = plans.find((q) => q.id === "tiny-lm" && q.dtype === "int8");
const empty = new WebAssembly.Module(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
const stubs = () => new Proxy({}, { get: (_, name) => (name === "then" ? undefined : () => 0) });
const [dim, hidden, layers, heads, kvHeads, signedVocab, seqLen] = p.header;
const plan = {
  arch: "llama", dim, hidden_dim: hidden, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads, head_size: p.head_size,
  vocab_size: Math.abs(signedVocab), seq_len: seqLen, rotary: p.head_size, parallel_residual: false, kv_start: 256,
  rms_norm_eps: 1e-5, shared_classifier: signedVocab > 0, int8: true, relaxed: true, tensors: p.tensors,
  derived: Object.fromEntries(Object.entries(p.derived).map(([n, b]) => [n, new Uint8Array(b)])), outliers: [], half_kv: true,
};
const after = footprint(p.header, p.size, { dtype: "int8", halfKV: true, shared: true });
const PAGE = 65536;
const pages = Math.ceil((CONTROL_BYTES + p.size + after + 2 ** 20) / PAGE) + 1;
const memory = new WebAssembly.Memory({ initial: Math.ceil((CONTROL_BYTES + p.size) / PAGE) + 1, maximum: pages, shared: true });
const spawn = async () => ({ terminate() {} });

// a seeded generator (mulberry32) and a normal (Box-Muller)
let seed = 12345;
const random = () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const normal = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());

export const DEVICES = {
  // the owner's Android, llm-jp-3 150M on the CPU alone (T223)
  android: { 1: 1000 / 110, 2: 1000 / 123, 4: 1000 / 38.6, 8: 1000 / 32.3 },
  // what CPU section measured for the made-up 2 layer model there (T190): 4 fastest
  androidSection: { 1: 16.4, 2: 10.5, 4: 8.8, 8: 12 },
  // a CI runner (T199's)
  ci: { 1: 13, 2: 10, 4: 11 },
};

/** one visit: a new engine; the count remembered (or 0); generations of `generation` tokens until the search ends or
 * `most` generations; the noise: sigma (a token's log-normal spread), pBlock and blockFactor (a whole block of 5 slowed:
 * what else runs), drift (the speed falls by this fraction over a visit: heat) */
export async function visit(ms, { remembered, hint = 8, generation = 20, most = 40, sigma = 0, pBlock = 0, blockFactor = 3, drift = 0, pSpike = 0, spike = 4 }) {
  let engine = null, starting = false, time = 0, comparison = -1, token = 0, tokensRun = 0, blockSlow = 1;
  const told = [];
  const clock = () => {
    starting = !starting;
    if (starting) return time;
    if (engine.searchLog.length !== comparison) [comparison, token] = [engine.searchLog.length, 0];
    if (token % 5 === 0) blockSlow = random() < pBlock ? blockFactor : 1;
    token++;
    const base = ms[engine.threads] ?? 3 * Math.max(...Object.values(ms));
    time += base * Math.exp(sigma * normal()) * blockSlow * (1 + drift * Math.min(1, tokensRun++ / 400)) * (random() < pSpike ? spike : 1);
    return time;
  };
  engine = createForward({ memory, base: CONTROL_BYTES, size: p.size, kernels: { plain: empty, relaxed: empty, wide: false }, plan,
    spawn, clock, wrap: stubs, halfKeys: true });
  const log = [];
  await engine.findThreads({ from: hint, remembered, chose: (count) => told.push(count), compared: (v) => log.push(v) });
  let generations = 0;
  for (; generations < most; generations++) {
    engine.newGeneration();
    for (let pos = 0; pos < generation; pos++) engine.forward(100 + (pos % 50), pos, true);
    await Promise.resolve();  // the helpers a larger count needs "start"
    await Promise.resolve();
    if (!engine.searching && told.length) break;
  }
  const result = { told, threads: engine.threads, generations: generations + 1, log: engine.searchLog.map(({ best, candidate, faster }) => `${best}|${candidate}:${faster ? candidate : best}`) };
  engine.release();
  return result;
}

if (process.argv[1].endsWith("sim.mjs")) {
  const visits = Number(process.argv[2] ?? 300);
  // the deterministic cases first (gpu-default-check's): from 8, and a remembered 4
  const a = await visit(DEVICES.android, { remembered: 0 });
  console.log("deterministic, first visit from 8:", JSON.stringify(a));
  const b = await visit(DEVICES.android, { remembered: 4 });
  console.log("deterministic, remembered 4:", JSON.stringify(b));
  const c = await visit(DEVICES.android, { remembered: 2 });
  console.log("deterministic, remembered 2:", JSON.stringify(c));
  const d = await visit(DEVICES.android, { remembered: 1 });
  console.log("deterministic, remembered 1:", JSON.stringify(d));
}
