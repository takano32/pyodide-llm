// P2: the page end to end under failures: what the Warnings block and the link hold
import { install, load, byId, F } from "./page-harness.mjs";
import { base, use } from "./page-scenarios.mjs";
import { checkVerdict } from "../../src/bench.js";

const ALL = ["device", "cpu", "model", "gpu", "storage", "line"];
const only = process.argv[2];

async function run(label, mutate, sections = ALL, { show = 200, quiet = false } = {}) {
  if (only && !label.includes(only)) return;
  install();
  const page = await load();
  const s = base();
  mutate(s);
  use(s);
  await page.runSections(sections);
  const markdown = window.__benchmark.markdown;
  const at = markdown.indexOf("#### Warnings");
  const block = at < 0 ? [] : markdown.slice(at, markdown.indexOf("\n#### ", at + 5)).split("\n").filter((l) => l.startsWith("- "));
  const href = byId("issue").href, body = new URL(href).searchParams.get("body");
  const login = `https://github.com/login?return_to=${encodeURIComponent(href)}`.length;
  console.log(`\n== ${label}`);
  console.log(`statuses: ${Object.entries(page.results).map(([k, v]) => `${k} ${v.status}`).join(", ")}; report ${markdown.length} chars; Warnings ${block.length}; ` +
    `link holds ${body.includes("#### Summary") ? "the summary" : body.endsWith("please paste them here.") ? "only the request" : "the whole"} (login ${login})`);
  if (!quiet) block.forEach((l) => console.log(`   ${l.slice(0, show)}`));
  const inSummary = body.includes("#### Summary") ? body.split("\n").filter((l) => l.startsWith("- ") && block.includes(l)).length : undefined;
  if (inSummary !== undefined) console.log(`   (the summary keeps ${inSummary} of ${block.length}${body.match(/- … and \d+ more[^\n]*/)?.[0] ? `; ${body.match(/- … and \d+ more[^\n]*/)[0]}` : ""})`);
  if (inSummary !== undefined && quiet) body.split("\n").filter((l) => block.includes(l)).forEach((l) => console.log(`   kept: ${l.slice(0, 110)}`));
  return { page, markdown, body };
}

const layerNames = [
  "a layer, llama.cpp, separate steps", "a layer, llama.cpp, fused (T150)", "a layer, DP4A, separate steps", "a layer, DP4A, fused (T175)"];
const stagesFloat = "stages: q 1.3e-7, K and V 0 to the nearest float16, 152 toward zero, 0 away from it, 232 farther, attention 3.8e-3, silu(gate) × up 2.9e-3, stream 2.6e-3; cache 1.7e-3; first to depart: K and V";
const wrongLayers = (s) => {
  for (const name of layerNames) s.check[name] = { worstRelative: 2.6e-3, ok: false, stages: stagesFloat };
  s.check["tokens on the GPU"] = { worstRelative: 0, ok: false, tokens: 8, edge: 0, problems: ["T 0.7, token 1: 331, the CPU 260"], steps: "steps: T 0: logits within 1.3e-2 of the largest, the most likely token the same at 6 of 6 steps" };
};

await run("everything ok", () => {});
await run("four layers and the tokens WRONG", wrongLayers);
await run("a shader the device refused (Dawn's message with line breaks)", (s) => {
  s.check["llama.cpp tiles 32×32, f16"] = { worstRelative: NaN, ok: false, error: 'Invalid ComputePipeline "tile".\n - While validating compute stage ([ShaderModule "main"]).\n - While calling [Device].CreateComputePipeline().' };
});
await run("the device lost at the layer step", (s) => { s.lostAfter = "a layer of a token"; });
await run("steps that failed as a whole", (s) => {
  s.steps["a layer of a token"] = "out of memory";
  s.steps["a prompt all at once"] = "Device is lost\nsecond line";
  s.steps["the device's ceilings"] = "x | y";
  s.steps["what a token costs besides the weights"] = "no timestamp";
  s.steps["tokens on the GPU"] = "unused";
});
await run("the check step failed as a whole", (s) => { s.steps["the shaders against JavaScript"] = "Error while parsing WGSL:\n :12:3 error: unknown type\n - While calling CreateShaderModule"; });
await run("the adapter step failed (the page calls that none)", (s) => { s.steps["the adapter"] = "requestDevice rejected: out of memory"; });
await run("the bridge's worker fails", (s) => { s.steps.bridge = "Atomics.wait cannot be called in this context"; });
await run("the CPU's logits not finite, a ceiling cut short", (s) => { s.cpu.rows[1].finite = false; s.cpu.ceilings.dot = { GMACs: 400, cutShort: true }; }, ["cpu"]);
await run("the CPU section fails, the device section fails", (s) => { s.steps.cpu = "the worker failed: out of memory"; s.steps.device = "no navigator"; }, ["device", "cpu"]);
await run("storage read back WRONG, the line fails on both", (s) => { s.storage.read = { seconds: 0.01, wrong: 3 }; s.steps.line = "NetworkError"; }, ["storage", "line"]);
await run("line: one fetch fails, a paced read errors", (s) => { s.line.hf = { error: "Failed to fetch" }; s.line.paced = [{ rate: 1, MBps: 1 }, { rate: 4, slower: true }, { rate: 8, error: "aborted" }]; }, ["line"]);
await run("the model page's path: the GPU stopped for a reason without 'failed'", (s) => {
  s.paths.gpu = { ...s.paths.gpu, lost: "the GPU computed logits that are not finite numbers (NaN or infinity) at position 3" };
  s.paths.rows[0].gpu = { skip: s.paths.gpu.lost };
}, ["model"]);
await run("the model page's path: a software thread stopped, the search not ended", (s) => {
  s.paths.how = { alone: "a software thread stopped" };
  s.paths.threads = 1;
}, ["model"]);
await run("the model page's path: threads did not start / stopped while timed / unfinished", (s) => { s.paths.how = { unfinished: 120, stopped: true }; }, ["model"]);
await run("the model section fails", (s) => { s.model = undefined; s.paths = { error: "the NumPy engine runs this model here: the page's path is the NumPy engine's" }; }, ["model"]);
await run("the model skips a round (deviceMemory not told)", (s) => { s.rounds[1] = { name: "without the kernels", without: ["kernels"], skip: "this browser does not say how much memory the device has" }; }, ["model"]);
await run("the GPU WRONG, then the CPU section run after it (the GPU's tables are written again)", (s) => wrongLayers(s), ["gpu", "cpu"]);
const allLayers = [
  "a layer, llama.cpp, separate steps", "a layer, llama.cpp, fused (T150)", "a layer, llama.cpp, fused (T150), flash_attn_vec (subgroups)",
  "a layer, llama.cpp, separate steps, subgroups", "a layer, llama.cpp, fused (T150), subgroups", "a layer, llama.cpp, fused (T150), subgroups, flash_attn_vec (subgroups)",
  "a layer, DP4A, separate steps", "a layer, DP4A, fused (T175), the norms apart", "a layer, DP4A, fused (T175), the norms apart, flash_attn_vec (subgroups)",
  "a layer, DP4A, fused (T175)", "a layer, DP4A, fused (T175), flash_attn_vec (subgroups)"];
const stagesDp4a = "stages: qkv quantized: scales 1.0e-7, 1 of 2112 off by 1, q 1.6e-7, K and V 1 to the nearest float16, 146 toward zero, 0 away from it, 237 farther, attention 3.2e-3, " +
  "o quantized: scales 8.8e-3, 32 of 2112 off by 1, ffn quantized: scales 2.3e-7, 0 of 2112 off by 1, silu(gate) × up 2.0e-7, down quantized: scales 5.1e-7, 0 of 2080 off by 1, stream 2.0e-7; cache 1.7e-3; first to depart: K and V";
const probe = (s) => {
  for (const name of allLayers) {
    const dp4a = name.includes("DP4A");
    s.check[name] = { worstRelative: dp4a ? 1.7e-3 : 2.6e-3, ok: false, stages: dp4a ? stagesDp4a : stagesFloat,
      ...(dp4a ? { quantized: [{ point: "qkv", wrong: null, scale: 1e-7, apart: 1, of: 2112 }, { point: "o", wrong: "far from quantize_x's", scale: 8.8e-3, apart: 32, of: 2112 }] } : {}) };
  }
  s.check["tokens on the GPU"] = { worstRelative: 0, ok: false, tokens: 8, edge: 0, problems: ["T 0.7, token 1: 331, the CPU 260"],
    steps: "steps: T 0: logits within 1.3e-2 of the largest, the most likely token the same at 6 of 6 steps, K and V 144 to the nearest float16, 739 toward zero, 58 away from it, 2131 farther; T 0.7: logits 1.2e-3 1.1e-3 1.3e-3 1.6e-3 3.4e-3 2.4e-3 of the largest by step, the most likely token the same at 6 of 6 steps, K and V 181 to the nearest float16, 782 toward zero, 58 away from it, 2051 farther" };
};
await run("PROBE: 11 layers and the tokens WRONG with T225's real stage lines, a quiet device", probe, ALL, { quiet: true });
await run("PROBE: the same on a noisy device (the model page's path unsteady all through)", (s) => {
  probe(s);
  s.paths.rows.forEach((row) => { row.cpu.unsteady = true; });
  s.paths.perCount = [{ threads: 1, speed: 107, low: 100, high: 108, unsteady: true }, { threads: 4, speed: 163, low: 120, high: 165, unsteady: true }];
}, ALL, { quiet: true });
if (process.env.WATCHDOG) {
  // a clock 2000 times faster, so that the 5 silent minutes take 150 ms
  const now = performance.now.bind(performance);
  performance.now = () => now() * 2000;
  const every = globalThis.setInterval;
  globalThis.setInterval = (fn, ms) => every(fn, ms / 2000);
  await run("the watchdog (GPU says nothing)", (s) => { s.stall = "gpu"; }, ["gpu"], {});
}
process.exit(0);
