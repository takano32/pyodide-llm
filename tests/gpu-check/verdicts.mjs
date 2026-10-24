// T.357.3: the verdicts of tests/gpu-check.mjs, a part of that tool: what the harness posted for the quantizers and
// for a model (its runs on the CPU and with the GPU, lines.mjs's lines) against the model's case (NumPy's
// answers), said in the log's lines, and whether any of them failed. Functions of what they are given: no GPU, no
// browser, no Pyodide (tests/gpu-check-parts-check.mjs runs them on made-up numbers).
import { floats, worstRow, worstAt, argmax, logSoftmax, divergence, scaleOff } from "./measures.mjs";
import { NOT_FINITE_STATUS, NOT_FINITE_KV_STATUS, GPU_LINE, HALF_LINE, K, K_PACKED, SCALE_LINE, TIE } from "./lines.mjs";
import { stepsRight } from "./steps.mjs";
import { layerTables } from "./tables.mjs";

// the status of a GPU that takes the prompts (T.152: what it says of the tokens after it)
// (T.152: "prompts and answers on WebGPU", or "prompts on WebGPU, answers on the CPU")
const promptsOnGpu = (note) => /^prompts (and answers )?on WebGPU($|, )/.test(note ?? "");
// T.241's review: the quantizers alone (the harness's quantizerProbe): the word of every group's scale as the contract in
// shaders.js says (a finite group's the largest |value| / 127, a group with a NaN or an infinity a NaN)
function quantizersWrong(outcome) {
  let failed = false;
  const q = outcome.quantizers;
  if (q?.rows) {
    const wrong = q.rows.filter((r) => !r.ok);
    console.log(`the quantizers alone (QUANTIZE, NORM_QUANTIZE): ${q.rows.length - wrong.length} of ${q.rows.length} groups' scale words as they are to be (a finite group's the largest |value| / 127, ` +
      `a group with a NaN or an infinity a NaN)${wrong.length ? ` — FAILED\n    - ${wrong.map((r) => `${r.shader}, ${r.what}: the word is ${r.word}, where ${r.want} is to be`).join("\n    - ")}` : ""}`);
    failed ||= wrong.length > 0;
  } else if (q?.error) {
    console.log(`the quantizers alone: FAILED\n    - ${q.error}`);
    failed = true;
  } else console.log(`the quantizers alone: not tried (${q?.skipped ?? "no answer"})`);
  return failed;
}
// a model's lines (the CPU's, each run's, the steps', the tables) and whether any of them failed
function modelWrong(cases, { id, cpu, gpu: runs, late, refused, remembered, alone, broken, nanSeconds }) {
  let failed = false;
  const c = cases.find((entry) => entry.id === id), ref = c.reference, n = ref.tokens.length - 1;
  const [dim, , every, heads, kvHeads] = ref.header, layers = ref.attending ?? every, kvDim = (c.headDim || dim / heads) * kvHeads;
  const exact = { keys: ref.keys, values: ref.values };
  // T.187: the worst row of every layer (the lines are held a layer at a time)
  const byLayer = (got, want = exact) => Array.from({ length: layers }, (_, l) => {
    const layer = (b64) => floats(b64).subarray(l * n * kvDim, (l + 1) * n * kvDim);
    return Math.max(worstRow(layer(got.keys), layer(want.keys), kvDim), worstRow(layer(got.values), layer(want.values), kvDim));
  });
  const kv = (run, want = exact) => Math.max(worstRow(floats(run.keys), floats(want.keys), kvDim), worstRow(floats(run.values), floats(want.values), kvDim));
  // T.153: the first layer's alone (a wrong step shows there already; the float16 of the cache grows over the layers)
  const first = (b64) => floats(b64).subarray(0, n * kvDim);
  const firstKv = (run, want = exact) => Math.max(worstRow(first(run.keys), first(want.keys), kvDim), worstRow(first(run.values), first(want.values), kvDim));
  // T.187: the first layer's scale against NumPy's, the larger of the keys' and the values'
  const scale = (run, want = exact) => [scaleOff(first(run.keys), first(want.keys)), scaleOff(first(run.values), first(want.values))]
    .reduce((worst, x) => (Math.abs(x) > Math.abs(worst) ? x : worst), 0);
  const cpuKv = kv(cpu);
  // T.153: E16, NumPy's answer with its cache in float16 against NumPy's. T.187: Q8, with the matrices' inputs in 8 bits
  const half = { keys: ref.keys16, values: ref.values16 }, eight = { keys: ref.keys8, values: ref.values8 };
  const e16 = kv(half), q8 = kv(eight), q8First = firstKv(eight), e16s = byLayer(half), q8s = byLayer(eight);
  const want = floats(ref.logits), largest = want.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
  const p = logSoftmax(want), top = argmax(want);
  const tenBest = [...want.keys()].sort((i, j) => want[j] - want[i]).slice(0, 10);
  const logits = (b64) => {
    const got = floats(b64);
    return { kl: divergence(p, got), ten: tenBest.reduce((m, i) => Math.max(m, Math.abs(got[i] - want[i])), 0) / largest, token: argmax(got) };
  };
  // the most likely token NumPy's, or one NumPy gives at least half the probability of its own (a near tie)
  const near = (token) => token === top || p[token] >= p[top] - TIE;
  const cpuLogits = logits(cpu.logits);
  const measures = { e16, q8, cpuLogits, logits, scale, kl16: divergence(p, floats(ref.logits16)), kl8: divergence(p, floats(ref.logits8)) };
  console.log(`${id} (${layers} layers, ${heads} heads, ${kvHeads} of keys and values, ${n} tokens): the CPU's keys and values ` +
    `${cpuKv.toExponential(2)} from NumPy's (the first layer's scale ${scale(cpu).toExponential(1)}), logits KL ${cpuLogits.kl.toExponential(2)}, ` +
    `most likely ${cpuLogits.token} ${cpuLogits.token === top ? "as NumPy's" : `(NumPy's ${top})`}, the prompt ${(cpu.promptMs / n).toFixed(2)} ms a token; ` +
    `E16 (NumPy's cache in float16) ${e16.toExponential(2)} (KL ${measures.kl16.toExponential(1)}, scale ${scale(half).toExponential(1)}); ` +
    `Q8 (and its matrices' inputs in 8 bits) ${q8.toExponential(2)}, the first layer ${q8First.toExponential(2)} (KL ${measures.kl8.toExponential(1)}, scale ${scale(eight).toExponential(1)})`);
  if (late) {
    // T.147: a request forward.js gave up on is answered by nothing
    const tried = promptsOnGpu(late.note) && late.gpuTokens === 0;
    // T.148: and the CPU did those blocks itself: its keys are the CPU's own run's, to the bit
    const cpuDid = late.keys === cpu.keys;
    const wrong = tried && (late.done !== 0 || late.failed !== 0 || !cpuDid);
    console.log(`  a GPU that answers late: ${!tried ? `not tried (${late.note}, ${late.gpuTokens} tokens on the GPU)` : wrong ? `it answered (done ${late.done}, failed ${late.failed}) or the CPU did not do its blocks (${cpuDid ? "it did" : "it did not"}) — FAILED` : "given up, it wrote nothing, and the CPU did the blocks"}`);
    failed ||= wrong;
  }
  if (refused) {
    // T.148: refused before anything was compiled (SwiftShader compiles a shader in 10 to 90 s): within a few seconds
    const right = /a fallback adapter/.test(refused.note) && refused.seconds < 30 && refused.ended;
    console.log(`  a fallback adapter without the tests' leave: ${refused.note} in ${refused.seconds.toFixed(1)} s, ` +
      `${refused.ended ? "its worker ended" : "its worker did not say it ended"}${right ? "" : " — FAILED"}`);
    failed ||= !right;
    // T.205: stopped while it got ready, it ends and says so (within forward.js's 5 s)
    const { whileStarting: w } = refused;
    console.log(`  a GPU let go while it got ready: ${w.ended ? "its worker ended" : "its worker did not say it ended"} in ${w.seconds.toFixed(1)} s${w.ended ? "" : " — FAILED"}`);
    failed ||= !w.ended;
  }
  if (alone) {
    // T.156: on the GPU alone the same shaders on the same weights write the same keys and values and ids as the run
    // with every byte in memory, to the bit; a step on the CPU throws; a GPU that fails as it opens is lost at once,
    // and says why
    const { right: r, pieces: p } = alone;
    const same = r.keys === alone.keys && r.values === alone.values && JSON.stringify(r.first) === JSON.stringify(alone.first);
    const good = !r.lost && r.gpuTokens === n && same && /^prompts and answers on WebGPU/.test(r.status ?? "") && /on it alone/.test(r.threw ?? "")
      && Boolean(p.lost) && /on it alone/.test(p.threw ?? "");
    // (the made-up model: a CPU far faster than the GPU is taken, and what the page keeps has the device's key)
    const f = alone.cpuFaster, cpuRight = !f || (/the CPU as \/benchmark\/ measured it/.test(f.lost ?? "") && f.verdict?.key === alone.key && /on it alone/.test(f.threw ?? ""));
    console.log(`  on the GPU alone (${(r.stored / 1e6).toFixed(2)} MB here): ${r.lost ? `lost (${r.lost})` : `${r.gpuTokens} tokens of the prompt, keys and values ` +
      `${r.keys === alone.keys && r.values === alone.values ? "the same to the bit" : "NOT the same"} as with the weights here, steps ${JSON.stringify(r.first)} ` +
      `(${JSON.stringify(alone.first)}), "${r.status}", a step on the CPU ${r.threw ? "threw" : "ran"}`}; a GPU that fails as it opens: ${p.lost ? `lost (${p.lost})` : "not lost"}, ` +
      `a step on the CPU ${p.threw ? "threw" : "ran"}${f ? `; a CPU far faster: ${f.lost ? `lost (${f.lost})` : "not lost"}, the verdict ${f.verdict?.key === alone.key ? "with the device's key" : "WITHOUT the device's key"}` : ""}` +
      `${good && cpuRight ? "" : " — FAILED"}`);
    failed ||= !good || !cpuRight;
  }
  if (remembered) {
    const right = remembered.same?.remembered === true && remembered.other?.remembered === false;
    console.log(`  the shaders remembered: for this adapter ${remembered.same?.remembered ? "taken" : "not taken"} (${remembered.same?.matrices}), ` +
      `for another ${remembered.other?.remembered ? "taken" : "not taken"}${right ? "" : " — FAILED"}`);
    failed ||= !right;
  }
  for (const gpu of runs) {
    const failures = [];
    if (!promptsOnGpu(gpu.note)) failures.push(`the GPU did not take it: ${gpu.note}`);
    // T.226: the status line of a run whose steps were asked for is the owner's words (forward.js's gpuLine): with the
    // steps on the GPU, or with why they are not (the reason itself in the console alone)
    const line = !gpu.steps ? gpu.note : gpu.steps.why ? "prompts on WebGPU, answers on the CPU" : "prompts and answers on WebGPU";
    if (promptsOnGpu(gpu.note) && gpu.note !== line) failures.push(`the status line is "${gpu.note}", not "${line}"`);
    if (gpu.gpuTokens !== n || gpu.again?.gpuTokens !== n) failures.push(`the GPU took ${gpu.gpuTokens} and ${gpu.again?.gpuTokens} of ${n} tokens`);
    if (gpu.ended === false) failures.push("the GPU's worker did not say it ended within 5 s of the release (T.205)");
    if (gpu.past?.gpuTokens !== 0) failures.push(`a block past the GPU's keys and values went to the GPU (${gpu.past?.gpuTokens} tokens)`);
    const kind = /DP4A/.test(gpu.form ?? "") ? "packed" : /f16/.test(gpu.form ?? "") ? "f16" : "float32";
    // T.187: the packed shaders' first layer and scale against Q8, their own arithmetic, and all their layers against
    // NumPy's by Q8's distance from it (see lines.mjs); the others against NumPy's
    const packed = kind === "packed", [firstStick, name] = packed ? [eight, "Q8"] : [exact, "NumPy's"];
    const shaderLine = kind === "f16" ? HALF_LINE : GPU_LINE;
    // T.187: a line a layer (the review: the largest of all layers against the largest of E16's or Q8's let a deep
    // model's early layers go 10 times their own E16 or Q8)
    const lines = e16s.map((e16l, l) => (packed ? K_PACKED * q8s[l] : Math.max(shaderLine, K[kind] * e16l)));
    const gpuKv = kv(gpu), againKv = gpu.again ? kv(gpu.again) : NaN, gpuFirst = firstKv(gpu, firstStick);
    const gpuLayers = byLayer(gpu), againLayers = gpu.again ? byLayer(gpu.again) : lines.map(() => NaN);
    const ratios = lines.map((line, l) => Math.max(gpuLayers[l], againLayers[l]) / line);
    // the worst layer (NaN the worst of all)
    const at = worstAt(ratios);
    Object.assign(gpu, { line: lines[at], ratio: ratios[at], layer: at });
    if (!(ratios[at] <= 1)) {
      failures.push(`the keys and values of the GPU's layer ${at} are ${gpuLayers[at].toExponential(2)} and ${againLayers[at].toExponential(2)} from NumPy's (its line ${lines[at].toExponential(2)})`);
    }
    // T.545: the packed shaders' first layer against Q8 is one layer's arithmetic where the first layer attends; an LFM2's
    // first keys and values are of a layer after convolution layers (their inputs rounded to 8 bits on the GPU as Q8
    // rounds them, then B × z × C), where the GPU's 8 bits and Q8's part a little at a tie and the product grows it (the
    // real 230M's and 350M's 1.2e-2 and 2.0e-2 from Q8, run 38852988481, where their layers were 0.38 and 0.41 of their
    // lines): the line of every layer holds those, the float forms' first layer is held as it was
    const oneLayer = !packed || c.plan.layers[0].kind === "attention";
    if (oneLayer && !(gpuFirst <= shaderLine)) failures.push(`the first layer's keys and values are ${gpuFirst.toExponential(2)} from ${name} (line ${shaderLine.toExponential(2)})`);
    const gpuScale = scale(gpu, firstStick), againScale = gpu.again ? scale(gpu.again, firstStick) : NaN;
    if (oneLayer && (!(Math.abs(gpuScale) <= SCALE_LINE) || !(Math.abs(againScale) <= SCALE_LINE))) {
      failures.push(`the first layer's keys and values are scaled by 1 + ${gpuScale.toExponential(2)} and 1 + ${againScale.toExponential(2)} of ${name} (line ${SCALE_LINE})`);
    }
    const gpuLogits = logits(gpu.logits), againLogits = logits(gpu.again.logits);
    if (!near(gpuLogits.token) || !near(againLogits.token)) failures.push(`another most likely token (${gpuLogits.token}, ${againLogits.token}) than NumPy's ${top}, and not a near tie`);
    // T.241: a block whose row held a NaN, or an infinity: the GPU answered it, and none of that token's keys and
    // values of any layer came back a finite number (the harness's brokenRows)
    for (const b of gpu.brokenRows?.cases ?? []) {
      if (!b.answered || b.failed || b.finite !== 0) {
        failures.push(`a block with ${b.value} in the row of token ${b.token} (at ${b.at}): the request ${!b.answered ? "was not answered" : b.failed ? "failed" : "was answered"}, and ${b.finite} of that token's ${b.of} keys and values are finite numbers (none is to be)`);
      }
    }
    // (the rows of a block: none where this browser does not put a NaN in this model's rows, gpu-check's NAN_ROUNDS)
    const said = [];
    if (gpu.brokenRows) {
      said.push(gpu.brokenRows.skipped ? `not tried (${gpu.brokenRows.skipped})`
        : gpu.brokenRows.cases.map((b) => `${b.value} ${b.finite} of ${b.of} finite`).join(", "));
    }
    // T.243: the same block through forward.js (the harness's refusedBlock): refused whole (no token of it on the GPU,
    // nor of the block after it), the status line says why, and the cache of the two blocks is the CPU's own run's, to
    // the bit (the CPU's kernels would have read the GPU's NaN as finite numbers: the cache held 65536 and more)
    const r = gpu.refusedBlock;
    if (r && !r.skipped) {
      const bits = (b64) => { const f = floats(b64); return new Uint32Array(f.buffer, f.byteOffset, f.length); };
      const cpuOwn = (got, want) => {
        const g = bits(got), w = bits(want), each = g.length / (32 * kvDim);
        return g.every((word, i) => word === w[Math.floor(i / (32 * kvDim)) * n * kvDim + (i % (32 * kvDim))]) && w.length === each * n * kvDim;
      };
      const own = cpuOwn(r.keys, cpu.keys) && cpuOwn(r.values, cpu.values);
      if (r.gpuTokens !== 0 || r.after !== 0 || !NOT_FINITE_KV_STATUS.test(r.status ?? "") || !own) {
        failures.push(`a block with ${r.value} in the row of token ${r.token} (at ${r.at}) through forward.js: ${r.gpuTokens} of its tokens and ${r.after} with the next block's taken on the GPU, ` +
          `the status line "${r.status}", the cache ${own ? "the CPU's own" : "not the CPU's own"}: the block is to be refused whole, the GPU stopped for keys or values that are not finite, the CPU to take both blocks`);
      }
      said.push(`through forward.js ${r.value} in token ${r.token}: ${r.gpuTokens === 0 && own ? "refused, the cache the CPU's own" : `${r.gpuTokens} tokens taken, the cache ${own ? "the CPU's own" : "not the CPU's own"}`}`);
    } else if (r?.skipped) said.push(`through forward.js not tried (${r.skipped})`);
    const rowsSaid = said.length ? `, the keys and values of a token whose row held ${said.join("; ")}` : "";
    console.log(`  ${gpu.form ?? "no form"}, ${gpu.attention ?? "no attention"}: keys and values ${gpuKv.toExponential(2)} (all at once ${againKv.toExponential(2)}, ` +
      `the first layer ${gpuFirst.toExponential(2)} from ${name}, the CPU's ${firstKv(cpu).toExponential(2)}; ${(gpuKv / e16).toFixed(1)} E16, ${(gpuKv / q8).toFixed(2)} Q8; ` +
      `layer ${at} ${ratios[at].toFixed(2)} of its line), ` +
      `the first layer's scale ${gpuScale.toExponential(1)} (${againScale.toExponential(1)}), logits KL ${gpuLogits.kl.toExponential(2)} (${againLogits.kl.toExponential(2)}), ` +
      `the prompt ${(gpu.promptMs / n).toFixed(2)} ms a token (${gpu.note})${rowsSaid}` +
      (failures.length ? ` — FAILED\n    - ${failures.join("\n    - ")}` : ""));
    failed ||= failures.length > 0;
    // (evaluated, and said, whatever came before)
    const stepsFine = gpu.steps ? stepsRight(c, gpu.steps, { e16s, q8s, kvDim, prompt: gpuLayers }) : true;
    failed ||= !stepsFine;
  }
  // T.241: a weight of the last layer that is no finite number on the GPU (the harness's brokenWeight): the step was
  // asked of the GPU and refused whole, the status line saying that the logits were not finite; a line a form
  for (const form of [...new Set((broken ?? []).map((b) => b.form))]) {
    const failures = [], said = [];
    for (const b of broken.filter((x) => x.form === form)) {
      const refusedStep = b.asked && b.taken === null && NOT_FINITE_STATUS.test(b.status ?? "");
      if (!refusedStep) {
        failures.push(`${b.name} ${b.value}: ${b.asked ? `the step gave ${JSON.stringify(b.taken)} and the status line "${b.status}"` : `the step was not asked of the GPU (${b.why}; "${b.note}")`}, where it is to be refused whole and say the logits were not finite`);
      }
      said.push(`${b.name} ${b.value} ${refusedStep ? "refused" : "NOT refused"}`);
    }
    console.log(`  a token by ${form}, the GPU holding in its last layer: ${said.join(", ")}${failures.length ? ` — FAILED\n    - ${failures.join("\n    - ")}` : ""}`);
    failed ||= failures.length > 0;
  }
  layerTables(c, cpu, runs, measures);
  // T.241's review: what the NaN rounds took of the seconds above, summed over the runs, where this browser runs them
  if (nanSeconds && Object.values(nanSeconds).some((s) => s > 0)) {
    console.log(`seconds of the NaN rounds of ${id}: ${Object.entries(nanSeconds).map(([round, s]) => `${round} ${s.toFixed(1)}`).join(", ")}`);
  }
  return failed;
}

export { promptsOnGpu, quantizersWrong, modelWrong };
