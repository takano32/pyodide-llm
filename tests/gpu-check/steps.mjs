// T.357.3: the steps of tests/gpu-check.mjs, a part of that tool: a generation's steps on the GPU against NumPy's
// greedy continuation, said in one line of the log, and whether it is right. A function of what it is given (the
// CPU's sampler in JavaScript is the runtime's, shaders/likecpu.ts).
//
// T.152: a generation's steps on the GPU (forward.js's generateMany, gpu.js's fused layer of a token and SAMPLE), after
// the prompt of a run whose GPU took them (the first, and one a form of a token's layer this adapter can make: each
// forced): NumPy continues the prompt greedy for GEN tokens (its logits of each step, and the keys and values of their
// positions); the GPU is asked for 4 steps greedy from the prompt's last token, the CPU then takes one step (the
// forward pass on the keys and values the GPU wrote back), and the GPU 3 more (the CPU's position goes up to it first).
// Checked: the ids are NumPy's, or where one is not, a near tie (T.187's: NumPy gives it at least half the probability
// of its most likely; the steps after it are not compared, their inputs differ; the CPU's step feeds NumPy's id to the
// GPU's next ones whatever it chose); the keys and values the GPU wrote back of its first 4 positions in forward.js's
// cache against NumPy's, a layer at a time (T.187), to the line of the prompt's shaders of the same arithmetic (float32 GPU_LINE or K × E16;
// DP4A K_PACKED_STEPS × Q8's distance) or the run's own prompt's distance, the larger (the steps' attention reads the
// prompt's keys and values; the next 3 read the CPU's step's too, its 7-bit activations, and are held to their ids); a
// stop token (NumPy's second) ends the steps after it; a penalty of 100 on NumPy's first id (greedy) gives the largest
// of NumPy's logits penalized so (or a near tie); two sampled steps at temperature 2 with random numbers of 0.02 and
// 0.98 give two tokens of NumPy's nucleus, not the same (the random numbers reach the GPU: where either lands is not
// held to NumPy's, the GPU's logits being others; the draw itself is SAMPLE's, checked on the device, gpu.js); and a
// step after the prompt went through the CPU while the GPU's own cache held the keys and values of other tokens
// (a prompt of them through the GPU first): the CPU's most likely token on its own keys and values, or a near tie of
// its logits (the CPU's keys and values went up). T.226: the made-up Qwen2 and Qwen3 too (their biases of q, k and v
// drawn around 0, an epsilon of 0.5 on the norms of heads of 32 where dim / heads is 16: what a step's ADD, HEAD_NORM
// and TOKEN_ROPE must get right in every layer, held by the keys and values a layer at a time), and the made-up GPT-2
// and GPT-NeoX (LayerNorm, the biases, GELU, the positions added to a step's embedding, RoPE on a part of a head, the
// parallel residual); the made-up GPT-2's final norm has eight weights of 12 (GPT-2's outlier channels, T.92: the
// engine takes their columns of the classifier apart on the CPU, and the GPU's classifier multiplies floats).
import { runtimeUrl } from "../tree.mjs";
import { floats, worstRow, worstAt, argmax, logSoftmax } from "./measures.mjs";
import { GEN, NOT_FINITE_STATUS, GPU_LINE, K, K_PACKED_STEPS, TIE } from "./lines.mjs";
const wgsl = await import(runtimeUrl("shaders.js"));

// T.213: the first matrix a token's layer would bind inside a joined buffer (q, k and v as one, gate and up as one,
// in gpu.js's tokensLayout's order; each after the ones before it) whose values or scales would not start where a device binds a buffer, or null where every
// one does (the model's steps then belong on the GPU). The device's alignment is WebGPU's default of 256 bytes
// (gpu.js asks for no other), the scales 4 bytes a group of 32: stories15M's k starts at 288 × 288 weights, whose
// scales, 10368 bytes, are not a multiple of 256 (T.150's (b))
// (T.232: a ternary weight is a quarter of a byte, and a scale goes with every 32 bytes as well)
function unbound(c) {
  const [dim, hidden, , heads, kvHeads] = c.reference.header, head = c.headDim || dim / heads, ALIGN = 256, GROUP = 32;
  // (T.226: GPT-2's and GPT-NeoX's FFN has no gate: no w3, and w1 a buffer of its own)
  const joined = { wk: head * heads * dim, wv: head * (heads + kvHeads) * dim, ...(c.arch === "llama" ? { w3: hidden * dim } : {}) };
  for (const [name, weights] of Object.entries(joined)) {
    const values = c.dtype === "ternary" ? weights / 4 : weights;
    if (values % ALIGN || (values / GROUP) * 4 % ALIGN) return name;
  }
  return null;
}

// T.152: the steps of a generation on the GPU against NumPy's greedy continuation (see the head of this file): a line
// "  a token by <form>: ..." and whether it is right
function stepsRight(c, steps, { e16s, q8s, kvDim, prompt }) {
  const ref = c.reference, greedy = ref.greedy, vocab = floats(ref.logits).length;
  const rows = floats(ref.greedyLogits), logitsOf = (i) => rows.subarray(i * vocab, (i + 1) * vocab);
  const failures = [], said = [];
  // a model whose steps stay on the CPU (not asked of the GPU), or the run in pieces (T.155: a token's layer reads a
  // matrix whole): said, and right
  if (steps.why) {
    // (T.152's review: or a classifier larger than what the adapter binds, a table being one piece; the rows of the
    // logits hold its size: vocabulary × dim)
    const table = floats(ref.logits).length * ref.header[0];
    // (T.226: no model of these has its steps left unasked for by its form any more: Qwen2's, Qwen3's, GPT-2's and
    // GPT-NeoX's go to the GPU too, and steps.planned === false, which passed them before, is a failure now: T.213's rule)
    const right = (c.force?.pieceBytes && !steps.forced && /past a buffer/.test(steps.why)) ||
      (/^the (classifier|embedding) is past a buffer/.test(steps.why ?? "") && table > steps.binds) ||
      // (T.261.4.2, T.545.2: a Gemma 3's and an LFM2's steps are the GPU's now: its GEGLU, norms behind, window and local
      // tables, and its convolution layers, are a token's fused layer's, and their steps on the CPU are a failure)
      steps.why === `${unbound(c)} would not start where this GPU binds a buffer`;
    // (T.547.3.1 took a model of a stack whose layers are each of a packing of their own's steps on the CPU, until
    // T.547.3.2 gave a token's layer its matrices apart)
    // (T.547.2.3 took a model of Q4_K blocks' steps on the CPU, until T.547.2.4 gave a token's layer their reading)
    console.log(`  a token: on the CPU (${steps.why})${right ? "" : " — FAILED"}`);
    return right;
  }
  // whether NumPy gives id at least half the probability of want (T.187's near tie: log-probabilities within TIE)
  const near = (logits, want, id) => id === want || logSoftmax(logits)[id] >= logSoftmax(logits)[want] - TIE;
  // the GPU's ids against NumPy's, step after step from `at`, up to the first that is not NumPy's (then a near tie; the
  // steps after it have other inputs). keep: the positions whose inputs were NumPy's are compared (theirs, [position
  // - n] each)
  const positions = [];
  const run = (ids, at, count, keep = true) => {
    if (!ids || ids.length !== count) {
      failures.push(`${count} steps from ${at} came back as ${JSON.stringify(ids)}`);
      return 0;
    }
    let same = 0;
    for (; same < count && ids[same] === greedy[at + same]; same++);
    if (keep) for (let i = 0; i <= Math.min(same, count - 1); i++) positions.push(at + i);
    if (same < count && !near(logitsOf(at + same), greedy[at + same], ids[same])) {
      // (T.547.3.6: and by how much, in NumPy's log-probabilities: a margin of NumPy's own or the GPU's error)
      const p = logSoftmax(logitsOf(at + same));
      failures.push(`step ${at + same} is ${ids[same]}, NumPy's ${greedy[at + same]}, and not a near tie (${(p[greedy[at + same]] - p[ids[same]]).toFixed(2)} below it in log-probability)`);
    }
    return same;
  };
  // (the keys and values of the second run's positions are not held to a line: they are the attention's over the
  // CPU's step between, its 7-bit activations, 1.02e-2 to 1.30e-2 on llm-jp-3 150M's float forms, run 36324147634)
  const first = run(steps.first, 0, 4), second = first === 4 ? run(steps.second, 5, 3, false) : 0;
  // the step after the prompt went through the CPU (the GPU's cache held another's): the CPU's most likely token on
  // the same keys and values, or a near tie of the CPU's logits
  const cpuLogits = floats(steps.uploadedLogits ?? ""), cpuTop = argmax(cpuLogits), uploaded = steps.uploaded?.[0];
  const cpuP = cpuLogits.length ? logSoftmax(cpuLogits) : [], gap = cpuP[cpuTop] - cpuP[uploaded];
  // (the CPU's logits carry its 7-bit activations at that token, the GPU's not: a quarter of the probability, not half.
  // synthetic-wide's GPU took a token 0.72 below the CPU's on every form, run 36326266335; the GPU without the CPU's
  // keys and values, see TASKS.md's T.152)
  if (!cpuLogits.length || !(gap <= 2 * TIE)) failures.push(`after a prompt on the CPU the step is ${uploaded}, the CPU's ${cpuTop}, and not a near tie (${gap?.toFixed(2)} below it in log-probability)`);
  said.push(`${first}${first === 4 ? ` and ${second}` : ""} greedy ids as NumPy's${first === 4 && second === 3 ? "" : " (then a near tie)"}` +
    `, after a prompt on the CPU ${uploaded}${uploaded === cpuTop ? " as the CPU's" : ` (the CPU's ${cpuTop}, ${gap.toFixed(2)} below it)`}` +
    (first === 4 ? `, the CPU's step between them ${steps.cpuStep === greedy[4] ? "NumPy's" : `${steps.cpuStep} (NumPy's ${greedy[4]})`}` : ""));
  // the keys and values the GPU wrote back of the positions whose inputs were NumPy's (not the CPU's step's: the CPU's)
  const layers = ref.attending ?? ref.header[2];  // (T.545: the attending layers alone hold keys and values)
  const rowsOf = (b64) => {
    const all = floats(b64), out = new Float32Array(layers * positions.length * kvDim);
    for (let l = 0; l < layers; l++) {
      positions.forEach((p, k) => out.set(all.subarray((l * GEN + p) * kvDim, (l * GEN + p + 1) * kvDim), (l * positions.length + k) * kvDim));
    }
    return out;
  };
  // a layer at a time (T.187's lines: the largest of all layers against the largest E16 or Q8 let a deep model's early
  // layers go 10 times their own; the review of T.152): the line of layer l is the prompt's shaders' of the same
  // arithmetic at layer l (T.187: DP4A by Q8's distance), or the run's own prompt's distance at layer l where that is
  // larger (the steps' attention reads its keys and values)
  const [gotK, gotV, wantK, wantV] = [steps.keys, steps.values, ref.greedyKeys, ref.greedyValues].map(rowsOf);
  const rowsAt = (all, l) => all.subarray(l * positions.length * kvDim, (l + 1) * positions.length * kvDim);
  const packed = /DP4A/.test(steps.form ?? "");
  const offs = Array.from({ length: layers }, (_, l) => Math.max(worstRow(rowsAt(gotK, l), rowsAt(wantK, l), kvDim), worstRow(rowsAt(gotV, l), rowsAt(wantV, l), kvDim)));
  const lines = offs.map((_, l) => Math.max(packed ? K_PACKED_STEPS * q8s[l] : Math.max(GPU_LINE, K.float32 * e16s[l]), prompt[l]));
  const ratios = offs.map((off, l) => off / lines[l]);
  const at = worstAt(ratios);
  if (!(ratios[at] <= 1)) failures.push(`the keys and values of the steps' layer ${at} are ${offs[at].toExponential(2)} from NumPy's (its line ${lines[at].toExponential(2)})`);
  said.push(`keys and values of ${positions.length} positions ${Math.max(...offs).toExponential(2)}, layer ${at} ${ratios[at].toFixed(2)} of its line ${lines[at].toExponential(2)}`);
  // (T.152's review) 4 sampled steps in one submission, and one at a time: the same ids to the bit
  if (JSON.stringify(steps.together) !== JSON.stringify(steps.apart)) failures.push(`4 sampled steps together ${JSON.stringify(steps.together)}, one at a time ${JSON.stringify(steps.apart)}`);
  said.push(`4 sampled steps ${JSON.stringify(steps.together)}${JSON.stringify(steps.together) === JSON.stringify(steps.apart) ? " one at a time too" : ""}`);
  // (T.152's review) the CPU's keys and values went up whole: over two other caches the step wrote the same bits
  if (!steps.uploadedSame || JSON.stringify(steps.uploaded) !== JSON.stringify(steps.uploadedTwice)) {
    failures.push(`after a prompt on the CPU over two other caches of the GPU, the step wrote other keys and values (${JSON.stringify(steps.uploaded)}, ${JSON.stringify(steps.uploadedTwice)})`);
  }
  // a stop token: NumPy's second id ends the steps after it
  const until = greedy.indexOf(greedy[1]) + 1, wanted = greedy.slice(0, until);
  if (first >= until && JSON.stringify(steps.stopped) !== JSON.stringify(wanted)) failures.push(`with ${greedy[1]} a stop token the steps were ${JSON.stringify(steps.stopped)}, not ${JSON.stringify(wanted)}`);
  said.push(`a stop token after ${steps.stopped?.length} steps`);
  // the penalty (100 on NumPy's first id, greedy): the largest of NumPy's logits penalized so, or a near tie
  const penalized = logitsOf(0).slice();
  wgsl.penalizeLikeCpu(penalized, steps.penaltyHistory ?? [], 100);
  const wantPenalized = wgsl.argmaxLikeCpu(penalized), gotPenalized = steps.penalized?.[0];
  if (!near(penalized, wantPenalized, gotPenalized)) failures.push(`with a penalty of 100 the step is ${gotPenalized}, NumPy's ${wantPenalized}, and not a near tie`);
  said.push(`penalized ${gotPenalized}${gotPenalized === wantPenalized ? " as NumPy's" : ` (NumPy's ${wantPenalized})`}`);
  // sampled at temperature 2 with a random number of 0.02 and one of 0.98: two tokens of NumPy's nucleus, and not
  // the same (the random numbers reach the GPU; its logits are not NumPy's, so where either lands is not held to it)
  // (T.226: they need not differ where NumPy's most likely token alone has more than 0.9 of the nucleus: the made-up
  // GPT-NeoX with heads of 256 gave 282 and 63 on five runs and 282 twice on one whose prompt went through other
  // shaders, run 36868289185: its first token's share is about 0.98, and 0.98 lands on either side of its border)
  const walk = wgsl.walkLikeCpu(logitsOf(0), 2, 0.999), [low, high] = steps.sampled ?? [], peaked = walk.cumulative[0] > 0.9 * walk.mass;
  if (!(walk.tokens.includes(low?.[0]) && walk.tokens.includes(high?.[0]) && (low[0] !== high[0] || peaked))) {
    failures.push(`sampled at 0.02 and 0.98: ${low?.[0]} and ${high?.[0]}, of NumPy's nucleus of ${walk.tokens.length}, which are to differ`);
  }
  said.push(`sampled ${low?.[0]} and ${high?.[0]}${peaked ? ` (NumPy's first token has ${(walk.cumulative[0] / walk.mass).toFixed(3)} of its nucleus)` : ""}`);
  // T.546: a top-k of 3 and min-p 0.5 at temperature 2, random numbers of 0.02 and 0.98: tokens of NumPy's walk with
  // them (or a near tie of one: the GPU's logits are not NumPy's), not the same where it has more than one and its
  // first token has no more than 0.9 of it; the presence penalty of 100 (greedy) on the sampled ones alone: the largest
  // of NumPy's logits penalized so, or a near tie; 4 steps with all three in one submission and one at a time the same
  if (steps.narrowed) {
    const { topK, minP, presence, together, apart } = steps.narrowed, logits = logitsOf(0);
    const within = (walked, id) => walked.tokens.includes(id) || walked.tokens.some((token) => near(logits, token, id));
    for (const [name, got, walked] of [["a top-k of 3", topK, wgsl.walkLikeCpu(logits, 2, 1, 3)], ["min-p 0.5", minP, wgsl.walkLikeCpu(logits, 2, 1, 0, 0.5)]]) {
      const [a, b] = (got ?? []).map((ids) => ids?.[0]), single = walked.tokens.length === 1 || walked.cumulative[0] > 0.9 * walked.mass;
      if (!(within(walked, a) && within(walked, b) && (a !== b || single))) {
        failures.push(`${name} sampled at 0.02 and 0.98: ${a} and ${b}, of NumPy's ${JSON.stringify(walked.tokens)}${single ? "" : ", which are to differ"}`);
      }
      said.push(`${name} ${a} and ${b} of NumPy's ${walked.tokens.length}`);
    }
    const presenceSaid = [];
    [1, 2].forEach((written, i) => {
      const penalized = logits.slice();
      wgsl.penalizeLikeCpu(penalized, steps.penaltyHistory ?? [], 1, 100, written);
      const want = wgsl.argmaxLikeCpu(penalized), got = presence?.[i]?.[0];
      if (!near(penalized, want, got)) failures.push(`with a presence penalty of 100 on the latest ${written} the step is ${got}, NumPy's ${want}, and not a near tie`);
      presenceSaid.push(`${got}${got === want ? "" : ` (NumPy's ${want})`}`);
    });
    said.push(`presence on the latest 1 and 2 ${presenceSaid.join(" and ")}`);
    if (JSON.stringify(together) !== JSON.stringify(apart)) failures.push(`4 steps with a top-k, min-p and presence together ${JSON.stringify(together)}, one at a time ${JSON.stringify(apart)}`);
    said.push(`4 of them ${JSON.stringify(together)}${JSON.stringify(together) === JSON.stringify(apart) ? " one at a time too" : ""}`);
  } else failures.push("no steps with a top-k, min-p or the presence penalty");
  // T.241: one value of the keys and values a NaN, or an infinity (the harness's brokenValues: the first layer and the
  // last, the first column and the last): every request answered and refused (nothing sampled, T.219's word set)
  if (steps.brokenValues?.cases) {
    const hidden = steps.brokenValues.cases.filter((b) => !(b.answered && !b.failed && b.sampled === 0 && b.notFinite === 1));
    for (const b of hidden) {
      failures.push(`${b.value} in the values of layer ${b.layer} (position ${b.position}, column ${b.column}): ` +
        (b.answered && !b.failed ? `the step sampled ${b.sampled} (id ${b.id}) and its word of logits that are not finite is ${b.notFinite}` : `the request ${b.answered ? "failed" : "was not answered"}`) +
        ", where it is to be refused");
    }
    said.push(`a NaN or an infinity among the values refused in ${steps.brokenValues.cases.length - hidden.length} of ${steps.brokenValues.cases.length}`);
  } else if (steps.brokenValues?.skipped) said.push(`a NaN among the values not tried (${steps.brokenValues.skipped})`);
  // T.219 (2): the GPU's logits made NaN (the harness's refusal()): the request refused whole (nothing of it taken), the GPU
  // stopped, and the status line says the logits were not finite (the word gpu.js passes on, not an id outside the
  // vocabulary or too few sampled: those are what a sampler that did not refuse, or a word that did not arrive, leaves).
  // T.241: the DP4A forms too. Their quantizers (QUANTIZE, NORM_QUANTIZE) took a group's largest with a float max and
  // made its values integers, both of which a device does as it likes for a NaN: on lavapipe and SwiftShader the NaN of
  // the keys and values became finite numbers, the logits stayed finite and 4 ids were taken where the float forms
  // refused (this check said "hidden by DP4A's 8-bit quantizer" then, and did not fail). The quantizers now give such a
  // group a NaN for its scale, by the bits (shaders.js, above QUANTIZE)
  if (steps.nanLogits && !steps.nanLogits.skipped) {
    const { taken, status } = steps.nanLogits;
    const refused = taken === null && NOT_FINITE_STATUS.test(status ?? "");
    if (!refused) {
      failures.push(`logits made NaN: the request gave ${JSON.stringify(taken)} and the status line "${status}", where it is to be refused whole and say the logits were not finite`);
    }
    said.push(refused ? "NaN logits refused" : `NaN logits taken as ${JSON.stringify(taken)}`);
  } else if (steps.nanLogits?.skipped) said.push(`NaN logits not tried (${steps.nanLogits.skipped})`);
  // T.209: the tables were cut where the run asked for it (a vocabulary of 192 rows or more is 3 pieces of 64)
  if (steps.cut && !(steps.pieces > 1)) failures.push(`the tables in ${steps.pieces} piece, not cut`);
  console.log(`  a token by ${steps.form}, its attention by ${steps.attention}${steps.pieces > 1 ? ` (the tables in ${steps.pieces} pieces)` : ""}: ${said.join(", ")}${failures.length ? ` — FAILED\n    - ${failures.join("\n    - ")}` : ""}`);
  return !failures.length;
}

export { unbound, stepsRight };
