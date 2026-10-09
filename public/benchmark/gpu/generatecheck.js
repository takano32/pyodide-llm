// The check of the sampling and of the generated tokens (T151, T178, T195, T219): against the CPU's sampling in JavaScript,
// with the logits that are not finite, and a few tokens against the layer in JavaScript.
// (T353: a module of /benchmark/'s GPU worker, public/benchmark/gpu.js, which asks for it with its own ?v=<build>)
const { shared, STORAGE, COPY_DST, COPY_SRC, UNIFORM, MAP_READ, buffer, floats, run, readBack, validated, scoped } = await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));
const { toHalf, heldHalves, halvesSaid, farthest } = await import(new URL(`halves.js${new URL(import.meta.url).search}`, import.meta.url));
const { EPS, layerShape, tokenForm, layerAngles } = await import(new URL(`layerparts.js${new URL(import.meta.url).search}`, import.meta.url));
const { INPUTS, dequantized, NORMED_SCALE_LINE, scaleLine, quantizedOff, layerReference } = await import(new URL(`layercheck.js${new URL(import.meta.url).search}`, import.meta.url));
const { GENERATE_SETTINGS, GENERATE_CHECK, generationPipes, samplerPipes, samplerDispatches, generationParts,
  generationRun, randomsOf, TIED_RUN, tiedLogits, sparseLogits, UNREFUSED, unfiniteLogits, madeUpLogits } = await import(new URL(`generate.js${new URL(import.meta.url).search}`, import.meta.url));

// The check of SAMPLE (T151) against the CPU's sampling in JavaScript (shaders.js's sampleLikeCpu and
// penalizeLikeCpu, which tests/smoke.mjs holds to the kernel): the same logits, history and random number must pick
// the same token. Where a float32 sum in another order moves a border, a token next to it is as right: the token
// passes when it is what the CPU picks, or where the CPU's walk (walkLikeCpu, with top-p as it is and moved by EDGE)
// passes it within EDGE of the mass of the random number's share (a relative 1e-4: about 3 times the worst the GPU's
// float32 sums and exp() can be off by, 501 × 2^-24 ≈ 3.0e-5 of a thread's run and 3.3e-5 with the rest; a wrong
// border moves the draw by a token's probability, 1e-3 of the mass or more), or a token of the same logit (equal probabilities, which the CPU
// takes in no set order); temperature 0 (the most likely token): the first index of the largest logit and, where
// logits come from the GPU's own forward pass, any within `band` of it (and of a token acceptable otherwise). The
// count of tokens that passed by an edge only is in the verdict.
const EDGE = 1e-4;
function acceptable(logits, { temperature, topp }, random, band = 0) {
  const picks = new Set([shared.WGSL.sampleLikeCpu(logits, temperature, topp, random)]), [first] = picks;
  const near = (token) => band > 0 ? logits.forEach((value, i) => Math.abs(value - logits[token]) <= band && picks.add(i)) : picks.add(token);
  // the most likely token: NumPy's first index of the largest logit, exactly where the logits are the same
  if (temperature === 0) {
    near(first);
    return { first, picks };
  }
  const nucleus = topp > 0 && topp < 1;
  for (const p of nucleus ? [topp, topp * (1 - EDGE), topp * (1 + EDGE)] : [topp]) {
    const { tokens, cumulative, mass } = shared.WGSL.walkLikeCpu(logits, temperature, p);
    const low = (random - EDGE) * mass, high = (random + EDGE) * mass;
    tokens.forEach((token, k) => (k ? cumulative[k - 1] : 0) <= high && cumulative[k] >= low && near(token));
  }
  // equal probabilities, which the CPU takes in no set order
  for (const token of [...picks]) logits.forEach((value, i) => value === logits[token] && picks.add(i));
  return { first, picks };
}
// The sampler alone: cases of logits (a vocabulary of 1003, not a multiple of 256, and Llama 3's 128256; normal
// spreads from flat to steep, and tokens far above them), top-p 0.9, 0.5 and none, random numbers at 0, inside and
// just under 1, temperature 0 and 1.3 too, the penalty (the window's tokens among the most likely, a repeated one,
// negative ones, and likely ones older than the window that must not be penalized), and equal logits (tiedLogits: the
// most likely token is the first index of the two at the top; the draw among equals is the exact token in the order of
// the index, as SAMPLE takes them and walkLikeCpu walks them). Then runs of four tokens in one submission (the i-th
// random number for the i-th token, the logits penalized again each time as the CPU's would be), and a run with a stop
// token: it is written, the state stops and nothing after it changes.
async function checkSampling(kind = "one") {
  const pipes = await samplerPipes();
  const cases = [];
  for (const vocab of [1003, 128256]) {
    // (a fallback adapter takes a second or so for each of the big vocabulary's: one spread there)
    for (const spread of vocab === 1003 ? [0.5, 2, 6, 12] : shared.fallback ? [2] : [2, 6]) {
      for (const topp of [0.9, 0.5, 1]) {
        for (const random of [0, Math.random(), 1 - 2 ** -24]) cases.push({ vocab, spread, topp, temperature: 0.7, penalty: 1.3, random });
      }
      cases.push({ vocab, spread, topp: 0.9, temperature: 0, penalty: 1.3, random: 0.5 });
      cases.push({ vocab, spread, topp: 0.9, temperature: 1.3, penalty: 1, random: Math.random() });
    }
  }
  // T191: without a nucleus on flat logits of Llama 3's vocabulary (no peaks), so that the mass is spread over the
  // chunks of the sampling in chunks and the draw passes in a chunk far from the first (a model's logits put nearly
  // all of it on one peak: the mass before that chunk is next to nothing, and forgetting it passed)
  for (const random of [0.3, 0.6, 0.9]) cases.push({ vocab: 128256, spread: 1, peaks: 0, topp: 1, temperature: 0.7, penalty: 1.3, random });
  // T191 (Fable's check): chunks with nothing over the floor, the tokens over it on the chunks' borders (sparseLogits);
  // the most likely token (temperature 0) is the last of the vocabulary (the fourth unpenalized peak: the window holds
  // the three most likely)
  for (const topp of [0.9, 0.5, 1]) {
    for (const random of [0.02, 0.5, 1 - 2 ** -24]) cases.push({ vocab: 128256, sparse: true, topp, temperature: 0.4, penalty: 1.3, random });
  }
  cases.push({ vocab: 128256, sparse: true, topp: 0.9, temperature: 0, penalty: 1.3, random: 0.5 });
  // a history shorter than the window (its empty slots must not count: token 0, among the most likely, is in none)
  for (const spread of [0.5, 2, 6]) for (let i = 0; i < 3; i++) cases.push({ vocab: 1003, spread, topp: 0.9, temperature: 0.7, penalty: 1.3, random: Math.random(), short: true });
  // equal logits (tiedLogits): the draw on the first of the two at the top, on the fifth of the run of 20, and the most
  // likely token (temperature 0)
  // (and over Llama 3's vocabulary, where they cross the chunks of the sampling in chunks: T191)
  for (const vocab of [1003, 128256]) {
    cases.push({ vocab, topp: 0.9, temperature: 0.7, penalty: 1, random: 0.05, ties: true });
    cases.push({ vocab, topp: 0.9, temperature: 0.7, penalty: 1, random: "fifth", ties: true });
    cases.push({ vocab, topp: 0.9, temperature: 0, penalty: 1, random: 0.3, ties: true });
  }
  // T219: logits the sampler must refuse (T195's rule: a NaN anywhere, +inf anywhere, or all -inf: the State's
  // not_finite word set, stopped set, nothing sampled) and ones it must not (UNREFUSED: a few -inf, which the CPU never
  // draws either, and the review's: denormals and -0, one finite logit among -inf): a NaN or +inf at the first token,
  // the last (a thread's last, the vocabulary's last chunk's) and in the middle, with a nucleus and without, at
  // temperature 0 too; the review's: NaNs of other bits, every logit NaN, +inf beside -inf, and a NaN or +inf on the
  // penalty's last token (which the penalty multiplies or divides)
  for (const vocab of [1003, 128256]) {
    const places = [["nan", 0], ["nan", vocab - 1], ["nan", (vocab / 2 | 0) + 1], ["+inf", vocab - 1], ["+inf", 777], ["-inf all", 0],
      ["nan, the sign set", 3], ["nan, signaling", vocab - 2], ["nan, all ones", 500], ["nan all", 0], ["+inf and -inf", 600],
      ["nan in the window", 0], ["+inf in the window", 0]];
    // (a fallback adapter takes a second or so for each of the big vocabulary's: six of them there)
    for (const [unfinite, at] of shared.fallback && vocab > 1003 ? [1, 4, 5, 6, 9, 11].map((i) => places[i]) : places) {
      for (const [topp, temperature] of [[0.9, 0.7], [1, 0.7], [0.9, 0]]) cases.push({ vocab, spread: 2, topp, temperature, penalty: 1.3, random: 0.5, unfinite, at });
    }
    for (const topp of [0.9, 1]) cases.push({ vocab, spread: 2, topp, temperature: 0.7, penalty: 1.3, random: 0.5, unfinite: "-inf some" });
    cases.push({ vocab, spread: 2, topp: 0.9, temperature: 0, penalty: 1.3, random: 0.5, unfinite: "-inf some" });
    for (const unfinite of shared.fallback && vocab > 1003 ? ["one finite"] : ["denormals and -0", "one finite"]) {
      for (const [topp, temperature] of [[0.9, 0.7], [1, 0.7], [0.9, 0]]) cases.push({ vocab, spread: 2, topp, temperature, penalty: 1.3, random: 0.5, unfinite, at: 321 });
    }
  }
  const most = 128256, owned = [];
  let wrong = 0, edge = 0, checked = 0;
  const problems = [];
  try {
    await validated(async () => {
      const make = (bytes, usage = STORAGE | COPY_DST | COPY_SRC) => {
        const b = buffer(bytes, usage);
        owned.push(b);
        return b;
      };
      const logitsBuffer = make(most * 4), probs = make(most * 4), order = make(most * 4), state = make(shared.WGSL.STATE_BYTES),
        chosen = make(16 * 4), randoms = make(16 * 4), settings = make(shared.WGSL.SAMPLING_BYTES, UNIFORM | COPY_DST),
        parts = make(shared.WGSL.samplePartsBytes(most));
      const b = { 0: logitsBuffer, 1: probs, 2: order, 3: state, 4: chosen, 5: randoms, 6: settings, 7: parts };
      // the sampler's dispatches for a vocabulary (the chunks' count is the vocabulary's)
      const lists = new Map();
      const listOf = (vocab) => lists.get(vocab) ?? lists.set(vocab, samplerDispatches(kind, pipes, b, vocab)).get(vocab);
      const back = make(16 * 4 + shared.WGSL.STATE_BYTES, MAP_READ | COPY_DST);
      // steps SAMPLE dispatches in one submission, from the case's history; the ids and the state back
      const sampled = async (c, logits, history, draws, steps, stops = []) => {
        shared.device.queue.writeBuffer(logitsBuffer, 0, logits);
        shared.device.queue.writeBuffer(state, 0, shared.WGSL.samplingState({ token: history[history.length - 1], pos: 40, history }));
        shared.device.queue.writeBuffer(chosen, 0, new Uint32Array(16).fill(SENTINEL_ID));
        shared.device.queue.writeBuffer(randoms, 0, new Float32Array(draws));
        shared.device.queue.writeBuffer(settings, 0, shared.WGSL.samplingSettings({ vocab: c.vocab, temperature: c.temperature, topp: c.topp, penalty: c.penalty, stops }));
        const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
        const list = listOf(c.vocab);
        for (let i = 0; i < steps; i++) list.forEach((d) => run(pass, d));
        pass.end();
        encoder.copyBufferToBuffer(chosen, 0, back, 0, 16 * 4);
        encoder.copyBufferToBuffer(state, 0, back, 16 * 4, shared.WGSL.STATE_BYTES);
        shared.device.queue.submit([encoder.finish()]);
        await back.mapAsync(MAP_READ);
        const words = new Uint32Array(back.getMappedRange().slice(0));
        back.unmap();
        return { ids: words.subarray(0, 16), state: words.subarray(16) };
      };
      // the tokens of a run held to the CPU's, step by step (the CPU's history goes on with the GPU's tokens)
      const judge = (c, logits, history, draws, got, stops = []) => {
        const cpu = Float32Array.from(logits), seen = [...history];
        let k = 0;
        for (; k < draws.length; k++) {
          shared.WGSL.penalizeLikeCpu(cpu, seen, c.penalty);
          const { first, picks } = acceptable(cpu, c, draws[k]);
          checked++;
          // the ties case holds SAMPLE to the token itself: equal probabilities in the order of their index, the
          // random number in the middle of the fifth's share (no border near): a token of the same logit is not enough
          const right = c.ties && c.temperature ? got.ids[k] === first : picks.has(got.ids[k]);
          if (!right) {
            wrong++;
            problems.push(`${c.vocab} spread ${c.spread ?? (c.sparse ? "sparse" : "tied")} top-p ${c.topp} T ${c.temperature} r ${draws[k].toFixed(3)}: ${got.ids[k]}, the CPU ${first}`);
            return;
          }
          if (got.ids[k] !== first) edge++;
          if (stops.includes(got.ids[k])) break;
          seen.push(got.ids[k]);
        }
        // the state: stopped after a stop token (its id written, nothing after it), else at the next position
        const stopped = k < draws.length, taken = stopped ? k + 1 : draws.length;
        const s = got.state, fed = stopped ? k : draws.length;
        const right = s[5] === taken && s[7] === (stopped ? 1 : 0) && s[shared.WGSL.STATE_NOT_FINITE] === 0 && s[1] === 40 + fed && s[6] === history.length + fed &&
          s[4] === (fed ? got.ids[fed - 1] : history[history.length - 1]) && got.ids[taken] === SENTINEL_ID &&
          (fed === 0 || s[8 + ((history.length + fed - 1) % shared.WGSL.REPETITION_WINDOW)] === got.ids[fed - 1]);
        if (!right) {
          wrong++;
          problems.push(`the state after ${draws.length} tokens${stops.length ? " and a stop token" : ""}: ${[...s.subarray(0, 8)].join(" ")}`);
        }
      };
      for (const c of cases) {
        const logits = c.ties ? tiedLogits(c.vocab) : c.sparse ? sparseLogits(c.vocab) : madeUpLogits(c.vocab, c.spread, c.peaks);
        if (c.unfinite) unfiniteLogits(logits, c.unfinite, c.at);
        const ranked = [...logits.keys()].sort((a, b) => logits[b] - logits[a]);
        // 70 tokens: the 6 before the window two of the most likely (3rd and 4th, which must not be penalized), then the
        // window: the three most likely twice each (a repeat is penalized once), early and late in it (both halves of
        // the ring), with likely ones and negative ones between. Short: 20 tokens, and token 0 made one of the most likely
        const history = c.short ? [ranked[1], ranked[5], ...ranked.slice(10, 28)]
          : [ranked[3], ranked[4], ...ranked.slice(-4), ranked[0], ...ranked.slice(5, 35), ranked[1], ...ranked.slice(-30, -10), ranked[0], ranked[2],
            ranked[1], ...ranked.slice(35, 43), ranked[2]];
        if (c.short) logits[0] = (logits[ranked[0]] + logits[ranked[1]]) / 2;
        // (T219's review) the penalty's last token a NaN or +inf: the penalty multiplies or divides it, and it stays one
        if (c.unfinite === "nan in the window") logits[history[history.length - 1]] = NaN;
        if (c.unfinite === "+inf in the window") logits[history[history.length - 1]] = Infinity;
        let random = c.random;
        if (random === "fifth") {
          // the middle of the fifth tied token's share in the CPU's walk (the ties in the order of their index)
          const walk = shared.WGSL.walkLikeCpu(logits, c.temperature, c.topp);
          const k = walk.tokens.map((token, at) => [token, at]).filter(([token]) => logits[token] === TIED_RUN)[4][1];
          random = (walk.cumulative[k - 1] + walk.cumulative[k]) / 2 / walk.mass;
        }
        if (c.unfinite && !UNREFUSED.includes(c.unfinite)) {
          // (T219) refused: the ids untouched (every one of the run's), the state's not_finite and stopped set, nothing
          // sampled, the position, the token and the history's length as they were; a run of 4 all refused too
          for (const draws of [[random], [random, 0.1, 0.9, 0.3]]) {
            const got = await sampled(c, logits, history, draws, draws.length), s = got.state;
            checked++;
            const right = draws.every((_, i) => got.ids[i] === SENTINEL_ID) && s[shared.WGSL.STATE_NOT_FINITE] === 1 && s[7] === 1 && s[5] === 0 && s[1] === 40 &&
              s[4] === history[history.length - 1] && s[6] === history.length;
            if (!right) {
              wrong++;
              problems.push(`${c.vocab} ${c.unfinite} at ${c.at} top-p ${c.topp} T ${c.temperature}, ${draws.length} steps: not refused (ids ${[...got.ids.subarray(0, draws.length)].join(" ")}, state ${[...s.subarray(0, 8)].join(" ")})`);
            }
          }
        } else {
          judge(c, logits, history, [random], await sampled(c, logits, history, [random], 1));
        }
        postMessage({ alive: true });
      }
      // runs: the i-th random number for the i-th token, and a stop token (the run's third token, taken again), second
      // of two and last of eight: the settings hold the stop tokens four to a vec4, and the list's models have up to
      // five (sarashina2.2, CAT-Translate, llm-jp-4: T151's review)
      for (const vocab of [1003, 128256]) {
        const c = { vocab, spread: 1, topp: 0.9, temperature: 0.7, penalty: 1.3 }, logits = madeUpLogits(vocab, 1, 40);
        const history = [...Array(20)].map(() => (Math.random() * vocab) | 0), draws = [0.05, 0.95, 0.5, 0.25];
        const got = await sampled(c, logits, history, draws, 4);
        judge(c, logits, history, draws, got);
        const stop = got.ids[2];
        for (const stops of [[NOT_A_TOKEN, stop], [...Array(shared.WGSL.STOPS_MOST - 1)].map((_, i) => NOT_A_TOKEN + i).concat(stop)]) {
          judge(c, logits, history, draws, await sampled(c, logits, history, draws, 4, stops), stops);
        }
      }
    });
  } catch (error) {
    return { worstRelative: NaN, ok: false, error: String(error?.message ?? error) };
  } finally {
    owned.forEach((b) => b.destroy());
  }
  return { worstRelative: 0, ok: wrong === 0, tokens: checked, edge, ...(problems.length ? { problems: problems.slice(0, 3) } : {}) };
}
const SENTINEL_ID = 0xdeadbeef;
// stop tokens past any vocabulary: the list's before the one that stops the run
const NOT_A_TOKEN = 200000;
// The check of a run of tokens on the GPU (T151): the check's small model (two layers of 256, 4 heads of 64 and 2 of
// K and V, a vocabulary of 1003) generates 6 tokens in one submission from position 5, greedy and sampled (top-p 0.9,
// the penalty), and each token is held to what the CPU's sampling picks from the logits of the same forward pass in
// JavaScript (layerReference's layers, the embedding and the classifier in float64), fed the GPU's tokens before it:
// the first index of the largest logit or one within 1e-4 of the logits' largest magnitude of it (the GPU's float32
// forward pass is off by about 1e-6 of it), or a token acceptable() takes. Then the sampled run again with its fourth
// token as a stop token: the same three tokens, the stop written, and the run stopped there.
// T175: the layers are tokenForm()'s. On DP4A the reference takes the vectors the GPU quantized (each pass records
// them: generationParts), as checkLayer does, and each is held to quantize_x of the reference's values where it was
// made (quantizedOff): quantizing its own values instead, a value on a rounding's edge in float32 goes either way and
// moved this model's logits by up to 6.5% of the largest (a JavaScript emulation, 360 tokens, .tmp/t175/band.mjs)
// T225: and the reference takes the GPU's own keys and values of each position it wrote, where each is a float16 next
// to the reference's (heldHalves: a device that rounds them toward zero, as Direct3D does, is as right as one that
// rounds to the nearest, and its logits were off by more than the band). The verdict says in `steps` how each step's
// logits held (the GPU's, read back, against the reference's over the largest, and whether the most likely token is
// the same) and how the keys and values were rounded: a line where it is ok, every step where it is not
const GENERATION_CHECK_POS = 5, GENERATION_CHECK_TOKENS = 6;
async function checkGeneration() {
  const model = GENERATE_CHECK, shape = layerShape(model), { dim, hidden, kvDim, headSize } = shape, form = tokenForm();
  const pos = GENERATION_CHECK_POS, count = GENERATION_CHECK_TOKENS, positions = pos + count + 1;
  const matrix = ([rows, n], scale) => ({ w: new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s: floats(rows * n / shared.GROUP, scale) });
  const cache = () => new Uint16Array(positions * kvDim).map((_, i) => (i < pos * kvDim ? toHalf((Math.random() - 0.5) * 4) : 0));
  const data = { layers: [...Array(model.layers)].map(() => Object.fromEntries(Object.entries(shape.matrices).map(([key, m]) => [key, matrix(m, 0.01 * Math.sqrt(256 / m[1]))]))),
    classifier: matrix([model.vocab, dim], 0.002), norms: new Float32Array((2 * model.layers + 1) * dim).map(() => 0.5 + Math.random()),
    keys: [...Array(model.layers)].map(cache), values: [...Array(model.layers)].map(cache), eps: EPS };
  const history = [...Array(pos + 1)].map(() => (Math.random() * model.vocab) | 0), draws = randomsOf(count);
  const embedded = (token) => {
    const signed = new Int8Array(data.classifier.w.buffer), h = new Float64Array(dim);
    for (let i = 0; i < dim; i++) h[i] = signed[token * dim + i] * data.classifier.s[(token * dim + i) / shared.GROUP | 0];
    return h;
  };
  // the logits of the GPU's tokens, position by position, in float64 (the caches go on as the GPU's). vectors: on DP4A
  // the GPU's quantized vectors of each pass ([token][layer × 4 + point, then the classifier's] {xq, xs}); quantizing
  // collects where one of them is not quantize_x's of the reference's values
  // caches: the GPU's keys and values after the run ([layer] { keys, values }); rounded collects heldHalves' counts
  const referenceLogits = (tokens, vectors, quantizing, caches, rounded) => {
    const keys = data.keys.map((k) => Uint16Array.from(k)), values = data.values.map((v) => Uint16Array.from(v)), out = [];
    for (let k = 0; k < tokens.length; k++) {
      const p = pos + k;
      let h = embedded(tokens[k]);
      // the GPU's quantized vector in place of x (T175), held to quantize_x of x
      const taken = (at, x, what) => {
        const { xq, xs } = vectors[k][at], { wrong } = quantizedOff(x, xq, xs, at === 4 * model.layers ? NORMED_SCALE_LINE : scaleLine(INPUTS[at % 4]));
        if (wrong) quantizing.push(`token ${k}, ${what}: ${wrong}`);
        return dequantized(xq, xs);
      };
      data.layers.forEach((m, l) => {
        const r = layerReference(shape, p, { ...m, h, norms: data.norms.subarray(2 * l * dim, (2 * l + 2) * dim), keys: keys[l], values: values[l],
          angles: layerAngles(headSize, p), eps: data.eps }, vectors ? (point, x) => taken(4 * l + INPUTS.indexOf(point), x, `layer ${l} ${point}`) : undefined,
        (which, x) => {
          const held = heldHalves(x, caches[l][which].subarray(p * kvDim, (p + 1) * kvDim));
          rounded.push(held);
          return held.bits;
        });
        keys[l].set(r.keys, p * kvDim);
        values[l].set(r.values, p * kvDim);
        h = r.h;
      });
      let squares = 0;
      for (const value of h) squares += value * value;
      const scale = 1 / Math.sqrt(squares / dim + data.eps), final = 2 * model.layers * dim;
      const normed = h.map((value, i) => data.norms[final + i] * (scale * value)), x = vectors ? taken(4 * model.layers, normed, "the classifier") : normed;
      const signed = new Int8Array(data.classifier.w.buffer), logits = new Float32Array(model.vocab);
      for (let r = 0; r < model.vocab; r++) {
        let sum = 0;
        for (let i = 0; i < dim; i++) sum += signed[r * dim + i] * data.classifier.s[(r * dim + i) / shared.GROUP | 0] * x[i];
        logits[r] = sum;
      }
      out.push(logits);
    }
    return out;
  };
  const verdicts = {};
  let tokens = 0, edge = 0;
  const problems = [], steps = [];
  try {
    await scoped(async (owned) => {
      const pipes = await generationPipes(headSize, form);
      const parts = generationParts(model, form, pipes, positions, owned, data);
      for (const settings of [{ temperature: 0, topp: 0.9, penalty: 1 }, { ...GENERATE_SETTINGS }]) {
        const runOnce = async (stops = []) => {
          // (the caches need no reset: a run writes each position's row before its attention reads it)
          parts.reset(shared.WGSL.samplingState({ token: history[pos], pos, history }), draws, shared.WGSL.samplingSettings({ vocab: model.vocab, ...settings, stops }));
          return generationRun(parts, count, count);
        };
        const got = await runOnce();
        const sampled = Math.min(got.state[5], count), fed = [history[pos], ...got.ids.subarray(0, sampled - 1)];
        const { vectors, logits: gpuLogits } = parts.recorded(await readBack(shared.device.createCommandEncoder(), parts.record, parts.recording), sampled);
        // T225: the GPU's keys and values of the positions before the run and of those it wrote
        const caches = [], cached = async (source) => new Uint16Array(await readBack(shared.device.createCommandEncoder(), source, (pos + sampled) * kvDim * 2));
        for (const { keys, values } of parts.caches) caches.push({ keys: await cached(keys), values: await cached(values) });
        const quantizing = [], rounded = [];
        const logits = referenceLogits(fed, vectors, quantizing, caches, rounded), seen = [...history];
        problems.push(...quantizing.slice(0, 3).map((why) => `T ${settings.temperature}, ${why}`));
        const most = (values) => values.reduce((best, value, i) => (value > values[best] ? i : best), 0), held = [];
        let stopped = false;
        for (let k = 0; k < sampled; k++) {
          const cpu = logits[k];
          shared.WGSL.penalizeLikeCpu(cpu, seen, settings.penalty);
          let largest = 0;
          for (const value of cpu) largest = Math.max(largest, Math.abs(value));
          // T225: the step's logits on the GPU (after the penalty, as the reference's) against the reference's
          held.push({ off: farthest(gpuLogits[k], cpu), most: most(gpuLogits[k]) === most(cpu) });
          seen.push(got.ids[k]);
          if (stopped) continue;
          const { first, picks } = acceptable(cpu, settings, draws[k], 1e-4 * largest);
          tokens++;
          if (!picks.has(got.ids[k])) {
            problems.push(`T ${settings.temperature}, token ${k}: ${got.ids[k]}, the CPU ${first}`);
            stopped = true;
          } else if (got.ids[k] !== first) edge++;
        }
        const wrong = stopped || quantizing.length > 0, same = held.filter((step) => step.most).length;
        steps.push(`T ${settings.temperature}: ${wrong ? `logits ${held.map((step) => `${step.off.toExponential(1)}${step.most ? "" : " (another most likely)"}`).join(" ")} of the largest by step`
          : `logits within ${Math.max(...held.map((step) => step.off)).toExponential(1)} of the largest`}, the most likely token the same at ${same} of ${held.length} steps, ${halvesSaid(rounded)}`);
        if (got.state[5] !== count || got.state[1] !== pos + count || got.state[7] !== 0) problems.push(`T ${settings.temperature}: the state ${[...got.state.subarray(0, 8)].join(" ")}`);
        if (settings.temperature) {
          // the fourth token a stop token (where it is not among the first three), sixth in the list of stop tokens
          const stop = got.ids[3];
          if (!got.ids.subarray(0, 3).includes(stop)) {
            // the stop token sixth of six (in the second vec4 of the settings)
            const again = await runOnce([...Array(5)].map((_, i) => NOT_A_TOKEN + i).concat(stop));
            const same = again.ids.subarray(0, 4).every((id, k) => id === got.ids[k]);
            if (!same || again.state[5] !== 4 || again.state[7] !== 1 || again.state[1] !== pos + 3) {
              problems.push(`a stop token: ${[...again.ids.subarray(0, 4)].join(" ")} against ${[...got.ids.subarray(0, 4)].join(" ")}, the state ${[...again.state.subarray(0, 8)].join(" ")}`);
            }
          }
        }
        postMessage({ alive: true });
      }
    });
    verdicts["tokens on the GPU"] = { worstRelative: 0, ok: problems.length === 0, tokens, edge, layer: form.name, steps: `steps: ${steps.join("; ")}`, ...(problems.length ? { problems } : {}) };
  } catch (error) {
    verdicts["tokens on the GPU"] = { worstRelative: NaN, ok: false, error: String(error?.message ?? error) };
  }
  return verdicts;
}

export { checkSampling, checkGeneration };
