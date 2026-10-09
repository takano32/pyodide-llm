// The attention of a generated token (T224): llama.cpp's decode form and the prompt's tiles, checked against
// JavaScript, timed and chosen, and their dispatches.
// (T352: a module of the model's GPU worker, public/gpu.js, which asks for it with its own ?v=<build>)

const { STORAGE, COPY_DST, COPY_SRC, common, within, buffer, uniform, readBack, validated, pipelineOf, bind, dispatch } =
  await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));
const { TIMED_MS, MOST_PASSES, PAIRS, LINE, threadsOf } =
  await import(new URL(`forms.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- T224: the attention of a generated token. The candidates: llama.cpp's decode form (shaders.js's flashVec and
// flashVecReduce: the positions split over nwg workgroups a head, then reduced), with subgroups where there are (and
// subgroup_id), and with the lanes of the workgroup standing for a subgroup; and the prompt's tiles (m.attention: one
// row of four used, a workgroup a head, T150's (a)). Each checked against JavaScript (checkTokenAttention), the right
// ones timed (timeTokenAttention), the fastest taken: m.gen.attention, with what each came to (m.gen.attentions).
// plan.force.tokenAttention (tests): that one alone; plan.force.quick: the first right one, untimed. None is
// remembered (T148 remembers the matrices, the prompt's attention and the token's layer): the two small shaders are
// compiled, checked and timed at every start.
const TOKEN_ATTENTION_VEC = "llama.cpp flash_attn_vec", TOKEN_ATTENTION_TILES = "the prompt's attention tiles";
async function chooseTokenAttention(m) {
  const { device, plan, wgsl, gen: g } = m;
  const subgroups = device.features.has("subgroups") && Boolean(navigator.gpu.wgslLanguageFeatures?.has("subgroup_id"));
  const vec = (withSubgroups) => ({ name: `${TOKEN_ATTENTION_VEC}${withSubgroups ? ", subgroups" : ""}`,
    shape: wgsl.flashVecShape({ headSize: plan.headSize, subgroups: withSubgroups, threads: threadsOf(device),
      subgroupMin: m.info.subgroupMinSize, subgroupMax: m.info.subgroupMaxSize }) });
  let candidates = [...(subgroups ? [vec(true)] : []), vec(false), { name: TOKEN_ATTENTION_TILES, tiles: true, pipeline: m.attention.pipeline }];
  if (plan.force.tokenAttention) candidates = candidates.filter((a) => a.name === plan.force.tokenAttention);
  const right = [];
  g.attentions = [];
  for (const a of candidates) {
    if (plan.force.quick && right.length) break;
    if (a.shape?.none) {
      g.attentions.push({ name: a.name, none: a.shape.none });
      continue;
    }
    try {
      if (!a.tiles) {
        a.pipeline = await within(validated(m, () => pipelineOf(m, wgsl.flashVec(a.shape))), `compiling ${a.name}`);
        a.reduce = await within(validated(m, () => pipelineOf(m, wgsl.flashVecReduce(a.shape))), `compiling ${a.name}'s reduce`);
      }
      const wrong = await within(checkTokenAttention(m, a), `checking ${a.name}`);
      if (wrong) g.attentions.push({ name: a.name, none: `wrong: ${wrong}` });
      else right.push(a);
    } catch (error) {
      if (error?.late) throw error;
      g.attentions.push({ name: a.name, none: String(error?.message ?? error) });
    }
    if (common.stopping) return;
  }
  if (!right.length) {
    throw new Error(`no attention of a token is right on this GPU (${g.attentions.map((a) => `${a.name}: ${a.none}`).join("; ") || `none named ${plan.force.tokenAttention}`})`);
  }
  const ms = plan.force.quick || right.length === 1 ? right.map(() => undefined) : await within(timeTokenAttention(m, right), "timing the attention of a token");
  const best = ms.reduce((b, t, i) => (t !== undefined && (ms[b] === undefined || t < ms[b]) ? i : b), 0);
  right.forEach((a, i) => g.attentions.push({ name: a.name, ms: ms[i] }));
  g.attention = right[best];
  // the vec form's parts (every head's, nwg of the most) and its Params a count of parts, made as a run needs them
  if (!g.attention.tiles) {
    g.parts = buffer(m, wgsl.flashVecPartsBytes(g.attention.shape, plan.heads));
    g.params = paramsOf(m, g.attention.shape, plan.heads, plan.kvHeads, m.owned);
  }
}
// flashVec's Params a count of parts, each made once (owned: where they go)
function paramsOf(m, shape, heads, kvHeads, owned) {
  const made = new Map();
  return (nwg) => {
    if (!made.has(nwg)) made.set(nwg, uniform(m, m.wgsl.flashVecParams(shape, heads, kvHeads, nwg), owned));
    return made.get(nwg);
  };
}
// The dispatches of attention a for a token of heads that reads positions: io's q, keys, values into out (flash: the
// tiles' Params; parts and params(nwg): the vec form's parts and Params; step: the token's Step), [pipeline, bind
// group, x, y] each: the tiles a workgroup a head; the vec form nwg a head, then (nwg more than 1) the reduce
function attentionPasses(m, a, io, heads, positions) {
  if (a.tiles) return [[a.pipeline, bind(m, a.pipeline, [io.q, io.keys, io.values, io.out, io.flash, io.step]), heads, 1]];
  const nwg = m.wgsl.flashVecSplits(a.shape, positions), params = io.params(nwg);
  return [[a.pipeline, bind(m, a.pipeline, [io.q, io.keys, io.values, io.parts, io.out, params, io.step]), heads * nwg, 1],
    ...(nwg > 1 ? [[a.reduce, bind(m, a.reduce, [io.parts, io.out, params]), heads, 1]] : [])];
}
// Made-up buffers of an attention of a token over positions (heads of q on kvHeads of keys and values, the model's
// headSize; shaders.js's tokenAttentionData: steep, the heads whose q is steep), the output, the Step of the token at
// positions - 1, the tiles' Params, and the vec form's parts and Params
function attentionIo(m, a, { heads, kvHeads, positions, steep = [] }, owned) {
  const size = m.plan.headSize, { q, keys, values } = m.wgsl.tokenAttentionData({ heads, kvHeads, size, positions, steep });
  const make = (data) => {
    const made = buffer(m, data.byteLength, STORAGE | COPY_DST, owned);
    m.device.queue.writeBuffer(made, 0, data);
    return made;
  };
  const flash = new ArrayBuffer(16);
  new Uint32Array(flash, 0, 2).set([heads, kvHeads]);
  new Float32Array(flash, 8, 1)[0] = 1 / Math.sqrt(size);
  return { data: { q, keys, values }, q: make(q), keys: make(keys), values: make(values), out: buffer(m, heads * size * 4, STORAGE | COPY_SRC, owned),
    step: uniform(m, new Uint32Array([1, positions - 1, 0, 0]), owned), flash: uniform(m, flash, owned),
    ...(a.tiles ? {} : { parts: buffer(m, m.wgsl.flashVecPartsBytes(a.shape, heads), STORAGE, owned), params: paramsOf(m, a.shape, heads, kvHeads, owned) }) };
}
// The check of an attention of a token, on made-up numbers against JavaScript's (shaders.js's tokenAttentionData and
// tokenAttentionOff): 4 heads of q on 2 of keys and values (each head of q to its own), a token that reads 40 positions
// (a KV_TILE of 32 and a part of the next: one part), 70 (two parts), 300 and 1100 (as many parts as the vec form takes,
// each of more than one tile: the reduce over them), with positions past the token's that it must not read, and head
// 3's q steep (a largest taken wrong shows only in a steep softmax). Each head's output against its softmax over the
// positions up to the token's, no farther than LINE of the largest |value| of its head (checkAttention's: the tiles
// hold the weights in float16)
const TOKEN_ATTENTION_LENGTHS = [40, 70, 300, 1100];
async function checkTokenAttention(m, a) {
  const size = m.plan.headSize, heads = 4, kvHeads = 2;
  for (const positions of TOKEN_ATTENTION_LENGTHS) {
    const owned = [];
    try {
      const io = attentionIo(m, a, { heads, kvHeads, positions, steep: [3] }, owned);
      const passes = await validated(m, () => attentionPasses(m, a, io, heads, positions));
      const encoder = m.device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (const [pipeline, group, x, y] of passes) dispatch(pass, pipeline, group, x, y);
      pass.end();
      m.device.queue.submit([encoder.finish()]);
      const got = new Float32Array(await readBack(m, io.out, heads * size * 4));
      const worst = m.wgsl.tokenAttentionOff(got, io.data, { heads, kvHeads, size, positions });
      if (!(worst <= LINE)) return `its output is ${worst.toExponential(2)} of the largest value from JavaScript's at ${positions} positions (line ${LINE})`;
    } finally {
      owned.forEach((b) => b.destroy());
    }
  }
  return null;
}
// ms of the attention of a token (every layer's) with each of right, on made-up numbers: at 128 positions and at 2048
// (the model's context where shorter), all in turn, as the tiled shaders are timed (timeForms: a submission of n
// attentions and one of 2n, n doubled from 1 until n takes TIMED_MS, up to MOST_PASSES layers of them, PAIRS pairs,
// the median), times the layers; the two lengths' ms added. (T224's review: n counts attentions, not layers of them.
// A submission of a whole token's layers at the least made the slowest one long: the prompt's tiles at 2048
// positions, if they take the time of T202's 0.90 ms at 127 positions in proportion (an estimate: the f32 tiles, on
// the owner's Android), 16 × 14 ms a submission and some 3.9 s of timing at every start, 0.5 s so)
async function timeTokenAttention(m, right) {
  const { device, plan } = m, owned = [];
  try {
    const lengths = [...new Set([128, 2048].map((n) => Math.min(n, plan.seqLen)))];
    const items = right.flatMap((a) => lengths.map((positions) => {
      const io = attentionIo(m, a, { heads: plan.heads, kvHeads: plan.kvHeads, positions }, owned);
      return { a, passes: attentionPasses(m, a, io, plan.heads, positions) };
    }));
    const submission = async ({ passes }, n) => {
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (let i = 0; i < n; i++) for (const [pipeline, group, x, y] of passes) dispatch(pass, pipeline, group, x, y);
      pass.end();
      const began = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - began;
    };
    const counts = [], differences = items.map(() => []);
    for (const item of items) {
      await submission(item, 1);  // warm
      let n = 1;
      while (!m.fallback && n < MOST_PASSES * plan.layers && (await submission(item, n)) < TIMED_MS) n *= 2;
      counts.push(n);
    }
    for (let round = 0; round < (m.fallback ? 1 : PAIRS); round++) {
      for (const [i, item] of items.entries()) {
        const once = await submission(item, counts[i]), twice = await submission(item, 2 * counts[i]);
        differences[i].push((twice - once) / counts[i]);
      }
      if (common.stopping) break;
    }
    const ms = differences.map((d) => d.sort((x, y) => x - y)[d.length >> 1] * plan.layers);
    return right.map((a) => items.reduce((sum, item, i) => (item.a === a ? sum + ms[i] : sum), 0));
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

export { chooseTokenAttention, attentionPasses };
