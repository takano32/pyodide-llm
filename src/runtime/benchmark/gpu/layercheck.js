// The check of a layer (T175, T187, T225): the layer in JavaScript, what a form's keys, values and stream are held to, and
// the attention of a token at several sizes and positions (T224).
// (T353: a module of /benchmark/'s GPU worker, public/benchmark/gpu.js, which asks for it with its own ?v=<build>)
const { shared, STORAGE, COPY_DST, COPY_SRC, UNIFORM, buffer, floats, run, readBack, scoped } = await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));
const { toHalf, fromHalf, heldHalves, halvesSaid, farthest } = await import(new URL(`halves.js${new URL(import.meta.url).search}`, import.meta.url));
const { layerShape, layerForms, hasSubgroupId, engineTiles, vecAttention, attentionSteps, layerCheck, compiled,
  layerPipes, layerParts, layerAngles, layerState } = await import(new URL(`layerparts.js${new URL(import.meta.url).search}`, import.meta.url));

// T175: where a matrix of a layer takes its input (layerReference's inputs), in the order of the DP4A forms' quantized
// vectors: the normed stream before q, k and v; the attention's output before o; the normed stream before gate and up;
// silu(gate) × up before down
const INPUTS = ["qkv", "o", "ffn", "down"];
// a quantized vector's values (int8 × the scale of its group of 32), as a matrix on DP4A takes it
function dequantized(xq, xs) {
  return Float64Array.from(xq, (value, i) => value * xs[(i / shared.GROUP) | 0]);
}
// T175: how a vector the GPU quantized (xq, xs) holds to quantize_x of x, the reference's values where it was made
// (from the GPU's own inputs before it: they differ as float32 and float64 sums in another order, about 1e-6, and a
// key or value of the position rounded the other way in float16 moves the attention's output by up to about 1e-4 of
// itself): each scale within `line` of the reference's (a scale of another group, a norm left out or read from the
// wrong weights is off by far more), each value within 1, and no more than 1% of them off by 1 (a value on a
// rounding's edge goes either way: about 1e-4 of them). The lines (T175's review): where a norm made the vector (q, k
// and v's; gate and up's) NORMED_SCALE_LINE, for the reference's float64 norm and the GPU's float32 differ by a few ulp
// (1.1e-7 to 4.1e-7 on lavapipe, 2026-09-27) and a mean over n - 1 in place of n moves the scales by 2.4e-4 (the
// stream stays within 3e-7: the quantized integers do not depend on the norm's scale, only the group's scale does);
// elsewhere (o's, down's) QUANTIZED_SCALE_LINE, for the attention's output moves by up to about 1e-4 with a key or value
// rounded the other way in float16 (T225: the reference now takes the GPU's own keys and values of the position where
// each is a float16 next to its own, heldHalves: which of the two a device rounds to is its own choice, and the line
// stays). Returns { wrong: why or null, scale: the worst relative difference of a scale, apart: the values off by 1 or
// more, more: those off by more than 1, of: how many }
const QUANTIZED_SCALE_LINE = 1e-3, NORMED_SCALE_LINE = 3e-5;
const scaleLine = (point) => (point === "qkv" || point === "ffn" ? NORMED_SCALE_LINE : QUANTIZED_SCALE_LINE);
function quantizedOff(x, xq, xs, line) {
  const mine = shared.WGSL.quantizedLikeCpu(Float32Array.from(x));
  let far = false, apart = 0, scale = 0, more = 0;
  mine.xs.forEach((want, g) => {
    const off = want > 0 ? Math.abs(xs[g] - want) / want : xs[g] === 0 ? 0 : Infinity;
    scale = Math.max(scale, off);
    far ||= !(off <= line);
  });
  mine.xq.forEach((value, i) => {
    more += Math.abs(xq[i] - value) > 1;
    apart += xq[i] !== value;
  });
  far ||= more > 0;
  const wrong = far ? "far from quantize_x's" : apart > 0.01 * x.length ? `${apart} of ${x.length} values not quantize_x's` : null;
  return { wrong, scale, apart, more, of: x.length };
}
// The layer in JavaScript (float64 sums), as the CPU's forward pass runs it: what every form is held to. d: the
// check's weights ({w, s} of each matrix), h, norms, keys, values (float16 bits), angles and eps. inputs(point, x): the
// vector a matrix takes where x comes in, at the points INPUTS names (T175: the DP4A forms' x quantized; else x itself).
// halves(which, x): the float16 bits of the position's "keys" or "values" x (T225: the GPU's own where they are a
// float16 next to x, heldHalves; else rounded to the nearest).
// Returns the residual stream after the layer, the float16 bits of the keys and values of the position, and (T225)
// stages: q after RoPE, the attention's output and silu(gate) × up, for a check to say where a form first departs
function layerReference({ dim, hidden, heads, kvHeads, headSize, kvDim }, pos, d, inputs = (point, x) => x, halves = (which, x) => Uint16Array.from(x, toHalf)) {
  const product = ({ w, s }, n, first, rows, x) => {
    const signed = new Int8Array(w.buffer, w.byteOffset, w.length), out = new Float64Array(rows);
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += signed[(first + r) * n + i] * s[((first + r) * n + i) / shared.GROUP | 0] * x[i];
      out[r] = sum;
    }
    return out;
  };
  const normed = (x, at) => {
    let squares = 0;
    for (const value of x) squares += value * value;
    const scale = 1 / Math.sqrt(squares / dim + d.eps);
    return x.map((value, i) => d.norms[at + i] * (scale * value));
  };
  const turned = (vector) => {
    for (let j = 0; j < vector.length; j += 2) {
      const i = (j % headSize) / 2, c = d.angles[i], s = d.angles[headSize / 2 + i], [a, b] = [vector[j], vector[j + 1]];
      vector[j] = a * c - b * s;
      vector[j + 1] = a * s + b * c;
    }
    return vector;
  };
  const h = Float64Array.from(d.h), xb = inputs("qkv", normed(h, 0));
  const q = turned(product(d.qkv, dim, 0, dim, xb)), k = turned(product(d.qkv, dim, dim, kvDim, xb));
  const v = product(d.qkv, dim, dim + kvDim, kvDim, xb);
  const keys = halves("keys", k), values = halves("values", v);
  const cached = (all, row) => (p, i) => fromHalf(p === pos ? row[i] : all[p * kvDim + i]);
  const K = cached(d.keys, keys), V = cached(d.values, values), att = new Float64Array(dim);
  for (let head = 0; head < heads; head++) {
    const kv = Math.floor(head / (heads / kvHeads)) * headSize, scores = [];
    for (let p = 0; p <= pos; p++) {
      let score = 0;
      for (let i = 0; i < headSize; i++) score += q[head * headSize + i] * K(p, kv + i);
      scores.push(score / Math.sqrt(headSize));
    }
    const most = Math.max(...scores), weights = scores.map((score) => Math.exp(score - most)), sum = weights.reduce((a, b) => a + b);
    for (let p = 0; p <= pos; p++) for (let i = 0; i < headSize; i++) att[head * headSize + i] += (weights[p] / sum) * V(p, kv + i);
  }
  const o = product(d.o, dim, 0, dim, inputs("o", att)), h1 = h.map((value, i) => value + o[i]), xb2 = inputs("ffn", normed(h1, dim));
  const gate = product(d.gateUp, dim, 0, hidden, xb2), up = product(d.gateUp, dim, hidden, hidden, xb2);
  const g = gate.map((value, i) => (value / (1 + Math.exp(-value))) * up[i]), down = product(d.down, hidden, 0, dim, inputs("down", g));
  return { h: h1.map((value, i) => value + down[i]), keys, values, stages: { q, att, g } };
}
// The check (T150; T175: the DP4A forms held to the reference fed their own quantized vectors, each held to
// quantize_x, quantizedOff): every form of the layer (Llama's shape: GQA, 33 heads of 64 and 3 of K and V; a width of 2112 = 66
// groups and a hidden width of 2080 = 65, each past one pass of mul_mat_vec's 64 groups a workgroup, so that the x²
// and the rows' sums of the second pass are seen: T150's review, whose three breakings of them the width of 256 let
// through), at position 70 (71 positions, two tiles of the attention), against layerReference. The weights' scales
// go as 1 / sqrt(width) and the stream's large channels as the width (3 in 256), so that the activations, the
// attention's scores and the norm's scale are as they were at a width of 256. The residual stream after it is held to LAYER_LINE of what the layer added to
// it (float32 sums in another order are off by about 1e-6 of it; a wrong index, a norm read from the wrong place, a
// residual left out or gate taken for up by a tenth or more), the key and value of the position to 2e-3 of the largest
// (a float16 rounded the other way is 2^-11 of itself), and the cache's other positions must stay as they were.
// The norm folded into the matrices' read (fusedMatVec) is held by the stream and the eps the check starts from: the
// stream is about ±2 with three channels at ±30 (a real stream has such channels: T92's GPT-2 at 1000× the median),
// so the norm's scale is far from 1 (about 0.28: a scale left out shows, and the sum of x² has a few large terms among
// many small), and the check's eps is about a twelfth of the mean of x² (a model's 1e-5 would hide a wrong or missing
// eps under LAYER_LINE; the timing keeps EPS). Larger channels (±60) make the attention's softmax steep enough that a
// key or value of the position rounded the other way in float16 moves the stream by up to 2e-4 (lavapipe, 2026-09-27);
// at ±30 both forms stay within 1e-6 over 45 draws
// T175 (Fable): the stream's first group of 32 is all zeros, so that the first quantized vector (the normed stream
// before q, k and v) has a group whose scale is 0 (NORM_QUANTIZE's and QUANTIZE's select of 1 / scale: a division by
// 0 there makes NaN, which quantizedOff holds to quantize_x's 0). And the two DP4A fused forms, which differ only in
// NORM_QUANTIZE against RMSNORM then QUANTIZE (the same expressions weight × (s × x), the largest / 127 and the
// rounding, in one dispatch or two), must agree within a few ulp (sameAsNormsApart): the scales of their quantized
// vectors within NORMS_APART_ULPS, the values within 1 (no more than 1% of them off by 1), and where every value is the
// same, the stream within 1e-6 of what the layer added. Not to the bit (T175's review): WGSL lets an implementation
// reassociate operations and fuse them where the result is at least as accurate, and a division is within 2.5 ulp
// (§15.7.5), so a device that compiles the two shaders differently (Metal's fast math, gpuweb #2076) may round a scale
// by an ulp or two and a value on an edge the other way, and still be right. Whether they agreed to the bit goes into
// the verdict as a fact. A mean over n - 1 in NORM_QUANTIZE alone moves its scales by 2.4e-4, far past a few ulp
const LAYER_CHECK = { dim: 2112, hidden: 2080, heads: 33, kvHeads: 3 }, LAYER_CHECK_POS = 70, LAYER_LINE = 1e-3, CACHE_LINE = 2e-3;
const NORMS_APART_ULPS = 4, NORMS_APART_STREAM = 1e-6;
const LAYER_CHECK_OUTLIERS = Math.round((3 * LAYER_CHECK.dim) / 256), LAYER_CHECK_OUTLIER = 30, LAYER_CHECK_EPS = 1;
// The check's data: the layer's shape, the position, the weights, the stream (about ±2 with the outlier channels), the norms'
// weights, the cache before the position, the angles and eps (judgeLayer() and the tests take them as they are)
function layerCheckData() {
  const shape = layerShape(LAYER_CHECK), pos = LAYER_CHECK_POS;
  const data = { ...layerState(shape, pos), angles: layerAngles(shape.headSize, pos), eps: LAYER_CHECK_EPS };
  for (let i = 0; i < LAYER_CHECK_OUTLIERS; i++) {
    data.h[Math.floor((i + 0.5) * shape.dim / LAYER_CHECK_OUTLIERS)] = LAYER_CHECK_OUTLIER * (i % 2 ? -1 : 1);
  }
  data.h.fill(0, 0, shared.GROUP);
  for (const [key, [rows, n]] of Object.entries(shape.matrices)) {
    data[key] = { w: new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s: floats(rows * n / shared.GROUP, 0.01 * Math.sqrt(256 / n)) };
  }
  return { shape, pos, data };
}
const sameBytes = (a, b) => {
  const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  return x.length === y.length && x.every((byte, i) => byte === y[i]);
};
// { ok, bitForBit, ulps: the most a scale is apart, apart: the values off by 1, stream: the stream's difference over
// what the layer added (where every value is the same) }
const sameAs = (got, other, added) => {
  let ulps = 0, apart = 0, far = false, values = 0;
  got.quantized.forEach((q, i) => {
    const bits = new Int32Array(q.xs.buffer, q.xs.byteOffset, q.xs.length), otherBits = new Int32Array(other.quantized[i].xs.buffer, other.quantized[i].xs.byteOffset, q.xs.length);
    // the scales are 0 or more: their bits are in the order of their values
    bits.forEach((b, g) => (ulps = Math.max(ulps, Math.abs(b - otherBits[g]))));
    q.xq.forEach((value, k) => {
      far ||= Math.abs(value - other.quantized[i].xq[k]) > 1;
      apart += value !== other.quantized[i].xq[k];
    });
    values += q.xq.length;
  });
  let stream = 0;
  got.h.forEach((value, k) => (stream = Math.max(stream, Math.abs(value - other.h[k]) / added)));
  const bitForBit = sameBytes(got.h, other.h) && sameBytes(got.keys, other.keys) && sameBytes(got.values, other.values)
    && got.quantized.every((q, i) => sameBytes(q.xq, other.quantized[i].xq) && sameBytes(q.xs, other.quantized[i].xs));
  const ok = ulps <= NORMS_APART_ULPS && !far && apart <= 0.01 * values && (apart > 0 || stream <= NORMS_APART_STREAM);
  return { ok, bitForBit, ulps, apart, stream };
};
// What the check says of a form from what the GPU left of it (got: the stream h, the cache's keys and values as float16 bits,
// on DP4A the quantized vectors, and the stages' q, att and g), in plain JavaScript: the reference, the lines and the
// words (tests/layer-check.mjs gives it a device that is JavaScript too). normsApart: what the DP4A fused form with the
// norms apart left, for the fused form to be held to (within a few ulp), by the attention (T224: the prompt's tiles or
// flash_attn_vec, whose sums go in another order): the one with them apart puts what it left (got) there. Returns the verdict
function judgeLayer(shape, pos, data, form, got, normsApart) {
  // T175: on DP4A the reference takes the GPU's own quantized vectors (a value on a rounding's edge may go either
  // way, and moves a layer's output by more than LAYER_LINE), and each of them is held to quantize_x of the
  // reference's values where it was made (quantizedOff)
  // T225: and the GPU's own keys and values of the position, where each is a float16 next to the reference's
  // (heldHalves: which of the two is the device's choice); the cache's line stays against the reference's nearest
  const made = [], rounded = {};
  const row = (cache) => cache.subarray(pos * shape.kvDim, (pos + 1) * shape.kvDim);
  const want = layerReference(shape, pos, data, form.dp4a ? (point, x) => {
    const i = INPUTS.indexOf(point);
    made[i] = x;
    return dequantized(got.quantized[i].xq, got.quantized[i].xs);
  } : undefined, (which, x) => (rounded[which] = heldHalves(x, row(got[which]))).bits);
  const quantizing = form.dp4a ? INPUTS.map((point, i) => [point, quantizedOff(made[i], got.quantized[i].xq, got.quantized[i].xs, scaleLine(point))]) : [];
  const wrongly = quantizing.filter(([, q]) => q.wrong);
  let added = 0, largestKey = 0, largestValue = 0;
  want.h.forEach((value, i) => (added = Math.max(added, Math.abs(value - data.h[i]))));
  rounded.keys.nearest.forEach((bits) => (largestKey = Math.max(largestKey, Math.abs(fromHalf(bits)))));
  rounded.values.nearest.forEach((bits) => (largestValue = Math.max(largestValue, Math.abs(fromHalf(bits)))));
  let off = 0, keyOff = 0, valueOff = 0, touched = false;
  got.h.forEach((value, i) => (off = Math.max(off, Math.abs(value - want.h[i]) / added)));
  for (let p = 0; p <= pos; p++) {
    for (let i = 0; i < shape.kvDim; i++) {
      const at = p * shape.kvDim + i;
      if (p === pos) {
        keyOff = Math.max(keyOff, Math.abs(fromHalf(got.keys[at]) - fromHalf(rounded.keys.nearest[i])) / largestKey);
        valueOff = Math.max(valueOff, Math.abs(fromHalf(got.values[at]) - fromHalf(rounded.values.nearest[i])) / largestValue);
      } else touched ||= got.keys[at] !== data.keys[at] || got.values[at] !== data.values[at];
    }
  }
  const cache = Math.max(keyOff, valueOff);
  // the DP4A fused form against the one with the norms apart, within a few ulp (undefined where that one was not run)
  let agreed;
  if (form.dp4a && form.fused) {
    const key = form.attention ?? "tiles";
    if (form.normApart) normsApart[key] = got;
    else if (normsApart[key]) agreed = sameAs(got, normsApart[key], added);
  }
  const ok = off < LAYER_LINE && cache < CACHE_LINE && !touched && !wrongly.length && agreed?.ok !== false;
  // T225: the stages in the order the layer runs them, [name, what to say, whether it departed] (a float vector
  // against the reference's, over its largest, departs past LAYER_LINE: no verdict, only where to look), in one
  // line for the report: short where the form is ok, every stage and the first that departed where it is not
  const quantizedAt = (point) => quantizing.filter(([at]) => at === point).map(([, q]) =>
    [`${point}'s quantizing`, `${point} quantized: scales ${q.scale.toExponential(1)}, ${q.apart} of ${q.of} off by 1${q.more ? ` (${q.more} by more)` : ""}`, Boolean(q.wrong)]);
  const float = (name, mine, theirs) => {
    const apart = farthest(mine, theirs);
    return [[name, `${name} ${apart.toExponential(1)}`, !(apart < LAYER_LINE)]];
  };
  const halves = [rounded.keys, rounded.values];
  const order = [...quantizedAt("qkv"), ...float("q", got.q, want.stages.q),
    ["K and V", halvesSaid(halves), halves.some((h) => h.far > 0) || !(cache < CACHE_LINE)], ...float("attention", got.att, want.stages.att),
    ...quantizedAt("o"), ...quantizedAt("ffn"), ...float("silu(gate) × up", got.g, want.stages.g), ...quantizedAt("down"),
    ["the stream", `stream ${off.toExponential(1)}`, !(off < LAYER_LINE)], ...(touched ? [["the cache's other positions", "the cache's other positions written", true]] : [])];
  const first = order.find(([, , departed]) => departed)?.[0] ?? (agreed?.ok === false ? "only against the norms apart" : "none");
  const stages = ok ? `stages: ${order.filter(([name]) => !name.endsWith("quantizing") && name !== "the stream").map(([, said]) => said).join(", ")}`
    : `stages: ${order.map(([, said]) => said).join(", ")}; cache ${cache.toExponential(1)}; first to depart: ${first}`;
  // on DP4A, how each quantized vector held (for CI's logs): the worst scale apart and the values off by 1
  return { worstRelative: Math.max(off, cache), ok, stages,
    stream: off, cache, ...(touched ? { wroteOtherPositions: true } : {}), ...(agreed === undefined ? {} : { sameAsNormsApart: agreed }),
    ...(quantizing.length ? { quantized: quantizing.map(([point, q]) => ({ point, ...q })) } : {}) };
}
async function checkLayer() {
  const { shape, pos, data } = layerCheckData(), verdicts = {};
  const normsApart = {};
  for (const form of layerForms()) {
    if (form.none) continue;
    try {
      const got = await scoped(async (owned) => {
        const pipes = await layerPipes(shape, form);
        const parts = layerParts(shape, pos, 1, owned, data);
        const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
        parts.dispatches(form, pipes, 0).forEach((d) => run(pass, d));
        pass.end();
        const h = new Float32Array(await readBack(encoder, parts.vectors.h, shape.dim * 4));
        const cacheBytes = (pos + 1) * shape.kvDim * 2;
        const back = async (source, bytes) => readBack(shared.device.createCommandEncoder(), source, bytes);
        // T175: the four vectors the DP4A form quantized, each as the matrix after it took it
        const quantized = [];
        if (form.dp4a) {
          for (const [i, n] of [shape.dim, shape.dim, shape.dim, shape.hidden].entries()) {
            quantized.push({ xq: new Int8Array(await back(parts.vectors.quantized[i].xq, n)),
              xs: new Float32Array(await back(parts.vectors.quantized[i].xs, (n / shared.GROUP) * 4)) });
          }
        }
        // T225: what the stages left behind (q after RoPE, the attention's output, silu(gate) × up), to say where a
        // form first departs
        const left = async (source, n) => new Float32Array(await back(source, n * 4));
        return { h, keys: new Uint16Array(await back(parts.vectors.keys, cacheBytes)), values: new Uint16Array(await back(parts.vectors.values, cacheBytes)), quantized,
          q: await left(parts.vectors.q, shape.dim), att: await left(parts.vectors.att, shape.dim), g: await left(parts.vectors.g, shape.hidden) };
      });
      verdicts[layerCheck(form)] = judgeLayer(shape, pos, data, form, got, normsApart);
    } catch (error) {
      verdicts[layerCheck(form)] = { worstRelative: NaN, ok: false, error: String(error?.message ?? error) };
    } finally {
      // the page stops a section that says nothing for 5 minutes: SwiftShader compiles each form's shaders for tens of s
      postMessage({ alive: true });
    }
  }
  shared.layerVerdicts = verdicts;
  return verdicts;
}
// T224: a token's attention, every form the layer rows and the engine may run (the prompt's tiles, flash_attn_vec with
// subgroups where there are and with the lanes of the workgroup standing for a subgroup), on made-up numbers against
// JavaScript's (shaders.js's tokenAttentionData and tokenAttentionOff, as the engine checks them, gpu.js): heads of 64,
// 128 and 256 values (the list's models'), 4 heads of q on 2 of keys and values, a token that reads 40, 70, 300 and 1100
// positions (flash_attn_vec's one part, two, and as many as it takes, of more than one tile each), positions past the
// token's that it must not read, and a steep head of q (a largest taken wrong shows only there). Each head's output no
// farther than TOKEN_ATTENTION_LINE of the largest |value| of its head (the tiles hold the weights in float16 where
// there is shader-f16: in float32 as the layer rows, and (T224's review) as the engine makes them here where that is
// another form). { "a token's attention, <form>": { ok, worstRelative, at } }
const TOKEN_ATTENTION_SIZES = [64, 128, 256], TOKEN_ATTENTION_POSITIONS = [40, 70, 300, 1100], TOKEN_ATTENTION_LINE = 4e-3;
async function checkTokenAttentions() {
  const heads = 4, kvHeads = 2, verdicts = {};
  const { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX } = shared.device.limits;
  const engine = engineTiles(64);
  const forms = [{ name: "the prompt's tiles" }, ...(engine ? [{ name: engine.name, engine: true }] : []),
    ...(hasSubgroupId() ? [true] : []).map(() => ({ name: vecAttention(true).name, attention: "vec", vecSubgroups: true })),
    { name: vecAttention(false).name, attention: "vec", vecSubgroups: false }];
  for (const form of forms) {
    const verdict = { ok: true, worstRelative: 0 };
    try {
      for (const size of TOKEN_ATTENTION_SIZES) {
        const pipes = {};
        if (form.attention === "vec") {
          pipes.vecShape = vecAttention(form.vecSubgroups).shape(size);
          if (pipes.vecShape.none) throw new Error(pipes.vecShape.none);
          pipes.vec = await compiled(shared.WGSL.flashVec(pipes.vecShape));
          pipes.vecReduce = await compiled(shared.WGSL.flashVecReduce(pipes.vecShape));
        } else {
          const flash = form.engine ? engineTiles(size)?.shape
            : shared.WGSL.flashShape({ headSize: size, half: false, subgroups: false, memory, threads: Math.min(maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX) });
          if (!flash) continue;  // (the engine's tiles for this size are the f32 ones: checked in the row above)
          if (flash.none) throw new Error(flash.none);
          pipes.flash = await compiled(shared.WGSL.flashTile(flash));
        }
        for (const positions of TOKEN_ATTENTION_POSITIONS) {
          const data = shared.WGSL.tokenAttentionData({ heads, kvHeads, size, positions, steep: [3] });
          const got = await scoped(async (owned) => {
            const make = (bytes, usage = STORAGE | COPY_DST | COPY_SRC) => {
              const b = buffer(bytes, usage);
              owned.push(b);
              return b;
            };
            const put = (values, usage) => {
              const b = make(values.byteLength, usage);
              shared.device.queue.writeBuffer(b, 0, values);
              return b;
            };
            const uniform = (values) => put(values, UNIFORM | COPY_DST);
            const flashParams = new ArrayBuffer(16);
            new Uint32Array(flashParams, 0, 2).set([heads, kvHeads]);
            new Float32Array(flashParams, 8, 1)[0] = 1 / Math.sqrt(size);
            const vecParams = new Map();
            const u = { step: uniform(new Uint32Array([1, positions - 1, 0, 0])), positions, flash: uniform(new Uint8Array(flashParams)),
              vecParams: (nwg) => vecParams.get(nwg) ?? vecParams.set(nwg, uniform(shared.WGSL.flashVecParams({ headSize: size }, heads, kvHeads, nwg))).get(nwg) };
            const v = { q: put(data.q), att: make(heads * size * 4), parts: make(pipes.vecShape ? shared.WGSL.flashVecPartsBytes(pipes.vecShape, heads) : 16) };
            const cache = { keys: put(data.keys), values: put(data.values) };
            const group = (pipeline, entries) => shared.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
              entries: entries.map(([binding, resource]) => ({ binding, resource: { buffer: resource } })) });
            const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
            attentionSteps(form, pipes, heads, cache, v, u, group).forEach((d) => run(pass, d));
            pass.end();
            return new Float32Array(await readBack(encoder, v.att, heads * size * 4));
          });
          const off = shared.WGSL.tokenAttentionOff(got, data, { heads, kvHeads, size, positions });
          // (a NaN stays the worst: `!(off <= NaN)` is true, and the next size's number would take its place: T224's review)
          if (!Number.isNaN(verdict.worstRelative) && !(off <= verdict.worstRelative)) Object.assign(verdict, { worstRelative: off, at: { headSize: size, positions } });
        }
        postMessage({ alive: true });
      }
      verdict.ok = verdict.worstRelative <= TOKEN_ATTENTION_LINE;
    } catch (error) {
      Object.assign(verdict, { ok: false, worstRelative: NaN, error: String(error?.message ?? error) });
    }
    verdicts[`a token's attention, ${form.name}`] = verdict;
  }
  return verdicts;
}

export { INPUTS, dequantized, NORMED_SCALE_LINE, scaleLine, quantizedOff, layerReference, layerCheckData, judgeLayer,
  checkLayer, checkTokenAttentions };
