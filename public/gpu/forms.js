// The forms of a prompt's shaders on this device, checked against JavaScript, timed and chosen: the attention's tiles
// and the matrices' tiled shaders (T146, T147), with how a matrix's pieces are bound and dispatched.
// (T352: a module of the model's GPU worker, public/gpu.js, which asks for it with its own ?v=<build>)

const { STORAGE, COPY_DST, COPY_SRC, common, within, buffer, uniform, readBack, validated, pipelineOf, bind, dispatch } =
  await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));

// The right tiled shaders are timed together, in turn, on the model's first layer (its seven matrices by a whole block,
// with the quantizations of a packed one): a submission of n passes of the layer and one of 2n, their difference the
// time of n passes (what a submission costs besides its work is in both and drops out: 2.6 to 8.4 ms waited for on
// the owner's Android, T134, where llm-jp-3 150M's gate by 64 tokens is 0.6 ms of work), n doubled until a submission
// of n takes TIMED_MS (at most MOST_PASSES), PAIRS pairs a shader, the shaders in turn within each round (a device that
// warms up or is loaded meanwhile falls on all of them alike: T168's review), the median of each shader's pairs. On a
// fallback adapter (the CPU in the GPU's place: its times are no GPU's) one pair of one pass
const TIMED_MS = 20, MOST_PASSES = 256, PAIRS = 5;

// T148: the adapter and the browser whose shaders the page remembers: another GPU, driver architecture or browser
// version chooses anew (the user agent carries the browser's version)
// the tiled shaders of T146 this device can make (shaders.js's promptForms)
const candidates = ({ device, wgsl, ternary }) => wgsl.devicePromptForms(device, ternary);

// ---- the attention: llama.cpp's flash attention with tiles, with f16 in the workgroup's memory where there is
// shader-f16 and its subgroups where there are (and subgroup_id), else the same without them; each checked first
// against JavaScript (checkAttention), the next one tried where it is wrong
async function chooseAttention(m) {
  const { device, plan, wgsl } = m;
  const half = device.features.has("shader-f16");
  const subgroups = device.features.has("subgroups") && Boolean(navigator.gpu.wgslLanguageFeatures?.has("subgroup_id"));
  const limits = { memory: device.limits.maxComputeWorkgroupStorageSize, threads: threadsOf(device),
    subgroupMin: m.info.subgroupMinSize, subgroupMax: m.info.subgroupMaxSize };
  const tried = [];
  const nameOf = (option) => `llama.cpp flash attention tiles${option.half ? ", f16" : ""}${option.subgroups ? ", subgroups" : ""}`;
  // T148: the one the page remembers for this adapter first (it was the first right one then: the other was wrong)
  const options = [{ half, subgroups }, { half: false, subgroups: false }].filter((o, i) => i === 0 || o.half !== half || o.subgroups !== subgroups)
    .sort((a, b) => (nameOf(b) === m.remembered?.attention) - (nameOf(a) === m.remembered?.attention));
  for (const option of options) {
    const shape = wgsl.flashShape({ headSize: plan.headSize, ...option, ...limits });
    const name = nameOf(option);
    if (plan.force.attention && plan.force.attention !== name) continue;
    if (shape.none) {
      tried.push(`${name}: ${shape.none}`);
      continue;
    }
    try {
      const pipeline = await within(validated(m, () => pipelineOf(m, wgsl.flashTile(shape))), `compiling ${name}`);
      const wrong = await within(checkAttention(m, pipeline), `checking ${name}`);
      if (!wrong) {
        m.attention = { name, pipeline, shape };
        return;
      }
      tried.push(`${name}: ${wrong}`);
    } catch (error) {
      if (error?.late) throw error;
      tried.push(`${name}: ${error?.message ?? error}`);
    }
  }
  throw new Error(`no attention is right on this GPU (${tried.join("; ") || `none named ${plan.force.attention}`})`);
}

// The attention on made-up numbers against JavaScript's: 9 tokens at positions 61 to 69 (two tiles of 4 and one
// token of a third), 4 heads of q on 2 of keys and values, 70 positions (a KV_TILE of 64 and a part of one) of keys and
// values as float16 of random bits between 2^-3 and 4 in size. Each token's output against its softmax over the
// positions up to its own, no more than LINE of the largest |value| of its head: f16 weights in the workgroup's
// memory are within 2^-10 of theirs (each rounded either way, WGSL leaves the direction to the device) and so the
// output within about 1e-3 of the largest value (the sum of the weights is float32), while a wrong mask, head or
// tile is off by a tenth and more.
const LINE = 4e-3;
async function checkAttention(m, pipeline) {
  const { plan } = m, size = plan.headSize, heads = 4, kvHeads = 2, tokens = 9, pos = 61, positions = pos + tokens;
  const kvDim = kvHeads * size, qDim = heads * size, scale = 1 / Math.sqrt(size);
  const q = new Float32Array(tokens * qDim).map(() => Math.random() * 2 - 1);
  const halfBits = () => ((Math.random() < 0.5 ? 0x8000 : 0) | ((12 + ((Math.random() * 5) | 0)) << 10) | ((Math.random() * 1024) | 0));
  const keys = new Uint16Array(positions * kvDim).map(halfBits), values = new Uint16Array(positions * kvDim).map(halfBits);
  const owned = [];
  const make = (data, usage = STORAGE | COPY_DST) => {
    const made = buffer(m, data.byteLength, usage, owned);
    m.device.queue.writeBuffer(made, 0, data);
    return made;
  };
  try {
    const out = buffer(m, tokens * qDim * 4, STORAGE | COPY_SRC, owned);
    const params = new ArrayBuffer(16);
    new Uint32Array(params, 0, 2).set([heads, kvHeads]);
    new Float32Array(params, 8, 1)[0] = scale;
    const group = bind(m, pipeline, [make(q), make(keys), make(values), out, uniform(m, params, owned),
      uniform(m, new Uint32Array([tokens, pos, 0, 0]), owned)]);
    const encoder = m.device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, pipeline, group, heads * Math.ceil(tokens / m.wgsl.FLASH_Q_TILE));
    pass.end();
    m.device.queue.submit([encoder.finish()]);
    const got = new Float32Array(await readBack(m, out, tokens * qDim * 4));
    const k = Float32Array.from(keys, halfToFloat), v = Float32Array.from(values, halfToFloat);
    let worst = 0;
    for (let t = 0; t < tokens; t++) {
      for (let h = 0; h < heads; h++) {
        const kv = Math.floor(h / (heads / kvHeads)) * size, row = t * qDim + h * size, seen = pos + t + 1;
        const scores = new Float64Array(seen);
        for (let p = 0; p < seen; p++) {
          for (let d = 0; d < size; d++) scores[p] += q[row + d] * scale * k[p * kvDim + kv + d];
        }
        const most = Math.max(...scores);
        let sum = 0, largest = 0;
        for (let p = 0; p < seen; p++) sum += (scores[p] = Math.exp(scores[p] - most));
        for (let p = 0; p < positions; p++) for (let d = 0; d < size; d++) largest = Math.max(largest, Math.abs(v[p * kvDim + kv + d]));
        for (let d = 0; d < size; d++) {
          let want = 0;
          for (let p = 0; p < seen; p++) want += scores[p] * v[p * kvDim + kv + d];
          worst = Math.max(worst, Math.abs(got[row + d] - want / sum) / largest);
        }
      }
    }
    return worst <= LINE ? null : `its output is ${worst.toExponential(2)} of the largest value from JavaScript's (line ${LINE})`;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}
function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 31, fraction = h & 1023;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

// ---- the matrices: the tiled shaders of T146 that this device can make (shaders.js's promptForms), each checked on a
// small matrix against JavaScript (checkForm: subgroups, f16 rounding and drivers differ from device to device, and
// only the device can say), the right ones timed on the model's own gate matrix by a whole block, and the fastest
// taken. forms: what each came to (ms, or why not), for the console. plan.force.matrices (tests only): that form alone, untimed.
async function chooseMatrices(m) {
  const { plan } = m;
  let forms = candidates(m);
  if (plan.force.matrices) forms = forms.filter((form) => form.name === plan.force.matrices);
  m.forms = [];
  // T148: the shader the page remembers for this adapter, alone, where it is one this device still makes and it is
  // still right here (a driver may have changed under the same names): no other is compiled or timed. Else all of them
  const kept = !plan.force.matrices && forms.find((form) => !form.none && form.name === m.remembered?.matrices);
  if (kept) {
    try {
      const tiled = { ...kept, remembered: true, pipeline: await within(validated(m, () => pipelineOf(m, kept.code, kept.constants)), `compiling ${kept.name}`) };
      const wrong = await within(checkForm(m, tiled), `checking ${kept.name}`);
      if (!wrong) {
        m.forms.push({ name: kept.name, remembered: true });
        m.form = tiled;
        return;
      }
      m.forms.push({ name: kept.name, none: `remembered, but wrong now: ${wrong}` });
    } catch (error) {
      if (error?.late) throw error;
      m.forms.push({ name: kept.name, none: `remembered, but ${error?.message ?? error}` });
    }
    if (common.stopping) return;
  }
  const right = [];
  for (const form of forms) {
    if (plan.force.quick && right.length) break;  // T148: the page's tests, the first right one untimed
    if (form === kept) continue;  // wrong just now
    if (form.none) {
      m.forms.push({ name: form.name, none: form.none });
      continue;
    }
    try {
      const pipeline = await within(validated(m, () => pipelineOf(m, form.code, form.constants)), `compiling ${form.name}`);
      const tiled = { ...form, pipeline };
      const wrong = await within(checkForm(m, tiled), `checking ${form.name}`);
      if (wrong) m.forms.push({ name: form.name, none: `wrong: ${wrong}` });
      else right.push(tiled);
    } catch (error) {
      if (error?.late) throw error;
      m.forms.push({ name: form.name, none: String(error?.message ?? error) });
    }
    if (common.stopping) return;
  }
  if (!right.length) throw new Error(`no tiled shader is right on this GPU (${m.forms.map((f) => `${f.name}: ${f.none}`).join("; ") || `none named ${plan.force.matrices}`})`);
  const ms = right.length > 1 ? await within(timeForms(m, right), "timing the tiled shaders") : [0];
  right.forEach((form, i) => m.forms.push({ name: form.name, ms: ms[i] }));
  m.form = right[ms.indexOf(Math.min(...ms))];
}
// the threads of a workgroup of one dimension this device takes
const threadsOf = (device) => Math.min(device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX);

// layer l's matrix name by the tokens of from into to (added where add): a bind group of form and its rows for each
// piece (T155), which writes its rows of to (bound from its first row; a token's row of to is the whole matrix's)
const productGroups = (m, form, name, l, from, to, add, owned = m.owned) => {
  const { rows, n, pieces } = m.matrices[name];
  return pieces.map(({ first, rows: count, layers }) => {
    const out = first ? { buffer: to, offset: first * 4, size: to.size - first * 4 } : to;
    return { rows: count, group: bind(m, form.pipeline, [...layers[l], form.packed ? m.xq : from, out,
      uniform(m, new Uint32Array([count, n / 4, n / m.wgsl.GROUP, 0, n, rows, add ? 1 : 0, 0]), owned), m.step, ...(form.packed ? [m.xs] : [])]) };
  });
};
// the workgroups of form for rows by count tokens: the tiles numbered over x, then y (as T146's shaders number them)
function multiply(m, pass, form, group, rows, count) {
  const tiles = Math.ceil(rows / form.tile.rows) * Math.ceil(count / form.tile.tokens);
  const across = Math.min(tiles, m.device.limits.maxComputeWorkgroupsPerDimension);
  dispatch(pass, form.pipeline, group, across, Math.ceil(tiles / across));
}

// T146's check (public/benchmark/gpu/check.js's checkTiled) of a form: 300 rows of 544 (17 groups of 32: a part of a tile
// of rows everywhere), 11 and 70 tokens (a part of a tile of tokens; two or one and a part), and 11 tokens whose x and
// y are wider than the product (xStride 608, yStride 320), each product twice into the same y (the second added:
// shape.add) against JavaScript's (shaders.js's tiledOff). The reason it is wrong, or null
// T232, a ternary form: rows of 640 weights (5 groups of 128 with a scale each, 20 of the activations' groups of 32:
// a weight's scale holds for four steps of the width, and the next is another), the weights random bytes of codes (all
// four codes, the one no file has too: every bit of a word matters), against JavaScript's on the codes' values
// (shaders.js's ternaryValues)
async function checkForm(m, form) {
  const { device, wgsl } = m, rows = 300, ternary = Boolean(form.ternary), n = ternary ? 640 : 544, perRow = n / wgsl.GROUP;
  const scaled = ternary ? 4 * wgsl.GROUP : wgsl.GROUP;  // the weights a scale
  const stored = ternary ? new Uint8Array((rows * n) / 4).map(() => (Math.random() * 256) | 0) : new Int8Array(rows * n).map(() => (Math.random() * 256) | 0);
  const w = ternary ? wgsl.ternaryValues(stored) : stored, s = new Float32Array((rows * n) / scaled).map(() => Math.random() * 0.01);
  for (const { tokens, wider } of [{ tokens: 11, wider: 0 }, { tokens: 70, wider: 0 }, { tokens: 11, wider: 64 }]) {
    const xStride = n + wider, yStride = rows + (wider ? 20 : 0), owned = [];
    const x = new Float32Array(tokens * xStride).map(() => (Math.random() - 0.5) * 2);
    const make = (data, usage = STORAGE | COPY_DST | COPY_SRC) => {
      const made = buffer(m, data.byteLength, usage, owned);
      device.queue.writeBuffer(made, 0, data);
      return made;
    };
    try {
      const io = { xq: buffer(m, tokens * xStride, STORAGE | COPY_SRC, owned), xs: buffer(m, tokens * (xStride / wgsl.GROUP) * 4, STORAGE | COPY_SRC, owned),
        step: uniform(m, new Uint32Array([tokens, 0, 0, 0]), owned) };
      const wb = make(stored), sb = make(s), xb = make(x), y = make(new Float32Array(tokens * yStride));
      const shape = (add) => uniform(m, new Uint32Array([rows, n / 4, perRow, 0, xStride, yStride, add ? 1 : 0, 0]), owned);
      const group = (add) => bind(m, form.pipeline, [wb, sb, form.packed ? io.xq : xb, y, shape(add), io.step, ...(form.packed ? [io.xs] : [])]);
      const groups = await validated(m, () => [group(false), group(true)]);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      if (form.packed) {
        dispatch(pass, m.quantize, bind(m, m.quantize, [xb, io.xq, io.xs, uniform(m, new Uint32Array([n, xStride, 0, 0]), owned), io.step]),
          Math.ceil(n / wgsl.GROUP / 64), tokens);
      }
      groups.forEach((g) => multiply(m, pass, form, g, rows, tokens));
      pass.end();
      device.queue.submit([encoder.finish()]);
      const got = new Float32Array(await readBack(m, y, tokens * yStride * 4));
      const xq = form.packed ? new Int8Array(await readBack(m, io.xq, tokens * xStride)) : null;
      const xs = form.packed ? new Float32Array(await readBack(m, io.xs, tokens * (xStride / wgsl.GROUP) * 4)) : null;
      const { wrong } = wgsl.tiledOff({ w, s, x, got, xq, xs, rows, n, tokens, xStride, yStride, half: form.half, group: scaled });
      if (wrong) return `${wrong} (${tokens} tokens${wider ? ", wider x and y" : ""})`;
    } finally {
      owned.forEach((b) => b.destroy());
    }
  }
  return null;
}

// ms of a pass of the model's first layer with each of forms (see TIMED_MS): the seven matrices by plan.batch tokens,
// each packed one's inputs quantized first
async function timeForms(m, forms) {
  const { device, plan } = m, owned = [], layer = [];
  try {
    device.queue.writeBuffer(m.step, 0, new Uint32Array([plan.batch, 0, 0, 0]));
    const products = [["wq", m.xb, m.q, false, m.quantizeXb], ["wk", m.xb, m.k], ["wv", m.xb, m.v], ["wo", m.xb, m.x, true, m.quantizeAttention],
      ["w1", m.ffnInput, m.gate, false, m.quantizeFfn], ["w3", m.ffnInput, m.up], ["w2", m.gate, m.x, true, m.quantizeGate]]
      .filter(([name]) => m.matrices[name]);
    // (a matrix's input quantized once, before its first piece)
    for (const form of forms) {
      layer.push(await validated(m, () => products.flatMap(([name, from, to, add, quantize]) =>
        productGroups(m, form, name, 0, from, to, add, owned).map((piece, i) => ({ ...piece, quantize: i ? null : quantize })))));
    }
    const submission = async (i, passes) => {
      const form = forms[i], encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (let p = 0; p < passes; p++) {
        for (const { group, rows, quantize } of layer[i]) {
          if (form.packed && quantize) dispatch(pass, m.quantize, quantize.group, quantize.x, plan.batch);
          multiply(m, pass, form, group, rows, plan.batch);
        }
      }
      pass.end();
      const began = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - began;
    };
    const passes = [], differences = forms.map(() => []);
    for (let i = 0; i < forms.length; i++) {
      await submission(i, 1);  // warm
      let n = 1;
      while (!m.fallback && n < MOST_PASSES && (await submission(i, n)) < TIMED_MS) n *= 2;
      passes.push(n);
    }
    for (let round = 0; round < (m.fallback ? 1 : PAIRS); round++) {
      for (let i = 0; i < forms.length; i++) {
        const once = await submission(i, passes[i]), twice = await submission(i, 2 * passes[i]);
        differences[i].push((twice - once) / passes[i]);
      }
    }
    return differences.map((d) => d.sort((a, b) => a - b)[d.length >> 1]);
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

export { TIMED_MS, MOST_PASSES, PAIRS, chooseAttention, LINE, halfToFloat, chooseMatrices, threadsOf, productGroups,
  multiply };
