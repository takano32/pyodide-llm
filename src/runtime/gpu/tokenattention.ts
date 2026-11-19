// The attention of a generated token (T224): llama.cpp's decode form and the prompt's tiles, checked against
// JavaScript, timed and chosen, and their dispatches.
// (T352: a module of the model's GPU worker, gpu.ts, which a bundler links it into: T367.3)

import { STORAGE, COPY_DST, COPY_SRC, stopping, within, isLate, buffer, uniform, readBack, validated, pipelineOf, bind, dispatch, whyOf } from "./device.ts";
import { LINE, threadsOf } from "./forms.ts";
import { MOST_PASSES, pairedMs } from "./timing.ts";
import { flashVecShape, flashVecSplits, flashVecPartsBytes, flashVecParams, flashParams, tokenAttentionData,
  tokenAttentionOff } from "../shaders/attention.ts";
import { ENGINE } from "../shaders/engine.ts";
import type { Model, Pass } from "./device.ts";
import type { FlashVecShape } from "../shaders/attention.ts";
import type { Tried } from "../contracts/gpu/protocol.ts";

// T368.6: an attention of a token: llama.cpp's decode form (its shape; its two pipelines once they are compiled), or
// the prompt's tiles (their pipeline, which is there already)
// (T334.2: sizes, its shape and pipelines of each size of heads the layers have, a Gemma 4's two: shape is the model's
// heads', whose parts a head and positions a tile every size shares)
type Sized = { shape: FlashVecShape; pipeline?: GPUComputePipeline; reduce?: GPUComputePipeline };
type Vec = { name: string; tiles?: undefined; shape: FlashVecShape; sizes: Record<number, Sized> };
type Tiles = { name: string; tiles: true; shape?: undefined; pipeline: GPUComputePipeline; reduce?: undefined };
type TokenAttention = Vec | Tiles;
// the model's record of a token's attention, m.tokenAttention, which chooseTokenAttention() sets (T466): the
// attention taken, what each came to, and the vec form's parts and Params (undefined with the tiles)
type Chosen = { attention: TokenAttention; attentions: Tried[]; parts: GPUBuffer | undefined; params: ((nwg: number, window: number, head: number) => GPUBuffer) | undefined };
// the buffers an attention's dispatches bind (parts and params: the vec form's; T261.4.2: window, the layer's, which the
// tiles' flash has in it and the vec form's params are made for; T334.2: head, the size of the layer's heads, whose
// pipelines it takes, the tiles' and the vec form's)
type Io = { q: GPUBuffer; keys: GPUBuffer; values: GPUBuffer; out: GPUBuffer; flash: GPUBuffer; step: GPUBuffer; parts?: GPUBuffer; params?: (nwg: number, window: number, head: number) => GPUBuffer;
  window?: number; head: number };

// ---- T224: the attention of a generated token. The candidates: llama.cpp's decode form (shaders/attention.ts's flashVec and
// flashVecReduce: the positions split over nwg workgroups a head, then reduced), with subgroups where there are (and
// subgroup_id), and with the lanes of the workgroup standing for a subgroup; and the prompt's tiles (m.forms.attention: one
// row of four used, a workgroup a head, T150's (a)). Each checked against JavaScript (checkTokenAttention), the right
// ones timed (timeTokenAttention), the fastest taken: attention, with what each came to (attentions), which
// T466: this module's record, m.tokenAttention (none where a stop was asked: tokenforms.ts's chooseTokens() ends then).
// plan.force.tokenAttention (tests): that one alone; plan.force.quick: the first right one, untimed. T296: the one the
// page remembered for this device's key (plan.remembered.tokenAttention, as T148 remembers the matrices, the prompt's
// attention and the token's layer), alone and untimed where it is still right here; else all of them again
const TOKEN_ATTENTION_VEC = "llama.cpp flash_attn_vec", TOKEN_ATTENTION_TILES = "the prompt's attention tiles";
async function chooseTokenAttention(m: Model): Promise<void> {
  const { device, plan } = m;
  const { subgroups } = m.features;
  // (T334.2: of each size of heads the layers have, a shape and its pipelines: one form, which none of them may refuse)
  const vec = (withSubgroups: boolean): Vec => {
    const shapeOf = (headSize: number) => flashVecShape({ headSize, subgroups: withSubgroups, threads: threadsOf(device), subgroupMin: m.info.subgroupMinSize, subgroupMax: m.info.subgroupMaxSize });
    const sizes = Object.fromEntries(headsOf(m).map((head) => [head, { shape: shapeOf(head) }] as const));
    const shape = shapeOf(plan.headSize), none = Object.values(sizes).find((each) => each.shape.none)?.shape.none;
    return { name: `${TOKEN_ATTENTION_VEC}${withSubgroups ? ", subgroups" : ""}`, shape: none ? { ...shape, none } : shape, sizes };
  };
  let candidates: TokenAttention[] = [...(subgroups ? [vec(true)] : []), vec(false), { name: TOKEN_ATTENTION_TILES, tiles: true, pipeline: m.forms.attention.pipeline }];
  if (plan.force.tokenAttention) candidates = candidates.filter((a) => a.name === plan.force.tokenAttention);
  const right: TokenAttention[] = [];
  const attentions: Tried[] = [];
  // T296: the remembered one, where this device still makes it: compiled and checked alone (a driver may have changed
  // under the same key), and taken untimed where right: no other is compiled, checked or timed (as forms.ts's matrices)
  const kept = !plan.force.tokenAttention && candidates.find((a) => !a.shape?.none && a.name === m.remembered?.tokenAttention);
  if (kept) {
    try {
      const wrong = await madeAndChecked(m, kept);
      if (!wrong) {
        attentions.push({ name: kept.name, remembered: true });
        taken(m, kept, attentions);
        return;
      }
      attentions.push({ name: kept.name, none: `remembered, but wrong now: ${wrong}` });
    } catch (error) {
      if (isLate(error)) throw error;
      attentions.push({ name: kept.name, none: `remembered, but ${whyOf(error)}` });
    }
    if (stopping) return;
  }
  for (const a of candidates) {
    if (plan.force.quick && right.length) break;
    if (a === kept) continue;  // wrong just now
    if (a.shape?.none) {
      attentions.push({ name: a.name, none: a.shape.none });
      continue;
    }
    try {
      const wrong = await madeAndChecked(m, a);
      if (wrong) attentions.push({ name: a.name, none: `wrong: ${wrong}` });
      else right.push(a);
    } catch (error) {
      if (isLate(error)) throw error;
      attentions.push({ name: a.name, none: whyOf(error) });
    }
    if (stopping) return;
  }
  if (!right.length) {
    throw new Error(`no attention of a token is right on this GPU (${attentions.map((a) => `${a.name}: ${a.none}`).join("; ") || `none named ${plan.force.tokenAttention}`})`);
  }
  const ms: (number | undefined)[] = plan.force.quick || right.length === 1 ? right.map(() => undefined) : await within(timeTokenAttention(m, right), "timing the attention of a token");
  const best = ms.reduce((b: number, t, i) => (t !== undefined && (ms[b] === undefined || t < ms[b]) ? i : b), 0);
  right.forEach((a, i) => attentions.push({ name: a.name, ms: ms[i] }));
  taken(m, right[best], attentions);
}
// attention a compiled (the vec form's two pipelines; the tiles' is the prompt's) and checked: why it is wrong, or null
// (T334.2: the vec form's of every size of heads)
async function madeAndChecked(m: Model, a: TokenAttention) {
  for (const sized of a.tiles ? [] : Object.values(a.sizes)) {
    sized.pipeline = await within(validated(m, () => pipelineOf(m, ENGINE.flashVec(sized.shape))), `compiling ${a.name}`);
    sized.reduce = await within(validated(m, () => pipelineOf(m, ENGINE.flashVecReduce(sized.shape))), `compiling ${a.name}'s reduce`);
  }
  return within(checkTokenAttention(m, a), `checking ${a.name}`);
}
// the record of attention taken, with what each tried came to: the vec form's parts (every head's, nwg of the most)
// and its Params a count of parts, made as a run needs them
function taken(m: Model, attention: TokenAttention, attentions: Tried[]) {
  const { plan } = m;
  let parts: Chosen["parts"], params: Chosen["params"];
  if (!attention.tiles) {
    // (T334.2: as many as the widest heads' take, and the Params of each size)
    parts = buffer(m, Math.max(...Object.values(attention.sizes).map(({ shape }) => flashVecPartsBytes(shape, plan.heads))));
    params = paramsOf(m, attention, plan.heads, plan.kvHeads, m.owned);
  }
  m.tokenAttention = { attention, attentions, parts, params };
}
// flashVec's Params a count of parts (T261.4.2: and a window; T334.2: and a size of heads, its shape's), each made once
// (owned: where they go)
function paramsOf(m: Model, a: Vec, heads: number, kvHeads: number, owned: GPUBuffer[]) {
  const made = new Map<string, GPUBuffer>();
  return (nwg: number, window: number, head: number) => {
    const key = `${nwg} ${window} ${head}`;
    if (!made.has(key)) made.set(key, uniform(m, flashVecParams(a.sizes[head].shape, heads, kvHeads, nwg, window), owned));
    return made.get(key)!;
  };
}
// T334.2: the sizes of heads the layers have (a Gemma 4's two; the model's alone where they are all alike, as a layer that
// keeps a state has: an LFM2's)
const headsOf = (m: Model) => [...new Set([m.plan.headSize, ...m.plan.shapes.map(({ head }) => head)])];
// The dispatches of attention a for a token of heads that reads positions: io's q, keys, values into out (flash: the
// tiles' Params; parts and params(nwg): the vec form's parts and Params; step: the token's Step), [pipeline, bind
// group, x, y] each: the tiles a workgroup a head; the vec form nwg a head, then (nwg more than 1) the reduce
// (T334.2: by the pipelines of the layer's size of heads, the prompt's tiles of it or the vec form's)
function attentionPasses(m: Model, a: TokenAttention, io: Io, heads: number, positions: number): Pass[] {
  if (a.tiles) {
    const pipeline = m.forms.attention.pipelines[io.head] ?? a.pipeline;
    return [[pipeline, bind(m, pipeline, [io.q, io.keys, io.values, io.out, io.flash, io.step]), heads, 1]];
  }
  const sized = a.sizes[io.head], nwg = flashVecSplits(sized.shape, positions), params = io.params!(nwg, io.window ?? 0, io.head);
  return [[sized.pipeline!, bind(m, sized.pipeline!, [io.q, io.keys, io.values, io.parts!, io.out, params, io.step]), heads * nwg, 1],
    ...(nwg > 1 ? [[sized.reduce!, bind(m, sized.reduce!, [io.parts!, io.out, params]), heads, 1]] satisfies Pass[] : [])];
}
// Made-up buffers of an attention of a token over positions (heads of q on kvHeads of keys and values, the model's
// headSize; shaders/attention.ts's tokenAttentionData: steep, the heads whose q is steep), the output, the Step of the token at
// positions - 1, the tiles' Params, and the vec form's parts and Params
// (T334.2: of a size of heads, size: each the layers have)
function attentionIo(m: Model, a: TokenAttention, { heads, kvHeads, positions, steep = [], window = 0, size }: { heads: number; kvHeads: number; positions: number; steep?: number[];
  window?: number; size: number }, owned: GPUBuffer[]) {
  const { q, keys, values } = tokenAttentionData({ heads, kvHeads, size, positions, steep });
  const make = (data: GPUAllowSharedBufferSource) => {
    const made = buffer(m, data.byteLength, STORAGE | COPY_DST, owned);
    m.device.queue.writeBuffer(made, 0, data);
    return made;
  };
  return { data: { q, keys, values }, window, head: size, q: make(q), keys: make(keys), values: make(values), out: buffer(m, heads * size * 4, STORAGE | COPY_SRC, owned),
    step: uniform(m, new Uint32Array([1, positions - 1, 0, 0]), owned), flash: uniform(m, flashParams(heads, kvHeads, size, window), owned),
    ...(a.tiles ? {} : { parts: buffer(m, flashVecPartsBytes(a.sizes[size].shape, heads), STORAGE, owned), params: paramsOf(m, a, heads, kvHeads, owned) }) };
}
// The check of an attention of a token, on made-up numbers against JavaScript's (shaders/attention.ts's tokenAttentionData and
// tokenAttentionOff): 4 heads of q on 2 of keys and values (each head of q to its own), a token that reads 40 positions
// (a KV_TILE of 32 and a part of the next: one part), 70 (two parts), 300 and 1100 (as many parts as the vec form takes,
// each of more than one tile: the reduce over them), with positions past the token's that it must not read, and head
// 3's q steep (a largest taken wrong shows only in a steep softmax). Each head's output against its softmax over the
// positions up to the token's, no farther than LINE of the largest |value| of its head (checkAttention's: the tiles
// hold the weights in float16). T261.4.2: a model with layers that attend a window (a Gemma 3's local ones) also at 300
// positions through a window of 37 (TOKEN_ATTENTION_WINDOW: its first position in the middle of a KV_TILE, and parts of
// the vec form that read nothing), whose output is of those 37 positions alone
const TOKEN_ATTENTION_LENGTHS = [40, 70, 300, 1100], TOKEN_ATTENTION_WINDOW = [300, 37];
// (T334.2: each size of heads the layers have)
async function checkTokenAttention(m: Model, a: TokenAttention) {
  const heads = 4, kvHeads = 2;
  const lengths = [...TOKEN_ATTENTION_LENGTHS.map((positions) => [positions, 0]), ...(m.plan.windows.some(Boolean) ? [TOKEN_ATTENTION_WINDOW] : [])];
  for (const [positions, window, size] of lengths.flatMap((each) => headsOf(m).map((head) => [...each, head]))) {
    const owned: GPUBuffer[] = [];
    try {
      const io = attentionIo(m, a, { heads, kvHeads, positions, steep: [3], window, size }, owned);
      const passes = await validated(m, () => attentionPasses(m, a, io, heads, positions));
      const encoder = m.device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (const [pipeline, group, x, y] of passes) dispatch(pass, pipeline, group, x, y);
      pass.end();
      m.device.queue.submit([encoder.finish()]);
      const got = new Float32Array(await readBack(m, io.out, heads * size * 4));
      const worst = tokenAttentionOff(got, io.data, { heads, kvHeads, size, positions, window });
      if (!(worst <= LINE)) {
        return `its output is ${worst.toExponential(2)} of the largest value from JavaScript's at ${positions} positions${window ? ` through a window of ${window}` : ""}${headsOf(m).length > 1 ? ` with heads of ${size}` : ""} (line ${LINE})`;
      }
    } finally {
      owned.forEach((b) => b.destroy());
    }
  }
  return null;
}
// ms of the attention of a token (every layer's) with each of right, on made-up numbers: at 128 positions and at 2048
// (the model's context where shorter), all in turn, as the tiled shaders are timed (timing.ts's pairedMs: n counts
// attentions, up to MOST_PASSES layers of them), times the layers; the two lengths' ms added. (T224's review: n counts attentions, not layers of them.
// A submission of a whole token's layers at the least made the slowest one long: the prompt's tiles at 2048
// positions, if they take the time of T202's 0.90 ms at 127 positions in proportion (an estimate: the f32 tiles, on
// the owner's Android), 16 × 14 ms a submission and some 3.9 s of timing at every start, 0.5 s so)
// (T334.2: each size of heads the layers have, times the layers of it)
async function timeTokenAttention(m: Model, right: TokenAttention[]) {
  const { device, plan } = m, owned: GPUBuffer[] = [];
  try {
    const lengths = [...new Set([128, 2048].map((n) => Math.min(n, plan.seqLen)))], heads = headsOf(m);
    const layersOf = (size: number) => (heads.length === 1 ? plan.layers : plan.shapes.filter(({ head }) => head === size).length);
    const items = right.flatMap((a) => lengths.flatMap((positions) => heads.map((size) => {
      const io = attentionIo(m, a, { heads: plan.heads, kvHeads: plan.kvHeads, positions, size }, owned);
      return { a, layers: layersOf(size), passes: attentionPasses(m, a, io, plan.heads, positions) };
    })));
    const ms = (await pairedMs(m, items, ({ passes }, n) => {
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (let i = 0; i < n; i++) for (const [pipeline, group, x, y] of passes) dispatch(pass, pipeline, group, x, y);
      pass.end();
      return encoder.finish();
    }, MOST_PASSES * plan.layers)).map((each, i) => each * items[i].layers);
    return right.map((a) => items.reduce((sum, item, i) => (item.a === a ? sum + ms[i] : sum), 0));
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

export { chooseTokenAttention, attentionPasses };
export type { TokenAttention, Chosen };
