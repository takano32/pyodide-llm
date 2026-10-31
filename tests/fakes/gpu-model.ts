// tests/fakes/gpu-model.ts (T.465.3, moved out of tests/gpu-forms-check.ts)
// A made-up model opened on the stand-in WebGPU (tests/fakes/webgpu.mjs) as the model's GPU worker opens one (openDevice,
// handed, upload, block.ts's prepare), with the dispatches of the checks that choose its forms answered as the shaders
// would (tests/fakes/gpu-answers.ts, whose arithmetic is tests/gpu_answers.py's): the prompt's attention, the
// quantization, every tiled form and its texts of llama.cpp's blocks, and (T.465.3) a token's attention. For the checks
// that run those parts: gpu-forms-check.ts, gpu-tokens-check.ts.
//
//   const { m, answers } = await opened({ model, features, spoil, weights })
//
// The same as the worker's: the plan as forward/gpuside.ts hands one (tests/fakes/gpu-plans.ts's), the device's
// options of each form. Simpler: a fallback adapter that the plan allows (forms.ts and tokenattention.ts time each right
// one by a pair of one pass there: every pass of a timing would be answered too), whose timing is no device's.
import { runtimeUrl } from "../tree.mjs";
import { standIn } from "./webgpu.mjs";
import { MODELS, planned, plausible } from "./gpu-plans.ts";
import { gpuAnswers } from "./gpu-answers.ts";

// (the worker's global, as a worker has it)
globalThis.self = globalThis;
globalThis.postMessage = () => {};
const device_ = await import(runtimeUrl("gpu/device.ts"));
const { upload } = await import(runtimeUrl("gpu/weights.ts"));
const block_ = await import(runtimeUrl("gpu/block.ts"));
const { ENGINE } = await import(runtimeUrl("shaders/engine.ts"));
const { flashShape, flashVecShape } = await import(runtimeUrl("shaders/attention.ts"));
const { devicePromptForms } = await import(runtimeUrl("shaders/key.ts"));

// a made-up model on the stand-in (model: one of tests/fakes/gpu-plans.ts's MODELS, a Llama where not said), its forms answered; spoil: a model to put in place of one ("attention", "form",
// "quantize") or null. T.465.2: weights, its matrices' own: "ternary" (all of them), or two types of llama.cpp's blocks,
// those of w2's two layers (the one matrix of rows of 256: dim is 128), the plan's codebooks the converter's after the
// checkpoint; spoil.block: a maker of a block type's model, given the right one and the type
// (a model in place of the right one: of the attention, of its f16 or subgroups texts alone (halves), of the quantization,
// of a token's attention by parts (vec) and of their reduce;
// form and block: makers given the right one, and the block's type)
type Answer = (entries: unknown[], x: number, y: number) => void;
type Spoil = { attention?: Answer; halves?: Answer; quantize?: Answer; vec?: Answer; reduce?: Answer; sample?: Answer; embed?: Answer; normQuantize?: Answer; headNorm?: Answer; normAdd?: Answer; bias?: Answer; fused?: (right: Answer, output: string) => Answer;
  dp4a?: (right: Answer, output: string) => Answer; form?: (right: Answer) => Answer; block?: (right: Answer, packing: string) => Answer };
type Opened = { model?: string; features?: string[]; spoil?: Spoil; weights?: "ternary" | string[] | null; sane?: boolean };
// (T.465.5) the stand-in device with a made-up model's plan and memory, and the answers of every form the device may
// check (registered before the worker makes its record: start.ts makes it in openDevice()); opened() and start's check
export async function made({ model = "llama", features = [], spoil = {}, weights = null, sane = false }: Opened = {}) {
  const answers = await gpuAnswers();
  const { flashTile, quantize, tiled } = answers;
  // (a fallback adapter, which the plan allows as the tests' gpuForce does: forms.ts times each right form by one pair of
  // one pass there, and every pass of a timing would be answered too)
  const { device, adapter, lose } = standIn({ fallback: true, language: ["packed_4x8_integer_dot_product", "subgroup_id"], answer: answers.answer });
  for (const feature of features) device.features.add(feature);
  const { plan, bytes } = planned(MODELS[model]);
  plan.force = { fallback: true };
  // (a ternary matrix in the place of an int8 one, which has room for it: a quarter of the bytes and its scales)
  if (weights === "ternary") for (const matrix of Object.values(plan.matrices)) Object.assign(matrix, { packing: "ternary", ternary: true });
  // (no codebooks where no type is an IQ one: a Q4_K_M's plan has none, T.465.2's review)
  if (Array.isArray(weights)) Object.assign(plan.matrices.w2, { packing: "layers", packings: weights });
  const codebooks = Array.isArray(weights) && weights.some((type) => type.startsWith("IQ")) ? answers.codebooks() : null;
  const memory = { buffer: new SharedArrayBuffer(bytes + (codebooks?.length ?? 0)) };
  new Uint8Array(memory.buffer).forEach((_, i, all) => { all[i] = (i * 131 + 7 * (i >> 8) + 89) & 255; });
  if (codebooks) {
    new Uint8Array(memory.buffer).set(codebooks, bytes);
    plan.codebooks = bytes;
  }
  // (T.465.4, sane: numbers a model could have where a check computes with the model's own: a token's layer)
  if (sane) plausible(plan, memory);
  // the attention's texts the device may make (forms.ts's options), the quantization, and every form's text (the model's
  // own forms: a copy of device.ts's ownOf, which is not exported, of the weights asked for)
  const own = weights === "ternary" ? "ternary" : Array.isArray(weights) ? "blocks" : null;
  const { subgroupMinSize: subgroupMin, subgroupMaxSize: subgroupMax } = adapter.info;
  const limits = { memory: device.limits.maxComputeWorkgroupStorageSize, threads: Math.min(device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX),
    subgroupMin, subgroupMax };
  for (const half of [false, true]) {
    for (const subgroups of [false, true]) {
      const shape = flashShape({ headSize: plan.headSize, half, subgroups, ...limits });
      // (spoil.halves: the f16 or subgroups texts alone, the device's first option)
      if (!shape.none) answers.add(ENGINE.flashTile(shape), ((half || subgroups) && spoil.halves) || spoil.attention || flashTile(plan.headSize));
    }
  }
  // (T.465.3: a token's attention, llama.cpp's decode form with subgroups and without: tokenattention.ts's candidates)
  for (const subgroups of [false, true]) {
    const shape = flashVecShape({ headSize: plan.headSize, subgroups, threads: limits.threads, subgroupMin, subgroupMax });
    if (shape.none) continue;
    answers.add(ENGINE.flashVec(shape), spoil.vec ?? answers.flashVec(plan.headSize));
    answers.add(ENGINE.flashVecReduce(shape), spoil.reduce ?? answers.flashVecReduce);
  }
  // (T.465.4: a token's step: its row of the embedding, every fused matrix's text, the sampler)
  answers.add(ENGINE.EMBED, spoil.embed ?? answers.embed);
  for (const input of ["norm", "plain"]) {
    for (const output of ["rope", "add", "swiglu", "write", "geglu"]) {
      for (const subgroups of [false, true]) {
        const right = answers.fusedMatVec({ norm: input === "norm", output });
        answers.add(ENGINE.fusedMatVec({ input, output, subgroups }), spoil.fused ? spoil.fused(right, output) : right);
      }
    }
  }
  answers.add(ENGINE.SAMPLE, spoil.sample ?? answers.sample);
  answers.add(ENGINE.RMSNORM, answers.rmsnorm());
  // (T.465.6: a Qwen3's norms of the heads, a Gemma 3's norms behind, a Qwen2's biases, RoPE apart from the matrix)
  answers.add(ENGINE.HEAD_NORM, spoil.headNorm ?? answers.rmsnorm({ inPlace: true }));
  answers.add(ENGINE.RMSNORM_ADD, spoil.normAdd ?? answers.rmsnorm({ add: true }));
  answers.add(ENGINE.ADD, spoil.bias ?? answers.bias);
  answers.add(ENGINE.TOKEN_ROPE, answers.tokenRope);
  answers.add(ENGINE.NORM_QUANTIZE, spoil.normQuantize ?? answers.normQuantize);
  // (the prompt's and a token's quantization: block.ts's pipeline of this text. A rotated basis's ROTATE_QUANTIZE is not
  // answered: no check here runs one, and its codes would stay 0)
  answers.add(ENGINE.QUANTIZE, spoil.quantize ?? quantize);
  for (const output of ["rope", "add", "swiglu", "write", "geglu"]) {
    const right = answers.dp4a({ output });
    answers.add(ENGINE.fusedDp4aMatVec({ output, weights: "int8" }), spoil.dp4a ? spoil.dp4a(right, output) : right);
  }
  // (spoil.form: a maker of each form's model, given the right one)
  for (const form of devicePromptForms(device, own)) {
    const right = tiled({ packed: form.packed, packing: form.ternary ? "ternary" : "int8", tile: form.tile });
    answers.add(form.code, spoil.form ? spoil.form(right) : right);
    // (T.465.2: and each text of a type of llama.cpp's blocks beside it)
    for (const [packing, code] of Object.entries(form.blocks ?? {})) {
      const block = tiled({ packed: true, packing, tile: form.tile });
      answers.add(code, spoil.block ? spoil.block(block, packing) : block);
    }
  }
  return { answers, device, adapter, plan, memory, lose };
}
export async function opened(options: Opened = {}) {
  const { answers, plan, memory } = await made(options);
  const m = await device_.openDevice(plan);
  device_.handed(memory, plan);
  await upload(m);
  await block_.prepare(m);
  return { m, answers };
}

// (T.465.4) the block's layers bound with the prompt's form chosen, and its cache made, as start.ts has them before a
// token's forms are chosen
export function blockMade(m: { forms: unknown }) {
  block_.bindLayers(m);
  block_.grow(m, 16);
}
