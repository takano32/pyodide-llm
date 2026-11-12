// The step "ternary" (T270): ternary weights (T232, Ternary Bonsai's) on the GPU beside int8's, for a device to say what
// T232 left unmeasured. Three tables, each timed as the matrix × vector is (T149: the forms in turn, n and 2n of each a
// submission, each dispatch on the next copy of the weights so that no cache of the GPU holds them):
//   - a token's matrix × vector: the engine's DP4A text for small M on int8 weights, on ternary codes made signed as
//     it reads them (the engine's, fusedDp4aMatVec's ternary), and on the codes as they are less the vector's sum
//     (shaders/matvec.ts's TERNARY_SUMS_MATVEC: the fork's and the CPU kernel's form, which the engine did not take). On
//     two shapes: Llama 3.2 1B's w1 (the bandwidth step's) and Ternary Bonsai 8B's;
//   - a prompt's tile: the engine's ORT DP4A 64×64 of int8 and of ternary weights (shaders/prompt.ts's promptForms,
//     with subgroups where they are), 64 tokens a dispatch on Llama 3.2 1B's w1;
//   - a layer of a token: the engine's fused DP4A layer (T175) of int8 and of ternary weights, at Llama 3.2 1B's width.
// The ternary texts are checked against JavaScript first (checkTernary(), which the step "check" runs too, so that CI's
// fallback adapter says whether they are right: T182). A form found wrong is not timed. The layer's ternary form reads
// its matrices by the lines the matrix × vector's check holds (fusedDp4aMatVec's reads are one for every output), and the
// engine's gpu-check holds the whole ternary layer (T232); it is not checked here as a layer.
// The page's table (src/bench/ternary.ts) says the rates: the weights a second, the bytes, and against int8's.
// (A module of /benchmark/'s GPU worker, benchmark/gpu.ts, which a bundler links it into: T367.3)
import { shared, MODELS, gpu, matrixBytes, ternaryBytes, TERNARY_GROUP, buffer, fill, floats, kindOf, placed, bound,
  vectors, destroyVectors, run, readBack, validated, interleaved, scoped, quantizer, STORAGE, COPY_DST, COPY_SRC } from "./device.ts";
import { MATVEC_BYTES, MATVEC_MOST } from "./matvec.ts";
import { LAYER_POS, LAYER_MOST, LAYER_KINDS, layerShape, layerPipes, layerParts, compiled } from "./layerparts.ts";
import type { Dispatch, Kind, MatVecShader, Owned, Placed, PromptShader, Shape, Submission, Thrown, Vectors } from "./device.ts";
import type { Form, MatrixKey, Matrices } from "./layerparts.ts";
import type { Alive, TernaryFound, TernaryRow, Verdicts } from "../../contracts/benchmark.ts";
// a form of a table: what it is called, whether its weights are ternary, the shader (kindOf's) or why it cannot run;
// checkedAs: the name of the form whose verdict of the check it has, where it is another row of that form
type Candidate = { name: string; ternary: boolean; shader: MatVecShader | PromptShader; none?: string; checkedAs?: string };

// the shapes of the matrix × vector, [rows, n] (Ternary Bonsai 8B is Qwen3 8B's shape: hidden 12288, dim 4096)
const SHAPES: [string, Shape][] = [["Llama 3.2 1B w1", [8192, 2048]], ["Ternary Bonsai 8B w1", [12288, 4096]]];
// the tile's shape and tokens (a dispatch of the 64×64 tile over 64 tokens)
const TILE_SHAPE: [string, Shape] = ["Llama 3.2 1B w1", [8192, 2048]], TILE_TOKENS = 64;
const NO_DOT = "no packed int8 dot here (a ternary model stays on the CPU)";

// ---- the forms
const matVecForms = (): Candidate[] => {
  const none = shared.features.packed ? undefined : NO_DOT, rows = shared.WGSL.ORT_DP4A_MATVEC_ROWS;
  const form = (name: string, ternary: boolean, code: string): Candidate => ({ name, ternary, none,
    shader: { name: `T270 ${name}`, code, rows, packed: true, engine: true, none } });
  return [form("int8 (engine)", false, shared.WGSL.fusedDp4aMatVec({ output: "write", weights: "int8" })),
    form("ternary, the codes made signed (engine)", true, shared.WGSL.fusedDp4aMatVec({ output: "write", weights: "ternary" })),
    form("ternary, the codes as they are less Σa", true, shared.WGSL.TERNARY_SUMS_MATVEC)];
};
// the engine's tiles of a model of int8 weights and of ternary ones on this device (promptForms, as the model's GPU
// worker makes them: own null and "ternary"), the DP4A ones
const tileForms = (): Candidate[] => {
  const { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX } = shared.device.limits;
  const asked = { half: shared.features.half, subgroups: shared.features.shuffles, packed: shared.features.packed, memory, threads: Math.min(maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX) };
  const int8 = shared.WGSL.promptForms(asked).filter((form) => form.packed), ternary = shared.WGSL.promptForms({ ...asked, own: "ternary" });
  // (without the packed dot, promptForms has no int8 DP4A form at all: the ternary ones say why)
  return [...int8.map((form) => ({ name: `${form.name}, int8`, ternary: false, shader: form, none: form.none })),
    ...ternary.map((form) => ({ name: form.name, ternary: true, shader: form, none: form.none ?? (shared.features.packed ? undefined : NO_DOT) }))];
};
const verdictOf = (name: string) => `ternary: ${name}`;
// the engine's ternary form once more at the end of a token's matrix × vector table, as bandwidth() times its first
// shader again: the interleaving takes the forms in one order every round, so the second row against the first says what
// that order and the device's drift make of a ratio, which the ratio of the two ternary forms is to be read against
const AGAIN = ", again at the end";
const withControl = (forms: Candidate[]): Candidate[] => [...forms, { ...forms[1], name: forms[1].name + AGAIN, checkedAs: forms[1].name }];

// random ternary codes, four a byte, 0 to 2 each (3, which no file has, is left out of the checks)
function codes(count: number) {
  const out = new Uint8Array(count / 4);
  for (let i = 0; i < out.length; i++) {
    out[i] = ((Math.random() * 3) | 0) | (((Math.random() * 3) | 0) << 2) | (((Math.random() * 3) | 0) << 4) | (((Math.random() * 3) | 0) << 6);
  }
  return out;
}

// ---- the check: the two ternary texts of a matrix × vector as check.ts's checkMatVec holds the engine's (301 rows in
// chunks of 128, widths of 640 and 2176: 5 and 17 groups of 128, 20 and 68 of the vector's 32, one step of ORT's 32
// groups and a part, then more than two; a sentinel past the rows that must stay; the products held to TILED_LINE of
// the sum of their |values| on the vector quantized in JavaScript), and the ternary tiles as checkTiled holds the int8
// ones (tiledOff, with the scales a group of 128). Kept for the step, which times no form found wrong.
const SENTINEL = 7.25, CASES = [{ tokens: 11, wider: 0 }, { tokens: 70, wider: 0 }, { tokens: 11, wider: 64 }];
let checked: Verdicts | undefined;
async function checkTernary(): Promise<Verdicts> {
  await gpu();
  const verdicts: Verdicts = {};
  if (!shared.features.packed) return (checked = verdicts);
  for (const form of matVecForms().filter((one) => one.ternary)) {
    const rows = 301, past = 16;
    try {
      const kind = await kindOf(form.shader);
      let worst = 0, over = false, touched = false;
      for (const n of [640, 2176]) {
        const longest = n + 64, w = codes(rows * n), s = floats(rows * n / TERNARY_GROUP, 0.01), values = shared.WGSL.ternaryValues(w);
        const io = vectors(longest, rows + past), x = floats(longest, 2), { xq, xs } = shared.WGSL.quantizedLikeCpu(x);
        shared.device.queue.writeBuffer(io.x, 0, x);
        shared.device.queue.writeBuffer(io.xq, 0, new Uint8Array(xq.buffer));
        shared.device.queue.writeBuffer(io.xs, 0, xs);
        shared.device.queue.writeBuffer(io.y, 0, new Float32Array(rows + past).fill(SENTINEL));
        const owned: Owned[] = [];
        const got = await validated(async () => {
          const held = placed([rows, n], io, { w, s }, { chunk: 128, ternary: true });
          owned.push(...held.owned);
          const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
          bound(held, io, kind).dispatches.forEach((d) => run(pass, d));
          pass.end();
          return new Float32Array(await readBack(encoder, io.y, (rows + past) * 4));
        }).finally(() => {
          owned.forEach((b) => b.destroy());
          destroyVectors(io);
        });
        for (let r = 0; r < rows; r++) {
          let want = 0, size = 0;
          for (let i = 0; i < n; i++) {
            const product = values[r * n + i] * xq[i] * s[r * (n / TERNARY_GROUP) + Math.floor(i / TERNARY_GROUP)] * xs[Math.floor(i / shared.GROUP)];
            want += product;
            size += Math.abs(product);
          }
          const off = Math.abs(got[r] - want);
          worst = Math.max(worst, off / size);
          over ||= !(off < shared.WGSL.TILED_LINE * size);
        }
        for (let r = rows; r < rows + past; r++) touched ||= got[r] !== SENTINEL;
      }
      verdicts[verdictOf(form.name)] = { worstRelative: worst, ok: !over && !touched, ...(touched ? { wrotePastTheRows: true } : {}) };
    } catch (error) {
      verdicts[verdictOf(form.name)] = { worstRelative: NaN, ok: false, error: String((error as Thrown)?.message ?? error) };
    } finally {
      postMessage({ alive: true } satisfies Alive);
    }
  }
  const rows = 300, n = 640, w = codes(rows * n), s = floats(rows * n / TERNARY_GROUP, 0.01), values = shared.WGSL.ternaryValues(w);
  for (const form of tileForms().filter((one) => one.ternary && !one.none)) {
    try {
      const kind = await kindOf(form.shader);
      let worst = 0, over = false, far = false, apart = 0, count = 0;
      for (const { tokens, wider } of CASES) {
        const xStride = n + wider, yStride = rows + (wider ? 20 : 0);
        const io = vectors(xStride, yStride, tokens), x = floats(tokens * xStride, 2);
        shared.device.queue.writeBuffer(io.x, 0, x);
        const owned: Owned[] = [];
        const [got, xq, xs] = await validated(async () => {
          // twice into the same y, the second added (shape.add), as checkTiled has it
          const made = [placed([rows, n], io, { w, s }, { chunk: 100, ternary: true }), placed([rows, n], io, { w, s }, { chunk: 100, add: true, ternary: true })];
          const quantize = quantizer(io, n);
          owned.push(...made.flatMap((m) => m.owned), ...quantize.owned);
          const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
          run(pass, quantize.dispatch);
          made.forEach((m) => bound(m, io, kind).dispatches.forEach((d) => run(pass, d)));
          pass.end();
          return [new Float32Array(await readBack(encoder, io.y, tokens * yStride * 4)),
            new Int8Array(await readBack(shared.device.createCommandEncoder(), io.xq, tokens * xStride)),
            new Float32Array(await readBack(shared.device.createCommandEncoder(), io.xs, tokens * (xStride / shared.GROUP) * 4))] as const;
        }).finally(() => {
          owned.forEach((b) => b.destroy());
          destroyVectors(io);
        });
        const off = shared.WGSL.tiledOff({ w: values, s, x, got, quantized: { xq, xs }, rows, n, tokens, xStride, yStride, half: false, group: TERNARY_GROUP });
        worst = Math.max(worst, off.worst);
        over ||= Boolean(off.wrong);
        far ||= off.far;
        apart += off.apart;
        count += off.values;
      }
      verdicts[verdictOf(form.name)] = { worstRelative: worst, ok: !over && !far && apart <= 0.01 * count, apart: apart / count, far };
    } catch (error) {
      verdicts[verdictOf(form.name)] = { worstRelative: NaN, ok: false, error: String((error as Thrown)?.message ?? error) };
    } finally {
      postMessage({ alive: true } satisfies Alive);
    }
  }
  return (checked = verdicts);
}

// ---- the times
// what n dispatches of each (a unit of dispatches on the next copy each time) take, submitted and waited for
const submitting = (units: Dispatch[][]): Submission => {
  let next = 0;
  return async (n: number) => {
    const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
    for (let i = 0; i < n; i++) units[next++ % units.length].forEach((d) => run(pass, d));
    pass.end();
    const began = performance.now();
    shared.device.queue.submit([encoder.finish()]);
    await shared.device.queue.onSubmittedWorkDone();
    return performance.now() - began;
  };
};
// the rows given, each with what submits it (or none, error: why not), timed in turn; most: the n a submission takes
// at the most
async function timeRows(timed: { row: TernaryRow; submission?: Submission }[], most: number): Promise<TernaryRow[]> {
  const running = timed.filter((t) => t.submission);
  try {
    const results = await validated(() => interleaved(running.map((t) => t.submission!), most));
    running.forEach((t, i) => {
      const r = results[i];
      Object.assign(t.row, r.error ? { error: r.error } : { ms: r.ms! / r.dispatches!, dispatches: r.dispatches, ...(r.ratio ? { ratio: r.ratio } : {}), ...(r.unsteady ? { unsteady: true } : {}) });
    });
  } catch (error) {
    running.forEach((t) => (t.row.error = String((error as Thrown)?.message ?? error)));
  }
  return timed.map((t) => t.row);
}
// a form that cannot be timed: why (it cannot run here, or the check found it wrong or could not run it)
const notTimed = (form: Candidate): Pick<TernaryRow, "none" | "wrong"> | undefined => {
  if (form.none) return { none: form.none };
  const verdict = form.ternary ? checked?.[verdictOf(form.checkedAs ?? form.name)] : undefined;
  if (verdict && !verdict.ok) return { wrong: verdict.error ?? `${verdict.worstRelative.toExponential(2)} of the products off JavaScript's` };
  return undefined;
};

// a table of matrices: every form on copies of its weights that make MATVEC_BYTES, x of tokens tokens
async function matrices(what: "matrix" | "tile", forms: Candidate[], [label, [rows, n]]: [string, Shape], tokens: number): Promise<TernaryRow[]> {
  return scoped(async (owned) => {
    const io: Vectors = vectors(n, rows, tokens);
    owned.push({ destroy: () => destroyVectors(io) });
    const held = new Map<boolean, Placed[]>();
    const weightsOf = (ternary: boolean) => {
      if (!held.has(ternary)) {
        const bytes = ternary ? ternaryBytes([rows, n]) : matrixBytes([rows, n]);
        const copies = shared.fallback ? 1 : Math.ceil(MATVEC_BYTES / bytes);
        held.set(ternary, [...Array(copies)].map(() => {
          const one = placed([rows, n], io, undefined, { ternary });
          owned.push(...one.owned);
          return one;
        }));
      }
      return held.get(ternary)!;
    };
    const timed: { row: TernaryRow; submission?: Submission }[] = [];
    for (const form of forms) {
      const row: TernaryRow = { what, form: form.name, ternary: form.ternary, shape: label, tokens, weights: rows * n,
        bytes: form.ternary ? ternaryBytes([rows, n]) : matrixBytes([rows, n]) };
      const not = notTimed(form);
      if (not) {
        timed.push({ row: { ...row, ...not } });
        continue;
      }
      try {
        const kind = await kindOf(form.shader) as Kind;
        const copies = weightsOf(form.ternary);
        await shared.device.queue.onSubmittedWorkDone();
        timed.push({ row: { ...row, copies: copies.length }, submission: submitting(await validated(() => copies.map((one) => bound(one, io, kind).dispatches))) });
      } catch (error) {
        timed.push({ row: { ...row, error: String((error as Thrown)?.message ?? error) } });
      } finally {
        postMessage({ alive: true } satisfies Alive);
      }
    }
    return timeRows(timed, MATVEC_MOST);
  });
}

// T175's fused DP4A layer of a token at Llama 3.2 1B's width, its matrices int8 and ternary (fusedDp4aMatVec's ternary
// texts in the place of the int8 ones: q, k and v with RoPE and the cache, o and down with the add, gate and up with
// SwiGLU), each on copies of its own that make MATVEC_BYTES, the attention the prompt's tiles at LAYER_POS as in layer()
async function layers(): Promise<TernaryRow[]> {
  const shape = layerShape(MODELS["Llama 3.2 1B"]);
  const form: Form = { ...LAYER_KINDS.find((kind) => kind.name === "DP4A, fused (T175)")!, subgroups: false };
  const label = "Llama 3.2 1B, a layer";
  const keys = Object.keys(shape.matrices) as MatrixKey[];
  const sized = (ternary: boolean) => keys.reduce((sum, key) => sum + (ternary ? ternaryBytes : matrixBytes)(shape.matrices[key]), 0);
  const rowOf = (ternary: boolean): TernaryRow => ({ what: "layer", form: `DP4A, fused (T175), ${ternary ? "ternary (engine)" : "int8 (engine)"}`, ternary, shape: label,
    tokens: 1, bytes: sized(ternary), weights: keys.reduce((sum, key) => sum + shape.matrices[key][0] * shape.matrices[key][1], 0) });
  // (the ternary layer's matrices read their codes by the lines of the engine's matrix × vector: its check's verdict)
  const signed = matVecForms()[1];
  if (!shared.features.packed) return [false, true].map((ternary) => ({ ...rowOf(ternary), none: NO_DOT }));
  return scoped(async (owned) => {
    const parts = layerParts(shape, LAYER_POS, 0, owned);
    const make = (bytes: number) => {
      const b = buffer(bytes, STORAGE | COPY_DST | COPY_SRC);
      owned.push(b);
      return b;
    };
    // a copy of the layer's four matrices, random (codes 0 to 3 where ternary: the time is the same)
    const copyOf = (ternary: boolean): Matrices => Object.fromEntries(keys.map((key) => {
      const [rows, n] = shape.matrices[key], w = make(ternary ? rows * n / 4 : rows * n), scales = rows * n / (ternary ? TERNARY_GROUP : shared.GROUP);
      fill(w, ternary ? rows * n / 4 : rows * n);
      const s = make(scales * 4);
      shared.device.queue.writeBuffer(s, 0, floats(scales, 0.002));
      return [key, { w, s, rows }];
    })) as Matrices;
    const int8Pipes = await layerPipes(shape, form);
    const timed: { row: TernaryRow; submission?: Submission }[] = [];
    for (const ternary of [false, true]) {
      const row = rowOf(ternary);
      const not = ternary ? notTimed(signed) : undefined;
      if (not) {
        timed.push({ row: { ...row, ...not } });
        continue;
      }
      try {
        const pipes = ternary ? { ...int8Pipes, qkv: await compiled(shared.WGSL.fusedDp4aMatVec({ output: "rope", weights: "ternary" })),
          add: await compiled(shared.WGSL.fusedDp4aMatVec({ output: "add", weights: "ternary" })),
          glu: await compiled(shared.WGSL.fusedDp4aMatVec({ output: "swiglu", weights: "ternary" })) } : int8Pipes;
        const copies = shared.fallback ? 1 : Math.ceil(MATVEC_BYTES / row.bytes!);
        const each = await validated(async () => [...Array(copies)].map(() => parts.dispatches(form, pipes, 0, copyOf(ternary))));
        await shared.device.queue.onSubmittedWorkDone();
        timed.push({ row: { ...row, copies, layerDispatches: each[0].length }, submission: submitting(each) });
      } catch (error) {
        timed.push({ row: { ...row, error: String((error as Thrown)?.message ?? error) } });
      } finally {
        postMessage({ alive: true } satisfies Alive);
      }
    }
    return timeRows(timed, LAYER_MOST);
  });
}

async function ternary(): Promise<TernaryFound> {
  await gpu();
  checked ??= await checkTernary();
  if (shared.fallback) return { rows: [], none: "not on a fallback adapter (its times are no GPU's: the check above says whether the shaders are right)" };
  const rows: TernaryRow[] = [];
  const step = async (made: () => Promise<TernaryRow[]>, what: TernaryRow["what"], shape: string) => {
    try {
      rows.push(...await made());
    } catch (error) {
      rows.push({ what, form: "every form", ternary: false, shape, tokens: 0, error: String((error as Thrown)?.message ?? error) });
    }
  };
  for (const shape of SHAPES) await step(() => matrices("matrix", withControl(matVecForms()), shape, 1), "matrix", shape[0]);
  await step(() => matrices("tile", tileForms(), TILE_SHAPE, TILE_TOKENS), "tile", TILE_SHAPE[0]);
  await step(layers, "layer", "Llama 3.2 1B, a layer");
  return { rows, pos: LAYER_POS };
}

export { checkTernary, ternary };
