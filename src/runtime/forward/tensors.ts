// forward/tensors.ts (T.641.2): the model's tensors as an engine reads them, by their names: floats(), the address of
// a float32 copy (widened once where the file holds another kind), and matrix(), a matrix as the kernels multiply by it
// (packed with its scales and the corrections of relaxed SIMD, the layers of a stack of its own kinds, on the GPU alone,
// or float32). Made once by createForward() (engine.ts), which hands what it reads to the kinds of layers (core) and to
// the GPU's side. They take of the memory as they are first asked for a name, so the order of what the engine asks is
// the order of the memory. A module of src/runtime/forward.ts.

import { KIND, packingOf } from "../jobs.ts";
import { halfToFloat } from "./memory.ts";
import { GPU_ALONE } from "./alone.ts";
import type { Kernels, Packed, Packing, Ready } from "../jobs.ts";
import type { Direct } from "./alone.ts";
import type { Held, Room } from "./kinds.ts";
import type { Derived, Part, Plan, Tensor, TensorName } from "../contracts/plan.ts";

// a tensor's parts: a stack whose layers are each of a kind of their own is its layers (T.363.5), any other is itself
export const partsOf = (t: Tensor): Part[] => t.layers ?? [t as Part];

// What the readers are made with: the plan, the checkpoint's address in the memory, the memory (room), the kernels
// (relaxed: null where the engine runs without relaxed SIMD), direct (T.156: the layers on the GPU alone), and the tables
// of llama.cpp's blocks and the scratch of their signed tile (blocks.ts's blockTables(); 0: the model has none)
export type Reading = { plan: Plan; base: number; room: Room; k: Kernels; relaxed: Kernels | null; direct?: Direct; tables: number; spread: number };

export function tensorsOf({ plan, base, room, k, relaxed, direct, tables, spread }: Reading) {
  const T = plan.tensors, derived: Derived = plan.derived ?? {}, { alloc } = room;
  const count = (t: Tensor) => t.shape.reduce((a, b) => a * b, 1);
  // a float32 copy of a tensor: float16 converted; a packed one (int8, int6, ternary: jobs.ts's PACKED says how each
  // weight is read) times the scale of its group (Math.fround is the float32 product NumPy computes); T.363.5: a stack
  // whose layers are each of a kind of their own layer by layer, and rows of llama.cpp's blocks by their kernel
  function widen(t: Tensor) {
    const n = count(t), at = alloc(n * 4), parts = partsOf(t), each = n / parts.length, width = t.shape[t.shape.length - 1];
    const { F, U, H } = room;
    parts.forEach((part, l) => {
      const to = at + l * each * 4, packing = packingOf(part.kind);
      if (part.kind === "f32") {
        F.copyWithin(to / 4, (base + part.offset) / 4, (base + part.offset) / 4 + each);
      } else if (part.kind === "f16") {
        for (let i = 0; i < each; i++) F[to / 4 + i] = halfToFloat(H[(base + part.offset) / 2 + i]);
      } else if (packing?.row) {
        const row = packing.row(k, tables), bytes = packing.rowBytes(width);
        for (let r = 0; r < each / width; r++) row(to + r * width * 4, base + part.offset + r * bytes, width);
      } else if (packing?.value) {
        const g = part.group, values = base + part.offset, scaleAt = scaleOf(packing, base + part.scales);
        for (let i = 0; i < each; i++) F[to / 4 + i] = Math.fround(packing.value(U, values, i) * scaleAt((i / g) | 0));
      } else {
        throw new Error(`This engine reads no weights stored as ${JSON.stringify(part.kind)}.`);
      }
    });
    return at;
  }
  // T.533: the scale of group g of a packed tensor whose scales start at address scales: a float32, or a float16
  // (ternary's: packing.scale says which), widened exactly. Of the views as they are when it is made (T.641.2: so a
  // token's embedding reads no getter a weight)
  function scaleOf(packing: Packing, scales: number) {
    const { F, H } = room;
    return packing.scale === 2 ? (g: number) => halfToFloat(H[scales / 2 + g]) : (g: number) => F[scales / 4 + g];
  }
  const widened = new Map<string, number>();
  // the address of a float32 tensor, widened once if the file does not hold float32
  function floats(name: TensorName | keyof Derived) {
    if (name in derived) {
      if (!widened.has(name)) {
        const bytes = derived[name as keyof Derived]!, at = alloc(bytes.length);
        room.U.set(bytes, at);
        widened.set(name, at);
      }
      return widened.get(name)!;
    }
    const t = T[name as TensorName];
    if (!t) return 0;
    if (t.kind === "f32") return base + t.offset;
    if (!widened.has(name)) widened.set(name, widen(t));
    return widened.get(name)!;
  }
  // a matrix: packed (values, scales, and the corrections of relaxed SIMD where its packing has them) when plan.int8,
  // else float32. T.375.4: how a packing is multiplied is its line of jobs.ts's PACKED, read here once: the matrix keeps
  // the kind of its job (with relaxed SIMD or without, as this engine runs) and what makes its input ready.
  // T.375.3: made once for a name, as floats() widens once: what the engine names for the GPU's side and what a kind of
  // layer takes for itself are the same thing
  const matrices = new Map<TensorName, Held | null>();
  function matrix(name: TensorName): Held | null {
    if (!matrices.has(name)) matrices.set(name, made(name));
    return matrices.get(name)!;
  }
  // (the activations as a packing's kernels take them: one function for a name, so that matrices packed alike in that
  // are told by it)
  const readies = new Map<string, Ready>();
  const readyOf = (packing: Packing) => {
    if (!readies.has(packing.takes)) readies.set(packing.takes, packing.ready(k, Boolean(relaxed)));
    return readies.get(packing.takes)!;
  };
  function made(name: TensorName): Held | null {
    const source: TensorName = plan.shared_classifier && name === "wcls" ? "token_embedding_table" : name;
    const t = T[source];
    if (!t) return null;
    const [rows, n] = t.shape.slice(-2);
    // T.363.5: a stack with a kind for each layer is its layers, each a matrix of its own (matmuls() takes the layer's)
    if (t.layers) {
      const layers = t.layers.map((part) => held({ ...part, shape: [rows, n] }, source, rows, n));
      return { rows, n, kind: "layers", layers, layer: () => { throw new Error(`The matrix ${source} is its layers, each of a kind of its own.`); } };
    }
    return held(t, source, rows, n);
  }
  function held(t: Tensor, source: TensorName, rows: number, n: number): Held {
    // (scaled: the bytes of a layer's scales, T.533: float16s for ternary; corrected: of its corrections, int32s)
    const packing = packingOf(t.kind), scaled = rows * (n / t.group) * (packing?.scale ?? 4), corrected = rows * (n / t.group) * 4;
    // T.156: on the GPU alone: where each layer's values and scales start in the checkpoint (what gpu.ts was opened
    // with), for the plan the GPU's worker is started with; the CPU never multiplies by it (T.210: nor by the classifier)
    if (direct && GPU_ALONE.includes(source)) {
      // (T.486: gpuOnlyUnfit() keeps a model of float32 or float16 matrices off the GPU alone before it is loaded; one
      // that came here anyway is said so, not read as a packing it has not)
      if (!packing) throw new Error(`The matrix ${source} is of ${t.kind}, which is not on the GPU alone.`);
      const rowBytes = packing.rowBytes(n);
      return { rows, n, kind: t.kind as Packed, group: t.group, onGpu: true, layer: (l) => [t.offset + l * rows * rowBytes, t.scales + l * scaled] };
    }
    if (packing && plan.int8) {
      const rowBytes = packing.rowBytes(n), values = base + t.offset, scales = base + t.scales, groups = count(t) / t.group;
      // (T.363.5: llama.cpp's blocks: the tables and the type where the others have scales and corrections, and last
      // the scratch of the signed tile, taken where there is one)
      if (packing.tabled !== undefined) {
        const type = packing.tabled, { jobs } = packing;
        const job = !relaxed?.matmul_sbr ? jobs.plain : spread && jobs.signed !== undefined ? jobs.signed : jobs.relaxed;
        return { rows, n, kind: t.kind as Packed, group: t.group, layer: (l) => [values + l * rows * rowBytes, tables, type, spread], job, ready: readyOf(packing) };
      }
      let corrections = 0;
      if (relaxed && packing.sums) {
        // relaxed SIMD multiplies by 7-bit unsigned activations with a bias of 64, which this takes out again:
        // dot(w, q - 64) = dot(w, q) - 64 * sum(w). -64 * sum of the group, an int32 (T.197: the kernels add it to the
        // group's integer sum; before it the float32 scale * sum)
        corrections = alloc(groups * 4);
        // the same numbers as a sum in JavaScript, a kernel's speed (T.98, T.123: 7B spent 266 s here one value at a
        // time). Groups of 32: relaxed runs only where every row is whole groups
        k[packing.sums](corrections, values, groups);
      }
      // (a layer's values, scales and corrections: 0 where the packing has none)
      const layer = (l: number) => [values + l * rows * rowBytes, scales + l * scaled, corrections && corrections + l * corrected];
      return { rows, n, kind: t.kind as Packed, group: t.group, layer, job: relaxed ? packing.jobs.relaxed : packing.jobs.plain, ready: readyOf(packing) };
    }
    // (a layer of a stack that is float32 by itself: widened or taken as it is, where it is)
    if (t !== T[source]) return { rows, n, kind: "f32", job: KIND.matmul_f32, ready: null, layer: ((w) => () => [w])(t.kind === "f32" ? base + t.offset : widen(t)) };
    const w = floats(source);  // float32 as it is, or widened once (shared with the embedding when it is the same table)
    return { rows, n, kind: "f32", job: KIND.matmul_f32, ready: null, layer: (l) => [w + l * rows * n * 4] };
  }
  return { floats, matrix, scaleOf };
}
