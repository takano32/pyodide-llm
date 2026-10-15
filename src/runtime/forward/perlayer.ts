// forward/perlayer.ts (T329.2): a Gemma 4's embedding of every layer, which the engine takes as it is made (null where
// the model has no table of it: a token's path then compares null twice). Made of what the engine hands a kind of layer
// (kinds.ts's Core), and of rows: what reads a token's row of the table (createForward()'s, the comments say). Python's
// side is engine/model.py's Llama.forward() (where it says T329), and the converter folded the model's scales into the
// table, the projection and its norm (TODO.md's T329). A module of src/runtime/forward.ts.
import { packingOf } from "../jobs.ts";
import { align, halfToFloat, perLayerBytes } from "./memory.ts";
import type { Core } from "./kinds.ts";
import type { Tensors } from "../contracts/plan.ts";

/** What reads bytes of the file: the bytes at offset (counted from the start of the checkpoint) into the memory at to,
 * at once (synchronously: a token's path waits on it) */
export type Rows = (offset: number, bytes: number, to: number) => void;

export type PerLayer = {
  /** count tokens' inputs of every layer, a frame apart in at.x (embedded, the table's row of each in ids) */
  inputs(ids: number[], count: number): void;
  /** what layer l makes of its input joins the stream of count tokens (after the FFN, before the layer's scale) */
  layer(l: number, count: number): void;
};

/** The step of every layer's input, or null where the model has no table of it. rows: where the table's rows come from */
export function perLayer({ plan, eps, k, room, floats, matrix, at, S, matmuls }: Core, rows: Rows): PerLayer | null {
  // (t: the table, as a plan says where a tensor is)
  const T = plan.tensors, t = T.per_layer_table;
  if (!t) return null;
  const { dim, n_layers: layers } = plan, norm = floats("per_layer_norm");
  // the width of a layer's input (P): the norm of the projection is that wide, one for every layer's
  const width = T.per_layer_norm!.shape[0], values = layers * width, W = width * 4, D = dim * 4;
  const projection = matrix("per_layer_projection"), gates = matrix("ple_gate"), projections = matrix("ple_proj"), post = floats("ple_post");
  if (plan.activation !== "gelu" && plan.activation !== "silu") throw new Error(`This engine has no gate whose activation is ${JSON.stringify(plan.activation)}.`);
  const gate = k[plan.activation === "gelu" ? "geglu" : "swiglu"];
  const { x, hb, xb2 } = at;
  // (what the inputs of BATCH tokens take, and what a token's row is read into: values, then scales where it has them)
  const group = t.group, [inputsBytes, readBytes, valuesBytes] = perLayerBytes(layers, width, t.kind, group);
  const inputs = room.alloc(inputsBytes), I = values * 4;
  const read = room.alloc(readBytes), scales = read + align(valuesBytes), scalesBytes = readBytes - align(valuesBytes);
  const packing = packingOf(t.kind), value = packing?.value;
  if (packing && !value) throw new Error(`This engine reads no row of a table stored as ${JSON.stringify(t.kind)}.`);
  // a token's row of the table, added to what is at to (float32, as NumPy: each weight times its scale, then the sum)
  const addRow = (id: number, to: number) => {
    rows(t.offset + id * valuesBytes, valuesBytes, read);
    if (value) {
      rows(t.scales + id * scalesBytes, scalesBytes, scales);
      const { U, F } = room;
      for (let i = 0; i < values; i++) F[to / 4 + i] += Math.fround(value(U, read, i) * F[scales / 4 + ((i / group) | 0)]);
    } else if (t.kind === "f16") {
      const { H, F } = room;
      for (let i = 0; i < values; i++) F[to / 4 + i] += halfToFloat(H[read / 2 + i]);
    } else {
      const { F } = room;
      for (let i = 0; i < values; i++) F[to / 4 + i] += F[read / 4 + i];
    }
  };
  return {
    inputs(ids, count) {
      // the projection of the embedded token, normalized every width values (one norm, every layer's)
      matmuls(x, count, [[projection, inputs, I, 0]]);
      for (let t = 0; t < count; t++) {
        const own = inputs + t * I;
        for (let l = 0; l < layers; l++) k.rmsnorm(own + l * W, own + l * W, norm, width, eps);
        addRow(ids[t], own);
      }
    },
    layer(l, count) {
      // gelu(gate · x) times the layer's input, projected back to dim, normalized, and added to the stream
      matmuls(x, count, [[gates, hb, S, l]]);
      for (let t = 0; t < count; t++) gate(hb + t * S, hb + t * S, inputs + t * I + l * W, width);
      matmuls(hb, count, [[projections, xb2, S, l]]);
      for (let t = 0; t < count; t++) {
        k.rmsnorm(xb2 + t * S, xb2 + t * S, post + l * D, dim, eps);
        k.add_inplace(x + t * S, xb2 + t * S, dim);
      }
    },
  };
}

/** T329.2.2: the stretch of the checkpoint the table of every layer's inputs takes ([start, end): its values, then its
 * scales where it has them, as llama2_numpy's Tensor places them), of the tensors llama2_numpy.external_tensors()
 * places; null where the model has no such table */
export function tableHole(tensors: Tensors): [number, number] | null {
  const t = tensors.per_layer_table;
  if (!t) return null;
  const [vocab, values] = t.shape, [, readBytes, valuesBytes] = perLayerBytes(1, values, t.kind, t.group);
  return [t.offset, packingOf(t.kind) ? t.scales + vocab * (readBytes - align(valuesBytes)) : t.offset + vocab * valuesBytes];
}
