// forward/kinds.ts (T375.3): the kinds of layers an engine has a step for, as one table, and what a kind is to the
// engine. A module of src/runtime/forward.ts.
//
// createForward() (engine.ts) compares no kind's name: for every kind the plan's layers have (contracts/plan.ts's
// Layer.kind, which Python's table of the architectures says: engine/layout.py) it looks the kind up here once, as the
// engine is made, has it make the layers of that kind, and keeps every layer's step in an array that a token's path
// calls. Python's side of the same table is engine/model.py's STATEFUL_LAYERS.
//
// A new kind of layer is a file of its own beside attention.ts, linear.ts and convolution.ts, and a line of
// LAYER_KINDS (AGENTS.md, "層の部品を作るときに選ぶ", has the steps).
import { attention } from "./attention.ts";
import { linear } from "./linear.ts";
import { convolution } from "./convolution.ts";
import type { Kernels } from "../jobs.ts";
import type { Cache } from "./cache.ts";
import type { Held } from "./gpuside.ts";
import type { Framed, KindFrame } from "./memory.ts";
import type { Sizes } from "./sizes.ts";
import type { Stateful } from "../contracts/form.ts";
import type { Derived, Layer, LayerKind, Plan, TensorName, Widths } from "../contracts/plan.ts";

// a matmul of one input: [the matrix (one the model has: matmuls() reads it as there), its output, the output's stride, the layer]
export type Product = [Held | null, number, number, number];
// The engine's memory after the checkpoint: alloc() takes bytes of it (and never gives them back), and F, U and H are
// the memory as it is now (getters: the views are made again when the memory grows, so they are read where they are
// used, not kept)
export type Room = { alloc(bytes: number): number; readonly F: Float32Array; readonly U: Uint8Array; readonly H: Uint16Array };
// What the engine hands a kind as it makes its layers (and the FFN, ffn.ts):
//   plan      the model's numbers and facts (Python's), eps the epsilon of its RMSNorms
//   sizes     what sizes the model (sizes.ts: the plan's, as frame() and state() were asked with)
//   k         the kernels on this memory (the calling thread's)
//   room      the memory
//   floats(), matrix()   a tensor of the model by its name: the address of its float32 values (0: the model has
//             none), a matrix as the kernels multiply by it (null: none). The same thing for the same name, whoever asks
//   at, S     where each array of token 0's frame is (the names the kinds' frame() said, and the engine's own: x, xb,
//             xb2, q, kNow, vNow, before, hb, hb2), and the bytes from a token's frame to the next token's. A kind's
//             arrays that are apart from the frame are in at too, a token's after another's by their own bytes
//   matmuls() matrices times count tokens of one input, shared out among the threads; phase() any jobs (jobs.ts)
//   cache     the keys and values of the layers that keep them (cache.ts)
export type Core = {
  plan: Plan; sizes: Sizes; eps: number; k: Kernels; room: Room;
  floats(name: TensorName | keyof Derived): number; matrix(name: TensorName): Held | null;
  at: Record<string, number>; S: number;
  matmuls(input: number, count: number, list: Product[]): void; phase(jobs: number[][]): void;
  cache: Cache;
};
// A layer's step: count tokens of a run that began at position pos0 (one, or a prompt's block), from the norm of the
// residual stream in at.xb into at.xb2, which the engine adds to the stream
export type Step = (count: number, pos0: number) => void;
// The layers of one kind in a model, which own their tensors and their state: layer() is the step of the place-th of
// them (fact: what the plan says of it). clear, where the kind keeps a state from token to token that cannot be
// written again: position 0 clears it, and the engine then takes the positions in turn only (its follow())
export type Layers = { layer(place: number, fact: Layer): Step; clear?(): void };
// A kind. What it says of a model's sizes is asked twice, as the engine is made (of the plan) and before a model is
// loaded (footprint(), of the header and the form: memory.ts), each time with a Sizes (sizes.ts):
//   frame()   what a token's frame holds for it (memory.ts's frameArrays(); apart: arrays of a token too large for
//             its frame, BATCH of each by themselves)
//   state()   the bytes of what its count layers keep besides, a number for each thing it takes of the memory, in
//             the order make() takes them (make() takes these, and footprint() counts them)
//   cached    its layers keep keys and values in the cache (cache.ts), which grows to the whole context
// What only a form can be asked, where there is no plan (Python's plan says both: its layers, its widths):
//   layers()  which of a model's count layers are of this kind (null: the form has none; a layer that no kind says
//             is its own attends)
//   widths()  its own widths, as the plan's widths has them
// And make(), its layers in a model that has count of them.
// (A kind whose layers keep a state from token to token says so by clear() of what make() returns: a model with such
// layers is not on the GPU, whatever the kind. One that keeps none there and is not attention's would be taken there.)
export type Kind = {
  frame(sizes: Sizes): KindFrame & { apart?: Framed[] }; state?(sizes: Sizes, count: number): number[]; cached?: boolean;
  layers?(form: Stateful, count: number): boolean[] | null; widths?(form: Stateful): Partial<Widths>;
  make(core: Core, count: number): Layers;
};
export type Kinds = Record<string, Kind>;

// (the order is the order the kinds of a model are made in, and so of what they take of the memory)
export const LAYER_KINDS: Record<LayerKind, Kind> = { attention, linear, convolution };
