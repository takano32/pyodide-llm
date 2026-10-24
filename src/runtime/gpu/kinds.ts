// gpu/kinds.ts (T.568.2): the kinds of layers the model's GPU worker has a step for, as one table, and what a kind is to
// the worker. (forward/kinds.ts's LAYER_KINDS is the CPU's table of the same kinds.)
// (a module of the model's GPU worker, gpu.ts, which a bundler links it into)
//
// block.ts, tokens.ts and tokenforms.ts compare no kind's name: for every kind the plan's layers have (Started's kinds)
// they look the kind up here once, have it make what it needs of a prompt's block and of a token, and call its layers'
// steps in turn, between what every layer has (the norm before, the norm behind, the FFN). The CPU stages the state of
// the layers that keep one in the order of its own table, which this table's order is.
//
// A new kind is a file of its own beside attention.ts, linear.ts and convolution.ts, and a line of GPU_KINDS (AGENTS.md,
// "層の種類（状態を持つ層）を足す", has the steps); its shaders are the catalogue's rows (shaders/engine.ts).
import { attention } from "./attention.ts";
import { linear } from "./linear.ts";
import { convolution } from "./convolution.ts";
import type { Bound, Model, Pass } from "./device.ts";
import type { Entry, Feed, Held, TokenForm, TokenRole } from "./tokens.ts";
import type { LayerKind } from "../contracts/plan.ts";
import type { Started, Vectors } from "../contracts/gpu/protocol.ts";

// ---- a prompt's block
// a matrix's product (a bind group and its rows for each piece, forms.ts's productGroups()), an input's quantization
// (QUANTIZE's bind group and workgroups), a bias added (ADD's)
// (T.547.2.3: and its pipeline: the form's, or its Q4_K text's for a matrix of Q4_K blocks)
export type Product = { rows: number; group: GPUBindGroup; pipeline: GPUComputePipeline }[];
// (T.271.1: its own pipeline: QUANTIZE, or ROTATE_QUANTIZE in a rotated basis, which block.ts's quantizing() picks)
export type Quantizing = { pipeline: GPUComputePipeline; group: GPUBindGroup; x: number };
export type Added = { group: GPUBindGroup; x: number };
// What a kind's layer l is handed as block.ts binds it: its place among the layers of its kind, and
//   product()  a matrix of the model (name) of the layer, from one buffer of a block's into another (add: onto it)
//   out()      the layer's matrix whose output joins the stream: onto x, or (a norm behind it: a Gemma 3's) into gate,
//              which block.ts norms into the stream after the step
//   added()    a vector of the layer (name) added to a buffer, where the model has it
export type Binding = {
  l: number; place: number;
  product(name: string, from: GPUBuffer, to: GPUBuffer, add?: boolean): Product;
  out(name: string, from: GPUBuffer): Product;
  added(to: GPUBuffer, name: keyof Vectors): Added | undefined;
};
// What a layer's step in a block is handed each time: the pass, the block's tokens, whether it is the last layer (which
// leaves what nothing reads after it), and what block.ts dispatches for it: a product by the form chosen, a quantizing
// where the form is packed, a bias added, and the FFN's norm of the layer's input (GPT-NeoX's parallel residual, which
// the attention dispatches before o adds to the stream)
export type Running = {
  pass: GPUComputePassEncoder; count: number; last: boolean;
  multiplied(pieces: Product): void; quantized(input: Quantizing): void; biased(bias: Added | undefined): void; ffnNorm(): void;
};
// a layer's step in a block, from the stream normed into xb (and quantized) to what joins the stream: true where the
// block ends with it (the last layer, which leaves the stream and the FFN alone)
export type BlockStep = (running: Running) => boolean;
// a buffer of the state the layers of a kind keep, as the CPU stages it (its bytes)
export type State = { buffer: GPUBuffer; bytes: number };

// ---- a token's step
// What a kind's layer l is handed as tokens.ts makes a step's dispatches: its place, the form of the step, and
//   matrix()    a dispatch of a fused matrix (its role, [values, scales], its input's bindings, its Params, its rows,
//               its output's bindings)
//   plain()     what a matrix of a vector as it is reads (the vector quantized first on DP4A)
//   joining()   a matrix whose output joins the stream (added on its write, or written and normed into it behind)
//   normed      the stream normed for the layer's first matrices (Feed: the dispatches before them and their input)
//   ffnNorm     the FFN's norm of the layer's input (GPT-NeoX's parallel residual)
//   biased()    a bias of the layer added after a matrix (GPT-2's bo), where the model has it
//   positions   how many positions the attention reads at the most
//   bindAt()    a bind group of the bindings given (the fused shaders skip a binding where they read no norm)
export type Stepping = {
  l: number; place: number; form: TokenForm;
  // (T.547.3.3: weights, its values, scales and packing, by which a matrix of llama.cpp's blocks takes its type's pipeline;
  // T.579: role, the key of the form's pipelines it takes, with the type after it for a matrix of blocks)
  matrix(role: TokenRole, weights: Held, input: Entry[], params: GPUBuffer, count: number, output: Entry[]): Pass;
  plain(x: GPUBuffer, params: GPUBuffer, n: number): Omit<Feed, "norm">;
  joining(post: GPUBuffer | undefined, weights: Held, input: Entry[], params: GPUBuffer): Pass[];
  normed: Feed; ffnNorm: Pass[]; biased(to: GPUBuffer, name: "bo", l: number): Pass[]; positions: number;
  bindAt(pipeline: GPUComputePipeline, entries: Entry[]): GPUBindGroup;
};
// What a kind makes a token's uniforms with (tokens.ts's): fusedMatVec's Params (rows, n, second, normAt, turned) and
// the norms' Norm with the model's epsilon (at, size, first), and every layer's offset of its norm (device.ts's normShape
// makes a Norm of another epsilon)
export type Making = {
  params(rows: number, n: number, second?: number, normAt?: number, turned?: number): GPUBuffer;
  norm(at: number, size?: number, first?: number): GPUBuffer;
  layers: number[];
};
// a kind's own of a token, made once (tokens.ts's tokenBuffers()): its layer l's dispatches of a step
export type Steps = { step(stepping: Stepping): Pass[] };

// ---- the check of a token's first layer (tokenforms.ts's checkTokens())
// What the first layer's check hands its kind: the model's first layer in JavaScript, in float64 (the stream x0 at
// position pos and xn normed; a matrix of the first layer times a vector; the first layer's vector of a name, where the
// model has it; RMSNorm with the model's epsilon; behind(), a norm behind, where the model has it; the largest of a
// list and how far one is from another), and the line of the form checked. And what an attending first layer reads:
// the keys and values of position 0 in its cache (random float16s, widened), those the step wrote at pos (as floats),
// and held(), the device's own float16 of a value where it is one of the value's neighbours (tokenforms.ts's heldFloats)
export type Checking = {
  x0: Float64Array; xn: Float64Array; pos: number; line: number;
  row0: [Float64Array, Float64Array]; got: [Float64Array, Float64Array]; held(x: Float64Array, got: ArrayLike<number>): Float64Array;
  matmul(name: string, x: ArrayLike<number>): Float64Array;
  vectorOf(name: keyof Vectors): Float64Array | undefined;
  plus(x: Float64Array, name: keyof Vectors): Float64Array;
  rms(x: Float64Array, weights: Float64Array): Float64Array;
  behind(x: Float64Array, name: keyof Vectors): Float64Array;
  largest(xs: ArrayLike<number> & { reduce(step: (sum: number, value: number) => number, from: number): number }): number;
  off(got: ArrayLike<number>, want: Float64Array): number;
};
// What the kind of the first layer does for its check: before the step, it writes what the layer goes on from (a state
// of random numbers) and says what it copies out after it (copies, into buffers of owned); after it, given the keys
// the check, it says the layer's output o in JavaScript, its keys and values (an attending layer's: the check holds the
// step's to them), and why its state is wrong (null: right)
export type First = {
  copies(encoder: GPUCommandEncoder): void;
  then(checking: Checking): Promise<{ o: Float64Array; k?: Float64Array; v?: Float64Array; wrong: string | null }>;
};

// ---- a kind
// What a kind is to the worker:
//   cached    its layers keep keys and values on the GPU (block.ts's cache), which go back and forth with the CPU's
//   widest()  the widest vector its layers quantize besides the stream, the attention's output and the FFN's
//   matrices() its matrices of a token's step by the pipeline each takes (tokens.ts's tokenShape(), T.579): qkv, q, k
//             and v; mix, a matrix of the normed stream written as it is; write, of a vector of its own; join, the matrix
//             whose output joins the stream (Stepping's joining())
//   block()   what it makes once of a prompt's block (its shaders, its buffers: nothing of m.block yet), which says
//             the state its layers keep (states: as the CPU stages them; none where they keep none), binds its layer l
//             once the forms are chosen (layer(): its step), makes its own of a token (tokens()) and checks its first
//             layer (first(): where the model's first layer is of this kind)
export type KindBlock = {
  states: State[];
  layer(binding: Binding): BlockStep;
  tokens(making: Making): Promise<Steps>;
  first(owned: GPUBuffer[]): First;
};
export type GpuKind = {
  cached?: boolean;
  widest?(plan: Started): number;
  matrices(plan: Started): Partial<Record<"qkv" | "mix" | "write" | "join", string[]>>;
  block(m: Model): Promise<KindBlock>;
};

// (the order is the CPU's, forward/kinds.ts's: the order the state of the layers that keep one is staged in)
export const GPU_KINDS: Record<LayerKind, GpuKind> = { attention, linear, convolution };
// the kinds a model's layers are of, in the table's order
export const kindsOf = (plan: Pick<Started, "kinds">) => (Object.keys(GPU_KINDS) as LayerKind[]).filter((kind) => plan.kinds.includes(kind));
// the layers that keep keys and values here (every one, but an LFM2's convolution layers and a Qwen3.5's linear-attention
// layers), in order: an attending layer's place among them is its index here
export const attendingOf = (plan: Pick<Started, "kinds">) => plan.kinds.flatMap((kind, l) => (GPU_KINDS[kind].cached ? [l] : []));
