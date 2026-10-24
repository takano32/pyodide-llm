// The weights onto the GPU: every layer's matrices and vectors and a token's tables, from the shared memory or (a
// model on the GPU alone, T.156) as routes of the checkpoint that start.ts's take() writes; a matrix past a buffer in
// pieces (T.155), the layout of a token's joined matrices (T.152), and the widening of int6 (T.155) with its check.
// (T.352: a module of the model's GPU worker, gpu.ts, which a bundler links it into: T.367.3)

import { STORAGE, COPY_DST, COPY_SRC, CHUNK, rowBytes, stopping, within, buffer, uniform, copyIn, widenIn, widenedHalves, readBack,
  validated, pipelineOf, bind, dispatch } from "./device.ts";
import { GROUP } from "../shaders/common.ts";
import { sixDispatch, sixValues } from "../shaders/steps.ts";
import { ENGINE } from "../shaders/engine.ts";
import { PACKINGS } from "./packings.ts";
import type { Model, Bound } from "./device.ts";
import type { GpuPacking, Table, Tables } from "../contracts/gpu/protocol.ts";

// T.368.6: this module's record, m.weights, as uploadLayers() makes it and the rest fills it (what is null there until
// its part has gone up is read with a ! by the modules that come after).
// a piece of a matrix: its first row, its rows, and for every layer its values and scales (buffers of its own, or its
// ranges of the layer's joined ones)
type Piece = { first: number; rows: number; layers: [Bound, Bound][] };
type MatrixOn = { rows: number; n: number; packing: GpuPacking; pieces: Piece[] };
// T.152: where a matrix is in a layer's joined buffers, and those buffers' sizes ([values, scales] each)
type Home = { joined: string; at: [number, number] };
type Together = { homes: { [name: string]: Home | undefined }; sizes: { [joined: string]: [number, number] } };
type Joined = { [joined: string]: [GPUBuffer, GPUBuffer] };
type Widen = { into(values: Bound, address: number, groups: number): void; done(): void };
// T.156: a stretch of the checkpoint, [start, end), the buffer it goes to and the offset in it
// (halves, T.533: a stretch of ternary's float16 scales, which go up widened to float32: the buffer's offset of the
// stretch's first, and twice its bytes there)
type Route = [number, number, GPUBuffer, number, boolean?];
// T.533: the bytes of a group's scale in the checkpoint: ternary's a float16, int8's (and int6's) a float32. The GPU
// holds every one as a float32 (scaleBytes below count those), ternary's widened as they go up (scalesIn, routedHalves)
// (T.547.2.1: the packing's, packings.ts)
const heldScale = ({ packing }: { packing: GpuPacking }) => PACKINGS[packing].scale;
function scalesIn(m: Model, spec: { packing: GpuPacking }, target: GPUBuffer, address: number, scaleBytes: number, offset = 0) {
  if (heldScale(spec) === 2) widenIn(m, target, address, scaleBytes / 4, offset);
  else copyIn(m, target, address, scaleBytes, offset);
}
// T.209: a piece of a table
type TablePiece = { first: number; rows: number; values: GPUBuffer; scales: GPUBuffer };
type Weights = {
  matrices: { [name: string]: MatrixOn };
  together: Together | null;
  unjoined: string | null;
  joined: Joined[];
  widen: Widen | null;
  routes: Route[] | null;
  partial: Map<GPUBuffer, Map<number, { bytes: Uint8Array<ArrayBuffer>; count: number }>> | null;
  vectors: { [name: string]: GPUBuffer } | null;
  tables: { classifier: TablePiece[]; embedding: TablePiece[] } | null;
  finalNorm: GPUBuffer | null;
  finalBias: GPUBuffer | null;
  positions: GPUBuffer | null;
  angleTable: GPUBuffer | null;
  // T.261.4.2: a Gemma 3's local layers' table of RoPE, laid out as angleTable (null: none)
  localAngleTable: GPUBuffer | null;
  // T.271.1: a rotated basis's signs, by the width a matrix reads (null: the model's own basis)
  signs: Record<number, GPUBuffer> | null;
};
// what tablesOf() reads of a plan: the tables of a model on the GPU alone as it opens, else a token's
type Tabled = { tables?: Tables; tokens: Tables | null };

// every layer's matrices (values and scales, as the checkpoint holds them; T.154: no w3 where there is no gate) and its
// vectors: the weights of its two norms, and (T.153) Qwen2's biases of q, k and v, Qwen3's norms of a head of q and of
// k, (T.154) GPT-2's and GPT-NeoX's biases of the two LayerNorms and of every matrix (plan.vectors: each one's address in
// the shared memory and its floats a layer). T.155: a matrix in pieces of whole rows (piecesOf), a buffer of values and
// one of scales each; int6 (plan.matrices' six) widened to int8 on the way (widener), its scales as they are (the
// quarter of an int6 group's is already the int8 values' scale, T.98). The addresses are Numbers in a 64-bit memory too
// (exact to 2^53; the views and copies take them as they are). T.232: ternary (plan.matrices' ternary) as it is, a row
// of n weights n / 4 bytes (rowBytes)
// T.152: where a token goes on the GPU too (plan.tokens), q, k and v are one matrix a layer, and gate and up one (the
// layer of a token reads them so: shaders.ts's fusedMatVec and fusedDp4aMatVec, T.150 and T.175), each a range of it for
// the prompt's tiled shaders (tokensLayout), and the classifier, the embedding, the final norm and RoPE's table go up
// as well (uploadTokens)
// T.156, a model on the GPU alone (alone: its record has routes): the layers' buffers are made as the worker opens,
// before a byte comes, and each stretch of the checkpoint they hold is a route ([start, end) in the checkpoint, the
// buffer and its offset) that routed() writes the bytes to as they come; the rest (the tables, the norms) goes up from
// the shared memory as start() is asked (uploadRest)
// T.378.4: what goes up is this module's record, m.weights, which it alone sets (uploadLayers() makes it, the rest
// fills it) and the others read: the matrices' pieces and the layers' joined buffers, the vectors, a token's tables,
// its final norm and positions and RoPE's table, and why a token's layer could not be joined (unjoined, T.152)
async function upload(m: Model) {
  try {
    const bytes = await uploadLayers(m);
    if (stopping) return bytes;
    return bytes + (await uploadRest(m));
  } finally {
    m.weights?.widen?.done();
  }
}
async function uploadLayers(m: Model, alone = false) {
  const { plan } = m, group = GROUP;
  let bytes = 0;
  const matrices = Object.fromEntries(Object.entries(plan.matrices).map(([name, { rows, n, packing }]): [string, MatrixOn] =>
    [name, { rows, n, packing, pieces: piecesOf(m, rows, rowBytes({ n, packing })).map(([first, count]) => ({ first, rows: count, layers: [] })) }]));
  // (routes, partial: T.156, a model on the GPU alone; the routes of the checkpoint's stretches, and the bytes of a word
  // that wait for the rest of it, routed())
  m.weights = { matrices, together: null, unjoined: null, joined: [], widen: null, routes: alone ? [] : null, partial: alone ? new Map() : null,
    vectors: null, tables: null, finalNorm: null, finalBias: null, positions: null, angleTable: null,
    localAngleTable: null, signs: null };
  const together = plan.tokens ? tokensLayout(m) : null;
  m.weights.together = together;
  // (a model on the GPU alone opens before its tables are placed, and is int8: T.156)
  const tables = together && !alone ? Object.values(tablesOf(plan)) : [];
  const six = [...Object.values(plan.matrices), ...tables].some((matrix) => PACKINGS[matrix.packing].widened);
  // (kept for the tables, uploadRest; upload() lets it go)
  const widen = m.weights.widen = six ? await widener(m, tables) : null;
  // (T.156: a model on the GPU alone reads its first layer back to check a token's layer against: firstLayer)
  const usage = STORAGE | COPY_DST | (alone ? COPY_SRC : 0);
  const { routes } = m.weights;
  for (let l = 0; l < plan.layers; l++) {
    // a layer's matrices that are one (T.152): a buffer of values and one of scales for all of them (T.271.3: where the
    // layer has them: q, k and v the attending layers' alone, where they are theirs: an LFM2's and a Qwen3.5's)
    const has = (joinedName: string) => Object.entries(together!.homes).some(([name, home]) => home?.joined === joinedName && (!plan.matrices[name].of || plan.matrices[name].of === plan.kinds[l]));
    const joined = together && Object.fromEntries(Object.entries(together.sizes).filter(([name]) => has(name)).map(([name, [valueBytes, scaleBytes]]): [string, [GPUBuffer, GPUBuffer]] =>
      [name, [buffer(m, valueBytes, usage), buffer(m, scaleBytes, usage)]]));
    if (joined) m.weights.joined.push(joined);
    for (const [name, matrix] of Object.entries(plan.matrices)) {
      // (T.545: a matrix of the layers of a kind, an LFM2's attending or convolution layers', at the layer's place among
      // them, and none in a layer of another kind: its pieces' layers are by those places)
      if (matrix.of && plan.kinds[l] !== matrix.of) continue;
      const [valuesAt, scalesAt] = matrix.layers[matrix.of ? plan.places[l] : l];
      const home = together?.homes[name], row = rowBytes(matrix);
      for (const piece of m.weights.matrices[name].pieces) {
        const valueBytes = piece.rows * row, scaleBytes = (valueBytes / group) * 4;
        // its own buffers, or its range of the layer's joined ones
        const [values, scales]: Bound[] = home
          ? joined![home.joined].map((b, i) => ({ buffer: b, offset: home.at[i], size: i ? scaleBytes : valueBytes }))
          : [buffer(m, valueBytes, usage), buffer(m, scaleBytes, usage)];
        const from = [valuesAt + piece.first * row, scalesAt + (piece.first * row / group) * heldScale(matrix)];
        if (alone) {
          // (T.156: int8 alone, and T.232 ternary, whose bytes are the buffer's as they come; T.533: its scales widened)
          routes!.push([from[0], from[0] + valueBytes, values.buffer ?? values, values.offset ?? 0],
            [from[1], from[1] + (scaleBytes / 4) * heldScale(matrix), scales.buffer ?? scales, scales.offset ?? 0, heldScale(matrix) === 2]);
        } else {
          // an int6 row is 3/4 of an int8 one (24 bytes a group of 32)
          if (PACKINGS[matrix.packing].widened) widen!.into(values, valuesAt + PACKINGS[matrix.packing].stored(piece.first * matrix.n), valueBytes / group);
          else copyIn(m, values.buffer ?? values, from[0], valueBytes, values.offset ?? 0);
          scalesIn(m, matrix, scales.buffer ?? scales, from[1], scaleBytes, scales.offset ?? 0);
        }
        piece.layers.push([values, scales]);
        bytes += valueBytes + scaleBytes;
      }
    }
    if (alone) continue;
    // what was written waits in memory until the GPU takes it: let it, before more comes
    await within(m.device.queue.onSubmittedWorkDone(), `layer ${l + 1}'s weights`);
    if (stopping) return bytes;
  }
  return bytes;
}
// the tables of a token (where the layers are joined for them, T.152), and the norms, from the shared memory
async function uploadRest(m: Model) {
  const { plan } = m;
  let bytes = 0;
  if (m.weights.together) bytes += await uploadTokens(m, m.weights.widen);
  m.weights.vectors = {};
  for (const [name, { at, size, of }] of Object.entries(plan.vectors)) {
    // (T.545: of the layers of a kind, where it is theirs alone)
    const vectorBytes = (of ? plan.kinds.filter((kind) => kind === of).length : plan.layers) * size * 4;

    m.weights.vectors[name] = buffer(m, vectorBytes, STORAGE | COPY_DST);
    copyIn(m, m.weights.vectors[name], at, vectorBytes);
    bytes += vectorBytes;
  }
  // T.271.1: a rotated basis's signs of each width, from the shared memory (Python's derived tables)
  if (plan.rotated) {
    m.weights.signs = {};
    for (const { width, at } of plan.rotated.signs) {
      m.weights.signs[width] = buffer(m, width * 4, STORAGE | COPY_DST);
      copyIn(m, m.weights.signs[width], at, width * 4);
      bytes += width * 4;
    }
  }
  return bytes;
}

// T.155: the pieces of a matrix of rows of n bytes (rowBytes), [first row, rows] each: whole rows, no more bytes than a buffer the device
// binds (m.limit, or plan.force.pieceBytes in the tests), each piece's first row where its part of the output may be
// bound (a multiple of minStorageBufferOffsetAlignment over the 4 bytes of a float32). One piece where the matrix
// fits, as every matrix of the models of the list does at WebGPU's least limit of 128 MiB (the largest, Qwen2.5 7B's
// w1, is 68 MB): llama.cpp's WebGPU keeps each tensor in one buffer within maxStorageBufferBindingSize and refuses a
// larger one (ggml-webgpu.cpp, ggml_backend_webgpu_buffer_type_get_max_size, commit 2145525a, MIT; no line copied),
// where this cuts it by rows, as T.94's design had it for a classifier past a buffer
function piecesOf(m: Model, rows: number, n: number, most = m.plan.force.pieceBytes): [number, number][] {
  const bytes = Math.min(m.limit, most ?? Infinity), align = m.device.limits.minStorageBufferOffsetAlignment / 4;
  if (rows * n <= bytes) return [[0, rows]];
  const step = Math.floor(bytes / n / align) * align;
  if (!step) throw new Error(`${align} rows of ${n} weights are more than a buffer of this GPU (${bytes} bytes)`);
  return Array.from({ length: Math.ceil(rows / step) }, (_, i): [number, number] => [i * step, Math.min(step, rows - i * step)]);
}

// T.152: the tables a token needs besides the layers (plan.tokens; T.210, a model on the GPU alone as it opens:
// plan.tables): the classifier, and the embedding where it is another table (else the classifier is the embedding
// too), each made by make(spec) where it is given, else the specs themselves
function tablesOf(plan: Tabled): { classifier: Table; embedding: Table };
function tablesOf<Made>(plan: Tabled, make: (spec: Table) => Made): { classifier: Made; embedding: Made };
function tablesOf(plan: Tabled, make: (spec: Table) => unknown = (spec) => spec) {
  const { classifier, embedding } = plan.tables ?? plan.tokens!, made = make(classifier);
  return { classifier: made, embedding: embedding ? make(embedding) : made };
}
// T.209: a table ({ rows, n }) in pieces of rows where it is past what the device binds (piecesOf), a buffer of values
// and one of scales each, [{ first, rows, values, scales }]; fill(piece, valueBytes, scaleBytes) puts its bytes there
function tablePieces(m: Model, spec: Table, fill: (piece: TablePiece, valueBytes: number, scaleBytes: number) => void, usage = STORAGE | COPY_DST): TablePiece[] {
  return piecesOf(m, spec.rows, rowBytes(spec), m.plan.force.tablePieceBytes ?? m.plan.force.pieceBytes).map(([first, count]) => {
    const valueBytes = count * rowBytes(spec), scaleBytes = (valueBytes / GROUP) * 4;
    const piece = { first, rows: count, values: buffer(m, valueBytes, usage), scales: buffer(m, scaleBytes, usage) };
    fill(piece, valueBytes, scaleBytes);
    return piece;
  });
}
// T.210: the tables of a model on the GPU alone, made as it opens (plan.tables: where each starts in the checkpoint),
// their stretches routes that take() writes the bytes to as they come; COPY_SRC for the check's rows (checkTokens).
// The bytes on the GPU
function tablesOn(m: Model) {
  const group = GROUP, { routes } = m.weights;
  let bytes = 0;
  m.weights.tables = tablesOf(m.plan, (spec) => tablePieces(m, spec, ({ first, values, scales }, valueBytes, scaleBytes) => {
    const [valuesAt, scalesAt] = spec.at, from = [valuesAt + first * rowBytes(spec), scalesAt + (first * rowBytes(spec) / group) * heldScale(spec)];
    routes!.push([from[0], from[0] + valueBytes, values, 0], [from[1], from[1] + (scaleBytes / 4) * heldScale(spec), scales, 0, heldScale(spec) === 2]);
    bytes += valueBytes + scaleBytes;
  }, STORAGE | COPY_DST | COPY_SRC));
  // (the last of the routes are made: in the checkpoint's order, for routed() to find a stretch's)
  routes!.sort((a, b) => a[0] - b[0]);
  return bytes;
}
// T.156: the bytes at offset of the checkpoint (bytes: a Uint8Array of their own) onto the buffers whose routes they
// fall in. writeBuffer takes whole words: the bytes of a word a stretch begins or ends in the middle of wait for the
// rest of it (partial), which another stretch brings
function routed(m: Model, offset: number, bytes: Uint8Array) {
  const { routes } = m.weights, end = offset + bytes.length;
  let lo = 0, hi = routes!.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (routes![mid][1] <= offset) lo = mid + 1;
    else hi = mid;
  }
  for (let i = lo; i < routes!.length && routes![i][0] < end; i++) {
    const [start, stop, target, at, halves] = routes![i];
    if (halves) {
      routedHalves(m, routes![i], Math.max(start, offset), Math.min(stop, end), offset, bytes);
      continue;
    }
    let a = Math.max(start, offset) - offset, here = at + (a + offset - start);
    const b = Math.min(stop, end) - offset;
    for (; a < b && here % 4; a++, here++) partialByte(m, target, here, bytes[a]);
    const whole = (b - a) & ~3;
    if (whole) m.device.queue.writeBuffer(target, here, bytes, a, whole);
    for (let k = a + whole; k < b; k++) partialByte(m, target, here + (k - a), bytes[k]);
  }
}
// T.533: the bytes [from, to) of the checkpoint (bytes holds those from offset on) of a route of float16 scales: each
// half widened to the float32 at the route's offset + twice its place; a half a stretch begins or ends in the middle
// of waits for its other byte (partial, as the words of partialByte do)
function routedHalves(m: Model, [start, , target, at]: Route, from: number, to: number, offset: number, bytes: Uint8Array) {
  let p = from;
  const place = (q: number) => at + ((q - start) >> 1) * 4;
  if ((p - start) & 1 && p < to) partialHalf(m, target, place(p), 1, bytes[p++ - offset]);
  const whole = (to - p) >> 1;
  if (whole) {
    const halves = new Uint16Array(whole);
    for (let i = 0; i < whole; i++) halves[i] = bytes[p - offset + 2 * i] | (bytes[p - offset + 2 * i + 1] << 8);
    m.device.queue.writeBuffer(target, place(p), widenedHalves(halves, new Float32Array(whole)));
    p += 2 * whole;
  }
  if (p < to) partialHalf(m, target, place(p), 0, bytes[p - offset]);
}
// byte k (0, the low one, or 1) of the half whose float32 is at the byte offset at of target, widened and written
// once both of its bytes are there
function partialHalf(m: Model, target: GPUBuffer, at: number, k: number, value: number) {
  const { partial } = m.weights;
  if (!partial!.has(target)) partial!.set(target, new Map());
  const halves = partial!.get(target)!;
  if (!halves.has(at)) halves.set(at, { bytes: new Uint8Array(2), count: 0 });
  const held = halves.get(at)!;
  held.bytes[k] = value;
  if (++held.count < 2) return;
  m.device.queue.writeBuffer(target, at, widenedHalves(Uint16Array.of(held.bytes[0] | (held.bytes[1] << 8)), new Float32Array(1)));
  halves.delete(at);
  if (!halves.size) partial!.delete(target);
}
// a byte of target at the byte offset at, the word it is in written once all four of its bytes are there
function partialByte(m: Model, target: GPUBuffer, at: number, value: number) {
  const { partial } = m.weights, word = at - (at % 4);
  if (!partial!.has(target)) partial!.set(target, new Map());
  const words = partial!.get(target)!;
  if (!words.has(word)) words.set(word, { bytes: new Uint8Array(4), count: 0 });
  const held = words.get(word)!;
  held.bytes[at % 4] = value;
  if (++held.count < 4) return;
  m.device.queue.writeBuffer(target, word, held.bytes);
  words.delete(word);
  if (!words.size) partial!.delete(target);
}
// whether bytes of a word still wait for the rest of it: the checkpoint's stretches did not all come
const unwritten = (m: Model) => m.weights.partial!.size > 0;
// T.152: how a token's layer holds its matrices (shaders.ts's fusedMatVec and fusedDp4aMatVec read q, k and v as one
// matrix, gate and up as one, up's rows after gate's): a buffer of values and one of scales a layer for each (sizes),
// and each matrix's range of them (homes: its offsets in bytes), which the prompt's tiled shaders bind as a matrix of
// its own. A range must start where the device binds a buffer (minStorageBufferOffsetAlignment: a matrix of a
// multiple of 2048 weights, the values and the scales alike at 256; stories15M's k starts at 82944 weights, T.150's
// (b); T.232: of 8192 ternary weights, whose scales are a byte to 32 weights), and a matrix a token reads must be one piece (T.155). The tables may be in pieces (T.209: uploadTokens). null
// where it cannot, and why in m.weights.unjoined: the prompts go on the GPU as before, the tokens stay on the CPU
function tokensLayout(m: Model): Together | null {
  const align = m.device.limits.minStorageBufferOffsetAlignment, group = GROUP;
  const why = (reason: string) => {
    m.weights.unjoined = reason;
    return null;
  };
  // (T.226: GPT-2's and GPT-NeoX's FFN has no gate: w1 is a matrix of its own, as wo and w2 are)
  const gated = Boolean(m.weights.matrices.w3);
  // (T.545.2: and an LFM2's matrices in and out of its convolution layers; T.271.3: a Qwen3.5's of its linear-attention
  // layers and the gate of its output)
  for (const name of ["wo", "w2", ...(gated ? [] : ["w1"]), "win", "wout", "wqkv", "wz", "wg"]) if ((m.weights.matrices[name]?.pieces.length ?? 1) > 1) return why(`${name} is past a buffer of this GPU`);
  const homes: Together["homes"] = {}, sizes: Together["sizes"] = {};
  for (const [joined, names] of Object.entries({ qkv: ["wq", "wk", "wv"], ...(gated ? { gateUp: ["w1", "w3"] } : {}) })) {
    let values = 0, scales = 0;
    for (const name of names) {
      const matrix = m.weights.matrices[name];
      if (matrix.pieces.length > 1) return why(`${name} is past a buffer of this GPU`);
      if (values % align || scales % align) {
        return why(`${name} would not start where this GPU binds a buffer`);
      }
      homes[name] = { joined, at: [values, scales] };
      values += matrix.rows * rowBytes(matrix);
      scales += (matrix.rows * rowBytes(matrix) / group) * 4;
    }
    if (values > m.limit) return why(`${names.join(" and ")} together are past a buffer of this GPU`);
    sizes[joined] = [values, scales];
  }
  return { homes, sizes };
}
// T.152: the tables onto the GPU (int6 widened as a layer's matrices are), the final norm's weights, and RoPE's table
// of every position, a row a position: the cos of its headSize / 2 angles, then their sin (shaders.ts's fusedMatVec
// reads it so, T.151; from the CPU's two tables, Llama 3's scaling in them)
// T.209: a table in pieces of rows where it is past what the device binds (piecesOf: Llama 3.2 3B's classifier is
// 394 MB, the owner's Android binds 256 MiB; T.152's review (e)), [{ first, rows, values, scales }] each. The
// classifier is then a dispatch a piece into its range of the logits, and EMBED one a piece (tokenPass)
// T.210: a model on the GPU alone makes them as it opens, their bytes routes of the checkpoint as the layers' are
// (tablesOn), and only the final norm and RoPE's table come from the shared memory here
async function uploadTokens(m: Model, widen: Widen | null) {
  const { plan } = m, group = GROUP, half = plan.headSize / 2;
  let bytes = 0;
  m.weights.tables ??= tablesOf(plan, (spec) => tablePieces(m, spec, (piece, valueBytes, scaleBytes) => {
    const { first, values, scales } = piece, { n, packing, at: [valuesAt, scalesAt] } = spec;
    if (PACKINGS[packing].widened) widen!.into(values, valuesAt + PACKINGS[packing].stored(first * n), valueBytes / group);
    else copyIn(m, values, valuesAt + first * rowBytes(spec), valueBytes);
    scalesIn(m, spec, scales, scalesAt + (first * rowBytes(spec) / group) * heldScale(spec), scaleBytes);
    bytes += valueBytes + scaleBytes;
  }));
  // (T.226: the final LayerNorm's bias, and GPT-2's learned positions, a row a position as the CPU widened them: a
  // step adds its position's row to the embedding's, as the CPU's embed() does)
  const vector = (address: number, floats: number) => {
    const made = buffer(m, floats * 4, STORAGE | COPY_DST);
    copyIn(m, made, address, floats * 4);
    bytes += floats * 4;
    return made;
  };
  m.weights.finalNorm = vector(plan.tokens!.final, plan.dim);
  m.weights.finalBias = plan.tokens!.finalBias ? vector(plan.tokens!.finalBias, plan.dim) : null;
  m.weights.positions = plan.tokens!.positions ? vector(plan.tokens!.positions, plan.seqLen * plan.dim) : null;
  const rows = Math.max(1, Math.floor(CHUNK / (plan.headSize * 4))), chunk = new Float32Array(rows * plan.headSize);
  // (T.261.4.2: and a Gemma 3's local layers' table, from its own two)
  const table = (cos: number, sin: number) => {
    const made = buffer(m, plan.seqLen * plan.headSize * 4, STORAGE | COPY_DST);
    // (GPT-2 turns nothing and has no tables: the buffer stays, bound and never read)
    for (let p = 0; plan.turned && p < plan.seqLen; p += rows) {
      const count = Math.min(rows, plan.seqLen - p);
      for (let r = 0; r < count; r++) {
        chunk.set(new Float32Array(m.memory.buffer, cos + (p + r) * half * 4, half), r * plan.headSize);
        chunk.set(new Float32Array(m.memory.buffer, sin + (p + r) * half * 4, half), r * plan.headSize + half);
      }
      m.device.queue.writeBuffer(made, p * plan.headSize * 4, chunk, 0, count * plan.headSize);
    }
    bytes += plan.seqLen * plan.headSize * 4;
    return made;
  };
  m.weights.angleTable = table(plan.cos, plan.sin);
  m.weights.localAngleTable = plan.localCos ? table(plan.localCos, plan.localSin) : null;
  await within(m.device.queue.onSubmittedWorkDone(), "the classifier's weights");
  return bytes;
}

// T.155: shaders.ts's WIDEN_SIX compiled and checked against JavaScript (sixValues), and into(values, address, groups):
// groups of int6 at address in the shared memory widened into values (a buffer of int8, or T.152 a range of one) on
// the GPU. The packed bytes go
// through one buffer of their own, written again for every piece (the queue runs a write after the submissions before
// it); done() lets it go.
async function widener(m: Model, tables: Table[] = []): Promise<Widen> {
  const { device } = m;
  const pipeline = await within(validated(m, () => pipelineOf(m, ENGINE.WIDEN_SIX)), "compiling the widening of int6");
  const owned: GPUBuffer[] = [];
  // (T.152: and the classifier's and the embedding's, one piece each)
  const largest = Math.max(CHECK_GROUPS * 24, ...Object.entries(m.plan.matrices).filter(([, matrix]) => PACKINGS[matrix.packing].widened)
    .flatMap(([name, { n }]) => m.weights.matrices[name].pieces.map((piece) => (piece.rows * n * 3) / 4)),
  ...tables.filter((table) => PACKINGS[table.packing].widened).map(({ rows, n }) => (rows * n * 3) / 4));
  const packed = buffer(m, largest, STORAGE | COPY_DST, owned);
  const run = (values: Bound, groups: number, most = device.limits.maxComputeWorkgroupsPerDimension) => {
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, pipeline, bind(m, pipeline, [packed, values, uniform(m, new Uint32Array([groups, 0, 0, 0]), owned)]),
      ...sixDispatch(groups, most));
    pass.end();
    device.queue.submit([encoder.finish()]);
  };
  const done = () => owned.splice(0).forEach((b) => b.destroy());
  try {
    const wrong = await within(checkWiden(m, packed, run), "checking the widening of int6");
    if (wrong) throw new Error(`the widening of int6 is wrong on this GPU: ${wrong}`);
  } catch (error) {
    done();
    throw error;
  }
  return {
    into(values, address, groups) {
      copyIn(m, packed, address, groups * 24);
      run(values, groups);
    },
    done,
  };
}
// The check: CHECK_GROUPS groups of random bytes (any 24 bytes are a group), dispatched over rows of 7 workgroups (a
// second dimension, and threads past the last group), into a buffer with 64 groups more of a known byte after them
// (which no thread may write), against JavaScript's to the bit. The reason it is wrong, or null
const CHECK_GROUPS = 1000;
async function checkWiden(m: Model, packed: GPUBuffer, run: (values: Bound, groups: number, most?: number) => void) {
  const bytes = new Uint8Array(CHECK_GROUPS * 24).map(() => (Math.random() * 256) | 0), after = 64 * 32, owned: GPUBuffer[] = [];
  try {
    const values = buffer(m, CHECK_GROUPS * 32 + after, STORAGE | COPY_DST | COPY_SRC, owned);
    m.device.queue.writeBuffer(values, 0, new Uint8Array(CHECK_GROUPS * 32 + after).fill(0x5a));
    m.device.queue.writeBuffer(packed, 0, bytes);
    run(values, CHECK_GROUPS, 7);
    const got = new Int8Array(await readBack(m, values, CHECK_GROUPS * 32 + after)), want = sixValues(bytes);
    const at = want.findIndex((v, i) => got[i] !== v);
    if (at >= 0) return `value ${at % 32} of group ${(at / 32) | 0} is ${got[at]}, JavaScript's ${want[at]}`;
    const past = got.subarray(want.length).findIndex((v) => v !== 0x5a);
    return past >= 0 ? `byte ${past} past the last group was written` : null;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

export { upload, uploadLayers, uploadRest, tablesOf, tablesOn, routed, unwritten };
export type { Weights, MatrixOn, TablePiece };
