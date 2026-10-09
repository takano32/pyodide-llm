// The weights onto the GPU: every layer's matrices and vectors and a token's tables, from the shared memory or (a
// model on the GPU alone, T156) as routes of the checkpoint that start.js's take() writes; a matrix past a buffer in
// pieces (T155), the layout of a token's joined matrices (T152), and the widening of int6 (T155) with its check.
// (T352: a module of the model's GPU worker, public/gpu.js, which asks for it with its own ?v=<build>)

const { STORAGE, COPY_DST, COPY_SRC, CHUNK, rowBytes, common, within, buffer, uniform, copyIn, readBack, validated,
  pipelineOf, bind, dispatch } =
  await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));

// every layer's matrices (values and scales, as the checkpoint holds them; T154: no w3 where there is no gate) and its
// vectors: the weights of its two norms, and (T153) Qwen2's biases of q, k and v, Qwen3's norms of a head of q and of
// k, (T154) GPT-2's and GPT-NeoX's biases of the two LayerNorms and of every matrix (plan.vectors: each one's address in
// the shared memory and its floats a layer). T155: a matrix in pieces of whole rows (piecesOf), a buffer of values and
// one of scales each; int6 (plan.matrices' six) widened to int8 on the way (widener), its scales as they are (the
// quarter of an int6 group's is already the int8 values' scale, T98). The addresses are Numbers in a 64-bit memory too
// (exact to 2^53; the views and copies take them as they are). T232: ternary (plan.matrices' ternary) as it is, a row
// of n weights n / 4 bytes (rowBytes)
// T152: where a token goes on the GPU too (plan.tokens), q, k and v are one matrix a layer, and gate and up one (the
// layer of a token reads them so: shaders.js's fusedMatVec and fusedDp4aMatVec, T150 and T175), each a range of it for
// the prompt's tiled shaders (tokensLayout), and the classifier, the embedding, the final norm and RoPE's table go up
// as well (uploadTokens)
// T156, a model on the GPU alone (m.direct): the layers' buffers are made as the worker opens, before a byte comes, and
// each stretch of the checkpoint they hold is a route ([start, end) in the checkpoint, the buffer and its offset) that
// take() writes the bytes to as they come; the rest (the tables, the norms) goes up from the shared memory as start()
// is asked (uploadRest)
async function upload(m) {
  try {
    const bytes = await uploadLayers(m);
    if (common.stopping) return bytes;
    return bytes + (await uploadRest(m));
  } finally {
    m.widen?.done();
  }
}
async function uploadLayers(m) {
  const { plan } = m, group = m.wgsl.GROUP;
  let bytes = 0;
  m.matrices = Object.fromEntries(Object.entries(plan.matrices).map(([name, { rows, n, ternary }]) =>
    [name, { rows, n, ternary, pieces: piecesOf(m, rows, rowBytes({ n, ternary })).map(([first, count]) => ({ first, rows: count, layers: [] })) }]));
  const together = plan.tokens ? tokensLayout(m) : null;
  m.together = together;
  // (a model on the GPU alone opens before its tables are placed, and is int8: T156)
  const tables = together && !m.direct ? Object.values(tablesOf(plan)) : [];
  const six = [...Object.values(plan.matrices), ...tables].some((matrix) => matrix.six);
  // (kept for the tables, uploadRest; upload() lets it go)
  const widen = m.widen = six ? await widener(m, tables) : null;
  // (T156: a model on the GPU alone reads its first layer back to check a token's layer against: firstLayer)
  const usage = STORAGE | COPY_DST | (m.direct ? COPY_SRC : 0);
  for (let l = 0; l < plan.layers; l++) {
    // a layer's matrices that are one (T152): a buffer of values and one of scales for all of them
    const joined = together && Object.fromEntries(Object.entries(together.sizes).map(([name, [valueBytes, scaleBytes]]) =>
      [name, [buffer(m, valueBytes, usage), buffer(m, scaleBytes, usage)]]));
    if (joined) (m.joined ??= []).push(joined);
    for (const [name, matrix] of Object.entries(plan.matrices)) {
      const [valuesAt, scalesAt] = matrix.layers[l];
      const home = together?.homes[name], row = rowBytes(matrix);
      for (const piece of m.matrices[name].pieces) {
        const valueBytes = piece.rows * row, scaleBytes = (valueBytes / group) * 4;
        // its own buffers, or its range of the layer's joined ones
        const [values, scales] = home
          ? joined[home.joined].map((b, i) => ({ buffer: b, offset: home.at[i], size: i ? scaleBytes : valueBytes }))
          : [buffer(m, valueBytes, usage), buffer(m, scaleBytes, usage)];
        const from = [valuesAt + piece.first * row, scalesAt + (piece.first * row / group) * 4];
        if (m.direct) {
          // (T156: int8 alone, and T232 ternary, whose bytes are the buffer's as they come)
          m.direct.routes.push([from[0], from[0] + valueBytes, values.buffer ?? values, values.offset ?? 0],
            [from[1], from[1] + scaleBytes, scales.buffer ?? scales, scales.offset ?? 0]);
        } else {
          // an int6 row is 3/4 of an int8 one (24 bytes a group of 32)
          if (matrix.six) widen.into(values, valuesAt + (piece.first * matrix.n * 3) / 4, valueBytes / group);
          else copyIn(m, values.buffer ?? values, from[0], valueBytes, values.offset ?? 0);
          copyIn(m, scales.buffer ?? scales, from[1], scaleBytes, scales.offset ?? 0);
        }
        piece.layers.push([values, scales]);
        bytes += valueBytes + scaleBytes;
      }
    }
    if (m.direct) continue;
    // what was written waits in memory until the GPU takes it: let it, before more comes
    await within(m.device.queue.onSubmittedWorkDone(), `layer ${l + 1}'s weights`);
    if (common.stopping) return bytes;
  }
  return bytes;
}
// the tables of a token (where the layers are joined for them, T152), and the norms, from the shared memory
async function uploadRest(m) {
  const { plan } = m;
  let bytes = 0;
  if (m.together) bytes += await uploadTokens(m, m.widen);
  m.vectors = {};
  for (const [name, { at, size }] of Object.entries(plan.vectors)) {
    const vectorBytes = plan.layers * size * 4;
    m.vectors[name] = buffer(m, vectorBytes, STORAGE | COPY_DST);
    copyIn(m, m.vectors[name], at, vectorBytes);
    bytes += vectorBytes;
  }
  return bytes;
}

// T155: the pieces of a matrix of rows of n bytes (rowBytes), [first row, rows] each: whole rows, no more bytes than a buffer the device
// binds (m.limit, or plan.force.pieceBytes in the tests), each piece's first row where its part of the output may be
// bound (a multiple of minStorageBufferOffsetAlignment over the 4 bytes of a float32). One piece where the matrix
// fits, as every matrix of the models of the list does at WebGPU's least limit of 128 MiB (the largest, Qwen2.5 7B's
// w1, is 68 MB): llama.cpp's WebGPU keeps each tensor in one buffer within maxStorageBufferBindingSize and refuses a
// larger one (ggml-webgpu.cpp, ggml_backend_webgpu_buffer_type_get_max_size, commit 2145525a, MIT; no line copied),
// where this cuts it by rows, as T94's design had it for a classifier past a buffer
function piecesOf(m, rows, n, most = m.plan.force.pieceBytes) {
  const bytes = Math.min(m.limit, most ?? Infinity), align = m.device.limits.minStorageBufferOffsetAlignment / 4;
  if (rows * n <= bytes) return [[0, rows]];
  const step = Math.floor(bytes / n / align) * align;
  if (!step) throw new Error(`${align} rows of ${n} weights are more than a buffer of this GPU (${bytes} bytes)`);
  return Array.from({ length: Math.ceil(rows / step) }, (_, i) => [i * step, Math.min(step, rows - i * step)]);
}

// T152: the tables a token needs besides the layers (plan.tokens; T210, a model on the GPU alone as it opens:
// plan.tables): the classifier, and the embedding where it is another table (else the classifier is the embedding
// too), each made by make(spec) where it is given, else the specs themselves
function tablesOf(plan, make = (spec) => spec) {
  const { classifier, embedding } = plan.tables ?? plan.tokens, made = make(classifier);
  return { classifier: made, embedding: embedding ? make(embedding) : made };
}
// T209: a table ({ rows, n }) in pieces of rows where it is past what the device binds (piecesOf), a buffer of values
// and one of scales each, [{ first, rows, values, scales }]; fill(piece, valueBytes, scaleBytes) puts its bytes there
function tablePieces(m, spec, fill, usage = STORAGE | COPY_DST) {
  return piecesOf(m, spec.rows, rowBytes(spec), m.plan.force.tablePieceBytes ?? m.plan.force.pieceBytes).map(([first, count]) => {
    const valueBytes = count * rowBytes(spec), scaleBytes = (valueBytes / m.wgsl.GROUP) * 4;
    const piece = { first, rows: count, values: buffer(m, valueBytes, usage), scales: buffer(m, scaleBytes, usage) };
    fill(piece, valueBytes, scaleBytes);
    return piece;
  });
}
// T210: the tables of a model on the GPU alone, made as it opens (plan.tables: where each starts in the checkpoint),
// their stretches routes that take() writes the bytes to as they come; COPY_SRC for the check's rows (checkTokens).
// The bytes on the GPU
function tablesOn(m) {
  const group = m.wgsl.GROUP;
  let bytes = 0;
  m.tables = tablesOf(m.plan, (spec) => tablePieces(m, spec, ({ first, values, scales }, valueBytes, scaleBytes) => {
    const [valuesAt, scalesAt] = spec.at, from = [valuesAt + first * rowBytes(spec), scalesAt + (first * rowBytes(spec) / group) * 4];
    m.direct.routes.push([from[0], from[0] + valueBytes, values, 0], [from[1], from[1] + scaleBytes, scales, 0]);
    bytes += valueBytes + scaleBytes;
  }, STORAGE | COPY_DST | COPY_SRC));
  return bytes;
}
// T152: how a token's layer holds its matrices (shaders.js's fusedMatVec and fusedDp4aMatVec read q, k and v as one
// matrix, gate and up as one, up's rows after gate's): a buffer of values and one of scales a layer for each (sizes),
// and each matrix's range of them (homes: its offsets in bytes), which the prompt's tiled shaders bind as a matrix of
// its own. A range must start where the device binds a buffer (minStorageBufferOffsetAlignment: a matrix of a
// multiple of 2048 weights, the values and the scales alike at 256; stories15M's k starts at 82944 weights, T150's
// (b); T232: of 8192 ternary weights, whose scales are a byte to 32 weights), and a matrix a token reads must be one piece (T155). The tables may be in pieces (T209: uploadTokens). null
// where it cannot, and why in m.tokensWhy: the prompts go on the GPU as before, the tokens stay on the CPU
function tokensLayout(m) {
  const { plan } = m, align = m.device.limits.minStorageBufferOffsetAlignment, group = m.wgsl.GROUP;
  const why = (reason) => {
    m.tokensWhy = reason;
    return null;
  };
  // (T226: GPT-2's and GPT-NeoX's FFN has no gate: w1 is a matrix of its own, as wo and w2 are)
  const gated = Boolean(m.matrices.w3);
  for (const name of ["wo", "w2", ...(gated ? [] : ["w1"])]) if (m.matrices[name].pieces.length > 1) return why(`${name} is past a buffer of this GPU`);
  const homes = {}, sizes = {};
  for (const [joined, names] of Object.entries({ qkv: ["wq", "wk", "wv"], ...(gated ? { gateUp: ["w1", "w3"] } : {}) })) {
    let values = 0, scales = 0;
    for (const name of names) {
      const matrix = m.matrices[name];
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
// T152: the tables onto the GPU (int6 widened as a layer's matrices are), the final norm's weights, and RoPE's table
// of every position, a row a position: the cos of its headSize / 2 angles, then their sin (shaders.js's fusedMatVec
// reads it so, T151; from the CPU's two tables, Llama 3's scaling in them)
// T209: a table in pieces of rows where it is past what the device binds (piecesOf: Llama 3.2 3B's classifier is
// 394 MB, the owner's Android binds 256 MiB; T152's review (e)), [{ first, rows, values, scales }] each. The
// classifier is then a dispatch a piece into its range of the logits, and EMBED one a piece (tokenPass)
// T210: a model on the GPU alone makes them as it opens, their bytes routes of the checkpoint as the layers' are
// (tablesOn), and only the final norm and RoPE's table come from the shared memory here
async function uploadTokens(m, widen) {
  const { plan } = m, group = m.wgsl.GROUP, half = plan.headSize / 2;
  let bytes = 0;
  m.tables ??= tablesOf(plan, (spec) => tablePieces(m, spec, (piece, valueBytes, scaleBytes) => {
    const { first, values, scales } = piece, { n, six, at: [valuesAt, scalesAt] } = spec;
    if (six) widen.into(values, valuesAt + (first * n * 3) / 4, valueBytes / group);
    else copyIn(m, values, valuesAt + first * rowBytes(spec), valueBytes);
    copyIn(m, scales, scalesAt + (first * rowBytes(spec) / group) * 4, scaleBytes);
    bytes += valueBytes + scaleBytes;
  }));
  // (T226: the final LayerNorm's bias, and GPT-2's learned positions, a row a position as the CPU widened them: a
  // step adds its position's row to the embedding's, as the CPU's embed() does)
  const vector = (address, floats) => {
    const made = buffer(m, floats * 4, STORAGE | COPY_DST);
    copyIn(m, made, address, floats * 4);
    bytes += floats * 4;
    return made;
  };
  m.finalNorm = vector(plan.tokens.final, plan.dim);
  m.finalBias = plan.tokens.finalBias ? vector(plan.tokens.finalBias, plan.dim) : null;
  m.positions = plan.tokens.positions ? vector(plan.tokens.positions, plan.seqLen * plan.dim) : null;
  m.angleTable = buffer(m, plan.seqLen * plan.headSize * 4, STORAGE | COPY_DST);
  const rows = Math.max(1, Math.floor(CHUNK / (plan.headSize * 4))), chunk = new Float32Array(rows * plan.headSize);
  // (GPT-2 turns nothing and has no tables: the buffer stays, bound and never read)
  for (let p = 0; plan.turned && p < plan.seqLen; p += rows) {
    const count = Math.min(rows, plan.seqLen - p);
    for (let r = 0; r < count; r++) {
      chunk.set(new Float32Array(m.memory.buffer, plan.cos + (p + r) * half * 4, half), r * plan.headSize);
      chunk.set(new Float32Array(m.memory.buffer, plan.sin + (p + r) * half * 4, half), r * plan.headSize + half);
    }
    m.device.queue.writeBuffer(m.angleTable, p * plan.headSize * 4, chunk, 0, count * plan.headSize);
  }
  bytes += plan.seqLen * plan.headSize * 4;
  await within(m.device.queue.onSubmittedWorkDone(), "the classifier's weights");
  return bytes;
}

// T155: shaders.js's WIDEN_SIX compiled and checked against JavaScript (sixValues), and into(values, address, groups):
// groups of int6 at address in the shared memory widened into values (a buffer of int8, or T152 a range of one) on
// the GPU. The packed bytes go
// through one buffer of their own, written again for every piece (the queue runs a write after the submissions before
// it); done() lets it go.
async function widener(m, tables = []) {
  const { device, wgsl } = m;
  const pipeline = await within(validated(m, () => pipelineOf(m, wgsl.WIDEN_SIX)), "compiling the widening of int6");
  const owned = [];
  // (T152: and the classifier's and the embedding's, one piece each)
  const largest = Math.max(CHECK_GROUPS * 24, ...Object.entries(m.plan.matrices).filter(([, matrix]) => matrix.six)
    .flatMap(([name, { n }]) => m.matrices[name].pieces.map((piece) => (piece.rows * n * 3) / 4)),
  ...tables.filter((table) => table.six).map(({ rows, n }) => (rows * n * 3) / 4));
  const packed = buffer(m, largest, STORAGE | COPY_DST, owned);
  const run = (values, groups, most = device.limits.maxComputeWorkgroupsPerDimension) => {
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, pipeline, bind(m, pipeline, [packed, values, uniform(m, new Uint32Array([groups, 0, 0, 0]), owned)]),
      ...wgsl.sixDispatch(groups, most));
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
async function checkWiden(m, packed, run) {
  const bytes = new Uint8Array(CHECK_GROUPS * 24).map(() => (Math.random() * 256) | 0), after = 64 * 32, owned = [];
  try {
    const values = buffer(m, CHECK_GROUPS * 32 + after, STORAGE | COPY_DST | COPY_SRC, owned);
    m.device.queue.writeBuffer(values, 0, new Uint8Array(CHECK_GROUPS * 32 + after).fill(0x5a));
    m.device.queue.writeBuffer(packed, 0, bytes);
    run(values, CHECK_GROUPS, 7);
    const got = new Int8Array(await readBack(m, values, CHECK_GROUPS * 32 + after)), want = m.wgsl.sixValues(bytes);
    const at = want.findIndex((v, i) => got[i] !== v);
    if (at >= 0) return `value ${at % 32} of group ${(at / 32) | 0} is ${got[at]}, JavaScript's ${want[at]}`;
    const past = got.subarray(want.length).findIndex((v) => v !== 0x5a);
    return past >= 0 ? `byte ${past} past the last group was written` : null;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

export { upload, uploadLayers, uploadRest, tablesOf, tablesOn };
