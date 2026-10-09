// shaders.js (T135): the WGSL of this project in one place. The GPU section of /benchmark/ (public/benchmark/gpu.js,
// T134) measures with some of them; the model's GPU worker (public/gpu.js) runs a prompt's tokens through the layers
// with the tiled matrices of T146 (the one each device runs fastest, T147), the flash attention and the steps of a
// layer below them. A plain ES module: both import it with the ?v= of their
// own URL (GitHub Pages keeps a file for ten minutes: all must come from the same deployment).
//
// The weights are this project's int8: values in groups of GROUP with one float32 scale each (llama2_numpy's layout),
// four values to a u32 as the shaders read them. Every matrix times vector: one workgroup of 64 per row, each thread a
// word (4 weights) at a time with the stride of the workgroup, so that neighbours read neighbouring words; the partial
// sums add up in the workgroup's memory. Rows past the most workgroups of a dimension go to its second one.

// T351: this file is the window of the shaders: every name it exported as one file, and nothing else. The texts and
// their makers are in the modules of shaders/, by family: common.js (what all share), bench.js (/benchmark/'s own: the
// first matrix products, the ceilings), prompt.js (a prompt's tiled matrix products), steps.js (the quantizing and the
// small steps of a layer), attention.js (flash attention by tiles and for one token), matvec.js (a matrix times one
// vector, the benchmark's forms), fused.js (the fused layer of a generated token), run.js (a run of generated tokens:
// its state, the embedding, the outliers), sample.js (the sampling), stages.js (the sampling in chunks, the benchmark's),
// likecpu.js (the CPU's sampling in JavaScript: no shader) and key.js (the device's key). Each is asked for with this
// file's own ?v=<build>, so that all come from one deployment, and all at once: a part's own import of a neighbour then
// finds it asked for already, where one after another's end would add a round trip for each.
const modules = Object.fromEntries(["common", "bench", "prompt", "steps", "attention", "matvec", "fused", "run", "sample", "stages", "likecpu", "key"].map((name) =>
  [name, import(new URL(`shaders/${name}.js${new URL(import.meta.url).search}`, import.meta.url))]));
const { GROUP } = await modules.common;
const { TILE, WIDEN, PACKED, BATCHED, ARGMAX, EMPTY, SMALL, CEILING_WORKGROUP, FMA_PER_LOOP, DOT4_PER_LOOP,
  SHARED_PER_LOOP, GLOBAL_PER_THREAD, FMA_SHAPES, fmaCeiling, DOT4_CEILING, SHARED_CEILING,
  GLOBAL_CEILING } = await modules.bench;
const { REG_TILES, regTileShape, regTileBytes, DP4A_SHAPE, regTile, TFJS_SHAPE, tfjsTile, ternaryValues, dp4a,
  promptForms, quantizedLikeCpu, TILED_LINE, tiledOff } = await modules.prompt;
const { QUANTIZE, WIDEN_SIX_WORKGROUP, WIDEN_SIX, sixDispatch, sixValues, RMSNORM, HEAD_NORM, LAYER_NORM, ADD, ROPE,
  SWIGLU, GELU } = await modules.steps;
const { FLASH_Q_TILE, flashShape, flashTile, FLASH_VEC_KV_TILE, FLASH_VEC_LANES, flashVecShape, flashVecSplits,
  flashVecPartsBytes, flashVecParams, flashVec, flashVecReduce, TOKEN_ATTENTION_PAST, TOKEN_ATTENTION_STEEP,
  TOKEN_ATTENTION_PEAK, tokenAttentionData, tokenAttentionOff } = await modules.attention;
const { MUL_MAT_VEC_ROWS, ORT_MATVEC_ROWS, ORT_DP4A_MATVEC_ROWS, mulMatVec, ortMatVec,
  ortDp4aMatVec } = await modules.matvec;
const { TOKEN_ROPE, fusedMatVec, NORM_QUANTIZE, fusedDp4aMatVec, ternaryMatVec } = await modules.fused;
const { REPETITION_WINDOW, STOPS_MOST, STATE_NOT_FINITE, STATE_BYTES, halvesOf, samplingState, SAMPLING_BYTES,
  samplingSettings, EMBED, EMBED_ROWS, EMBED_TERNARY, EMBED_ROWS_TERNARY, OUTLIERS_MOST, TAKE_OUTLIERS, TERNARY_COLUMNS,
  outliersOf } = await modules.run;
const { SAMPLE } = await modules.sample;
const { SAMPLE_CHUNK, sampleChunks, samplePartsBytes, SAMPLE_MAX, SAMPLE_SUM, SAMPLE_GATHER, SAMPLE_PICK,
  SAMPLER_STAGES } = await modules.stages;
const { penalizeLikeCpu, sampleLikeCpu, argmaxLikeCpu, finiteLikeCpu, walkLikeCpu } = await modules.likecpu;
const { devicePromptForms, deviceKey } = await modules.key;
export { GROUP, TILE, WIDEN, PACKED, BATCHED, REG_TILES, regTileShape, regTileBytes, DP4A_SHAPE, regTile, TFJS_SHAPE,
  tfjsTile, ternaryValues, dp4a, QUANTIZE, WIDEN_SIX_WORKGROUP, WIDEN_SIX, sixDispatch, sixValues, promptForms,
  quantizedLikeCpu, TILED_LINE, tiledOff, ARGMAX, EMPTY, SMALL, RMSNORM, HEAD_NORM, LAYER_NORM, ADD, ROPE, FLASH_Q_TILE,
  flashShape, flashTile, FLASH_VEC_KV_TILE, FLASH_VEC_LANES, flashVecShape, flashVecSplits, flashVecPartsBytes,
  flashVecParams, flashVec, flashVecReduce, TOKEN_ATTENTION_PAST, TOKEN_ATTENTION_STEEP, TOKEN_ATTENTION_PEAK,
  tokenAttentionData, tokenAttentionOff, SWIGLU, GELU, CEILING_WORKGROUP, FMA_PER_LOOP, DOT4_PER_LOOP, SHARED_PER_LOOP,
  GLOBAL_PER_THREAD, FMA_SHAPES, fmaCeiling, DOT4_CEILING, SHARED_CEILING, GLOBAL_CEILING, MUL_MAT_VEC_ROWS,
  ORT_MATVEC_ROWS, ORT_DP4A_MATVEC_ROWS, mulMatVec, ortMatVec, ortDp4aMatVec, TOKEN_ROPE, fusedMatVec, NORM_QUANTIZE,
  fusedDp4aMatVec, ternaryMatVec, REPETITION_WINDOW, STOPS_MOST, STATE_NOT_FINITE, STATE_BYTES, halvesOf, samplingState,
  SAMPLING_BYTES, samplingSettings, EMBED, EMBED_ROWS, EMBED_TERNARY, EMBED_ROWS_TERNARY, OUTLIERS_MOST, TAKE_OUTLIERS,
  TERNARY_COLUMNS, outliersOf, SAMPLE, SAMPLE_CHUNK, sampleChunks, samplePartsBytes, SAMPLE_MAX, SAMPLE_SUM,
  SAMPLE_GATHER, SAMPLE_PICK, SAMPLER_STAGES, penalizeLikeCpu, sampleLikeCpu, argmaxLikeCpu, finiteLikeCpu, walkLikeCpu,
  devicePromptForms, deviceKey };
