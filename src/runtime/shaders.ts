// shaders.ts (T135): the WGSL of this project in one place. The GPU section of /benchmark/ (src/runtime/benchmark/gpu.js,
// T134) measures with some of them; the model's GPU worker (src/runtime/gpu.ts) runs a prompt's tokens through the layers
// with the tiled matrices of T146 (the one each device runs fastest, T147), the flash attention and the steps of a
// layer below them. A plain ES module, which a bundler links into each of them (T367.3).
//
// The weights are this project's int8: values in groups of GROUP with one float32 scale each (llama2_numpy's layout),
// four values to a u32 as the shaders read them. Every matrix times vector: one workgroup of 64 per row, each thread a
// word (4 weights) at a time with the stride of the workgroup, so that neighbours read neighbouring words; the partial
// sums add up in the workgroup's memory. Rows past the most workgroups of a dimension go to its second one.

// T351: this file is the window of the shaders: every name it exported as one file, and nothing else. The texts and
// their makers are in the modules of shaders/, by family: common.ts (what all share), bench.ts (/benchmark/'s own: the
// first matrix products, the ceilings), prompt.ts (a prompt's tiled matrix products), steps.ts (the quantizing and the
// small steps of a layer), attention.ts (flash attention by tiles and for one token), matvec.ts (a matrix times one
// vector, the benchmark's forms), fused.ts (the fused layer of a generated token), run.ts (a run of generated tokens:
// its state, the embedding, the outliers), sample.ts (the sampling), stages.ts (the sampling in chunks, the benchmark's),
// likecpu.ts (the CPU's sampling in JavaScript: no shader), engine.ts (T379.1: the catalogue of the engine's shaders,
// which the key hashes and the model's worker takes them from) and key.ts (the device's key). Each is a static import
// (T367.3: a bundler links them).
// T367.5: who reads the window. /benchmark/'s GPU worker (benchmark/gpu/device.js), which measures with nearly all of
// them and fetches this module when its section runs; and the tools (the net's shaders section, tests/gpu-check.mjs,
// tests/smoke.mjs and others), to which shaders.ts is the name of all the WGSL there is. The model's GPU worker does
// not: its modules take the shaders they run from the catalogue (shaders/engine.ts's ENGINE, T379.1) and the other
// names they use from the parts (gpu/*.ts), linked into the worker's own file without /benchmark/'s shaders, and the
// model's worker takes the device's key from shaders/key.ts.
import { GROUP } from "./shaders/common.ts";
import { TILE, WIDEN, PACKED, BATCHED, ARGMAX, EMPTY, SMALL, CEILING_WORKGROUP, FMA_PER_LOOP, DOT4_PER_LOOP,
  SHARED_PER_LOOP, GLOBAL_PER_THREAD, FMA_SHAPES, fmaCeiling, DOT4_CEILING, SHARED_CEILING,
  GLOBAL_CEILING } from "./shaders/bench.ts";
import { REG_TILES, regTileShape, regTileBytes, DP4A_SHAPE, regTile, TFJS_SHAPE, tfjsTile, ternaryValues, dp4a,
  promptForms, quantizedLikeCpu, TILED_LINE, tiledOff } from "./shaders/prompt.ts";
import { QUANTIZE, WIDEN_SIX_WORKGROUP, WIDEN_SIX, sixDispatch, sixValues, RMSNORM, HEAD_NORM, LAYER_NORM, ADD, ROPE,
  SWIGLU, GELU, RMSNORM_ADD, GEGLU, SHORT_CONV, CONV_STATE,
  DELTA_GATES, DELTA_CONV, DELTA_CONV_STATE, DELTA_NET, DELTA_THREADS, OUTPUT_GATE } from "./shaders/steps.ts";
import { FLASH_Q_TILE, flashShape, flashTile, FLASH_VEC_KV_TILE, FLASH_VEC_LANES, flashVecShape, flashVecSplits,
  flashVecPartsBytes, flashVecParams, flashVec, flashVecReduce, TOKEN_ATTENTION_PAST, TOKEN_ATTENTION_STEEP,
  TOKEN_ATTENTION_PEAK, tokenAttentionData, tokenAttentionOff } from "./shaders/attention.ts";
import { MUL_MAT_VEC_ROWS, ORT_DP4A_MATVEC_ROWS, TOKEN_ROPE, fusedMatVec, NORM_QUANTIZE,
  fusedDp4aMatVec } from "./shaders/fused.ts";
import { ORT_MATVEC_ROWS, mulMatVec, ortMatVec, ortDp4aMatVec } from "./shaders/matvec.ts";
import { REPETITION_WINDOW, STOPS_MOST, STATE_NOT_FINITE, STATE_BYTES, halvesOf, samplingState, SAMPLING_BYTES,
  samplingSettings, EMBED, EMBED_ROWS, EMBED_TERNARY, EMBED_ROWS_TERNARY, OUTLIERS_MOST, TAKE_OUTLIERS, TERNARY_COLUMNS,
  outliersOf } from "./shaders/run.ts";
import { SAMPLE } from "./shaders/sample.ts";
import { ROTATE_MOST, ROTATE_QUANTIZE, UNROTATE, rotateLikeCpu, unrotateLikeCpu, rotateGroups } from "./shaders/rotate.ts";
import { SAMPLE_CHUNK, sampleChunks, samplePartsBytes, SAMPLE_MAX, SAMPLE_SUM, SAMPLE_GATHER, SAMPLE_PICK,
  SAMPLER_STAGES } from "./shaders/stages.ts";
import { penalizeLikeCpu, sampleLikeCpu, argmaxLikeCpu, finiteLikeCpu, walkLikeCpu } from "./shaders/likecpu.ts";
import { ENGINE_SHADERS, TERNARY_SHADERS, ENGINE } from "./shaders/engine.ts";
import { devicePromptForms, deviceKey } from "./shaders/key.ts";
export { GROUP, TILE, WIDEN, PACKED, BATCHED, REG_TILES, regTileShape, regTileBytes, DP4A_SHAPE, regTile, TFJS_SHAPE,
  tfjsTile, ternaryValues, dp4a, QUANTIZE, WIDEN_SIX_WORKGROUP, WIDEN_SIX, sixDispatch, sixValues, promptForms,
  quantizedLikeCpu, TILED_LINE, tiledOff, ARGMAX, EMPTY, SMALL, RMSNORM, HEAD_NORM, LAYER_NORM, ADD, ROPE, FLASH_Q_TILE,
  flashShape, flashTile, FLASH_VEC_KV_TILE, FLASH_VEC_LANES, flashVecShape, flashVecSplits, flashVecPartsBytes,
  flashVecParams, flashVec, flashVecReduce, TOKEN_ATTENTION_PAST, TOKEN_ATTENTION_STEEP, TOKEN_ATTENTION_PEAK,
  tokenAttentionData, tokenAttentionOff, SWIGLU, GELU, RMSNORM_ADD, GEGLU, SHORT_CONV, CONV_STATE, DELTA_GATES, DELTA_CONV, DELTA_CONV_STATE, DELTA_NET, DELTA_THREADS, OUTPUT_GATE, CEILING_WORKGROUP, FMA_PER_LOOP, DOT4_PER_LOOP, SHARED_PER_LOOP,
  GLOBAL_PER_THREAD, FMA_SHAPES, fmaCeiling, DOT4_CEILING, SHARED_CEILING, GLOBAL_CEILING, MUL_MAT_VEC_ROWS,
  ORT_MATVEC_ROWS, ORT_DP4A_MATVEC_ROWS, mulMatVec, ortMatVec, ortDp4aMatVec, TOKEN_ROPE, fusedMatVec, NORM_QUANTIZE,
  fusedDp4aMatVec, REPETITION_WINDOW, STOPS_MOST, STATE_NOT_FINITE, STATE_BYTES, halvesOf, samplingState,
  SAMPLING_BYTES, samplingSettings, EMBED, EMBED_ROWS, EMBED_TERNARY, EMBED_ROWS_TERNARY, OUTLIERS_MOST, TAKE_OUTLIERS,
  TERNARY_COLUMNS, outliersOf, SAMPLE, SAMPLE_CHUNK, sampleChunks, samplePartsBytes, SAMPLE_MAX, SAMPLE_SUM,
  SAMPLE_GATHER, SAMPLE_PICK, SAMPLER_STAGES, penalizeLikeCpu, sampleLikeCpu, argmaxLikeCpu, finiteLikeCpu, walkLikeCpu,
  ROTATE_MOST, ROTATE_QUANTIZE, UNROTATE, rotateLikeCpu, unrotateLikeCpu, rotateGroups, ENGINE_SHADERS, TERNARY_SHADERS, ENGINE, devicePromptForms, deviceKey };
