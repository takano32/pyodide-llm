// WASM SIMD128 kernels for a llama2.c-style transformer (AssemblyScript).
// Loaded into Pyodide with ctypes as an Emscripten side module, they work in place on NumPy-owned memory.
// No static data and no std math on purpose: nothing relocates a data segment in this hand-made side module.
//
// T356: this file is the window. The kernels are in kernel/, by what they are, and every one of them is named here:
// what this file exports is what the module exports, in this order (the order they had as one file, which is the
// order of the binary's exports: the built files are the same to the byte as before the division). A new kernel is
// written in the file of its kind and added to a line here; a new file of kernel/ is added to the Makefile's rule
// too, or make kernels does not see it change.
//   kernel/matmul.ts       the matrix products (float32, int8, 6 bits, ternary) and what they take beside the
//                          weights: the corrections' sums, interleave, the outliers' columns
//   kernel/quantize.ts     numbers from one form to another: the stored forms widened to float32, float32
//                          quantized to int8, six bits and ternary (the converter's, and quantize_x every token)
//   kernel/attention.ts    rope and attention over a cache of float32 or float16
//   kernel/halves.ts       float16: the cache's keys and values to it and back, and whether they are finite
//   kernel/activations.ts  a vector between the matrix products: the norms, gelu, swiglu, the add, the rotated basis
//   kernel/stateful.ts     the layers with a state: gate, convolve and delta_rule (Qwen3.5), short_conv (LFM2)
//   kernel/sample.ts       argmax, the penalties and the sampling
//   kernel/math.ts         what several of them use: the group's size, a vector's sum, exp, the largest of floats

export { matmul_f32 } from "./kernel/matmul";
export { widen_bf16, widen_q8_0, widen_pq2_0, widen_ptq1_0, quantize_x, quantize6_x } from "./kernel/quantize";
export { six_sums, int8_sums, matmul_q8, interleave, matmul_t2 } from "./kernel/matmul";
export { ternary_x } from "./kernel/quantize";
export { matmul_q6 } from "./kernel/matmul";
export { rmsnorm } from "./kernel/activations";
export { rope, attention, attention_f16 } from "./kernel/attention";
export { to_f16, from_f16, finite_f16 } from "./kernel/halves";
export { layernorm, gelu, swiglu } from "./kernel/activations";
export { add_columns } from "./kernel/matmul";
export { add_inplace } from "./kernel/activations";
export { argmax } from "./kernel/sample";
export { gate, convolve, short_conv, delta_rule } from "./kernel/stateful";
export { penalize, sample } from "./kernel/sample";
export { rotate, unrotate } from "./kernel/activations";
