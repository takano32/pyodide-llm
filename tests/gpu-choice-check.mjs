// T148: how forward.js weighs a block of a prompt on the GPU against the CPU (promptTimes), in Node, without a GPU:
//   node tests/gpu-choice-check.mjs
// Made-up times: the CPU's ms a token of its blocks, gpu.js's two blocks timed as it starts, the blocks the GPU then ran.
import assert from "node:assert/strict";
import fs from "node:fs";
import { aloneHolds, BOTH_ON_8, aloneVerdict, cpuReadBytes, footprint, gpuBytes, gpuLine, gpuOnlyPlan, gpuOnlyUnfit, gpuOnlyWeights, gpuHoles, layerWeightsOf, placer, PROMPTS_CPU,
  PROMPTS_GPU, PROMPTS_UNTIMED, promptTimes, tokenTimes, USAGE_UNKNOWN, weightsPlace } from "../public/forward.js";
import { deviceKey, fusedDp4aMatVec, halvesOf, OUTLIERS_MOST, outliersOf, quantizedLikeCpu, ternaryMatVec, ternaryValues, tiledOff, tokenAttentionData,
  tokenAttentionOff } from "../public/shaders.js";
import { usedAfter } from "../src/bench.js";

// a GPU with a fixed cost of 40 ms a block and 0.5 ms a token (16 tokens 48 ms, 64 tokens 72 ms)
const started = [{ count: 16, ms: 48 }, { count: 64, ms: 72 }];

// nothing is known before the CPU has run TIMED blocks on the threads in use: the prompt stays on the CPU
{
  const times = promptTimes();
  assert.equal(times.of(64, 4), null, "no GPU yet");
  times.started(started);
  assert.equal(times.of(64, 4), null, "no CPU yet");
  times.cpu(4, 2);
  assert.equal(times.of(64, 4), null, "one CPU block is not enough (the first after a pause is the slowest)");
  times.cpu(4, 2);
  assert.ok(times.of(64, 4), "two are");
  assert.equal(times.of(64, 2), null, "another number of threads is timed anew");
}

// a CPU of 2 ms a token: 64 tokens 128 ms against the GPU's 72: the GPU; 16 tokens 32 ms against 48: the CPU. The
// threshold is the fewest tokens the GPU is faster for by more than BETTER (0.95): 40 + 0.5 n < 0.95 × 2 n, n > 28.6
{
  const times = promptTimes();
  times.started(started);
  times.cpu(4, 2);
  times.cpu(4, 2);
  const whole = times.of(64, 4), short = times.of(16, 4);
  assert.equal(whole.cpu, 128);
  assert.equal(whole.gpu, 72);
  assert.equal(whole.faster, true, "a whole block goes to the GPU");
  assert.equal(short.faster, false, "a short one stays on the CPU");
  assert.equal(times.threshold(64, 4), 29);
}

// a GPU that runs its blocks for real three times as slow as it timed itself: the line is scaled, and the CPU wins
{
  const times = promptTimes();
  times.started(started);
  times.cpu(1, 1.5);
  times.cpu(1, 1.5);
  assert.equal(times.of(64, 1).faster, true, "timed at the start: 72 against 96");
  for (let i = 0; i < 3; i++) times.gpu(64, 216);
  assert.equal(times.of(64, 1).gpu, 216);
  assert.equal(times.of(64, 1).faster, false, "216 against 96");
  assert.equal(times.threshold(64, 1), 65, "none: every block on the CPU");
}

// the medians of the last five: one slow block does not move the verdict, five do
{
  const times = promptTimes();
  times.started(started);
  for (const ms of [2, 2, 2, 2, 50]) times.cpu(4, ms);
  assert.equal(times.of(64, 4).cpu, 128, "one slow block among five");
  for (let i = 0; i < 5; i++) times.cpu(4, 50);
  assert.equal(times.of(64, 4).cpu, 3200, "only the last five count");
}

// a GPU whose blocks of 16 took as long as those of 64 (the tiles): a flat line, never below 0 a token
{
  const times = promptTimes();
  times.started([{ count: 16, ms: 80 }, { count: 64, ms: 70 }]);
  times.cpu(4, 1);
  times.cpu(4, 1);
  assert.equal(times.of(16, 4).gpu, 80);
  assert.equal(times.of(64, 4).gpu, 80);
}

// T152: a generation's steps (tokenTimes): nothing before the GPU has a time and the CPU TIMED ones on the threads in
// use; the GPU where a step takes less than 0.95 of the CPU's; the lower medians of the last five
{
  const steps = tokenTimes();
  assert.equal(steps.of(4), null, "nothing timed");
  steps.gpu(10);
  steps.cpu(4, 12);
  assert.equal(steps.of(4), null, "one CPU step is not enough");
  steps.cpu(4, 12);
  assert.deepEqual(steps.of(4), { cpu: 12, gpu: 10, faster: true }, "10 < 0.95 × 12");
  assert.equal(steps.of(2), null, "another number of threads is timed anew");
  for (const ms of [12, 12, 12]) steps.gpu(ms);
  assert.equal(steps.of(4).faster, false, "12 is not below 0.95 × 12");
  for (const ms of [5, 5, 50]) steps.gpu(ms);
  assert.equal(steps.of(4).gpu, 12, "the lower median of the last five: 12, 12, 5, 5, 50");
  steps.gpu(5);
  assert.equal(steps.of(4).gpu, 5, "12, 5, 5, 50, 5");
  assert.equal(steps.of(4).faster, true);
}
// T152's review (T160): halvesOf, a float32 cache's keys and values narrowed on the way up to the GPU. Every float16
// back to itself (NaN to a NaN), the float32 half way between two neighbours to the even one, and a step of float32
// either side of it to the nearer one; past the largest to the infinity, and below half the least subnormal to 0
{
  const toFloat = (h) => {
    const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
    if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
    return exponent ? sign * 2 ** (exponent - 15) * (1 + fraction / 1024) : sign * 2 ** -24 * fraction;
  };
  const one = (x) => halvesOf(Float32Array.of(x), new Uint16Array(1))[0];
  const nan = (h) => (h & 0x7c00) === 0x7c00 && (h & 0x3ff) !== 0;
  for (let h = 0; h < 0x10000; h++) {
    if (nan(h)) assert.ok(nan(one(toFloat(h))), `NaN ${h.toString(16)}`);
    else assert.equal(one(toFloat(h)), h, `float16 ${h.toString(16)} back to itself`);
  }
  const f32 = new Float32Array(1), bits = new Uint32Array(f32.buffer);
  const beside = (x, step) => { f32[0] = x; bits[0] += step; return f32[0]; };
  for (const sign of [0, 0x8000]) {
    // (h and h + 1 finite; a float32 one step nearer 0 than the middle goes to h, one step farther to h + 1)
    for (let h = sign; h < sign + 0x7bff; h++) {
      const middle = (toFloat(h) + toFloat(h + 1)) / 2;  // exact in float32
      assert.equal(one(middle), h & 1 ? h + 1 : h, `half way above ${h.toString(16)} to the even one`);
      assert.equal(one(beside(middle, -1)), h, `just short of half way above ${h.toString(16)}`);
      assert.equal(one(beside(middle, 1)), h + 1, `just past half way above ${h.toString(16)}`);
    }
  }
  assert.equal(one(65520), 0x7c00, "past the largest: the infinity");
  assert.equal(one(2 ** -25), 0, "half the least subnormal: 0 (the even one)");
  assert.equal(one(-(2 ** -26)), 0x8000, "below it: -0");
}
// T225's review: heldHalves, with which /benchmark/'s layer check holds a key or a value of the cache to a float16 next
// to its own (WGSL leaves to the device which of the two neighbours a conversion gives, §15.7.6, and Direct3D takes the
// one toward zero, D3D11.3 3.2.2: the owner's NVIDIA PC on Windows rounded every key and value that way). A device that
// rounds to the nearest, toward zero or away from zero is right, and the reference goes on with its bits; one that is
// a float16 farther off is not (counted "farther", and the reference keeps its own nearest). Nothing else in the
// repository rounds toward zero (lavapipe and SwiftShader round to the nearest), so this is where that is held
{
  globalThis.onmessage = null;  // (the worker's file sets it as a module's plain assignment)
  const { fromHalf, halvesSaid, farthest, heldHalves, toHalf } = await import("../public/benchmark/gpu.js");
  const toFloat = (h) => {
    const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
    return exponent ? sign * 2 ** (exponent - 15) * (1 + fraction / 1024) : sign * 2 ** -24 * fraction;
  };
  const nearest = (x) => halvesOf(Float32Array.of(x), new Uint16Array(1))[0];
  // the neighbours by hand: toward zero (the nearest, or the one below it where the nearest is the larger) and away
  const inwards = (x) => (Math.abs(toFloat(nearest(x))) > Math.abs(x) ? nearest(x) - 1 : nearest(x));
  const outwards = (x) => (Math.abs(toFloat(nearest(x))) < Math.abs(x) ? nearest(x) + 1 : nearest(x));
  let seed = 12345;
  const random = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32;
  // 600 values of the size of keys and values (either sign, 2^-13 to 2^6 times 1 to 2: past the subnormals' edge at
  // 2^-14 too), float32 as the GPU's are, and a few the float16 holds exactly
  const values = Float64Array.from({ length: 600 }, () => Math.fround((random() < 0.5 ? -1 : 1) * 2 ** (-13 + 19 * random()) * (1 + random())));
  values.set([0, 2 ** -24, -(2 ** -24), 1, -2, 0.5, 8], 0);
  const counts = (got) => {
    const held = heldHalves(values, Uint16Array.from(values, got));
    return { held, line: [held.same, held.inward, held.outward, held.far] };
  };
  // toHalf (the check's) and halvesOf (the engine's) are the same rounding; fromHalf takes every finite float16 back
  for (const x of values) assert.equal(toHalf(x), nearest(x), `toHalf of ${x}`);
  for (let h = 0; h < 0x10000; h++) if ((h & 0x7c00) !== 0x7c00) assert.equal(fromHalf(h), toFloat(h), `fromHalf of ${h.toString(16)}`);
  // to the nearest: all of them
  assert.deepEqual(counts(nearest).line, [values.length, 0, 0, 0]);
  // toward zero: the ones that differ from the nearest are neighbours toward zero, about half of them, taken as the
  // GPU's; and away from zero the same the other way
  const inward = values.filter((x) => inwards(x) !== nearest(x)).length, outward = values.filter((x) => outwards(x) !== nearest(x)).length;
  assert.ok(inward > 200 && inward < 400 && outward > 200 && outward < 400, `about half round the other way: ${inward} and ${outward}`);
  {
    const { held, line } = counts(inwards);
    assert.deepEqual(line, [values.length - inward, inward, 0, 0]);
    assert.deepEqual([...held.bits], [...Uint16Array.from(values, inwards)], "the reference goes on with the GPU's bits");
    assert.deepEqual([...held.nearest], [...Uint16Array.from(values, nearest)], "and has its own nearest beside them");
  }
  assert.deepEqual(counts(outwards).line, [values.length - outward, 0, outward, 0]);
  // a float16 farther than that (one below the neighbour toward zero: 1 to 2 ulp off the value), on the elements in the top
  // quarter of the row's largest, where the slack (HALF_SLACK, 1e-5 of the largest) is under a tenth of an ulp: not
  // taken (the reference keeps its own nearest), counted farther, but for the few that lie within the slack of a float16
  {
    const largest = values.reduce((most, x) => Math.max(most, Math.abs(x)), 0), top = [...values.keys()].filter((i) => Math.abs(values[i]) >= largest / 4);
    const got = Uint16Array.from(values, (x) => (Math.abs(x) >= largest / 4 ? inwards(x) - 1 : inwards(x)));
    const held = heldHalves(values, got);
    assert.ok(top.length > 30, `${top.length} in the top quarter`);
    assert.ok(held.far >= 0.85 * top.length, `${held.far} farther of ${top.length}`);
    for (const i of top) if (held.bits[i] !== got[i]) assert.equal(held.bits[i], held.nearest[i], `element ${i} (${values[i]}) not taken`);
    assert.equal(held.same + held.inward + held.outward + held.far, values.length);
  }
  // the slack: a tiny element (1e-7) the GPU gives as 5e-5 is within 1e-5 of the largest (8: 8e-5), as 2e-4 is not
  {
    const x = Float64Array.of(8, 1e-7), within = heldHalves(x, Uint16Array.of(nearest(8), nearest(5e-5))), beyond = heldHalves(x, Uint16Array.of(nearest(8), nearest(2e-4)));
    assert.deepEqual([within.same, within.outward, within.far], [1, 1, 0]);
    assert.equal(within.bits[1], nearest(5e-5));
    assert.deepEqual([beyond.same, beyond.outward, beyond.far], [1, 0, 1]);
    assert.equal(beyond.bits[1], nearest(1e-7));
  }
  // an infinity or a NaN of the GPU's is farther; an exact value one float16 either side is a neighbour, two is not
  {
    const x = Float64Array.of(8, 2.5), exact = nearest(2.5);
    for (const wrong of [0x7c00, 0x7e00, 0xfc00]) assert.equal(heldHalves(x, Uint16Array.of(nearest(8), wrong)).far, 1, `float16 ${wrong.toString(16)}`);
    assert.equal(heldHalves(x, Uint16Array.of(nearest(8), exact + 1)).outward, 1);
    assert.equal(heldHalves(x, Uint16Array.of(nearest(8), exact - 1)).inward, 1);
    assert.equal(heldHalves(x, Uint16Array.of(nearest(8), exact + 2)).far, 1);
  }
  // the words
  assert.equal(halvesSaid([{ same: 100, inward: 0, outward: 0, far: 0 }, { same: 92, inward: 0, outward: 0, far: 0 }]), "K and V 192 to the nearest float16");
  assert.equal(halvesSaid([{ same: 50, inward: 50, outward: 0, far: 0 }, { same: 40, inward: 52, outward: 0, far: 0 }]),
    "K and V 90 to the nearest float16, 102 toward zero, 0 away from it, 0 farther");
  assert.equal(farthest([1, 2.5], [1, 2]), 0.25);
  assert.ok(Number.isNaN(farthest([NaN, 1], [1, 2])), "a NaN is not within any line");
  // the engine's own check of a form (public/gpu.js's checkTokens: the first layer's keys and values at position 1) takes
  // the same rule on floats (heldFloats): the device's float16 where it is a neighbour, the nearest where not. Toward zero
  // the stream of a float form on llm-jp-3 150M came to 3.2 times its line off the nearest's (CI, Dawn with every
  // conversion cut toward zero), and the form was refused
  {
    const { heldFloats } = await import("../public/gpu.js");
    const asked = (how) => Float64Array.from(values, (x) => toFloat(how(x)));
    for (const how of [nearest, inwards, outwards]) assert.deepEqual([...heldFloats(values, asked(how))], [...asked(how)], `${how.name}: the device's own`);
    const largest = values.reduce((most, x) => Math.max(most, Math.abs(x)), 0), top = [...values.keys()].filter((i) => Math.abs(values[i]) >= largest / 4);
    const far = Float64Array.from(values, (x) => toFloat(Math.abs(x) >= largest / 4 ? inwards(x) - 1 : inwards(x)));
    const held = heldFloats(values, far);
    assert.ok(top.filter((i) => held[i] !== far[i]).length >= 0.85 * top.length, "one more float16 off: not taken, for all but the few within the slack");
    for (const i of top) if (held[i] !== far[i]) assert.equal(held[i], toFloat(nearest(values[i])), `element ${i} (${values[i]}) is the nearest's`);
    // the engine's rule and the benchmark's are one: the same values taken, on the same cases (a change to one alone fails here)
    const bitsOf = (how) => Uint16Array.from(values, (x) => how(x));
    const oneMoreBits = Uint16Array.from(values, (x) => (Math.abs(x) >= largest / 4 ? inwards(x) - 1 : inwards(x)));
    for (const got of [bitsOf(nearest), bitsOf(inwards), bitsOf(outwards), oneMoreBits]) {
      assert.deepEqual([...heldFloats(values, Float64Array.from(got, toFloat))], [...Float64Array.from(heldHalves(values, got).bits, toFloat)]);
    }
    assert.deepEqual([...heldFloats(Float64Array.of(8, 2.5), Float64Array.of(8, NaN))], [8, 2.5], "a NaN: the nearest");
    assert.deepEqual([...heldFloats(Float64Array.of(8, 2.5), Float64Array.of(8, Infinity))], [8, 2.5], "an infinity: the nearest");
  }
}
// T225's review: tests/rounding.mjs's rewriting of a shader's text for a device that rounds a float32 to a float16 another
// way (the arithmetic is held by tests/rounding-check.mjs on Dawn: every key and value must come out as the rounding asked)
{
  const { rounded } = await import("./rounding.mjs");
  const keys = "keys[row] = pack2x16float(key);\nlet back = unpack2x16float(h);";
  assert.equal(rounded(keys, ""), keys);
  assert.equal(rounded(keys, "nearest"), keys);
  const zero = rounded(keys, "toward-zero");
  assert.ok(zero.startsWith("keys[row] = pack2x16float_rounded(key);\nlet back = unpack2x16float(h);"), "pack2x16float( is the call changed, not unpack2x16float(");
  assert.ok(zero.includes("fn pack2x16float_rounded(") && zero.includes("fn rounded16(") && !zero.includes("fn toward16("), "its helpers after it");
  assert.ok(rounded(keys, "away").includes("fn toward16("), "away from zero is toward zero and one more");
  assert.equal(rounded("var a = 1;", "toward-zero"), "var a = 1;", "a shader that converts nothing is left as it is");
  const tiles = (type) => `alias shmem_t = ${type};\nshmem[at] = shmem_t(x);\nvar<workgroup> p: array<vec4<shmem_t>, 4>;`;
  assert.equal(rounded(tiles("f16"), "toward-zero"), tiles("f16"), "the tiles' conversions are left to the device unless asked");
  assert.ok(rounded(tiles("f16"), "everything").includes("shmem[at] = to_shmem(x);") && rounded(tiles("f16"), "everything").includes("return shmem_t(unpack2x16float(rounded16(x)).x);"));
  assert.ok(rounded(tiles("f32"), "everything").includes("fn to_shmem(x: f32) -> shmem_t { return x; }"), "in the float32 tiles it converts nothing");
  assert.ok(rounded(tiles("f16"), "everything").includes("array<vec4<shmem_t>, 4>"), "a type is not a conversion");
  assert.throws(() => rounded("", "sideways"), /unknown rounding/);
}
// T152: the status line's words (the owner's, 2026-09-27): both on the GPU, before either is timed, the answers alone on
// the CPU (faster here), and a reason for the answers (in the console alone)
{
  assert.equal(gpuLine(PROMPTS_GPU, "gpu"), "prompts and answers on WebGPU");
  assert.equal(gpuLine(PROMPTS_UNTIMED, "untimed"), "WebGPU where it is faster than the CPU");
  assert.equal(gpuLine(PROMPTS_GPU, "cpu"), "prompts on WebGPU, answers on the CPU (faster here)");
  assert.equal(gpuLine(PROMPTS_GPU, "why"), "prompts on WebGPU, answers on the CPU");
  assert.equal(gpuLine(PROMPTS_UNTIMED, "why"), "prompts on WebGPU where it is faster than the CPU, answers on the CPU");
  assert.equal(gpuLine(PROMPTS_CPU, "cpu"), "prompts and answers on the CPU (faster here than WebGPU)");
  assert.equal(gpuLine(PROMPTS_CPU, "gpu"), "prompts on the CPU (faster here than WebGPU), answers on WebGPU");
  assert.equal(gpuLine("prompts of 32 tokens and more on WebGPU", "gpu"), "prompts of 32 tokens and more on WebGPU, answers on WebGPU");
  assert.equal(gpuLine(PROMPTS_GPU, "untimed"), "prompts on WebGPU, answers on WebGPU where it is faster than the CPU");
  assert.equal(gpuLine("prompts on the CPU (no GPU adapter here)", null), "prompts on the CPU (no GPU adapter here)");
  assert.equal(gpuLine(undefined, null), undefined);
}
// T156: where a model goes (the owner's B, 2026-09-27): both where both fit half of the device (8: 8 GB, 4 GB for
// both), else the GPU alone where the model can be and fits (8: no limit, as the CPU alone has none), else the CPU
{
  const GB = 2 ** 30;
  // (the owner, 2026-09-27: both up to 6.5 GiB on a device that says 8, so that the list's models up to 2B stay as they
  // were and 3B and larger go on the GPU alone. The second review of T156, the list's configs with footprint() and
  // gpuBytes(), 4096 positions: sarashina2.2 1B 2.81 + 1.94 GB (4.42 GiB) and llm-jp-3.1 1.8B 3.543 + 2.910 GB (6.010
  // GiB, no GQA: its keys and values are 0.8 GB on either side; 11 MB past the first line of 6 GiB) both as before;
  // Qwen2.5 3B 4.10 + 3.63 GB (7.20 GiB; until T226 not eligible for its biases: the CPU with its layers past the
  // room left; with its steps on the GPU, the GPU alone); Llama 3.2 3B 8.83 GB the GPU alone)
  assert.equal(BOTH_ON_8, 6.5 * GB, "the line of both on a device that says 8");
  assert.deepEqual(weightsPlace({ cpu: 1.5 * GB, gpu: 1.4 * GB, deviceMemory: 8 }), { mode: "both", gpuRoom: 5 * GB }, "Llama 3.2 1B on 8: both");
  assert.equal(weightsPlace({ cpu: 2.812e9, gpuOnly: 0.98e9, gpu: 1.939e9, deviceMemory: 8, eligible: true }).mode, "both", "sarashina2.2 1B on 8: both");
  assert.equal(weightsPlace({ cpu: 3.543e9, gpuOnly: 1.72e9, gpu: 2.910e9, deviceMemory: 8, eligible: true }).mode, "both", "llm-jp-3.1 1.8B: 6.01 GiB, within 6.5");
  assert.equal(weightsPlace({ cpu: 3.25 * GB, gpuOnly: 0.6 * GB, gpu: 3.25 * GB, deviceMemory: 8, eligible: true }).mode, "both", "6.5 GiB: both");
  assert.equal(weightsPlace({ cpu: 3.3 * GB, gpuOnly: 0.6 * GB, gpu: 3.3 * GB, deviceMemory: 8, eligible: true }).mode, "gpu", "6.6 GiB: the GPU alone");
  const qwen3B = weightsPlace({ cpu: 4.103e9, gpu: 3.627e9, deviceMemory: 8 });
  assert.ok(qwen3B.mode === "cpu" && qwen3B.gpuRoom < 3.273e9, "Qwen2.5 3B on 8 where it is not eligible: the CPU, its layers (3.27 GB) past the room left");
  assert.equal(weightsPlace({ cpu: 4.103e9, gpuOnly: 1.2e9, gpu: 3.627e9, deviceMemory: 8, eligible: true }).mode, "gpu", "T226: Qwen2.5 3B on 8, eligible: the GPU alone");
  assert.equal(weightsPlace({ cpu: 4.9e9, gpuOnly: 1.3e9, gpu: 3.93e9, deviceMemory: 8, eligible: true }).mode, "gpu", "Llama 3.2 3B on 8: the GPU alone");
  assert.equal(weightsPlace({ cpu: 4.4 * GB, gpuOnly: 1.2 * GB, gpu: 4.1 * GB, deviceMemory: 8, eligible: true }).mode, "gpu", "3B on 8: the GPU alone");
  assert.equal(weightsPlace({ cpu: 9.2 * GB, gpuOnly: 1.5 * GB, gpu: 8 * GB, deviceMemory: 8, eligible: true }).mode, "gpu", "7B on 8: no limit");
  const notEligible = weightsPlace({ cpu: 4.4 * GB, gpu: 4.1 * GB, deviceMemory: 8 });
  assert.ok(notEligible.mode === "cpu" && Math.abs(notEligible.gpuRoom - 2.1 * GB) < 1, "not eligible: the CPU, 2.1 GiB for the GPU's layers");
  assert.equal(weightsPlace({ cpu: 1.5 * GB, gpuOnly: 0.7 * GB, gpu: 1.4 * GB, deviceMemory: 4, eligible: true }).mode, "cpu", "1B on 4 GB: 2.1 GB past 2");
  assert.equal(weightsPlace({ cpu: 1.5 * GB, gpuOnly: 0.3 * GB, gpu: 1.4 * GB, deviceMemory: 4, eligible: true }).mode, "gpu", "within 2 GB: the GPU alone");
  assert.equal(weightsPlace({ cpu: 0.1 * GB, gpuOnly: 0.05 * GB, gpu: 0.1 * GB, deviceMemory: 8, eligible: true, forced: true }).mode, "gpu", "?gpuTest=only");
  assert.equal(weightsPlace({ cpu: 0.1 * GB, gpu: 0.1 * GB, deviceMemory: 8, forced: true }).mode, "both", "forced, not eligible");
  // gpuBytes: Llama 3.2 1B (dim 2048, hidden 8192, 16 layers, 32 heads, 8 of keys and values, vocab 128256, 4096)
  const oneB = gpuBytes([2048, 8192, 16, 32, 8, 128256, 4096]);
  const want = 16 * (2 * 2048 * 2048 + 2 * 512 * 2048 + 3 * 8192 * 2048) * 1.125 + 16 * 2 * 2048 * 4 + 2 * 16 * 4096 * 512 * 2 +
    128256 * 2048 * 1.125 + 4096 * 64 * 4 + 3 * 128256 * 4;
  assert.ok(Math.abs(oneB - want) < 1, `gpuBytes ${oneB} against ${want}`);
  assert.ok(gpuBytes([64, 128, 2, 4, 2, 320, 256], { arch: "gpt2" }) < gpuBytes([64, 128, 2, 4, 2, 320, 256]), "no gate on GPT-2");
  // T232: ternary weights go up as they are, two bits a weight and a float32 scale a group of 128 (0.28125 bytes a
  // weight). Ternary Bonsai 1.7B (a Qwen3: dim 2048, hidden 6144, 28 layers, 16 heads of 128 on 8 of keys and values)
  const bonsai = [2048, 6144, 28, 16, 8, 151936, 4096], form = { qk_norm: true, head_dim: 128 };
  const ternary = 28 * (2 * 2048 * 2048 + 2 * 1024 * 2048 + 3 * 6144 * 2048) * 0.28125 + 28 * 2 * 2048 * 4 + 2 * 28 * 4096 * 1024 * 2 +
    151936 * 2048 * 0.28125 + 4096 * 128 * 4 + 3 * 151936 * 4;
  assert.ok(Math.abs(gpuBytes(bonsai, { ...form, dtype: "ternary" }) - ternary) < 1, `gpuBytes of ternary weights ${gpuBytes(bonsai, { ...form, dtype: "ternary" })} against ${ternary}`);
  assert.ok(gpuBytes(bonsai, { ...form, dtype: "ternary" }) < gpuBytes(bonsai, form) / 2, "the keys and values are the same, the weights a quarter");
  // where the three listed models go (their checkpoints' bytes; the worker's options; 4096 positions), by what the
  // device says it has: 8, both and measured (2.24, 3.83 and 6.01 GiB of 6.5); 4, the 1.7B and the 4B on the GPU alone
  // (0.91 and 1.64 GiB of 2) and the 8B on the CPU (2.74); a browser that does not say (4, and never on the GPU alone,
  // T205): the CPU, and not even the prompts on the GPU (the layers with their keys and values past the room left)
  const adapter = { fallback: false, packed: true, limits: { maxStorageBufferBindingSize: 128 * 2 ** 20, maxBufferSize: 256 * 2 ** 20 } };
  const listed = [["1.7B", bonsai, 484372508, form, "both", "gpu"], ["4B", [2560, 9728, 36, 32, 8, 151936, 4096], 1132048412, form, "both", "gpu"],
    ["8B", [4096, 12288, 36, 32, 8, -151936, 4096], 2304790556, { qk_norm: true }, "both", "cpu"]];
  for (const [name, header, size, shape, on8, on4] of listed) {
    const options = { ...shape, dtype: "ternary", int8: true, relaxed: true, halfKV: true, shared: true, outliers: 8, gpu: true };
    const [dim, hidden, layers, heads, kvHeads, signedVocab, seqLen] = header, headSize = shape.head_dim || dim / heads;
    const layerWeights = layers * (2 * heads * headSize * dim + 2 * kvHeads * headSize * dim + 3 * hidden * dim);
    const stored = size - (layerWeights + Math.abs(signedVocab) * dim * (signedVocab > 0 ? 1 : 2)) * 0.28125;
    const cpu = size + footprint(header, size, options), gpu = gpuBytes(header, { ...shape, dtype: "ternary" });
    const gpuOnly = stored + footprint(header, stored, { ...options, direct: true });
    assert.equal(gpuOnlyUnfit(header, "ternary", shape, adapter), null, `Ternary Bonsai ${name} may go on the GPU alone`);
    assert.ok(stored > 0 && stored < 2e6, `Ternary Bonsai ${name}: ${stored} bytes besides the matrices`);
    assert.equal(weightsPlace({ cpu, gpuOnly, gpu, deviceMemory: 8, eligible: true }).mode, on8, `Ternary Bonsai ${name} on 8: ${((cpu + gpu) / GB).toFixed(2)} GiB`);
    assert.equal(weightsPlace({ cpu, gpuOnly, gpu, deviceMemory: 4, eligible: true }).mode, on4, `Ternary Bonsai ${name} on 4: ${((gpuOnly + gpu) / GB).toFixed(2)} GiB on the GPU alone`);
    const unsaid = weightsPlace({ cpu, gpu, deviceMemory: 4 });
    const onGpu = layerWeights * 0.28125 + 2 * layers * seqLen * kvHeads * headSize * 2;
    assert.ok(unsaid.mode === "cpu" && onGpu > unsaid.gpuRoom, `Ternary Bonsai ${name} where the browser does not say: the CPU, the layers past the room left`);
  }
}
// T156: why a model cannot go on the GPU alone
{
  const adapter = { fallback: false, limits: { maxStorageBufferBindingSize: 128 * 2 ** 20, maxBufferSize: 256 * 2 ** 20, minStorageBufferOffsetAlignment: 256 } };
  const oneB = [2048, 8192, 16, 32, 8, 128256, 4096];
  assert.equal(gpuOnlyUnfit(oneB, "int8", {}, adapter), null);
  assert.match(gpuOnlyUnfit(oneB, "int8", {}, undefined), /no GPU adapter/);
  assert.match(gpuOnlyUnfit(oneB, "int8", {}, { ...adapter, fallback: true }), /fallback/);
  assert.equal(gpuOnlyUnfit(oneB, "int8", {}, { ...adapter, fallback: true }, { fallback: true }), null, "the tests' leave");
  // T226: Qwen2 (biases) and Qwen3 (norms of the heads, heads of another size) as well: Qwen2.5 3B's and Qwen3 4B's
  // shapes (the owner's NVIDIA PC: "Qwen3 4B ですら GPU 使われないんだが？"); not GPT-2 or GPT-NeoX
  assert.equal(gpuOnlyUnfit(oneB, "int8", { bias: true }, adapter), null);
  assert.equal(gpuOnlyUnfit([2048, 11008, 36, 16, 2, 151936, 4096], "int8", { bias: true }, adapter), null, "Qwen2.5 3B");
  assert.equal(gpuOnlyUnfit([2560, 9728, 36, 32, 8, 151936, 4096], "int8", { qk_norm: true, head_dim: 128 }, adapter), null, "Qwen3 4B");
  assert.equal(gpuOnlyUnfit([4096, 12288, 36, 32, 8, 151936, 4096], "int8", { qk_norm: true, head_dim: 128 }, adapter), null, "Qwen3 8B");
  assert.match(gpuOnlyUnfit(oneB, "int8", { arch: "gpt2" }, adapter), /GPT-2 and GPT-NeoX/);
  // T237: a model in a rotated basis, whose inputs the GPU does not turn
  assert.match(gpuOnlyUnfit(oneB, "int8", { rotated: { block: 1024, signs: {} } }, adapter), /rotated basis/);
  assert.match(gpuOnlyUnfit(oneB, "int8", { arch: "neox" }, adapter), /GPT-2 and GPT-NeoX/);
  assert.match(gpuOnlyUnfit(oneB, "int8", { arch: "qwen35" }, adapter), /Qwen3\.5/, "T229: a hybrid model is not placed on the GPU alone either");
  assert.match(gpuOnlyUnfit(oneB, "int6", {}, adapter), /int6/);
  // stories15M (dim 288, 6 heads): k starts at 288 × 288 weights, no multiple of 2048 (T152)
  assert.match(gpuOnlyUnfit([288, 768, 6, 6, 6, 32000, 256], "int8", {}, adapter), /would not start/);
  // T220: by the device's 256, not by what the adapter could give: stories15M's k starts on 64 (82944 values, 10368
  // bytes of scales), and gpu.js's device, which asks for no alignment, binds on 256 and would refuse it
  const loose = { ...adapter, limits: { ...adapter.limits, minStorageBufferOffsetAlignment: 64 } };
  assert.match(gpuOnlyUnfit([288, 768, 6, 6, 6, 32000, 256], "int8", {}, loose), /would not start/);
  assert.equal(gpuOnlyUnfit(oneB, "int8", {}, loose), null);
  // Qwen2.5 7B's shape (no biases here): gate and up are 135.8 MB (T152's (d)), past 128 MiB
  assert.match(gpuOnlyUnfit([3584, 18944, 28, 28, 4, 152064, 4096], "int8", {}, adapter), /past a buffer/);
  // T232: ternary weights too, where the browser's WGSL has the packed int8 dot (the worker's adapter.packed); a
  // quarter of the bytes, so the same shape's gate and up are within a buffer, and a matrix starts where the device
  // binds where its weights are a multiple of 8192 (a byte of scales to 32 weights)
  const packed = { ...adapter, packed: true }, bonsai = { qk_norm: true, head_dim: 128 };
  assert.equal(gpuOnlyUnfit([2048, 6144, 28, 16, 8, 151936, 4096], "ternary", bonsai, packed), null, "Ternary Bonsai 1.7B");
  assert.match(gpuOnlyUnfit([2048, 6144, 28, 16, 8, 151936, 4096], "ternary", bonsai, adapter), /packed int8 dot/, "no packed int8 dot: the CPU");
  assert.equal(gpuOnlyUnfit([3584, 18944, 28, 28, 4, 152064, 4096], "ternary", {}, packed), null, "a quarter of the bytes: within a buffer");
  // (q of one head of 32 in a dim of 128: 4096 weights, 1024 bytes of codes and 128 of scales, where k would start)
  assert.match(gpuOnlyUnfit([128, 384, 2, 1, 1, 320, 256], "ternary", { head_dim: 32 }, packed), /would not start/);
  assert.equal(gpuOnlyUnfit([128, 384, 2, 8, 4, 320, 256], "ternary", { qk_norm: true, head_dim: 32 }, packed), null, "gpu-check's made-up ternary model");
  // (the review) and the untied one: a dim of 256 (two groups of 128 a row), a hidden size of 1152 (a second pass of a
  // token's matrix), its classifier apart (a negative vocabulary)
  assert.equal(gpuOnlyUnfit([256, 1152, 3, 8, 4, -320, 256], "ternary", { qk_norm: true, head_dim: 32 }, packed), null, "gpu-check's made-up untied ternary model");
  // (the review of T237) with the refusal of ternary weights gone, the others still stand, each with its own reason:
  // the 27B is ternary, in a rotated basis, with linear-attention layers (a Qwen3.5)
  assert.match(gpuOnlyUnfit([5120, 17408, 64, 24, 4, 248320, 4096], "ternary", { arch: "qwen35", head_dim: 256, rotated: { block: 1024, signs: {} } }, packed), /Qwen3\.5/);
  assert.match(gpuOnlyUnfit([2048, 6144, 28, 16, 8, 151936, 4096], "ternary", { ...bonsai, rotated: { block: 1024, signs: {} } }, packed), /rotated basis/);
}
// T156: the checkpoint's stretches: the layers' matrices to the GPU's worker, the rest packed into memory, whatever the
// stretches it comes in; each byte posted once, and the writer waits on flow
{
  // (T232, ternary: the matrices two bits a weight, a scale a group of 128)
  const tensor = (offset, shape, int8, ternary = false) => ({ kind: !int8 ? "f32" : ternary ? "ternary" : "int8", offset, shape, group: !int8 ? 0 : ternary ? 128 : 32,
    scales: int8 ? offset + shape.reduce((a, b) => a * b, 1) / (ternary ? 4 : 1) : 0 });
  // a made-up Llama: 2 layers, dim 64, hidden 96, the tensors in file order
  let end = 28;
  const tensors = {};
  for (const [name, shape, int8] of [["token_embedding_table", [100, 64], true], ["rms_att_weight", [2, 64], false], ["wq", [2, 64, 64], true],
    ["wk", [2, 32, 64], true], ["wv", [2, 32, 64], true], ["wo", [2, 64, 64], true], ["rms_ffn_weight", [2, 64], false], ["w1", [2, 96, 64], true],
    ["w2", [2, 64, 96], true], ["w3", [2, 96, 64], true], ["rms_final_weight", [64], false]]) {
    tensors[name] = tensor(end, shape, int8);
    const count = shape.reduce((a, b) => a * b, 1);
    end += int8 ? count + count / 8 : count * 4;
  }
  const size = end, holes = gpuHoles(tensors), place = placer(holes);
  // (T210: the embedding too, which is the classifier here: one stretch)
  assert.equal(holes.length, 8);
  assert.equal(gpuHoles({ ...tensors, wcls: tensors.token_embedding_table }).length, 8, "a shared classifier is one stretch");
  assert.equal(place(tensors.rms_att_weight.offset), 28, "the first norm goes where the embedding began");
  assert.equal(place(tensors.rms_ffn_weight.offset), 28 + 2 * 64 * 4, "the norm after wo goes after the first norm");
  const checkpoint = Uint8Array.from({ length: size }, (_, i) => (i * 7 + 3) & 255);
  const memory = new WebAssembly.Memory({ initial: 2, maximum: 4, shared: true }), base = 64;
  let flow;
  const posted = new Uint8Array(size), counts = new Uint8Array(size);
  const worker = {
    postMessage({ offset, bytes }) {
      posted.set(bytes, offset);
      for (let i = 0; i < bytes.length; i++) counts[offset + i]++;
      Atomics.add(flow, 0, BigInt(bytes.length));
    },
  };
  const weights = gpuOnlyWeights({ memory, base, size, tensors, worker });
  flow = new BigInt64Array(weights.flow);
  const steps = [1, 3, 4093, 17, 65536];
  for (let at = 0, k = 0; at < size; at += steps[k++ % steps.length]) weights.write(at, checkpoint.subarray(at, Math.min(size, at + steps[k % steps.length])));
  await weights.room();
  await weights.drained();
  const inHole = (i) => holes.some(([a, b]) => i >= a && i < b);
  const stored = new Uint8Array(memory.buffer, base, weights.stored);
  for (let i = 0; i < size; i++) {
    if (inHole(i)) assert.ok(counts[i] === 1 && posted[i] === checkpoint[i], `byte ${i} posted once`);
    else assert.ok(counts[i] === 0 && stored[place(i)] === checkpoint[i], `byte ${i} in memory at ${place(i)}`);
  }
  assert.equal(weights.stored, size - holes.reduce((sum, [a, b]) => sum + b - a, 0));
  // what gpu.js opens with: each layer's values and scales where they start in the checkpoint
  const plan = gpuOnlyPlan([64, 96, 2, 4, 2, 100, 32], tensors);
  assert.deepEqual(plan.matrices.w2.layers[1], [tensors.w2.offset + 64 * 96, tensors.w2.scales + (64 * 96 / 32) * 4]);
  // T210: and the tables, where they start in the checkpoint (the embedding null where the classifier is it)
  const e = tensors.token_embedding_table;
  assert.deepEqual(plan.tables, { classifier: { rows: 100, n: 64, ternary: false, at: [e.offset, e.scales] }, embedding: null });
  const apart = gpuOnlyPlan([64, 96, 2, 4, 2, 100, 32], { ...tensors, wcls: tensor(end, [100, 64], true) });
  assert.deepEqual(apart.tables, { classifier: { rows: 100, n: 64, ternary: false, at: [end, end + 6400] }, embedding: { rows: 100, n: 64, ternary: false, at: [e.offset, e.scales] } });
  assert.equal(gpuHoles({ ...tensors, wcls: tensor(end, [100, 64], true) }).length, 9, "a classifier of its own is a stretch of its own");
  // T232: a ternary model's: a layer of a matrix is a quarter of a byte a weight on, and its scales one a group of 128
  let at = 28;
  const codes = {};
  for (const [name, shape, matrix] of [["token_embedding_table", [100, 128], true], ["rms_att_weight", [2, 128], false], ["wq", [2, 128, 128], true],
    ["wk", [2, 64, 128], true], ["wv", [2, 64, 128], true], ["wo", [2, 128, 128], true], ["rms_ffn_weight", [2, 128], false], ["w1", [2, 256, 128], true],
    ["w2", [2, 128, 256], true], ["w3", [2, 256, 128], true], ["rms_final_weight", [128], false]]) {
    codes[name] = tensor(at, shape, matrix, true);
    const count = shape.reduce((a, b) => a * b, 1);
    at += matrix ? count / 4 + count / 32 : count * 4;
  }
  const ternary = gpuOnlyPlan([128, 256, 2, 4, 2, 100, 32], codes);
  assert.deepEqual(ternary.matrices.w2, { rows: 128, n: 256, ternary: true, layers: [[codes.w2.offset, codes.w2.scales],
    [codes.w2.offset + 128 * 256 / 4, codes.w2.scales + (128 * 256 / 128) * 4]] });
  assert.deepEqual(ternary.tables, { classifier: { rows: 100, n: 128, ternary: true, at: [codes.token_embedding_table.offset, codes.token_embedding_table.scales] }, embedding: null });
  // (its stretches: every matrix's codes and then its scales, to where the next tensor begins)
  assert.deepEqual(gpuHoles(codes).find(([start]) => start === codes.w1.offset), [codes.w1.offset, codes.w2.offset]);
  // (the review of T232: a ternary classifier apart from the embedding, as the 8B's: two tables, each its own codes and
  // scales, and a stretch of its own, its codes and then its scales; no gpu-check model was untied before)
  const wcls = tensor(at, [100, 128], true, true);
  const untied = gpuOnlyPlan([128, 256, 2, 4, 2, -100, 32], { ...codes, wcls });
  assert.deepEqual(untied.tables, { classifier: { rows: 100, n: 128, ternary: true, at: [wcls.offset, wcls.scales] },
    embedding: { rows: 100, n: 128, ternary: true, at: [codes.token_embedding_table.offset, codes.token_embedding_table.scales] } });
  assert.equal(gpuHoles({ ...codes, wcls }).length, gpuHoles(codes).length + 1, "a ternary classifier of its own is a stretch of its own");
  assert.deepEqual(gpuHoles({ ...codes, wcls }).at(-1), [wcls.offset, wcls.scales + ((100 * 128) / 128) * 4], "its codes, then its scales");
}

// T156 (the owner, 2026-09-27): a model on the GPU alone weighed on the prompts too, by the page's use, and the verdict
// kept for the device and /benchmark/'s CPU reading
{
  // Llama 3.2 3B (dim 3072, hidden 8192, 28 layers, 24 heads, 8 of keys and values): 2.8 G multiply-adds a token
  const header = [3072, 8192, 28, 24, 8, 128256, 4096], weights = layerWeightsOf(header);
  assert.equal(weights, 28 * (2 * 3072 * 3072 + 2 * 1024 * 3072 + 3 * 8192 * 3072));
  const size = 3.6e9, cpu = { GBps: 28.7, promptGMACs: 40 };
  // the CPU's step 125 ms (3.6 GB at 28.7 GB/s), its prompt's token 70.6 ms (2.8 G at 40 G MAC/s)
  const slowStep = { stepMs: 160, promptMs: 14 };
  // the answers alone: the CPU faster (125 < 0.95 × 160)
  assert.equal(aloneVerdict({ size, layerWeights: weights, cpu: { GBps: 28.7 }, gpu: slowStep }).cpuFaster, true, "the steps alone: the CPU");
  // with the prompts, as many tokens: 125 + 70.6 against 160 + 14: the CPU still (195.6 < 165.3? no): the GPU
  const even = aloneVerdict({ size, layerWeights: weights, cpu, gpu: slowStep, usage: USAGE_UNKNOWN });
  assert.equal(even.cpuFaster, false, `as many prompt tokens as written: the GPU (${even.cpu} against ${even.gpu})`);
  // mostly writing (1 prompt token to 10 written): the CPU
  assert.equal(aloneVerdict({ size, layerWeights: weights, cpu, gpu: slowStep, usage: { prompt: 1, written: 10 } }).cpuFaster, true, "writing: the CPU");
  // nothing known of the CPU: the GPU stays
  assert.equal(aloneVerdict({ size, layerWeights: weights, cpu: {}, gpu: slowStep }).cpuFaster, false, "no /benchmark/: the GPU");
  // T232: the CPU's token of a ternary model is read as the same weights of int8 (four times the checkpoint's bytes:
  // its kernel is bound by its arithmetic, at about the int8 kernel's weights a second, T231)
  assert.equal(cpuReadBytes(1e9, "ternary"), 4e9);
  assert.equal(cpuReadBytes(1e9, "int8"), 1e9);
  // the use kept: each generation's added to four fifths of what was there
  assert.deepEqual(usedAfter(usedAfter(undefined, 100, 50), 20, 200), { prompt: 100 * 0.8 + 20, written: 50 * 0.8 + 200 });
  // the key: the adapter as the device made of it (the same features and limits), another description another key;
  // the verdict holds for the same key and the same reading of /benchmark/'s CPU, and for nothing else
  const adapter = { info: { vendor: "arm", architecture: "valhall", device: "", description: "Mali-G615" },
    features: new Set(["shader-f16", "subgroups"]), limits: { maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 } };
  const device = { features: new Set(adapter.features), limits: { ...adapter.limits } };
  const key = deviceKey(adapter);
  assert.equal(deviceKey(adapter, device), key, "the worker's key is gpu.js's");
  // T232: a model of ternary weights has a key of its own (its shaders' text after the others'), the worker's and
  // gpu.js's the same
  assert.notEqual(deviceKey(adapter, device, true), key, "a ternary model's key is not the others'");
  assert.equal(deviceKey(adapter, adapter, true), deviceKey(adapter, device, true), "the worker's ternary key is gpu.js's");
  assert.notEqual(deviceKey({ ...adapter, info: { ...adapter.info, description: "Mali-G715" } }), key, "another GPU");
  assert.notEqual(deviceKey({ ...adapter, features: new Set(["subgroups"]) }), key, "other shaders");
  const alone = { key, cpu };
  assert.equal(aloneHolds(alone, key, { ...cpu, threads: 4 }), true);
  assert.equal(aloneHolds(alone, key + "x", cpu), false, "another device, browser or shaders");
  assert.equal(aloneHolds(alone, key, { ...cpu, GBps: 30.1 }), false, "/benchmark/'s CPU measured again");
  assert.equal(aloneHolds(alone, key, undefined), false, "no /benchmark/ now");
  assert.equal(aloneHolds(undefined, key, cpu), false);
}
// T224's review: shaders.js's tokenAttentionOff (how far a token's attention from the GPU is from JavaScript's, the check
// of the engine and of /benchmark/) keeps a NaN. `if (!(off <= worst)) worst = off` let the next value take its place,
// and a NaN in one head, or in the first values of the last, was a pass (a reduce that wrote NaN for head 0 only: 4e-8)
{
  const dims = { heads: 4, kvHeads: 2, size: 8, positions: 40 };
  const data = tokenAttentionData({ ...dims, steep: [3] });
  // the answer in float64, apart from the function's: the weights of a head over its positions, then their values
  const half = (h) => (h & 0x8000 ? -1 : 1) * ((h >> 10) & 31 ? 2 ** (((h >> 10) & 31) - 15) * (1 + (h & 1023) / 1024) : 2 ** -14 * ((h & 1023) / 1024));
  const exact = new Float32Array(dims.heads * dims.size), kvDim = dims.kvHeads * dims.size;
  for (let h = 0; h < dims.heads; h++) {
    const kv = Math.floor(h / (dims.heads / dims.kvHeads)) * dims.size;
    const scores = Array.from({ length: dims.positions }, (_, p) => {
      let sum = 0;
      for (let d = 0; d < dims.size; d++) sum += data.q[h * dims.size + d] * half(data.keys[p * kvDim + kv + d]);
      return sum / Math.sqrt(dims.size);
    });
    const top = Math.max(...scores), weights = scores.map((s) => Math.exp(s - top)), total = weights.reduce((a, b) => a + b, 0);
    for (let d = 0; d < dims.size; d++) {
      exact[h * dims.size + d] = weights.reduce((sum, w, p) => sum + w * half(data.values[p * kvDim + kv + d]), 0) / total;
    }
  }
  assert.ok(tokenAttentionOff(exact, data, dims) < 1e-6, "the right answer is off by rounding to float32 only");
  const wrong = Float32Array.from(exact);
  wrong[5] += 0.5;
  assert.ok(tokenAttentionOff(wrong, data, dims) > 1e-2, "a value off by 0.5 is seen");
  const withNaN = (from, to) => Float32Array.from(exact).fill(NaN, from, to);
  for (const [where, got] of [["the first value", withNaN(0, 1)], ["all of head 0", withNaN(0, 8)], ["head 1's middle", withNaN(10, 12)],
    ["the first values of the last head", withNaN(24, 28)], ["the last value", withNaN(31, 32)]]) {
    assert.ok(Number.isNaN(tokenAttentionOff(got, data, dims)), `a NaN in ${where} stays`);
  }
}
// T232: the check of a tiled shader against JavaScript (shaders.js's tiledOff, gpu.js's checkForm) on ternary weights:
// their values are the codes less one, a scale covers 128 of them (four of the vector's groups of 32); and a product
// that is no number is wrong (a NaN is neither over a line nor under it: the check passed every form, the int8 ones
// too, while gpu.js handed it a group that was no number; CI's mutants of the ternary tiles found it)
{
  const rows = 9, n = 384, tokens = 3, group = 128;
  let seed = 232;
  const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  const stored = new Uint8Array((rows * n) / 4).map(() => (random() * 256) | 0), w = ternaryValues(stored);
  assert.deepEqual(Array.from(ternaryValues(new Uint8Array([0b11100100, 0b00000110]))), [-1, 0, 1, 2, 1, 0, -1, -1], "a byte's four codes, the lowest first, each less one");
  const s = new Float32Array((rows * n) / group).map(() => random() * 0.01), x = new Float32Array(tokens * n).map(() => random() * 2 - 1);
  const xq = new Int8Array(tokens * n), xs = new Float32Array(tokens * (n / 32));
  for (let t = 0; t < tokens; t++) {
    const q = quantizedLikeCpu(x.subarray(t * n, (t + 1) * n));
    xq.set(q.xq, t * n);
    xs.set(q.xs, t * (n / 32));
  }
  // the product twice (as the check adds it), by sign times the weights and the scale at scaleAt(row, group of 32)
  const product = (sign, scaleAt) => Float32Array.from({ length: tokens * rows }, (_, at) => {
    const t = Math.floor(at / rows), r = at % rows;
    let sum = 0;
    for (let g = 0; g < n / 32; g++) {
      let part = 0;
      for (let i = g * 32; i < (g + 1) * 32; i++) part += sign * w[r * n + i] * xq[t * n + i];
      sum += part * s[scaleAt(r, g)] * xs[t * (n / 32) + g];
    }
    return 2 * sum;
  });
  const right = (r, g) => r * (n / group) + Math.floor(g / 4);
  const off = (got, more = {}) => tiledOff({ w, s, x, got, xq, xs, rows, n, tokens, xStride: n, yStride: rows, half: false, group, ...more }).wrong;
  assert.equal(off(product(1, right)), null, "the right products");
  assert.match(off(product(-1, right)), /products/, "the codes' signs swapped");
  assert.match(off(product(1, (r, g) => r * (n / group) + Math.min(Math.floor(g / 4) + 1, 2))), /products/, "the scale of the next group of 128");
  assert.match(off(product(1, right).fill(NaN, 5, 6)), /products/, "one product that is no number");
  assert.match(off(product(1, right), { group: () => 128 }), /products/, "a group that is no number makes every product NaN: wrong, not right");
}
// T232's review: ternaryMatVec is fusedDp4aMatVec's text with the lines that read the weights and their scales changed (and
// ternary_packed after SDP8AI), a copy because deviceKey() hashed fusedDp4aMatVec's source (until T366: the key holds
// the text a maker makes now, and the two may be made one maker without moving an int8 model's key). A copy drifts: a line changed in
// one and not the other is a shader of one kind of weights that is not the other's, and CI's runs of both go on being ok.
// Every line but those that differ by design is the same in the two, for every output a layer writes
{
  const byDesign = [/var<storage, read> b: /, /var<storage, read> scales_b: /, /k_offset - covers 32 values of k in input_b/, /let block_idx = k_offset;/,
    /let K128 = /, /the scales of a row of the weights: one a group of 128/, /let own_scale_b = /, /let own_b = /, /let own_b1 = /, /let up_scale_b = /,
    /\+= SDP8AI\(own_a, (ternary_packed\()?b\[up_offset/];
  const lines = (text) => text.replace(/\/\/ T232: a word of 16 ternary codes[\s\S]*?\n}\n/, "").split("\n");
  const kept = (text) => lines(text).filter((line) => !byDesign.some((re) => re.test(line)));
  const different = (text) => lines(text).length - kept(text).length;
  // (what differs by design: 7 lines of the int8 text, 8 of the ternary one (the scales' K128 and its comment are two lines
  // for the int8's block_idx one), and where SwiGLU reads up's rows 2 more in each)
  for (const [output, int8, ternary] of [["write", 7, 8], ["rope", 7, 8], ["add", 7, 8], ["swiglu", 9, 10]]) {
    assert.equal(different(fusedDp4aMatVec({ output })), int8, `${output}: the lines of fusedDp4aMatVec that read the weights`);
    assert.equal(different(ternaryMatVec({ output })), ternary, `${output}: the lines of ternaryMatVec that read the weights`);
    assert.deepEqual(kept(ternaryMatVec({ output })), kept(fusedDp4aMatVec({ output })),
      `ternaryMatVec(${output}) drifted from fusedDp4aMatVec(${output}): a line changed in one is to be changed in the other (or the two made one)`);
  }
  // and what is not by design is seen: one other line changed in the copy is not the same text
  const changed = ternaryMatVec({ output: "add" }).replace("var output_value = f32(0);", "var output_value = f32(1);");
  assert.notDeepEqual(kept(changed), kept(fusedDp4aMatVec({ output: "add" })), "a changed line is a difference");
}
// T232's review: the outlier channels' two shaders take as many as OUTLIERS_MOST, and Python's OUTLIER_CHANNELS is that
// many (a ninth channel past the Outliers' room would throw as the plan is made: the GPU then lost for a model that
// has them): one number in two languages
{
  const python = fs.readFileSync(new URL("../public/engine/checkpoint.py", import.meta.url), "utf8");
  assert.equal(Number(/^OUTLIER_CHANNELS = (\d+)$/m.exec(python)?.[1]), OUTLIERS_MOST, "llama2_numpy.OUTLIER_CHANNELS is shaders.js's OUTLIERS_MOST");
  const words = outliersOf([5, 70, 130, 255, 256, 300, 2047, 3], 151936, 2048);
  assert.deepEqual(Array.from(words), [8, 151936, 2048, 0, 5, 70, 130, 255, 256, 300, 2047, 3], "count, rows, n, a word of nothing, then the channels: 48 bytes of the uniform");
  assert.equal(outliersOf([1, 2]).length, 4 + OUTLIERS_MOST);
  assert.throws(() => outliersOf(Array.from({ length: OUTLIERS_MOST + 1 }, (_, i) => i)), RangeError, "a ninth channel has no room");
}
// T232's review: the check of a tiled shader sees a quantized scale that is no number and a form that writes nothing but NaN
{
  const rows = 4, n = 256, tokens = 2, group = 128, s = new Float32Array((rows * n) / group).fill(0.01), w = new Int8Array(rows * n).map((_, i) => (i % 3) - 1);
  const x = new Float32Array(tokens * n).map((_, i) => Math.sin(i)), xq = new Int8Array(tokens * n), xs = new Float32Array(tokens * (n / 32));
  for (let t = 0; t < tokens; t++) {
    const q = quantizedLikeCpu(x.subarray(t * n, (t + 1) * n));
    xq.set(q.xq, t * n);
    xs.set(q.xs, t * (n / 32));
  }
  const exact = Float32Array.from({ length: tokens * rows }, (_, at) => {
    const t = Math.floor(at / rows), r = at % rows;
    let sum = 0;
    for (let g = 0; g < n / 32; g++) {
      let part = 0;
      for (let i = g * 32; i < (g + 1) * 32; i++) part += w[r * n + i] * xq[t * n + i];
      sum += part * s[r * (n / group) + Math.floor(g / 4)] * xs[t * (n / 32) + g];
    }
    return 2 * sum;
  });
  const off = (got, more = {}) => tiledOff({ w, s, x, got, xq, xs, rows, n, tokens, xStride: n, yStride: rows, half: false, group, ...more }).wrong;
  assert.equal(off(exact), null);
  assert.match(off(new Float32Array(exact.length).fill(NaN)), /products/, "a form that writes only NaN is wrong, not right");
  assert.match(off(exact, { xs: Float32Array.from(xs, (v, i) => (i === 3 ? NaN : v)) }), /products/, "a quantized scale that is a NaN is wrong");
  assert.match(off(exact.map((v, i) => (i === 2 ? Infinity : v))), /products/, "an infinity is wrong");
  // the f16 forms' line takes the same road
  assert.match(off(new Float32Array(exact.length).fill(NaN), { half: true }), /products/, "an f16 form that writes only NaN is wrong");
}
console.log("ok");
