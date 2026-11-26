// tests/bench-nan-check.ts (T263): /benchmark/'s GPU step "nan" (src/runtime/benchmark/gpu/nan.ts) and its reading
// (src/bench/nan.ts) on made-up devices, in Node:
//
//   node tests/bench-nan-check.ts
//
// The device is tests/fakes/webgpu.ts's, whose dispatches compute nothing: here its answer() writes what a device of a
// kind would make of each case of shaders.ts's NAN_CASES (a model of each case in JavaScript below, by its name). The
// kinds: one that carries a NaN and an infinity as IEEE 754 has it; one that makes every NaN 0 (as it loads one and as
// it makes one); one whose max() and min() drop a NaN for the other operand (and clamp() for its bound); one whose
// pack2x16float() clamps to float16's largest and makes a NaN 0; one that folds x != x (lavapipe's: T219); and one
// without shader-f16. Each is a process of its own (the worker's device is made once a process). The reading must
// list as broken the marks of the engine each device breaks, and those alone, and say no warning for what no mark
// stands on (the max() of a device that drops a NaN). Simpler than a GPU: no WGSL runs, so that a case's WGSL that
// does not do what its name says is CI's to see (bench.yml, gpu-prompt.yml's Dawn); the names of the cases, the
// buffers the step binds, its reading back and the reading are this check's.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { runtimeUrl } from "./tree.ts";
import { standIn } from "./fakes/webgpu.ts";

const DEVICES = ["ieee", "nan to zero", "max drops NaN", "pack clamps", "folds x != x", "no f16"];
if (process.argv[2] !== "device") {
  for (const one of DEVICES) {
    const ran = spawnSync(process.execPath, [import.meta.filename, "device", one], { encoding: "utf8", timeout: 120000 });
    assert.equal(ran.status, 0, `${one}:\n${ran.stdout}${ran.stderr}`);
    process.stdout.write(ran.stdout);
  }
  console.log(`bench-nan-check: the step "nan" and its reading on ${DEVICES.length} made-up devices: ok`);
  process.exit(0);
}
const kind = process.argv[3];

const { halvesOf, floatOfHalf, NAN_CASES } = await import(runtimeUrl("shaders.js"));
const { NAN_READINGS, nanBroken, nanTable, nanSummary, warnings, gpuMarkdown, cellsOf } = await import("../src/bench.ts");

// ---- the devices: how one loads a float, makes one from what it computed, takes max(), min(), clamp(), packs to
// float16, and compares x != x
const same = (x) => x;
const zero = (x) => (Number.isNaN(x) ? 0 : x);
const ieeeMax = (x, y) => (Number.isNaN(x) || Number.isNaN(y) ? NaN : Math.max(x, y));
const ieeeMin = (x, y) => (Number.isNaN(x) || Number.isNaN(y) ? NaN : Math.min(x, y));
const dropMax = (x, y) => (Number.isNaN(x) ? y : Number.isNaN(y) ? x : Math.max(x, y));
const dropMin = (x, y) => (Number.isNaN(x) ? y : Number.isNaN(y) ? x : Math.min(x, y));
const half = (x) => { const out = new Uint16Array(1); halvesOf([x], out); return out[0]; };
const clampedHalf = (x) => (Number.isNaN(x) ? 0 : half(Math.max(-65504, Math.min(65504, x))));
const IEEE = { load: same, made: same, max: ieeeMax, min: ieeeMin, clamp: (x, lo, hi) => ieeeMin(ieeeMax(x, lo), hi), pack: half, toHalf: half, differs: (x, y) => x !== y, f16: true };
const BEHAVIOURS = {
  ieee: IEEE,
  "nan to zero": { ...IEEE, load: zero, made: zero, pack: (x) => half(zero(x)), toHalf: (x) => half(zero(x)) },
  "max drops NaN": { ...IEEE, max: dropMax, min: dropMin, clamp: (x, lo, hi) => dropMin(dropMax(x, lo), hi) },
  "pack clamps": { ...IEEE, pack: clampedHalf },
  "folds x != x": { ...IEEE, differs: () => false },
  "no f16": { ...IEEE, f16: false },
};
const device = BEHAVIOURS[kind];

// ---- each case in JavaScript, by its name, as a device of that behaviour makes it: (o, a, b, ua) → a float (float), a
// float16's bits (half) or 0 / 1 (flag). a, b are loaded already; ua is the word of a float16's bits
const f = Math.fround;
const mag = (x) => new Uint32Array(new Float32Array([x]).buffer)[0] & 0x7fffffff;
const softmax2 = (o, s0, s1) => {
  const m = o.max(s0, s1), p0 = o.made(f(Math.exp(f(s0 - m)))), p1 = o.made(f(Math.exp(f(s1 - m))));
  return f(f(p0 * 2 + p1 * 3) / f(p0 + p1));
};
const halfIn = (o, ua) => o.load(floatOfHalf(ua & 0xffff));
const MODELS = {
  "a NaN loaded and stored": (o, a) => a,
  "+Inf loaded and stored": (o, a) => a,
  "−Inf loaded and stored": (o, a) => a,
  "1.5 + NaN": (o, a, b) => f(a + b),
  "NaN × 0": (o, a, b) => f(a * b),
  "+Inf × 0": (o, a, b) => f(a * b),
  "+Inf − +Inf": (o, a, b) => f(a - b),
  "exp(NaN)": (o, a) => f(Math.exp(a)),
  "exp(+Inf)": (o, a) => f(Math.exp(a)),
  "exp(−Inf)": (o, a) => f(Math.exp(a)),
  "max(NaN, 1)": (o, a, b) => o.max(a, b),
  "max(1, NaN)": (o, a, b) => o.max(a, b),
  "min(NaN, 1)": (o, a, b) => o.min(a, b),
  "clamp(NaN, −9, 9)": (o, a) => o.clamp(a, -9, 9),
  "tanh(NaN)": (o, a) => f(Math.tanh(a)),
  "SiLU of a NaN gate": (o, a, b) => f(f(a / f(1 + f(Math.exp(-a)))) * b),
  "SiLU of a −Inf gate": (o, a, b) => f(f(a / f(1 + f(Math.exp(-a)))) * b),
  "GEGLU of a NaN gate": (o, a, b) => f(0.5 * a * (1 + Math.tanh(o.clamp(0.7978845608028654 * (a + 0.044715 * a * a * a), -9.010913, 9.010913))) * b),
  "softmax of the scores 1 and NaN": (o, a, b) => softmax2(o, a, b),
  "softmax of the scores NaN and 1": (o, a, b) => softmax2(o, a, b),
  "softmax of the scores 1 and +Inf": (o, a, b) => softmax2(o, a, b),
  "NaN × 2 found by its bits": (o, a, b) => Number(mag(o.made(f(a * b))) > 0x7f800000),
  "+Inf × 2 found by its bits": (o, a, b) => Number(mag(o.made(f(a * b))) === 0x7f800000),
  "3e38 × 1 taken by its bits for a finite number": (o, a, b) => Number(mag(o.made(f(a * b))) < 0x7f800000),
  "NaN × 2 found by x != x": (o, a, b) => Number(o.differs(o.made(f(a * b)), o.made(f(a * b)))),
  "pack2x16float(NaN)": (o, a) => o.pack(a),
  "pack2x16float(+Inf)": (o, a) => o.pack(a),
  "pack2x16float(−Inf)": (o, a) => o.pack(a),
  "pack2x16float(1e6), past float16's largest": (o, a) => o.pack(a),
  "unpack2x16float of a float16 NaN": (o, a, b, ua) => halfIn(o, ua),
  "unpack2x16float of a float16 +Inf": (o, a, b, ua) => halfIn(o, ua),
  "f16(NaN)": (o, a) => o.toHalf(a),
  "f16(+Inf)": (o, a) => o.toHalf(a),
  "f16(70000), past float16's largest": (o, a) => o.toHalf(a),
  "a float16 NaN × 1 in f16": (o, a, b, ua) => o.toHalf(o.made(halfIn(o, ua) * b)),
  "a float16 NaN + 1 in f16": (o, a, b, ua) => o.toHalf(o.made(halfIn(o, ua) + b)),
  "f32 of a float16 NaN": (o, a, b, ua) => halfIn(o, ua),
};
// the three lists of the cases' names are one: the shaders', the reading's and this check's
const names = NAN_CASES.map((one) => one.name);
assert.equal(new Set(names).size, names.length, "a case's name twice in NAN_CASES");
assert.deepEqual(Object.keys(NAN_READINGS).sort(), [...names].sort(), "src/bench/nan.ts's NAN_READINGS has a row of each case of NAN_CASES, and no other");
assert.deepEqual(Object.keys(MODELS).sort(), [...names].sort(), "this check models each case of NAN_CASES, and no other");
for (const one of NAN_CASES) assert.ok(one.kind === "float" || one.kind === "half" || one.kind === "flag", `${one.name}: a kind of ${one.kind}`);

// ---- the device: answer() runs the cases of the dispatched shader (the f16 one where its text enables f16)
const { device: fake, adapter } = standIn({ fallback: true, answer: (pipeline, entries) => {
  const these = NAN_CASES.filter((one) => Boolean(one.half) === pipeline.code.startsWith("enable f16;"));
  const view = (binding, Type) => { const { resource } = entries.find((entry) => entry.binding === binding), { data } = resource.buffer ?? resource; return new Type(data.buffer, data.byteOffset, data.byteLength / 4); };
  const words = view(0, Uint32Array), floats = view(1, Float32Array), bits = view(2, Uint32Array), stored = view(3, Float32Array);
  these.forEach((one, i) => {
    const result = MODELS[one.name](device, device.load(floats[2 * i]), device.load(floats[2 * i + 1]), words[2 * i]);
    if (one.kind === "float") {
      const made = device.made(f(result));
      stored[i] = made;
      bits[i] = new Uint32Array(new Float32Array([made]).buffer)[0];
    } else bits[i] = result;
  });
} });
if (device.f16) { fake.features.add("shader-f16"); adapter.features.add("shader-f16"); }
globalThis.self = globalThis;
const posted = [];
globalThis.postMessage = (message) => posted.push(message);
let receiver;
Object.defineProperty(globalThis, "onmessage", { configurable: true, get: () => receiver, set: (handler) => { receiver = handler; } });
await import(runtimeUrl("benchmark/gpu.js"));
await receiver({ data: { step: "nan" } });
const answer = posted.find((message) => message.step === "nan");
assert.ok(answer && !answer.error, `nan: ${answer?.error}`);
const found = answer.result;
assert.deepEqual(found.cases.map((one) => one.name), names, "a case a row, in NAN_CASES' order");

// ---- what each device breaks: the marks the reading says do not hold here
const broken = nanBroken(found).map((line) => line.replace(/ \(made .*\)$/, ""));
const reliedOn = (test) => names.filter((name) => NAN_READINGS[name].relied && test(name));
const WANTS = {
  ieee: [],
  // every case a mark stands on that has a NaN in it (the device zeroes NaNs it loads or makes), and the two that make
  // one from an infinity (−Inf / +Inf in SiLU, +Inf − +Inf in the softmax)
  "nan to zero": reliedOn((name) => /NaN/.test(name) && !/^\+Inf/.test(name)).concat("softmax of the scores 1 and +Inf", "SiLU of a −Inf gate"),
  // the engine stands on no max() of a NaN: exp() carries it through the softmax
  "max drops NaN": [],
  "pack clamps": ["pack2x16float(NaN)", "pack2x16float(+Inf)", "pack2x16float(−Inf)"],
  "folds x != x": [],
  "no f16": [],
};
assert.deepEqual(broken.sort(), [...WANTS[kind]].sort(), `${kind}: the marks that do not hold`);

// ---- the table: a row a case; "failed" in the rows of the marks broken and in no other, so that warnings() lists
// those (and the head says none of the words)
const markdown = gpuMarkdown({ bandwidth: [], token: [], info: { result: { adapter: "a GPU", fallback: false } }, nan: { name: "NaN and infinities", result: found } }, {}, undefined, undefined).markdown;
const lines = nanTable({ name: "NaN and infinities", result: found });
assert.ok(lines.every((line) => markdown.includes(line)), "the section's Markdown holds the table");
const rowOf = (name) => cellsOf(lines.find((line) => line.startsWith(`| ${name.replace(/\|/g, "\\|")} |`)));
for (const name of names) {
  const row = rowOf(name), failed = /\bfailed\b/.test(row.join(" "));
  assert.equal(failed, WANTS[kind].includes(name), `${kind}: ${name}'s row ${failed ? "says" : "does not say"} failed: ${row.join(" | ")}`);
}
const listed = warnings([{ title: "GPU", status: "ok", markdown }]);
assert.equal(listed.length, WANTS[kind].length, `${kind}: a warning a mark broken:\n${listed.join("\n")}`);
const f16Rows = names.filter((name) => NAN_CASES.find((one) => one.name === name).half);
if (kind === "no f16") for (const name of f16Rows) assert.equal(rowOf(name)[2], "no shader-f16 here", `${name} where there is no shader-f16`);
// what each device makes of a case no mark stands on, as the row says it
if (kind === "max drops NaN") assert.deepEqual(["max(NaN, 1)", "max(1, NaN)", "min(NaN, 1)", "clamp(NaN, −9, 9)"].map((name) => rowOf(name).slice(2)),
  [["1", "nothing"], ["1", "nothing"], ["1", "nothing"], ["−9", "nothing"]]);
if (kind === "folds x != x") assert.deepEqual(rowOf("NaN × 2 found by x != x").slice(2), ["no", "nothing"]);
if (kind === "ieee") {
  assert.deepEqual(rowOf("pack2x16float(1e6), past float16's largest").slice(1), ["+Inf", "+Inf", "nothing: as IEEE 754 has it"]);
  assert.deepEqual(rowOf("NaN × 0").slice(2), ["NaN", "T241: a group's NaN scale times its integer sum (0 where its values are) stays NaN"]);
}
if (kind === "pack clamps") assert.equal(rowOf("pack2x16float(+Inf)")[2], "65504");
// the summary's line
const summary = nanSummary({ name: "NaN and infinities", result: found });
const relied = names.filter((name) => NAN_READINGS[name].relied && !(kind === "no f16" && f16Rows.includes(name)));
assert.ok(WANTS[kind].length ? summary.includes(`${relied.length - WANTS[kind].length} of the ${relied.length} cases the engine stands on carried, failed: `)
  : summary.includes(`the ${relied.length} cases the engine stands on all carried`), summary);
// a step that failed, and one not run
assert.deepEqual(nanTable({ name: "t", error: "the worker stopped" }).at(-1), "failed: the worker stopped");
assert.deepEqual(nanTable(undefined).at(-1), "not measured");
assert.equal(nanSummary(undefined), undefined);
console.log(`bench-nan-check: ${kind}: ${found.cases.length} cases, ${broken.length} marks that do not hold${broken.length ? ` (${broken.join(", ")})` : ""}: ok`);
