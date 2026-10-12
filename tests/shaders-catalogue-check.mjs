// tests/shaders-catalogue-check.mjs (T379.1): the catalogue of the engine's shaders (src/runtime/shaders/engine.ts) by
// itself: no GPU, no worker, no model. Node only, under a second:
//
//   node tests/shaders-catalogue-check.mjs
//
// The catalogue is one list three read: the device's key hashes its rows' texts in their order, the model's GPU worker
// takes its shaders from it by name, and the checks walk it. Held here:
//   - the order of the rows is the recorded one (ORDER, below: a second copy on purpose). The order is the key's: a row
//     moved, added or dropped has every device choose its forms again, and no other check of the light suite says so
//     (the net's shaders does, tests/unchanged.mjs, which compares the keys with another commit's);
//   - a row is a text or a maker with its arguments, and its text() is that text or what the maker makes of them;
//     every row's text is a shader, and no two rows give one text:
//     a choice given to a maker that does not take it (a misspelt name of a packing, an argument left out) makes a
//     second copy of another row's text and leaves a text a device runs out of the key;
//   - a maker's rows are every choice of it: each of its arguments that is no number takes every value in every
//     combination with the others (subgroups × half; input × output × subgroups), and a number is the same in all;
//   - ENGINE, which the worker reads, is the rows by name and nothing else: a text is the row's own, a maker the
//     function itself (called with a model's numbers), and it cannot be written to by a name the rows do not have;
//   - the key is the fold of these rows and of nothing else of the catalogue: computed here a second time from the
//     rows (FNV-1a, the unit separator after each text) for eight made-up devices and their ternary models.
// What a maker's text branches on that no row walks (a new parameter) is tests/device-key-check.mjs's, which changes
// each piece of the shaders' source by a character; that the worker takes shaders nowhere else, too.
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";

const { ENGINE, ENGINE_SHADERS, TERNARY_SHADERS, deviceKey, devicePromptForms, ...window } = await import(runtimeUrl("shaders.js"));

// ---- the order. A row: its name, and for a maker the choices it is given (what is no number), in the row's own order
const said = ({ name, given }) => (given ? `${name}(${Object.entries(given).filter(([, value]) => typeof value !== "number").map(([key, value]) => `${key}=${value}`).join(", ")})` : name);
const EITHER = [false, true];
const ORDER = {
  engine: ["RMSNORM", "HEAD_NORM", "ADD", "ROPE", "SWIGLU", "QUANTIZE", "LAYER_NORM", "GELU", "EMBED", "EMBED_ROWS", "SAMPLE", "NORM_QUANTIZE", "TOKEN_ROPE", "WIDEN_SIX",
    // (T379.2: each maker's rows together; within a maker subgroups is the outer choice)
    ...EITHER.flatMap((subgroups) => EITHER.map((half) => `flashTile(half=${half}, subgroups=${subgroups})`)),
    ...EITHER.map((subgroups) => `flashVec(subgroups=${subgroups})`),
    ...EITHER.map((subgroups) => `flashVecReduce(subgroups=${subgroups})`),
    ...EITHER.flatMap((subgroups) => ["norm", "plain"].flatMap((input) => ["rope", "add", "swiglu", "write"].map((output) => `fusedMatVec(input=${input}, output=${output}, subgroups=${subgroups})`))),
    "fusedDp4aMatVec(output=rope, weights=int8)", "fusedDp4aMatVec(output=add, weights=int8)", "fusedDp4aMatVec(output=swiglu, weights=int8)", "fusedDp4aMatVec(output=write, weights=int8)"],
  ternary: [
    "fusedDp4aMatVec(output=rope, weights=ternary)", "fusedDp4aMatVec(output=add, weights=ternary)", "fusedDp4aMatVec(output=swiglu, weights=ternary)", "fusedDp4aMatVec(output=write, weights=ternary)",
    "EMBED_TERNARY", "EMBED_ROWS_TERNARY", "TAKE_OUTLIERS", "TERNARY_COLUMNS"],
};
const moved = "the order of the catalogue's rows is the device's key: a row moved, added or taken away has every device choose anew (such changes are gathered into one deployment, as T379.2 was). If that is meant, this list is to say the same";
assert.deepEqual(ENGINE_SHADERS.map(said), ORDER.engine, `every model's shaders: ${moved}`);
assert.deepEqual(TERNARY_SHADERS.map(said), ORDER.ternary, `a ternary model's shaders: ${moved}`);
console.log(`shaders-catalogue-check: the rows are in the recorded order (${ENGINE_SHADERS.length} of every model, ${TERNARY_SHADERS.length} of a ternary model alone)`);

// ---- a row
const rows = [...ENGINE_SHADERS, ...TERNARY_SHADERS];
const seen = new Map();
for (const row of rows) {
  const what = said(row), text = row.text();
  assert.deepEqual(Object.keys(row).sort(), (row.given ? ["given", "name", "shader", "text"] : ["name", "shader", "text"]), `${what}: a row's fields`);
  if (row.given) {
    assert.equal(typeof row.shader, "function", `${what}: a row with arguments is a maker's`);
    assert.equal(text, row.shader(row.given), `${what}: its text is what the maker makes of its arguments`);
  } else assert.equal(text, row.shader, `${what}: a text's row gives that text`);
  // (the row's name is its shader's own: the window has every shader by its name, and no piece)
  assert.equal(window[row.name], row.shader, `${what}: the row's shader is the one of that name (the window's)`);
  assert.equal(text, row.text(), `${what}: the same text twice`);
  assert.ok(typeof text === "string" && text.length > 100, `${what}: a text`);
  // (T379.2: a piece of text shaders share is no row: it is in the key by the texts that hold it)
  assert.ok(/@compute/.test(text), `${what}: a row is a shader, with an entry point`);
  assert.ok(!seen.has(text), `${what} gives the text ${seen.get(text)} gives: an argument the maker does not take, or a row twice (and a text a device runs that the key does not hold?)`);
  seen.set(text, what);
}
console.log(`shaders-catalogue-check: ${rows.length} rows, each a text of its own (${rows.filter((row) => row.given).length} made by ${new Set(rows.filter((row) => row.given).map((row) => row.name)).size} makers)`);

// ---- a maker's rows: every combination of its choices, one set of numbers
const makers = new Map();
for (const row of rows) if (row.given) makers.set(row.name, [...(makers.get(row.name) ?? []), row.given]);
for (const [name, givens] of makers) {
  const keys = Object.keys(givens[0]);
  for (const given of givens) assert.deepEqual(Object.keys(given), keys, `${name}: every row gives it the same arguments`);
  const numbers = keys.filter((key) => typeof givens[0][key] === "number"), choices = keys.filter((key) => !numbers.includes(key));
  for (const key of numbers) assert.deepEqual([...new Set(givens.map((given) => given[key]))], [givens[0][key]], `${name}'s ${key}: a number is the same in every row (the key walks one set of numbers)`);
  const values = choices.map((key) => new Set(givens.map((given) => given[key])));
  const combinations = values.reduce((count, set) => count * set.size, 1);
  assert.equal(new Set(givens.map((given) => JSON.stringify(choices.map((key) => given[key])))).size, givens.length, `${name}: a choice twice`);
  assert.equal(givens.length, combinations, `${name}: ${givens.length} rows for ${choices.map((key, i) => `${values[i].size} of ${key}`).join(" × ")}: a combination of its choices has no row, and its text is in no key`);
  for (const [i, key] of choices.entries()) assert.ok(values[i].size > 1, `${name}'s ${key} takes one value in every row: no choice`);
}
console.log(`shaders-catalogue-check: ${[...makers].map(([name, givens]) => `${name} × ${givens.length}`).join(", ")}: every combination of each maker's choices, one set of numbers`);

// ---- ENGINE: the rows by name
assert.deepEqual(Object.keys(ENGINE), [...new Set(rows.map((row) => row.name))], "ENGINE's names are the rows', in their order");
for (const row of rows) assert.equal(ENGINE[row.name], row.shader, `ENGINE.${row.name} is the row's shader`);
assert.equal(ENGINE.fusedDp4aMatVec({ output: "add", weights: "ternary" }), TERNARY_SHADERS.find((row) => said(row) === "fusedDp4aMatVec(output=add, weights=ternary)").text(), "a maker of ENGINE, called as the worker calls it, makes the row's text");
assert.equal(ENGINE.NO_SUCH_SHADER, undefined, "a name without a row is no shader of ENGINE (the type check says so of the worker's modules)");
console.log(`shaders-catalogue-check: ENGINE is the rows by name (${Object.keys(ENGINE).length} names)`);

// ---- the key: the fold of the tiles a device can make and of the rows, and of nothing else
const DEVICES = [false, true].flatMap((half) => [false, true].map((subgroups) => ({ info: { vendor: "v", architecture: "a", device: "d", description: "x" },
  features: new Set([...(half ? ["shader-f16"] : []), ...(subgroups ? ["subgroups"] : [])]),
  limits: { maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 } })));
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { userAgent: "UA", gpu: { wgslLanguageFeatures: new Set(["packed_4x8_integer_dot_product"]) } } });
function folded(texts) {
  let hash = 0x811c9dc5;
  for (const text of texts) {
    for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193);
    hash = Math.imul(hash ^ 0x1f, 0x01000193);
  }
  return (hash >>> 0).toString(16);
}
const tiles = (device, ternary) => devicePromptForms(device, ternary).flatMap((form) => [form.name, form.code]);
const texts = (list) => list.map((row) => row.text());
const hashes = new Set();
for (const device of DEVICES) {
  for (const ternary of [false, true]) {
    const key = deviceKey(device, device, ternary);
    const expected = folded([...tiles(device, false), ...texts(ENGINE_SHADERS), ...(ternary ? [...tiles(device, true), ...texts(TERNARY_SHADERS)] : [])]);
    assert.equal(key.split("|").at(-1), expected, `the key's hash is the fold of the device's tiles and the catalogue's rows${ternary ? ", a ternary model's after every model's" : ""}`);
    hashes.add(expected);
  }
}
assert.equal(hashes.size, 2 * DEVICES.length, "a hash for each device, and for its ternary models");
// (and the fold is of the order: two rows changed about are another hash)
const swapped = texts(ENGINE_SHADERS);
[swapped[0], swapped[1]] = [swapped[1], swapped[0]];
assert.notEqual(folded(swapped), folded(texts(ENGINE_SHADERS)), "the fold moves with the order");
console.log(`shaders-catalogue-check: the key of ${DEVICES.length} made-up devices, and of their ternary models, is the fold of their tiles and of the rows`);
console.log("shaders-catalogue-check: ok");
