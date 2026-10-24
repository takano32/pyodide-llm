// tests/shaders-catalogue-check.mjs (T.379.1): the catalogue of the engine's shaders (src/runtime/shaders/engine.ts) by
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
//   - (T.379.2) every maker of WGSL the window exports has rows here, or arguments in tests/shader-makers.mjs (the
//     tiles', the benchmark's), each set of which makes a text of its own.
//   - (T.379.2) a text declares nothing it does not use, and declares the struct of what it binds.
// What a maker's text branches on that no row walks (a new parameter) is tests/device-key-check.mjs's, which changes
// each piece of the shaders' source by a character; that the worker takes shaders nowhere else, too.
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";
import { MAKERS_APART, makersOfWgsl } from "./shader-makers.mjs";

const { ENGINE, ENGINE_SHADERS, TERNARY_SHADERS, BLOCK_SHADERS, deviceKey, devicePromptForms, ...window } = await import(runtimeUrl("shaders.js"));

// ---- the order. A row: its name, and for a maker the choices it is given (what is no number), in the row's own order
const said = ({ name, given }) => (given ? `${name}(${Object.entries(given).filter(([, value]) => typeof value !== "number").map(([key, value]) => `${key}=${value}`).join(", ")})` : name);
const EITHER = [false, true];
const ORDER = {
  engine: ["RMSNORM", "HEAD_NORM", "ADD", "ROPE", "SWIGLU", "QUANTIZE", "LAYER_NORM", "GELU", "EMBED", "EMBED_ROWS", "SAMPLE", "NORM_QUANTIZE", "TOKEN_ROPE", "WIDEN_SIX",
    "RMSNORM_ADD", "GEGLU",  // (T.261.4: a Gemma 3's norms behind and its gate)
    "SHORT_CONV", "CONV_STATE",  // (T.545: an LFM2's convolution layers)
    "DELTA_GATES", "DELTA_CONV", "DELTA_CONV_STATE", "DELTA_NET", "OUTPUT_GATE",  // (T.271.2: a Qwen3.5's linear-attention layers and its output gate)
    // (T.379.2: each maker's rows together; within a maker subgroups is the outer choice)
    ...EITHER.flatMap((subgroups) => EITHER.map((half) => `flashTile(half=${half}, subgroups=${subgroups})`)),
    ...EITHER.map((subgroups) => `flashVec(subgroups=${subgroups})`),
    ...EITHER.map((subgroups) => `flashVecReduce(subgroups=${subgroups})`),
    ...EITHER.flatMap((subgroups) => ["norm", "plain"].flatMap((input) => ["rope", "add", "swiglu", "write", "geglu"].map((output) => `fusedMatVec(input=${input}, output=${output}, subgroups=${subgroups})`))),
    "fusedDp4aMatVec(output=rope, weights=int8)", "fusedDp4aMatVec(output=add, weights=int8)", "fusedDp4aMatVec(output=swiglu, weights=int8)", "fusedDp4aMatVec(output=write, weights=int8)",
    "fusedDp4aMatVec(output=geglu, weights=int8)"],
  ternary: [
    "fusedDp4aMatVec(output=rope, weights=ternary)", "fusedDp4aMatVec(output=add, weights=ternary)", "fusedDp4aMatVec(output=swiglu, weights=ternary)", "fusedDp4aMatVec(output=write, weights=ternary)",
    "fusedDp4aMatVec(output=geglu, weights=ternary)",
    "EMBED_TERNARY", "EMBED_ROWS_TERNARY", "TAKE_OUTLIERS", "TERNARY_COLUMNS", "ROTATE_QUANTIZE", "UNROTATE"],
  // T.547.2.4: a model's of llama.cpp's blocks besides int8 (T.547.3.3: a type after another)
  blocks: ["Q4_K", "Q2_K", "IQ2_XXS", "IQ2_XS", "IQ2_S"].flatMap((weights) => ["rope", "add", "swiglu", "write", "geglu"].map((output) => `fusedDp4aMatVec(output=${output}, weights=${weights})`)),
};
const moved = "the order of the catalogue's rows is the device's key: a row moved, added or taken away has every device choose anew (such changes are gathered into one deployment, as T.379.2 was). If that is meant, this list is to say the same";
assert.deepEqual(ENGINE_SHADERS.map(said), ORDER.engine, `every model's shaders: ${moved}`);
assert.deepEqual(TERNARY_SHADERS.map(said), ORDER.ternary, `a ternary model's shaders: ${moved}`);
assert.deepEqual(BLOCK_SHADERS.map(said), ORDER.blocks, `a model of blocks' shaders: ${moved}`);
console.log(`shaders-catalogue-check: the rows are in the recorded order (${ENGINE_SHADERS.length} of every model, ${TERNARY_SHADERS.length} of a ternary model alone, ${BLOCK_SHADERS.length} of a model of blocks alone)`);

// ---- a row
const rows = [...ENGINE_SHADERS, ...TERNARY_SHADERS, ...BLOCK_SHADERS];
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
  // (T.379.2: a piece of text shaders share is no row: it is in the key by the texts that hold it)
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

// ---- (T.379.2) every maker of WGSL the window exports is walked by someone: the catalogue's rows (above), or the list
// of the makers apart (tests/shader-makers.mjs: the prompt's tiles and the benchmark's own, which the net compares
// with another commit's by the texts those arguments make). A maker in neither would be compared by its syntax tree
// alone, and a branch of it by nothing; a maker in both is the catalogue's
const shadersModule = await import(runtimeUrl("shaders.js"));
assert.deepEqual(makersOfWgsl(shadersModule), [...new Set([...makers.keys(), ...Object.keys(MAKERS_APART)])].sort(),
  "the makers of WGSL that shaders.ts exports are the catalogue's and tests/shader-makers.mjs's: a new one is to have rows in shaders/engine.ts (the engine's) or arguments in that list");
for (const name of Object.keys(MAKERS_APART)) assert.ok(!makers.has(name), `${name} has rows in the catalogue: tests/shader-makers.mjs is for the makers that have none`);
for (const [name, givens] of Object.entries(MAKERS_APART)) {
  const made = givens.map((given) => shadersModule[name](...given));
  for (const text of made) assert.ok(typeof text === "string" && /@compute/.test(text), `${name}: a shader of each set of arguments`);
  assert.equal(new Set(made).size, givens.length, `${name}: two sets of arguments of tests/shader-makers.mjs make one text: an argument the maker does not take (its branches are not walked)`);
}
console.log(`shaders-catalogue-check: ${makersOfWgsl(shadersModule).length} makers of WGSL, each the catalogue's (${makers.size}) or of the list apart (${Object.keys(MAKERS_APART).join(", ")})`);

// ---- (T.379.2) what a text declares, it uses, and the type of what it binds is declared: every text a device is given
// (the rows' and the tiles'), read as text (no WGSL compiler here: that a text compiles is CI's gpu-check). Until T.379.2
// a token's matrices declared the Step struct where no step was bound, and a constant only the norm's form read; a
// declaration that comes back unused, or a struct dropped where it is bound, is said here. Names a piece of text
// shared by two shaders gives both, of which one uses fewer, are listed (LEFT): taking them out is another text, and
// every device's key.
const LEFT = { "flashVec(subgroups=true)": ["lanes_add4"],
  "flashVecReduce(subgroups=false)": ["shuffle_down", "shuffle_down4", "shuffle"], "flashVecReduce(subgroups=true)": ["shuffle_down", "shuffle_down4", "shuffle"] };
const given = [...rows.map((row) => [said(row), row.text()]),
  // (T.547.2.3: and a Q4_K model's, each with its Q4_K text)
  ...[null, "ternary", "blocks"].flatMap((own) => devicePromptForms({ features: new Set(["shader-f16", "subgroups"]), limits: { maxComputeWorkgroupStorageSize: 1e9, maxComputeInvocationsPerWorkgroup: 1e9, maxComputeWorkgroupSizeX: 1e9 } }, own)
    .concat(devicePromptForms({ features: new Set(), limits: { maxComputeWorkgroupStorageSize: 1e9, maxComputeInvocationsPerWorkgroup: 1e9, maxComputeWorkgroupSizeX: 1e9 } }, own))
    .flatMap((form) => [[`the tile ${form.name}`, form.code], ...Object.entries(form.blocks ?? {}).map(([packing, code]) => [`the ${packing} text of ${form.name}`, code])]))];
let declarations = 0;
for (const [what, raw] of given) {
  const text = raw.replace(/\/\/.*$/gm, "");
  const unused = [];
  for (const found of text.matchAll(/^(?:@group\(0\) @binding\(\d+\) )?(?:(?:struct|const|fn|override|alias)\s+(\w+)|var(?:<[^>]*>)?\s+(\w+)\s*:)/gm)) {
    const declared = found[1] ?? found[2];
    if (declared === "main") continue;
    declarations += 1;
    if ([...text.matchAll(new RegExp(`\\b${declared}\\b`, "g"))].length === 1) unused.push(declared);
  }
  assert.deepEqual(unused, LEFT[what] ?? [], `${what}: what its text declares and never uses (but what tests/shaders-catalogue-check.mjs's LEFT lists)`);
  for (const bound of text.matchAll(/^@group\(0\) @binding\(\d+\) var<[^>]*>\s+\w+\s*:\s*([A-Z]\w*)\s*;/gm)) {
    assert.ok(new RegExp(`^struct ${bound[1]} \\{`, "m").test(text), `${what} binds a ${bound[1]} and declares no struct of that name`);
  }
}
console.log(`shaders-catalogue-check: ${given.length} texts (the rows' and the tiles'): each of their ${declarations} declarations is used, and each struct that is bound is declared`);

// ---- (T.379.2) a prompt's tile of a ternary model reads codes, and no other tile does: the packing dp4a() is given by
// its name is the one of the form's own name (read backwards, a ternary model was handed the int8 text under the
// ternary name, and no local check said so)
{
  const every = { features: new Set(["shader-f16", "subgroups"]), limits: { maxComputeWorkgroupStorageSize: 1e9, maxComputeInvocationsPerWorkgroup: 1e9, maxComputeWorkgroupSizeX: 1e9 } };
  const reads = (form) => /fn ternary_packed\(/.test(form.code) && /var<storage, read> b: array<u32>;/.test(form.code);
  const ternary = devicePromptForms(every, "ternary"), others = devicePromptForms(every, null);
  assert.ok(ternary.length > 0 && ternary.every((form) => form.ternary && / ternary$/.test(form.name) && reads(form)), "every tile of a ternary model unpacks its weights' codes (ternary_packed), under a name that says ternary");
  assert.ok(others.length > ternary.length && others.every((form) => !form.ternary && !/ternary/.test(form.name) && !reads(form)), "no tile of another model reads ternary codes");
  console.log(`shaders-catalogue-check: the ${ternary.length} tiles of a ternary model read codes, the ${others.length} of the others do not`);
}

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
const tiles = (device, ternary) => devicePromptForms(device, ternary ? "ternary" : null).flatMap((form) => [form.name, form.code]);
const texts = (list) => list.map((row) => row.text());
const hashes = new Set();
for (const device of DEVICES) {
  for (const ternary of [false, true]) {
    const key = deviceKey(device, device, ternary ? "ternary" : null);
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
