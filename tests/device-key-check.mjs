// tests/device-key-check.mjs (T366): what public/shaders.js's deviceKey() is made of.
//
// The key says whether what a device measured (the forms it chose, T148; that the CPU was faster than a model on the GPU
// alone, T156) still holds. It is to change when a shader the engine runs changes by a character, and not otherwise:
// not when the file is formatted, minified or bundled, a local or a maker renamed, a parameter or a comment added.
// (Until T366 the key hashed String(maker) of four makers, and four shaders gpu.js runs were not in it.) Node only:
//
//   node tests/device-key-check.mjs
//
// (1) Every piece of text of the engine's shaders reaches the hash. A piece is a string, a number or a run of a
//     template's text in the source of public/shaders.js; the engine's shaders are every text and maker that
//     public/gpu.js names (wgsl.<name>) and the prompt's tiles, with all they are made of (the names their source
//     reaches). Each piece is changed by one character, one at a time, the changed file run, and the key of one of
//     eight made-up devices must differ: so every branch of every maker is walked by what deviceKey() makes of it.
//     A piece of a ternary model's shaders must leave the key of every other model alone (T232); a piece of anything
//     else in the file (the benchmark's shaders, the JavaScript the checks compare with) must leave every key alone.
// (2) The key is the same when the source is not: esbuild's reprint, its minifier (white space, names, syntax), its
//     wrapping as another format of module, a comment in every function, a parameter more for every function, every
//     maker renamed. A copy of the file that hashes one maker's source as well (what the key did) is the control:
//     every one of these changes moves that copy's key.
// The parser is @babel/parser and the minifier esbuild, both of which Astro's packages bring.
import assert from "node:assert/strict";
import fs from "node:fs";
import { parse } from "@babel/parser";
import { transformSync } from "esbuild";

const read = (file) => fs.readFileSync(new URL(`../public/${file}`, import.meta.url), "utf8");
const source = read("shaders.js");
const program = parse(source, { sourceType: "module" }).program;

// ---- the made-up devices: the features that choose the prompt's tiles, and whether the model is of ternary weights
const DEVICES = [false, true].flatMap((half) => [false, true].map((subgroups) => ({ info: { vendor: "v", architecture: "a", device: "d", description: "x" },
  features: new Set([...(half ? ["shader-f16"] : []), ...(subgroups ? ["subgroups"] : [])]),
  limits: { maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 } })));
Object.defineProperty(globalThis, "navigator", { configurable: true,
  value: { userAgent: "UA", gpu: { wgslLanguageFeatures: new Set(["packed_4x8_integer_dot_product"]) } } });
const keysOf = (deviceKey, ternary) => DEVICES.map((device) => deviceKey(device, device, ternary));

// ---- the file's syntax tree: every node under one, and the top-level declarations by name
function* under(node) {
  if (!node || typeof node.type !== "string") return;
  yield node;
  for (const [key, value] of Object.entries(node)) {
    if (key === "loc" || key === "leadingComments" || key === "trailingComments" || key === "innerComments") continue;
    for (const child of Array.isArray(value) ? value : [value]) if (child && typeof child === "object") yield* under(child);
  }
}
const declared = new Map(), exportWords = [];
for (const statement of program.body) {
  const s = statement.type === "ExportNamedDeclaration" ? statement.declaration : statement;
  assert.ok(s, "an export without a declaration: this check knows `export const` and `export function`");
  if (statement !== s) exportWords.push(statement.start);
  if (s.type === "FunctionDeclaration") declared.set(s.id.name, s);
  else if (s.type === "VariableDeclaration") {
    for (const d of s.declarations) {
      assert.equal(d.id.type, "Identifier", "a top-level pattern: this check knows names");
      declared.set(d.id.name, d);
    }
  } else assert.fail(`a top-level ${s.type}: this check knows declarations`);
}
// an identifier that is a name in use, not a property's or a key's
function* namesIn(node) {
  const skipped = new Set();
  for (const n of under(node)) {
    if ((n.type === "MemberExpression" || n.type === "OptionalMemberExpression") && !n.computed) skipped.add(n.property);
    if ((n.type === "ObjectProperty" || n.type === "ObjectMethod") && !n.computed && !n.shorthand) skipped.add(n.key);
    if (n.type === "Identifier" && !skipped.has(n)) yield n;
  }
}
/** the top-level names a declaration's source reaches, itself among them */
function reached(names) {
  const found = new Set();
  const walk = (name) => {
    if (found.has(name) || !declared.has(name)) return;
    found.add(name);
    for (const n of namesIn(declared.get(name))) walk(n.name);
  };
  names.forEach(walk);
  return found;
}
/** the pieces of text of a declaration: [where a character goes in, what the piece says] */
function pieces(name) {
  const found = [];
  for (const n of under(declared.get(name))) {
    if (n.type === "TemplateElement") found.push({ name, at: n.start, says: n.value.raw, insert: "~" });
    if (n.type === "StringLiteral") found.push({ name, at: n.start + 1, says: n.value, insert: "~" });
    // (a number: a digit more before it; 0x…, a float and a BigInt are not written so in what the shaders are made of)
    if (n.type === "NumericLiteral") found.push({ name, at: n.start, says: source.slice(n.start, n.end), insert: /^[1-9]\d*$/.test(source.slice(n.start, n.end)) ? "1" : null });
  }
  return found;
}

/** the text with these [at, what goes in] */
const withInserts = (text, inserts) => inserts.sort((a, b) => b[0] - a[0]).reduce((now, [at, what]) => `${now.slice(0, at)}${what}${now.slice(at)}`, text);

// ---- running a text of the file: `export` blanked (the places stay), the top level a function's
function run(text, places = exportWords) {
  let plain = text;
  for (const at of places) {
    assert.equal(plain.slice(at, at + 7), "export ", "an export where the tree says one");
    plain = `${plain.slice(0, at)}       ${plain.slice(at + 7)}`;
  }
  return new Function(`${plain}\nreturn { deviceKey };`)().deviceKey;
}
const real = (await import("../public/shaders.js"));
const KEYS = { plain: keysOf(real.deviceKey, false), ternary: keysOf(real.deviceKey, true) };
assert.deepEqual(keysOf(run(source), false), KEYS.plain, "the file run as a function's body gives the module's keys");
assert.equal(new Set([...KEYS.plain, ...KEYS.ternary]).size, 2 * DEVICES.length, "a key for each device, and for its ternary models");

// ---- (1) what the engine runs, and that each piece of it reaches the hash
// the prompt's tiles (gpu.js takes them through devicePromptForms), and the makers' own
const TILES = ["regTile", "tfjsTile", "dp4a"];
// the key's own lines (their strings choose what is made: no text of a shader), and what names the prompt's forms and
// says why one is not made (a form's name is hashed with its text: the tiles' sizes are in it)
const KEY_ITSELF = ["deviceKey", "engineShaders", "ternaryShaders", "EITHER", "OUTPUTS"], FORMS = ["devicePromptForms"];
// the shaders of a model of ternary weights alone (T232): no other model's key holds them
const TERNARY = ["ternaryMatVec", "EMBED_TERNARY", "EMBED_ROWS_TERNARY", "TAKE_OUTLIERS", "TERNARY_COLUMNS"];
// (and the makers of every model's shaders that have branches for ternary weights, and what only those branches write)
const TERNARY_BRANCHES = ["dp4a", "TERNARY_PACKED"];
const isShader = (name) => typeof real[name] === "string"
  || (typeof real[name] === "function" && [...reached([name])].some((part) => pieces(part).some((piece) => typeof piece.says === "string" && piece.says.includes("@compute"))));
const used = [...new Set([...read("gpu.js").matchAll(/\bwgsl\.([A-Za-z_][A-Za-z_0-9]*)/g)].map((match) => match[1]))].sort();
for (const name of used) assert.ok(name in real, `gpu.js names wgsl.${name}, which shaders.js does not export`);
const shaders = [...used.filter((name) => ![...KEY_ITSELF, ...FORMS].includes(name) && isShader(name)), ...TILES];
assert.ok(shaders.length >= 26, `the shaders gpu.js runs: ${shaders.length} found (27 when this was written: a pattern that no longer finds them?)`);
for (const name of TERNARY) assert.ok(shaders.includes(name), `${name} is no shader of gpu.js any more`);
const others = shaders.filter((name) => !TERNARY.includes(name));
const engine = reached(shaders), general = reached(others);
const itself = new Set([...KEY_ITSELF, ...reached(FORMS)].filter((name) => !engine.has(name)));

// (the devices one at a time, those with most of the forms first: most pieces move the first one's key)
const moves = (deviceKey, ternary) => DEVICES.toReversed().some((device) => deviceKey(device, device, ternary) !== KEYS[ternary ? "ternary" : "plain"][DEVICES.indexOf(device)]);
const changedBy = (pieces) => run(withInserts(source, pieces.map((piece) => [piece.at, piece.insert])),
  exportWords.map((at) => at + pieces.filter((piece) => piece.at < at).reduce((sum, piece) => sum + piece.insert.length, 0)));
const counted = { engine: 0, ternary: 0, branches: 0, numbers: 0, apart: [] };
for (const name of declared.keys()) {
  if (itself.has(name)) continue;  // (a reason a form is not made is no shader's text; a form's name is in the key)
  const ternaryOnly = !general.has(name);
  for (const piece of pieces(name)) {
    if (piece.insert === null) continue;
    if (!engine.has(name)) {
      counted.apart.push(piece);
      continue;
    }
    const said = `${name}, ${JSON.stringify(piece.says.slice(0, 60))} (at ${piece.at})`;
    const deviceKey = changedBy([piece]);
    if (ternaryOnly) {
      assert.ok(moves(deviceKey, true), `${said}, of a ternary model's shaders, did not reach the hash of a ternary model's key`);
      assert.ok(!moves(deviceKey, false), `${said}, of a ternary model's shaders, moved the key of the other models`);
      counted.ternary += 1;
    } else if (moves(deviceKey, false)) {
      counted.engine += 1;
    } else {
      // (the branches for ternary weights of a maker every model's shaders come from)
      assert.ok(TERNARY_BRANCHES.includes(name) && moves(deviceKey, true),
        `${said}, of the engine's shaders, did not reach the hash: a branch of a maker that deviceKey() does not walk?`);
      counted.branches += 1;
    }
    if (piece.insert === "1") counted.numbers += 1;
  }
}
// the rest of the file (the benchmark's shaders, the JavaScript the checks compare with): all changed at once
const apart = changedBy(counted.apart);
assert.ok(!moves(apart, false) && !moves(apart, true), "what gpu.js does not run moved the key");
assert.ok(counted.engine > 100 && counted.ternary > 10 && counted.branches > 0 && counted.apart.length > 50, `too few pieces: ${JSON.stringify({ ...counted, apart: counted.apart.length })}`);
console.log(`device-key-check: ${shaders.length} shaders of gpu.js (${shaders.join(", ")}), made of ${engine.size} declarations`);
console.log(`device-key-check: one character more in each of ${counted.engine + counted.ternary + counted.branches} pieces of them moved the key ` +
  `(${counted.numbers} of the pieces are numbers; ${counted.ternary} pieces of the ternary shaders and ${counted.branches} of the branches for ternary weights moved a ternary model's alone), ` +
  `and one more in all ${counted.apart.length} pieces of the rest of the file did not`);

// ---- (2) the same key from another source
const functions = (tree) => [...under(tree)].filter((n) => n.type === "ArrowFunctionExpression" || n.type === "FunctionDeclaration" || n.type === "FunctionExpression");
const makers = [...declared].filter(([name, d]) => name !== "deviceKey" && (d.type === "FunctionDeclaration" || /Function/.test(d.init?.type ?? ""))).map(([name]) => name);
const CHANGES = {
  // a printer's own white space, quotes and line breaks
  "reprinted by esbuild": (text) => transformSync(text, { format: "esm" }).code,
  "minified by esbuild (white space, names, syntax)": (text) => transformSync(text, { format: "esm", minify: true }).code,
  "wrapped as another format of module (esbuild's cjs, minified)": (text) => transformSync(text, { format: "cjs", minify: true }).code,
  "a comment in every function": (text, tree) => withInserts(text, functions(tree).map((f) => [f.body.start + (f.body.type === "BlockStatement" ? 1 : 0), "/* a comment a tool left */ "])),
  "a parameter more for every function": (text, tree) => withInserts(text, functions(tree).flatMap((f) => {
    if (f.params.some((p) => p.type === "RestElement")) return [];
    if (f.params.length) return [[f.params.at(-1).end, ", notUsed$ = 0"]];
    const open = text.indexOf("(", f.type === "ArrowFunctionExpression" ? f.start : f.id.end);
    return [[open + 1, "notUsed$ = 0"]];
  })),
  "every maker renamed": (text, tree) => withInserts(text, [...namesIn(tree)].filter((n) => makers.includes(n.name)).map((n) => [n.end, "$moved"])),
};
// esbuild's output has its exports at the end, or is CommonJS: run as it is said
function runAny(text) {
  if (/\bmodule\.exports\b/.test(text)) {
    const module = { exports: {} };
    new Function("module", "exports", text)(module, module.exports);
    return module.exports.deviceKey;
  }
  const tree = parse(text, { sourceType: "module" }).program;
  const lists = tree.body.filter((s) => s.type === "ExportNamedDeclaration" && !s.declaration);
  if (!lists.length) return run(text, tree.body.filter((s) => s.type === "ExportNamedDeclaration").map((s) => s.start));
  assert.equal(lists.length, 1, "one list of exports");
  const key = lists[0].specifiers.find((specifier) => specifier.exported.name === "deviceKey").local.name;
  return new Function(`${text.slice(0, lists[0].start)}\nreturn ${key};${text.slice(lists[0].end)}`)();
}
// the control: the same file with one maker's own source hashed too, as every maker's was until T366
const hashed = "...engineShaders(),";
assert.equal(source.split(hashed).length, 2, "deviceKey() names engineShaders() once");
const control = source.replace(hashed, `${hashed} String(fusedMatVec),`);
const controlTree = parse(control, { sourceType: "module" }).program;
const controlKeys = keysOf(runAny(control), false);
assert.notDeepEqual(controlKeys, KEYS.plain, "the control hashes more");
for (const [name, change] of Object.entries(CHANGES)) {
  const changed = change(source, program);
  assert.notEqual(changed, source, `${name}: the source is another`);
  const deviceKey = runAny(changed);
  assert.deepEqual(keysOf(deviceKey, false), KEYS.plain, `${name}: the key moved`);
  assert.deepEqual(keysOf(deviceKey, true), KEYS.ternary, `${name}: a ternary model's key moved`);
  assert.notDeepEqual(keysOf(runAny(change(control, controlTree)), false), controlKeys, `${name}: the control's key did not move (the change does nothing a function's text shows?)`);
}
// and all of them at once, each on the last one's text
let all = source;
for (const name of ["a comment in every function", "a parameter more for every function", "every maker renamed", "minified by esbuild (white space, names, syntax)"]) {
  all = CHANGES[name](all, parse(all, { sourceType: "module" }).program);
}
assert.deepEqual(keysOf(runAny(all), true), KEYS.ternary, "all the changes at once: the key moved");
console.log(`device-key-check: the key is the same with the source ${Object.keys(CHANGES).join("; ")}; and all at once (${source.length} characters to ${all.length}). ` +
  "A copy that hashes a maker's own source moved with every one of them");
console.log("device-key-check: ok");
