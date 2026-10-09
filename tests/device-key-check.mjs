// tests/device-key-check.mjs (T366): what public/shaders.js's deviceKey() is made of.
//
// The key says whether what a device measured (the forms it chose, T148; that the CPU was faster than a model on the GPU
// alone, T156) still holds. It is to change when a shader the engine runs changes by a character, and not otherwise:
// not when the file is formatted, minified or bundled, a local or a maker renamed, a parameter or a comment added.
// (Until T366 the key hashed String(maker) of four makers, and four shaders gpu.js runs were not in it.) Node only:
//
//   node tests/device-key-check.mjs
//
// T351: shaders.js is a window over the modules of public/shaders/ now. What is read, changed and run here is those
// modules as the one module they were (tests/shaders-source.mjs's oneSource(): their statements in the window's order,
// without the lines that hand names from one to another), and "the file" below is that text. That it is the modules'
// own behaviour is held where the check starts: run as it is, it gives the keys of the window imported for real.
// (Whether each module takes the names it uses is tests/shaders-modules-check.mjs's.)
//
// (1) Every piece of text of the engine's shaders reaches the hash. A piece is a string, a number or a run of a
//     template's text in the source of public/shaders.js's modules; the engine's shaders are every text and maker that
//     public/gpu.js names (wgsl.<name>) and the prompt's tiles, with all they are made of (the names their source
//     reaches). Each piece is changed by one character, one at a time, the changed file run, and the key of one of
//     eight made-up devices must differ: so every branch of every maker is walked by what deviceKey() makes of it.
//     A piece of a ternary model's shaders must leave the key of every other model alone (T232); a piece of anything
//     else in the file (the benchmark's shaders, the JavaScript the checks compare with) must leave every key alone.
//     (The review: what gpu.js runs is found by name from its syntax tree, whatever the object is called and however the
//     name is taken (a property, a destructuring), and every place gpu.js compiles a shader is listed, so that a new one
//     is looked at; and what a maker's numbers may do is held by its syntax tree: a number is written into the text and
//     nothing else, since the key walks one set of them.)
// (2) The key is the same when the source is not: esbuild's reprint, its minifier (white space, names, syntax), its
//     wrapping as another format of module, rolldown's bundle (Vite 8's, the site's), a comment in every function, a
//     parameter more for every function, every maker renamed. A copy of the file that hashes one maker's source as well
//     (what the key did) is the control: every one of these changes moves that copy's key.
// (3) The key moves with what the texts are cut at, and with the two language features of the browser that choose shaders.
// The parser is @babel/parser, the minifier esbuild and the bundler rolldown, all of which Astro's packages bring.
import assert from "node:assert/strict";
import fs from "node:fs";
import { parse } from "@babel/parser";
import { transformSync } from "esbuild";
import { rolldown } from "rolldown";
import { oneSource } from "./shaders-source.mjs";

const read = (file) => fs.readFileSync(new URL(`../public/${file}`, import.meta.url), "utf8");
const source = oneSource();
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
assert.deepEqual(keysOf(run(source), false), KEYS.plain, "the modules run as one function's body give the keys of shaders.js imported");
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
// What gpu.js takes of shaders.js, by name and not by what it calls the module: the property of any member access, the key of
// any destructuring, a subscript that is a string. (A match for `wgsl.<name>` found a name taken any other way as no shader
// of gpu.js: a destructured shader was in no key and this check said ok.) A name that is also something else in gpu.js
// (a property of the same spelling) is taken for a shader and must then reach the hash: it fails loudly, never quietly.
// T352: gpu.js is a window over the modules of public/gpu/ now, and the worker is all of those files: each is read, and
// what is said of "gpu.js" below is said of them together (a shader named or compiled in any of them). Every file of the
// folder is read, whether or not the window asks for it (that it asks for them all is tests/gpu-modules-check.mjs's).
const gpuFiles = ["gpu.js", ...fs.readdirSync(new URL("../public/gpu/", import.meta.url)).filter((name) => name.endsWith(".js")).sort().map((name) => `gpu/${name}`)];
assert.ok(gpuFiles.length > 1, "public/gpu/ holds the worker's modules");
const gpuNodes = gpuFiles.flatMap((file) => {
  const text = read(file);
  return [...under(parse(text, { sourceType: "module" }).program)].map((n) => ({ n, file, text }));
});
const taken = new Set();
for (const { n } of gpuNodes) {
  if ((n.type === "MemberExpression" || n.type === "OptionalMemberExpression") && !n.computed) taken.add(n.property.name);
  if ((n.type === "MemberExpression" || n.type === "OptionalMemberExpression") && n.computed && n.property.type === "StringLiteral") taken.add(n.property.value);
  if (n.type === "ObjectPattern") for (const property of n.properties) if (property.type === "ObjectProperty" && !property.computed) taken.add(property.key.name ?? property.key.value);
}
const used = [...taken].filter((name) => name in real).sort();
// gpu.js compiles a shader in one place (pipelineOf) and hands it the code at the calls below. A new call, or one whose code
// comes from somewhere else, is looked at here: every shader it can compile must be one of those the key hashes
const compiled = gpuNodes.filter(({ n }) => n.type === "CallExpression" && n.callee.name === "pipelineOf").map(({ n, text }) => text.slice(n.arguments[1].start, n.arguments[1].end));
assert.equal(gpuNodes.filter(({ n }) => n.type === "Identifier" && n.name === "createShaderModule").length, 1, "gpu.js makes shader modules in one place");
// (T352: and under that one name: a module that took pipelineOf as another name, or made a pipeline itself, would compile unseen)
assert.equal(gpuNodes.filter(({ n }) => n.type === "Identifier" && /^createComputePipeline(Async)?$/.test(n.name)).length, 1, "gpu.js makes pipelines in one place");
for (const { n, file } of gpuNodes) {
  if (n.type === "ObjectProperty" && !n.computed && (n.key.name ?? n.key.value) === "pipelineOf") assert.ok(n.shorthand, `${file} takes pipelineOf under another name`);
}
assert.deepEqual([...new Set(compiled)].sort(), [
  "code",  // the small steps' (RMSNORM or LAYER_NORM, HEAD_NORM, ADD, ROPE, SWIGLU or GELU, QUANTIZE) and a token's layer, tokenCodes' (the fused matrices, NORM_QUANTIZE)
  "embedCode",  // EMBED or EMBED_TERNARY
  "form.code",  // a prompt's tile (devicePromptForms)
  "kept.code",  // the same, the remembered one
  "rowsCode",  // EMBED_ROWS or EMBED_ROWS_TERNARY
  "wgsl.SAMPLE", "wgsl.TAKE_OUTLIERS", "wgsl.TERNARY_COLUMNS", "wgsl.TOKEN_ROPE", "wgsl.WIDEN_SIX",
  "wgsl.flashTile(shape)", "wgsl.flashVec(a.shape)", "wgsl.flashVecReduce(a.shape)",
].sort(), "a place where gpu.js compiles a shader that this check does not know: is every shader it can compile in the key (engineShaders(), ternaryShaders(), the tiles)?");
for (const { n, file } of gpuNodes) {
  const text = n.type === "TemplateElement" ? n.value.raw : n.type === "StringLiteral" ? n.value : "";
  assert.ok(!/@compute|@workgroup_size|\bfn main\b|@group\(/.test(text), `${file} writes WGSL itself (at ${n.start}): the key hashes the texts of shaders.js alone`);
}
const shaders = [...new Set([...used.filter((name) => ![...KEY_ITSELF, ...FORMS].includes(name) && isShader(name)), ...TILES])];
assert.ok(shaders.length >= 27, `the shaders gpu.js runs: ${shaders.length} found (27 when this was written)`);
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

// ---- (1b) the arguments of the makers: the key walks every choice a text branches on and one set of numbers. A piece of
// text that only some arguments reach is found above (it did not move the key); what is not text is found here.
// (a) the key calls each maker with the arguments the maker reads, no fewer: a parameter added to a maker and not to
// engineShaders() or ternaryShaders() is an argument the key does not walk. (b) The numbers the key gives (a literal in its
// call) are only written into a text: each use of one in the maker is inside a template's ${}, in a sum or a product at
// most, or is an argument to a function that does the same with it. A number that is compared, branched on or looked up
// chooses a text the key may not make (`${wgSize > kvTile ? STEP : ""}` has no piece of text of its own to move the key).
const functionOf = (name) => { const d = declared.get(name); return d?.type === "FunctionDeclaration" ? d : d?.init; };
const paramsOf = (fn) => fn.params.map((p) => (p.type === "ObjectPattern" ? p.properties.map((q) => q.key.name) : p.type === "AssignmentPattern" ? p.left.name : p.name));
function* withParents(node, parents = []) {
  if (!node || typeof node.type !== "string") return;
  yield [node, parents];
  for (const [key, value] of Object.entries(node)) {
    if (key === "loc" || /Comments$/.test(key)) continue;
    for (const child of Array.isArray(value) ? value : [value]) if (child && typeof child === "object") yield* withParents(child, [...parents, node]);
  }
}
/** whether a parameter is only written into a text: every use of its name in the function is in a template's ${} (through
 * + - * / % only) or an argument at a position another declared function treats the same way */
function onlyWritten(fn, param, seen = new Set()) {
  for (const [node, parents] of withParents(fn.body)) {
    if (node.type !== "Identifier" || node.name !== param) continue;
    const up = parents.at(-1);
    if ((up.type === "MemberExpression" && up.property === node && !up.computed) || (up.type === "ObjectProperty" && up.key === node && !up.shorthand)) continue;
    let at = parents.length - 1, child = node;
    while (at >= 0 && parents[at].type === "BinaryExpression" && "+-*/%".includes(parents[at].operator) || parents[at]?.type === "ParenthesizedExpression") child = parents[at--];
    const parent = parents[at];
    if (parent?.type === "TemplateLiteral" && parent.expressions.includes(child)) continue;
    if (parent?.type === "CallExpression" && parent.callee.type === "Identifier" && functionOf(parent.callee.name) && parent.arguments.includes(child)) {
      const callee = functionOf(parent.callee.name), position = parent.arguments.indexOf(child), name = paramsOf(callee)[position];
      const key = `${parent.callee.name}:${position}`;
      if (typeof name === "string" && !seen.has(key) && onlyWritten(callee, name, new Set([...seen, key]))) continue;
    }
    return false;
  }
  return true;
}
const walked = new Map();  // maker -> the properties of the object the key gives it
for (const given of ["engineShaders", "ternaryShaders"]) {
  for (const [call] of withParents(declared.get(given))) {
    if (call.type !== "CallExpression" || call.callee.type !== "Identifier" || !functionOf(call.callee.name) || !/Function/.test(functionOf(call.callee.name).type)) continue;
    const maker = functionOf(call.callee.name), name = call.callee.name;
    const first = call.arguments[0];
    if (first?.type !== "ObjectExpression") { assert.fail(`the key calls ${name} with something other than an object of arguments: this check reads those`); }
    const kept = new Map(first.properties.map((p) => [p.key.name, p.shorthand || p.value.type !== "NumericLiteral" ? "choice" : "number"]));
    assert.deepEqual([...kept.keys()].sort(), paramsOf(maker)[0].slice().sort(), `${name}'s parameters are not the arguments the key gives it`);
    for (const [property, kind] of kept) if (kind === "number") assert.ok(onlyWritten(maker, property), `${name}'s ${property}: a number the key gives once, used for more than writing (a comparison or a branch chooses a text the key does not make)`);
    walked.set(name, kept);
  }
}
assert.ok(walked.size >= 6, `the makers the key calls: ${[...walked.keys()]}`);
console.log(`device-key-check: the key gives ${[...walked].map(([name, kept]) => `${name}(${[...kept].map(([k, v]) => v === "number" ? `${k}#` : k).join(", ")})`).join("; ")} (# a number, only written)`);

// ---- (2) the same key from another source
const functions = (tree) => [...under(tree)].filter((n) => n.type === "ArrowFunctionExpression" || n.type === "FunctionDeclaration" || n.type === "FunctionExpression");
const makers = [...declared].filter(([name, d]) => name !== "deviceKey" && (d.type === "FunctionDeclaration" || /Function/.test(d.init?.type ?? ""))).map(([name]) => name);
async function rolldownOf(text, minify) {
  const bundle = await rolldown({ input: "virtual:shaders", treeshake: true, logLevel: "silent",
    plugins: [{ name: "virtual", resolveId: (id) => (id === "virtual:shaders" ? "\0shaders" : null), load: (id) => (id === "\0shaders" ? text : null) }] });
  const { output } = await bundle.generate({ format: "esm", minify });
  return output[0].code;
}
const CHANGES = {
  // a printer's own white space, quotes and line breaks
  "reprinted by esbuild": (text) => transformSync(text, { format: "esm" }).code,
  "minified by esbuild (white space, names, syntax)": (text) => transformSync(text, { format: "esm", minify: true }).code,
  "wrapped as another format of module (esbuild's cjs, minified)": (text) => transformSync(text, { format: "cjs", minify: true }).code,
  // what the site's build does (Vite 8 bundles with rolldown and minifies with oxc, not esbuild): the file as a module of its own
  "bundled and minified by rolldown (Vite 8's)": (text) => rolldownOf(text, true),
  "bundled by rolldown, not minified": (text) => rolldownOf(text, false),
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
  const changed = await change(source, program);
  assert.notEqual(changed, source, `${name}: the source is another`);
  const deviceKey = runAny(changed);
  assert.deepEqual(keysOf(deviceKey, false), KEYS.plain, `${name}: the key moved`);
  assert.deepEqual(keysOf(deviceKey, true), KEYS.ternary, `${name}: a ternary model's key moved`);
  assert.notDeepEqual(keysOf(runAny(await change(control, controlTree)), false), controlKeys, `${name}: the control's key did not move (the change does nothing a function's text shows?)`);
}
// and all of them at once, each on the last one's text
let all = source;
for (const name of ["a comment in every function", "a parameter more for every function", "every maker renamed", "minified by esbuild (white space, names, syntax)", "bundled and minified by rolldown (Vite 8's)"]) {
  all = await CHANGES[name](all, parse(all, { sourceType: "module" }).program);
}
assert.deepEqual(keysOf(runAny(all), true), KEYS.ternary, "all the changes at once: the key moved");
console.log(`device-key-check: the key is the same with the source ${Object.keys(CHANGES).join("; ")}; and all at once (${source.length} characters to ${all.length}). ` +
  "A copy that hashes a maker's own source moved with every one of them");

// ---- (3) what the key is cut at, and the browser's language features
// a character moved from the end of one text to the start of the next leaves the concatenation as it was
const moved = source.replace("[RMSNORM, HEAD_NORM, ADD,", "[RMSNORM.slice(0, -1), RMSNORM.slice(-1) + HEAD_NORM, ADD,");
assert.notEqual(moved, source, "the engine's list of texts starts as this check expects");
assert.notDeepEqual(keysOf(runAny(moved), false), KEYS.plain, "a character moved from one text to the next did not move the key");
// the two features of the browser's WGSL that choose which shaders a device makes (packed int8 dot, subgroup_id)
{
  const was = globalThis.navigator;
  const features = (...names) => Object.defineProperty(globalThis, "navigator", { configurable: true, value: { ...was, gpu: { wgslLanguageFeatures: new Set(names) } } });
  const keys = [[], ["packed_4x8_integer_dot_product"], ["subgroup_id"], ["packed_4x8_integer_dot_product", "subgroup_id"], ["a_feature_no_shader_reads"]].map((names) => {
    features(...names);
    return [keysOf(real.deviceKey, false)[0], keysOf(real.deviceKey, true)[0]];
  });
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: was });
  assert.equal(new Set(keys.map(([plain]) => plain)).size, 4, "packed and subgroup_id each move the key (a feature that no shader reads is the same as none)");
  assert.equal(keys[0][0], keys[4][0], "a language feature no shader reads moved the key");
  assert.equal(new Set(keys.map(([, ternary]) => ternary)).size, 4, "a ternary model's key too");
}
console.log("device-key-check: a character moved from one text to the next moves the key; so do packed int8 dot and subgroup_id");
console.log("device-key-check: ok");
