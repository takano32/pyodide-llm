// tests/device-key-check.ts (T.366): what src/runtime/shaders.js's deviceKey() is made of.
//
// The key says whether what a device measured (the forms it chose, T.148; that the CPU was faster than a model on the GPU
// alone, T.156) still holds. It is to change when a shader the engine runs changes by a character, and not otherwise:
// not when the file is formatted, minified or bundled, a local or a maker renamed, a parameter or a comment added.
// (Until T.366 the key hashed String(maker) of four makers, and four shaders gpu.js runs were not in it.) Node only:
//
//   node tests/device-key-check.ts
//
// T.351: shaders.js is a window over the modules of src/runtime/shaders/ now. What is read, changed and run here is those
// modules as the one module they were (tests/shaders-source.ts's oneSource(): their statements in the window's order,
// without the lines that hand names from one to another), and "the file" below is that text. That it is the modules'
// own behaviour is held where the check starts: run as it is, it gives the keys of the window imported for real.
// (Whether each module takes the names it uses is the type check's: tests/types.ts.)
//
// (1) Every piece of text of the engine's shaders reaches the hash. A piece is a string, a number or a run of a
//     template's text in the source of src/runtime/shaders.js's modules; the engine's shaders are every text and maker that
//     src/runtime/gpu.js's modules import from the parts of shaders/ and the prompt's tiles, with all they are made of (the names their source
//     reaches). Each piece is changed by one character, one at a time, the changed file run, and the key of one of
//     eight made-up devices must differ: so every branch of every maker is walked by what deviceKey() makes of it.
//     A piece of a ternary model's shaders must leave the key of every other model alone (T.232); a piece of anything
//     else in the file (the benchmark's shaders, the JavaScript the checks compare with) must leave every key alone.
//     (The review: every place gpu.js compiles a shader is listed, so that a new one is looked at; and what a maker's
//     numbers may do is held by its syntax tree: a number is written into the text and nothing else, since the key
//     walks one set of them.)
//     T.379.1: what gpu.js runs is the catalogue's (shaders/engine.ts): its modules take a shader as ENGINE.<name> and
//     import none of a part, and the key hashes the catalogue's rows. So the shaders here are the catalogue's names,
//     those of a ternary model alone are its second list's, and the arguments the key gives a maker are its rows'.
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
import { oneSource } from "./shaders-source.ts";
import { runtime, runtimeUrl } from "./tree.ts";
import { typed, untyped } from "./untyped.ts";

const read = (file) => fs.readFileSync(runtime(file), "utf8");
const source = oneSource();
const program = parse(source, { sourceType: "module" }).program;

// ---- the made-up devices: the features that choose the prompt's tiles, and whether the model is of ternary weights
const DEVICES = [false, true].flatMap((half) => [false, true].map((subgroups) => ({ info: { vendor: "v", architecture: "a", device: "d", description: "x" },
  features: new Set([...(half ? ["shader-f16"] : []), ...(subgroups ? ["subgroups"] : [])]),
  limits: { maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 } })));
Object.defineProperty(globalThis, "navigator", { configurable: true,
  value: { userAgent: "UA", gpu: { wgslLanguageFeatures: new Set(["packed_4x8_integer_dot_product"]) } } });
// (T.547.2.3: the key takes a model's own shaders by name: none, a ternary model's or a Q4_K one's)
const OWN = { plain: null, ternary: "ternary", q4k: "blocks" };
const keysOf = (deviceKey, which) => DEVICES.map((device) => deviceKey(device, device, OWN[which]));

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
const real = (await import(runtimeUrl("shaders.js")));
const KEYS = { plain: keysOf(real.deviceKey, "plain"), ternary: keysOf(real.deviceKey, "ternary"), q4k: keysOf(real.deviceKey, "q4k") };
assert.deepEqual(keysOf(run(source), "plain"), KEYS.plain, "the modules run as one function's body give the keys of shaders.js imported");
assert.equal(new Set([...KEYS.plain, ...KEYS.ternary, ...KEYS.q4k]).size, 3 * DEVICES.length, "a key for each device, for its ternary models and for its Q4_K ones");

// ---- (1) what the engine runs, and that each piece of it reaches the hash
// the prompt's tiles (gpu.js takes them through devicePromptForms), and the makers' own
const TILES = ["regTile", "tfjsTile", "dp4a"];
// the key's own lines (their strings choose what is made: no text of a shader), and what names the prompt's forms and
// says why one is not made (a form's name is hashed with its text: the tiles' sizes are in it)
// (T.379.1: and the catalogue's own: its rows, what makes a row, and the lists of the choices they walk)
const KEY_ITSELF = ["deviceKey", "ENGINE_SHADERS", "TERNARY_SHADERS", "BLOCK_SHADERS", "ENGINE", "EITHER", "texts", "made", "FUSED_INPUTS", "FUSED_OUTPUTS"], FORMS = ["devicePromptForms"];
for (const name of KEY_ITSELF) assert.ok(declared.has(name), `${name} is no declaration of the shaders' modules any more: this check's list of the key's own lines`);
// the catalogue (T.379.1): the rows the key hashes, [name, the arguments of a maker's row] each
const rowsOf = (rows) => rows.map(({ name, given }) => ({ name, given }));
const ROWS = { engine: rowsOf(real.ENGINE_SHADERS), ternary: rowsOf(real.TERNARY_SHADERS) };
const named = (rows) => [...new Set(rows.map(({ name }) => name))];
// the shaders of a model of ternary weights alone (T.232): no other model's key holds them
const TERNARY = named(ROWS.ternary).filter((name) => !named(ROWS.engine).includes(name));
// (and the makers of every model's shaders that have branches for ternary weights, and what only those branches write:
// TERNARY_PACKED is a piece of the tiles' ternary text and of DP4A_WEIGHTS' ternary row)
const TERNARY_BRANCHES = ["dp4a", "TERNARY_PACKED", "DP4A_WEIGHTS"];
// (T.547.2.3: and those for Q4_K blocks: dp4a's branch and (T.547.2.4) DP4A_WEIGHTS' row, and what only they write)
// (T.547.3.3: of every type of BLOCK_READS: what they share and each one's reading)
const Q4K_BRANCHES = ["dp4a", "BLOCK_COMMON", "BLOCK_READS", "blockLines", "DP4A_WEIGHTS", "blockRow", "blockRows", "fusedDp4aMatVec", "K_BITS", "IQ_GRID", "IQ_SEVEN", "IQ_WORDS", "IQ_BIND", "IQ_SIGNS", "IQ_EIGHT", "IQ_FOUR", "IQ_SHIFTED", "IQ_NL", "BLK_F16", "BLK_READ", "BLK_DOT", "embedBlocks"];
const isShader = (name) => typeof real[name] === "string"
  || (typeof real[name] === "function" && [...reached([name])].some((part) => pieces(part).some((piece) => typeof piece.says === "string" && piece.says.includes("@compute"))));
// What gpu.js takes of the shaders (T.379.1): the catalogue's, as ENGINE.<name>, and no shader by an import. What its
// modules import from the parts of shaders/ is by name, each line read: a namespace (`import * as`), the window
// (shaders.js: every name, /benchmark/'s too) or a part imported when it is used (`import("…")`) would hand names round
// that no line names, and is refused here; and a name imported so is no shader (a text with @compute, or a maker of
// one): that is the catalogue's to give. ENGINE itself is read by its properties alone (ENGINE.SAMPLE): handed on,
// taken apart or subscripted, a shader would be taken that no line names. (T.367.5 to T.379.1: the import lines were
// the list of the worker's shaders. Until T.367.5 the worker handed the window's module round as m.wgsl.)
// T.352: gpu.js is a window over the modules of src/runtime/gpu/ now, and the worker is all of those files: each is read, and
// what is said of "gpu.js" below is said of them together (a shader named or compiled in any of them). Every file of the
// folder is read, whether or not the window asks for it (that it asks for them all is tests/gpu-modules-check.ts's).
// (T.368.6: the worker is TypeScript: its files in either spelling, each parsed with its types blanked, so that a type's
// import of a part is no name taken and a place in the text is its place in the source)
const gpuFiles = ["gpu.js", ...fs.readdirSync(runtime("gpu")).filter((name) => name.endsWith(".js") || typed(name)).sort().map((name) => `gpu/${name}`)];
assert.ok(gpuFiles.length > 1, "gpu/ holds the worker's modules");
const gpuNodes = gpuFiles.flatMap((file) => {
  const text = untyped(read(file), runtime(file));
  return [...under(parse(text, { sourceType: "module" }).program)].map((n) => ({ n, file, text }));
});
const taken = new Set();
for (const { n, file, text } of gpuNodes) {
  const asked = n.type === "ImportDeclaration" ? n.source.value : n.type === "ImportExpression" ? text.slice(n.source.start, n.source.end)
    : n.type === "CallExpression" && n.callee.type === "Import" ? text.slice(n.arguments[0].start, n.arguments[0].end) : null;
  if (asked === null || !/shaders/.test(asked)) continue;
  assert.ok(n.type === "ImportDeclaration" && /^\.\.?\/shaders\/\w+\.[jt]s$/.test(asked), `${file} asks for ${asked}: a module of the GPU's worker takes the shaders it uses by name, from the parts of shaders/ (import { NAME } from "../shaders/<part>.js")`);
  for (const specifier of n.specifiers) {
    assert.ok(specifier.type === "ImportSpecifier" && specifier.imported.name === specifier.local.name, `${file} takes ${text.slice(specifier.start, specifier.end)} of ${asked}: a plain name, under its own (a namespace would hand shaders round unseen)`);
    assert.ok(specifier.imported.name in real, `${file} takes ${specifier.imported.name} of ${asked}, which shaders.js (the window) does not export`);
    assert.ok([...KEY_ITSELF, ...FORMS].includes(specifier.imported.name) || !isShader(specifier.imported.name), `${file} imports the shader ${specifier.imported.name} of ${asked}: the worker takes a shader from the catalogue (ENGINE.${specifier.imported.name}, shaders/engine.ts), which the device's key hashes`);
    taken.add(specifier.imported.name);
  }
}
// (the worker's modules with their parents: how each mention of ENGINE is used)
const taking = new Set();
for (const file of gpuFiles) {
  for (const [n, parents] of withParents(parse(untyped(read(file), runtime(file)), { sourceType: "module" }).program)) {
    if (n.type !== "Identifier" || n.name !== "ENGINE") continue;
    const up = parents.at(-1);
    if (up.type === "ImportSpecifier") continue;
    assert.ok(up.type === "MemberExpression" && up.object === n && !up.computed, `${file} uses ENGINE otherwise than as ENGINE.<name> (at ${n.start}): the shaders the worker runs are read off these lines`);
    assert.ok(up.property.name in real.ENGINE, `${file} takes ENGINE.${up.property.name}, which the catalogue has no row of`);
    taking.add(up.property.name);
  }
}
const used = [...taking].sort();
// every row is a shader the worker runs: a row no module takes is in every device's key for nothing (TERNARY_PACKED,
// a piece of text, was one from T.232 until T.379.2 moved the keys anyway)
assert.deepEqual(Object.keys(real.ENGINE).filter((name) => !taking.has(name)), [], "the catalogue's rows that no module of gpu.js takes");
// gpu.js compiles a shader in one place (pipelineOf) and hands it the code at the calls below. A new call, or one whose code
// comes from somewhere else, is looked at here: every shader it can compile must be one of those the key hashes
const compiled = gpuNodes.filter(({ n }) => n.type === "CallExpression" && n.callee.name === "pipelineOf").map(({ n, text }) => text.slice(n.arguments[1].start, n.arguments[1].end));
assert.equal(gpuNodes.filter(({ n }) => n.type === "Identifier" && n.name === "createShaderModule").length, 1, "gpu.js makes shader modules in one place");
// (T.352: and under that one name: a module that took pipelineOf as another name, or made a pipeline itself, would compile unseen)
assert.equal(gpuNodes.filter(({ n }) => n.type === "Identifier" && /^createComputePipeline(Async)?$/.test(n.name)).length, 1, "gpu.js makes pipelines in one place");
for (const { n, file } of gpuNodes) {
  if (n.type === "ObjectProperty" && !n.computed && (n.key.name ?? n.key.value) === "pipelineOf") assert.ok(n.shorthand, `${file} takes pipelineOf under another name`);
}
assert.deepEqual([...new Set(compiled)].sort(), [
  "code",  // the small steps' (RMSNORM or LAYER_NORM, HEAD_NORM, ADD, ROPE, SWIGLU or GELU, QUANTIZE) and a token's layer, tokenCodes' (the fused matrices, NORM_QUANTIZE)
  "embedCode",  // EMBED or EMBED_TERNARY
  "form.code",  // a prompt's tile (devicePromptForms; the remembered one too: forms.ts's compiled())
  "form.blocks [packing]",  // (T.547.2.3) the same form's text of a type of llama.cpp's blocks (T.547.3.3), a tile of devicePromptForms too
  "rowsCode",  // EMBED_ROWS or EMBED_ROWS_TERNARY
  "ENGINE.SAMPLE", "ENGINE.TAKE_OUTLIERS", "ENGINE.TERNARY_COLUMNS", "ENGINE.TOKEN_ROPE", "ENGINE.WIDEN_SIX",
  "ENGINE.OUTPUT_GATE",  // (T.271.2: a Qwen3.5's gate of the attention's output)
  "ENGINE.flashTile(shape)", "ENGINE.flashVec(a.shape)", "ENGINE.flashVecReduce(a.shape)",
  'ENGINE.fusedDp4aMatVec({ output: "write", weights: type })',  // (T.547.3.6) a token's matrix of a type of blocks, checked on its own (tokenforms.ts's checkTokenBlocks)
].sort(), "a place where gpu.js compiles a shader that this check does not know: is every shader it can compile of the catalogue (ENGINE.<name>) or a tile (devicePromptForms)?");
for (const { n, file } of gpuNodes) {
  const text = n.type === "TemplateElement" ? n.value.raw : n.type === "StringLiteral" ? n.value : "";
  assert.ok(!/@compute|@workgroup_size|\bfn main\b|@group\(/.test(text), `${file} writes WGSL itself (at ${n.start}): the key hashes the texts of shaders.js alone`);
}
for (const name of used) assert.ok(isShader(name), `ENGINE.${name} is no shader: a text with @compute or a maker of one`);
const shaders = [...new Set([...used, ...TILES])];
assert.ok(shaders.length >= 26, `the shaders gpu.js runs: ${shaders.length} found (26 when this was written: 23 of the catalogue and the 3 makers of the tiles)`);
const others = shaders.filter((name) => !TERNARY.includes(name));
const engine = reached(shaders), general = reached(others);
const itself = new Set([...KEY_ITSELF, ...reached(FORMS)].filter((name) => !engine.has(name)));

// (the devices one at a time, those with most of the forms first: most pieces move the first one's key)
const moves = (deviceKey, which) => DEVICES.toReversed().some((device) => deviceKey(device, device, OWN[which]) !== KEYS[which][DEVICES.indexOf(device)]);
const changedBy = (pieces) => run(withInserts(source, pieces.map((piece) => [piece.at, piece.insert])),
  exportWords.map((at) => at + pieces.filter((piece) => piece.at < at).reduce((sum, piece) => sum + piece.insert.length, 0)));
const counted = { engine: 0, ternary: 0, branches: 0, q4k: 0, numbers: 0, looked: 0, apart: [] };
const LOOKED_UP = ["BLOCK_PACKINGS"];
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
    // (T.547.3.3: the names of the types of llama.cpp's blocks, which the makers look their readings up by: one changed is
    // no reading at all, and the shaders are not made: loud, as no piece of a text is)
    let deviceKey;
    try {
      deviceKey = changedBy([piece]);
    } catch (error) {
      if (!LOOKED_UP.includes(name)) throw error;
      counted.looked += 1;
      continue;
    }
    if (ternaryOnly) {
      assert.ok(moves(deviceKey, "ternary"), `${said}, of a ternary model's shaders, did not reach the hash of a ternary model's key`);
      assert.ok(!moves(deviceKey, "plain"), `${said}, of a ternary model's shaders, moved the key of the other models`);
      counted.ternary += 1;
    } else if (moves(deviceKey, "plain")) {
      counted.engine += 1;
    } else if (TERNARY_BRANCHES.includes(name) && moves(deviceKey, "ternary")) {
      // (the branches for ternary weights of a maker every model's shaders come from)
      counted.branches += 1;
    } else {
      // (T.547.2.3: the branches for Q4_K blocks: a Q4_K model's key alone)
      assert.ok(Q4K_BRANCHES.includes(name) && moves(deviceKey, "q4k") && !moves(deviceKey, "ternary"),
        `${said}, of the engine's shaders, did not reach the hash: a branch of a maker that deviceKey() does not walk?`);
      counted.q4k += 1;
    }
    if (piece.insert === "1") counted.numbers += 1;
  }
}
// the rest of the file (the benchmark's shaders, the JavaScript the checks compare with): all changed at once
const apart = changedBy(counted.apart);
assert.ok(!moves(apart, "plain") && !moves(apart, "ternary") && !moves(apart, "q4k"), "what gpu.js does not run moved the key");
assert.ok(counted.engine > 100 && counted.ternary > 10 && counted.branches > 0 && counted.q4k > 10 && counted.apart.length > 50, `too few pieces: ${JSON.stringify({ ...counted, apart: counted.apart.length })}`);
console.log(`device-key-check: ${shaders.length} shaders of gpu.js (${shaders.join(", ")}), made of ${engine.size} declarations`);
console.log(`device-key-check: one character more in each of ${counted.engine + counted.ternary + counted.branches} pieces of them moved the key ` +
  `(${counted.numbers} of the pieces are numbers; ${counted.ternary} pieces of the ternary shaders and ${counted.branches} of the branches for ternary weights moved a ternary model's alone, ` +
  `${counted.q4k} of those for Q4_K blocks a Q4_K model's alone), ` +
  `and one more in all ${counted.apart.length} pieces of the rest of the file did not`);

// ---- (1b) the arguments of the makers: the key walks every choice a text branches on and one set of numbers. A piece of
// text that only some arguments reach is found above (it did not move the key); what is not text is found here.
// (a) the key calls each maker with the arguments the maker reads, no fewer: a parameter added to a maker and not to
// the catalogue's rows is an argument the key does not walk. (b) The numbers the key gives (a literal in its
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
// (T.379.1: what the key gives a maker is the catalogue's rows of it: every row of a maker gives it all the maker reads,
// and what is a number in one row is the same number in all of them: one set of numbers)
for (const { name, given } of [...ROWS.engine, ...ROWS.ternary]) {
  if (!given) { assert.equal(typeof real.ENGINE[name], "string", `${name}: a row without arguments is a text`); continue; }
  const maker = functionOf(name);
  assert.ok(maker && /Function/.test(maker.type), `${name}: a row with arguments is a maker's`);
  const kept = new Map(Object.entries(given).map(([property, value]) => [property, typeof value === "number" ? "number" : "choice"]));
  assert.deepEqual([...kept.keys()].sort(), paramsOf(maker)[0].slice().sort(), `${name}'s parameters are not the arguments the catalogue's row gives it (${JSON.stringify(given)})`);
  for (const [property, kind] of kept) {
    if (kind !== "number") continue;
    assert.ok(onlyWritten(maker, property), `${name}'s ${property}: a number the key gives once, used for more than writing (a comparison or a branch chooses a text the key does not make)`);
    const first = [...ROWS.engine, ...ROWS.ternary].find((row) => row.name === name).given[property];
    assert.equal(given[property], first, `${name}'s ${property}: a number, and another in another row (the key walks one set of numbers; a choice is no number)`);
  }
  assert.deepEqual(walked.get(name) ?? kept, kept, `${name}: the rows give it arguments of different kinds`);
  walked.set(name, kept);
}
assert.ok(walked.size >= 5, `the makers the key calls: ${[...walked.keys()]}`);
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
  // (a name at a place once: a shorthand property, { flashTile }, is a key and a value there)
  "every maker renamed": (text, tree) => withInserts(text, [...new Set([...namesIn(tree)].filter((n) => makers.includes(n.name)).map((n) => n.end))].map((end) => [end, "$moved"])),
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
// the control: the same file with one maker's own source hashed too, as every maker's was until T.366
const hashed = "...written(ENGINE_SHADERS),";
assert.equal(source.split(hashed).length, 2, "deviceKey() names the catalogue's rows of every model once");
const control = source.replace(hashed, `${hashed} String(fusedMatVec),`);
const controlTree = parse(control, { sourceType: "module" }).program;
const controlKeys = keysOf(runAny(control), "plain");
assert.notDeepEqual(controlKeys, KEYS.plain, "the control hashes more");
for (const [name, change] of Object.entries(CHANGES)) {
  const changed = await change(source, program);
  assert.notEqual(changed, source, `${name}: the source is another`);
  const deviceKey = runAny(changed);
  assert.deepEqual(keysOf(deviceKey, "plain"), KEYS.plain, `${name}: the key moved`);
  assert.deepEqual(keysOf(deviceKey, "ternary"), KEYS.ternary, `${name}: a ternary model's key moved`);
  assert.notDeepEqual(keysOf(runAny(await change(control, controlTree)), "plain"), controlKeys, `${name}: the control's key did not move (the change does nothing a function's text shows?)`);
}
// and all of them at once, each on the last one's text
let all = source;
for (const name of ["a comment in every function", "a parameter more for every function", "every maker renamed", "minified by esbuild (white space, names, syntax)", "bundled and minified by rolldown (Vite 8's)"]) {
  all = await CHANGES[name](all, parse(all, { sourceType: "module" }).program);
}
assert.deepEqual(keysOf(runAny(all), "ternary"), KEYS.ternary, "all the changes at once: the key moved");
console.log(`device-key-check: the key is the same with the source ${Object.keys(CHANGES).join("; ")}; and all at once (${source.length} characters to ${all.length}). ` +
  "A copy that hashes a maker's own source moved with every one of them");

// ---- (3) what the key is cut at, and the browser's language features
// a character moved from the end of one text to the start of the next leaves the concatenation as it was
const moved = source.replace("texts({ RMSNORM, HEAD_NORM, ADD,", "texts({ RMSNORM: RMSNORM.slice(0, -1), HEAD_NORM: RMSNORM.slice(-1) + HEAD_NORM, ADD,");
assert.notEqual(moved, source, "the engine's list of texts starts as this check expects");
assert.notDeepEqual(keysOf(runAny(moved), "plain"), KEYS.plain, "a character moved from one text to the next did not move the key");
// the two features of the browser's WGSL that choose which shaders a device makes (packed int8 dot, subgroup_id)
{
  const was = globalThis.navigator;
  const features = (...names) => Object.defineProperty(globalThis, "navigator", { configurable: true, value: { ...was, gpu: { wgslLanguageFeatures: new Set(names) } } });
  const keys = [[], ["packed_4x8_integer_dot_product"], ["subgroup_id"], ["packed_4x8_integer_dot_product", "subgroup_id"], ["a_feature_no_shader_reads"]].map((names) => {
    features(...names);
    return [keysOf(real.deviceKey, "plain")[0], keysOf(real.deviceKey, "ternary")[0]];
  });
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: was });
  assert.equal(new Set(keys.map(([plain]) => plain)).size, 4, "packed and subgroup_id each move the key (a feature that no shader reads is the same as none)");
  assert.equal(keys[0][0], keys[4][0], "a language feature no shader reads moved the key");
  assert.equal(new Set(keys.map(([, ternary]) => ternary)).size, 4, "a ternary model's key too");
}
console.log("device-key-check: a character moved from one text to the next moves the key; so do packed int8 dot and subgroup_id");
console.log("device-key-check: ok");
