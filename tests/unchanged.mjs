// unchanged.mjs (T346)
// The net under a refactoring: what the working tree does, held to what another commit does (main, by default).
// Nothing here knows how the files are divided; it asks both trees the same questions and compares the answers.
//
//   node tests/unchanged.mjs [--before <a commit, default origin/main>] [shaders models calls choices exports layouts python page bench gpuworker sizes]
//   (PYTHON=.venv/bin/python; in CI: tests.yml's extra="node tests/unchanged.mjs")
//
//   shaders  every export of public/shaders.js (a text as it is; T366: a maker of WGSL as the texts it makes, for every
//            choice its text branches on; another function as its source) and deviceKey() for 192 made-up adapters: a
//            device forgets the forms it measured when the key changes (AGENTS.md). The key holds the WGSL a device is
//            given and no function's source (tests/device-key-check.mjs holds that every piece of the engine's shaders
//            reaches it), so a maker may be formatted, moved or given a parameter and this still says "the same"
//   models   every export of src/models.js: the data as JSON, and what every exported function answers for every
//            entry of the list (a builder moved to another file leaves both as they were)
//   calls    tests/unchanged-calls.mjs: the plan Python hands forward.js and every call of a kernel, 150 hashes
//   layouts  (T357) tests/unchanged_layouts.py: what a tree says of a checkpoint's file over a grid of made-up headers, forms
//            and dtypes (18 families of the four layouts, five dtypes): layout(), Writer's places and checkpoint_size(),
//            checkpoint_dtype(), the plan Llama(external=) hands forward.js (the engine's own order of the tensors),
//            external_tensors() and conversion_plan(); what the converter reads of 23 made-up config.json files (the
//            header, the form, an LFM2's FFN); and the repetition penalty's window. The order of a file's tensors is
//            written in four places and the unit tests walk the branches of the few models they make: here all four
//            answer for the same grid, and the next one to move them (T359) is shown what moved
//   python   the unit tests under tests/unchanged_recorder.py: the checkpoints, options and tokenizers of every
//            conversion the tests make, the logits of every forward pass, the ids of every encode (two runs of pytest)
//   choices  (review) tests/unchanged-choices.mjs: what forward.js decides without a browser over a grid of made-up numbers (GPU or CPU for
//            a block and a step, where the weights go, the memory a model needs): the constants that choose a device are in no other check
//   exports  (review) the names a window keeps: every export of forward.js, jobs.js, kept.js, gpu.js and src/bench.js (a function by its
//            arity, a constant as JSON), and every public name of llama2_convert and llama2_numpy with its signature, a class's methods too
//            (tests/unchanged_exports.py): the tests reach only the names they use, a facade that forgets one breaks the page
//   page     (T355 review) the model page's script (src/pages/index.astro's <script> and src/page/*.ts): what it is made of, counted: every
//            string, number and regular expression, every operator, every `.name` and object key, every kind of statement and
//            expression. A moved statement counts the same wherever it is; a branch dropped, a key spelled otherwise, a limit
//            changed or `===` turned to `!==` is a count that differs. (No test outside CI's browsers runs the page's script.)
//            (T353) and the same of /benchmark/'s script (src/pages/benchmark.astro's <script> and src/benchmark/*.ts)
//   bench    (T353 review) every call tests/bench.mjs makes to src/bench.js, written down (tests/unchanged-bench.mjs), in both trees:
//            the arguments and the Markdown or numbers that come back. The tables' words, cells and warnings, read by nothing else
//   gpuworker (T353 review) the statements of /benchmark/'s GPU worker (public/benchmark/gpu.js and public/benchmark/gpu/*.js), and (T352)
//            of the model's (public/gpu.js and public/gpu/*.js, whose modules share `common` where the benchmark's share `shared`), each
//            function and constant as its syntax tree: `shared.<x>` (the fields the modules share) read as the `let` it was, the
//            ../ of a URL dropped, positions and comments dropped. A field taken for another (`shared.fallback` for `shared.packed`:
//            a device that is not a fallback and has the packed dot product tells them apart, one that is neither or both does not),
//            a condition, a number or a name changed in a moved statement is a statement that differs. The worker runs nowhere but in CI's
//            GPU jobs; this is the one check that reads every line of it, against the commit it was divided from
//   sizes    the files past the size a file should have (50 KB or 800 lines): said, never failed
//
// The other tree is tests/other-tree.mjs's: `git archive` of the commit under .tmp/unchanged/<its hash> (made once).
// Exit 1 if anything differs.
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { transformSync } from "esbuild";
import { parse } from "@babel/parser";
import { otherTree } from "./other-tree.mjs";
import { treeOf } from "./tree.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const args = process.argv.slice(2);
const flag = (name, otherwise) => (args.includes(name) ? args.splice(args.indexOf(name), 2)[1] : otherwise);
const before = flag("--before", "origin/main");
const kinds = args.length ? args : ["shaders", "models", "calls", "choices", "exports", "layouts", "python", "page", "bench", "gpuworker", "sizes"];
const git = (...command) => execFileSync("git", command, { cwd: root, maxBuffer: 1 << 28 });
const python = process.env.PYTHON ?? "python3";
const LIMIT_BYTES = 50 * 1024, LIMIT_LINES = 800;

const text = (value) => (typeof value === "function" ? String(value) : typeof value === "string" ? value : JSON.stringify(value));
// the differences of two objects of texts, by name
function differences(was, now) {
  const names = [...new Set([...Object.keys(was), ...Object.keys(now)])].sort();
  return { count: names.length, differ: names.filter((name) => name in was && name in now && was[name] !== now[name]),
    gone: names.filter((name) => !(name in now)), added: names.filter((name) => !(name in was)) };
}
function said(what, { count, differ, gone, added }, more = "") {
  const wrong = differ.length + gone.length + added.length;
  const list = (label, names) => (names.length ? `; ${label}: ${names.slice(0, 12).join(", ")}${names.length > 12 ? ` and ${names.length - 12} more` : ""}` : "");
  console.log(`unchanged: ${what}: ${wrong ? "CHANGED" : "the same"}: ${count} compared${more}${list("differ", differ)}${list("gone", gone)}${list("new", added)}`);
  return wrong === 0;
}

// T366: the makers of WGSL among public/shaders.js's exports, and the arguments that walk every branch of each one's
// text (a number is written into the text and chooses nothing). The engine's makers are held by deviceKey() as well
// (below); the benchmark's (fmaCeiling, mulMatVec) by this table alone: a branch added to one of those is to be added here
const EITHER = [false, true], OUTPUTS = ["rope", "add", "swiglu", "write"];
const MAKERS = {
  regTile: EITHER.map((half) => [half]),
  dp4a: EITHER.flatMap((subgroups) => EITHER.map((ternary) => [subgroups, ternary])),
  flashTile: EITHER.flatMap((half) => EITHER.map((subgroups) => [{ headSize: 64, half, subgroups, wgSize: 32, kvTile: 8, minSubgroup: 4 }])),
  flashVec: EITHER.map((subgroups) => [{ headSize: 64, subgroups, wgSize: 32, kvTile: 32, dSplit: 16 }]),
  flashVecReduce: EITHER.map((subgroups) => [{ headSize: 64, subgroups, reduceSize: 32 }]),
  fmaCeiling: EITHER.flatMap((half) => ["square", "affine"].map((shape) => [half, shape])),
  mulMatVec: EITHER.flatMap((packed) => EITHER.map((subgroups) => [{ packed, subgroups }])),
  fusedMatVec: ["norm", "plain"].flatMap((input) => OUTPUTS.flatMap((output) => EITHER.map((subgroups) => [{ input, output, subgroups }]))),
  fusedDp4aMatVec: OUTPUTS.map((output) => [{ output }]),
  ternaryMatVec: OUTPUTS.map((output) => [{ output }]),
};

async function shaders(other) {
  const [was, now] = await Promise.all([other, root].map((tree) => import(treeOf(tree).runtimeUrl("shaders.js"))));
  const texts = (module) => Object.fromEntries(Object.entries(module).map(([name, value]) =>
    [name, typeof value === "function" && MAKERS[name] ? MAKERS[name].map((given) => value(...given)).join("\n----\n") : text(value)]));
  let ok = said("shaders, the exports", differences(texts(was), texts(now)));
  const keys = { was: {}, now: {} };
  for (const packed of [false, true]) {
    Object.defineProperty(globalThis, "navigator", { configurable: true,
      value: { userAgent: "UA", gpu: { wgslLanguageFeatures: new Set(packed ? ["packed_4x8_integer_dot_product"] : []) } } });
    for (const f16 of [false, true]) for (const subgroups of [false, true]) for (const memory of [16384, 32768, 49152, 65536]) {
      for (const threads of [128, 256, 1024]) for (const ternary of [false, true]) {
        const device = { info: { vendor: "v", architecture: "a", device: "d", description: "x" },
          features: new Set([...(f16 ? ["shader-f16"] : []), ...(subgroups ? ["subgroups"] : [])]),
          limits: { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup: threads, maxComputeWorkgroupSizeX: threads } };
        const name = JSON.stringify([packed, f16, subgroups, memory, threads, ternary]);
        keys.was[name] = was.deviceKey(device, device, ternary);
        keys.now[name] = now.deviceKey(device, device, ternary);
      }
    }
  }
  ok = said("shaders, deviceKey()", differences(keys.was, keys.now), ` (one of them: ${Object.values(keys.now)[0]})`) && ok;
  return ok;
}

async function models(other) {
  const [was, now] = await Promise.all([other, root].map((tree) => import(pathToFileURL(path.join(tree, "src/models.js")))));
  const answers = (module) => {
    const found = {};
    for (const [name, value] of Object.entries(module)) {
      if (typeof value !== "function") { found[name] = JSON.stringify(value); continue; }
      // a function: what it answers for every entry (and for nothing), an error's message where it throws
      module.MODELS.forEach((entry) => {
        for (const [label, more] of [["", []], [", asked 6 of a device of 4", [6, 4]], [", asked nothing of a device of 8", [undefined, 8]]]) {
          let answer;
          try { answer = JSON.stringify(value(entry, ...more)); } catch (error) { answer = `throws ${error.message}`; }
          found[`${name}(${entry.id}${label})`] = answer;
        }
      });
    }
    return found;
  };
  return said("models", differences(answers(was), answers(now)), ` (${now.MODELS.length} entries)`);
}

function calls(other) {
  const run = (tree) => JSON.parse(execFileSync("node", [path.join(root, "tests/unchanged-calls.mjs"), tree], { maxBuffer: 1 << 28, env: { ...process.env, PYTHON: python } }));
  return said("the plans and the kernels' calls", differences(run(other), run(root)));
}

// (T357) what both trees say of a checkpoint's file, a config.json and the penalty's window, over one grid
function layouts(other) {
  const run = (tree) => JSON.parse(execFileSync(python, [path.join(root, "tests/unchanged_layouts.py"), tree], { maxBuffer: 1 << 28, env: { ...process.env, PYTHONHASHSEED: "0" } }));
  return said("the checkpoints' layouts", differences(run(other), run(root)));
}

function pythonTests(other) {
  const run = (tree, name) => {
    const record = path.join(root, ".tmp", "unchanged", `${name}.json`);
    fs.rmSync(record, { force: true });
    const ran = spawnSync(python, ["-m", "pytest", "tests", "-q", "-p", "unchanged_recorder", "-p", "no:cacheprovider"], { cwd: tree, maxBuffer: 1 << 28,
      env: { ...process.env, UNCHANGED_RECORD: record, PYTHONPATH: path.join(root, "tests"), PYTEST_DEBUG_TEMPROOT: path.join(root, ".tmp"),
        PYTHONHASHSEED: "0" } });  // (a test that takes its text from a set would write another every run)
    const last = ran.stdout.toString().trim().split("\n").at(-1);
    if (!fs.existsSync(record)) throw new Error(`pytest wrote no record in ${tree}: ${last}\n${ran.stderr.toString().slice(-2000)}`);
    const flat = {};
    for (const [test, kinds] of Object.entries(JSON.parse(fs.readFileSync(record, "utf8")))) for (const [kind, value] of Object.entries(kinds)) flat[`${test}: ${kind}`] = value;
    return { flat, last, status: ran.status };
  };
  const was = run(other, "before"), now = run(root, "now");
  console.log(`unchanged: pytest before: ${was.last}`);
  console.log(`unchanged: pytest now: ${now.last}`);
  // (review) a test that ran in one tree and was skipped in the other is compared with nothing: the counts must be the same
  const counts = (line) => line.replace(/ in [\d.]+s.*$/, "").replace(/^=+\s*|\s*=+$/g, "");
  const same = counts(was.last) === counts(now.last);
  if (!same) console.log(`unchanged: pytest ran different tests (before: ${counts(was.last)}; now: ${counts(now.last)}): CHANGED`);
  return said("what the unit tests convert and compute", differences(was.flat, now.flat)) && now.status === 0 && same;
}

function choices(other) {
  const run = (tree) => JSON.parse(execFileSync("node", [path.join(root, "tests/unchanged-choices.mjs"), tree], { maxBuffer: 1 << 28 }));
  return said("forward.js's choices", differences(run(other), run(root)));
}

async function exports(other) {
  let ok = true;
  const [was, now] = await Promise.all([other, root].map(async (tree) => {
    const found = {};
    // (T367.1: a window of the runtime by its name there, wherever the tree keeps the runtime: tests/tree.mjs)
    for (const file of ["forward.js", "jobs.js", "kept.js", "gpu.js", "src/bench.js"]) {
      let module;
      try { module = await import(file.startsWith("src/") ? pathToFileURL(path.join(tree, file)) : treeOf(tree).runtimeUrl(file)); } catch (error) { found[`${file}: import`] = `throws ${error.message}`; continue; }
      for (const [name, value] of Object.entries(module)) {
        found[`${file}: ${name}`] = typeof value === "function" ? `${/^class\b/.test(String(value)) ? "class" : "function"}/${value.length}` : JSON.stringify(value) ?? String(value);
      }
    }
    return found;
  }));
  ok = said("exports, the JavaScript windows", differences(was, now)) && ok;
  const python_ = (tree) => JSON.parse(execFileSync(python, [path.join(root, "tests/unchanged_exports.py"), tree], { maxBuffer: 1 << 28, env: { ...process.env, PYTHONHASHSEED: "0" } }));
  return said("exports, the Python windows", differences(python_(other), python_(root))) && ok;
}

// the parts of a page's script, counted (T355): the script of index.astro and every .ts of src/page/ of a tree (T353: and
// of benchmark.astro and src/benchmark/)
function pageShape(tree, astro = "index.astro", modules = "page") {
  const sources = [];
  const page = path.join(tree, "src/pages", astro);
  sources.push(/<script>\n([\s\S]*?)<\/script>/.exec(fs.readFileSync(page, "utf8"))[1]);
  const folder = path.join(tree, "src", modules);
  if (fs.existsSync(folder)) for (const file of fs.readdirSync(folder).sort()) sources.push(fs.readFileSync(path.join(folder, file), "utf8"));
  const found = {};
  const add = (key) => { found[key] = (found[key] ?? 0) + 1; };
  // (what the division itself changes: imports and exports, `page.x` for a `let x`, the shape of declarations, comments, `undefined` for `let x;`)
  const SKIPPED = /^(Identifier|MemberExpression|OptionalMemberExpression|ImportDeclaration|ExportNamedDeclaration|CommentLine|CommentBlock|Program|File|VariableDeclaration|VariableDeclarator|ObjectProperty|ObjectExpression)$/;
  const visit = (node) => {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) { node.forEach(visit); return; }
    if (!node.type || node.type === "ImportDeclaration") return;
    if (node.type === "UnaryExpression" && node.operator === "void") return;  // (`void 0`: esbuild's `undefined`)
    const type = node.type;
    // (the one object the parts share: the names of its fields were the names of the `let`s)
    if (type === "VariableDeclarator" && node.id.name === "page" && node.init?.type === "ObjectExpression") { node.init.properties.forEach((property) => visit(property.value)); return; }
    if (!SKIPPED.test(type)) add(`node ${type}`);
    if (type === "StringLiteral" || type === "NumericLiteral" || type === "BooleanLiteral") add(`literal ${JSON.stringify(node.value)}`);
    if (type === "RegExpLiteral") add(`regular expression /${node.pattern}/${node.flags}`);
    if (type === "TemplateElement") add(`template ${JSON.stringify(node.value.cooked)}`);
    if (/^(Binary|Logical|Assignment|Unary|Update)Expression$/.test(type)) add(`operator ${type} ${node.operator}`);
    if ((type === "MemberExpression" || type === "OptionalMemberExpression") && !node.computed && !(node.object.type === "Identifier" && node.object.name === "page")) add(`property .${node.property.name}${node.optional ? "?" : ""}`);
    if (type === "ObjectProperty" && !node.computed && node.key.type === "Identifier" && node.key.name !== "id") add(`key ${node.key.name}`);
    for (const key of Object.keys(node)) if (key !== "loc" && key !== "extra") visit(node[key]);
  };
  for (const text of sources) {
    const { code } = transformSync(text, { loader: "ts", tsconfigRaw: { compilerOptions: { verbatimModuleSyntax: true } } });
    visit(parse(code, { sourceType: "module" }).program);
  }
  return found;
}
function page(other) {
  let ok = true;
  for (const [what, astro, modules] of [["the model page's script", "index.astro", "page"], ["/benchmark/'s script", "benchmark.astro", "benchmark"]]) {
    const was = pageShape(other, astro, modules), now = pageShape(root, astro, modules);
    ok = said(what, differences(was, now), ` (${Object.values(now).reduce((sum, count) => sum + count, 0)} parts)`) && ok;
  }
  return ok;
}

function bench(other) {
  const run = (tree) => {
    const record = path.join(root, ".tmp", "unchanged", `bench-${path.basename(tree)}.json`);
    fs.rmSync(record, { force: true });
    const ran = spawnSync("node", ["--import", path.join(root, "tests/unchanged-bench.mjs"), path.join(tree, "tests/bench.mjs")], { cwd: tree, maxBuffer: 1 << 28,
      env: { ...process.env, UNCHANGED_RECORD: record } });
    if (ran.status !== 0 || !fs.existsSync(record)) throw new Error(`tests/bench.mjs did not finish in ${tree}: ${ran.stderr.toString().slice(-2000)}`);
    const flat = {};
    JSON.parse(fs.readFileSync(record, "utf8")).forEach(([name, args, answer], index) => { flat[`${String(index).padStart(4, "0")} ${name}`] = `${args} => ${answer}`; });
    return flat;
  };
  const was = run(other), now = run(root);
  if (Object.keys(now).length < 100) throw new Error(`only ${Object.keys(now).length} calls were written down: the preload does not reach src/bench.js`);
  return said("src/bench.js, the calls tests/bench.mjs makes", differences(was, now));
}

// the statements of /benchmark/'s GPU worker (T353 review), as syntax trees. What the division adds is told apart from what it moves:
// imports, the window's loading and early queue, the `shared` object, the destructuring of what a module takes from another
// (T352: of the model's GPU worker too. worker: { window: the file a worker starts from, object: the name of the object its modules share })
function gpuWorker(tree, fields, worker) {
  const files = [worker.window], inFolder = worker.window.replace(/\.js$/, ""), runtime = treeOf(tree).runtime;
  const folder = runtime(inFolder);
  if (fs.existsSync(folder)) for (const file of fs.readdirSync(folder).sort()) files.push(`${inFolder}/${file}`);
  const asts = files.map((file) => parse(fs.readFileSync(runtime(file), "utf8"), { sourceType: "module" }).program);
  const declared = (statement) => {
    const names = [];
    const pattern = (node) => { if (!node) return; if (node.type === "Identifier") names.push(node.name); else if (node.type === "ObjectPattern") node.properties.forEach((p) => pattern(p.type === "RestElement" ? p.argument : p.value)); else if (node.type === "ArrayPattern") node.elements.forEach(pattern); else if (node.type === "AssignmentPattern") pattern(node.left); };
    statement.declarations.forEach((d) => pattern(d.id));
    return names;
  };
  // (fields: the names the modules share, the keys of the `shared` object of the tree being checked, and the `let`s of a tree before
  // the division that these became)
  const norm = (node) => {
    if (Array.isArray(node)) return node.map(norm);
    if (!node || typeof node !== "object") return node;
    if (node.type === "MemberExpression" && !node.computed && node.object.type === "Identifier" && node.object.name === worker.object && fields.has(node.property.name)) return { type: "Identifier", name: node.property.name };
    const out = {};
    for (const key of Object.keys(node)) {
      if (["loc", "start", "end", "extra", "range", "leadingComments", "trailingComments", "innerComments", "shorthand"].includes(key)) continue;
      out[key] = norm(node[key]);
    }
    if (node.type === "TemplateElement") out.value = { raw: node.value.raw.replace(/^(\.\.\/)+/, ""), cooked: node.value.cooked.replace(/^(\.\.\/)+/, "") };
    return out;
  };
  // (`await import(…)` of a module's, or `await modules.x` of the window's)
  const importsAModule = (init) => JSON.stringify(init, (key, v) => (key === "loc" ? undefined : v)).includes('"type":"Import"') ||
    (init.type === "AwaitExpression" && init.argument.type === "MemberExpression" && init.argument.object.name === "modules");
  const found = {};
  for (const [index, program] of asts.entries()) for (let statement of program.body) {
    const wrapped = statement.type === "ExportNamedDeclaration";
    if (wrapped && statement.declaration) statement = statement.declaration;
    if (statement.type === "ImportDeclaration") continue;
    if (wrapped && !statement.declarations && !statement.id) {
      // (the window's names, as the tests import them; what a module exports to its neighbours is the division's)
      if (index === 0) found["export { }"] = JSON.stringify(statement.specifiers.map((s) => s.exported.name).sort());
      continue;
    }
    let key;
    if (statement.type === "FunctionDeclaration") key = `function ${statement.id.name}`;
    else if (statement.type === "VariableDeclaration") {
      // (not moved: the `let`s that became fields (T352: where a `let` declared one of them with others, the others are what is
      // compared), `shared`, the window's `modules` and `early`, what a module takes from another)
      if (statement.kind === "let" && statement.declarations.some((d) => fields.has(d.id.name))) {
        statement = { ...statement, declarations: statement.declarations.filter((d) => !fields.has(d.id.name)) };
        if (!statement.declarations.length) continue;
      }
      const names = declared(statement);
      if (names.some((name) => [worker.object, "modules", "early"].includes(name)) || statement.declarations.some((d) => d.init && importsAModule(d.init) && d.id.type === "ObjectPattern")) continue;
      key = `${statement.kind === "let" ? "let" : "const"} ${names.join(",")}`;
    } else if (statement.type === "ExpressionStatement" && statement.expression.type === "AssignmentExpression" && statement.expression.left.name === "onmessage") {
      // (the early queue is the division's; T352: the model's worker's receiver is no async function)
      const right = statement.expression.right, queue = right.body.type === "CallExpression" && right.body.callee.object?.name === "early";
      if (queue) continue;
      key = right.async ? "onmessage = async" : "onmessage =";
    } else continue;   // (the replay of the early queue)
    if (key in found) throw new Error(`${key} twice in the GPU worker`);
    found[key] = JSON.stringify(norm(statement));
  }
  return found;
}
// the keys of the `shared` object a tree's modules hold between them (none before the division)
function sharedFields(tree, worker) {
  const file = treeOf(tree).runtime(worker.window.replace(/\.js$/, "/device.js"));
  if (!fs.existsSync(file)) return new Set();
  const program = parse(fs.readFileSync(file, "utf8"), { sourceType: "module" }).program;
  const shared = program.body.find((s) => s.type === "VariableDeclaration" && s.declarations[0].id.name === worker.object);
  return new Set(shared.declarations[0].init.properties.map((p) => p.key.name));
}
// (T352: the model's GPU worker, public/gpu.js and public/gpu/*.js, whose modules share `common`: open() has a parameter named shared)
// (window: the file a worker starts from, as the runtime names it)
const GPU_WORKERS = [{ window: "benchmark/gpu.js", object: "shared", name: "/benchmark/'s GPU worker" },
  { window: "gpu.js", object: "common", name: "the model's GPU worker" }];
function gpuworker(other) {
  let same = true;
  for (const worker of GPU_WORKERS) {
    const fields = sharedFields(root, worker);
    const was = gpuWorker(other, fields, worker), now = gpuWorker(root, fields, worker);
    if (!fields.size) console.log(`unchanged: gpuworker: ${worker.name} is one file in the working tree`);
    same = said(`${worker.name}, statement by statement`, differences(was, now)) && same;
  }
  return same;
}

function sizes() {
  // (public and src: wherever a tree keeps the runtime and the Python, it is under one of them)
  const files = git("ls-files", "public", "src", "kernels", "tests", "*.py", "*.mjs").toString().trim().split("\n")
    .filter((file) => /\.(js|mjs|py|ts|astro)$/.test(file) && !file.startsWith("tests/fixtures/") && fs.existsSync(path.join(root, file)));
  const large = files.map((file) => { const body = fs.readFileSync(path.join(root, file), "utf8"); return { file, bytes: Buffer.byteLength(body), lines: body.split("\n").length }; })
    .filter(({ bytes, lines }) => bytes > LIMIT_BYTES || lines > LIMIT_LINES).sort((a, b) => b.bytes - a.bytes);
  console.log(`unchanged: sizes: ${large.length} of ${files.length} files are past ${LIMIT_BYTES / 1024} KB or ${LIMIT_LINES} lines` +
    `${large.length ? `, ${Math.round(large.reduce((sum, { bytes }) => sum + bytes, 0) / 1024)} KB in them` : ""}`);
  for (const { file, bytes, lines } of large) console.log(`unchanged: sizes:   ${String(Math.round(bytes / 1024)).padStart(4)} KB ${String(lines).padStart(5)} lines  ${file}`);
  return true;
}

const { commit, folder } = kinds.some((kind) => kind !== "sizes") ? otherTree(before) : {};
if (commit) console.log(`unchanged: the working tree against ${before} (${commit.slice(0, 7)})`);
const checks = { shaders: () => shaders(folder), models: () => models(folder), calls: () => calls(folder), choices: () => choices(folder), exports: () => exports(folder), layouts: () => layouts(folder), python: () => pythonTests(folder), page: () => page(folder), bench: () => bench(folder), gpuworker: () => gpuworker(folder), sizes };
let ok = true;
for (const kind of kinds) {
  if (!checks[kind]) throw new Error(`no check "${kind}": ${Object.keys(checks).join(", ")}`);
  ok = (await checks[kind]()) && ok;
}
console.log(ok ? "unchanged: ok" : "unchanged: FAILED");
process.exit(ok ? 0 : 1);
