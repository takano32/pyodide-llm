// unchanged.mjs (T346)
// The net under a refactoring: what the working tree does, held to what another commit does (main, by default).
// Nothing here knows how the files are divided; it asks both trees the same questions and compares the answers.
//
//   node tests/unchanged.mjs [--before <a commit, default origin/main>] [shaders models calls python sizes]
//   (PYTHON=.venv/bin/python; in CI: tests.yml's extra="node tests/unchanged.mjs")
//
//   shaders  every export of public/shaders.js (a text as it is, a function as its text) and deviceKey() for 192
//            made-up adapters: a device forgets the forms it measured when the key changes (AGENTS.md)
//   models   every export of src/models.js: the data as JSON, and what every exported function answers for every
//            entry of the list (a builder moved to another file leaves both as they were)
//   calls    tests/unchanged-calls.mjs: the plan Python hands forward.js and every call of a kernel, 150 hashes
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

const root = fileURLToPath(new URL("..", import.meta.url));
const args = process.argv.slice(2);
const flag = (name, otherwise) => (args.includes(name) ? args.splice(args.indexOf(name), 2)[1] : otherwise);
const before = flag("--before", "origin/main");
const kinds = args.length ? args : ["shaders", "models", "calls", "choices", "exports", "python", "page", "sizes"];
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

async function shaders(other) {
  const [was, now] = await Promise.all([other, root].map((tree) => import(pathToFileURL(path.join(tree, "public/shaders.js")))));
  const texts = (module) => Object.fromEntries(Object.entries(module).map(([name, value]) => [name, text(value)]));
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
    for (const file of ["public/forward.js", "public/jobs.js", "public/kept.js", "public/gpu.js", "src/bench.js"]) {
      let module;
      try { module = await import(pathToFileURL(path.join(tree, file))); } catch (error) { found[`${file}: import`] = `throws ${error.message}`; continue; }
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

// the parts of the model page's script, counted (T355): the script of index.astro and every .ts of src/page/ of a tree
function pageShape(tree) {
  const sources = [];
  const page = path.join(tree, "src/pages/index.astro");
  sources.push(/<script>\n([\s\S]*?)<\/script>/.exec(fs.readFileSync(page, "utf8"))[1]);
  const folder = path.join(tree, "src/page");
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
  const was = pageShape(other), now = pageShape(root);
  return said("the model page's script", differences(was, now), ` (${Object.values(now).reduce((sum, count) => sum + count, 0)} parts)`);
}

function sizes() {
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
const checks = { shaders: () => shaders(folder), models: () => models(folder), calls: () => calls(folder), choices: () => choices(folder), exports: () => exports(folder), python: () => pythonTests(folder), page: () => page(folder), sizes };
let ok = true;
for (const kind of kinds) {
  if (!checks[kind]) throw new Error(`no check "${kind}": ${Object.keys(checks).join(", ")}`);
  ok = (await checks[kind]()) && ok;
}
console.log(ok ? "unchanged: ok" : "unchanged: FAILED");
process.exit(ok ? 0 : 1);
