// tests/python-archive-check.mjs (T367.2, in tests/python-files-check.mjs's place)
// The site's Python goes to a visitor as two archives (python_archive.py: engine.zip, converter.zip), which the worker
// fetches whole and unpacks in Pyodide. Three things are held here, each alone:
//
//   (1) the builder, on a made-up folder of sources: what goes in (a module's window and every .py under its package's
//       folder, and nothing else of the folder), in which order, with which dates, modes and compression; and that the
//       same sources give the same bytes whatever the files' dates and modes are;
//   (2) this tree's archives: what is in them is what the tools walk and place (tests/tree.mjs: a tool in Node reads no
//       archive), byte for byte; every .py of the sources' folder is in one of them (a file that is nobody's would be
//       fetched by no one, and the page would fail at an import where every test in Node passes); two builds are the
//       same bytes; and an archive that lies built in the tree is the one its sources make now (an old one beside new
//       sources is what a server of public/ would hand out). What python-files-check held of the sources stays: a
//       package has an __init__.py, and its parts import by the package's own name;
//   (3) the worker's way with them (public/worker/pyodide.js and convert.js, in a vm: tests/worker-harness.mjs): what it
//       asks for and with which ?v=, what it says when the archive is not there, that the converter's archive is asked
//       for when a model is first converted and not before, on a stand-in of Pyodide; and with the real Pyodide, that
//       what it unpacks is found by `import`, with the sources' bytes.
//
//   node tests/python-archive-check.mjs       (PYTHON=.venv/bin/python; about three seconds)
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";
import { loadPyodide } from "pyodide";
import { RULE, treeOf } from "./tree.mjs";
import { workerHarness } from "./worker-harness.mjs";
import { workerScripts } from "./worker-source.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const python = process.env.PYTHON ?? "python3";
const work = path.join(root, ".tmp", "python-archive-check");
fs.rmSync(work, { recursive: true, force: true });
const ok = (line) => console.log(`ok: ${line}`);
// an archive's name -> its module (the builder's own table is ARCHIVES of python_archive.py; the walk is tests/tree.mjs's)
const ARCHIVES = { "engine.zip": "llama2_numpy", "converter.zip": "llama2_convert" };

/** python_archive.py of this tree: the archives of a folder of sources, written into another */
function build(sources, out) {
  execFileSync(python, [path.join(root, "python_archive.py"), sources, out], { stdio: ["ignore", "pipe", "inherit"] });
  return Object.fromEntries(Object.keys(ARCHIVES).map((name) => [name, fs.readFileSync(path.join(out, name))]));
}

/** A zip's entries as its central directory lists them: { name, method, time, date, system, mode, data, packed } */
function entriesOf(zip) {
  const end = zip.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  assert.ok(end >= 0, "no end of a central directory: not a zip");
  const count = zip.readUInt16LE(end + 10);
  let at = zip.readUInt32LE(end + 16);
  const entries = [];
  for (let i = 0; i < count; i++) {
    assert.equal(zip.readUInt32LE(at), 0x02014b50, "an entry of the central directory");
    const named = zip.readUInt16LE(at + 28), extra = zip.readUInt16LE(at + 30), comment = zip.readUInt16LE(at + 32), local = zip.readUInt32LE(at + 42);
    const packedSize = zip.readUInt32LE(at + 20), size = zip.readUInt32LE(at + 24), method = zip.readUInt16LE(at + 10);
    const begins = local + 30 + zip.readUInt16LE(local + 26) + zip.readUInt16LE(local + 28);
    const packed = zip.subarray(begins, begins + packedSize);
    const data = method === 8 ? zlib.inflateRawSync(packed) : Buffer.from(packed);
    assert.equal(data.length, size);
    assert.equal(zlib.crc32(data), zip.readUInt32LE(at + 16), "an entry's CRC-32");
    entries.push({ name: zip.toString("utf8", at + 46, at + 46 + named), method, time: zip.readUInt16LE(at + 12), date: zip.readUInt16LE(at + 14),
      system: zip[at + 5], mode: zip.readUInt32LE(at + 38) >>> 16, data, packed, local });
    at += 46 + named + extra + comment;
  }
  return entries;
}

// ---- (1) the builder, on a made-up folder
{
  const sources = path.join(work, "made-up", "sources");
  // a text that deflates to fewer bytes at level 9 than at the usual 6 (the check of the level needs one that does)
  let seed = 7;
  const word = () => ["alpha", "beta", "gamma", "delta", "tensor", "layer", "norm", "head", "plan", "rows"][(seed = (seed * 1103515245 + 12345) >>> 0) % 10];
  const long = Array.from({ length: 30000 }, (_, i) => `${word()}_${word()} = ${word()}(${word()}, ${i % 97})\n`).join("");
  const files = {
    "llama2_numpy.py": "import engine\n",
    "engine/__init__.py": "# the parts\n",
    "engine/b.py": "B = 2\n",
    "engine/a.py": long,
    "engine/deep/c.py": "C = 'ふじ'\n",
    // what is in the package's folder and is no source
    "engine/__pycache__/a.cpython-313.pyc": "\0compiled",
    "engine/__pycache__/trap.py": "raise SystemExit('a .py of a __pycache__')\n",
    "engine/notes.txt": "not Python\n",
    "engine/a.py.orig": "an editor's copy\n",
    "llama2_convert.py": "import convert\n",
    "convert/__init__.py": "",
    "convert/x.py": "X = 1\r\n",  // (a file's bytes are kept as they are, line ends too)
    // Python that is no module's part
    "stray.py": "raise SystemExit('nobody asked')\n",
    "tests/test_a.py": "def test_a(): pass\n",
    "convert_old/y.py": "Y = 0\n",
  };
  for (const [name, text] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(sources, name)), { recursive: true });
    fs.writeFileSync(path.join(sources, name), text);
  }
  const first = build(sources, path.join(work, "made-up", "one"));
  const expected = {
    "engine.zip": ["engine/__init__.py", "engine/a.py", "engine/b.py", "engine/deep/c.py", "llama2_numpy.py"],
    "converter.zip": ["convert/__init__.py", "convert/x.py", "llama2_convert.py"],
  };
  for (const [name, names] of Object.entries(expected)) {
    const entries = entriesOf(first[name]);
    assert.deepEqual(entries.map((entry) => entry.name), names, `${name} holds the module's window and every .py under its package's folder, in the order of their names, and nothing else`);
    assert.deepEqual([...entries].sort((a, b) => a.local - b.local).map((entry) => entry.name), names, `${name}'s files lie in the order of their names`);
    for (const entry of entries) {
      assert.ok(entry.data.equals(Buffer.from(files[entry.name])), `${name}: ${entry.name} has the source's bytes`);
      // (a zip's date: years since 1980 << 9 | month << 5 | day; its time: 0 is midnight)
      assert.deepEqual([entry.date, entry.time], [(0 << 9) | (1 << 5) | 1, 0], `${name}: ${entry.name} is dated 1980-01-01 00:00:00`);
      assert.equal(entry.method, 8, `${name}: ${entry.name} is deflated`);
      assert.deepEqual([entry.system, entry.mode], [3, 0o100644], `${name}: ${entry.name} is a file of mode 644, as of Unix`);
    }
  }
  // deflate at level 9: Python's own zlib at 9 gives the entry's bytes, and at 6 (zipfile's own choice) it gives more
  const sizes = JSON.parse(execFileSync(python, ["-c",
    "import json, sys, zlib\ndata = open(sys.argv[1], 'rb').read()\ndef size(level):\n    c = zlib.compressobj(level, zlib.DEFLATED, -15)\n    return len(c.compress(data) + c.flush())\nprint(json.dumps([size(9), size(6)]))",
    path.join(sources, "engine/a.py")]).toString());
  assert.ok(sizes[0] < sizes[1], `the made-up text deflates to as many bytes at level 6 as at 9 (${sizes}): it shows no level`);
  assert.equal(entriesOf(first["engine.zip"]).find((entry) => entry.name === "engine/a.py").packed.length, sizes[0], "engine/a.py is deflated at level 9");

  // the same sources, other dates and modes on the disk, another day: the same bytes
  for (const name of Object.keys(files)) fs.utimesSync(path.join(sources, name), new Date("2031-05-06T07:08:09Z"), new Date("2031-05-06T07:08:09Z"));
  fs.chmodSync(path.join(sources, "engine/b.py"), 0o755);
  fs.chmodSync(path.join(sources, "llama2_convert.py"), 0o600);
  const second = build(sources, path.join(work, "made-up", "two"));
  for (const name of Object.keys(ARCHIVES)) assert.ok(first[name].equals(second[name]), `${name} is other bytes when its sources' dates and modes on the disk are others`);
  // and other sources, other bytes: one byte more in a part
  fs.appendFileSync(path.join(sources, "convert/x.py"), "#");
  const third = build(sources, path.join(work, "made-up", "three"));
  assert.ok(!first["converter.zip"].equals(third["converter.zip"]) && first["engine.zip"].equals(third["engine.zip"]), "a part's change changes its own archive and no other");
  ok("the builder, on a made-up folder: a window and its package's .py files alone, by name, dated 1980-01-01, mode 644, deflate 9; the same bytes from the same sources");
}

// ---- (2) this tree's archives
const tree = treeOf(), where = `${tree.folders.python}/`;
const made = build(tree.python(), path.join(work, "tree"));
{
  const walked = Object.fromEntries(Object.entries(ARCHIVES).map(([name, module]) => [name, tree.pythonFiles(module)]));
  assert.deepEqual(Object.values(ARCHIVES).sort(), Object.keys(RULE.packages).sort(), "the archives are of the modules the tools walk (tests/tree.json)");
  for (const [name, files] of Object.entries(walked)) {
    const entries = entriesOf(made[name]);
    assert.deepEqual(entries.map((entry) => entry.name), [...files].sort(), `${name} is not the files the tools walk and place (tests/tree.mjs)`);
    for (const entry of entries) assert.ok(entry.data.equals(fs.readFileSync(tree.python(entry.name))), `${name}: ${entry.name} has not the source's bytes`);
    // what python-files-check held of the sources: the parts are one package with an __init__.py, named by its own name
    const [window, ...parts] = files;
    assert.equal(window, `${ARCHIVES[name]}.py`);
    const folders = new Set(parts.map((part) => part.split("/")[0]));
    assert.ok(folders.size === 1, `${ARCHIVES[name]}'s parts are in more than one folder, or in none`);
    for (const folder of folders) assert.ok(parts.includes(`${folder}/__init__.py`), `${folder}/ has no __init__.py`);
    for (const part of parts) {
      assert.equal(fs.readFileSync(tree.python(part), "utf8").match(/^from \.+\w* import|^import \./m), null, `${part} imports by a relative name`);
    }
  }
  const there = fs.readdirSync(tree.python(), { recursive: true }).map((name) => name.split(path.sep).join("/"))
    .filter((name) => name.endsWith(".py") && !name.includes("__pycache__")).sort();
  assert.deepEqual(Object.values(walked).flat().sort(), there, `a .py file of ${where} is in no archive (a module's window, or a file under its package's folder), or in two`);
  const again = build(tree.python(), path.join(work, "tree-again"));
  for (const name of Object.keys(ARCHIVES)) assert.ok(made[name].equals(again[name]), `${name}: two builds of the same sources are other bytes`);
  // an archive that lies built in the tree (npm run build, npm run dev and make python build them) is today's
  const beside = [];
  for (const name of Object.keys(ARCHIVES)) {
    const built = tree.served(name, { maybe: true });
    if (!fs.existsSync(built)) continue;
    assert.ok(fs.readFileSync(built).equals(made[name]), `${path.relative(root, built)} is not what python_archive.py makes of ${where} now: an old archive beside newer sources (make python)`);
    beside.push(path.relative(root, built));
  }
  // and they are built wherever the site is: before a build and before the development server (a site without them
  // says "Could not fetch engine.zip: 404" and nothing in Node would have said so)
  const scripts = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8")).scripts;
  for (const script of ["build", "dev"]) assert.match(scripts[script], /^python3 python_archive\.py && astro /, `npm run ${script} does not build the archives first`);
  // the build's own scripts find the sources where they are: run with no arguments from another folder, each gets past
  // its imports to its missing argument (tests/test_convert_hf.py imports the converter itself, by pytest's path)
  for (const script of ["convert_hf.py", "quantize.py"]) {
    const said = spawnSync(python, [path.join(root, script)], { cwd: work, encoding: "utf8", env: { ...process.env, PYTHONPATH: "", PYTHONDONTWRITEBYTECODE: "1" } }).stderr;
    assert.ok(/IndexError/.test(said) && !/ModuleNotFoundError|ImportError/.test(said), `${script} does not find the Python it imports: ${said.trim().split("\n").pop()}`);
  }
  ok(`this tree's archives hold the ${there.length} Python files of ${where} that the tools walk (${Object.entries(walked).map(([name, files]) => `${name}: ${files.length} files, ${made[name].length} bytes`).join("; ")}); ` +
    (beside.length ? `${beside.join(" and ")} ${beside.length > 1 ? "are" : "is"} those bytes` : "none is built in the tree now"));
}

// ---- (3) the worker's way with them
{
  const { context, run, requests, fresh } = workerHarness();
  assert.equal(requests.length, 0, "the worker asked for something as its modules ran");
  const beside = (name) => new URL(name, tree.runtimeUrl("worker.js")).href;
  context.location.search = "?v=0123abc";
  const answers = (bytes) => (url) => (/\.zip\?v=0123abc$/.test(url) ? new Response(bytes) : new Response("", { status: 404 }));

  // what is asked for: the archive beside the worker, with the worker's ?v=<build>, once; its bytes whole
  fresh(answers(made["engine.zip"]));
  const came = await run("pythonArchive")("engine.zip");
  assert.deepEqual(requests.map(({ url, method }) => `${method} ${url}`), [`GET ${beside("engine.zip")}?v=0123abc`]);
  assert.ok(Buffer.from(came).equals(made["engine.zip"]), "the archive's bytes, whole");
  // an archive that is not there: the words a file of the site has when it cannot be fetched
  fresh(() => new Response("", { status: 404 }));
  await assert.rejects(run("pythonArchive")("engine.zip"), { message: "Could not fetch engine.zip: 404" });
  // a load that was cancelled: the fetch is the load's to stop
  const cancelled = new AbortController();
  fresh(() => "hang");
  const waiting = run("pythonArchive")("converter.zip", cancelled.signal);
  cancelled.abort();
  await assert.rejects(waiting, { name: "AbortError" });
  assert.equal(requests[0].signal, cancelled.signal);

  // placed: unpacked as a zip, into the folder Pyodide is in (no other folder is named); nothing where the archive did not come
  const unpacked = [];
  const standIn = { unpackArchive: (...given) => { unpacked.push(given); } };
  await run("placePython")(standIn, Promise.resolve(came));
  assert.equal(unpacked.length, 1);
  assert.deepEqual([unpacked[0][0] === came, ...unpacked[0].slice(1)], [true, "zip"], "placePython() unpacks the bytes it was given as a zip, where Pyodide is");
  await assert.rejects(run("placePython")(standIn, Promise.reject(new Error("Could not fetch engine.zip: 404"))), { message: "Could not fetch engine.zip: 404" });
  assert.equal(unpacked.length, 1, "something was unpacked of an archive that did not come");

  // the converter: asked for by the first conversion and not before, after jinja2 was asked of Pyodide, imported after
  // it is placed; the second conversion asks for nothing
  const told = [];
  const pyodide = {
    unpackArchive: (bytes, format) => { told.push(`unpack ${format} of ${bytes.byteLength} bytes`); setTimeout(() => pyodide.jinjaCame?.(), 30); },
    // (jinja2 comes late: only after the converter's archive is unpacked)
    loadPackage: (name) => { told.push(`loadPackage ${name}`); return new Promise((resolve) => { pyodide.jinjaCame = resolve; }); },
    pyimport: (name) => { told.push(`import ${name}`); return { name, destroy() {} }; },
  };
  context.stand = { pyodide };
  run("state.pyodide = stand.pyodide; state.llama2_convert = undefined;");
  fresh((url, init) => { told.push(`fetch ${url.slice(url.lastIndexOf("/") + 1)}`); return answers(made["converter.zip"])(url, init); });
  assert.equal(requests.length, 0);
  const load = new AbortController();
  const converter = await run("converter")(load.signal);
  assert.equal(converter.name, "llama2_convert");
  assert.equal(run("state.llama2_convert"), converter);
  assert.deepEqual(told, ["loadPackage jinja2", "fetch converter.zip?v=0123abc", `unpack zip of ${made["converter.zip"].length} bytes`, "import jinja2", "import llama2_convert"],
    "jinja2 is asked for first, the archive fetched and unpacked meanwhile, and the converter imported once both are there");
  assert.deepEqual(requests.map(({ url, signal }) => [url, signal === load.signal]), [[`${beside("converter.zip")}?v=0123abc`, true]], "one request, the load's to cancel");
  told.length = 0;
  assert.equal(await run("converter")(load.signal), converter);
  assert.deepEqual([told, requests.length], [[], 1], "the second conversion fetched or placed the converter again");
  // a converter that did not come: the conversion fails with the fetch's words, and the next one asks again
  run("state.llama2_convert = undefined;");
  fresh(() => new Response("", { status: 503 }));
  await assert.rejects(run("converter")(load.signal), { message: "Could not fetch converter.zip: 503" });
  assert.equal(run("state.llama2_convert"), undefined);

  // who names the archives: init() the engine's, converter() the converter's, and convert() alone calls converter().
  // (init() runs in no check in Node: it loads Pyodide from the CDN. What it asks for and when is seen in a browser,
  // tests/e2e.mjs; here its text says that the one archive it names is the engine's.)
  const sources = Object.fromEntries(workerScripts(tree).map(({ name, source }) => [name, source]));
  const naming = (pattern) => Object.entries(sources).filter(([, source]) => pattern.test(source)).map(([name]) => name);
  const body = (source, head) => { const from = source.indexOf(head); assert.ok(from >= 0, `no ${head}`); return source.slice(from, source.indexOf("\n}\n", from)); };
  // (the names as the code says them, in quotes: the comments name them too)
  assert.deepEqual(naming(/"engine\.zip"/), ["worker.js"]);
  assert.deepEqual(naming(/"converter\.zip"/), ["convert"]);
  assert.match(body(sources["worker.js"], "async function init(search) {"), /pythonArchive\("engine\.zip"\)[^]*await pyodideSteps\([^]*await placePython\(state\.pyodide, engine\);\s*state\.llama2_numpy = state\.pyodide\.pyimport\("llama2_numpy"\);/,
    "init() asks for the engine's archive before it waits for Pyodide, and places it before it imports the engine");
  assert.equal(Object.values(sources).join("\n").match(/"converter\.zip"/g).length, 1);
  assert.match(body(sources.convert, "async function converter(signal) {"), /"converter\.zip"/);
  assert.deepEqual(Object.entries(sources).flatMap(([name, source]) => [...source.matchAll(/\bconverter\(/g)].map(() => name)), ["convert", "convert"], "converter() is called in one place");
  assert.match(body(sources.convert, "async function convert(model, signal, id) {"), /if \(kept && !kept\.miss\) \{\s*return [^]*?\}\s*const keptMiss = [^\n]*\n\s*await converter\(signal\);/,
    "convert() asks for the converter only once the model is not among the kept ones");
  assert.ok(!/\.py\b[^"'`\n]*\$\{self\.location\.search\}|python\.js/.test(Object.values(sources).join("\n")), "the worker still fetches a .py or python.js");
  ok("the worker asks for an archive beside itself with its own ?v=, says the fetch's words where it is not there, unpacks it where Pyodide is; the converter's on the first conversion alone");

  // the real Pyodide: what the worker unpacks is where `import` finds it, with the sources' bytes
  const real = await loadPyodide();
  fresh((url) => new Response(made[url.slice(url.lastIndexOf("/") + 1, url.indexOf("?"))]));
  const spent = {};
  for (const [name, module] of Object.entries(ARCHIVES)) {
    const began = performance.now();
    await run("placePython")(real, run("pythonArchive")(name));
    spent[name] = performance.now() - began;
    // (the window and the package are asked for by name, which runs none of them: the parts import NumPy, which is not here)
    const found = (name) => real.runPython(`import importlib.util\nspec = importlib.util.find_spec(${JSON.stringify(name)})\nspec and spec.origin`);
    const home = path.posix.dirname(found(module) ?? "");
    assert.equal(found(module), `${home}/${module}.py`, `import does not find ${module} of ${name}`);
    assert.equal(found(RULE.packages[module]), `${home}/${RULE.packages[module]}/__init__.py`, `import does not find the package of ${name}`);
    for (const file of tree.pythonFiles(module)) {
      assert.ok(Buffer.from(real.FS.readFile(`${home}/${file}`)).equals(fs.readFileSync(tree.python(file))), `${file} was unpacked to other bytes`);
    }
  }
  assert.deepEqual(requests.map(({ url }) => url.slice(url.lastIndexOf("/") + 1)), ["engine.zip?v=0123abc", "converter.zip?v=0123abc"]);
  ok(`in Pyodide ${real.version}: the two archives unpacked by the worker's own functions are found by import, with the sources' bytes (${Object.entries(spent).map(([name, ms]) => `${name} in ${ms.toFixed(1)} ms`).join(", ")})`);
}

fs.rmSync(work, { recursive: true, force: true });
console.log("python-archive-check: ok");
// (T384: Node may stand still in process.exit() right after heavy work; a moment first)
setTimeout(() => process.stdout.write("", () => process.exit(0)), 200);
