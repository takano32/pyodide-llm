// tests/tree.ts (T.367.1): where in a tree what is. The one place the tools ask, and it holds no table of places: a
// file is found by looking at the tree that is asked about, when it is asked for. (The owner, 2026-10-11:
// 「ファイルの場所はあらかじめ書くんじゃなくて、そのときに探すようにしたほうがいいんじゃないの？」) So a tool reads a tree with
// everything in public/, one with the Python in src/python/ and one with the runtime in src/runtime/ alike, and a move
// of the files needs no line here.
//
// A tree is this working tree or another commit's (tests/other-tree.ts's folder: searched under its own root). Its
// files are asked for by what they are, a kind and a name under it:
//
//   runtime   the JavaScript a browser runs ("forward.js", "worker/conduct.js", "benchmark/gpu.js")
//   python    the Python sources ("llama2_numpy.py", "convert/conduct.py")
//   built     what `make kernels` builds ("simdkernel.so", "simdkernel_plain.wasm")
//   served    what is served as it is ("coi.js", "models/tokenizer.bin")
//
// How a file is found: the kind's folder is where the kind's anchor is (tests/tree.json: one file's name a kind, looked
// for in the whole tree), and the name is looked for under it; a name that is not there is looked for in the whole tree
// by the end of its path. What stops a search from taking the wrong file without a word:
//   - found in two places: an error that names both (there are two worker.js, two gpu.js, several device.js: a kind's
//     anchor must be one file, and a name is asked under its kind);
//   - found nowhere: an error that says where it was looked for;
//   - never looked into: the folders of the tree's root that are no part of what is built and served (tree.json's
//     "skip": node_modules, dist, .tmp, .git, .claude, and the tests' and the documents' own, T.429: a fixture or a note
//     with an anchor's name is nobody's runtime), __pycache__ at any depth (tree.json's "anywhere"), and links: another
//     tree's copy under .tmp/ is searched only when it is the tree asked about. A "skip" name is skipped at the root
//     alone: a folder called dist or docs deeper in the tree is looked into like any other (T.429: it was passed over
//     without a word);
//   - a tree is walked once a process, and what is found is remembered.
// T.368.1, one name in two spellings: a file of the runtime is asked for as x.js or as x.ts, and what is found is the one
// the tree has (tree.json's "either"; the anchor too). The runtime's files become TypeScript one at a time, the tools
// read two trees (this one and another commit's, where the same file may be spelled the other way), and no tool is to
// be touched each time a file's ending changes. Both spellings in one tree: an error that names both.
//
//   const tree = treeOf(folder)        (no folder: this tree)
//   tree.runtime("forward.js")         a path; tree.runtimeUrl("forward.js") a file: URL, for import() and new Worker()
//   tree.python("engine/layout.py")    tree.python() is the folder Python's sys.path wants
//   tree.built("simdkernel_plain.wasm")   tree.served("models/tokenizer.bin")
//   tree.built("simdkernel.so", { maybe: true })   where the file would be, whether or not it is there (for a tool that
//                                      asks whether, or says "make kernels")
//   tree.folders                       { runtime, python, built, served }: the folders as the tree names them, for a
//                                      tool's words ("public", "src/runtime")
//   tree.pythonFiles("llama2_numpy")   the module's files under tree.python(): its window, then every .py of its
//                                      package's folder (a commit of before the packages has the window alone)
//   placePython(pyodide, tree, modules)   those files into Pyodide's file system, where `import` finds them
//   placeKernels(pyodide, tree)        simdkernel.so and simdkernel_relaxed.wasmlib of its built files, beside them
//
// and for this tree, by name:   import { runtime, runtimeUrl, python, built, served } from "./tree.ts";
//
//   node tests/tree.ts [--root <a tree>] runtime|python|built|served [<a file>]     the path, for the shell scripts
//
// What is still written down, in tests/tree.json, because it cannot be found by looking: the name of one file a kind
// (which file is "the runtime" is a decision, and "gpu.js" alone is two files); the folders that are no part of a tree;
// which folder is a Python module's package. And one thing is read off the Makefile: where the kernels are built, in a
// tree where nothing is built yet (the rule that makes simdkernel.so names the place).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = fileURLToPath(new URL("..", import.meta.url));
export const RULE = JSON.parse(fs.readFileSync(new URL("tree.json", import.meta.url), "utf8"));
const KINDS = Object.keys(RULE.anchors);
const trees = new Map();  // a root -> its tree: a tree is walked once a process

/** every file of a tree, as its path from the root with "/" between: no link is followed, no skipped folder entered */
function walked(root, { skip, anywhere = [] }) {
  const files = [];
  (function walk(folder, under) {
    const passed = under ? anywhere : [...skip, ...anywhere];  // (skip: names of the root's own folders)
    for (const entry of fs.readdirSync(folder, { withFileTypes: true })) {
      if (entry.isDirectory()) { if (!passed.includes(entry.name)) walk(path.join(folder, entry.name), `${under}${entry.name}/`); }
      else if (entry.isFile()) files.push(`${under}${entry.name}`);
    }
  })(root, "");
  return files;
}

export function treeOf(root = HERE, rule = RULE) {
  root = path.resolve(root);
  if (rule === RULE && trees.has(root)) return trees.get(root);
  let files;
  const tree = { root, walks: 0 };
  const looked = `looked in every folder but the root's ${rule.skip.join(", ")}${rule.anywhere?.length ? ` and any ${rule.anywhere.join(", ")}` : ""}`;
  const all = () => {
    if (!files) { files = walked(root, rule); tree.walks += 1; }
    return files;
  };
  /** the files whose path ends with name (a whole name, or the last names of a path) */
  const ending = (...names) => all().filter((file) => names.some((name) => file === name || file.endsWith(`/${name}`))).sort();
  /** a name as a kind may spell it (T.368.1): itself, then the same with the kind's other endings */
  const spelled = (kind, name) => {
    const endings = rule.either?.[kind] ?? [], has = endings.find((end) => name.endsWith(end) && !name.endsWith(`.d${end}`));
    return has ? [name, ...endings.filter((end) => end !== has).map((end) => name.slice(0, -has.length) + end)] : [name];
  };
  const one = (names, what) => {
    const found = ending(...names);
    if (found.length > 1) throw new Error(`${root}: ${what} is in ${found.length} places: ${found.join(" and ")}`);
    if (!found.length) throw new Error(`${root}: no ${what} anywhere in the tree (${looked})`);
    return found[0];
  };
  const folders = {};
  // a kind's folder: where its anchor is. The built files' of a tree where nothing is built: where its Makefile builds them
  const folder = (kind) => {
    if (kind in folders) return folders[kind];
    const anchor = rule.anchors[kind];
    if (kind === "built" && !ending(...spelled(kind, anchor)).length) {
      const made = fs.existsSync(path.join(root, "Makefile")) && new RegExp(`^(\\S*?)/?${anchor.replace(".", "\\.")}:`, "m").exec(fs.readFileSync(path.join(root, "Makefile"), "utf8"));
      if (!made) throw new Error(`${root}: no ${anchor} anywhere in the tree, and no rule of its Makefile makes one: where are the kernels built?`);
      return (folders[kind] = made[1]);
    }
    return (folders[kind] = path.posix.dirname(one(spelled(kind, anchor), `${anchor} (what says where the ${kind} files are)`)).replace(/^\.$/, ""));
  };
  const at = (kind) => (name = "", { maybe = false } = {}) => {
    const names = spelled(kind, name), under = names.map((spelling) => path.join(root, folder(kind), spelling));
    if (!name || maybe) return under[0];
    const there = under.filter((file) => fs.existsSync(file));
    if (there.length > 1) throw new Error(`${root}: ${name} is in ${folder(kind) || "the root"}/ (the ${kind} files) in ${there.length} spellings: ${there.map((file) => path.basename(file)).join(" and ")}`);
    if (there.length) return there[0];
    // not under the kind's folder: anywhere in the tree, by the end of its path
    const found = ending(...names.map((spelling) => spelling.replace(/\/$/, "")));
    if (found.length > 1) throw new Error(`${root}: ${name} is not in ${folder(kind) || "the root"}/ (the ${kind} files), and is in ${found.length} places: ${found.join(" and ")}`);
    if (!found.length) throw new Error(`${root}: no ${name} in ${folder(kind) || "the root"}/ (the ${kind} files) nor anywhere in the tree (${looked})`);
    return path.join(root, found[0]);
  };
  for (const kind of KINDS) tree[kind] = at(kind);
  tree.runtimeUrl = (name = "", options) => pathToFileURL(tree.runtime(name, options));
  Object.defineProperty(tree, "folders", { enumerable: true, get: () => Object.fromEntries(KINDS.map((kind) => [kind, folder(kind)])) });
  tree.pythonFiles = (module) => {
    const package_ = rule.packages[module];
    if (!package_) throw new Error(`tests/tree.json names no package of ${module}`);
    const window = tree.python(`${module}.py`), inside = path.join(tree.python(), package_);
    const parts = !fs.existsSync(inside) ? [] : fs.readdirSync(inside, { recursive: true })
      .map((name) => `${package_}/${name.split(path.sep).join("/")}`).filter((name) => name.endsWith(".py") && !name.includes("__pycache__")).sort();
    return [path.basename(window), ...parts];
  };
  if (rule === RULE) trees.set(root, tree);
  return tree;
}

/** The modules' Python files of a tree into Pyodide's file system (each folder made first). `under`, where two trees'
 * files are placed side by side: the folder in Pyodide to put them in. Returns the names placed. */
export function placePython(pyodide, tree, modules = Object.keys(RULE.packages), under = "") {
  const names = modules.flatMap((module) => tree.pythonFiles(module));
  for (const name of names) {
    const to = under ? `${under}/${name}` : name;
    if (to.includes("/")) pyodide.FS.mkdirTree(to.slice(0, to.lastIndexOf("/")));
    pyodide.FS.writeFile(to, fs.readFileSync(tree.python(name)));
  }
  return names;
}

/** The kernels Pyodide loads as side modules (ctypes), of a tree's built files, into Pyodide's file system */
export function placeKernels(pyodide, tree, names = ["simdkernel.so", "simdkernel_relaxed.wasmlib"]) {
  for (const name of names) pyodide.FS.writeFile(name, fs.readFileSync(tree.built(name)));
}

// this tree's, by name
const mine = (kind) => (...given) => treeOf()[kind](...given);
export const runtime = mine("runtime"), runtimeUrl = mine("runtimeUrl"), python = mine("python"), built = mine("built"), served = mine("served");

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const root = args[0] === "--root" ? args.splice(0, 2)[1] : HERE;
  const [kind, name = ""] = args;
  if (!KINDS.includes(kind) || args.length > 2) {
    console.error(`node tests/tree.ts [--root <a tree>] ${KINDS.join("|")} [<a file>]`);
    process.exit(2);
  }
  try { console.log(treeOf(root)[kind](name)); } catch (error) { console.error(error.message); process.exit(1); }
}
