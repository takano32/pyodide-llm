// kernels-same.mjs (T356)
// Are the kernels this tree builds the ones another commit builds? For a change that only moves the kernels' sources
// (kernels/kernel.ts divided into kernel/*.ts): both trees' kernels/build.py is run (with this tree's node_modules)
// and every built file is compared, first byte for byte, then function by function.
//   node tests/kernels-same.mjs [--before <commit>]        (origin/main; needs `npm ci`, and about two minutes)
// A file's answer is one of
//   the same bytes
//   the same functions, in another order      every export is there, in the same order of the exports, and its code
//                                             is the same instruction for instruction (with what it calls); the
//                                             types, the globals and the memory are the same. Only the order of the
//                                             functions in the code section (and of the types) differs: Binaryen
//                                             sorts the functions that are used as often as each other by their
//                                             names, and AssemblyScript's name of a function begins with its file.
//   DIFFERENT                                 with the exports whose code differs: exit 1
// and public/'s built files must be this tree's (a kernel file the Makefile's rule does not name leaves them stale:
// then the checks test old kernels), or exit 1.
// The functions are compared as Binaryen's text of them (wasm-dis), where a call names the function called by its
// export, or by its own text where it has none (a function that calls itself: "itself"), and a type by what it is.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { otherTree } from "./other-tree.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const args = process.argv.slice(2);
const before = args.includes("--before") ? args[args.indexOf("--before") + 1] : "origin/main";
const work = path.join(root, ".tmp", "kernels-same");
const hash = (text) => createHash("sha256").update(text).digest("hex");

function build(tree, name) {
  const out = path.join(work, name);
  fs.rmSync(out, { recursive: true, force: true });
  fs.mkdirSync(out, { recursive: true });
  execFileSync("python3", [path.join(tree, "kernels", "build.py"), out], { cwd: root, stdio: ["ignore", "ignore", "inherit"] });
  return out;
}

/** a module as what does not depend on the order of its functions: the lines that are no function and no export (the
 * types sorted), the exports in their order, and each export's code */
function described(file) {
  const bytes = fs.readFileSync(file);
  // (a side module: Binaryen does not read "dylink.0", and it is the same few bytes in front of every one)
  const side = bytes[8] === 0 && bytes.subarray(10, 19).toString("latin1") === "\x08dylink.0";
  const plain = side ? Buffer.concat([bytes.subarray(0, 8), bytes.subarray(10 + bytes[9])]) : bytes;
  const taken = path.join(work, "module.wasm");
  fs.writeFileSync(taken, plain);
  const text = execFileSync("npx", ["wasm-dis", "--all-features", taken], { cwd: root, encoding: "utf8", maxBuffer: 1 << 28 });
  const functions = new Map(), exports = [], rest = [], types = [], typed = new Map();
  let current = null;
  for (const line of text.split("\n")) {
    const begun = line.match(/^ \(func (\$\S+)(.*)$/);
    if (begun) { current = [begun[2]]; functions.set(begun[1], current); continue; }
    if (current && line.startsWith("  ")) { current.push(line); continue; }
    if (current && line === " )") { current = null; continue; }
    current = null;
    const exported = line.match(/^ \(export "([^"]+)" \(func (\$\S+)\)\)$/);
    if (exported) exports.push([exported[1], exported[2]]);
    else if (/^ \(type /.test(line)) { const [, name, what] = line.match(/^ \(type (\$\S+) (.*)\)$/); typed.set(name, what); types.push(what); }
    else rest.push(line);
  }
  const names = new Map(exports.map(([name, f]) => [f, name]));
  const done = new Map(), open = new Set();
  const code = (f) => {
    if (done.has(f)) return done.get(f);
    if (open.has(f)) throw new Error(`${file}: functions that call each other in a ring`);
    open.add(f);
    if (!functions.has(f)) throw new Error(`${file}: no function ${f}`);
    const body = functions.get(f).join("\n").replace(/\(type (\$[^\s()]+)\)/g, (all, type) => `(type ${typed.get(type)})`).replace(/\((call|ref\.func|return_call) (\$[^\s()]+)/g,
      (all, how, callee) => `(${how} ${names.get(callee) ?? (callee === f ? "itself" : "the one of " + code(callee))}`);
    open.delete(f);
    done.set(f, hash(body));
    return done.get(f);
  };
  for (const f of functions.keys()) code(f);
  const helpers = [...functions.keys()].filter((f) => !names.has(f)).map(code).sort();
  return { rest: [...types.sort(), ...rest].join("\n"), order: exports.map(([name]) => name), code: new Map(exports.map(([name, f]) => [name, code(f)])), helpers, functions: functions.size, side };
}

const other = otherTree(before);
console.log(`the kernels of this tree against those of ${before} (${other.commit.slice(0, 7)})`);
const [mine, theirs] = [build(root, "tree"), build(other.folder, "before")];
const files = [...new Set([...fs.readdirSync(mine), ...fs.readdirSync(theirs)])].sort();
let failed = false, reordered = 0, identical = 0;
for (const file of files) {
  const [a, b] = [path.join(mine, file), path.join(theirs, file)];
  if (!fs.existsSync(a) || !fs.existsSync(b)) { console.log(`${file}: DIFFERENT: only ${fs.existsSync(a) ? "this tree" : before} builds it`); failed = true; continue; }
  const [x, y] = [fs.readFileSync(a), fs.readFileSync(b)];
  const sums = `sha256 ${hash(x).slice(0, 16)}${x.equals(y) ? "" : " (it was " + hash(y).slice(0, 16) + ")"}, ${x.length} bytes${x.length === y.length ? "" : " (it was " + y.length + ")"}`;
  if (x.equals(y)) { console.log(`${file}: the same bytes (${sums})`); identical += 1; continue; }
  const [p, q] = [described(a), described(b)];
  const said = [];
  if (p.rest !== q.rest || p.side !== q.side) said.push("the types, the imports, the globals or the memory");
  if (p.order.join() !== q.order.join()) said.push(`the exports or their order (${p.order.filter((n) => !q.order.includes(n)).concat(q.order.filter((n) => !p.order.includes(n))).join(", ") || "the order"})`);
  const moved = p.order.filter((name) => q.code.has(name) && p.code.get(name) !== q.code.get(name));
  if (moved.length) said.push(`the code of ${moved.join(", ")}`);
  if (p.helpers.join() !== q.helpers.join()) said.push("the functions that are not exported");
  if (said.length) { console.log(`${file}: DIFFERENT: ${said.join("; ")} (${sums})`); failed = true; continue; }
  console.log(`${file}: the same functions, in another order (${p.functions} functions, ${p.order.length} exports; ${sums})`);
  reordered += 1;
}
// public/'s are this tree's
let stale = 0;
for (const file of fs.readdirSync(mine)) {
  const built = path.join(root, "public", file);
  if (!fs.existsSync(built) || !fs.readFileSync(built).equals(fs.readFileSync(path.join(mine, file)))) {
    console.log(`public/${file}: ${fs.existsSync(built) ? "is not what this tree builds" : "is not there"}: make kernels (and is every file of kernels/ in the Makefile's rule?)`);
    stale += 1;
  }
}
if (!stale) console.log("public/: what this tree builds");
console.log(`kernels-same: ${failed || stale ? "FAILED" : "ok"} (${files.length} files: ${identical} the same bytes, ${reordered} the same functions in another order${failed ? ", the rest different" : ""}${stale ? `, ${stale} of public/ stale` : ""})`);
process.exit(failed || stale ? 1 : 0);
