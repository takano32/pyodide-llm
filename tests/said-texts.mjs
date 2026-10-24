// tests/said-texts.mjs (T.380.1): every sentence the sources of what a browser runs may say, as one list, and whether
// two commits have the same ones.
//
//   node tests/said-texts.mjs                       the list of this tree (one JSON line a text: [count, text])
//   node tests/said-texts.mjs --before <commit>     what the two trees differ in (--before alone: origin/main);
//                                                   ends 1 where a text is in one and not in the other, or its count differs
//   node tests/said-texts.mjs --root <folder>       the list of another tree
//
// For a step that moves texts and is to change none (a failure's sentence to the place that says it, a reason to the
// table of the reasons): what a visitor reads is made of these, wherever they stand. A text: a string or a template of
// a .ts or .js source under src/ that has a space in it (a sentence or a part of one; a key, a name and a number are
// not). Of a template, its written parts with ${} where an expression stands: an expression may be renamed, the words
// around it not. Counted over all the files together, for a text may move from one to another. What is not read: the
// shaders (src/runtime/shaders/: WGSL, which tests/unchanged.mjs holds), the types (src/runtime/contracts/, and every
// type of a .ts file), the .astro pages' markup, the Python. A text built of parts in another way than before (two
// strings joined where one was) differs here although it reads the same: such a difference is to be read, each one.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "@babel/parser";
import { typed, untyped } from "./untyped.mjs";

const here = fileURLToPath(new URL("..", import.meta.url));
const LEFT_OUT = ["src/runtime/shaders/", "src/runtime/contracts/"];

function sources(root, folder = "src") {
  const found = [];
  for (const entry of fs.readdirSync(path.join(root, folder), { withFileTypes: true })) {
    const name = `${folder}/${entry.name}`;
    if (entry.isDirectory()) {
      if (!LEFT_OUT.includes(`${name}/`) && entry.name !== "built") found.push(...sources(root, name));
    } else if (/\.(?:ts|js)$/.test(entry.name) && !/\.d\.ts$/.test(entry.name)) found.push(name);
  }
  return found.sort();
}

/** The texts of one source: [text, …], a template as its parts around ${}. */
export function textsOf(text, name) {
  const tree = parse(typed(name) ? untyped(text, name) : text, { sourceType: "module", plugins: ["importAttributes"] });
  const said = [];
  const walk = (node, parent) => {
    if (!node || typeof node.type !== "string") return;
    // (what names a module is no sentence)
    if (node.type === "StringLiteral" && !/^(?:Import|Export)/.test(parent?.type ?? "") && parent?.type !== "ImportExpression") said.push(node.value);
    if (node.type === "TemplateLiteral") said.push(node.quasis.map((part) => part.value.cooked ?? part.value.raw).join("${}"));
    for (const [key, value] of Object.entries(node)) {
      if (key === "loc" || key === "leadingComments" || key === "trailingComments" || key === "innerComments") continue;
      if (Array.isArray(value)) value.forEach((child) => walk(child, node));
      else if (value && typeof value === "object") walk(value, node);
    }
  };
  walk(tree.program, null);
  return said.filter((one) => /\S \S/.test(one) && /[A-Za-z]{2}/.test(one));
}

/** Map(text → how often) over a tree's sources. */
export function saidIn(root) {
  const counts = new Map();
  for (const name of sources(root)) {
    for (const one of textsOf(fs.readFileSync(path.join(root, name), "utf8"), name)) counts.set(one, (counts.get(one) ?? 0) + 1);
  }
  return counts;
}

/** What two lists differ in: [[text, before, after], …] (0 where one has it not), by the text. */
export function differences(before, after) {
  return [...new Set([...before.keys(), ...after.keys()])].sort()
    .map((text) => [text, before.get(text) ?? 0, after.get(text) ?? 0]).filter(([, a, b]) => a !== b);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const value = (flag) => (args.includes(flag) ? args[args.indexOf(flag) + 1] : undefined);
  const root = value("--root") ?? here;
  if (!args.includes("--before")) {
    for (const [text, count] of [...saidIn(root)].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) console.log(JSON.stringify([count, text]));
  } else {
    const { otherTree } = await import("./other-tree.mjs");
    const { commit, folder } = otherTree(value("--before") && !value("--before").startsWith("--") ? value("--before") : "origin/main");
    const before = saidIn(folder), after = saidIn(root);
    const differ = differences(before, after);
    const total = (counts) => [...counts.values()].reduce((sum, count) => sum + count, 0);
    console.log(`said-texts: ${before.size} texts (${total(before)} written) at ${commit.slice(0, 7)}, ${after.size} (${total(after)}) here`);
    for (const [text, a, b] of differ) console.log(`  ${a} -> ${b}  ${JSON.stringify(text)}`);
    console.log(differ.length ? `said-texts: ${differ.length} texts differ` : "said-texts: the same texts, each as often");
    process.exitCode = differ.length ? 1 : 0;
  }
}
