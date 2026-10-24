// tests/silent.mjs (T.380.2): where a source catches a failure and tells nobody, read off the syntax tree. What
// tests/silent-check.mjs holds to its table, and what lists them for a person:
//
//   node tests/silent.mjs [--root <tree>]      every one of the tree: file, function, the text of the catch
//
//   silentIn(text, name)    the silent catches of one source: [{ where, what, line }]
//   sourcesOf(root)         the names of the sources that are read: src/**/*.{ts,js}, the scripts of src/pages/*.astro,
//                           and public/coi.js
//   silentOf(root)          the silent catches of a tree: [{ file, where, what, line }]
//
// A catch is silent when it neither reads what it caught nor throws: `catch {}`, `catch { kernels = null; }`,
// `catch { return false; }`, `.catch(() => {})`, `.catch(() => [])`, `.then(taken, () => undefined)`, and a handler
// that names what it caught and leaves it unread. One that reads the caught value (hands it on, says it, looks at its
// kind) or throws another is not: what becomes of the failure is then written there. `where` is the nearest function
// or variable that has a name ("(the module)" outside all), `what` the catch's own text on one line: for a try its
// handler, for a promise the whole call (`cache.delete(url).catch(() => {})`; a long one as its beginning, …, and its handler). No line is in
// either, so that a row of the table stays a row when lines are added above it.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "@babel/parser";
import { untyped } from "./untyped.mjs";

const SKIP = new Set(["loc", "start", "end", "range", "extra", "leadingComments", "trailingComments", "innerComments"]);
const children = (node) => Object.entries(node).filter(([key]) => !SKIP.has(key)).flatMap(([, value]) => (Array.isArray(value) ? value : [value]))
  .filter((value) => value && typeof value.type === "string");
const isFunction = (node) => /^(?:Arrow)?Function(?:Expression|Declaration)$|^(?:Object|Class)(?:Private)?Method$/.test(node.type);

// whether a name is read anywhere under a node (a shadowing declaration of the same name is rare enough to count as a read)
function reads(node, name) {
  if (node.type === "Identifier" && node.name === name) return true;
  return children(node).some((child) => reads(child, name));
}
// whether something is thrown under a node, by this function itself (a function written inside throws for its caller)
function throws(node) {
  if (node.type === "ThrowStatement") return true;
  return children(node).some((child) => !isFunction(child) && throws(child));
}
// the names a pattern binds
function bound(pattern) {
  if (!pattern) return [];
  if (pattern.type === "Identifier") return [pattern.name];
  return children(pattern).flatMap(bound);
}
const silentBody = (params, body) => !throws(body) && !params.flatMap(bound).some((name) => reads(body, name));

export function silentIn(text, name) {
  const code = untyped(text, name);
  const program = parse(code, { sourceType: "module", errorRecovery: false }).program;
  const one = (node) => code.slice(node.start, node.end).replace(/\/\/.*$/gm, "").replace(/\/\*[\s\S]*?\*\//g, "").replace(/\s+/g, " ")
    .replace(/ (?=[.,;)\]])/g, "").replace(/\{ \}/g, "{}").replace(/([(\[]) /g, "$1").trim();
  const found = [];
  const say = (node, where, what) => found.push({ where, what, line: node.loc.start.line });
  (function walk(node, where) {
    // the name a function or a variable gives to what is under it
    if (node.type === "FunctionDeclaration" && node.id) where = node.id.name;
    else if (node.type === "VariableDeclarator" && node.id.type === "Identifier" && node.init && isFunction(node.init)) where = node.id.name;
    else if (/Method$/.test(node.type) && node.key?.name) where = node.key.name;
    else if (/Property$/.test(node.type) && node.key?.name && node.value && isFunction(node.value)) where = node.key.name;
    else if (node.type === "AssignmentExpression" && isFunction(node.right)) where = one(node.left);

    if (node.type === "TryStatement" && node.handler && silentBody(node.handler.param ? [node.handler.param] : [], node.handler.body)) {
      say(node.handler, where, one(node.handler));
    }
    if (/^(?:Optional)?CallExpression$/.test(node.type) && /^(?:Optional)?MemberExpression$/.test(node.callee.type) && !node.callee.computed) {
      const method = node.callee.property.name;
      const handler = method === "catch" ? node.arguments[0] : method === "then" ? node.arguments[1] : undefined;
      if (handler) {
        // the whole call, so that what was asked is in it; a long one as its beginning and its handler
        const whole = one(node), tail = `.${method}(${node.arguments.map(one).join(", ")})`;
        const what = whole.length <= 100 ? whole : `${whole.slice(0, 60)} … ${tail.length <= 40 ? tail : `…${tail.slice(-38)}`}`;
        if (isFunction(handler) ? silentBody(handler.params, handler.body) : handler.type !== "Identifier" || /^(?:noop|quiet|nothing)$/.test(handler.name)) {
          say(node, where, what);
        }
      }
    }
    for (const child of children(node)) walk(child, where);
  })(program, "(the module)");
  return found.sort((one, other) => one.line - other.line);
}

// a page's scripts, each blanked around so that a line is the page's line
function scriptsOf(text) {
  const out = [];
  for (const match of text.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)) {
    const start = match.index + match[0].indexOf(">") + 1;
    out.push(text.slice(0, start).replace(/[^\n]/g, " ") + match[1]);
  }
  return out;
}

export function sourcesOf(root) {
  const names = fs.readdirSync(path.join(root, "src"), { recursive: true }).map((name) => `src/${String(name).split(path.sep).join("/")}`)
    .filter((name) => (/\.(?:ts|js|astro)$/.test(name) && !/\.d\.ts$/.test(name)) && !name.startsWith("src/runtime/built/"));
  return [...names, "public/coi.js"].sort();
}

export function silentOf(root) {
  return sourcesOf(root).flatMap((file) => {
    const text = fs.readFileSync(path.join(root, file), "utf8");
    if (file.endsWith(".astro")) return scriptsOf(text).flatMap((script) => silentIn(untyped(script, "page.ts"), "page.js")).map((one) => ({ file, ...one }));
    return silentIn(text, file).map((one) => ({ file, ...one }));
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const at = process.argv.indexOf("--root");
  const root = at > 0 ? path.resolve(process.argv[at + 1]) : fileURLToPath(new URL("..", import.meta.url));
  const all = silentOf(root);
  for (const { file, where, what, line } of all) console.log(`${file}:${line}  ${where}  ${what}`);
  const files = new Map();
  for (const { file } of all) files.set(file, (files.get(file) ?? 0) + 1);
  console.log(`\n${all.length} silent catches in ${files.size} files`);
}
