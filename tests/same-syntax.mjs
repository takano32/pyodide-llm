// tests/same-syntax.mjs (T.481): whether the sources of two commits are the same code, by their syntax trees.
//
//   node tests/same-syntax.mjs [--before <commit>] [file …]      (--before: origin/main; the files: every .js and .ts
//                                                                 source git knows under src/, and public/coi.js)
//
// For a step that is to change no code (types added, a file renamed .ts, lines moved by a formatter): each file of the
// working tree against the same file of the other commit, statement by statement, with the types blanked
// (tests/untyped.mjs). tests/dist-same.mjs says the same of two builds, and a bundler prints some differences away
// before it can: a shorthand property opened ({ ms } to { ms: ms }), 1e-3 written .001, a string's quotes. Here those
// are differences. What is held:
//   - every statement's tree, with each string's and each number's text as it is written (raw), a template's raw and
//     cooked texts, and whether a property is a shorthand;
//   - the /*! … */ and /** … */ comments, in their order (the ones a bundler keeps: licences are /*! … */).
// What is evened out, and nothing else: where a node stands, the other comments, parentheses, and the ending of a
// relative specifier ("./x.js" and "./x.ts" are one name: the file was renamed).
// The file of the other commit: the one git says this one was renamed from, else the same path, else the same path
// with the other ending (.js for .ts); a file with none is new, and said. A file the other commit had and this tree
// has not is not looked for. Not read: the scripts inside the .astro pages.
// Ends 1 where a file differs.
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parse } from "@babel/parser";
import { typed, untyped } from "./untyped.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const PLACES = new Set(["loc", "start", "end", "range", "leadingComments", "trailingComments", "innerComments"]);
const SPECIFIER = /^(\.\.?\/.*)\.(?:js|ts)$/;

/** a node as what is compared: no places, no comments, of `extra` only how a literal is written */
function plain(node) {
  if (Array.isArray(node)) return node.map(plain);
  if (!node || typeof node !== "object") return node;
  const out = {};
  for (const [key, value] of Object.entries(node)) {
    if (PLACES.has(key)) continue;
    if (key === "extra") { if (value?.raw !== undefined) out.raw = value.raw; continue; }
    out[key] = plain(value);
  }
  if (node.type === "StringLiteral" && SPECIFIER.test(node.value)) {
    out.value = node.value.replace(SPECIFIER, "$1.js");
    out.raw = node.extra.raw.replace(/\.(?:js|ts)(["'])$/, ".js$1");
  }
  return out;
}

/** { statements: each top-level statement as a text to compare, lines: where each starts, notices: the comments a bundler keeps } */
export function shape(text, name) {
  const file = parse(untyped(text, name), { sourceType: "module" });
  return {
    statements: [...file.program.directives, ...file.program.body].map((statement) => JSON.stringify(plain(statement))),
    lines: [...file.program.directives, ...file.program.body].map((statement) => statement.loc.start.line),
    notices: file.comments.filter((comment) => comment.type === "CommentBlock" && /^[!*]/.test(comment.value)).map((comment) => comment.value),
  };
}

/** what differs between two shapes, as lines to print (none: the same code) */
export function differences(was, now) {
  const said = [];
  if (was.statements.length !== now.statements.length) said.push(`${was.statements.length} statements were, ${now.statements.length} are`);
  else was.statements.forEach((statement, i) => { if (statement !== now.statements[i]) said.push(`statement ${i + 1} differs (line ${now.lines[i]})`); });
  if (JSON.stringify(was.notices) !== JSON.stringify(now.notices)) said.push(`its /*! */ and /** */ comments differ (${was.notices.length} were, ${now.notices.length} are)`);
  return said;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2), at = args.indexOf("--before");
  const before = at < 0 ? "origin/main" : args.splice(at, 2)[1];
  if (!before) { console.error("usage: node tests/same-syntax.mjs [--before <commit>] [file …]"); process.exit(2); }
  const git = (...command) => execFileSync("git", command, { cwd: root, encoding: "utf8", maxBuffer: 1 << 28, stdio: ["ignore", "pipe", "pipe"] });
  const source = (name) => /\.(js|ts)$/.test(name) && !/\.d\.ts$/.test(name);
  const files = args.length ? args.map((name) => path.relative(root, path.resolve(name)).split(path.sep).join("/"))
    : git("ls-files", "--cached", "--others", "--exclude-standard", "--", "src", "public/coi.js").trim().split("\n").filter(source).filter((name) => fs.existsSync(path.join(root, name)));
  // (what git reads as renamed between that commit and this tree: R<score> <old> <new>)
  const renamed = new Map(git("diff", "-M", "--name-status", before, "--").split("\n").filter((line) => line.startsWith("R")).map((line) => line.split("\t")).map(([, old, name]) => [name, old]));
  const there = (name) => { try { return git("show", `${before}:${name}`); } catch { return null; } };
  const other = (name) => [renamed.get(name), name, typed(name) ? name.replace(/\.ts$/, ".js") : name.replace(/\.js$/, ".ts")].filter(Boolean);
  let differing = 0, statements = 0, notices = 0, fresh = 0;
  for (const name of files) {
    const now = shape(fs.readFileSync(path.join(root, name), "utf8"), name);
    const old = other(name).map((candidate) => [candidate, there(candidate)]).find(([, text]) => text !== null);
    if (!old) { fresh += 1; console.log(`same-syntax: ${name}: new (${before} has no such file)`); continue; }
    const said = differences(shape(old[1], old[0]), now);
    statements += now.statements.length;
    notices += now.notices.length;
    if (said.length) differing += 1;
    for (const line of said) console.log(`same-syntax: ${name}${old[0] === name ? "" : ` (${old[0]})`}: ${line}`);
  }
  console.log(`same-syntax: ${files.length} files against ${before}: ${statements} statements and ${notices} /*! */ and /** */ comments; ` +
    `${fresh} new; ${differing ? `${differing} DIFFERENT` : "the same code"}`);
  process.exit(differing ? 1 : 0);
}
