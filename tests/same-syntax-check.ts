// tests/same-syntax-check.ts (T.481): tests/same-syntax.ts on made-up texts, and once as the tool it is. Node alone,
// under a second.
//
// What is the same code: types added, a file's specifiers renamed .ts, parentheses, the place of everything, the
// comments a bundler drops. What is not, each said: a number or a string written otherwise, a shorthand property
// opened, a template's text, a statement more, a name, a /*! */ or /** */ comment.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { differences, shape } from "./same-syntax.ts";

const JS = `/*! a licence */
// a line of words
import { b } from "./b.js";
/** what f is */
export function f(ms, ids) {
  const line = \`\${ms} ms\\n\`;
  return { ms, ids, line, small: 1e-3, name: "f", file: "model.js", more: (ms + 1) * 2, later: () => import("./c.js"), url: new URL("./w.js", import.meta.url) };
}
`;
const said = (was, now, wasName = "a.js", nowName = "a.js") => differences(shape(was, wasName), shape(now, nowName));
const same = (now, why, name = "a.js") => assert.deepEqual(said(JS, now, "a.js", name), [], why);
const differs = (now, expected, why, name = "a.js") => assert.deepEqual(said(JS, now, "a.js", name), expected, why);
const swap = (find, replace) => { assert.ok(JS.includes(find), find); return JS.replace(find, replace); };

// ---- the same code
same(JS, "a text against itself");
same(swap('import { b } from "./b.js";', 'import type { T } from "./t.ts";\nimport { b } from "./b.ts";').replace("export function f(ms, ids) {", "export function f(ms: number, ids: T[]): object {")
  .replace('import("./c.js")', 'import("./c.ts")').replace('"./w.js"', '"./w.ts"'), "types added and the specifiers renamed .ts", "a.ts");
same(swap("(ms + 1) * 2", "((ms + 1)) * (2)"), "parentheses that change nothing");
same(swap("// a line of words\n", "\n\n/* other words */\n"), "the comments a bundler drops, and where the lines are");
same(swap("// a line of words\n", "//* other words\n"), "a line's comment, whatever it starts with");
same(swap("  return {", "  return   {\n   "), "white space");

// ---- not the same, each said
differs(swap("{ ms, ids,", "{ ms: ms, ids,"), ["statement 2 differs (line 5)"], "a shorthand property opened: a bundler prints the two alike");
differs(swap("1e-3", ".001"), ["statement 2 differs (line 5)"], "a number written otherwise");
differs(swap('name: "f"', "name: 'f'"), ["statement 2 differs (line 5)"], "a string's quotes");
differs(swap("ms\\n`", "ms\\u000a`"), ["statement 2 differs (line 5)"], "a template's text as it is written (the same text when it is read)");
differs(swap("(ms + 1) * 2", "ms + 1 * 2"), ["statement 2 differs (line 5)"], "parentheses that change the tree");
differs(swap('import { b } from "./b.js";', 'import { b } from "./d.js";'), ["statement 1 differs (line 3)"], "another file imported");
differs(swap('import { b } from "./b.js";', 'import { b } from "b.ts";'), ["statement 1 differs (line 3)"], "a specifier that is no relative one is not evened out");
differs(swap('"model.js"', '"model.ts"'), ["statement 2 differs (line 5)"], "a string that ends .js and names no neighbour is a string");
differs(`${JS}f();\n`, ["2 statements were, 3 are"], "a statement more");
differs(swap("/*! a licence */", "/*! a license */"), ["its /*! */ and /** */ comments differ (2 were, 2 are)"], "a licence's text");
differs(swap("/** what f is */", "// what f is"), ["its /*! */ and /** */ comments differ (2 were, 1 are)"], "a /** */ comment made one a bundler drops");
differs(swap("export function f(ms, ids) {", "export function f(ms: number, ids = []) {"), ["statement 2 differs (line 5)"], "a default beside a type", "a.ts");
differs(`"use strict";\n${JS}`, ["2 statements were, 3 are"], "a directive");

// ---- the tool: this tree against its own commit is the same code but for what is being worked on, and it says its count
const run = spawnSync(process.execPath, [fileURLToPath(new URL("./same-syntax.ts", import.meta.url)), "--before", "HEAD", "src/runtime/jobs.ts", "src/models.ts"], { encoding: "utf8" });
assert.match(run.stdout, /^same-syntax: 2 files against HEAD: \d+ statements and \d+ \/\*! \*\/ and \/\*\* \*\/ comments; 0 new; (the same code|\d+ DIFFERENT)$/m, `the tool's last line: ${run.stdout}${run.stderr}`);
const none = spawnSync(process.execPath, [fileURLToPath(new URL("./same-syntax.ts", import.meta.url)), "--before"], { encoding: "utf8" });
assert.equal(none.status, 2, "--before without a commit is a usage error");
console.log("same-syntax-check: ok (6 texts that are the same code, 13 that are not, and the tool on this tree)");
