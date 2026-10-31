// tests/receivers-check.ts (T.483, out of T.368.8's review): the three windows of a worker, the model's worker
// (src/runtime/worker.ts), its GPU worker (gpu.ts) and /benchmark/'s GPU worker (benchmark/gpu.ts), say at their heads
// that nothing is awaited before their receiver is set: a module worker's first message is lost where its top level
// waits before onmessage is there (AGENTS.md's pitfall: Node's worker_threads keep it, a browser does not; the page
// stood still for about 40 minutes once). Read here by the syntax tree: every top-level statement before the one
// that sets onmessage, outside the functions it declares, awaits nothing. The rule is tried on made-up texts first.
// Node only, under a second:
//
//   node tests/receivers-check.ts
import assert from "node:assert/strict";
import fs from "node:fs";
import { parse } from "@babel/parser";
import { runtime } from "./tree.ts";

const WINDOWS = ["worker.ts", "gpu.ts", "benchmark/gpu.ts"];

/** The awaits at a module's top level before its receiver is set ({ line, what }), or throws where it sets none */
export function awaitedBefore(text, name) {
  const { program } = parse(text, { sourceType: "module", plugins: ["typescript"] });
  const sets = (node) => node.type === "ExpressionStatement" && node.expression.type === "AssignmentExpression" &&
    ((node.expression.left.type === "Identifier" && node.expression.left.name === "onmessage") ||
     (node.expression.left.type === "MemberExpression" && node.expression.left.object.name === "self" && node.expression.left.property.name === "onmessage"));
  const at = program.body.findIndex(sets);
  if (at < 0) throw new Error(`${name} sets no onmessage at its top level`);
  const found = [];
  const walk = (node) => {
    if (!node || typeof node.type !== "string") return;
    // (a function's body runs when it is called, not as the module loads)
    if (/Function|Method|ClassBody/.test(node.type)) return;
    if (node.type === "AwaitExpression" || (node.type === "ForOfStatement" && node.await)) found.push({ line: node.loc.start.line, what: node.type });
    for (const [key, value] of Object.entries(node)) {
      if (key === "loc" || key.endsWith("Comments") || !value || typeof value !== "object") continue;
      (Array.isArray(value) ? value : [value]).forEach(walk);
    }
  };
  program.body.slice(0, at).forEach(walk);
  return found;
}

// the rule on made-up windows
const made = (head) => `${head}\nfunction later() { return (async () => { await 1; })(); }\nonmessage = ({ data }) => later(data);\nawait 2;\n`;
assert.deepEqual(awaitedBefore(made("const a = 1;"), "made"), [], "an await inside a function, or after the receiver, is none");
assert.deepEqual(awaitedBefore(made("await null;"), "made").map(({ line }) => line), [1], "a top-level await before the receiver");
assert.deepEqual(awaitedBefore(made("const b = [await null];"), "made").length, 1, "one inside an expression of a declaration");
assert.deepEqual(awaitedBefore(made("if (x) { for await (const y of z) {} }"), "made").length, 1, "a for await in a block");
assert.deepEqual(awaitedBefore(`await null;\nself.onmessage = () => {};\n`, "made").length, 1, "self.onmessage is a receiver too");
assert.throws(() => awaitedBefore("const a = 1;\n", "made"), /sets no onmessage/, "a window without a receiver");

for (const window of WINDOWS) {
  const found = awaitedBefore(fs.readFileSync(runtime(window), "utf8"), window);
  assert.deepEqual(found, [], `${window} awaits at its top level before its receiver is set (lines ${found.map(({ line }) => line).join(", ")}): its first message would be lost`);
}
console.log(`receivers-check: ok (${WINDOWS.join(", ")} await nothing before their receiver; the rule tried on 6 made-up texts)`);
