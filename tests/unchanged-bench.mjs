// unchanged-bench.mjs (T353 review): a preload for `node --import`, that writes down every call tests/bench.mjs makes to a
// name src/bench.js exports (the arguments and what came back, a function as its text), into the file UNCHANGED_RECORD.
// tests/unchanged.mjs runs it in two trees and compares the records: a table's header word, a cell's rounding, a
// warning's condition or a summary line that a division of src/bench.js moved (and mistyped) is a call that answers
// otherwise. (tests/bench.mjs holds a few of the tables' words; the GPU section's tables only here, and in CI's browsers
// with a GPU, which read no word of them.)
import fs from "node:fs";
import { register } from "node:module";

const record = [];
globalThis.__benchRecord = record;
const written = (value) => {
  try { return JSON.stringify(value, (key, inner) => (typeof inner === "function" ? String(inner) : typeof inner === "bigint" ? String(inner) : inner)) ?? "undefined"; }
  catch (error) { return `unwritable ${error.message}`; }
};
globalThis.__benchWritten = written;
process.on("exit", () => fs.writeFileSync(process.env.UNCHANGED_RECORD, JSON.stringify(record)));

register("data:text/javascript," + encodeURIComponent(`
export async function load(url, context, next) {
  if (!/\\/src\\/bench\\.js$/.test(url)) return next(url, context);
  const names = Object.keys(await import(url + "?real"));
  const source = 'import * as real from ' + JSON.stringify(url + "?real") + ';\\n' +
    'const wrap = (name, f) => typeof f !== "function" ? f : function (...args) { const answer = f.apply(this, args);' +
    ' globalThis.__benchRecord.push([name, globalThis.__benchWritten(args), globalThis.__benchWritten(answer)]); return answer; };\\n' +
    names.map((name) => 'export const ' + name + ' = wrap(' + JSON.stringify(name) + ', real.' + name + ');').join("\\n");
  return { format: "module", source, shortCircuit: true };
}`));
