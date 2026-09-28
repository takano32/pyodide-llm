// T129's broken-on-purpose run (the probe branch only, never main): each mutation undoes one fix in public/worker.js,
// and tests/worker-check.mjs must fail on it. Prints one line per mutation and exits 1 if any passed.
//   node tests/t129-mutants.mjs
import { execFileSync } from "node:child_process";
import fs from "node:fs";

const file = "public/worker.js";
const original = fs.readFileSync(file, "utf8");
const MUTANTS = [
  ["(1) the version asked without a deadline", 'resolved?specifier=latest", { signal: given.signal })', 'resolved?specifier=latest")'],
  ["(1) the version's error not Pyodide's", "seconds`), { pyodide: true });", "seconds`), {});"],
  ["(2) NumPy fetched with integrity", 'loaded.loadPackage("numpy", { checkIntegrity: false })', 'loaded.loadPackage("numpy")'],
  ["(3) a failed part does not stop the others", "      inner.abort(error);  // T129 (3): the other connections stop with the part that failed for good\n", ""],
  ["(3) the parts fetched on the load's signal", "fetchPart(partUrl(part), model, part, inner.signal)", "fetchPart(partUrl(part), model, part, signal)"],
  ["(3) a refused write fetched again", "throw Object.assign(error, { final: true });", "throw error;"],
  ["(3) the load cannot stop the download", "source.stop = (why) => inner.abort(why);", "source.stop = () => {};"],
  ["(3) a failed range does not stop the others", "  } catch (error) {\n    inner.abort(error);\n    throw error;\n  } finally {\n    inner.done();", "  } catch (error) {\n    throw error;\n  } finally {\n    inner.done();"],
  ["(3) the ranges fetched on the load's signal", "const inner = innerAbort(outer), signal = inner.signal;", "const inner = innerAbort(outer), signal = outer;"],
  ["(4) a part answered 5xx not fetched again", "{ final: !worthRetrying(res.status) }", "{ final: true }"],
  ["(4) the progress goes back", "if (percent > reported) {", "if (percent !== reported) {"],
  ["(4) the third failure does not name the part", "throw new Error(`Part ${part} of ${model.checkpoint} failed three times: ${error.message ?? error}`, { cause: error });", "throw error;"],
  ["(5) 429 in the words of any status", "    : res.status === 429 ?", "    : false ?"],
  ["(5) 408 not asked again (ranges)", "(error.status >= 400 && !worthRetrying(error.status))", "(error.status >= 400 && error.status < 500)"],
  ["(5) 408 not asked again (anywhere)", "status >= 500 || status === 408", "status >= 500"],
  ["(6) the wrapper tried on every response", "if (!wraps || !res.body", "if (!res.body"],
  ["(7) a model past a 64-bit memory not refused", "if (forwardModule.pastWide(size, after)) {", "if (false) {"],
];
let passed = 0;
try {
  for (const [name, from, to] of MUTANTS) {
    const at = original.indexOf(from);
    if (at < 0 || original.indexOf(from, at + 1) >= 0) throw new Error(`${name}: the text to change is not there once`);
    fs.writeFileSync(file, original.replace(from, to));
    let failed = false, last = "";
    try {
      execFileSync("node", ["tests/worker-check.mjs"], { stdio: "pipe", timeout: 240000 });
    } catch (error) {
      failed = true;
      // where it failed: after which check, and the assertion's own line
      const oks = String(error.stdout ?? "").trim().split("\n").filter((line) => line.startsWith("ok: "));
      const why = String(error.stderr ?? error.message).split("\n").filter((line) => /^(AssertionError|\w*Error:|worker-check: still)/.test(line.trim()))[0] ?? "";
      last = `after ${oks.length} checks: ${why.trim()}`;
    }
    console.log(`${failed ? "caught" : "PASSED"}: ${name}${failed ? ` — ${last.slice(0, 200)}` : ""}`);
    if (!failed) passed++;
  }
} finally {
  fs.writeFileSync(file, original);
}
console.log(`t129-mutants: ${MUTANTS.length - passed} of ${MUTANTS.length} caught`);
process.exit(passed ? 1 : 0);
