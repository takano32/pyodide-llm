// T233 review probe (a throwaway branch): the 27B's section of tests/smoke.mjs, run alone in Pyodide (NumPy's integers are 32 bits
// there) against public/llama2_convert.py as it is on disk: tests/t233_int32_probe.sh puts the old arithmetic (int(np.prod(shape)))
// back one place at a time, and the section must fail on each. Needs `make kernels` for tests/engine.mjs only.
import fs from "node:fs";
import { pyodideWithEngine } from "./engine.mjs";

const smoke = fs.readFileSync(new URL("./smoke.mjs", import.meta.url), "utf8");
const from = smoke.indexOf("# T233: Ternary Bonsai 2 27B as the ternary checkpoint");
const to = smoke.indexOf("dim, hidden, layers, heads, vocab, positions = 32, 64, 2, 4, 320, 16");
if (from < 0 || to < from) throw new Error("the 27B's section of smoke.mjs is not where this probe expects it");
const block = smoke.slice(from, to).replaceAll("\\\\", "\\");
const { pyodide } = await pyodideWithEngine();
try {
  pyodide.runPython(`
import numpy as np
import llama2_convert
class Sized:
    def open(self, size, header, dtype, form):
        self.size = size
    def write(self, offset, array):
        pass
print("probe: NumPy's default integer here is", np.dtype(int).name, "and np.prod((64, 17408, 5120)) is", int(np.prod((64, 17408, 5120))))
${block}
print("probe: the 27B's section passes")
`);
} catch (error) {
  console.log(`probe: the 27B's section FAILS: ${String(error.message).trim().split("\n").slice(-2).join(" | ")}`);
  process.exit(1);
}
