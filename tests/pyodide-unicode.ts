// T678: the Unicode of the latest Pyodide's Python against the one engine.tokenizer's UNNORMALIZED was made against
// (UNNORMALIZED_UNICODE, T663). The page follows the latest Pyodide (the second rule of AGENTS.md); a Python of a newer
// Unicode reads the characters it adds otherwise than the real tokenizers' NFC and NFKC, and the test of every code
// point runs on CI's Python 3.14 (T630), which never names them. Said, never failed (only new characters differ, which is
// no reason to stop a deploy): the full suite runs it every night, and the line starts with "suite: " so that
// tests/ci.ts's lines of tests.yml (and so `node tests/ci.ts nightly`) show it, besides a warning on the run's page.
//
//   node tests/pyodide-unicode.ts          (after `npm install --no-save pyodide@latest`, as the suite does)
import { loadPyodide, version } from "pyodide";
import { leave } from "./leave.ts";
import { placePython, treeOf } from "./tree.ts";

const pyodide = await loadPyodide();
placePython(pyodide, treeOf(), ["llama2_numpy"]);
const [python, table] = pyodide.runPython(`
import unicodedata
from engine.tokenizer import UNNORMALIZED_UNICODE
(unicodedata.unidata_version, UNNORMALIZED_UNICODE)
`).toJs();
const at = `Pyodide ${version}, Python ${pyodide.runPython("import sys; sys.version.split()[0]")}`;
if (python === table) {
  console.log(`pyodide-unicode: ${at} is of Unicode ${python}, as engine.tokenizer's UNNORMALIZED was made against`);
} else {
  const said = `${at} is of Unicode ${python}, and engine.tokenizer's UNNORMALIZED was made against ${table}: ` +
    "the characters the two tell apart are read otherwise than by the real tokenizers' NFC and NFKC. Make the table " +
    "again (AGENTS.md, Tokenizers, T663's item)";
  console.log(`suite: warning: ${said}`);
  console.log(`::warning title=Pyodide's Unicode moved::${said}`);
}
await leave(0);
