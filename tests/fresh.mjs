// tests/fresh.mjs (T.584): a module of the sources made anew, with nothing of an earlier import of it: what the tests of
// an owner of state (page-parts-check, worker-parts-check, worker-outdated-check) start each case from. Node makes a new
// module of a URL with another query (?case=N), and the modules it imports are the ones there are; Bun makes one module
// of a file whatever the query, so there the module is a copy of the file beside it (the same folder: its imports are
// the same files), taken away once it is imported (the tools that read the folder are not to see it), or at the
// process's end where the import failed.
//
//   fresh(url)      the module of a file: URL (a string or a URL), new
import fs from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

let made = 0;
const copies = [];
process.on("exit", () => {
  for (const copy of copies) fs.rmSync(copy, { force: true });
});

export async function fresh(url) {
  const file = fileURLToPath(String(url).replace(/[?#].*$/, ""));
  if (!process.versions.bun) return import(`${pathToFileURL(file).href}?case=${made++}`);
  const copy = file.replace(/(\.[cm]?[jt]s)$/, `.fresh-${process.pid}-${made++}$1`);
  fs.copyFileSync(file, copy);
  copies.push(copy);
  try {
    return await import(pathToFileURL(copy).href);
  } finally {
    fs.rmSync(copy, { force: true });
  }
}
