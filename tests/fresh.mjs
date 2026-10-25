// tests/fresh.mjs (T.584): a module of the sources made anew, with nothing of an earlier import of it: what the tests of
// an owner of state (page-parts-check, worker-parts-check, worker-outdated-check) start each case from. A specifier with
// another query (?case=N) makes a new module, and the modules it imports are the ones there are. Node takes it as a
// file: URL; Bun drops the query of a file: URL and makes one module of the file (oven-sh/bun#46900), but keeps it on
// an absolute path, so there the specifier is the path (T.584.2).
//
//   fresh(url)      the module of a file: URL (a string or a URL), new
import { fileURLToPath, pathToFileURL } from "node:url";

let made = 0;

export function fresh(url) {
  const file = fileURLToPath(String(url).replace(/[?#].*$/, ""));
  return import(`${process.versions.bun ? file : pathToFileURL(file).href}?case=${made++}`);
}
