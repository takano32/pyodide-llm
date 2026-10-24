import fs from "node:fs";
import path from "node:path";
// tests/leave.mjs (T.357)
// The end of a tool: everything it wrote is out before the process ends. A pipe takes what is written to it when it has
// room, and process.exit() does not wait: gpu-check's log of a failed run ended in the middle of a line with no FAILED in
// it (T.226), and answers.mjs's 12 answers stopped at the seventh and the run was a success (T.274). One place for it.
//   await leave(failed ? 1 : 0);
export const flushed = () => Promise.all([process.stdout, process.stderr].map((stream) => new Promise((resolve) => stream.write("", resolve))));
// (T.580: and 200 ms after it, as T.575 has plan-facts-check end: Node now and then stands still in process.exit() right
// after heavy work (T.384), as blocks-forward-check did once in the light suite after its "ok")
export async function leave(code = 0) {
  await flushed();
  await new Promise((resolve) => setTimeout(resolve, 200));
  process.exit(code);
}

// T.391.6: and the browser a tool drove, closed without waiting on it for long. A browser's close() may never return
// (T.141: WebKit on Windows, three times in CI; the job then sat until its own limit, and each browser of a list waits
// for the one before it), and by then the tool knows its verdict. So the close is given 15 seconds, the tool goes on
// whether it came back, and ends with leave(): what the browser left behind does not keep the process.
//   const done = await closed(() => browser.close());     // true: it closed; false: it did not in time, or it threw
//   await leave(failed ? 1 : 0);
// It never throws: a close that fails after the verdict is no verdict.
export async function closed(close, seconds = 15) {
  let timer;
  try {
    return await Promise.race([Promise.resolve(close()).then(() => true), new Promise((resolve) => { timer = setTimeout(() => resolve(false), seconds * 1000); })]);
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

// T.512: a folder a tool needs for its run (a browser's profile on disk, a made-up model), under the repository's .tmp/
// and gone when the process ends, however it ends. Never the system's temp folder: on the development machine /tmp is
// memory, and the browser profiles of E2E_TWICE (hundreds of MB each, a model kept in OPFS) were left there, 3 GB of
// them in two days (the owner's rule: work files in .tmp/, nothing in $HOME or /tmp).
//   const profile = scratch("e2e-profile-");
const made = [];
export function scratch(prefix) {
  const root = path.join(new URL("../", import.meta.url).pathname, ".tmp");
  fs.mkdirSync(root, { recursive: true });
  const folder = fs.mkdtempSync(path.join(root, prefix));
  if (!made.length) process.on("exit", () => { for (const one of made) fs.rmSync(one, { recursive: true, force: true }); });
  made.push(folder);
  return folder;
}
