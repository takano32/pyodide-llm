// tests/leave.mjs (T357)
// The end of a tool: everything it wrote is out before the process ends. A pipe takes what is written to it when it has
// room, and process.exit() does not wait: gpu-check's log of a failed run ended in the middle of a line with no FAILED in
// it (T226), and answers.mjs's 12 answers stopped at the seventh and the run was a success (T274). One place for it.
//   await leave(failed ? 1 : 0);
export const flushed = () => Promise.all([process.stdout, process.stderr].map((stream) => new Promise((resolve) => stream.write("", resolve))));
export async function leave(code = 0) {
  await flushed();
  process.exit(code);
}

// T391.6: and the browser a tool drove, closed without waiting on it for long. A browser's close() may never return
// (T141: WebKit on Windows, three times in CI; the job then sat until its own limit, and each browser of a list waits
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
