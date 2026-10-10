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
