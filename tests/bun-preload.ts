// tests/bun-preload.ts (T.584): what Bun runs before a tool of tests/ (bunfig.toml's preload), where the owner runs the
// tools with Bun too (2026-10-25). On the main thread, Bun keeps the process alive for ever once onmessage is set
// (oven-sh/bun#24256), and the tests import the workers' modules there (worker.ts, gpu.ts, the benchmark's), which set
// it as they load: a test then said ok and never ended. Node has no such global. So on the main thread onmessage is a
// plain property here, which holds what is set and is read back as it was; a worker thread keeps Bun's own (it is
// how a worker hears its messages).
import { isMainThread } from "node:worker_threads";

if (isMainThread) {
  let handler = null;
  Object.defineProperty(globalThis, "onmessage", { get: () => handler, set: (value) => { handler = value; }, configurable: true });
}
