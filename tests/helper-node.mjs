// tests/helper-node.mjs (T367.5): a software thread (src/runtime/helper.js) in Node. helper.js is a browser's module
// worker and nothing else: it claims its one message through self.onmessage, says "ready" with self.postMessage and
// ends with self.close(). Node's worker_threads has no `self`, so this file is the worker Node starts: it makes one
// of the thread's port and then imports the helper it was given, which runs as it is, to its last line. That is the
// sources' helper.js, or a build's _astro/helper.<hash>.js (its jobs.js linked into it: T367.5, what a browser runs).
//
//   softwareThread()                  a Worker of the helper the environment says: the sources', or with
//                                     ENGINE_FROM=dist [ENGINE_DIST=<a dist>] a build's (as tests/bundle.mjs's forwardUrl())
//   softwareThread(url)               a Worker of that helper
//
// The one message is taken with once() before the helper is imported (Node queues what comes before a listener, a
// browser does not: helper.js sets self.onmessage before its first await, and tests/e2e.mjs sees a browser's way).
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { builtModule } from "./bundle.mjs";
import { runtimeUrl } from "./tree.mjs";

/** helper.js as the tools read it: the sources', or with ENGINE_FROM=dist a build's (one file, jobs.js in it) */
export function helperUrl(env = process.env) {
  if (!env.ENGINE_FROM || env.ENGINE_FROM === "sources") return runtimeUrl("helper.js");
  if (env.ENGINE_FROM !== "dist") throw new Error(`ENGINE_FROM=${env.ENGINE_FROM}: "dist" (a build's helper) or "sources"`);
  return builtModule("helper", env.ENGINE_DIST || undefined);
}

export const softwareThread = (url = helperUrl()) => new Worker(new URL(import.meta.url), { workerData: { softwareThreadOf: String(url) } });

if (!isMainThread && workerData?.softwareThreadOf) {
  globalThis.self = {
    set onmessage(handler) { parentPort.once("message", (data) => handler({ data })); },
    postMessage: (message) => parentPort.postMessage(message),
    close: () => process.exit(0),
  };
  await import(workerData.softwareThreadOf);
}
