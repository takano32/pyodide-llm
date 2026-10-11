// worker/outdated.ts (T439): the one thing this visit's worker tells the page of the files it went on without. A page
// left open across a deployment asks for files of its own build that are gone (updated.js's head). Where that ends
// what was asked, the visitor is told so by an error (T367.4). Where the engine has a fallback (NumPy for the kernels
// and forward.js, fewer threads for a software thread, the CPU for the GPU's worker) it takes it, as it always did,
// and the model answers: slower than it would, and until this nothing said why. Now the page is told, once,
// { type: "updated" }, and its status line says updated.js's SLOWER after what it said before.
//
// Only where the file is gone (updated.js's line: a built file's own 404, or the 404 of the module that names a
// module or a worker). A line that dropped, a page that is offline, an answer of the 500s, a request something
// blocked, a development server and Node fall back as before and nothing is added.
// Nothing waits for this: the caller goes on to its fallback at once, and the word comes when the server has answered
// (a HEAD, after a failure only; never on a visit where nothing failed). Nothing here throws, and no failure is
// turned into another: the fallbacks are what they were.
import { gone, isUpdated } from "../updated.js";
import type { Updated } from "../contracts/messages.ts";

// (the tests' own fetch and address, which gone() takes: nothing of the worker hands one)
type Probe = Parameters<typeof gone>[1];

let noticed = false;  // the page was told: it holds for the visit, and nothing is asked again
const asking = new Map<string, Promise<boolean>>();  // the HEADs under way, by the address asked for: failures at the same time share one
function goneNow(namer: string, probe?: Probe) {
  if (!asking.has(namer)) asking.set(namer, gone(namer, probe).finally(() => asking.delete(namer)));
  return asking.get(namer);
}
async function notice(whether: () => boolean | Promise<boolean> | undefined) {
  if (noticed || !(await whether()) || noticed) return noticed;
  noticed = true;
  postMessage({ type: "updated" } satisfies Updated);
  return true;
}

/** The engine goes on without a file that did not come: `failure` is what its fetch or its import threw. A built
 * file's own answer says whether it is gone (built()'s updated()); for a module, namer is the import.meta.url of the
 * module that names it, which is asked for. Resolves to whether the page has been told (no caller waits for it). */
export const wentWithout = (failure: unknown, namer?: string, probe?: Probe) =>
  notice(() => isUpdated(failure) || (namer !== undefined && goneNow(namer, probe)));

/** The engine goes on without a worker that did not start: `event` is its error event, namer the import.meta.url of
 * the module that made it. A script that threw says what (a message) and is no missing file; one that did not load
 * says nothing, and then the namer is asked for (updated.js's unstarted() has the same line). */
export const workerWentWithout = (event: { message?: string } | undefined, namer: string, probe?: Probe) => notice(() => !event?.message && goneNow(namer, probe));
