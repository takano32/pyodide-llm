// Cross-origin isolation (T93): the service worker registered as the model page registers it, and the one reload.
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
// T643.8: the steps are the model page's own isolating() (src/page/isolation.ts); what /benchmark/ does apart is said here
import { parameters } from "./dom.ts";
import { isolating } from "../page/isolation.ts";

// whether the worker keeps its copies for offline use: as the model page's ?offline=off left it (remembered there;
// /benchmark/ reads it and writes nothing)
function offlineAsKept(): boolean {
  try {
    return localStorage.getItem("offline") !== "off";
  } catch {
    // no storage: the copies are kept, as by default
    return true;
  }
}

// ---- cross-origin isolation, for the software threads and the GPU section's bridge: ?coi=off as on the model page
// (T582, T507), and a tab in which the model page gave up on the worker (Pyodide did not load under it, T113) stays
// without it
if (await isolating({ off: parameters.get("coi") === "off", fallback: "stays", offline: offlineAsKept })) await new Promise(() => {});
