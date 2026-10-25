// Cross-origin isolation (T.93): the service worker registered as the model page registers it, and the one reload.
// (T.353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
import { BASE, V, parameters } from "./dom.ts";

// ---- cross-origin isolation, for the software threads and the GPU section's bridge: the site's service worker
// (coi.js, T.93), registered as the model page registers it, and one reload the first time it takes the page over
async function isolating(): Promise<boolean> {
  if (!("serviceWorker" in navigator)) return false;
  try {
    // ?coi=off, as on the model page (T.582, T.507): the worker unregistered, and a page that came through it with its
    // headers loaded again without them (not a page that is that load itself)
    if (parameters.get("coi") === "off") {
      for (const registration of await navigator.serviceWorker.getRegistrations()) await registration.unregister();
      const loaded = performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined;
      if (!self.crossOriginIsolated || loaded?.type === "reload") return false;
      location.reload();
      return true;
    }
    // a tab in which the model page gave up on it (Pyodide did not load under it, T.113) stays without it
    if (sessionStorage.getItem("coi-fallback")) return false;
    let offline = true;
    try {
      offline = localStorage.getItem("offline") !== "off";  // the model page's ?offline=off, remembered
    } catch {
      // no storage: the copies are kept, as by default
    }
    await navigator.serviceWorker.register(`${BASE}coi.js${V}${offline ? "" : "&offline=0"}`, { scope: BASE });
    if (self.crossOriginIsolated) return false;
    await navigator.serviceWorker.ready;
    if (sessionStorage.getItem("coi-reloaded")) return false;
    sessionStorage.setItem("coi-reloaded", "1");
    location.reload();
    return true;
  } catch {
    return false;
  }
}
if (await isolating()) await new Promise(() => {});
