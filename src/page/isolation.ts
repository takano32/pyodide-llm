// src/page/isolation.ts (T355; T643.8: the one of both pages): the site's service worker (public/coi.js), the one
// reload of a first visit, and taking the worker away. The model page (receiver.ts) and /benchmark/
// (src/benchmark/isolation.ts) call the same isolating(), each saying what it does apart (Isolating), and the model
// page's error row takes the worker away with the same unregistered(). It imports nothing, so that /benchmark/'s script
// takes nothing more of the model page with it.

declare const __BUILD__: string;

/** What a page does apart, of the steps of isolating() */
export type Isolating = {
  /** ?coi=off: the worker taken away (and the page has no isolation) */
  off: boolean;
  /** A tab in which the model page gave up on the worker (receiver.ts's mark coi-fallback: Pyodide did not load
   * under it, T113): "unregisters" takes the worker away as ?coi=off does (the model page, which wrote the mark),
   * "stays" goes on without it and leaves the registration as it is (/benchmark/) */
  fallback: "unregisters" | "stays";
  /** whether the worker keeps its copies for offline use (T111): read once, after ?coi=off and the mark */
  offline: () => boolean;
};

/** Every registration of the site's worker taken away (each awaited) */
export async function unregistered(): Promise<void> {
  for (const registration of await navigator.serviceWorker.getRegistrations()) {
    await registration.unregister();
  }
}

// ?v=<commit>, on the Service Worker's registration: a new deployment installs its own (the workers and the Python are
// named by a hash of their text since T367.3, and need none)
// T93 stage 3: cross-origin isolation, for the software threads of forward.ts (and /benchmark/'s GPU section's
// bridge). GitHub Pages cannot send the headers, so a service worker (coi.js) adds them; it controls the page from the
// next load on, so the first visit reloads once. Nothing else starts before that. If it cannot register, or the page
// is still not isolated after the reload, the model runs on one thread as before. ?coi=off unregisters it (and stays
// on one thread).
/** Whether the page reloads (nothing else may start then) */
export async function isolating(how: Isolating): Promise<boolean> {
  if (!("serviceWorker" in navigator)) {
    return false;
  }
  try {
    // ?coi=off, or a tab in which Pyodide did not load under the worker (the error branch of receiver.ts): without it
    // (the mark read only where it is asked, in the order the two pages asked it)
    const fallback = () => sessionStorage.getItem("coi-fallback");
    if (how.off || (how.fallback === "unregisters" && fallback())) {
      await unregistered();
      // (T507, T522, T582: a page that came through the worker has its headers, isolated, and the model's worker did
      // not start under them: it loads again without them. Not a page that is that load itself, where a browser still
      // gave it the headers. No mark in sessionStorage: the isolated page and the one without them do not share it)
      const loaded = performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined;
      if (self.crossOriginIsolated && loaded?.type !== "reload") {
        location.reload();
        return true;
      }
      return false;
    }
    if (how.fallback === "stays" && fallback()) {
      return false;
    }
    // Registered every time: with the same URL nothing changes, with a new ?v= the browser installs the new
    // version and it takes over (skipWaiting, claim). A visitor who was isolated already kept the old version for
    // ever when the page skipped this (2026-09-25: their fetches through it and the new worker did not agree).
    // T111: it keeps copies for offline use, on every visit (the owner, 2026-09-25), unless the page says not to
    // (the model page's ?offline=off, remembered: remembered.ts's offlineKept())
    const offline = how.offline();
    await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}coi.js?v=${__BUILD__}${offline ? "" : "&offline=0"}`,
      { scope: import.meta.env.BASE_URL });
    if (self.crossOriginIsolated) {
      return false;
    }
    await navigator.serviceWorker.ready;
    // once per tab: a browser that is not isolated even under the worker must not reload for ever
    if (sessionStorage.getItem("coi-reloaded")) {
      return false;
    }
    sessionStorage.setItem("coi-reloaded", "1");
    location.reload();
    return true;
  } catch {
    return false;
  }
}
