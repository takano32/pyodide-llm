// src/page/isolation.ts (T355): the service worker (public/coi.js) and the one reload of a first visit.
import { parameters } from "./address.ts";

declare const __BUILD__: string;

// ?v=<commit>: a new page must never run with the cached worker or Python code of the previous deployment
// T93 stage 3: cross-origin isolation, for the software threads of forward.js. GitHub Pages cannot send the headers,
// so a service worker (coi.js) adds them; it controls the page from the next load on, so the first visit reloads
// once. Nothing else starts before that. If it cannot register, or the page is still not isolated after the
// reload, the model runs on one thread as before. ?coi=off unregisters it (and stays on one thread).
export async function isolating(): Promise<boolean> {
  if (!("serviceWorker" in navigator)) {
    return false;
  }
  try {
    // ?coi=off, or a tab in which Pyodide did not load under the worker (the error branch of receiver.ts): without it
    if (parameters.get("coi") === "off" || sessionStorage.getItem("coi-fallback")) {
      for (const registration of await navigator.serviceWorker.getRegistrations()) {
        await registration.unregister();
      }
      return false;
    }
    // Registered every time: with the same URL nothing changes, with a new ?v= the browser installs the new
    // version and it takes over (skipWaiting, claim). A visitor who was isolated already kept the old version for
    // ever when the page skipped this (2026-09-25: their fetches through it and the new worker did not agree).
    // T111: it keeps copies for offline use, on every visit (the owner, 2026-09-25). ?offline=off turns that off
    // (remembered, so that every later visit registers the same one; it then throws the copies away), ?offline=on
    // back on.
    let offline = parameters.get("offline") !== "off";
    try {
      if (parameters.get("offline") === "off") localStorage.setItem("offline", "off");
      if (parameters.get("offline") === "on") localStorage.removeItem("offline");
      offline &&= localStorage.getItem("offline") !== "off";
    } catch {
      // no storage: only the address says
    }
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
