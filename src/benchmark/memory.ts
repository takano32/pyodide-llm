// The section "Page memory" (T173), which may end the tab: "Run all" leaves it out.
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
import { memoryResult, holdMemory, memoryLimit } from "../page-memory.js";
import { parameters, type Result } from "./dom.ts";
import { type Stage, staged, ask, worker } from "./section.ts";

// T173: the tab's sessionStorage, where it works (null: the page-memory section does not run)
export const tabStorage = (() => {
  try {
    sessionStorage.setItem("benchmark-probe", "1");
    sessionStorage.removeItem("benchmark-probe");
    return sessionStorage;
  } catch {
    return null;
  }
})();
export async function pageMemory(): Promise<Result> {
  const w = worker("benchmark/memory.js");
  // T173's review: a page hidden (another app, the screen locked) or left (a reload, another page) stops the run in
  // the event itself, and the memory is let go at once: a phone ends a hidden page far sooner, so a tab ended then is
  // no limit, and a page left is no tab ended. The run's own finally removes the mark in the microtasks right after
  // the event, before the page can be put away (tests/page-memory-check.mjs reloads a page in the middle of a run),
  // so only a tab that ends with no such event leaves its mark.
  let hide: () => void = () => {};
  const hidden = new Promise<void>((resolve) => (hide = () => resolve()));
  const stop = () => {
    w.terminate();
    hide();
  };
  const onVisibility = () => { if (document.visibilityState === "hidden") stop(); };
  document.addEventListener("visibilitychange", onVisibility);
  addEventListener("pagehide", stop);
  if (document.visibilityState === "hidden") stop();
  try {
    return memoryResult(await holdMemory({ ask: (message: any) => ask(w, message), storage: tabStorage, hidden,
      limit: memoryLimit(parameters.get("memoryMB")), stage: (s: Stage) => staged("memory", s) })) as Result;
  } finally {
    document.removeEventListener("visibilitychange", onVisibility);
    removeEventListener("pagehide", stop);
    w.terminate();
  }
}
