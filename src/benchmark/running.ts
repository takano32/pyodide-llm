// src/benchmark/running.ts (T.378.5): the run that is going on: its sections, the one that is running, and when each
// began. The owner of that: run.ts says what happens to the functions below, and section.ts and run.ts read these by
// the names they import (a static import of src/ is a live binding); neither sets one. No clock is read here (the time
// is told) and nothing of a page: Node imports this file as it is (tests/page-parts-check.mjs).
import type { Name } from "./dom.ts";

// the section that is running, and when it began
export let current: Name | undefined = undefined;
export let sectionAt = 0;
// the sections of the run and when it began: the seconds of the whole run are shown when it is more than one section
export let runNames: readonly Name[] = [];
export let runAt = 0;
let running = false;

/** A run of these sections begins. While one is running another does not: false. */
export function runBegan(names: readonly Name[], now: number) {
  if (running) return false;
  running = true;
  runNames = names;
  runAt = now;
  return true;
}
/** The next section of the run begins. */
export function sectionBegan(name: Name, now: number) {
  current = name;
  sectionAt = now;
}
/** The run is over: no section is running, and another run may begin. */
export function runEnded() {
  current = undefined;
  running = false;
}
