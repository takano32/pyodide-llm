// src/benchmark/running.ts (T378.5): the run that is going on: its sections, the one that is running, and when each
// began; and (T643.14) what each section found. The owner of that: run.ts says what happens to the functions below,
// and the other parts read these by the names they import (a static import of src/ is a live binding); none sets one. No clock is read here (the time
// is told) and nothing of a page: Node imports this file as it is (tests/page-parts-check.ts).
import type { Name, Result, Results } from "../bench.ts";

// the section that is running, and when it began
export let current: Name | undefined = undefined;
export let sectionAt = 0;
// the sections of the run and when it began: the seconds of the whole run are shown when it is more than one section
export let runNames: readonly Name[] = [];
export let runAt = 0;
let running = false;
// what the sections found, by their names: a new object each time one is told (window.__benchmark.results reads it,
// dom.ts), so a holder of an earlier one holds what was found then
export let results: Results = {};

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
/** A section found this: its own run ended, or (the page-memory section) a tab ended while it grew. */
export function sectionEnded<Section extends Name>(name: Section, result: Result<Section>) {
  results = { ...results, [name]: result };
}
/** The GPU section's tables were written again (T157: against a CPU section run after it). */
export function gpuRewritten(result: Result<"gpu">) {
  results = { ...results, gpu: result };
}
