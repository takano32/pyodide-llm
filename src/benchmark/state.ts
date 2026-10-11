// The one object the parts of /benchmark/'s script share.
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
import { type Name } from "./dom.ts";

// T353: what more than one part of this script reads and runSections() sets (a module's `let` cannot be assigned from
// another module). A field is no variable, and a wrong name reads undefined without a word:
// tests/page-modules-check.mjs holds every page.<name> to this list.
export const page = {
  // the section that is running, and when it began
  current: undefined as Name | undefined,
  sectionBegan: 0,
  // the seconds of the running section, and of the whole run when it is more than one section
  runBegan: 0,
  runNames: [] as readonly Name[],
};

// ---- T378.5: what happens to it. run.ts says these and sets no field itself. No clock is read here: the time is told.
let running = false;
/** A run of these sections begins. While one is running another does not: false. */
export function runBegan(names: readonly Name[], now: number) {
  if (running) return false;
  running = true;
  page.runNames = names;
  page.runBegan = now;
  return true;
}
/** The next section of the run begins. */
export function sectionBegan(name: Name, now: number) {
  page.current = name;
  page.sectionBegan = now;
}
/** The run is over: no section is running, and another run may begin. */
export function runEnded() {
  page.current = undefined;
  running = false;
}
