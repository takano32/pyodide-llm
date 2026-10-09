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
