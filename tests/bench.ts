// src/bench.ts: the table a visitor pastes into an issue (T45). A plain module, so this runs in Node alone.
//
//   node tests/bench.ts
//
// T643.13: the asserts are in tests/bench/, a part for each of src/bench/'s modules, run here in the order they were
// written in (a part reads by name what an earlier part made; tests/unchanged.ts's bench writes down every call to
// src/bench.ts in that order, so the parts are not run alone and not reordered).
import "./bench/rounds.ts";
import "./bench/cpu.ts";
import "./bench/gpu.ts";
import "./bench/layersteps.ts";
import "./bench/generate.ts";
import "./bench/path.ts";
import "./bench/summary.ts";
import "./bench/warnings.ts";
import "./bench/sections.ts";
import "./bench/gpuplan.ts";
import "./bench/speed.ts";

console.log("ok");
