// worker/request.ts (T378.3): what the page asked for with the load that is going on, and with the benchmark's rounds.
// This module alone sets these: the others say what happened (a load was asked for, the page path found a number of
// threads, a round began) and read the names.
// A module of worker.ts, which a bundler links (T367.3). It imports nothing (but types, which are no import once built).

import type { Asked, GpuAsked, ThreadsAsked } from "../contracts/messages.ts";

// what the page's own URL said to leave out (T52), as the first message says it: a load has these unless it is a round
// of the benchmark, which comes back to them (T77)
let pageSwitches: string[] | undefined;
// Only what the engine has a fallback for. A name it does not know is refused there, and the page says so.
function switchesOf(search: string) {
  const parameters = new URLSearchParams(search);
  const names = (parameters.get("without") ?? "").split(",").map((name) => name.trim()).filter(Boolean);
  if (parameters.get("kernel") === "off" && !names.includes("kernels")) {
    names.push("kernels");
  }
  return names;
}

// T93 stage 2: what the page asked about the number of threads ({ fixed, remembered, hint })
export let threadsRequest: ThreadsAsked | undefined;
// what the page kept of the GPU's shaders for the model asked for ({ remembered }), as threadsRequest
export let gpuRequest: GpuAsked | undefined;
// the URL of the site's root, which the page hands with a load (T367.3): models/ is under it
export let root: string | undefined;
// T242: the switches of each load the page says will follow on the model of its init, or undefined (weights.ts's pooledWeights())
export let loadsAhead: string[][] | undefined;
// the benchmark's rounds are running (T45); see worker.ts's bench message
export let benching = false;
// the switches of the round that is going on, or undefined
let round: string[] | undefined;

/** An init or a load came: what it asks for besides the model. (The root stays where a load says none.) */
export function loadAsked(data: Asked) {
  pageSwitches ??= switchesOf(data.search);  // (the page's URL is the same for every load of a visit: read once)
  threadsRequest = data.threads;
  gpuRequest = data.gpu;
  root = data.root ?? root;  // (T367.3: the site's root, where models/ is; the page knows it)
  loadsAhead = Array.isArray(data.ahead) ? data.ahead : undefined;  // (T242; a load of the model page says none)
}

/** T184: the page path found the number of threads: the rounds that follow load the model again on that count */
export function threadsFound(count: number) {
  threadsRequest = { ...threadsRequest, remembered: count };
}

export function roundsBegan() {
  benching = true;
}
/** A round of the benchmark loads the model with its own switches (T45) */
export function roundBegan(without: string[]) {
  round = without;
}
// also when a round failed or a change of model cancelled it (the review of T76): the next model must not load with a
// round's switches off while the panel shows them on
export function roundsEnded() {
  round = undefined;  // (the page's again, not a round's)
  benching = false;
}

/** The optimizations the load going on leaves out (T52): a round's own, else the page's (?without=relaxed,sampler,
 * and ?kernel=off as it always was) */
export const switchesNow = () => round ?? pageSwitches!;
