// The section "Model": the model page's worker itself, the page's path timed and the rounds (T184, T190).
// (T353: a part of /benchmark/'s script; src/pages/benchmark.astro imports the parts in the order they ran as one script)
import { roundsHere, FULL_ROUNDS, ROUNDS, threadsKey, PATH_PROMPTS, PATH_WRITES, pathTable, pathWarnings, roundsTable } from "../bench.js";
import { $, parameters, type Result } from "./dom.ts";
import { staged, worker, model } from "./section.ts";

// the model page's worker itself (public/worker.js): the model loads as the model page loads it (the GPU too), the
// model page's own path is timed on it (T184: prompts and writing as the page chooses between the GPU and the CPU,
// and on each alone), then the rounds of ?bench=1 (T45), which load it again each, on the CPU alone
// what the model section measured, under the name of the model it ran (the choice may change after)
// said (T227's review): what the page's path says went wrong in its first line (src/bench.js's pathWarnings()), for the warnings
export let measured: { name: string; rows: any[]; pyodide: string; paths?: string; said?: string[] } | undefined;
export async function modelSection(): Promise<Result> {
  const entry = model();
  const w = worker("worker.js");
  // T214: where the browser does not say its memory, the round without the kernels is a row that says why
  const asked = roundsHere(($("full") as HTMLInputElement).checked ? FULL_ROUNDS : ROUNDS, (navigator as any).deviceMemory);
  // T177: the load, the page's path (the worker's words as the stage), then each round that runs (loaded again and
  // run: the worker says ready at the end of each round's load; a skipped round loads nothing)
  const rounds = asked.filter((round) => round.skip === undefined);
  const stages = 2 + rounds.length;
  let readies = 0, pathing = false;
  const prompt = entry.prompt || "Once upon a time";
  const megabytes = (data: any) => `the model: ${(data.received / 1e6 || 0).toFixed(1)} / ${(data.total / 1e6).toFixed(1)} MB`;
  try {
    const listen = (resolve: (value: any) => void) => {
      w.onmessage = ({ data }) => {
        // T205's review: a round's words and MB are of its load (the first load's ready counted, readies - 1 is the
        // round being loaded), and its ready begins its writing: the stage names the round of either (it said the
        // round before while the next one loaded, so a tab that went down in the load of "without the kernels" read
        // as one in "everything")
        const round = readies && !pathing ? rounds[Math.min(readies - 1, rounds.length - 1)].name : "";
        if (data.type === "status" && pathing) staged("model", { stage: `the page's path: ${data.text}`, at: 2, of: stages });
        else if (data.type === "status" && !readies) staged("model", { stage: data.text, at: 1, of: stages });
        else if (data.type === "progress" && !readies) {
          staged("model", { stage: megabytes(data), at: 1, of: stages, part: data.received / data.total || 0 });
        } else if (data.type === "status" && round) staged("model", { stage: `round: ${round}: ${data.text}`, at: 2 + readies, of: stages });
        else if (data.type === "progress" && round) {
          staged("model", { stage: `round: ${round}: ${megabytes(data)}`, at: 2 + readies, of: stages, part: (data.received / data.total || 0) / 2 });
        } else if (data.type === "ready") {
          if (readies) staged("model", { stage: `round: ${round}: writing`, at: 2 + readies, of: stages, part: 0.5 });
          readies++;
          if (readies === 1) resolve(data);
        } else if (data.type === "bench" || data.type === "paths") resolve(data);
        else if (data.type === "error") {
          // T242: where it happened goes to the console, as on the model page (the report holds the message alone)
          console.error(`worker error (weights memory ${data.weights} bytes):`, data.stack);
          resolve({ error: data.message });
        }
      };
      w.onerror = (event) => resolve({ error: `the worker failed: ${event.message ?? "error"}` });
    };
    const request = (message: any) => new Promise<any>((resolve) => {
      listen(resolve);
      w.postMessage(message);
    });
    // ?gpuTest=on (tests only, as on the model page: T148): CI's fallback adapter taken as a GPU, to see the GPU's side
    // T190: the number of threads the model page remembers for this model here (it reads it the same way), else a
    // search from the logical cores that the page path lets end (worker/timing.js); nothing is remembered from here
    let remembered = 0;
    try {
      remembered = Number(localStorage.getItem(threadsKey(entry.id, navigator))) || 0;
    } catch {
      // no storage here: searched
    }
    // T242, ahead: the switches of each round that follows on this model, the only one this worker loads: its memory is
    // made for the largest of them, and not for a next model as well (that gigabyte took down Windows' WebKit here)
    const loaded = await request({ type: "init", search: parameters.get("gpuTest") === "on" ? "?gpuTest=on" : "", model: entry, load: 1,
      threads: { remembered, hint: navigator.hardwareConcurrency || 1 }, ahead: rounds.map((round) => round.without) });
    if (loaded.error) return { status: "error", markdown: loaded.error };
    pathing = true;
    staged("model", { stage: "the page's path", at: 2, of: stages });
    const paths = await request({ type: "paths", load: 1, prompt, counts: PATH_PROMPTS, sampled: PATH_WRITES });
    pathing = false;
    const answer = await request({ type: "bench", model: entry, load: 1, rounds: asked, prompt, steps: 64 });
    if (answer.error) return { status: "error", markdown: answer.error };
    const table = pathTable(paths, entry.name);
    measured = { name: entry.name, rows: answer.rows, pyodide: answer.pyodide, paths: table, said: pathWarnings(paths) };
    const lines = roundsTable(answer.rows);
    return { status: "ok", data: { rows: answer.rows, paths },
             markdown: `${entry.name} · Pyodide ${answer.pyodide}\n\n${lines.join("\n")}\n\n${table}` };
  } finally {
    w.terminate();
  }
}
