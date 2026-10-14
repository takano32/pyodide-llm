// tests/page-parts-check.mjs (T378.5): the parts of the two pages' scripts that hold something, each by itself.
//
// What the model page's script keeps has two owners, and /benchmark/'s one. The other parts say what happened, to a
// function of the owner, and read what it holds by the names they import. The owners read nothing of a page (no
// element, no address, no storage, no clock), so Node imports them as the modules they are (Node 24 takes the types
// off a .ts file), with no browser, no worker and no model:
//
//   src/page/model.ts         the model the page has and how far its load has come: each event changes what it says
//                             and nothing else (the others are compared whole, before and after); events out of their
//                             order do no harm; nothing of it can be assigned from outside
//   src/page/answer.ts        the answer being written: its bubble, the settings it is written with (the steps held to
//                             the context, the seed drawn once where the run samples and gone where it is greedy), and
//                             that it is being written
//   src/benchmark/running.ts  the run of /benchmark/ that is going on: one at a time, its sections and the times it is told
//   src/page/opening.ts       the model a visit opens with, in a process of its own for each address and storage (it
//                             reads both as it loads, through address.ts and remembered.ts): the URL's first, then the
//                             one remembered, then the first of the list
//
// What these are still bound to: model.ts to src/models.js (the first of the list is what a visit starts with);
// opening.ts to `location` and `localStorage` (stand-ins here). What the page draws of all this is the browsers' to
// see (tests/e2e.mjs, tests/bench-check.mjs).
//
//   node tests/page-parts-check.mjs        (Node only, about a second)
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { MODELS } from "../src/models.js";

let made = 0, cases = 0;
const fresh = (file) => import(`${new URL(`../src/${file}`, import.meta.url).href}?case=${made++}`);
// every export that is no function: what the owner holds now
const held = (module) => Object.fromEntries(Object.entries(module).filter(([, value]) => typeof value !== "function").map(([name, value]) =>
  [name, value && typeof value === "object" && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype ? { ...value } : value]));
/** Run `event`, and hold that what the owner holds changed in `changes` and in nothing else. */
function only(module, event, changes, what) {
  const before = held(module);
  const answer = event();
  assert.deepEqual(held(module), { ...before, ...changes }, what);
  cases += 1;
  return answer;
}
const unassignable = (module, what) => {
  for (const name of Object.keys(module)) assert.throws(() => { module[name] = 1; }, { name: "TypeError" }, `${what}: ${name} can be assigned from outside its module`);
  cases += 1;
};
const [first, second, third] = MODELS;

// ---- src/page/model.ts
{
  const m = await fresh("page/model.ts");
  assert.deepEqual(held(m), { model: first, loads: 1, ready: false, longest: 256, fromTemplate: undefined, readyLine: undefined, gpuNote: undefined, benchmarking: false, outdated: false },
    "a visit starts with the first of the list, its first load, nothing ready");
  assert.equal(m.takesPrompts(), false, "a model that is not ready takes no prompt");
  unassignable(m, "model.ts");

  // what the visit opens with, as often as it is said before a model is chosen
  assert.equal(only(m, () => m.opensWith(second), { model: second }, "opening with a model sets the model alone"), true);
  assert.equal(only(m, () => m.opensWith(third), { model: third }, "and again, while nothing was chosen"), true);

  // the worker's ready
  const ready = { type: "ready", load: 1, seq_len: 4096, template: "<|user|>{prompt}", backend: "SIMD kernels, int8", threads: 4, pyodide: "314.0.7", seconds: {}, gpu: "prompts on the CPU" };
  only(m, () => m.becameReady(ready), { ready: true, longest: 4096, fromTemplate: "<|user|>{prompt}", readyLine: { backend: "SIMD kernels, int8", threads: 4, pyodide: "314.0.7" } },
    "ready: the model takes prompts, its context, its own format and how it runs; the GPU's word is not its to set");
  assert.equal(m.takesPrompts(), true);
  assert.notEqual(m.readyLine, ready, "the status line's record is the owner's own, not the worker's message");

  // the GPU's word: news once, kept when nothing is said
  assert.equal(only(m, () => m.gpuSaid("prompts on WebGPU"), { gpuNote: "prompts on WebGPU" }, "the GPU's word"), true, "a new word is news");
  assert.equal(only(m, () => m.gpuSaid("prompts on WebGPU"), {}, "the same word again"), false, "the same word is no news");
  assert.equal(only(m, () => m.gpuSaid(undefined), {}, "nothing said keeps the word"), false);
  assert.equal(only(m, () => m.gpuSaid(), {}, "nothing said keeps the word"), false);
  assert.equal(only(m, () => m.gpuSaid("prompts on the CPU (slower here)"), { gpuNote: "prompts on the CPU (slower here)" }, "another word"), true);

  // the threads an answer's end says
  only(m, () => m.threadsNow(2), { readyLine: { backend: "SIMD kernels, int8", threads: 2, pyodide: "314.0.7" } }, "the threads in use, and nothing else of the line");

  // the benchmark: no prompt while its rounds run, whatever is ready
  only(m, () => m.benchBegan(), { benchmarking: true }, "the benchmark began");
  assert.equal(m.takesPrompts(), false, "a prompt sent during the benchmark would find no model ready");
  only(m, () => m.benchBegan(), {}, "said twice, it began once");
  only(m, () => m.benchEnded(), { benchmarking: false }, "the benchmark ended");
  assert.equal(m.takesPrompts(), true);
  only(m, () => m.benchEnded(), {}, "an end without a beginning changes nothing");

  // the worker's error
  m.benchBegan();
  only(m, () => m.failed(false), { benchmarking: false }, "an error ends the benchmark, and a model that was ready still is");
  assert.equal(m.takesPrompts(), true, "after a failed round the run button must not stay off (the review of T76)");
  only(m, () => m.failed(false), {}, "an error with nothing going on changes nothing");
  only(m, () => m.failed(true), { ready: false }, "a model the worker loads again on the CPU is not ready (T156)");
  assert.equal(m.takesPrompts(), false);
  only(m, () => m.becameReady({ ...ready, template: undefined, threads: 1, seq_len: 512 }), { ready: true, longest: 512, fromTemplate: undefined, readyLine: { backend: "SIMD kernels, int8", threads: 1, pyodide: "314.0.7" } },
    "ready again: a model that says no format has none (the one before it is gone)");
  for (const template of [null, 7, {}, ["{prompt}"]]) {
    m.becameReady({ ...ready, template });
    assert.equal(m.fromTemplate, undefined, `a template that is no text (${JSON.stringify(template)}) is none`);
  }
  m.becameReady(ready);

  // a model is chosen: nothing of the one before it holds, but what only the next ready says (the context, the format)
  m.benchBegan();
  assert.equal(only(m, () => m.chose(first), { model: first, loads: 2, ready: false, benchmarking: false, readyLine: undefined, gpuNote: undefined },
    "a choice: the model, its load's number, and nothing ready, measured or said of the one before"), 2, "chose() answers the load's number");
  assert.equal(m.takesPrompts(), false);
  assert.equal(only(m, () => m.chose(first), { loads: 3 }, "the same model chosen again is another load"), 3);
  // ... and from then on the visit does not open with anything else (the review of T116: a model chosen while the page
  // looked at what is kept wins)
  assert.equal(only(m, () => m.opensWith(second), {}, "opening with a model after one was chosen changes nothing"), false, "and says so");

  // out of their order: nothing here throws, and nothing is ready that was not said to be
  only(m, () => m.threadsNow(8), {}, "the threads of a model that is not ready are nobody's");
  only(m, () => m.benchEnded(), {}, "");
  only(m, () => m.failed(true), {}, "an error while loading: still not ready");
  assert.equal(only(m, () => m.gpuSaid(""), { gpuNote: "" }, "an empty word is a word (it is kept), and no news"), false);
}
{
  // every case of whether a prompt may be sent
  for (const [ready, benchmarking, takes] of [[false, false, false], [true, false, true], [true, true, false], [false, true, false]]) {
    const m = await fresh("page/model.ts");
    if (ready) m.becameReady({ seq_len: 1, backend: "", threads: 1, pyodide: "" });
    if (benchmarking) m.benchBegan();
    assert.equal(m.takesPrompts(), takes, `ready ${ready}, benchmarking ${benchmarking}`);
    cases += 1;
  }
}
{
  // T439: the worker's word that it runs slower for a file of this page's build that is gone. It is of the page, and of
  // no model: whenever it comes (before a model is ready, during a benchmark, twice), and whatever comes after it
  // (another model, its ready, an error, a model loaded again on the CPU), it holds for the rest of the visit
  const m = await fresh("page/model.ts");
  only(m, () => m.siteUpdated(), { outdated: true }, "the site was updated: that, and nothing of the model");
  only(m, () => m.siteUpdated(), {}, "said again: the same");
  m.becameReady({ seq_len: 1, backend: "NumPy", threads: 1, pyodide: "" });
  assert.equal(m.outdated, true, "a ready model's page is still an old one");
  assert.equal(m.takesPrompts(), true, "and the model takes prompts: nothing is stopped");
  m.gpuSaid("prompts on the CPU");
  m.threadsNow(2);
  m.benchBegan();
  m.benchEnded();
  m.failed(false);
  m.failed(true);
  assert.equal(m.outdated, true, "whatever the worker says next");
  m.chose(second);
  assert.equal(m.outdated, true, "another model on the same page: the page is as old as it was");
  m.becameReady({ seq_len: 1, backend: "NumPy", threads: 1, pyodide: "" });
  assert.equal(m.outdated, true);
  // ... and a page that was told nothing says nothing, whatever else happens to its model
  const quiet = await fresh("page/model.ts");
  quiet.becameReady({ seq_len: 1, backend: "NumPy", threads: 1, pyodide: "" });
  quiet.gpuSaid("prompts on the CPU (the GPU's worker did not start (an error))");
  quiet.failed(true);
  quiet.chose(third);
  assert.equal(quiet.outdated, false, "no event but the worker's word makes a page an old one");
  cases += 2;
}
console.log(`page-parts-check: model.ts by itself: what a visit opens with, a choice, ready, the GPU's word, the threads, the benchmark, an error, the site's update; each changes what it says alone`);

// ---- src/page/answer.ts
{
  const a = await fresh("page/answer.ts");
  assert.deepEqual(held(a), { generating: false, answer: null, used: {} }, "nothing is being written as the page starts");
  unassignable(a, "answer.ts");
  const drawn = [];
  const seed = () => (drawn.push(1), 4242);
  const bubble = { bubble: 1 }, settings = { steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty: 1.1 };
  let how = only(a, () => a.answerBegan(bubble, settings, 4096, seed), { generating: true, answer: bubble, used: { ...settings, seed: 4242 } },
    "an answer that samples: its bubble, its settings, and a seed drawn for it");
  assert.equal(how, a.used, "what answerBegan() answers is what the worker is sent: the settings of this run");
  assert.equal(a.answer, bubble, "the bubble is the one handed over");
  assert.deepEqual([drawn.length, settings], [1, { steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty: 1.1 }], "the seed is drawn once, and the page's settings are not written to");
  only(a, () => a.idle(), { generating: false }, "idle: nothing is written any more; the bubble and its settings stay for the line under it");
  only(a, () => a.idle(), {}, "idle twice");

  // a seed the visitor fixed is the seed, and none is drawn
  how = a.answerBegan(bubble, { ...settings, seed: 7 }, 4096, seed);
  assert.deepEqual([how.seed, drawn.length], [7, 1], "a fixed seed is kept");
  how = a.answerBegan(bubble, { ...settings, seed: 0 }, 4096, seed);
  assert.deepEqual([how.seed, drawn.length], [0, 1], "the seed 0 is a seed");
  // greedy: no seed, not even a fixed one (showing one would suggest it mattered)
  how = a.answerBegan(bubble, { ...settings, temperature: 0, seed: 7 }, 4096, seed);
  assert.deepEqual(["seed" in how, drawn.length], [false, 1], "a greedy run has no seed and draws none");
  // the steps: 0 is all of the context and stays 0; more than the context is the context; fewer stay
  for (const [steps, longest, sent] of [[0, 4096, 0], [128, 4096, 128], [4096, 4096, 4096], [5000, 4096, 4096], [300, 256, 256], [-1, 256, 0]]) {
    assert.equal(a.answerBegan(bubble, { ...settings, steps }, longest, seed).steps, sent, `${steps} steps asked of a context of ${longest}`);
    cases += 1;
  }
  // settings the page has no field for go to the worker as they are (T274)
  how = a.answerBegan(bubble, { ...settings, top_k: 20, min_p: 0.05, presence_penalty: 1.5 }, 4096, seed);
  assert.deepEqual([how.top_k, how.min_p, how.presence_penalty], [20, 0.05, 1.5]);
  // the next answer's bubble replaces the last one's
  const next = { bubble: 2 };
  a.answerBegan(next, settings, 4096, seed);
  assert.equal(a.answer, next);
  cases += 4;
}
console.log("page-parts-check: answer.ts by itself: the bubble, the settings of the run (steps within the context, a seed where it samples), idle");

// ---- src/benchmark/running.ts
{
  const r = await fresh("benchmark/running.ts");
  assert.deepEqual(held(r), { current: undefined, sectionAt: 0, runNames: [], runAt: 0 }, "no run as the page starts");
  unassignable(r, "running.ts");
  only(r, () => r.runEnded(), {}, "an end before any run changes nothing");
  const names = ["device", "cpu", "model"];
  assert.equal(only(r, () => r.runBegan(names, 1000), { runNames: names, runAt: 1000 }, "a run begins: its sections and when; no section runs yet"), true);
  only(r, () => r.sectionBegan("device", 1010), { current: "device", sectionAt: 1010 }, "its first section");
  // one run at a time: a second one is refused and changes nothing (a Run button pressed, ?run= while one goes on)
  assert.equal(only(r, () => r.runBegan(["gpu"], 2000), {}, "a run while one is going on"), false, "is refused");
  only(r, () => r.sectionBegan("cpu", 5000), { current: "cpu", sectionAt: 5000 }, "the next section: the run's own time stays");
  only(r, () => r.runEnded(), { current: undefined }, "the run ended: no section runs; when the last ones began stays for what is written after");
  assert.equal(only(r, () => r.runBegan(["gpu"], 9000), { runNames: ["gpu"], runAt: 9000 }, "another run after it"), true);
  r.sectionBegan("gpu", 9001);
  only(r, () => r.runEnded(), { current: undefined }, "");
  only(r, () => r.runEnded(), {}, "ended twice");
}
console.log("page-parts-check: running.ts by itself: one run at a time, its sections and the times it is told");

// ---- src/page/opening.ts: a process for each visit (address.ts and remembered.ts read the address and the storage as
// they load, and a module is made once)
{
  const large = MODELS.find((entry) => (entry.download ?? 0) > 500e6);
  assert.ok(large && MODELS.length > 2, "the list has a model fetched from Hugging Face anew (more than 500 MB)");
  const opens = (search, stored) => {
    const script = `
      globalThis.location = { search: ${JSON.stringify(search)} };
      const stored = ${JSON.stringify(stored)};
      if (stored !== null) globalThis.localStorage = { getItem: (name) => (name === "model" ? stored : null) };
      await import(${JSON.stringify(new URL("../src/page/opening.ts", import.meta.url).href)});
      const { model, loads } = await import(${JSON.stringify(new URL("../src/page/model.ts", import.meta.url).href)});
      const { recalled } = await import(${JSON.stringify(new URL("../src/page/remembered.ts", import.meta.url).href)});
      console.log(JSON.stringify({ id: model.id, name: model.name, repo: model.hf?.repo, weights: model.hf?.repo ? model.hf.weights : undefined, note: model.hf?.repo ? model.note : undefined, loads, recalled: recalled?.id ?? null }));`;
    cases += 1;
    return JSON.parse(execFileSync(process.execPath, ["--input-type=module", "-e", script], { encoding: "utf8" }));
  };
  const kept = (id, was = false) => JSON.stringify({ id, kept: was });
  assert.deepEqual(opens("", null), { id: first.id, name: first.name, loads: 1, recalled: null }, "a first visit (and a browser without storage) opens the first of the list");
  assert.deepEqual(opens("", kept(third.id)), { id: third.id, name: third.name, loads: 1, recalled: third.id }, "the model chosen on an earlier visit");
  assert.deepEqual(opens(`?model=${second.id}`, kept(third.id)), { id: second.id, name: second.name, loads: 1, recalled: null }, "the URL's model of the list before the one remembered");
  assert.deepEqual(opens("?model=no-such-model", kept(third.id)), { id: third.id, name: third.name, loads: 1, recalled: third.id }, "a name of no model: the one remembered");
  assert.deepEqual(opens("?model=no-such-model", null).id, first.id, "a name of no model and nothing remembered: the first");
  const fromLink = opens(`?hf=owner/name&model=${second.id}`, kept(third.id));
  assert.deepEqual([fromLink.id, fromLink.name, fromLink.repo, fromLink.recalled], ["local", "owner/name", "owner/name", null], "?hf= before everything");
  assert.deepEqual([fromLink.weights, fromLink.note], ["model.safetensors", "Hugging Face"], "?hf= with no file named is not opened by model.safetensors (until the page has looked at the repository)");
  // T505.3: &file= names the file that is the model, and &revision= the revision, both as names an address may hold
  const named = opens("?hf=owner/name-GGUF&file=name-Q4_K_M.gguf&revision=v1.0", null);
  assert.deepEqual([named.id, named.repo, named.weights, named.note], ["local", "owner/name-GGUF", "name-Q4_K_M.gguf", "Hugging Face · v1.0 · name-Q4_K_M.gguf"], "?hf=&file= does not open the file it names");
  assert.equal(opens("?hf=owner/name&file=", null).weights, "model.safetensors", "an empty &file= names a file");
  for (const file of ["../other/resolve/main/model.gguf", "a/../b.gguf", "name.gguf?download=true", "https://example.com/m.gguf"]) {
    assert.equal(opens(`?hf=owner/name&file=${encodeURIComponent(file)}`, null).id, first.id, `a link opened ${file} as a file of a repository: it goes into the address the worker fetches`);
  }
  assert.equal(opens("", kept(large.id, false)).id, first.id, "a large model that was not kept is not started unasked");
  assert.equal(opens("", kept(large.id, true)).id, large.id, "one that was kept is (index.astro then looks whether it still is)");
  assert.equal(opens("", "not JSON").id, first.id, "something else under the name: the first");
}
console.log(`page-parts-check: opening.ts: the URL's model, the one remembered, the first of the list\npage-parts-check: ok (${cases} cases)`);
