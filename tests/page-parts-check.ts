// tests/page-parts-check.ts (T.378.5): the parts of the two pages' scripts that hold something, each by itself.
//
// What the model page's script keeps has two owners, and /benchmark/'s one. The other parts say what happened, to a
// function of the owner, and read what it holds by the names they import. The owners read nothing of a page (no
// element, no address, no storage, no clock), so Node imports them as the modules they are (Node 24 takes the types
// off a .ts file), with no browser, no worker and no model:
//
//   src/page/model.ts         the model the page has and how far its load has come (T.553: and the time its fetch has
//                             left, from the bytes said and the clock the caller hands over): each event changes what it says
//                             and nothing else (the others are compared whole, before and after); events out of their
//                             order do no harm; nothing of it can be assigned from outside
//   src/page/answer.ts        the answer being written: its bubble, the settings it is written with (the steps held to
//                             the context, the seed drawn once where the run samples and gone where it is greedy), and
//                             that it is being written
//   src/benchmark/running.ts  the run of /benchmark/ that is going on: one at a time, its sections and the times it is told
//   src/page/opening.ts       the model a visit opens with, in a process of its own for each address and storage (it
//                             reads both as it loads, through address.ts and remembered.ts): the URL's first, then the
//                             one remembered, then the default (T.511.6)
//
// What these are still bound to: model.ts to src/models.ts (its DEFAULT_MODEL is what a visit starts with);
// opening.ts to `location` and `localStorage` (stand-ins here). What the page draws of all this is the browsers' to
// see (tests/e2e.ts, tests/bench-check.ts).
//
//   node tests/page-parts-check.ts        (Node only, about a second)
import { fresh as made } from "./fresh.ts";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { DEFAULT_MODEL, MODELS } from "../src/models.ts";

let cases = 0;
// (T.584: tests/fresh.ts, on Node and on Bun)
const fresh = (file) => made(new URL(`../src/${file}`, import.meta.url));
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
// (T.511.6: the default is not the first of the list; second and third are two others)
const first = DEFAULT_MODEL, [second, third] = MODELS.filter((entry) => entry !== first);

// ---- src/page/model.ts
{
  const m = await fresh("page/model.ts");
  assert.deepEqual(held(m), { model: first, loads: 1, ready: false, longest: 256, fromTemplate: undefined, readyLine: undefined, gpuNote: undefined, benchmarking: false, outdated: false, arriving: undefined },
    "a visit starts with the default, its first load, nothing ready");
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
  assert.equal(m.takesPrompts(), true, "after a failed round the run button must not stay off (the review of T.76)");
  only(m, () => m.failed(false), {}, "an error with nothing going on changes nothing");
  only(m, () => m.failed(true), { ready: false }, "a model the worker loads again on the CPU is not ready (T.156)");
  assert.equal(m.takesPrompts(), false);
  only(m, () => m.becameReady({ ...ready, template: undefined, threads: 1, seq_len: 512 }), { ready: true, longest: 512, fromTemplate: undefined, readyLine: { backend: "SIMD kernels, int8", threads: 1, pyodide: "314.0.7" } },
    "ready again: a model that says no format has none (the one before it is gone)");
  for (const template of [null, 7, {}, ["{prompt}"]]) {
    m.becameReady({ ...ready, template });
    assert.equal(m.fromTemplate, undefined, `a template that is no text (${JSON.stringify(template)}) is none`);
  }
  m.becameReady(ready);

  // a model is chosen: nothing of the one before it holds (T.470: nor its context and format, which the next ready says:
  // the first value of each till then)
  m.benchBegan();
  assert.equal(only(m, () => m.chose(first), { model: first, loads: 2, ready: false, benchmarking: false, readyLine: undefined, gpuNote: undefined, longest: 256, fromTemplate: undefined },
    "a choice: the model, its load's number, and nothing ready, measured or said of the one before"), 2, "chose() answers the load's number");
  assert.equal(m.takesPrompts(), false);
  assert.equal(only(m, () => m.chose(first), { loads: 3 }, "the same model chosen again is another load"), 3);
  // ... and from then on the visit does not open with anything else (the review of T.116: a model chosen while the page
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
  // T.439: the worker's word that it runs slower for a file of this page's build that is gone. It is of the page, and of
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
{
  // T.553: the reports of a load's fetch, and the time it has left (remaining.ts: the clock is the caller's)
  const m = await fresh("page/model.ts");
  const { secondsLeft, leftSaid, advanced } = await fresh("page/remaining.ts");
  // (receiver.ts says each report so)
  const report = (bytes, now) => m.arrived(advanced(m.arriving, bytes, now));
  report(1e6, 1000);
  assert.deepEqual(m.arriving, { began: 1000, first: 1e6, at: 1000, bytes: 1e6, moved: 1000 }, "a first report: when and how much, no speed yet");
  assert.equal(secondsLeft(m.arriving, 100e6, 1000), undefined, "no speed: nothing said");
  for (let t = 1250; t <= 4000; t += 250) report(1e6 + (t - 1000) * 1000, t);  // 1 MB/s
  assert.equal(m.arriving.speed, 1e6, "the plain average of the first seconds");
  assert.equal(secondsLeft(m.arriving, 100e6, 3999), undefined, "nothing in the first 3 seconds (the owner's words)");
  report(4.25e6, 4250);
  assert.equal(Math.round(secondsLeft(m.arriving, 100e6, 4250)), 96, "after them, the bytes left over the speed");
  assert.equal(leftSaid(secondsLeft(m.arriving, 100e6, 4250)), " · about 100 s left", "said in tens above a minute");
  assert.ok(secondsLeft(m.arriving, 100e6, 14249) > 0, "said until 10 seconds pass without an arrival");
  assert.equal(secondsLeft(m.arriving, 100e6, 14250), undefined, "nothing once nothing arrived for 10 seconds");
  report(4.25e6, 14000);  // a report of no more bytes (the share converted moved): still nothing arrived
  assert.equal(secondsLeft(m.arriving, 100e6, 14250), undefined, "a report without new bytes is no arrival");
  // the speed changes: after 10 seconds the weighted average follows the new one, and forgets the old one in about 10
  for (let t = 14250; t <= 44000; t += 250) report(m.arriving.bytes + 250e3 * 4, t);  // 4 MB/s from now
  const a = m.arriving;
  assert.ok(Math.abs(a.speed - 4e6) < 0.05 * 4e6, `the speed follows a faster line (${a.speed})`);
  assert.equal(secondsLeft(a, a.bytes, 44000), undefined, "nothing left: nothing said");
  assert.equal(leftSaid(secondsLeft(a, a.bytes + 1, 44000)), " · about 1 s left", "never under a second");
  assert.deepEqual([leftSaid(undefined), leftSaid(7.4), leftSaid(33), leftSaid(57.6), leftSaid(61)], ["", " · about 7 s left", " · about 35 s left", " · about 60 s left", " · about 60 s left"]);
  // a fetch begun again (fewer bytes), and a load chosen anew
  report(5, 50000);
  const again = m.arriving;
  assert.deepEqual(again, { began: 50000, first: 5, at: 50000, bytes: 5, moved: 50000 }, "fewer bytes than before: a fetch begun anew");
  report(6, 50000);
  assert.equal(m.arriving, again, "a report at the same time adds nothing to the speed");
  only(m, () => m.chose(second), { model: second, loads: 2, arriving: undefined }, "another model: nothing of the fetch before");
  report(1, 1);
  only(m, () => m.becameReady({ seq_len: 1, backend: "NumPy", threads: 1, pyodide: "" }), { ready: true, longest: 1, readyLine: { backend: "NumPy", threads: 1, pyodide: "" }, arriving: undefined }, "ready: the fetch is over");
  report(1, 1);
  only(m, () => m.failed(false), { arriving: undefined }, "an error: the fetch is over");
  cases += 6;
}
console.log(`page-parts-check: model.ts by itself: what a visit opens with, a choice, ready, the GPU's word, the threads, the benchmark, an error, the site's update; each changes what it says alone`);

// ---- src/page/answer.ts
{
  const a = await fresh("page/answer.ts");
  assert.deepEqual(held(a), { generating: false, answer: null, used: {}, written: "", undrawn: 0, drawnAt: undefined, shape: { markdown: false, thinking: false } }, "nothing is being written as the page starts");
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
  // T.468.3: none while one is being written, and nothing of that one changes
  a.answerBegan(bubble, { ...settings, seed: 9 }, 4096, seed);  // (a fixed seed: none drawn for it)
  assert.equal(only(a, () => a.answerBegan({ bubble: "second" }, { ...settings, seed: 1 }, 4096, seed), {}, "a second answer while one is written changes nothing"),
    undefined, "and is not begun");
  assert.equal(drawn.length, 1, "nor draws a seed");
  a.idle();
  cases += 2;

  // (each answer after the worker went idle, as the page asks: T.468.3)
  const begin = (...args) => (a.idle(), a.answerBegan(...args));
  // a seed the visitor fixed is the seed, and none is drawn
  how = begin(bubble, { ...settings, seed: 7 }, 4096, seed);
  assert.deepEqual([how.seed, drawn.length], [7, 1], "a fixed seed is kept");
  how = begin(bubble, { ...settings, seed: 0 }, 4096, seed);
  assert.deepEqual([how.seed, drawn.length], [0, 1], "the seed 0 is a seed");
  // greedy: no seed, not even a fixed one (showing one would suggest it mattered)
  how = begin(bubble, { ...settings, temperature: 0, seed: 7 }, 4096, seed);
  assert.deepEqual(["seed" in how, drawn.length], [false, 1], "a greedy run has no seed and draws none");
  // the steps: 0 is all of the context and stays 0; more than the context is the context; fewer stay
  for (const [steps, longest, sent] of [[0, 4096, 0], [128, 4096, 128], [4096, 4096, 4096], [5000, 4096, 4096], [300, 256, 256], [-1, 256, 0]]) {
    assert.equal(begin(bubble, { ...settings, steps }, longest, seed).steps, sent, `${steps} steps asked of a context of ${longest}`);
    cases += 1;
  }
  // settings the page has no field for go to the worker as they are (T.274)
  how = begin(bubble, { ...settings, top_k: 20, min_p: 0.05, presence_penalty: 1.5 }, 4096, seed);
  assert.deepEqual([how.top_k, how.min_p, how.presence_penalty], [20, 0.05, 1.5]);
  // the next answer's bubble replaces the last one's
  const next = { bubble: 2 };
  begin(next, settings, 4096, seed);
  assert.equal(a.answer, next);
  cases += 4;
}
{
  // T.554: the tokens of an answer, drawn at once, then every 8 tokens or 32 ms; a new answer begins with nothing written
  const a = await fresh("page/answer.ts");
  a.answerBegan({}, { steps: 0, temperature: 0 }, 256, () => 1);
  assert.equal(only(a, () => a.wrote("A", 1000), { written: "A", undrawn: 1 }, "the first token"), true, "is drawn at once");
  only(a, () => a.drew(1000), { undrawn: 0, drawnAt: 1000 }, "drawn: nothing undrawn, and when");
  const due = [];
  for (let i = 0; i < 9; i += 1) due.push(a.wrote(String(i), 1001 + i));
  assert.deepEqual(due, [false, false, false, false, false, false, false, true, true], "the eighth token after a draw is drawn (and each after it until one is)");
  assert.equal(a.written, "A012345678");
  a.drew(1010);
  assert.equal(a.wrote("x", 1041), false, "31 ms after a draw: not yet");
  assert.equal(a.wrote("y", 1042), true, "32 ms: drawn");
  a.drew(1042);
  a.idle();  // (T.468.3: the next answer after the worker said this one's end)
  const bubble = {};
  only(a, () => a.answerBegan(bubble, { steps: 0, temperature: 0 }, 256, () => 1), { generating: true, answer: bubble, written: "", undrawn: 0, drawnAt: undefined }, "the next answer begins with nothing written or drawn");
  assert.equal(a.wrote("z", 0), true, "and its first token is drawn at once");
  cases += 2;
}
{
  // T.556: how an answer is drawn is said as it begins, a copy of what is handed over, and nothing of it stays to the next
  const a = await fresh("page/answer.ts");
  const drawn = { markdown: true, thinking: true }, bubble = {};
  only(a, () => a.answerBegan(bubble, { steps: 0, temperature: 0 }, 256, () => 1, drawn), { generating: true, answer: bubble, used: { steps: 0, temperature: 0 }, shape: { markdown: true, thinking: true } }, "an answer drawn as Markdown, in a thought");
  assert.notEqual(a.shape, drawn, "the shape is the owner's own");
  a.idle();
  a.answerBegan({}, { steps: 0, temperature: 0 }, 256, () => 1);
  assert.deepEqual(a.shape, { markdown: false, thinking: false }, "an answer that says nothing of it is plain text");
  cases += 2;
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
  assert.deepEqual((({ id, name, loads, recalled }) => ({ id, name, loads, recalled }))(opens("", null)), { id: first.id, name: first.name, loads: 1, recalled: null }, "a first visit (and a browser without storage) opens the default");
  assert.equal(opens("", kept("tiny-lm")).id, "tiny-lm", "a visitor who chose tiny-lm before the default changed keeps it (T.511.6)");
  assert.deepEqual(opens("", kept(third.id)), { id: third.id, name: third.name, loads: 1, recalled: third.id }, "the model chosen on an earlier visit");
  assert.deepEqual(opens(`?model=${second.id}`, kept(third.id)), { id: second.id, name: second.name, loads: 1, recalled: null }, "the URL's model of the list before the one remembered");
  assert.deepEqual(opens("?model=no-such-model", kept(third.id)), { id: third.id, name: third.name, loads: 1, recalled: third.id }, "a name of no model: the one remembered");
  assert.deepEqual(opens("?model=no-such-model", null).id, first.id, "a name of no model and nothing remembered: the default");
  const fromLink = opens(`?hf=owner/name&model=${second.id}`, kept(third.id));
  assert.deepEqual([fromLink.id, fromLink.name, fromLink.repo, fromLink.recalled], ["local", "owner/name", "owner/name", null], "?hf= before everything");
  assert.deepEqual([fromLink.weights, fromLink.note], ["model.safetensors", "Hugging Face"], "?hf= with no file named is not opened by model.safetensors (until the page has looked at the repository)");
  // T.505.3: &file= names the file that is the model, and &revision= the revision, both as names an address may hold
  const named = opens("?hf=owner/name-GGUF&file=name-Q4_K_M.gguf&revision=v1.0", null);
  assert.deepEqual([named.id, named.repo, named.weights, named.note], ["local", "owner/name-GGUF", "name-Q4_K_M.gguf", "Hugging Face · v1.0 · name-Q4_K_M.gguf"], "?hf=&file= does not open the file it names");
  assert.equal(opens("?hf=owner/name&file=", null).weights, "model.safetensors", "an empty &file= names a file");
  for (const file of ["../other/resolve/main/model.gguf", "a/../b.gguf", "name.gguf?download=true", "https://example.com/m.gguf"]) {
    assert.equal(opens(`?hf=owner/name&file=${encodeURIComponent(file)}`, null).id, first.id, `a link opened ${file} as a file of a repository: it goes into the address the worker fetches`);
  }
  assert.equal(opens("", kept(large.id, false)).id, first.id, "a large model that was not kept is not started unasked");
  assert.equal(opens("", kept(large.id, true)).id, large.id, "one that was kept is (index.astro then looks whether it still is)");
  assert.equal(opens("", "not JSON").id, first.id, "something else under the name: the default");
  // T.508.3: what the page keeps of a model (its threads, its GPU's shaders) is under keptAs(): a model of the list by
  // its id, one of ?hf= by its repository, revision and file (a repository's model.safetensors as it was), a folder's
  // by its name
  const keys = JSON.parse(execFileSync(process.execPath, ["--input-type=module", "-e", `
    globalThis.location = { search: "" };
    const { keptAs, gpuKey } = await import(${JSON.stringify(new URL("../src/page/remembered.ts", import.meta.url).href)});
    const hf = (weights) => ({ id: "local", name: "owner/name", hf: { repo: "owner/name", revision: "main", weights } });
    console.log(JSON.stringify([keptAs({ id: "tiny-lm" }), keptAs(hf("model.safetensors")), keptAs(hf("m-Q8_0.gguf")), keptAs(hf("m-Q4_K_M.gguf")),
      keptAs({ id: "local", name: "a folder" }), gpuKey(hf("m-Q8_0.gguf"))]));`], { encoding: "utf8" }));
  assert.deepEqual(keys, ["tiny-lm", "hf:owner/name@main", "hf:owner/name@main/m-Q8_0.gguf", "hf:owner/name@main/m-Q4_K_M.gguf", "file:a folder", "gpu:hf:owner/name@main/m-Q8_0.gguf"],
    "the GGUF files of one repository, and the models of ?hf=, are kept apart");
  cases += 1;
}
{
  // T.520: thought.ts by itself: an answer's thought where the format opened it or the model did, its end, and its line
  const { thoughtOf, thoughtLine } = await fresh("page/thought.ts");
  const none = { thought: null, closed: false };
  assert.deepEqual(thoughtOf("Hello <think>", false), { ...none, answer: "Hello <think>" }, "an answer that does not begin with a thought is all answer");
  assert.deepEqual(thoughtOf("", true), { thought: "", closed: false, answer: "" }, "a format that opened the thought: nothing written yet is an empty thought");
  assert.deepEqual(thoughtOf("\nLet me see.\n", true), { thought: "Let me see.\n", closed: false, answer: "" }, "a thought that goes on");
  assert.deepEqual(thoughtOf("\nLet me see.\n</think>\n\nTokyo.", true), { thought: "Let me see.", closed: true, answer: "Tokyo." }, "a thought the format opened, closed");
  assert.deepEqual(thoughtOf("<think>\nHm.\n</think>\n\nTokyo. </think> again", false), { thought: "Hm.", closed: true, answer: "Tokyo. </think> again" }, "a thought the model opened (Qwen3's thinking form): it ends at its first closing mark");
  assert.deepEqual(thoughtOf("\nso 17 × 4</think>\n</think>\n\n408.", true), { thought: "so 17 × 4", closed: true, answer: "408." }, "a thought the budget closed and the model closed once more");
  assert.deepEqual(thoughtOf("<think>\n\n</think>\n\nTokyo.", true),{ thought: "", closed: true, answer: "Tokyo." }, "an empty thought (its opening mark written again) before the answer");
  assert.deepEqual([thoughtLine(false), thoughtLine(true)], ["Thinking…", "Thought"], "while it goes on, and once it closed");
  assert.deepEqual([thoughtLine(true, { thought: 412, thought_cut: false }), thoughtLine(true, { thought: 1, thought_cut: false }),
    thoughtLine(true, { thought: 3000, thought_cut: true }), thoughtLine(false, { thought: 80, thought_cut: false }), thoughtLine(true, { thought: null, thought_cut: false })],
  ["Thought for 412 tokens", "Thought for 1 token", "Thought for 3000 tokens, up to its limit", "Thought for 80 tokens, not finished", "Thought"], "at the answer's end, by the engine's count");
  cases += 1;
}
console.log(`page-parts-check: thought.ts by itself: a thought, its end and its line`);

// ---- T.411: folder.ts by itself: the models the page makes of the files of the visitor's disk (Node has File)
{
  const { openHuggingFace, openLegacy } = await fresh("page/folder.ts");
  const file = (name, size, text) => new File([text ?? new Uint8Array(size)], name);
  // the legacy form: the larger of the two is the checkpoint, its size the note's MB; a .json beside them is the settings
  const legacy = await openLegacy([file("tok512.bin", 6000), file("stories260K.bin", 1_100_000)], () => assert.fail("asked of a small file"));
  assert.deepEqual([legacy.name, legacy.note, legacy.file.name, legacy.tokenizerFile.name, legacy.bytes, legacy.prompt, legacy.generation],
    ["stories260K.bin", "local · 1 MB", "stories260K.bin", "tok512.bin", 1_100_000, "Once upon a time", { steps: 0, temperature: 0.0 }], "two files of llama2.c's form");
  const named = await openLegacy([file("a.bin", 3_400_000), file("settings.json", 0, JSON.stringify({ name: "Mine", options: { bias: true }, prompt: "Hi" })), file("t.bin", 10)], () => true);
  assert.deepEqual([named.name, named.note, named.options, named.prompt], ["Mine", "local · 3 MB", { bias: true }, "Hi"], "and the settings of a .json");
  for (const chosen of [[file("a.bin", 10)], [file("a.bin", 10), file("b.bin", 5), file("c.bin", 1)]]) {
    await assert.rejects(openLegacy(chosen, () => true), /Choose two files together: the checkpoint and its tokenizer\.bin/, `${chosen.length} files`);
  }
  // a checkpoint of more than a GB is asked about (its size is a File's: none is made here)
  const large = Object.defineProperty(file("big.bin", 0), "size", { value: 2.5e9 });
  const questions = [];
  assert.equal(await openLegacy([large, file("t.bin", 10)], (question) => { questions.push(question); return false; }), undefined, "declined: no model");
  assert.equal((await openLegacy([large, file("t.bin", 10)], () => true)).bytes, 2.5e9, "taken");
  assert.deepEqual(questions, ["big.bin has 2.5 GB. That may be more than this browser can hold. Read it anyway?"]);
  // what Hugging Face publishes: the weights' size is the note's MB
  const hf = await openHuggingFace([file("model.safetensors", 12_600_000), file("config.json", 0, "{}"), file("tokenizer.json", 0, "{}")]);
  assert.deepEqual([hf.name, hf.note, hf.hf.weights], ["model.safetensors", "local · Hugging Face · 13 MB", "model.safetensors"], "a folder of Hugging Face's files");
  await assert.rejects(openHuggingFace([file("model.safetensors", 10), file("config.json", 0, "{}")]), /A Hugging Face model needs three files together/, "no tokenizer");
  cases += 6;
}
{
  // T.468.1: heard.ts, what the page reads of a report first: a late report of a cancelled load is passed over, and
  // only the reports that end a piece of the worker's work give the page its buttons back. Every kind of report the
  // contract has (contracts/messages.ts's FromWorker) is asked, so a kind added there is asked too
  const { ofThisLoad, endsWork } = await fresh("page/heard.ts");
  const { contract } = await import("./contracts.ts");
  assert.deepEqual([ofThisLoad({ type: "status", load: 3 }, 3), ofThisLoad({ type: "status", load: 2 }, 3), ofThisLoad({ type: "threads" }, 3)],
    [true, false, true], "a report of this load is read, one of a load before it is not, and one of no load is");
  const kinds = [...contract("messages.ts").membersOf("FromWorker").keys()].sort();
  assert.deepEqual(kinds.filter((type) => endsWork({ type })), ["bench", "done", "error", "paths", "ready"],
    `of the reports (${kinds.join(", ")}), the ends of a piece of work`);
  // (receiver.ts reads elements at its top, and Node cannot import it: its text asks the two, first and last)
  const receiver = (await import("node:fs")).readFileSync(new URL("../src/page/receiver.ts", import.meta.url), "utf8");
  const handler = receiver.slice(receiver.indexOf("worker.onmessage = ({ data }) => {"));
  assert.ok(/^worker\.onmessage = \(\{ data \}\) => \{\n(\s*\/\/[^\n]*\n)*\s*if \(!ofThisLoad\(data, loads\)\) \{\n\s*return;/.test(handler), "receiver.ts passes over another load's report before anything else");
  assert.ok(/\n  if \(endsWork\(data\)\) \{\n/.test(handler), "receiver.ts gives the page back its buttons where a report ends a piece of work");
  cases += 2;
}
{
  // T.468.2: lines.ts, the status line's words: a ready model's, and a load's as it fetches and converts
  const { readyStatus, progressLine } = await fresh("page/lines.ts");
  const { SLOWER } = await fresh("runtime/updated.ts");
  const ready = { name: "Tiny", backend: "SIMD kernels, int8, relaxed SIMD", threads: 1, pyodide: "314.0.7", gpuTest: false, outdated: false };
  assert.equal(readyStatus(ready), "Tiny · SIMD kernels, int8, relaxed SIMD · Pyodide 314.0.7", "one thread, no GPU, a page of this build");
  assert.equal(readyStatus({ ...ready, threads: 4, gpuNote: "prompts on WebGPU" }), "Tiny · SIMD kernels, int8, relaxed SIMD, 4 threads, prompts on WebGPU · Pyodide 314.0.7",
    "the threads where more than one, then what the GPU takes");
  assert.equal(readyStatus({ ...ready, gpuTest: true }), readyStatus(ready), "?gpuTest=on says nothing where the GPU says nothing");
  assert.equal(readyStatus({ ...ready, gpuNote: "prompts on WebGPU", gpuTest: true, outdated: true }),
    `Tiny · SIMD kernels, int8, relaxed SIMD, prompts on WebGPU (gpuTest) · Pyodide 314.0.7 · ${SLOWER}`, "the tests' flag after the GPU's words, and an old page's words last");
  const progress = { name: "Tiny", received: 2.5e6, total: 10e6 };
  assert.equal(progressLine(progress), "Tiny: 2.5 / 10.0 MB", "what has arrived of how much");
  assert.equal(progressLine({ ...progress, perSecond: 1.25e6, converted: 0.237 }), "Tiny: 2.5 / 10.0 MB · 1.3 MB/s · 23 % converted",
    "T.119: how fast, and the share converted apart (rounded down)");
  assert.equal(progressLine({ ...progress, received: 0, converted: 0.5 }), "Tiny: 50 % converted", "a file of the visitor's disk: the share alone");
  assert.equal(progressLine({ ...progress, received: 0 }), "Tiny: 0.0 / 10.0 MB", "nothing yet, nothing converted: the bytes");
  cases += 8;
}
console.log(`page-parts-check: folder.ts by itself: the models of a folder, of llama2.c's form and of Hugging Face's`);
console.log(`page-parts-check: opening.ts: the URL's model, the one remembered, the default\npage-parts-check: ok (${cases} cases)`);
