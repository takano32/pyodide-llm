// tests/threads-part-check.mjs (T378.1): the software threads' part of the engine (src/runtime/forward/threads.js's
// softwareThreads()) by itself, with stand-ins for everything beside it: a small shared memory of its own, "threads"
// that are started at once and do nothing (the coordinator then takes every chunk itself), a runRows that writes down
// what it was asked for, a GPU that is getting ready when the test says so, and times the test makes up (the part reads
// no clock: it is told a token's milliseconds). No engine, no kernels, no model, no Pyodide. What the part is still
// bound to: jobs.js (where the words of the control area are) and choice.js's lowerMedian and BETTER.
//
// The state has one owner (T378.1): the rest of the engine tells it what happened (use, find, generationBegan,
// tokenBegan and tokenEnded, stop) and reads count, lost and searching. Held here:
//   - each of those changes what it says it changes, and nothing else that can be read;
//   - a phase's rows are each run once, whole by one caller, with any count, and alone where there are no threads;
//   - the search's turns: best, candidate, candidate, best in blocks of 5, the count in use the best one between tokens;
//   - what is told in a wrong order is harmless (said one by one below), and nothing is told the page after a give-up;
//   - T120: a thread that took a chunk and never did it is given up, the phase is run again here, one thread from then on;
//   - no threads at all where none can be started (no spawn: a model on the GPU alone, or a memory that is not shared).
//
//   node tests/threads-part-check.mjs        (Node only, under two seconds)
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";

const { softwareThreads } = await import(runtimeUrl("forward/threads.js"));
const { controlArea } = await import(runtimeUrl("forward/control.js"));
const { CONTROL_BYTES, JOB, ROWS, COUNT, COUNTER, QUIT, JOB_TABLE, BATCH } = await import(runtimeUrl("jobs.ts"));

const PAGE = 65536;
assert.ok(JOB_TABLE + BATCH * JOB * 8 <= PAGE && CONTROL_BYTES <= PAGE, "the control area fits the one page of this test's memory");
const jobOf = (rows, count = 1) => {
  const job = new Array(JOB).fill(0);
  job[ROWS] = rows;
  job[COUNT] = count;
  return job;
};

/** The part and its stand-ins. spawn: false for none; late: the threads come up only when started.release() is called */
function made({ spawn = true, sharedMemory = true, stalledMs = 40, late = false } = {}) {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1, shared: sharedMemory });
  const beside = { memory, started: 0, ended: 0, ran: [], gpuGettingReady: false, onRun: null, waiting: [] };
  const helper = () => ({ terminate() { beside.ended += 1; } });
  const start = (data) => {
    beside.started += 1;
    assert.equal(data.memory, memory, "a thread is started on the engine's memory");
    assert.equal(data.share, beside.started - beside.ended, "a thread is told which of them it is");
    assert.deepEqual([data.plain, data.relaxed, data.wide], ["plain", null, false], "a thread is started with the two kernels the part was handed");
    return late ? new Promise((resolve) => beside.waiting.push(() => resolve(helper()))) : Promise.resolve(helper());
  };
  // (the control area is the real one, on this test's page of memory: T378.2, it is handed to the part)
  const control = controlArea({ memory, sharedMemory, stalledMs });
  beside.ctl = control.ctl;
  const part = softwareThreads({ memory, wide: false, kernels: { plain: "plain", relaxed: null }, spawn: spawn ? start : null, control,
    runRows: (job, from, to) => {
      beside.ran.push([job, from, to]);
      beside.onRun?.();
    },
    gpuGettingReady: () => beside.gpuGettingReady });
  return { part, beside };
}
const read = (part) => ({ threads: part.count, lost: part.lost, searching: part.searching, log: part.searchLog.length });
// every row of every job run once, and in how many pieces
function covered(beside, jobs) {
  for (const job of jobs) {
    const seen = new Array(job[ROWS]).fill(0);
    for (const [ran, from, to] of beside.ran) if (ran === job) for (let r = from; r < to; r++) seen[r] += 1;
    assert.ok(seen.every((times) => times === 1), `every row of a phase's job is run once (${seen.join("")})`);
  }
  const pieces = beside.ran.length;
  beside.ran.length = 0;
  return pieces;
}
// a token that makes logits, as the engine runs one: timed where the part says it is the search's. ms: a count's time
const token = (part, ms) => {
  if (!part.tokenBegan(true)) return null;
  const on = part.count;
  part.tokenEnded(typeof ms === "function" ? ms(on) : ms[on] ?? 30);
  return on;
};
const until = (part, ms, limit = 400) => {
  let tokens = 0;
  while (part.searching && tokens < limit) tokens += token(part, ms) === null ? 0 : 1;
  assert.ok(!part.searching, "the search ends");
  return tokens;
};
const ANDROID = { 1: 1000 / 110, 2: 1000 / 123, 4: 1000 / 38.6, 8: 1000 / 32.3 };
const quietly = async (run) => {
  const warn = console.warn, said = [];
  console.warn = (...words) => said.push(words.join(" "));
  try { await run(); } finally { console.warn = warn; }
  return said;
};

// ---- as made: one thread, nothing lost, no search; a phase is run here, whole
{
  const { part, beside } = made();
  assert.deepEqual(read(part), { threads: 1, lost: false, searching: false, log: 0 });
  const jobs = [jobOf(64), jobOf(7), jobOf(1)];
  part.phase(jobs);
  assert.deepEqual(beside.ran.map(([job, from, to]) => [jobs.indexOf(job), from, to]), [[0, 0, 64], [1, 0, 7], [2, 0, 1]], "one thread: each job whole, in order");
  assert.equal(beside.started, 0, "no thread is started before one is asked for");
  assert.deepEqual(Object.keys(part).sort(), ["count", "find", "generationBegan", "lost", "phase", "searchLog", "searching", "stop", "tokenBegan", "tokenEnded", "use"],
    "what the part hands out");
  for (const name of ["count", "lost", "searching"]) {
    assert.equal(Object.getOwnPropertyDescriptor(part, name).set, undefined, `${name} cannot be assigned from outside`);
  }
}
console.log("ok: as made the part runs on one thread, and nothing of its state can be assigned from outside");

// ---- use(n): the count, the threads that are missing started, a search dropped; a phase's rows each once
{
  const { part, beside } = made();
  assert.equal(await part.use(4), 4);
  assert.deepEqual([read(part), beside.started], [{ threads: 4, lost: false, searching: false, log: 0 }, 3], "use(4): 4 in use, 3 started");
  const jobs = [jobOf(64), jobOf(7), jobOf(64, 4)];
  part.phase(jobs);
  const pieces = covered(beside, jobs);
  assert.ok(pieces > jobs.length, `4 threads: a phase's rows are cut into chunks (${pieces})`);
  assert.equal(await part.use(2), 2);
  assert.equal(beside.started, 3, "use(2) after use(4): none more started");
  part.phase(jobs);
  assert.ok(covered(beside, jobs) < pieces, "2 threads: fewer chunks than 4");
  assert.equal(await part.use(0), 1, "a count below 1 is 1");
  assert.equal(part.count, 1);
  await part.find({ from: 4, chose() {} });
  assert.equal(part.searching, true);
  assert.equal(await part.use(2), 2);
  assert.deepEqual(read(part), { threads: 2, lost: false, searching: false, log: 0 }, "use() ends a search under way");
  assert.equal(token(part, ANDROID), null, "and no token is the search's after it");
}
console.log("ok: use(n) sets the count and starts the threads that are missing; a phase runs every row once with any count");

// ---- find(): from a hint a search begins at once; from a remembered count it is owed, and begins with a generation
{
  const { part, beside } = made();
  const told = [], verdicts = [];
  assert.equal(await part.find({ from: 8, chose: (count) => told.push(count), compared: (verdict) => verdicts.push(verdict) }), 8);
  assert.deepEqual([read(part), beside.started], [{ threads: 8, lost: false, searching: true, log: 0 }, 7], "find from 8: 8 in use, searching");
  // the turns of one comparison: best, candidate, candidate, best, 5 tokens each; the best between tokens
  const turns = [];
  for (let i = 0; i < 20; i++) {
    assert.equal(part.tokenBegan(true), true);
    turns.push(part.count);
    part.tokenEnded(ANDROID[part.count]);
    if (i < 19) assert.equal(part.count, 8, "between the search's tokens the count in use is the best so far");
  }
  assert.deepEqual(turns, [...Array(5).fill(8), ...Array(10).fill(4), ...Array(5).fill(8)], "a comparison's turns");
  assert.deepEqual(part.searchLog.map(({ best, candidate, faster, times }) => [best, candidate, faster, times[best].length, times[candidate].length]), [[8, 4, true, 8, 8]],
    "the first token of each block of 5 is not timed");
  assert.deepEqual([part.count, part.searching, told], [4, true, []], "the faster candidate is the best now, and the search goes on");
  until(part, ANDROID);
  assert.deepEqual([part.count, told, verdicts.map(({ best, candidate, faster }) => `${best} ${candidate} ${faster}`)], [2, [2], ["8 4 true", "4 2 true", "2 1 false"]], "the Android's search ends on 2 and says so once");
  assert.equal(verdicts.length, part.searchLog.length, "every comparison is told");
  assert.equal(token(part, ANDROID), null, "no token is the search's once it has ended");
  // generationBegan(): the count is checked again every `recheck` generations (8 unless said), counted from the load
  const began = [];
  for (let generation = 1; generation <= 17; generation++) {
    part.generationBegan();
    if (part.searching) began.push(generation);
    until(part, ANDROID);
  }
  assert.deepEqual(began, [8, 16], "a search again on every 8th generation");
  assert.deepEqual(told, [2, 2, 2]);
}
{
  const { part } = made();
  const told = [];
  assert.equal(await part.find({ from: 8, remembered: 4, recheck: 3, chose: (count) => told.push(count) }), 4);
  assert.deepEqual(read(part), { threads: 4, lost: false, searching: false, log: 0 }, "a remembered count is used as it is, with no search yet");
  assert.equal(part.tokenBegan(true), false, "T240: no search inside a text before the page began a generation (/benchmark/)");
  part.generationBegan();
  assert.equal(part.searching, true, "T223: the first generation searches again from a remembered count");
  until(part, ANDROID);
  assert.deepEqual([part.count, told], [2, [2]]);
  const began = [];
  for (let generation = 2; generation <= 7; generation++) {
    part.generationBegan();
    if (part.searching) began.push(generation);
    until(part, ANDROID);
  }
  assert.deepEqual(began, [3, 6], "recheck: 3 is every third generation");
}
{
  // T223, T240: nothing is searched beside a GPU that is getting ready; what was timed beside it is not remembered
  const { part, beside } = made();
  const told = [];
  beside.gpuGettingReady = true;
  await part.find({ from: 8, remembered: 4, chose: (count) => told.push(count) });
  part.generationBegan();
  assert.equal(part.searching, false, "no search begins while the GPU gets ready");
  assert.equal(part.tokenBegan(true), false);
  beside.gpuGettingReady = false;
  assert.equal(part.tokenBegan(false), false, "a token that makes no logits is never the search's");
  assert.equal(part.searching, true, "T240: the search owed begins at the first token after the GPU is ready");
  const before = read(part);
  for (let i = 0; i < 30; i++) assert.equal(part.tokenBegan(false), false);
  assert.deepEqual(read(part), before, "a prompt's tokens leave the search where it was");
  until(part, ANDROID);
  assert.deepEqual(told, [2]);
  // a first search beside the GPU: its verdict is used, not told, and searched again once the GPU is ready
  const other = made(), remembered = [];
  other.beside.gpuGettingReady = true;
  await other.part.find({ from: 8, chose: (count) => remembered.push(count) });
  until(other.part, ANDROID);
  assert.deepEqual([other.part.count, remembered, other.part.searchLog.every(({ whileGpu }) => whileGpu)], [2, [], true], "a verdict beside the GPU is used and not told");
  other.beside.gpuGettingReady = false;
  other.part.generationBegan();
  assert.equal(other.part.searching, true);
  until(other.part, ANDROID);
  assert.deepEqual(remembered, [2], "and told once it was found again with the GPU ready");
}
console.log("ok: find() searches from a hint at once and from a remembered count with the first generation; generationBegan() again every recheck; never beside a GPU getting ready");

// ---- T451: a token's time is counted for the count it ran on, and for no other. Every token is told a time of its
// own (its count's hundreds and its turn), so a time put with another count of the comparison, which is there to take
// it, shows as that: before this, such a break ended in a TypeError or in another search's verdict
{
  const { part } = made();
  await part.find({ from: 4, chose() {} });
  const ran = { 4: [], 2: [] }, order = [];
  for (let i = 0; i < 20; i++) {
    assert.equal(part.tokenBegan(true), true);
    const on = part.count, ms = 100 * on + i;
    order.push(on);
    if (i % 5) ran[on].push(ms);  // the first token of a block of 5 is the switch: not timed
    part.tokenEnded(ms);
  }
  assert.deepEqual(order, [...Array(5).fill(4), ...Array(10).fill(2), ...Array(5).fill(4)], "best, candidate, candidate, best");
  const [{ best, candidate, times }] = part.searchLog;
  assert.deepEqual([best, candidate], [4, 2]);
  assert.deepEqual(times[4], ran[4], "the times of the tokens that ran on 4 threads are 4's, each once, in order");
  assert.deepEqual(times[2], ran[2], "the times of the tokens that ran on 2 threads are 2's, each once, in order");
}
console.log("ok: T451, a token's time is counted for the count it ran on");

// ---- a count that needs more threads: untimed on the best count until they are there
{
  const { part, beside } = made({ late: true });
  const finding = part.find({ from: 1, chose() {} });  // (1 needs none: resolves; its candidate 2 needs one)
  assert.equal(await finding, 1);
  assert.deepEqual([part.searching, beside.started, beside.waiting.length], [true, 1, 1]);
  for (let i = 0; i < 25; i++) {
    assert.equal(part.tokenBegan(true), true);
    assert.equal(part.count, 1, "the best count while the candidate's thread starts");
    part.tokenEnded(5);
  }
  assert.equal(part.searchLog.length, 0, "and nothing is timed meanwhile");
  beside.waiting.shift()();
  await new Promise((resolve) => setTimeout(resolve, 0));
  const turns = [];
  for (let i = 0; i < 20; i++) {
    part.tokenBegan(true);
    turns.push(part.count);
    part.tokenEnded(5);
  }
  assert.deepEqual([turns.slice(0, 10).join(""), part.searchLog.length], ["1111122222", 1], "timed from the token after the thread is there");
  // T120's review: a thread that comes up after the part let its threads go is ended at once
  const using = part.use(4);
  part.stop();
  beside.waiting.splice(0).forEach((release) => release());
  assert.equal(await using, 1, "use() that was overtaken by stop() says 1");
  assert.deepEqual([part.count, beside.started, beside.ended], [1, beside.started, beside.started], "and every thread started is ended");
}
console.log("ok: a candidate's threads are waited for on the best count, untimed; a thread that comes up after stop() is ended");

// ---- what is told in a wrong order
{
  const { part, beside } = made();
  // before anything: a generation, a token and a token's end change nothing
  part.generationBegan();
  assert.equal(part.tokenBegan(true), false);
  part.tokenEnded(5);
  part.stop();
  assert.deepEqual([read(part), beside.started], [{ threads: 1, lost: false, searching: false, log: 0 }, 0], "a generation, a token, an end and a stop before any count was asked for: nothing");
  await part.find({ from: 4, chose() {} });
  // a token's end with no token begun, and one told twice: the comparison still takes its 20 tokens
  part.tokenEnded(1);
  part.tokenEnded(1);
  for (let i = 0; i < 19; i++) {
    part.tokenBegan(true);
    part.tokenEnded(ANDROID[part.count]);
    part.tokenEnded(0.001);  // told again: nothing
  }
  assert.equal(part.searchLog.length, 0, "an end with no token begun counts for nothing");
  part.tokenBegan(true);
  part.tokenBegan(true);  // begun twice: the one turn
  part.tokenEnded(ANDROID[part.count]);
  assert.deepEqual([part.searchLog.length, part.searchLog[0].times[4].length, Math.min(...part.searchLog[0].times[2])], [1, 8, ANDROID[2]], "a comparison is 20 tokens' ends, each counted once");
  // stop() in the middle of a search: one thread now; the search finds none of its candidate's threads and ends
  part.stop();
  part.stop();
  assert.equal(part.count, 1, "stop(): one thread");
  assert.equal(beside.ended, beside.started, "stop(): every thread ended");
  assert.equal(Atomics.load(beside.ctl, QUIT), 1, "stop(): the word the threads end on is set");
  // and the part is usable again
  assert.equal(await part.use(2), 2);
  assert.equal(Atomics.load(beside.ctl, QUIT), 0, "a fresh start clears it");
  const jobs = [jobOf(64)];
  beside.ran.length = 0;
  part.phase(jobs);
  covered(beside, jobs);
  // find() twice: the second one's callbacks and count
  const first = [], second = [];
  await part.find({ from: 4, chose: (count) => first.push(count) });
  await part.find({ from: 8, chose: (count) => second.push(count) });
  until(part, ANDROID);
  assert.deepEqual([first, second], [[], [2]], "find() told twice: the second is the one in force");
}
console.log("ok: a generation, a token or its end told out of turn change nothing; stop() twice, use() after stop(), find() twice are what they say");

// ---- T120: a thread that took a chunk and never did it
{
  const { part, beside } = made({ stalledMs: 40 });
  const told = [];
  await part.find({ from: 2, chose: (count) => told.push(count) });
  assert.equal(part.searching, true);
  const jobs = [jobOf(64), jobOf(9)];
  beside.onRun = () => {
    beside.onRun = null;
    Atomics.add(beside.ctl, COUNTER, 1);  // as a thread that took the next chunk and was stopped by the browser
  };
  assert.equal(part.tokenBegan(true), true);
  const said = await quietly(() => part.phase(jobs));
  assert.deepEqual(read(part), { threads: 1, lost: true, searching: false, log: 0 }, "given up: one thread, no search");
  assert.equal(said.length, 1, "and one line says so");
  assert.equal(beside.ended, beside.started, "its threads are ended");
  // the phase was run again here, whole: after the chunks of the first try, each job once from its first row to its last
  assert.deepEqual(beside.ran.slice(-2).map(([job, from, to]) => [jobs.indexOf(job), from, to]), [[0, 0, 64], [1, 0, 9]], "the phase is run again on this thread");
  part.tokenEnded(5);  // the token that was the search's ends: nothing
  assert.deepEqual([read(part), told], [{ threads: 1, lost: true, searching: false, log: 0 }, []], "one thread after a give-up is not told the page");
  assert.equal(await part.use(4), 1, "no threads again under this engine");
  assert.equal(await part.find({ from: 8, chose: (count) => told.push(count) }), 1);
  part.generationBegan();
  for (let i = 0; i < 40; i++) token(part, ANDROID);
  assert.deepEqual([read(part), told, beside.ended], [{ threads: 1, lost: true, searching: false, log: 0 }, [], beside.started], "nor a search");
  beside.ran.length = 0;
  part.phase(jobs);
  assert.equal(beside.ran.length, 2, "its phases run here");
}
console.log("ok: T120, a thread that never does its chunk is given up, the phase run again here, and the engine keeps to one thread");

// ---- no threads where none can be started
for (const [title, options] of [["no spawn (a model on the GPU alone)", { spawn: false }], ["a memory that is not shared", { sharedMemory: false }]]) {
  const { part, beside } = made(options);
  const told = [];
  assert.equal(await part.use(4), 1, title);
  assert.equal(await part.find({ from: 8, remembered: 4, chose: (count) => told.push(count) }), 1, title);
  part.generationBegan();
  for (let i = 0; i < 40; i++) assert.equal(token(part, ANDROID), null, title);
  part.stop();
  assert.deepEqual([read(part), told, beside.started], [{ threads: 1, lost: false, searching: false, log: 0 }, [], 0], title);
  if (beside.ctl) assert.equal(Atomics.load(beside.ctl, QUIT), 0, `${title}: stop() has nothing to stop and writes nothing`);
  else assert.equal(options.sharedMemory, false, "no control area only on a memory that is not shared");
  const jobs = [jobOf(64)];
  part.phase(jobs);
  assert.deepEqual(beside.ran.map(([, from, to]) => [from, to]), [[0, 64]], title);
}
console.log("ok: with no spawn or on a memory that is not shared there is one thread whatever is asked, and no search");
