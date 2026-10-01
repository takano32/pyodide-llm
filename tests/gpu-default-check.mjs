// T148 (the review's must-fix 2): the page's default choice of the GPU or the CPU for the blocks of a prompt (not the
// tests' gpuForce.always), on forward.js's real CPU forward pass of llm-jp-3 150M, with a made-up GPU's worker that only
// sleeps for as long as its block would take and answers in the control area (no WebGPU: Node alone). Its times are
// set as multiples of the CPU's own ms a prompt token, measured first, so that the verdicts do not hang on this
// machine's speed. Each generation feeds a prompt as Python does (promptBlock at a time). Checked:
//   - a GPU far faster: the first prompt on the CPU (it is timed first), then every block on the GPU; the eighth
//     generation after the first verdict times the CPU again on the prompt's last 32 tokens only; a short prompt
//     stays on the CPU; another number of threads is timed on the CPU again;
//   - a GPU far slower: every prompt on the CPU, but the eighth after the first verdict, whose first block of 64 goes to
//     the GPU once;
//   - a GPU faster from 36 tokens on (the band of 17 to 64 where the first version of T148 put the prompt on blocks of
//     16 on the GPU in its check, 1.47 times as long): the generation that checks again takes no longer than the others.
// T184 (the review's must-fix): /benchmark/'s page path (forward.js's timePrompts, src/bench.js's pathTable) on the same
// made-up GPU: far faster, every block of the GPU's side on it and a ratio; failing on its sixth block, the GPU's cells
// empty with why and no ratio anywhere (its time was the CPU's: 0.90× before); no GPU, the CPU's side alone.
// T190: the page path's number of threads: a search run to its end (forward.js's endSearch) and timed on the count it
// chose; a count the model page remembers taken with no search.
// T199: the search's verdict on a made-up clock (forward.js's clock), with one block of every comparison slowed 3 times:
// the count that is fastest still chosen (4 of the owner's Android's 8 logical cores, 2 of CI's runner's 4).
// T223: on the same clock, the owner's Android's llm-jp-3 150M (2 threads fastest, 4 a third of 1's speed) chooses 2;
// a search timed while the GPU gets ready is not remembered, and is searched again once the GPU is ready (the times
// beside the GPU say 4, those after it 2: 2 is remembered); a count remembered from an earlier visit is searched again
// on the first generation (not only on the eighth); and the page's key of the count is one a model.
// T240: the GPU ready inside a generation: the search owed begins at that generation's next token and ends in it.
// T239: the owner's PC's times (8 remembered, 4 no faster, 2 fastest): 2, by the quarter the search compares after half.
// T152: the steps of a generation (forward.js's tokenBlock and generateMany, tokenTimes) on the same made-up GPU, whose
// step sleeps a multiple of the CPU's own ms of a token and writes made-up ids: generations of STEPS steps after a short
// prompt, as Python takes them (tokenBlock at a time on the GPU, else one on the CPU). Far faster: the first steps on the
// CPU (timed first), then every one on the GPU, and the eighth generation after the first verdict its first 4 on the
// CPU; far slower: every step on the CPU but that generation's first run; failing on its second request: the CPU from
// there on, the step it gave back taken by the CPU. And Python's generate() through external(), as the page's worker
// runs it, on the far faster one: sampled and greedy, the steps on the GPU and the counts right (the made-up GPU fails a
// request whose random numbers, history or settings are not what Python should hand over); and through Python on one
// that fails on its second request: the generation goes on whole on the CPU (T152's review: JavaScript's null is jsnull).
// T219: a GPU that samples -1 (SAMPLE's NONE) or an id past the vocabulary: that request refused whole, the CPU from
// there on, and the status line says why; a model on the GPU alone stops with words and is not loaded again.
// T243: a GPU that writes back a key or a value that is no finite number (a float16 with every bit of its exponent set: a
// NaN, an infinity of either sign), which the CPU's kernels would read as a finite one: the block of the prompt, or the
// request for steps, refused whole (nothing of it in the cache, to the bit), the GPU stopped and the status line says
// why, the CPU on from its own keys and values; in a cache kept in float16 and in a float32 one; a finite block taken
// as the GPU wrote it, the largest halves and the subnormals among them.
// T205: release() waits for the GPU's worker to say "ended" (a slow one; one that never does, terminated after 5 s; one
// stopped while it got ready), and so does Llama.release() called from JavaScript through Python, as the page's worker
// calls it (the review); a browser that does not say the device's memory keeps the steps on the CPU.
//   node tests/gpu-default-check.mjs [--forward <another forward.js, to see a broken one fail>]
import fs from "node:fs";
import path from "node:path";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : path.join(root, "public", "forward.js");

// T243: the keys and values the made-up GPU writes back where it is asked to (line.kv), as float16 bits: of part (the
// keys of every layer, then the values), of the t-th position of the request, the i-th of a position. Finite numbers
// all: of a size a key has, and every so often the largest half, of either sign, zeros, subnormals and the smallest
// normal (none of which is to be taken for a NaN or an infinity)
const madeUpHalf = (part, t, i) => {
  const n = part * 131 + t * 17 + i;
  if (n % 29 === 0) return [0x7bff, 0xfbff, 0x0000, 0x8000, 0x0001, 0x83ff, 0x7800, 0x0400][(n / 29) % 8];
  return (n & 1 ? 0x8000 : 0) | ((12 + (n % 5)) << 10) | ((n * 37) & 0x3ff);
};
// the made-up GPU: ready with the times of a block of 16 and of 64 on its line (fixed + a token), then each block a
// sleep of that long and the answer
const FAKE = `
const { parentPort, workerData: line } = require("node:worker_threads");
let ctl, words, ids, most, blocks = 0, requests = 0, asked = 0, held = null, memory, plan;
// T243: the keys and values of count positions into the staging place ([keys, values][layer][plan.batch positions], in
// float16, as gpu.js writes them back), where line.kv asks for them (zeros where it does not); and in the kv.at-th request of kv.request (a
// "prompt"'s block, or "tokens"), kv.bad: its bits in one of them (side 0 a key, 1 a value; layer, token, column: 0 the
// first, 1 the last). kv.stale: bad halves (this one, and -inf) in the positions after those of the request; kv.stopAfter:
// a request for steps that stops after that many (a stop token: fewer ids than asked for)
const madeUpHalf = ${madeUpHalf.toString()};
const kvAsked = { prompt: 0, tokens: 0 };
const writeBack = (request, count) => {
  const row = plan.kvHeads * plan.headSize, H = new Uint16Array(memory.buffer, plan.staging, 2 * plan.layers * plan.batch * row);
  // (zeros where not asked: the staging place is in a memory the engines before this one wrote other things into,
  // which forward.js would now read as the keys and values of this request)
  for (let part = 0; part < 2 * plan.layers; part++) {
    for (let t = 0; t < count; t++) for (let i = 0; i < row; i++) H[(part * plan.batch + t) * row + i] = line.kv ? madeUpHalf(part, t, i) : 0;
  }
  if (!line.kv) return;
  // (T243's review: the positions after the request's, which the GPU's read-back buffer still holds from before, hold bad
  // halves where line.kv.stale says so: a look at them is none of forward.js's business, the request did not write them)
  if (line.kv.stale !== undefined) {
    for (let part = 0; part < 2 * plan.layers; part++) {
      for (let t = count; t < plan.batch; t++) for (let i = 0; i < row; i++) H[(part * plan.batch + t) * row + i] = (part + t + i) & 1 ? line.kv.stale : 0xfc00;
    }
  }
  const bad = line.kv.request === request && ++kvAsked[request] === line.kv.at ? line.kv.bad : null;
  if (bad) H[((bad.side * plan.layers + (bad.layer ? plan.layers - 1 : 0)) * plan.batch + (bad.token ? count - 1 : 0)) * row + (bad.column ? row - 1 : 0)] = bad.bits;
};
const nap = new Int32Array(new SharedArrayBuffer(4)), ms = (n) => line.fixed + line.perToken * n;
parentPort.on("message", (data) => {
  if (data.type === "start") {
    ctl = new Int32Array(data.memory.buffer, 0, 2048);
    ({ memory, plan } = data);
    words = data.plan.words;
    most = data.plan.tokens?.most;
    // T152: where forward.js asked for the steps and the line has a cost of a step
    ids = data.plan.tokens && line.step !== undefined ? new Int32Array(data.memory.buffer, data.plan.tokens.ids, 2 + data.plan.tokens.most) : null;
    const ready = { type: "ready", adapter: "made up", key: "k", bytes: 1, seconds: 0, form: "made up", attention: "made up",
      forms: [], remembered: false, blocks: [{ count: 16, ms: ms(16) }, { count: 64, ms: ms(64) }],
      ...(ids ? { tokens: { form: "made up", ms: line.step, forms: [] } } : {}) };
    // T223: a GPU that gets ready only when the test says so ({ type: "go" }, sent to this worker alone)
    if (line.readyOnGo) held = ready;
    else parentPort.postMessage(ready);
  } else if (data.type === "go") {
    parentPort.postMessage(held);
  } else if (data.type === "prompt") {
    Atomics.wait(nap, 0, 0, ms(data.count));
    if (Atomics.load(ctl, words.wanted) !== data.serial) return;
    const fail = Boolean(line.failAt) && ++blocks >= line.failAt;  // T184: a GPU that fails on its failAt-th block
    writeBack("prompt", data.count);
    Atomics.store(ctl, words.failed, fail ? 1 : 0);
    Atomics.store(ctl, words.done, data.serial);
    Atomics.notify(ctl, words.done);
  } else if (data.type === "tokens") {
    // T152: count steps of a made-up generation: each id the one after the token fed
    Atomics.wait(nap, 0, 0, line.step * data.count);
    if (Atomics.load(ctl, words.wanted) !== data.serial) return;
    // (T152: what Python hands over, as a step on the GPU takes it: a random number a step where sampled, the end of
    // the history and its length, the settings)
    const odd = data.randoms.length !== (data.settings.temperature ? data.count : 0) || data.history.length > 64 ||
      data.length < data.history.length || data.history.at(-1) !== data.token || !Array.isArray(data.settings.stops);
    const fail = odd || (Boolean(line.failTokensAt) && ++requests >= line.failTokensAt);
    asked++;
    // T219: a GPU whose outsideAt-th request samples outsideId (-1, SAMPLE's NONE, or one past the vocabulary) last
    const outside = Boolean(line.outsideAt) && asked === line.outsideAt;
    // T219 (2): a GPU whose refuseAt-th request's sampler refused the step after refuseAfter sampled ones (its logits
    // not finite): the State's not_finite word after the ids
    const refuse = Boolean(line.refuseAt) && asked === line.refuseAt, sampled = refuse ? line.refuseAfter : Math.min(data.count, line.kv?.stopAfter ?? data.count);
    ids[0] = sampled;
    for (let i = 0; i < sampled; i++) ids[1 + i] = outside && i === data.count - 1 ? line.outsideId : data.token + 1 + i;
    ids[1 + most] = refuse ? 1 : 0;
    writeBack("tokens", sampled);
    Atomics.store(ctl, words.failed, fail ? 1 : 0);
    Atomics.store(ctl, words.done, data.serial);
    Atomics.notify(ctl, words.done);
  } else if (data.type === "stop") {
    // T205: as gpu.js's end(): { type: "ended" } once its device is let go (endAfter ms: a slow let-go; never: a worker
    // that hangs in a compilation), then the worker ends
    if (line.endAfter === Infinity) return;
    setTimeout(() => {
      parentPort.postMessage({ type: "ended" });
      parentPort.close();
    }, line.endAfter ?? 0);
  }
});`;

if (isMainThread) {
  const { pyodideWithEngine } = await import("./engine.mjs");
  const { MODELS } = await import("../src/models.js");
  const { weightsMemory, footprint } = await import(forwardFile);
  const entry = MODELS.find((m) => m.id === "llm-jp-3-150m");
  const { pyodide: py } = await pyodideWithEngine();
  const checkpoint = fs.readFileSync(path.join(root, entry.checkpoint));
  // the plan forward.js gets from Python (where every tensor is), recorded
  let plan;
  py.globals.set("OUTSIDE", { size: checkpoint.length, read: (o, l) => new Uint8Array(checkpoint.buffer, checkpoint.byteOffset + o, l).slice(),
    start: (p) => { plan = p.toJs({ dict_converter: Object.fromEntries }); return { backend: "", bind() {}, forward() {}, release() {} }; } });
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(path.join(root, entry.tokenizer)));
  py.globals.set("OPTIONS", py.toPy(entry.options ?? {}));
  py.runPython(`from llama2_numpy import Llama\nLlama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS).release()`);
  const header = [plan.dim, plan.hidden_dim, plan.n_layers, plan.n_heads, plan.n_kv_heads, plan.vocab_size, plan.seq_len];
  const { memory, base } = weightsMemory(checkpoint.length, { shared: true, after: footprint(header, checkpoint.length, { dtype: "int8", halfKV: true, shared: true, gpu: true }) });
  new Uint8Array(memory.buffer).set(checkpoint, base);
  // forward.js waits in Atomics.wait: in a worker, as in the page
  const worker = new Worker(new URL(import.meta.url), { workerData: { memory, base, size: checkpoint.length, plan, forwardFile,
    tokenizer: path.join(root, entry.tokenizer), options: entry.options ?? {} } });
  const code = await new Promise((resolve) => {
    worker.on("message", (line) => console.log(line));
    worker.once("exit", resolve);
    worker.once("error", (error) => { console.error(`FAILED\n- ${error.stack ?? error}`); resolve(1); });
  });
  process.exit(code);
} else {
  const { memory, base, size, plan, forwardFile, tokenizer, options } = workerData;
  const { compileKernels, createForward, endSearch, timePrompts, external, OUTSIDE_VOCABULARY } = await import(forwardFile);
  const { pathTable } = await import(path.join(root, "src/bench.js"));
  const kernels = compileKernels(fs.readFileSync(path.join(root, "public/simdkernel_shared.wasm")), fs.readFileSync(path.join(root, "public/simdkernel_relaxed_shared.wasm")));
  const spawn = (data) => new Promise((resolve) => {
    const helper = new Worker(path.join(root, "public/helper.js"));
    helper.once("message", () => resolve({ terminate: () => helper.terminate() }));
    helper.postMessage(data);
  });
  const say = (line) => parentPort.postMessage(line);
  const failures = [];
  console.info = () => {};  // forward.js's verdicts: the harness reads what ran where instead
  const PROMPT = 230;  // fed: blocks of 64, 64, 64 and 38
  const prompt = (count) => [plan.bos ?? 1, ...Array.from({ length: count - 1 }, (_, i) => 100 + (i % 500))];
  // one generation's prompt as Python hands it over: [ms, tokens on the GPU]
  const feed = (engine, count = PROMPT) => {
    engine.newGeneration();
    const fed = prompt(count), began = performance.now();
    for (let at = 0; at < fed.length;) {
      const block = fed.slice(at, at + engine.promptBlock);
      engine.forwardMany(block, at);
      at += block.length;
    }
    return [performance.now() - began, engine.gpuTokens];
  };
  // the CPU's ms a prompt token here, on one thread (the median of five prompts of a GPU-less engine)
  const alone = createForward({ memory, base, size, kernels, plan });
  const perToken = [0, 1, 2, 3, 4].map(() => feed(alone)[0] / PROMPT).sort((a, b) => a - b)[2];
  alone.release();
  say(`the CPU: ${perToken.toFixed(2)} ms a prompt token`);

  const run = async (name, fixed, tokenCost, generations, after, failAt = 0) => {
    const line = { fixed: fixed * perToken, perToken: tokenCost * perToken, failAt };
    const gpu = () => {
      const fake = new Worker(FAKE, { eval: true, workerData: line });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
    };
    const engine = createForward({ memory, base, size, kernels, plan, spawn, gpu });
    await engine.setThreads(1);
    await engine.gpu;  // ready before the first prompt
    const seen = [];
    for (let g = 1; g <= generations; g++) seen.push(feed(engine));
    const more = after ? await after(engine) : {};
    engine.release();
    say(`${name}: tokens on the GPU ${seen.map(([, t]) => t).join(" ")}; ms ${seen.map(([ms]) => ms.toFixed(0)).join(" ")}` +
      `${Object.keys(more).length ? `; ${JSON.stringify(more)}` : ""}; status ${engine.gpuStatus}`);
    return { seen, more };
  };
  const expect = (what, got, want) => {
    if (JSON.stringify(got) !== JSON.stringify(want)) failures.push(`${what}: ${JSON.stringify(got)}, not ${JSON.stringify(want)}`);
  };
  const all = (from, to, value) => Array.from({ length: to - from + 1 }, () => value);

  // a GPU far faster: 10 ms-a-token of fixed cost, a twentieth of the CPU's a token (faster from 12 tokens on)
  {
    const { seen, more } = await run("a GPU far faster", 10, 0.05, 12, async (engine) => {
      const short = feed(engine, 6)[1];
      await engine.setThreads(2);
      return { short, twoThreads: [feed(engine)[1], feed(engine)[1]] };
    });
    const tokens = seen.map(([, t]) => t);
    expect("the first prompt on the CPU", tokens[0], 0);
    expect("then every block on the GPU", tokens.slice(1, 8), all(2, 8, PROMPT));
    expect("the eighth generation after the first verdict: the last 32 tokens on the CPU", tokens[8], PROMPT - 38);
    expect("and the GPU again", tokens.slice(9), all(10, 12, PROMPT));
    expect("a short prompt on the CPU", more.short, 0);
    expect("two threads: the CPU timed again, then the GPU", more.twoThreads, [0, PROMPT]);
  }
  // a GPU far slower: 60 fixed and twice the CPU's a token
  {
    const tokens = (await run("a GPU far slower", 60, 2, 12)).seen.map(([, t]) => t);
    expect("every prompt on the CPU but the GPU's check, its first block", tokens, [0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0]);
  }
  // a GPU faster from 36 tokens on (30 fixed, a tenth of the CPU's a token): whole blocks on the GPU, the last block of
  // 38 either way; the check must cost little (the first version of T148: about three times here)
  {
    const { seen } = await run("a GPU faster from 36 tokens", 30, 0.1, 12);
    const normal = seen.slice(1, 8).map(([ms]) => ms).sort((a, b) => a - b)[3], check = seen[8][0];
    expect("the check: the last 32 tokens on the CPU", seen[8][1], PROMPT - 38);
    if (!(check <= 1.5 * normal)) failures.push(`the check took ${check.toFixed(0)} ms, the others ${normal.toFixed(0)} (more than 1.5 times)`);
  }
  // T184: the page path of /benchmark/, as worker.js's timedPaths times it (2 rounds here), and its table
  const paths = (engine) => {
    const ready = engine.gpuReady;  // as timedPaths: what the GPU was before the sides were timed
    const rows = timePrompts(engine, { words: [100, 101, 102, 103, 104], counts: [64, 256], rounds: 2 });
    const gpu = ready ? { seconds: 0, matrices: "made up", attention: "made up", ...(engine.gpuWhyNot ? { lost: engine.gpuWhyNot } : {}) }
      : { why: engine.gpuWhyNot };
    return { rows, table: pathTable({ threads: engine.threads, gpu, status: engine.gpuStatus, rows }) };
  };
  {
    const { more } = await run("the page path, a GPU far faster", 10, 0.05, 0, async (engine) => paths(engine));
    say(more.table);
    expect("the GPU's side all on the GPU", more.rows.map((row) => row.gpu.gpuTokens), [64, 256]);
    expect("as chosen: the GPU", more.rows.map((row) => row.chosen.gpuTokens), [64, 256]);
    expect("the CPU's side on the CPU", more.rows.map((row) => row.cpu.gpuTokens), [0, 0]);
    if ((more.table.match(/×/g) ?? []).length !== 2) failures.push("the page path: a ratio a prompt where the GPU ran them");
  }
  {
    const { more } = await run("the page path, a GPU that fails on its sixth block", 10, 0.05, 0, async (engine) => paths(engine), 6);
    say(more.table);
    // the sixth block is the first of 256's warm-up: 64's GPU side ran whole before it, 256's on the CPU after it
    expect("the GPU's cell of 256 empty, with why", more.rows.map((row) => row.gpu.skip), [undefined, "the GPU failed on a block of the prompt"]);
    if (/×/.test(more.table) || !more.table.includes("WebGPU stopped while timed")) failures.push("the page path: a ratio, or no word of the GPU that stopped");
  }
  {
    const engine = createForward({ memory, base, size, kernels, plan, spawn });
    await engine.setThreads(1);
    const { rows, table } = paths(engine);
    engine.release();
    say(table);
    expect("no GPU: the CPU's side alone", rows.map((row) => [row.chosen.same, row.gpu.skip]), [["cpu", "no WebGPU in a worker here"], ["cpu", "no WebGPU in a worker here"]]);
  }
  // T190: the page path is timed on the model page's number of threads. A search from the logical cores runs to its end
  // (endSearch), on generations of few logits each (the owner's Android timed it on 1 thread where the page runs 4:
  // T184 stopped the search after 8 generations); a count the model page remembers is taken as is, with no search
  {
    const engine = createForward({ memory, base, size, kernels, plan, spawn });
    let chose = 0, tokens = 0;
    await engine.findThreads({ from: 2, chose: (count) => (chose = count) });
    // a generation of 6 tokens with logits: a comparison takes 20 of them, so more than 3 generations
    const write = () => { for (let pos = 0; pos < 6; pos++, tokens++) engine.forward(100 + pos, pos, true); };
    const ended = await endSearch(engine, write);
    const log = engine.searchLog.map(({ best, candidate, faster }) => `${best} or ${candidate}: ${faster ? candidate : best}`);
    engine.release();
    say(`the search to its end: ${ended.threads} threads after ${ended.generations} generations of 6 tokens (${log.join(", ")}), the search chose ${chose}`);
    expect("the search ended", [ended.ended, engine.searching], [true, false]);
    expect("timed on the count the search chose", ended.threads, chose);
    if (!log.length) failures.push("the search to its end: no comparison made");
  }
  {
    const engine = createForward({ memory, base, size, kernels, plan, spawn });
    let tokens = 0;
    await engine.findThreads({ from: 2, remembered: 3 });
    const ended = await endSearch(engine, () => tokens++);
    engine.release();
    expect("a count the model page remembers: no search, that count", [ended.threads, ended.generations, tokens], [3, 0, 0]);
  }
  // T199: the search's verdict stands where one block of 4 timed tokens is slowed whole (a collection of the garbage,
  // another tab). On a made-up clock (forward.js's clock) a token of n threads takes ms[n], and in every comparison the
  // tokens of one block (0 and 3 are the best count's, 1 and 2 the candidate's) take 3 times as long: the owner's
  // Android (4 threads fastest, from its 8 logical cores) and CI's runner where 2 threads were 1.3 times as fast as 1
  // (the upper median chose 1 twice, T190's review)
  for (const [name, ms, from, want] of [["the owner's Android", { 1: 16.4, 2: 10.5, 4: 8.8, 8: 12 }, 8, 4],
                                        ["CI's runner", { 1: 13, 2: 10, 4: 11 }, 4, 2]]) {
    const chosen = [];
    for (const slowed of [null, 0, 1, 2, 3]) {
      let engine = null, starting = false, time = 0, comparison = -1, token = 0;
      const clock = () => {
        starting = !starting;  // the search reads the clock as a token starts and as it ends
        if (starting) return time;
        // the end of a token: its count is the one the engine runs it on. No helper is started on the way to the
        // count wanted (the search goes down from counts whose helpers findThreads started), so a comparison's 20 tokens
        // are its 4 blocks of 5. (A count past ms: a search gone the wrong way)
        if (engine.searchLog.length !== comparison) [comparison, token] = [engine.searchLog.length, 0];
        const block = Math.floor(token++ / 5);
        time += (ms[engine.threads] ?? 30) * (block === slowed ? 3 : 1);
        return time;
      };
      engine = createForward({ memory, base, size, kernels, plan, spawn, clock });
      await engine.findThreads({ from });
      const ended = await endSearch(engine, () => { for (let pos = 0; pos < 6; pos++) engine.forward(100 + pos, pos, true); });
      const log = engine.searchLog.map(({ best, candidate, faster }) => `${best} or ${candidate}: ${faster ? candidate : best}`);
      engine.release();
      chosen.push(`${slowed === null ? "none slowed" : `block ${slowed} slowed`}: ${ended.threads} (${log.join(", ")})`);
      expect(`${name}, ${slowed === null ? "no block" : `block ${slowed}`} slowed 3 times: the count chosen`, [ended.ended, ended.threads], [true, want]);
    }
    say(`the search on a made-up clock, ${name}: ${chosen.join("; ")}`);
  }
  // T223: the owner's Android's writing of llm-jp-3 150M on the CPU alone (/benchmark/, site d531cd5): 1 thread 110,
  // 2 123, 4 38.6, 8 32.3 tok/s; as ms a token
  const LLM_JP = { 1: 1000 / 110, 2: 1000 / 123, 4: 1000 / 38.6, 8: 1000 / 32.3 };
  const madeUpClock = (msNow) => {
    let starting = false, time = 0;
    return () => {
      starting = !starting;  // the search reads the clock as a token starts and as it ends
      if (!starting) time += msNow();
      return time;
    };
  };
  const generation = (engine) => {
    engine.newGeneration();
    for (let pos = 0; pos < 6; pos++) engine.forward(100 + pos, pos, true);
  };
  const verdicts = (engine) => engine.searchLog.map(({ best, candidate, faster, whileGpu }) => `${best} or ${candidate}: ${faster ? candidate : best}${whileGpu ? " (GPU getting ready)" : ""}`);
  {
    let engine = null;
    engine = createForward({ memory, base, size, kernels, plan, spawn, clock: madeUpClock(() => LLM_JP[engine.threads] ?? 40) });
    const told = [];
    await engine.findThreads({ from: 8, chose: (count) => told.push(count) });
    const ended = await endSearch(engine, () => generation(engine));
    say(`T223, llm-jp-3 150M of the owner's Android on a made-up clock: ${ended.threads} (${verdicts(engine).join(", ")})`);
    engine.release();
    expect("T223: llm-jp-3 150M of the owner's Android from its 8 logical cores: 2 threads, remembered", [ended.threads, told], [2, [2]]);
  }
  {
    // the GPU gets ready when told; until then a token's times are those beside it (4 fastest), then llm-jp's
    let engine = null, gpuReady = false, fake = null;
    const beside = { 1: 16, 2: 12, 4: 8, 8: 10 };
    const gpu = () => {
      fake = new Worker(FAKE, { eval: true, workerData: { fixed: 1e3, perToken: 1e3, readyOnGo: true } });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
    };
    engine = createForward({ memory, base, size, kernels, plan, spawn, gpu, clock: madeUpClock(() => (gpuReady ? LLM_JP : beside)[engine.threads] ?? 40) });
    const told = [];
    await engine.findThreads({ from: 8, chose: (count) => told.push(count) });
    const turn = () => new Promise((resolve) => setTimeout(resolve, 0));
    for (let g = 0; g < 200 && engine.searching; g++) {
      generation(engine);
      await turn();
    }
    const beforeReady = { threads: engine.threads, told: [...told], searched: engine.searchLog.length };
    for (let g = 0; g < 3; g++) generation(engine);  // no search begins while the GPU gets ready
    const idle = engine.searching || engine.searchLog.length !== beforeReady.searched;
    fake.postMessage({ type: "go" });
    await engine.gpu;
    gpuReady = true;
    for (let g = 0; g < 200 && (g === 0 || engine.searching); g++) {
      generation(engine);
      await turn();
    }
    say(`T223, a search while the GPU gets ready: ${beforeReady.threads} then ${engine.threads}, remembered ${JSON.stringify(told)} (${verdicts(engine).join(", ")})`);
    const log = engine.searchLog, after = engine.threads;
    engine.release();  // (one thread from here on)
    expect("T223: while the GPU gets ready, the search's 4 used and not remembered", [beforeReady.threads, beforeReady.told], [4, []]);
    expect("T223: no search begins while the GPU gets ready", idle, false);
    expect("T223: the verdicts beside the GPU marked, those after it not", log.map((v) => v.whileGpu),
      log.map((_, i) => i < beforeReady.searched));
    expect("T223: searched again once the GPU is ready: 2, remembered", [after, told], [2, [2]]);
  }
  {
    // T240: the GPU ready inside a generation. The first search ends beside the GPU's getting ready (4, not remembered);
    // the next generation writes 3 tokens, the GPU is ready, and the same generation goes on: the search begins at its
    // next token (not at the next generation's head) and ends in it, 2 remembered
    let engine = null, gpuReady = false, fake = null;
    const beside = { 1: 16, 2: 12, 4: 8, 8: 10 };
    const gpu = () => {
      fake = new Worker(FAKE, { eval: true, workerData: { fixed: 1e3, perToken: 1e3, readyOnGo: true } });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
    };
    engine = createForward({ memory, base, size, kernels, plan, spawn, gpu, clock: madeUpClock(() => (gpuReady ? LLM_JP : beside)[engine.threads] ?? 40) });
    const told = [];
    await engine.findThreads({ from: 8, chose: (count) => told.push(count) });
    const turn = () => new Promise((resolve) => setTimeout(resolve, 0));
    for (let g = 0; g < 200 && engine.searching; g++) {
      generation(engine);
      await turn();
    }
    const before = { threads: engine.threads, told: [...told], searched: engine.searchLog.length };
    engine.newGeneration();
    for (let pos = 0; pos < 3; pos++) engine.forward(100 + pos, pos, true);
    const idle = engine.searching || engine.searchLog.length !== before.searched;
    fake.postMessage({ type: "go" });
    await engine.gpu;
    gpuReady = true;
    engine.forward(103, 3, true);  // the generation goes on: no newGeneration() from here
    const began = engine.searching;
    for (let t = 0; t < 600 && engine.searching; t++) {
      engine.forward(100 + (t % 6), t % 6, true);
      if (t % 6 === 5) await turn();
    }
    say(`T240, the GPU ready inside a generation: ${before.threads} then ${engine.threads}, remembered ${JSON.stringify(told)} (${verdicts(engine).join(", ")})`);
    const log = engine.searchLog, after = engine.threads;
    engine.release();
    expect("T240: beside the GPU's getting ready, the search's 4 used and not remembered", [before.threads, before.told], [4, []]);
    expect("T240: no search begins inside a generation while the GPU gets ready", idle, false);
    expect("T240: the search begins at the token after the GPU is ready, inside the generation", began, true);
    expect("T240: ...and ends in it: 2, remembered, its verdicts not marked", [after, told, log.slice(before.searched).map((v) => v.whileGpu)], [2, [2], [false, false]]);
  }
  {
    // T239: the owner's PC's llm-jp-3 150M (2 threads 171 tok/s, 8 threads 158; 1, 4 and 16 made up to agree with its
    // verdicts "16 or 8: 8, 8 or 4: 8"): the 8 the page remembers there is left for 2, by the quarter (8 or 2)
    const PC = { 1: 1000 / 100, 2: 1000 / 171, 4: 1000 / 150, 8: 1000 / 158, 16: 1000 / 120 };
    let engine = null;
    engine = createForward({ memory, base, size, kernels, plan, spawn, clock: madeUpClock(() => PC[engine.threads] ?? 40) });
    const told = [];
    await engine.findThreads({ from: 16, remembered: 8, chose: (count) => told.push(count) });
    const turn = () => new Promise((resolve) => setTimeout(resolve, 0));
    for (let g = 0; g < 200 && (g === 0 || engine.searching); g++) {
      generation(engine);
      await turn();
    }
    const after = engine.threads, log = verdicts(engine);
    say(`T239, a remembered 8 on the owner's PC's times: ${after}, remembered ${JSON.stringify(told)} (${log.join(", ")})`);
    engine.release();
    expect("T239: a remembered 8 on the owner's PC: 8 or 2 after 8 or 4, then 2 remembered", [log, after, told], [["8 or 4: 8", "8 or 2: 2", "2 or 1: 2"], 2, [2]]);
  }
  {
    // a count remembered from an earlier visit (4, the owner's Android's) is searched again on the first generation
    let engine = null;
    engine = createForward({ memory, base, size, kernels, plan, spawn, clock: madeUpClock(() => LLM_JP[engine.threads] ?? 40) });
    const told = [];
    await engine.findThreads({ from: 8, remembered: 4, chose: (count) => told.push(count) });
    const idle = engine.searching;
    generation(engine);
    const began = engine.searching;
    const turn = () => new Promise((resolve) => setTimeout(resolve, 0));
    for (let g = 0; g < 200 && engine.searching; g++) {
      generation(engine);
      await turn();
    }
    const after = engine.threads;
    say(`T223, a remembered 4 on its first generation: ${after}, remembered ${JSON.stringify(told)} (${verdicts(engine).join(", ")})`);
    engine.release();
    expect("T223: a remembered count: no search as it loads, one on the first generation, 2 remembered", [idle, began, after, told], [false, true, 2, [2]]);
  }
  {
    const { threadsKey } = await import(path.join(root, "src/bench.js"));
    const device = { hardwareConcurrency: 8, deviceMemory: 8, userAgent: "made up" };
    expect("T223: the page remembers the count one a model", threadsKey("llm-jp-3-150m", device) !== threadsKey("tiny-lm", device), true);
  }
  // T190's review: a software thread that stops in the search (T120: the engine gives its helpers up and goes on with
  // one) leaves found at 1 as well, so "fewer than found" never says it: the page path's head reads lostThreads. This
  // helper takes a chunk and ends without counting it (threads-check's)
  {
    const { WAKE, COUNTER, ACTIVE, CONTROL_BYTES } = await import(path.join(root, "public/jobs.js"));
    const dying = ({ memory: shared, share }) => new Promise((resolve) => {
      const helper = new Worker(`
        const { parentPort, workerData: { memory, share, WAKE, COUNTER, ACTIVE, CONTROL_BYTES } } = require("node:worker_threads");
        const ctl = new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4);
        parentPort.postMessage("ready");
        for (let gen = Atomics.load(ctl, WAKE + share); ; gen = Atomics.load(ctl, WAKE + share)) {
          Atomics.wait(ctl, WAKE + share, gen);
          if (Atomics.load(ctl, WAKE + share) & 1) continue;
          Atomics.add(ctl, ACTIVE, 1);
          Atomics.add(ctl, COUNTER, 1);  // a chunk taken, never done
          process.exit(0);
        }`, { eval: true, workerData: { memory: shared, share, WAKE, COUNTER, ACTIVE, CONTROL_BYTES } });
      helper.once("message", () => resolve({ terminate: () => helper.terminate() }));
    });
    const engine = createForward({ memory, base, size, kernels, plan, spawn: dying, stalledMs: 300 });
    await engine.findThreads({ from: 2 });
    const warned = console.warn;
    console.warn = () => {};  // the one line forward.js writes about it
    const ended = await endSearch(engine, () => { for (let pos = 0; pos < 6; pos++) engine.forward(100 + pos, pos, true); });
    console.warn = warned;
    const lost = engine.lostThreads;
    engine.release();
    expect("a software thread that stops in the search: one thread, found 1 too, lostThreads", [ended.threads, ended.found, lost], [1, 1, true]);
  }
  // T152: the steps of a generation. The CPU's ms of a token here (the median of five of a GPU-less engine); a prompt of
  // 16 (on the CPU: the made-up GPU's prompt is slow here), then STEPS steps as Python takes them: [on the GPU, on the CPU]
  const STEPS = 20;
  const cpuStep = (() => {
    const engine = createForward({ memory, base, size, kernels, plan });
    engine.forwardMany(prompt(16), 0);
    const ms = [16, 17, 18, 19, 20].map((pos) => {
      const began = performance.now();
      engine.forward(100, pos);
      return performance.now() - began;
    }).sort((a, b) => a - b)[2];
    engine.release();
    return ms;
  })();
  say(`the CPU: ${cpuStep.toFixed(2)} ms a token with its logits`);
  const write = (engine) => {
    engine.newGeneration();
    const fed = prompt(16);
    engine.forwardMany(fed, 0);
    let token = 100, pos = fed.length, onGpu = 0, onCpu = 0;
    const history = [...fed, token];
    while (pos < fed.length + STEPS) {
      const many = Math.min(engine.tokenBlock, fed.length + STEPS - pos);
      const ids = many > 0 ? engine.generateMany(token, pos, history.slice(-64), history.length, many, 0, 0.9, 1, [], []) : null;
      const chosen = ids ?? (engine.forward(token, pos), [token + 1]);
      if (ids) onGpu += ids.length;
      else onCpu += 1;
      for (const id of chosen) {
        history.push(id);
        token = id;
        pos += 1;
      }
    }
    return [onGpu, onCpu];
  };
  const steps = async (name, stepCost, generations, more = {}) => {
    const line = { fixed: 60 * perToken, perToken: 2 * perToken, step: stepCost * cpuStep, ...more };
    const gpu = () => {
      const fake = new Worker(FAKE, { eval: true, workerData: line });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
    };
    const engine = createForward({ memory, base, size, kernels, plan, spawn, gpu });
    await engine.setThreads(1);
    await engine.gpu;
    const seen = [];
    for (let g = 1; g <= generations; g++) seen.push(write(engine));
    say(`${name}: [on the GPU, on the CPU] ${seen.map((pair) => pair.join("/")).join(" ")}; status ${engine.gpuStatus}`);
    const status = engine.gpuStatus;
    engine.release();
    return Object.assign(seen, { status });
  };
  {
    const seen = await steps("the steps, a GPU far faster", 0.2, 12);
    expect("the first generation: 2 steps on the CPU (timed), the rest on the GPU", seen[0], [STEPS - 2, 2]);
    expect("then every step on the GPU", seen.slice(1, 8), all(2, 8, [STEPS, 0]));
    expect("the eighth after the first verdict: its first 4 on the CPU", seen[8], [STEPS - 4, 4]);
    expect("and the GPU again", seen.slice(9), all(10, 12, [STEPS, 0]));
    // (the owner's words, 2026-09-27: "prompts and answers on WebGPU", or the prompts' own verdict and "answers on WebGPU")
    expect(`the status line says the answers are on WebGPU (${seen.status})`, /answers on WebGPU$/.test(seen.status), true);
  }
  {
    const seen = await steps("the steps, a GPU far slower", 5, 12);
    expect("every step on the CPU but the check's first run", [...seen], [...all(1, 8, [0, STEPS]), [4, STEPS - 4], ...all(10, 12, [0, STEPS])]);
    // ("prompts on WebGPU, answers on the CPU (faster here)", or both on the CPU where the prompts are too)
    expect(`the status line says the answers are on the CPU, faster (${seen.status})`,
      /(, answers on the CPU \(faster here\)|^prompts and answers on the CPU \(faster here than WebGPU\))$/.test(seen.status), true);
  }
  {
    const seen = await steps("the steps, a GPU that fails on its second request", 0.2, 2, { failTokensAt: 2 });
    expect("the CPU from the failure on", [...seen], [[4, STEPS - 4], [0, STEPS]]);
    expect(`the status line says why the prompts are on the CPU, and nothing of the answers (${seen.status})`,
      /^prompts on the CPU \(.*failed on a token/.test(seen.status) && !/answers/.test(seen.status), true);
  }
  // T219: a GPU whose second request samples an id outside the vocabulary last (what SAMPLE gives for logits that are
  // not finite: -1, or one past the end): the whole request refused, the CPU takes the steps from there (and would stop
  // with T195's NOT_FINITE where its own logits were not finite either; here they are), the GPU stopped
  for (const outsideId of [-1, plan.vocab_size]) {
    const seen = await steps(`the steps, a GPU that samples ${outsideId} on its second request`, 0.2, 2, { outsideAt: 2, outsideId });
    expect(`${outsideId}: the CPU from that request on, none of its ids taken`, [...seen], [[4, STEPS - 4], [0, STEPS]]);
    expect(`${outsideId}: the status line says the GPU sampled outside the vocabulary (${seen.status})`,
      new RegExp(`^prompts on the CPU \\(the GPU sampled ${outsideId}, outside the vocabulary`).test(seen.status), true);
  }
  // T219 (2): a GPU whose sampler refused a step (its logits not finite: the State's not_finite word after the ids) on
  // its second request, after one sampled token: the whole request refused (that token not taken either), the CPU from
  // there on, the GPU stopped and the status line says why (as an id outside the vocabulary above)
  {
    const seen = await steps("the steps, a GPU whose sampler refuses a step (logits not finite) on its second request", 0.2, 2, { refuseAt: 2, refuseAfter: 1 });
    expect("refused: the CPU from that request on, its one id not taken", [...seen], [[4, STEPS - 4], [0, STEPS]]);
    expect(`refused: the status line says the GPU computed logits that are not finite (${seen.status})`,
      /^prompts on the CPU \(the GPU computed logits that are not finite numbers/.test(seen.status), true);
  }
  // T219: a model on the GPU alone (T156's direct) cannot take the step on the CPU: generateMany stops with words (the
  // page says them), and does not load the model again on the CPU (onLost: the CPU's logits would be those of the same
  // weights); a request of good ids after it is taken on the GPU as before. (2): the same where the sampler refused the
  // first request's first step
  for (const [outsideId, more] of [[-1, { outsideAt: 1, outsideId: -1 }], [plan.vocab_size, { outsideAt: 1, outsideId: plan.vocab_size }], ["a refused step", { refuseAt: 1, refuseAfter: 0 }]]) {
    const line = { fixed: 10 * perToken, perToken: 0.05 * perToken, step: 0.2 * cpuStep, ...more };
    const fake = new Worker(FAKE, { eval: true, workerData: line });
    const gpu = () => ({ postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
      set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() });
    let lost = null;
    const engine = createForward({ memory, base, size, kernels, plan, gpu, direct: { onLost: (why) => { lost = why; } } });
    await engine.gpu;
    engine.newGeneration();
    const fed = prompt(16);
    engine.forwardMany(fed, 0);
    let thrown = null, first;
    try {
      first = engine.generateMany(100, fed.length, [...fed, 100].slice(-64), fed.length + 1, 4, 0, 0.9, 1, [], []);
    } catch (error) {
      thrown = error.message;
    }
    const second = thrown ? engine.generateMany(100, fed.length, [...fed, 100].slice(-64), fed.length + 1, 4, 0, 0.9, 1, [], []) : undefined;
    say(`T219: on the GPU alone, ${outsideId} sampled: ${thrown ?? `no error (${JSON.stringify(first)})`}; lost ${lost}; then ${JSON.stringify(second)}`);
    expect(`${outsideId} on the GPU alone: stopped with words, not loaded again on the CPU, the GPU still taking the steps`,
      [thrown === OUTSIDE_VOCABULARY, lost, second], [true, null, [101, 102, 103, 104]]);
    await engine.release();
  }
  // T243: a made-up GPU that writes keys and values back (madeUpHalf), one of them no finite number. In a cache kept in
  // float16 (this model's on a shared memory) and in a float32 one (halfKeys false: a grouped-query model's, T160, into
  // which the GPU's are widened); every block and step is put to the GPU (gpuSide), on one thread
  {
    const fast = { fixed: 10 * perToken, perToken: 0.05 * perToken, step: 0.2 * cpuStep };
    const LAYERS = plan.n_layers, ROW = plan.n_kv_heads * (plan.dim / plan.n_heads), FED = 150;  // blocks of 64, 64 and 22
    const wide = new Float16Array(1), narrow = new Uint16Array(wide.buffer);
    const float = (bits) => { narrow[0] = bits; return wide[0]; };
    const bitsOf = (floats) => new Uint32Array(floats.buffer, floats.byteOffset, floats.length);
    const same = (a, b) => { const x = bitsOf(a), y = bitsOf(b); return x.length === y.length && x.every((word, i) => word === y[i]); };
    const sameKv = (a, b) => same(a.keys, b.keys) && same(a.values, b.values);
    // what the cache holds of count positions the made-up GPU wrote back, whose place in their request at(p) gives
    const madeUp = (count, at) => {
      const side = (s) => Float32Array.from({ length: LAYERS * count * ROW }, (_, j) =>
        float(madeUpHalf(s * LAYERS + Math.floor(j / (count * ROW)), at(Math.floor(j / ROW) % count), j % ROW)));
      return { keys: side(0), values: side(1) };
    };
    const notFinite = (where) => new RegExp(`^prompts on the CPU \\(the GPU computed keys or values that are not finite numbers \\(NaN or infinity\\) ${where}\\)`);
    const NAN = 0x7e00, INF = 0x7c00, MINUS_INF = 0xfc00;
    const names = { [NAN]: "a NaN", [INF]: "+inf", [MINUS_INF]: "-inf", 0x7c01: "a NaN of the smallest mantissa", 0xffff: "a NaN of every bit" };
    // one key or one value: of the first layer and of the last, of the request's first position and of its last, in the
    // first column and in the last, each of the three kinds on either side
    const BAD = [{ side: 0, layer: 0, token: 0, column: 0, bits: NAN }, { side: 0, layer: 1, token: 1, column: 1, bits: INF },
      { side: 0, layer: 0, token: 1, column: 0, bits: MINUS_INF }, { side: 1, layer: 0, token: 0, column: 1, bits: INF },
      { side: 1, layer: 1, token: 1, column: 1, bits: NAN }, { side: 1, layer: 1, token: 0, column: 0, bits: MINUS_INF },
      { side: 1, layer: 0, token: 1, column: 0, bits: 0x7c01 }, { side: 0, layer: 1, token: 0, column: 1, bits: 0xffff }];
    const said = (bad) => `${names[bad.bits]} in a ${bad.side ? "value" : "key"} of the ${bad.layer ? "last" : "first"} layer, the ${bad.token ? "last" : "first"} position, the ${bad.column ? "last" : "first"} column`;
    for (const halfKeys of [true, false]) {
      const cache = halfKeys ? "a float16 cache" : "a float32 cache";
      const open = (kv) => {
        const gpu = () => {
          const fake = new Worker(FAKE, { eval: true, workerData: { ...fast, kv } });
          return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
            set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
        };
        return createForward({ memory, base, size, kernels, plan, spawn, halfKeys, ...(kv ? { gpu } : {}) });
      };
      const ready = async (kv) => {
        const engine = open(kv);
        await engine.setThreads(1);
        await engine.gpu;
        engine.gpuSide = "gpu";
        return engine;
      };
      // the CPU alone: what a prompt refused from its first block on is to leave. (The cache is the kind asked for: the
      // CPU's own keys are float16 numbers in the one, and not all of them in the other)
      const cpu = open(null);
      await cpu.setThreads(1);
      feed(cpu, FED);
      const cpuKv = cpu.keysAndValues(0, FED);
      cpu.release();
      expect(`T243, ${cache}: the CPU's own keys are ${halfKeys ? "" : "not all "}float16 numbers`, cpuKv.keys.every((x) => Math.f16round(x) === x), halfKeys);
      // every block finite: all taken, and the cache holds what the GPU wrote, to the bit
      {
        const engine = await ready({ stale: NAN });
        const tokens = feed(engine, FED)[1], got = engine.keysAndValues(0, FED), status = engine.gpuStatus;
        await engine.release();
        // (T243's review: and the positions after the last block's 22, bad ones in the read-back buffer's, are left alone)
        expect(`T243, ${cache}: a prompt of finite keys and values is taken whole, as the GPU wrote it (${status})`,
          [tokens, sameKv(got, madeUp(FED, (p) => p % 64)), /not finite/.test(status ?? "")], [FED, true, false]);
      }
      // the first block refused: nothing on the GPU, the cache the CPU's own, and the next prompt on the CPU too
      const outcomes = [];
      for (const bad of BAD) {
        const engine = await ready({ request: "prompt", at: 1, bad });
        const tokens = feed(engine, FED)[1], got = engine.keysAndValues(0, FED), status = engine.gpuStatus, next = feed(engine, FED)[1];
        await engine.release();
        outcomes.push(`${names[bad.bits]}: ${tokens}`);
        expect(`T243, ${cache}, a block of the prompt with ${said(bad)}: refused, the cache the CPU's own, the next prompt on the CPU (${status})`,
          [tokens, sameKv(got, cpuKv), notFinite("in a block of the prompt at position 0").test(status ?? ""), next], [0, true, true, 0]);
      }
      // the second block refused: the first stays as the GPU wrote it, and the CPU goes on from it as it would have
      // (a GPU whose first block is taken and whose side is then the CPU's)
      {
        const engine = await ready({});
        engine.newGeneration();
        engine.forwardMany(prompt(FED).slice(0, 64), 0);
        engine.gpuSide = "cpu";
        for (let at = 64; at < FED; at += 16) engine.forwardMany(prompt(FED).slice(at, Math.min(at + 16, FED)), at);
        const want = engine.keysAndValues(0, FED);
        await engine.release();
        const bad = BAD[4], refused = await ready({ request: "prompt", at: 2, bad });
        const tokens = feed(refused, FED)[1], got = refused.keysAndValues(0, FED), status = refused.gpuStatus;
        await refused.release();
        outcomes.push(`the second block: ${tokens}`);
        expect(`T243, ${cache}, the second block with ${said(bad)}: refused, the first block the GPU's, the rest the CPU's from it (${status})`,
          [tokens, sameKv(got, want), notFinite("in a block of the prompt at position 64").test(status ?? "")], [64, true, true]);
      }
      say(`T243, ${cache}: tokens on a GPU of a prompt of ${FED} whose keys or values hold ${outcomes.join(", ")}`);
      // the steps: a prompt of 16 and a request of 4 on the GPU (taken: the cache holds them), then a request refused:
      // nothing of it written (the cache at its positions as it was, and not the GPU's), the GPU stopped, and the CPU's
      // step from there what it is after a GPU that only stopped taking steps
      const fed = prompt(16), history = [...fed, 100, 101, 102, 103, 104];
      const ask = (engine, token, pos) => engine.generateMany(token, pos, history.slice(0, pos + 1).slice(-64), pos + 1, 4, 0, 0.9, 1, [], []) ?? null;
      const begin = async (kv) => {
        const engine = await ready(kv);
        engine.newGeneration();
        engine.forwardMany(fed, 0);
        return [engine, ask(engine, 100, 16)];
      };
      const [good, first] = await begin({ stale: NAN });
      const taken = good.keysAndValues(16, 4);
      good.gpuSide = "cpu";
      good.forward(104, 20);
      const want = good.keysAndValues(0, 21);
      await good.release();
      expect(`T243, ${cache}: steps of finite keys and values are taken as the GPU wrote them`, [first, sameKv(taken, madeUp(4, (p) => p))], [[101, 102, 103, 104], true]);
      // (T243's review) a request that stops early, at a stop token: 2 of the 4 steps, the positions after them bad ones in the
      // read-back buffer: only the 2 are looked at and taken
      {
        const [engine, early] = await begin({ stale: NAN, stopAfter: 2 });
        const got = engine.keysAndValues(16, 2), status = engine.gpuStatus, sampled = engine.gpuSampled;
        await engine.release();
        expect(`T243, ${cache}: a request that stops after 2 of 4 steps takes the 2 (${status})`,
          [early, sameKv(got, madeUp(2, (p) => p)), sampled, /not finite/.test(status ?? "")], [[101, 102], true, 2, false]);
      }
      const steps = [];
      for (const bad of BAD) {
        const [engine, before] = await begin({ request: "tokens", at: 2, bad });
        const was = engine.keysAndValues(20, 4), second = ask(engine, 104, 20), now = engine.keysAndValues(20, 4), status = engine.gpuStatus;
        const third = ask(engine, 104, 20), sampled = engine.gpuSampled;
        engine.forward(104, 20);
        const got = engine.keysAndValues(0, 21), finite = engine.logits().every((x) => Number.isFinite(x));
        await engine.release();
        steps.push(`${names[bad.bits]}: ${JSON.stringify(second)}`);
        expect(`T243, ${cache}, steps with ${said(bad)}: the request refused whole, nothing of it written, the GPU stopped, the CPU's step from its own cache (${status})`,
          [before, second, sameKv(was, now), sameKv(now, madeUp(4, (p) => p)), notFinite("in a step at position 20").test(status ?? ""), third, sampled, sameKv(got, want), finite],
          [[101, 102, 103, 104], null, true, false, true, null, 4, true, true]);
      }
      say(`T243, ${cache}: the second request for steps of a GPU whose keys or values hold ${steps.join(", ")}`);
    }
  }
  // T205: the next model is read after the GPU's worker let go of its device (an iPhone's tab went down in
  // /benchmark/'s rounds where the one before still held it). release() waits for its "ended": a slow let-go of 400 ms
  // is waited for; one that never comes is given up after 5 s and the worker terminated; a model let go while its GPU
  // is getting ready waits the same (the late "ready" after the stop changes nothing)
  {
    const fast = { fixed: 10 * perToken, perToken: 0.05 * perToken, step: 0.2 * cpuStep };
    const released = async (more, ready = true) => {
      let exited = false;
      const gpu = () => {
        const fake = new Worker(FAKE, { eval: true, workerData: { ...fast, ...more } });
        fake.once("exit", () => { exited = true; });
        return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
          set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
      };
      const warned = console.warn;
      console.warn = () => {};  // the line forward.js writes of a worker that did not end
      const engine = createForward({ memory, base, size, kernels, plan, gpu });
      if (ready) await engine.gpu;
      const began = performance.now(), ended = await engine.release(), ms = performance.now() - began;
      console.warn = warned;
      await new Promise((resolve) => setTimeout(resolve, 300));
      say(`T205: a GPU's worker ${more.endAfter === Infinity ? "that never ends" : `ending ${more.endAfter} ms after the stop`}` +
        `${ready ? "" : ", stopped while it got ready"}: release() ${ended ? "saw it end" : "gave it up"} in ${ms.toFixed(0)} ms` +
        `, the worker ${exited ? "gone" : "still there"}; status ${engine.gpuStatus}`);
      return { ended, ms, exited, status: engine.gpuStatus };
    };
    const slow = await released({ endAfter: 400 });
    expect("release() waits for a slow GPU's worker to end", [slow.ended, slow.ms >= 380 && slow.ms < 2500, slow.exited], [true, true, true]);
    const hung = await released({ endAfter: Infinity });
    expect("release() gives up a GPU's worker that never ends after 5 s, and terminates it",
      [hung.ended, hung.ms >= 4900 && hung.ms < 7000, hung.exited], [false, true, true]);
    const early = await released({ endAfter: 200 }, false);
    expect("a model let go while its GPU gets ready: waited for, and the late ready ignored",
      [early.ended, early.ms >= 180 && early.ms < 2500, early.exited, /WebGPU/.test(early.status ?? "") && !/CPU/.test(early.status ?? "")], [true, true, true, false]);
  }
  // T205: a browser that does not say the device's memory (Safari, Firefox: no navigator.deviceMemory) keeps a
  // generation's steps on the CPU (no classifier and embedding on the GPU), and the prompts' blocks on the GPU
  {
    const fast = { fixed: 10 * perToken, perToken: 0.05 * perToken, step: 0.2 * cpuStep };
    const gpu = () => {
      const fake = new Worker(FAKE, { eval: true, workerData: fast });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
    };
    const outcome = async (memoryUnsaid) => {
      const engine = createForward({ memory, base, size, kernels, plan, spawn, gpu, memoryUnsaid });
      await engine.setThreads(1);
      await engine.gpu;
      const prompts = [1, 2, 3].map(() => feed(engine)[1]);
      const written = [1, 2, 3].map(() => write(engine));
      const got = { planned: engine.gpuTokensPlanned, why: engine.gpuTokensWhyNot, prompts, written, status: engine.gpuStatus };
      await engine.release();
      say(`T205: memoryUnsaid ${memoryUnsaid}: ${JSON.stringify(got)}`);
      return got;
    };
    const unsaid = await outcome(true), said = await outcome(false);
    expect("no deviceMemory: the steps not asked of the GPU, and why", [unsaid.planned, /does not say how much memory/.test(unsaid.why ?? "")], [false, true]);
    expect("no deviceMemory: the prompts' blocks still on the GPU (after the first, timed on the CPU)", unsaid.prompts.slice(1), [PROMPT, PROMPT]);
    expect("no deviceMemory: every step on the CPU", unsaid.written, all(1, 3, [0, STEPS]));
    expect("no deviceMemory: the status line says no reason (the console does)", /memory/.test(unsaid.status ?? ""), false);
    expect("with deviceMemory: the steps asked of the GPU, and taken there", [said.planned, said.written.at(-1)[0] > 0], [true, true]);
  }
  // T152: Python's generate() through forward.js's external() (as the page's worker has it) on the made-up GPU far
  // faster: the steps go to it (Python draws the random numbers and hands the history over as the made-up GPU expects),
  // and the text and the counts are those of the steps
  {
    const { loadPyodide } = await import("pyodide");
    const py = await loadPyodide();
    await py.loadPackage("numpy", { messageCallback: () => {} });
    for (const name of ["llama2_numpy.py", "llama2_convert.py", "simdkernel.so", "simdkernel_relaxed.wasmlib"]) {
      py.FS.writeFile(name, fs.readFileSync(path.join(root, "public", name)));
    }
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(tokenizer));
    const line = { fixed: 60 * perToken, perToken: 2 * perToken, step: 0.2 * cpuStep };
    const gpuOf = (workerData) => () => {
      const fake = new Worker(FAKE, { eval: true, workerData });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
    };
    // (T205's review: its worker says ended 300 ms after the stop, for the release through Python below)
    const outside = external({ memory, base, size, kernels, gpu: gpuOf({ ...line, endAfter: 300 }) });
    py.globals.set("OUTSIDE", outside);
    py.globals.set("OPTIONS", py.toPy(options));
    py.runPython(`from llama2_numpy import Llama\nllama = Llama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
    await outside.engine.gpu;
    const written = [];
    for (const settings of ["temperature=0.8, topp=0.9, repetition_penalty=1.1, seed=3", "temperature=0.0"]) {
      outside.engine.newGeneration();
      py.runPython(`text = "".join(llama.generate("こんにちは、今日は", steps=48, ${settings}))`);
      const stats = py.runPython("llama.stats").toJs({ dict_converter: Object.fromEntries });
      written.push({ sampled: stats.sampled, gpu: outside.engine.gpuSampled, prompt: stats.prompt_tokens, text: py.globals.get("text").length });
    }
    // T205's review: the release as the page's worker makes it (worker.js's load()): Llama.release() called from
    // JavaScript hands forward.js's promise back through Python, and the next model is read once it settles. An await
    // of anything else (None, a PyProxy) waits for nothing, and every other test here would still pass
    const pythonLlama = py.globals.get("llama");
    const releasedAt = performance.now(), released = pythonLlama.release?.();
    const promised = typeof released?.then === "function";
    const endedThrough = promised ? await released : undefined, releaseMs = performance.now() - releasedAt;
    pythonLlama.destroy();
    say(`T205: Llama.release() through Python: ${promised ? `a promise, ${endedThrough ? "its worker ended" : "not ended"} after ${releaseMs.toFixed(0)} ms` : `${released} (no promise)`}`);
    if (!(promised && endedThrough === true && releaseMs >= 280)) {
      failures.push(`Llama.release() through Python: ${promised ? `${endedThrough} after ${releaseMs.toFixed(0)} ms (its worker ends 300 ms after the stop)` : "no promise to wait for"}`);
    }
    say(`Python's generate() on the made-up GPU: ${JSON.stringify(written)}; status ${outside.engine.gpuStatus}`);
    written.forEach(({ sampled, gpu, prompt, text }, i) => {
      if (!(sampled === 48 - prompt && gpu >= sampled - 2 && text > 0)) {
        failures.push(`Python's generate() ${i ? "greedy" : "sampled"}: ${sampled} sampled, ${gpu} on the GPU, ${text} characters`);
      }
    });
    // T152's review: the same through Python on a GPU that fails on its second request. generateMany gives the steps
    // back (null), and Python takes them on the CPU: the generation goes on whole (Pyodide makes JavaScript's null
    // jsnull, not None, unless asked: generateMany says undefined, which is None)
    const failing = external({ memory, base, size, kernels, gpu: gpuOf({ ...line, failTokensAt: 2 }) });
    py.globals.set("OUTSIDE", failing);
    py.runPython(`llama = Llama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
    await failing.engine.gpu;
    failing.engine.newGeneration();
    let broke = null;
    try {
      py.runPython(`text = "".join(llama.generate("こんにちは、今日は", steps=48, temperature=0.8, topp=0.9, repetition_penalty=1.1, seed=3))`);
    } catch (error) {
      broke = String(error?.message ?? error).split("\n").filter(Boolean).at(-1);
    }
    const after = py.runPython("llama.stats").toJs({ dict_converter: Object.fromEntries });
    py.runPython("llama.release()");
    say(`Python's generate() on a GPU that fails on its second request: ${broke ?? `${after.sampled} sampled, ${failing.engine.gpuSampled} on the GPU`}; status ${failing.engine.gpuStatus}`);
    if (broke || !(after.sampled === 48 - after.prompt_tokens && failing.engine.gpuSampled === 4 && /failed on a token/.test(failing.engine.gpuStatus ?? ""))) {
      failures.push(`Python's generate() on a GPU that fails: ${broke ?? `${after.sampled} sampled, ${failing.engine.gpuSampled} on the GPU, ${failing.engine.gpuStatus}`}`);
    }
  }
  if (failures.length) say(`FAILED\n- ${failures.join("\n- ")}`);
  else say("ok");
  process.exit(failures.length ? 1 : 0);
}
