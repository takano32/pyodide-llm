// tests/worker-conduct-check.mjs (T374.2.1)
// The worker's side of a conversion's conduct (public/worker/conduct.js, and convert() of public/worker/convert.js
// around it), alone: the conduct is a stand-in that plays a list of requests (a JavaScript generator behind the face
// Pyodide gives a Python one: next(answer) and return(), each request a proxy to be destroyed), the network is
// tests/worker-harness.mjs's, and there is no Pyodide. What is seen: every kind of request answered with what it asks
// for, a file that is not there answered with undefined (never null), the 404 of the file a conduct ends on thrown in
// its own words, every other failure and a cancelled load ending the loop with the generator closed and nothing left
// open, every proxy destroyed once, and (T403) the file opened to keep the conversion in let go whatever ended it.
//
//   node tests/worker-conduct-check.mjs
//
// What this does not see: the real conduct (tests/worker-fetches-check.mjs runs it in Pyodide with this loop, on a
// made-up hub; tests/test_conduct.py alone), a real line, and a browser's own file system (tests/e2e.mjs).
import assert from "node:assert/strict";
import { workerHarness } from "./worker-harness.mjs";

setTimeout(() => {
  console.error("worker-conduct-check: still waiting after 120 s");
  process.exit(1);
}, 120000).unref();

const REVISION = "0123456789abcdef0123456789abcdef01234567", OTHER = REVISION.split("").reverse().join("");
const HEADER = [64, 128, 2, 4, 4, 256, 128], FORM = { bias: false, arch: "llama", qk_norm: false, head_dim: 0, linear: null, rotated: null, convolution: null };
let passed = 0;
const ok = (name) => { passed++; console.log(`ok: ${name}`); };

// ---- the stand-ins. Every proxy counts how often it was destroyed, and refuses to be read afterwards
function world({ gpu = false } = {}) {
  const harness = workerHarness({ told: true, gpu });
  const { context, run, MiB } = harness;
  const proxies = [];
  const proxy = (name, fields = {}) => {
    // (the fields as they are written: a getter stays one, and makes its proxy when the worker reads it)
    const made = Object.defineProperties({ name, destroyed: 0, destroy() { made.destroyed++; } }, Object.getOwnPropertyDescriptors(fields));
    proxies.push(made);
    return made;
  };
  const readable = (name, value) => proxy(name, { toJs(options) {
    assert.equal(this.destroyed, 0, `${name} was read after it was destroyed`);
    return typeof value === "function" ? value(options) : value;
  } });
  // the generator's face: body is a JavaScript generator function (make): it yields requests (arrays) and is sent the answers
  const played = { made: [], listed: [], steps: [] };
  let body;
  const conductStandIn = (listed, make) => {
    const running = body(make, listed.value);
    const steps = proxy("the generator", { returned: 0, afterReturn: 0, answers: [],
      next(answer) {
        assert.equal(steps.destroyed, 0, "next() of a generator that was destroyed");
        if (steps.returned) steps.afterReturn++;
        steps.answers.push(answer);
        const step = running.next(answer);
        return step.done ? step : { done: false, value: readable(`the request ${step.value.filter((part) => typeof part !== "object").join(" ")}`, (options) => {
          assert.equal(JSON.stringify(options), "{\"depth\":1}", "a request is read one level deep: the conversion of the last one stays a proxy");
          return step.value;
        }) };
      },
      return() {
        assert.equal(steps.destroyed, 0, "return() of a generator that was destroyed");
        steps.returned++;
        return running.return();
      } });
    played.steps.push(steps);
    played.made.push(make);
    return steps;
  };
  const buffers = [];
  const kept = { open: 0, opened: 0, refused: 0, dropped: 0, finished: 0, keeps: [] };
  context.stand = {
    // (the converter's window: asked for the kernels' readers alone)
    converter: { kernel_readers: (kernels) => proxy(`the readers of ${kernels}`) },
    numpy: { Llama: { callKwargs: () => ({}) }, OUTLIER_CHANNELS: 8, KV_START: 256, kernel_quantizer: (kernels) => proxy(`the quantizer of ${kernels}`) },
    pyodide: {
      toPy: (value) => { const made = proxy("the model as it is listed", { value }); played.listed.push(value); return made; },
      pyimport: (name) => {
        assert.equal(name, "convert.conduct");
        return proxy("the module", { get conduct() { return proxy("the function", { callKwargs: conductStandIn }); } });
      },
      // (no kernels of forward.js: the checkpoint's place is a Python bytearray)
      globals: { get: () => (size) => { const buffer = proxy("the checkpoint's buffer", { getBuffer: () => ({ data: new Uint8Array(size), release() {} }) }); buffers.push(buffer); return buffer; } },
    },
    forward: { aloneHolds: () => false },
    // the kept models: a file opened to keep a conversion in as it comes has the one handle its file can have (OPFS)
    kept: {
      openKept: async () => null, replaced: async () => [], forget: async () => {},
      keep: async (model, manifest) => { kept.keeps.push(manifest); },
      keeper: async () => {
        if (kept.open) {
          kept.refused++;
          throw new Error("Access Handles cannot be created if there is another open Access Handle");
        }
        kept.open++;
        kept.opened++;
        return { write() {}, finish: async () => { kept.open--; kept.finished++; },
          drop: async () => {
            kept.open--;
            kept.dropped++;
            if (kept.dropFails) throw new Error("the file system let go of nothing");
          } };
      },
    },
  };
  run("state.llama2_convert = stand.converter; state.keptModule = stand.kept; state.pyodide = stand.pyodide; state.llama2_numpy = stand.numpy; " +
    "state.forwardModule = stand.forward; state.jsKernels = undefined; state.kernels = undefined; state.disabled = [];");
  // what a conversion does as it is made: the place of the weights opened through the sink, as Writer opens it
  const opened = (make) => make.sink.open(1000, readable("the header", HEADER), "int8", readable("the form", FORM));
  // the conversion a conduct ends with, as the worker gets it: a proxy
  const conversion = () => proxy("the conversion", {
    get options() { return readable("the options", { dtype: "int8", bos: 1 }); },
    get tokenizer() { return proxy("the tokenizer", { getBuffer: () => ({ data: new Uint8Array(4), release() {} }) }); } });
  const convert = (hf, play, { signal = new AbortController().signal, conversion: asked } = {}) => {
    body = play;
    proxies.length = 0;
    return harness.failure(context.convert({ id: "made-up", name: "Made up", hf, ...(asked && { conversion: asked }) }, signal, 1));
  };
  // every proxy made since convert() began was destroyed exactly once
  const allDestroyed = () => {
    for (const made of proxies) assert.equal(made.destroyed, 1, `${made.name} was destroyed ${made.destroyed} times`);
    return proxies.length;
  };
  return { ...harness, MiB, proxies, proxy, played, buffers, kept, opened, conversion, convert, allDestroyed };
}
const hf = (more = {}) => ({ repo: "owner/model", revision: REVISION, weights: "model.safetensors", tokenizer: "tokenizer.json", ...more });
const url = (repository, revision, name) => `https://huggingface.co/${repository}/resolve/${revision}/${name}`;
const notFound = () => new Response("", { status: 404, headers: { "X-Error-Code": "EntryNotFound" } });
const equalBytes = (a, b, what) => assert.ok(a.length === b.length && Buffer.from(a.buffer, a.byteOffset, a.length).equals(Buffer.from(b.buffer, b.byteOffset, b.length)), what);

// ---- every kind of request is answered with what it asks for, and the end is the conversion
{
  const w = world();
  const { MiB, bytesOf, body, requests, messages, fresh } = w;
  const SIZE = 30 * MiB, BASE = 1000, SECOND = 5 * MiB + 3;
  const files = {
    [url("owner/model", REVISION, "config.json")]: () => new Response("{\"a\":\"config\"}"),
    [url("vocabulary/owner", OTHER, "tokenizer.model")]: () => new Response(bytesOf(7, 77)),
    [url("owner/model", REVISION, "model.safetensors")]: (init) => {
      if (init.method === "HEAD") return new Response(null, { headers: { "Content-Length": String(SIZE) } });
      const [, from, to] = /bytes=(\d+)-(\d+)/.exec(init.headers.Range);
      const end = Math.min(SIZE, Number(to) + 1);
      // (the first range of all does not show its size: the conduct then asks for it)
      return new Response(body(Number(from), end, { delay: Number(from) >= BASE ? 300 : 0 }), { status: 206, headers: Number(to) === 99 ? {} : { "Content-Range": `bytes ${from}-${end - 1}/${SIZE}` } });
    },
    [url("owner/model", REVISION, "second.safetensors")]: (init) => {
      const [, from, to] = /bytes=(\d+)-(\d+)/.exec(init.headers.Range);
      // (a MiB every 300 ms of the worker's clock: the page is told as a stream comes, four times a second at most)
      return new Response(body(Number(from), Number(to) + 1, { delay: 300 }), { status: 206, headers: { "Content-Range": `bytes ${from}-${to}/${SECOND}` } });
    },
  };
  fresh((address, init) => files[address]?.(init) ?? notFound());
  const got = {}, fed = [];
  const failed = await w.convert(hf({ config: undefined, vocabulary: { repo: "vocabulary/owner", revision: OTHER, tokenizer: "tokenizer.model" } }), function* (make) {
    got.config = yield ["text", "weights", "config.json"];
    got.optional = yield ["text", "vocabulary", "tokenizer_config.json"];
    got.tokenizer = yield ["bytes", "vocabulary", "tokenizer.model"];
    got.noBytes = yield ["bytes", "weights", "tokenizer.json"];
    got.unsaid = yield ["range", "weights", "model.safetensors", 0, 100];
    got.size = yield ["size", "weights", "model.safetensors"];
    got.said = yield ["range", "weights", "model.safetensors", 10, 20];
    got.noRange = yield ["range", "weights", "model.safetensors.index.json", 0, 100];
    w.opened(make);
    for (const [name, begin, end, before] of [["model.safetensors", BASE, SIZE, 0], ["second.safetensors", 3, SECOND, SIZE - BASE]]) {
      let at = begin;
      for (let part = yield ["stream", "weights", name, begin, end, before, SIZE - BASE + SECOND - 3]; part !== undefined; ) {
        fed.push([name, at, part.slice()]);
        at += part.length;
        part = yield ["more", (before + at - begin) / (SIZE - BASE + SECOND - 3)];
      }
      got[name] = at;
    }
    got.empty = yield ["stream", "weights", "second.safetensors", 7, 7, 0, 0];
    got.conversion = w.conversion();
    yield ["done", got.conversion];
    assert.fail("the conduct was asked on after its end");
  });
  assert.equal(failed, undefined, `the conversion failed: ${failed?.error?.stack}`);
  assert.equal(got.config, "{\"a\":\"config\"}", "a text is the file's text");
  assert.equal(got.optional, undefined, "a text that is not there is undefined (None in Python; null is not)");
  equalBytes(got.tokenizer, bytesOf(7, 77), "bytes are the file's bytes, of the vocabulary's repository where the request says so");
  assert.ok(ArrayBuffer.isView(got.tokenizer) && got.tokenizer.BYTES_PER_ELEMENT === 1, "bytes are a Uint8Array (a buffer for Python)");
  assert.equal(got.noBytes, undefined, "bytes that are not there are undefined");
  assert.equal(got.unsaid.length, 2);
  equalBytes(got.unsaid[0], bytesOf(0, 100), "a range is the bytes asked for");
  assert.equal(got.unsaid[1], undefined, "a size the answer did not show is undefined, not NaN");
  assert.equal(got.size, SIZE, "the size is the file's");
  equalBytes(got.said[0], bytesOf(10, 20), "a range is the bytes asked for, from where it begins");
  assert.equal(got.said[1], SIZE, "a range's answer says the size of the whole file where the server showed it");
  assert.equal(got.noRange, undefined, "a range of a file that is not there is undefined");
  // the streams: every byte once, in the order of the file, each stream ended with undefined
  assert.equal(got["model.safetensors"], SIZE, "the first stream did not end at the end asked for");
  assert.equal(got["second.safetensors"], SECOND, "the second stream did not end at the end asked for");
  assert.equal(got.empty, undefined, "a stream of nothing is answered with undefined at once");
  for (const [name, at, part] of fed) equalBytes(part, bytesOf(at, at + part.length), `a part of ${name} at ${at} is not the file's bytes there`);
  assert.ok(fed.filter(([name]) => name === "model.safetensors").length >= 3, "the stream came in one piece");
  // what was asked of the network, and where
  const asked = requests.map((r) => `${r.method} ${r.url.replace("https://huggingface.co/", "")}${r.range ? ` ${r.range}` : ""}`);
  assert.deepEqual(asked.slice(0, 8), [
    `GET owner/model/resolve/${REVISION}/config.json`,
    `GET vocabulary/owner/resolve/${OTHER}/tokenizer_config.json`,
    `GET vocabulary/owner/resolve/${OTHER}/tokenizer.model`,
    `GET owner/model/resolve/${REVISION}/tokenizer.json`,
    `GET owner/model/resolve/${REVISION}/model.safetensors bytes=0-99`,
    `HEAD owner/model/resolve/${REVISION}/model.safetensors`,
    `GET owner/model/resolve/${REVISION}/model.safetensors bytes=10-19`,
    `GET owner/model/resolve/${REVISION}/model.safetensors.index.json bytes=0-99`,
  ]);
  assert.equal(asked[8], `GET owner/model/resolve/${REVISION}/model.safetensors bytes=${BASE}-${BASE + 8 * MiB - 1}`, "the stream begins where the request says");
  assert.ok(asked.every((line) => !line.includes("second.safetensors") || /bytes=\d+-\d+$/.test(line)));
  assert.equal(asked.filter((line) => line.includes("second.safetensors")).at(-1).split("-").at(-1), String(SECOND - 1), "the second stream was asked for past its end, or not to it");
  // the progress: of the total the request said, what was in before this stream counted, and the share the conduct said
  const progress = messages.filter((m) => m.type === "progress");
  const TOTAL = SIZE - BASE + SECOND - 3;
  assert.ok(progress.length >= 2, "no progress was told");
  assert.ok(progress.every((m) => m.total === TOTAL || m.total === 0), `the progress is of another total than the stream's: ${JSON.stringify(progress[0])}`);
  const whole = progress.filter((m) => m.total === TOTAL);
  assert.ok(whole.every((m, i) => i === 0 || m.received >= whole[i - 1].received), "what arrived went back between two streams");
  assert.equal(whole.at(-1).received, TOTAL, "the last progress of the streams is not all of them: before counts what the earlier streams brought");
  assert.equal(whole.at(-1).converted, 1, "the share converted is the conduct's");
  assert.ok(whole.some((m) => m.received > SIZE - BASE && m.received < TOTAL), "nothing was told while the second stream came");
  // the model as it is listed: no key of undefined; and what the converter takes besides the files
  // (compared as text: the worker's objects are of its own context)
  assert.equal(JSON.stringify(w.played.listed), JSON.stringify([{ repo: "owner/model", revision: REVISION, weights: "model.safetensors", tokenizer: "tokenizer.json",
    vocabulary: { repo: "vocabulary/owner", revision: OTHER, tokenizer: "tokenizer.model" } }]));
  assert.ok(!Object.hasOwn(w.played.listed[0], "config"), "a key whose value is undefined was handed to Python");
  assert.deepEqual(Object.keys(w.played.made[0]).sort(), ["dtype", "quantize_rows", "readers", "sink"]);
  assert.equal(typeof w.played.made[0].dtype, "function", "no bits asked for: the worker's choice");
  // the end: the generator closed and let go, the conversion let go, every proxy once
  const [steps] = w.played.steps;
  assert.equal(steps.returned, 1, "the generator was not closed after its end");
  assert.equal(steps.afterReturn, 0);
  assert.equal(steps.answers[0], undefined, "the first next() sends nothing");
  assert.ok(w.allDestroyed() >= 20, "fewer proxies than the requests alone make");
  assert.equal(got.conversion.destroyed, 1);
  assert.equal(w.kept.keeps.length, 1, "the conversion was not kept");
  assert.equal(w.run("state.llama !== undefined"), true, "no engine was made of the conversion");
  ok("every kind of request is answered with what it asks for: text, bytes, range, size, streams and their end; a file that is not there with undefined");
  ok(`the end of a conduct: the conversion kept and made an engine of, the generator closed, ${w.proxies.length} proxies destroyed once each`);

  // ---- the dtype asked for and the kernels' quantizer and readers go to the conduct, and are let go
  w.run("state.kernels = 'simdkernel.so'; state.llama = undefined");
  fresh(() => notFound());
  const refused = await w.convert(hf(), function* () {
    yield ["text", "weights", "config.json"];
    yield ["missing", "weights", "config.json"];
  }, { conversion: { dtype: "int6", max_seq_len: 512 } });
  assert.match(refused.error.message, /^owner\/model has no config\.json at /);
  const make = w.played.made.at(-1);
  assert.equal(make.dtype, "int6");
  assert.equal(make.max_seq_len, 512);
  assert.equal(make.quantize_rows.name, "the quantizer of simdkernel.so");
  assert.equal(make.readers.name, "the readers of simdkernel.so");
  w.allDestroyed();
  assert.equal(make.quantize_rows.destroyed, 1);
  w.run("state.kernels = undefined");
  ok("the bits asked for, the context and the kernels' quantizer and readers go to the conduct; a conversion that ends early lets go of them");

  // ---- "missing": the 404 of that very file, in refused()'s words; the place decides the repository
  const vocabulary = { repo: "vocabulary/owner", revision: OTHER, tokenizer: "tokenizer.model" };
  for (const [where, repository, revision] of [["weights", "owner/model", REVISION], ["vocabulary", "vocabulary/owner", OTHER]]) {
    fresh(() => notFound());
    const lost = await w.convert(hf({ vocabulary }), function* () {
      assert.equal(yield ["text", "weights", "other.json"], undefined);
      assert.equal(yield ["text", "vocabulary", "other.json"], undefined);
      assert.equal(yield ["bytes", where, "tokenizer.model"], undefined);
      assert.equal(yield ["range", where, "tokenizer.json", 0, 10], undefined);
      yield ["missing", where, "tokenizer.model"];
      assert.fail("the conduct was asked on after a file was missing");
    });
    assert.equal(lost.error.message, `${repository} has no tokenizer.model at ${revision} on huggingface.co, or has no commit ${revision}.`);
    assert.equal(lost.error.status, 404);
    assert.equal(w.played.steps.at(-1).returned, 1);
    w.allDestroyed();
  }
  ok("a conduct that ends on a missing file: the 404 of that file of that repository is thrown, in the words the worker had for it");

  // ---- a failure that is not a 404 is never an answer: the loop ends, the generator is closed
  for (const [kind, request, answer, words] of [
    ["text", ["text", "weights", "tokenizer_config.json"], () => new Response("", { status: 503 }), /^huggingface\.co answered 503 for tokenizer_config\.json of owner\/model\.$/],
    ["text (gated)", ["text", "weights", "config.json"], () => new Response("", { status: 401, headers: { "X-Error-Code": "GatedRepo" } }), /^owner\/model is gated on huggingface\.co/],
    ["bytes", ["bytes", "weights", "tokenizer.json"], () => new Response("", { status: 429 }), /asks this address to make fewer requests/],
    ["range", ["range", "weights", "model.safetensors", 0, 100], () => new Response("", { status: 403 }), /^huggingface\.co answered 403 for model\.safetensors of owner\/model\.$/],
    ["text (the line)", ["text", "weights", "config.json"], () => Promise.reject(new TypeError("Failed to fetch")), /^Failed to fetch$/],
    ["size", ["size", "weights", "model.safetensors"], () => new Response(null, { status: 404 }), /^Could not learn the size of /],
  ]) {
    fresh(answer);
    let resumed = false;
    const failed = await w.convert(hf(), function* (make) {
      w.opened(make);  // (a place for the weights is open, as after a first candidate)
      yield request;
      resumed = true;
    });
    assert.match(failed.error.message, words, kind);
    assert.equal(resumed, false, `${kind}: a failure was answered to the conduct`);
    const steps = w.played.steps.at(-1);
    assert.equal(steps.returned, 1, `${kind}: the generator was not closed`);
    assert.equal(steps.answers.length, 1, `${kind}: the generator was sent something after the failure`);
    w.allDestroyed();
    assert.equal(w.buffers.at(-1).destroyed, 1, `${kind}: the place of the weights was left behind`);
  }
  ok("a failure that is not a 404 (5xx, 401, 403, 429, the line, a size that cannot be learnt) is not answered: the generator is closed, nothing is left open");

  // ---- a part of a stream that fails for good, and a part cut short
  {
    fresh((address, init) => (init.headers?.Range === `bytes=${8 * MiB}-${16 * MiB - 1}` ? new Response("", { status: 403 })
      : new Response(body(...(/bytes=(\d+)-(\d+)/.exec(init.headers.Range).slice(1).map(Number).map((n, i) => n + i))), { status: 206 })));
    const parts = [];
    const failed = await w.convert(hf(), function* (make) {
      w.opened(make);
      for (let part = yield ["stream", "weights", "model.safetensors", 0, 20 * MiB, 0, 20 * MiB]; part !== undefined; part = yield ["more", 0.5]) parts.push(part.length);
      assert.fail("a stream with a hole in it was ended as a whole one");
    });
    assert.match(failed.error.message, /answered 403 for model\.safetensors/);
    assert.ok(parts.length <= 1, "parts after the one that failed went to the conduct");
    assert.equal(w.played.steps.at(-1).returned, 1);
    w.allDestroyed();
    assert.equal(w.buffers.at(-1).destroyed, 1);
    ok("a part of a stream that fails for good ends the loop: no later part goes to the conduct, the generator is closed");
  }

  // ---- the conduct's own failure (a refusal of the converter's, in Python) is thrown as it is, and the rest let go
  {
    fresh(() => new Response("{}"));
    const failed = await w.convert(hf(), function* (make) {
      w.opened(make);
      yield ["text", "weights", "config.json"];
      throw Object.assign(new Error("Traceback\nValueError: This model cannot be converted."), { type: "ValueError" });
    });
    assert.equal(failed.error.type, "ValueError");
    assert.equal(w.played.steps.at(-1).returned, 1);
    w.allDestroyed();
    assert.equal(w.buffers.at(-1).destroyed, 1);
    // a request nothing answers, a conduct that ends without a word, another request in the middle of a stream
    for (const [play, words] of [
      [function* () { yield ["folder", "weights", "x"]; }, /asked for folder/],
      [function* () { yield ["text", "weights", "config.json"]; }, /ended without a word/],
      [function* () { yield ["stream", "weights", "model.safetensors", 0, 10, 0, 10]; yield ["text", "weights", "config.json"]; }, /in the middle of a file/],
    ]) {
      fresh((address, init) => (init.headers?.Range ? new Response(body(0, 10), { status: 206 }) : new Response("{}")));
      const wrong = await w.convert(hf(), play);
      assert.match(wrong.error.message, words);
      assert.equal(w.played.steps.at(-1).returned, 1);
      w.allDestroyed();
    }
    ok("a failure of the conduct's own is thrown as it came, and a conduct that asks what nothing answers is refused; the generator is closed all the same");
  }

  // ---- a cancelled load: in the middle of a stream, while an answer waits, and as an answer comes
  {
    const cancel = new AbortController();
    fresh((address, init) => new Response(body(...(/bytes=(\d+)-(\d+)/.exec(init.headers.Range).slice(1).map(Number).map((n, i) => n + i)), { signal: init.signal }), { status: 206 }));
    let parts = 0;
    const failed = await w.convert(hf(), function* (make) {
      w.opened(make);
      for (let part = yield ["stream", "weights", "model.safetensors", 0, 40 * MiB, 0, 40 * MiB]; part !== undefined; part = yield ["more", 0.1]) {
        if (++parts === 2) cancel.abort();
      }
      assert.fail("a cancelled stream was ended as a whole one");
    }, { signal: cancel.signal });
    assert.equal(failed.error.name, "AbortError");
    assert.equal(parts, 2, "a part went to the conduct after the load was cancelled");
    const steps = w.played.steps.at(-1);
    assert.equal(steps.returned, 1, "the generator of a cancelled load was not closed");
    assert.equal(steps.afterReturn, 0);
    w.allDestroyed();
    assert.equal(w.buffers.at(-1).destroyed, 1, "a cancelled load left the place of its weights behind");
    const asked = requests.length;
    await w.sleep(1000);
    assert.equal(requests.length, asked, "a cancelled load went on asking");
    assert.equal(steps.afterReturn, 0, "a part went to the generator after it was closed");
  }
  for (const [name, route, when] of [
    ["while an answer waits", () => "hang", (cancel) => setImmediate(() => cancel.abort())],
    ["as an answer comes", (cancel) => () => { const answer = new Response("{}"); cancel.abort(); return answer; }, () => {}],
  ]) {
    const cancel = new AbortController();
    fresh(typeof route() === "string" ? route : route(cancel));
    let resumed = false;
    const play = w.convert(hf(), function* (make) {
      w.opened(make);
      yield ["text", "weights", "config.json"];
      resumed = true;
    }, { signal: cancel.signal });
    when(cancel);
    const failed = await play;
    assert.equal(failed.error.name, "AbortError", name);
    assert.equal(resumed, false, `${name}: a cancelled load was answered`);
    assert.equal(w.played.steps.at(-1).returned, 1, name);
    w.allDestroyed();
    assert.equal(w.buffers.at(-1).destroyed, 1, name);
  }
  ok("a cancelled load (in the middle of a stream, while an answer waits, as an answer comes) sends the conduct nothing more, closes it and leaves nothing open");
}

// ---- T403: the file opened to keep a conversion in as it comes (a model that may go on the GPU alone) is let go
// whatever ended the conversion: its handle is the only one its file can have, and the next conversion of the model
// got none (and was not kept) after one that ended before its weights came
{
  const w = world({ gpu: true });
  w.context.crossOriginIsolated = true;
  w.run("state.sharedKernels = {}; state.wideKernels = { plain: {}, shared: {} }; state.gpuAdapter = { fallback: false };");
  assert.equal(w.run("gpuOnlyPossible('int8')"), true, "the check's device may not put a model on the GPU alone: nothing is opened to keep it in");
  const endings = [
    ["a config.json that is not there", () => notFound(), function* () { yield ["text", "weights", "config.json"]; yield ["missing", "weights", "config.json"]; }, /has no config\.json/],
    ["a head the server refuses", () => new Response("", { status: 503 }), function* () { yield ["range", "weights", "model.safetensors", 0, 100]; }, /answered 503/],
    ["a tokenizer the converter refuses", () => new Response("x"), function* () { yield ["bytes", "weights", "tokenizer.json"]; throw new Error("This model cannot be converted."); }, /cannot be converted/],
  ];
  for (const [name, route, play, words] of endings) {
    w.fresh(route);
    const before = { ...w.kept };
    const failed = await w.convert(hf(), play);
    assert.match(failed.error.message, words, name);
    assert.equal(w.kept.opened, before.opened + 1, `${name}: no file was opened to keep the conversion in (after an earlier one that ended early?)`);
    assert.equal(w.kept.open, 0, `${name}: the file opened to keep the conversion in was left open`);
    assert.equal(w.kept.dropped, before.dropped + 1, name);
  }
  // a load cancelled while its first file comes
  const cancel = new AbortController();
  w.fresh(() => "hang");
  const play = w.convert(hf(), function* () { yield ["text", "weights", "config.json"]; }, { signal: cancel.signal });
  setImmediate(() => cancel.abort());
  assert.equal((await play).error.name, "AbortError");
  assert.equal(w.kept.open, 0, "a load cancelled before its weights came left the file to keep it in open");
  assert.equal(w.kept.refused, 0, "a conversion found the file of an earlier one still open");
  assert.equal(w.kept.opened, endings.length + 1);
  // and where letting go of that file fails, the rest is let go all the same
  w.kept.dropFails = true;
  w.fresh(() => new Response("", { status: 503 }));
  const failed = await w.convert(hf(), function* (make) { w.opened(make); yield ["text", "weights", "config.json"]; });
  assert.match(failed.error.message, /let go of nothing/);
  assert.equal(w.played.steps.at(-1).returned, 1, "the generator was left open where the kept file could not be let go");
  w.allDestroyed();
  assert.equal(w.buffers.at(-1).destroyed, 1, "the place of the weights was left behind where the kept file could not be let go");
  ok("T403: a conversion that ends before its weights come (a file that is not there, a refusal, a cancelled load) lets go of the file opened to keep it in: the next one opens it");
}

console.log(`worker-conduct-check: ${passed} checks passed`);
// (T384: not at once, as tests/worker-fetches-check.mjs)
setTimeout(() => process.stdout.write("", () => process.exit(0)), 200);
