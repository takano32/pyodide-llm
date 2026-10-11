// conducting.mjs (T374.4): what a Node tool that converts a model in Pyodide answers the conduct of a conversion with
// (src/python/convert/conduct.py: a generator that asks which file comes next; the page's worker answers the same one,
// public/worker/conduct.js). A tool says where the files are and what the conversion is made with, and nothing of
// which files a model needs, which tokenizer is tried or where its template comes from:
//
//   const { hf, folder } = listed(path);
//   const conversion = converted(py, hf, { dtype, sink, quantize_rows, readers }, fromFolder(folder));
//
//   listed      the model a path stands for, as tests/hf_fetch.py lays its files out (tests/conducting.py's listed())
//   fromFolder  an answerer of the shape the worker's have, { answers: { text, bytes, range, size, stream }, missing },
//               from the files of a folder: a file that is not there is answered with undefined (None in Python; null
//               is not), a stream's parts go to the conversion's own feed, which the request brings
//   answered    the loop: a conduct's requests answered until it ends; every proxy on the way let go
//   converted   the conduct of a model made in a Pyodide and answered: the conversion, finished (a proxy)
//
// The tools: tests/profile-convert.mjs, tests/page-27b.mjs. The Python tools have the same in tests/conducting.py.
// Tested alone, with a stand-in for the generator and no Pyodide, by tests/conducting-check.mjs.
import fs from "node:fs";
import path from "node:path";

/** { hf, folder } of a path: a .gguf file that says everything itself (T74); a folder with the original's config.json
 * and tokenizer and a GGUF beside them (T136's second stage: the first .gguf by its name); or a folder with
 * model.safetensors, or with the shards its index names. No tokenizer is named: the conduct's candidates, in its order. */
export function listed(target) {
  if (target.endsWith(".gguf")) return { hf: { weights: path.basename(target) }, folder: path.dirname(target) };
  const gguf = fs.readdirSync(target).filter((name) => name.endsWith(".gguf")).sort()[0];
  // (any vocabulary that is not empty: the conduct then asks "vocabulary" for all but the weights, and the one folder
  // is both places)
  return { hf: gguf ? { weights: gguf, vocabulary: { folder: path.basename(target) } } : { weights: "model.safetensors" }, folder: target };
}

/** What answers a conduct from the files of a folder (every place is that folder).
 * part: the bytes of a stream that go to the conversion at a time (the worker's first parts are 8 MiB).
 * starting(total): once, when the first stream is asked for (of total bytes of weights: the head, the tokenizer and
 * the template are read by then, and so is what of the tensors came with a GGUF's head).
 * reading(ms): after each part is read from the disk, the time that took. close(): the files opened, closed. */
export function fromFolder(folder, { part = 8 << 20, starting = () => {}, reading = () => {} } = {}) {
  const opened = new Map();  // a file is opened once: a part costs one read
  const open = (name) => {
    if (!opened.has(name)) {
      const fd = fs.openSync(`${folder}/${name}`, "r");
      opened.set(name, { fd, size: fs.fstatSync(fd).size });
    }
    return opened.get(name);
  };
  const range = (name, begin, end) => {
    const { fd, size } = open(name), bytes = new Uint8Array(Math.max(Math.min(end, size) - begin, 0));
    fs.readSync(fd, bytes, 0, bytes.length, begin);
    return bytes;
  };
  const there = (name, read) => (fs.existsSync(`${folder}/${name}`) ? read() : undefined);
  let started = false;
  return {
    answers: {
      text: (where, name) => there(name, () => fs.readFileSync(`${folder}/${name}`, "utf8")),
      bytes: (where, name) => there(name, () => new Uint8Array(fs.readFileSync(`${folder}/${name}`))),
      // (the bytes [begin, end), fewer where the file ends first, and the size of the file)
      range: (where, name, begin, end) => there(name, () => [range(name, begin, end), open(name).size]),
      size: (where, name) => open(name).size,
      stream(where, name, begin, end, before, total, feed) {
        if (!started) {
          started = true;
          starting(total);
        }
        for (let at = begin; at < end; at += part) {
          const began = performance.now(), bytes = range(name, at, Math.min(at + part, end));
          reading(performance.now() - began);
          if (bytes.length !== Math.min(part, end - at)) throw new Error(`${name} ends at ${at + bytes.length}, before the ${end} bytes its head says it has`);
          feed(bytes);
        }
      },
    },
    missing: (where, name) => new Error(`${folder} has no ${name}`),
    close() {
      for (const { fd } of opened.values()) fs.closeSync(fd);
      opened.clear();
    },
  };
}

/** Answers the requests of steps (a conduct as Pyodide gives a Python generator: next(answer) gives { done, value },
 * each value a proxy of a tuple) from an answerer until it ends, and returns the conversion it ends with, finished: a
 * proxy for the caller. A proxy a request brings (the feed of a stream) is let go here once the request is answered or
 * failed, as the worker's loop does. */
export function answered(steps, { answers, missing }) {
  const requestOf = (step) => {
    if (step.done) throw new Error("The conduct of the conversion ended without a word.");
    try {
      return step.value.toJs({ depth: 1 });
    } finally {
      step.value.destroy();
    }
  };
  for (let request = requestOf(steps.next()); ;) {
    const [kind, ...asked] = request;
    if (kind === "done") return asked[0];
    let answer;
    try {
      if (kind === "missing") throw missing(...asked);
      if (!Object.hasOwn(answers, kind)) throw new Error(`The conduct of the conversion asked for ${kind}, which nothing here answers.`);
      answer = answers[kind](...asked);
    } finally {
      for (const brought of asked) if (typeof brought?.destroy === "function") brought.destroy();
    }
    request = requestOf(steps.next(answer));
  }
}

/** The conversion of a model by its conduct, in the Pyodide py. hf: the model as src/python/convert/conduct.py takes it
 * (names only); make: what the converter takes besides the files (dtype, max_seq_len, sink, quantize_rows, readers). */
export function converted(py, hf, make, answerer) {
  const module = py.pyimport("convert.conduct");
  // (JSON: a key whose value is undefined is left out. In Python it would be there, with a value that is not None)
  const model = py.toPy(JSON.parse(JSON.stringify(hf)));
  const steps = module.conduct.callKwargs(model, make);
  try {
    return answered(steps, answerer);
  } finally {
    try { steps.return(); } catch { /* (ended already) */ }
    steps.destroy();
    model.destroy();
    module.destroy();
    answerer.close?.();
  }
}
