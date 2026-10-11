// worker/conduct.js (T374.2.1): the worker's side of a conversion's conduct. Which file a model needs, how much of its
// head, which tokenizer is tried next and which parts of the weights follow is Python's (public/convert/conduct.py: a
// generator that asks). Here is what answers it (T374.2.2): the loop, which knows nothing of where a file is, and the
// two answerers it is handed one of: huggingface.co's (the fetches, their parts and their order, what is tried again)
// and a folder's of the visitor's disk (its Files, read as the disk gives them). Nothing of a model's layout is known
// here, and nothing of a line or a disk there.
//
// An answerer: { answers: { text, bytes, range, size, stream }, missing }. Each answer takes what its request has
// after the kind (where, name, ...) and gives what public/convert/conduct.py says the request is answered with; a file
// that is not there is answered with undefined, which is None in Python (null is not). Every other failure is thrown
// and never answered. missing(where, name): the error of a conduct that ended for want of that file, in the words of
// whoever knows the place (or nothing: the loop has a sentence of its own, which neither answerer here leaves it to).
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

const { state } = await import(new URL(`state.js${new URL(import.meta.url).search}`, import.meta.url));
const { fileSize, refused, fetchRange, inOrder } =
  await import(new URL(`ranges.js${new URL(import.meta.url).search}`, import.meta.url));
const { weightsRoom } = await import(new URL(`weights.js${new URL(import.meta.url).search}`, import.meta.url));

// The conduct of the conversion of a model as it is listed: the generator, not yet asked anything.
// hf: the item's (weights, config, tokenizer, vocabulary: names, wherever the files are); make: what the converter
// takes besides the files.
// Every proxy on the way is let go: each keeps its Python object alive.
export function conductOf(hf, make) {
  const module = state.pyodide.pyimport("convert.conduct"), conduct = module.conduct;
  // (JSON: a key whose value is undefined is left out. In Python it would be there, with a value that is not None)
  const listed = state.pyodide.toPy(JSON.parse(JSON.stringify(hf)));
  try {
    return conduct.callKwargs(listed, make);
  } finally {
    listed.destroy();
    conduct.destroy();
    module.destroy();
  }
}

// What next() of the generator gave, as an array, its proxy let go. The conversion of ("done", conversion) stays a
// proxy (depth 1), which whoever asked for it destroys; and so does the feed of a stream, which the loop destroys.
function requestOf(step) {
  if (step.done) {
    throw new Error("The conduct of the conversion ended without a word.");
  }
  try {
    return step.value.toJs({ depth: 1 });
  } finally {
    step.value.destroy();
  }
}

// Answers the requests of steps (a conduct: next(answer) sends, and gives { done, value }) from an answerer until it
// ends, and returns the conversion it ends with, finished: a proxy for the caller to destroy.
//
// A file that is not there is an answer (undefined): the conduct goes on to another file, or ends with ("missing",
// where, name), and what is thrown then is the answerer's to say. Every other failure (the line, a refusal of the
// server, a file of the disk that cannot be read, a cancelled load, a part cut short) is thrown from here as it came
// and never answered: whoever called closes the generator (return()) and lets go of it, as it does after the end.
//
// A proxy a request brings (the feed of a stream: the conversion's own) is the loop's to let go of, once the request is
// answered, or failed, or was of a kind nothing answers (T407): no answerer has to remember it.
export async function answered(steps, { answers, missing }, signal) {
  for (let request = requestOf(steps.next()); ;) {
    const [kind, ...asked] = request;
    if (kind === "done") {
      return asked[0];  // (the conversion)
    }
    let answer;
    try {
      if (kind === "missing") {
        throw missing(...asked) ?? new Error(`The conversion needs ${asked[1]}, which is not there.`);
      }
      if (!Object.hasOwn(answers, kind)) {
        throw new Error(`The conduct of the conversion asked for ${kind}, which nothing here answers.`);
      }
      answer = await answers[kind](...asked);
    } finally {
      for (const brought of asked) {
        if (typeof brought?.destroy === "function") brought.destroy();
      }
    }
    // (a load cancelled while its answer came converts nothing more)
    signal.throwIfAborted();
    request = requestOf(steps.next(answer));
  }
}

// What answers a conduct from huggingface.co. hf: where the places are ({ repo, revision }, and the same under
// vocabulary where the model names another repository for it, T136).
//
// A 404 is the answer "not there", and the 404 of the file a conduct ends on is what missing() gives: the words are
// refused()'s.
//
// progress: of(total) once the size of what is streamed is known, arriving(bytes in so far), converting(feed): feed()
// hands a part to the conversion and returns the share converted (the time it takes is the converter's, T84).
// A stream's parts go from here to the conversion's own feed, which the request brings: the call the worker's own
// steps made for a part, and nothing more.
export function fromHub(hf, signal, progress) {
  const places = { weights: hf, vocabulary: hf.vocabulary };
  const at = (where, name) => `https://huggingface.co/${places[where].repo}/resolve/${places[where].revision}/${name}`;
  const notThere = new Map();  // by URL, the error of each 404 answered with undefined
  const orNot = async (url, fetched) => {
    try {
      return await fetched();
    } catch (error) {
      if (error.status !== 404) {
        throw error;
      }
      notThere.set(url, error);
      return undefined;
    }
  };
  const file = async (url) => {
    const res = await fetch(url, { signal });
    if (!res.ok) {
      throw refused(url, res);
    }
    return res;
  };
  // (each answer by the address of its file)
  const of = (answer) => (where, name, ...more) => answer(at(where, name), ...more);
  return {
    answers: {
      text: of((url) => orNot(url, async () => (await file(url)).text())),
      bytes: of((url) => orNot(url, async () => new Uint8Array(await (await file(url)).arrayBuffer()))),
      // [the bytes, the size of the whole file]; undefined for a size the answer did not show (Content-Range is not a
      // header CORS shows by default, T112): the conduct then asks for it
      range: of((url, begin, end) => orNot(url, async () => {
        const { bytes, total } = await fetchRange(url, begin, end, signal);
        return [bytes, Number.isFinite(total) && total > 0 ? total : undefined];
      })),
      size: of((url) => fileSize(url, signal)),
      // the parts go to the conversion itself (feed) as inOrder() has them in the order of the file: nothing of the
      // conduct runs for a part, which costs what it did. What is answered is the end of the stream. Of total bytes,
      // before were in when this stream began (the shards count one after another: the review of T119)
      stream: of(async (url, begin, end, before, total, feed) => {
        progress.of(total);
        await inOrder(url, begin, end, (part) => progress.converting(() => feed(part)), signal,
          (received) => progress.arriving(before + received - begin));
        return undefined;
      }),
    },
    missing: (where, name) => notThere.get(at(where, name)),
  };
}

// What answers a conduct from a folder of the visitor's disk (T374.2.2). files: the Files chosen, all of them; the
// conduct asks for each by its name, and a name is found whatever its letters' case (as the page finds the three files
// a model needs; of two names that differ in that alone, the first). where is not looked at: a folder is one place.
//
// A name the folder does not have is the answer "not there". A file that is there and cannot be read is a failure,
// thrown as the browser says it. A file a conduct ends for want of is said in the folder's words (T427: the page asks
// for a model's three files before the worker hears of the folder, so it is a file one of those names).
//
// progress: as fromHub()'s, but for arriving(): what a disk gives has all arrived. A part goes to the conversion's
// feed as the disk gives it, one call a part, and waits for the room its weights need where they go on (T156).
export function fromFolder(files, signal, progress) {
  const named = new Map();
  for (const file of files) {
    const name = file.name.toLowerCase();
    if (!named.has(name)) named.set(name, file);
  }
  // (each answer by the File of that name, where there is one)
  const of = (answer) => (where, name, ...more) => {
    const file = named.get(name.toLowerCase());
    return file && answer(file, ...more);
  };
  return {
    answers: {
      text: of((file) => file.text()),
      bytes: of(async (file) => new Uint8Array(await file.arrayBuffer())),
      range: of(async (file, begin, end) => [new Uint8Array(await file.slice(begin, end).arrayBuffer()), file.size]),
      size: of((file) => file.size),
      stream: of(async (file, begin, end, before, total, feed) => {
        progress.of(total);
        const reader = file.slice(begin, end).stream().getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) {
            return undefined;
          }
          if (signal.aborted) {
            reader.cancel();
            signal.throwIfAborted();
          }
          progress.converting(() => feed(value));
          await weightsRoom();  // (T156)
        }
      }),
    },
    missing: (where, name) => new Error(`The folder has no ${name}, which the model needs.`),
  };
}
