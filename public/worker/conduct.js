// worker/conduct.js (T374.2.1): the worker's side of a conversion's conduct. Which file a model needs, how much of its
// head, which tokenizer is tried next and which parts of the weights follow is Python's (public/convert/conduct.py: a
// generator that asks). Here is what answers it from huggingface.co: the fetches, their parts and their order, what is
// tried again, what a cancelled load stops. Nothing of a model's layout is known here, and nothing of a line there.
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

const { state } = await import(new URL(`state.js${new URL(import.meta.url).search}`, import.meta.url));
const { fileSize, refused, fetchRange, inOrder } =
  await import(new URL(`ranges.js${new URL(import.meta.url).search}`, import.meta.url));

// The conduct of the conversion of a model as huggingface.co lists it: the generator, not yet asked anything.
// hf: the item's (weights, config, tokenizer, vocabulary); make: what the converter takes besides the files.
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
// proxy (depth 1), which whoever asked for it destroys.
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

// Answers the requests of steps (a conduct: next(answer) sends, and gives { done, value }) from huggingface.co until
// it ends, and returns the conversion it ends with, finished: a proxy for the caller to destroy. hf: where the places
// are ({ repo, revision }, and the same under vocabulary where the model names another repository for it, T136).
//
// A file that is not there (a 404) is an answer: undefined, which is None in Python (null is not). The conduct goes on
// to another file, or ends with ("missing", where, name), and the 404 of that file is what is thrown here: the words
// are refused()'s. Every other failure (the line, a refusal of the server, a cancelled load, a part cut short) is
// thrown from here as it came and never answered: whoever called closes the generator (return()) and lets go of it,
// as it does after the end.
//
// progress: of(total) once the size of what is streamed is known, arriving(bytes in so far), converting(feed): feed()
// hands a part to the conversion and returns the share converted (the time it takes is the converter's, T84).
export async function answered(steps, hf, signal, progress) {
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
  const answers = {
    text: (url) => orNot(url, async () => (await file(url)).text()),
    bytes: (url) => orNot(url, async () => new Uint8Array(await (await file(url)).arrayBuffer())),
    // [the bytes, the size of the whole file]; undefined for a size the answer did not show (Content-Range is not a
    // header CORS shows by default, T112): the conduct then asks for it
    range: (url, begin, end) => orNot(url, async () => {
      const { bytes, total } = await fetchRange(url, begin, end, signal);
      return [bytes, Number.isFinite(total) && total > 0 ? total : undefined];
    }),
    size: (url) => fileSize(url, signal),
    // the parts go to the conduct as inOrder() has them in the order of the file, each answered by ("more", share);
    // what is answered here is the end of the stream. Of total bytes, before were in when this stream began (the
    // shards count one after another: the review of T119)
    stream: async (url, begin, end, before, total) => {
      progress.of(total);
      await inOrder(url, begin, end, (part) => progress.converting(() => {
        const [kind, share] = requestOf(steps.next(part));
        if (kind !== "more") {
          throw new Error(`The conduct of the conversion asked for ${kind} in the middle of a file.`);
        }
        return share;
      }), signal, (received) => progress.arriving(before + received - begin));
      return undefined;
    },
  };
  for (let request = requestOf(steps.next()); ;) {
    const [kind, where, name, ...rest] = request;
    if (kind === "done") {
      return where;  // (the conversion)
    }
    if (kind === "missing") {
      throw notThere.get(at(where, name)) ?? new Error(`The conversion needs ${name}, which is not there.`);
    }
    if (!Object.hasOwn(answers, kind)) {
      throw new Error(`The conduct of the conversion asked for ${kind}, which nothing here answers.`);
    }
    const answer = await answers[kind](at(where, name), ...rest);
    // (a load cancelled while its answer came converts nothing more)
    signal.throwIfAborted();
    request = requestOf(steps.next(answer));
  }
}
