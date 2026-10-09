// worker/sources.js (T350): where the bytes of a llama2.c checkpoint come from: the parts of this site's models
// (download(), with the Cache API), a file of the visitor's disk (readFile()) and a URL (readUrl()). Each gives a source
// whose into(write) writes every byte where it belongs.
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

const { weightsRoom } = await import(new URL(`state.js${new URL(import.meta.url).search}`, import.meta.url));
const { since } = await import(new URL(`clock.js${new URL(import.meta.url).search}`, import.meta.url));
const { innerAbort, worthRetrying, inOrder } = await import(new URL(`ranges.js${new URL(import.meta.url).search}`, import.meta.url));
const { told } = await import(new URL(`told.js${new URL(import.meta.url).search}`, import.meta.url));

// Checkpoints are deployed in parts of 8 MiB (see the Makefile). Several parts download at once, which is
// about twice as fast as one stream, and the download runs while Pyodide is still loading: until the Python
// buffer exists the chunks wait in a queue, after that every chunk is written straight into it.
const PART_BYTES = 8 * 1024 * 1024;
const CONNECTIONS = 8;

// GitHub Pages lets the browser keep a file for ten minutes only, so the parts also go into the Cache API: the
// next visit starts without downloading the model again. The size is part of the key, so a rebuilt model of
// another size is fetched anew. Without the Cache API (some private modes) this is a plain fetch.
// v2: llm-jp-3-150m got its whole context of 4096 tokens, which changed its header and not its size
const MODEL_CACHE = "models-v2";

// what an earlier version of this page stored
globalThis.caches?.delete("models-v1").catch(() => {});

// A part's bytes: PART_BYTES, the last one less
const partBytes = (model, part) => Math.min(PART_BYTES, model.bytes - part * PART_BYTES);
const partKey = (url, model) => `${url}?bytes=${model.bytes}`;

async function fetchPart(url, model, part, signal) {
  const cache = await globalThis.caches?.open(MODEL_CACHE).catch(() => undefined);
  const key = partKey(url, model);
  // a Cache API that fails here (T117 met it in the service worker) leaves the network to answer
  const cached = await cache?.match(key).catch(() => undefined);
  if (cached) {
    return cached;
  }
  const res = await fetch(url, { signal });
  if (res.ok && cache) {
    // stored while the other copy streams into Python; a full disk must not stop the download. A part that is
    // cancelled half way is not stored at all, the finished ones stay for the next time. Nor is one whose body
    // ends short without an error (the review of T97): it would come back from here on every visit
    const expected = partBytes(model, part);
    let count = 0;
    const whole = new TransformStream({
      transform(chunk, out) {
        count += chunk.byteLength;
        out.enqueue(chunk);
      },
      flush() {
        if (count !== expected) throw new Error(`part ${part}: ${count} of ${expected} bytes`);
      },
    });
    cache.put(key, new Response(res.clone().body.pipeThrough(whole))).catch(() => {});
  }
  return res;
}
// a part that broke is not read from the cache again: the next try asks the network
async function forgetPart(url, model) {
  const cache = await globalThis.caches?.open(MODEL_CACHE).catch(() => undefined);
  await cache?.delete(partKey(url, model)).catch(() => {});
}

// parts of this checkpoint that were cached for another size are of no use any more
export async function dropStaleParts(model) {
  const cache = await globalThis.caches?.open(MODEL_CACHE).catch(() => undefined);
  for (const request of (await cache?.keys().catch(() => undefined)) ?? []) {
    const url = new URL(request.url);
    if (url.pathname.includes(`/models/${model.checkpoint}.`) && url.searchParams.get("bytes") !== String(model.bytes)) {
      cache.delete(request).catch(() => {});
    }
  }
}

export function download(model, signal, load) {
  const parts = Math.ceil(model.bytes / PART_BYTES);
  const queue = [];
  const started = performance.now();
  const inner = innerAbort(signal);
  let sink, next = 0, received = 0, reported = -1;
  // T115: the checkpoint's first bytes (its header), as soon as the first part brings them
  let head = new Uint8Array(0), tell;
  const header = new Promise((resolve) => { tell = resolve; });
  // T97: Firefox on Windows breaks the body of a part now and then ("Error in input stream", 1 load in 12 on the CI
  // runners, with the service worker and without it alike): the part is fetched again, twice at most. Its chunks go
  // to the same offsets, so what arrived before the break is written over with the same bytes. T129 (4): so is a
  // part the server failed (5xx, 408), as a fetch from huggingface.co is.
  const partUrl = (part) => new URL(`../models/${model.checkpoint}.${String(part).padStart(3, "0")}`, import.meta.url).href;
  const fetchOnce = async (part) => {
    const res = await fetchPart(partUrl(part), model, part, inner.signal);
    if (!res.ok) {
      throw Object.assign(new Error(`Could not fetch part ${part} of ${model.checkpoint}: ${res.status}`), { final: !worthRetrying(res.status) });
    }
    const reader = res.body.getReader();
    let got = 0;
    try {
      for (let offset = part * PART_BYTES; ;) {
        const { done, value } = await reader.read();
        if (done) {
          // a body that ends short without an error is a break too (the review of T97)
          if (got !== partBytes(model, part)) throw new Error(`part ${part} of ${model.checkpoint} ended after ${got} of ${partBytes(model, part)} bytes`);
          return;
        }
        // a chunk that was already on its way when the load was cancelled: its buffer is gone
        inner.signal.throwIfAborted();
        if (sink) {
          // T129 (3): a write the memory refused (gone, or too small) is not cured by fetching the part again, nor is a
          // GPU's worker that takes no more of the weights (T156: a model on the GPU alone, its worker no more than
          // FLOW_BYTES behind; it gives up after FLOW_STALL_MS, and three more tries would wait that long each)
          try {
            sink(offset, value);
            await weightsRoom();
          } catch (error) {
            throw Object.assign(error, { final: true });
          }
        } else {
          queue.push([offset, value]);
        }
        // the header's bytes that this chunk brings (a part fetched again brings some a second time)
        if (head.length < HEADER_BYTES && offset <= head.length && offset + value.length > head.length) {
          head = new Uint8Array([...head, ...value.subarray(head.length - offset, HEADER_BYTES - offset)]);
          if (head.length === HEADER_BYTES) tell(head);
        }
        offset += value.length;
        got += value.length;
        received += value.length;
        // one message per percent is plenty; none that goes back (T129 (4): a part fetched again takes back what its
        // broken body had brought, and the page showed the bar going back for a moment)
        const percent = Math.floor((received / model.bytes) * 100);
        if (percent > reported) {
          reported = percent;
          postMessage({ type: "progress", load, received, total: model.bytes });
        }
      }
    } catch (error) {
      received -= got;  // counted again when the part comes again
      reader.cancel().catch(() => {});
      throw error;
    }
  };
  const connection = async () => {
    while (next < parts) {
      const part = next++;
      for (let attempt = 0; ; attempt++) {
        try {
          await fetchOnce(part);
          break;
        } catch (error) {
          if (inner.signal.aborted || error.final) {
            throw error;
          }
          if (attempt === 2) {
            // T129 (4): which part, where the browser's words do not say ("TypeError: Error in input stream")
            throw new Error(`Part ${part} of ${model.checkpoint} failed three times: ${error?.message ?? told(error)}`, { cause: error });
          }
          console.warn(`part ${part} of ${model.checkpoint} broke off (${error?.message ?? told(error)}): fetched again`);
          await forgetPart(partUrl(part), model);  // it may have come from the cache: the next try is the network's
        }
      }
    }
  };
  // The download runs while Pyodide loads, so it usually ends long before into() can write anything into Python.
  // Its own seconds are the time until the last byte arrived, not the time until the waiting was over as well.
  const source = { overlapped: true };
  const finished = Promise.all(Array.from({ length: Math.min(CONNECTIONS, parts) }, connection))
    .then(() => { source.seconds = since(started); }, (error) => {
      inner.abort(error);  // T129 (3): the other connections stop with the part that failed for good
      throw error;
    })
    .finally(() => inner.done());
  // a load that is cancelled while Pyodide still loads never gets to into(): that is no unhandled rejection
  finished.catch(() => {});
  // a download that fails before the header came fails the wait for it
  source.header = Promise.race([header, finished.then(() => head)]);
  source.header.catch(() => {});
  // write(offset, chunk) receives everything queued so far, and every later chunk
  // T129 (3): the load stops the download where it failed without it (the memory refused the queued chunks, or the
  // checkpoint before into(): weightsBuffer() said no)
  source.stop = (why) => inner.abort(why);
  source.into = async (write) => {
    sink = write;
    try {
      queue.splice(0).forEach(([offset, chunk]) => write(offset, chunk));
    } catch (error) {
      source.stop(error);
      throw error;
    }
    await finished;
    if (received !== model.bytes) {
      throw new Error(`${model.checkpoint}: got ${received} bytes instead of ${model.bytes}`);
    }
  };
  return source;
}

// The same for a file of the visitor's own disk: read in chunks straight into the Python buffer, never as a whole.
export function readFile(model, signal, load) {
  // a file is read only once there is somewhere to put it, so these seconds begin here and not at the choice
  const source = {
    async into(write) {
      const started = performance.now();
      const reader = model.file.stream().getReader();
      let reported = -1;
      for (let offset = 0; ;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (signal.aborted) {
          reader.cancel();
          signal.throwIfAborted();
        }
        write(offset, value);
        await weightsRoom();  // (T156)
        offset += value.length;
        const percent = Math.floor((offset / model.bytes) * 100);
        if (percent !== reported) {
          reported = percent;
          postMessage({ type: "progress", load, received: offset, total: model.bytes });
        }
      }
      source.seconds = since(started);
    },
  };
  return source;
}

// A llama2.c checkpoint at a URL (?checkpoint=&tokenizer=): range requests in parallel, written where they belong
export function readUrl(model, signal, load) {
  const source = {
    async into(write) {
      const started = performance.now();
      let offset = 0, reported = -1;
      await inOrder(model.url.checkpoint, 0, model.bytes, (bytes) => {
        write(offset, bytes);
        offset += bytes.length;
        const percent = Math.floor((offset / model.bytes) * 100);
        if (percent !== reported) {
          reported = percent;
          postMessage({ type: "progress", load, received: offset, total: model.bytes });
        }
      }, signal);
      source.seconds = since(started);
    },
  };
  return source;
}

// The header of the legacy format: 7 ints, the shape of the model. T115: they also say what the forward pass puts
// after the checkpoint (forward.js's footprint()), which the memory is chosen by.
export const HEADER_BYTES = 28;
export const headerInts = (bytes) => [...new Int32Array(bytes.slice(0, HEADER_BYTES).buffer)];
