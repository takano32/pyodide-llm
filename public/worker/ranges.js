// worker/ranges.js (T350): a file of huggingface.co (or of any server that answers range requests) in parts, fed in
// order, with what is tried again and what stops the other connections (T107, T112, T119, T129).
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

const { state } = await import(new URL(`state.js${new URL(import.meta.url).search}`, import.meta.url));
const { breathe } = await import(new URL(`clock.js${new URL(import.meta.url).search}`, import.meta.url));
const { weightsRoom } = await import(new URL(`weights.js${new URL(import.meta.url).search}`, import.meta.url));

// T129 (3): an AbortController of one download or fetch in order, which also stops where signal (the load's) does.
// A part that failed for good ends the download: the other connections stop with it rather than fetching the rest of
// the model until the next load cancels this one (the review of T97). done() lets go of the load's signal.
export function innerAbort(signal) {
  const inner = new AbortController();
  const outer = () => inner.abort(signal.reason);
  if (signal.aborted) outer();
  else signal.addEventListener("abort", outer, { once: true });
  inner.done = () => signal.removeEventListener("abort", outer);
  return inner;
}
// T129 (4, 5): the statuses another try may cure: the server's own failures (5xx) and 408 (it gave up waiting). The
// rest of 4xx is answered the same the next time; 429 is huggingface.co's limit on the requests of an address, which
// a try within the second only adds to (refused() says to wait).
export const worthRetrying = (status) => status >= 500 || status === 408;

// Every await in here may end with the AbortError of signal: a newer load has taken over, and this one must
// leave nothing behind, least of all a Python buffer as large as its model.
// A Hugging Face model of huggingface.co ({repo, revision, weights, config, tokenizer} are names; one of the visitor's
// disk is read as the disk gives it, conduct.js): model.safetensors arrives in the order of the file, a few
// megabytes at a time, and the Python code that builds the models of this site converts every tensor as it comes and
// writes it to its place in a buffer of the final size. Reading in the order of the output instead would mean
// hundreds of range requests, and each one takes a second.
// 16 MiB over 6 connections (T107, measured in CI against huggingface.co): parts of 8 MiB took 1.36 times as long
// for Qwen2.5 0.5B, of 4 MiB 2.8 times; more connections gained 6% at most. But the first bytes then come late on a
// slow line, and a phone has less room for what waits in the queue (two parts per connection: 192 MB at 16 MiB),
// so the first part is 8 MiB wherever the size is not fixed by the URL, and the rest follow what it measured
// (the owner's ask, 2026-09-25): 16 MiB where that part came in at 4 MB/s or more and the device says nothing of a
// small memory, 8 MiB otherwise.
const HF_PART_BYTES = 16 * 1024 * 1024;
const HF_SMALL_PART_BYTES = 8 * 1024 * 1024;
const HF_FAST_BYTES_PER_SECOND = 4e6;

// The size of a file, for the few places that need it (the whole of a model: how many parts to ask for). A range
// response says it in Content-Range, but that header is not one CORS shows by default: huggingface.co exposes it by
// name, its CDN by "*", and a browser that does not honour "*" (WebKit; T112) sees none and the fetch never began
// ("The file ended before all of its tensors were read"). Content-Length of a HEAD is always shown.
export async function fileSize(url, signal) {
  const res = await fetch(url, { method: "HEAD", signal });
  const length = Number(res.headers.get("Content-Length"));
  if (!res.ok || !Number.isFinite(length) || length <= 0) {
    throw new Error(`Could not learn the size of ${url}: ${res.status}`);
  }
  return length;
}
// the size a range response reported, or the file's size asked for separately when it did not
export const sized = async (url, result, signal) => (Number.isFinite(result.total) && result.total > 0 ? result : { ...result, total: await fileSize(url, signal) });

// Bytes from..to of a response's body, taken as they stream past and no further: the rest is cancelled.
// arriving(count) is told of every stretch kept. Fewer bytes than asked for when the body ends first.
async function bodyBetween(res, from, to, arriving) {
  const bytes = new Uint8Array(to - from);
  let at = 0, kept = 0;  // at: where in the body the next chunk begins
  const reader = res.body.getReader();
  try {
    while (at < to) {
      const { done, value } = await reader.read();
      if (done) break;
      const start = Math.max(from - at, 0), stop = Math.min(to - at, value.length);
      if (stop > start) {
        bytes.set(value.subarray(start, stop), at + start - from);
        kept += stop - start;
        arriving?.(stop - start);
      }
      at += value.length;
    }
  } finally {
    reader.cancel().catch(() => {});
  }
  return bytes.subarray(0, kept);
}

// arriving(count): told of every stretch of the body as it comes, for a progress line before a whole part is in
// A fetch that huggingface.co refused, in the visitor's words (T119). The status alone does not tell: a gated
// repository and one that is not there (or private) both answer 401; X-Error-Code (which CORS shows) says which.
// Such an answer is the same the next time: it is not asked again (status says so).
export function refused(url, res) {
  const [, repository, revision, file] = /^https:\/\/huggingface\.co\/(.+?)\/resolve\/([^/]+)\/(.+)$/.exec(url) ?? [];
  const code = res.headers.get("X-Error-Code");
  const error = new Error(!repository ? `Could not fetch ${url}: ${res.status}`
    : code === "GatedRepo" ? `${repository} is gated on huggingface.co: its owner lets it be fetched only after a login and an accepted license, which this page cannot do. A copy of it that someone else published openly may work.`
    : code === "RevisionNotFound" ? `${repository} has no revision ${revision} on huggingface.co.`
    : code === "EntryNotFound" ? `${repository} has no ${file} at ${revision} on huggingface.co` +
      // a commit that does not exist is answered so too (only a branch or tag that does not is RevisionNotFound)
      (/^[0-9a-f]{40}$/.test(revision) ? `, or has no commit ${revision}.` : ".")
    : res.status === 401 || res.status === 404 ? `huggingface.co has no public repository ${repository}: check its name.`
    // T129 (5): its limit on the requests of one address, which another try at once only adds to
    : res.status === 429 ? `huggingface.co asks this address to make fewer requests for a while. Wait a few minutes, then choose ${repository} again.`
    : `huggingface.co answered ${res.status} for ${file} of ${repository}.`);
  error.status = res.status;
  return error;
}

export async function fetchRange(url, begin, end, signal, arriving) {
  for (let attempt = 0; ; attempt++) {
    // the bytes of a body that broke come again with the next try: they are taken back (the review of T119)
    let counted = 0;
    const counting = arriving && ((count) => { counted += count; arriving(count); });
    try {
      const res = await fetch(url, { headers: { Range: `bytes=${begin}-${end - 1}` }, signal });
      if (res.status !== 206 && res.status !== 200) {
        throw refused(url, res);
      }
      // 200: the server ignored the range and sends the whole file (T112: a browser whose stack does this is one to
      // know about). What was asked for is cut out as it streams past, and the rest is never fetched: taking the
      // whole file for every part fetched SmolLM2's 145 MB ten times over, and held it whole for each (the review)
      const whole = res.status === 200;
      if (whole) {
        console.warn(`${url} answered a range request with the whole file`);
      }
      const bytes = await bodyBetween(res, whole ? begin : 0, whole ? end : end - begin, counting);
      const total = Number(whole ? res.headers.get("Content-Length") : (res.headers.get("Content-Range") ?? "").split("/")[1]);
      return { bytes, total };
    } catch (error) {
      if (counted) arriving(-counted);
      // (T129 (5): the rest of 4xx is answered the same the next time; 408 and 5xx not)
      if (signal.aborted || attempt === 2 || (error.status >= 400 && !worthRetrying(error.status))) {
        throw error;
      }
    }
  }
}

// feed(bytes) gets the file from position start to its end, in order, although the parts arrive as they like.
// The parts are cut as they are asked for: the first small, the rest by what the first one measured (see above).
// arriving(bytes): how much of the file is in so far, told as it comes (the page shows it until the conversion of
// the first part gives it percentages: on a slow line the first part alone takes a while, and a line that says
// nothing looks stuck).
export async function inOrder(url, start, size, feed, outer, arriving = () => {}) {
  const small = navigator.deviceMemory !== undefined && navigator.deviceMemory <= 4;
  let partBytes = state.hfPartBytes || HF_SMALL_PART_BYTES;
  const ranges = [];  // [begin, end] of every part asked for so far, in the order of the file
  const arrived = new Map();
  let scheduled = start, fed = 0, waiting = [], received = start;
  // T129 (3): a part that failed for good (or a feed the converter refused) stops the other connections rather than
  // fetching the rest of the file until the next load cancels this one. (One that waits for room is never woken, and
  // goes with the rest of this call.)
  const inner = innerAbort(outer), signal = inner.signal;
  const connection = async () => {
    for (;;) {
      // no more than two parts per connection wait in memory for an earlier one
      while (ranges.length - fed >= 2 * state.hfConnections) {
        await new Promise((resolve) => waiting.push(resolve));
      }
      if (scheduled >= size) {
        return;
      }
      const part = ranges.length, begin = scheduled, end = Math.min(begin + partBytes, size);
      ranges.push([begin, end]);
      scheduled = end;
      const began = performance.now();
      const { bytes } = await fetchRange(url, begin, end, signal, (count) => { received += count; arriving(received); });
      // every part lies inside the file: a short one would feed the converter a file with a hole in it, which it
      // would convert without a word (the review of T112; the header's fetches may ask past the end, these not)
      if (bytes.length !== end - begin) {
        throw new Error(`${url} gave ${bytes.length} of the ${end - begin} bytes asked for at ${begin}`);
      }
      arrived.set(part, bytes);
      if (part === 0 && !state.hfPartBytes) {
        const rate = (end - begin) / ((performance.now() - began) / 1000);
        partBytes = rate >= HF_FAST_BYTES_PER_SECOND && !small ? HF_PART_BYTES : HF_SMALL_PART_BYTES;
      }
      while (arrived.has(fed)) {
        signal.throwIfAborted();
        feed(arrived.get(fed));
        arrived.delete(fed++);
        // the conversion of a part takes a moment: let messages in (T156: and the GPU's worker catch up)
        await breathe();
        await weightsRoom();
      }
      waiting.splice(0).forEach((resolve) => resolve());
    }
  };
  try {
    await Promise.all(Array.from({ length: state.hfConnections }, connection));
  } catch (error) {
    inner.abort(error);
    throw error;
  } finally {
    inner.done();
  }
}
