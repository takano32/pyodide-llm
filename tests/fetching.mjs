// tests/fetching.mjs (T357.6)
// Fetching for the Node tools, in one place: a file to disk, and an answer's body as bytes. The Node side of
// tests/fetching.py, with its rules:
//
//   - a body is held to the length its answer promises (Content-Length);
//   - a body that stopped short, a connection that broke, a request that timed out and a server's own failure (5xx)
//     are asked for again, `tries` times in all;
//   - what the server refuses (4xx: a file the repository does not have) is not asked for again: it is thrown, an
//     Answered with the status, or is null where the file is optional;
//   - a file goes to its place through "<target>.part", so a file that is there is a whole one; one that is there
//     is not fetched.
//
//   await download(url, target, { tries = 3, timeout = 60000, optional = false })   -> target, or null
//   await fetched(url, { tries = 3, timeout = 60000 })                             -> a Buffer
//   both: { wait: ms more before each next try (wait, 2 * wait, ...; none by default), said(error, attempt): told of
//           each try that failed but the last, headers: more of them, fetch and sleep: the tests' own }
//
// Where Node differs from Python, and this file with it:
//   - Node's fetch fails by itself where a connection closes before the promised length ("TypeError: terminated";
//     http.client's read() returns what came, without a word: the review of T247). The comparison of the lengths is
//     what holds past that: an answer that ends in order and short of what it said.
//   - Node's fetch asks for a compressed body and unpacks it, and Content-Length is then the length that travelled
//     (GitHub Pages says 9,634 of a page of 55,199 bytes). So the body is asked for as it is (Accept-Encoding:
//     identity), and an answer that is encoded all the same is not held to a length.
//   - the time limit is of the whole request, the body too (AbortSignal.timeout), not of each read.
//   - the ".part" of a download that failed for good is removed.
// Not for what measures fetching itself (tests/fetch-node.mjs), nor for a file that is resumed and held to a sha256
// (the 27B's GGUF: tests/reference_27b.sh's curl).
import fs from "node:fs";
import path from "node:path";

/** A server's answer that was not the file: its status (4xx: refused, and not asked again; 5xx: asked again). */
export class Answered extends Error {
  constructor(url, status) {
    super(`${url}: the server answered ${status}`);
    this.status = status;
  }
}

const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** One try: the body of the answer, piece by piece to take(piece), all of what the answer said would come. */
async function once(url, { timeout, headers, fetch }, take) {
  const response = await fetch(url, { headers: { "Accept-Encoding": "identity", ...headers }, signal: AbortSignal.timeout(timeout) });
  if (!response.ok) {
    await response.body?.cancel();
    throw new Answered(url, response.status);
  }
  const promised = response.headers.get("content-encoding") ? null : response.headers.get("content-length");
  let came = 0;
  for await (const piece of response.body ?? []) {
    came += piece.length;
    take(piece);
  }
  if (promised !== null && came !== Number(promised)) throw new Error(`${url}: ${came} of ${Number(promised)} bytes came`);
}

/** attempt(), again where it failed with what another try may mend. */
async function tried({ tries, wait, said, sleep }, attempt) {
  for (let n = 1; ; n++) {
    try {
      return await attempt();
    } catch (error) {
      if (error.status < 500 || n >= tries) throw error;
      said?.(error, n);
      if (wait) await sleep(wait * n);
    }
  }
}

const settled = ({ tries = 3, timeout = 60000, wait = 0, optional = false, headers = {}, said, fetch = globalThis.fetch, sleep = pause } = {}) =>
  ({ tries, timeout, wait, optional, headers, said, fetch, sleep });

/** The file at url as target (a path), fetched unless it is there: target, or null where the server refused a file
 * that is optional. */
export async function download(url, target, options) {
  const asked = settled(options);
  if (fs.existsSync(target)) return target;
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const partial = `${target}.part`;
  try {
    await tried(asked, async () => {
      const out = fs.openSync(partial, "w");
      try {
        await once(url, asked, (piece) => fs.writeFileSync(out, piece));
      } finally {
        fs.closeSync(out);
      }
    });
  } catch (error) {
    fs.rmSync(partial, { force: true });
    if (asked.optional && error.status < 500) return null;
    throw error;
  }
  fs.renameSync(partial, target);
  return target;
}

/** The body of the answer at url, whole, as a Buffer. */
export async function fetched(url, options) {
  const asked = settled(options);
  return tried(asked, async () => {
    const pieces = [];
    await once(url, asked, (piece) => pieces.push(piece));
    return Buffer.concat(pieces);
  });
}
