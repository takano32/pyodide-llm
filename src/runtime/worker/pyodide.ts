// worker/pyodide.ts (T.350): the version of Pyodide, its loading in named steps, and the watch that tells a slow line
// from one that stopped (T.118, T.129). It holds nothing of the worker's state: init() of worker.ts keeps what this loads.
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T.367.3).

import { Failure, isError, told } from "../failure.ts";
import type { Status } from "../contracts/messages.ts";
import type { Pyodide, PyodideModule } from "../contracts/python.ts";

// the version becomes part of a CDN URL, so accept nothing but a plain version number
const PYODIDE_VERSION_PATTERN = /^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$/;

// the latest release on npm (the "latest" tag never points at an alpha), or ?pyodide=<version> to force one.
// T.129 (1): its answer is a hundred bytes, so a connection that opened and never answered left "Loading Pyodide" for
// ever (the review of T.118). It is given up after QUIET_SECONDS like a step of Pyodide's (the page may try again
// without the service worker). Not a step under watchArrivals(): that would count every part of the model, which
// downloads meanwhile, and find a stop of Pyodide only after the model's end. But it is not always quick on a line the
// model's parts fill: the first bytes of a new connection wait behind them (slow.yml, the review of T.129: 20.5 s behind
// a 1 MB model at 0.4 Mbps with the oldest connection served first, 14.2 s through one queue of 256 KB at 0.4 Mbps,
// 0.8 s with the line shared by turns), so 30 seconds is little more than such a line's queue.
export async function resolvePyodideVersion(search: string) {
  const forced = new URLSearchParams(search).get("pyodide");
  if (PYODIDE_VERSION_PATTERN.test(forced!)) {  // (none: null, which is read as "null" and is no version)
    return forced!;
  }
  const given = new AbortController();
  const timer = setTimeout(() => given.abort(), QUIET_SECONDS * 1000);
  let version: string;
  try {
    const res = await fetch("https://data.jsdelivr.com/v1/packages/npm/pyodide/resolved?specifier=latest", { signal: given.signal });
    version = (await res.json()).version;
  } catch (error) {
    if (!given.signal.aborted) throw error;
    throw new Failure("runtime", `The latest version of Pyodide: data.jsdelivr.com did not answer in ${QUIET_SECONDS} seconds`);
  } finally {
    clearTimeout(timer);
  }
  if (!PYODIDE_VERSION_PATTERN.test(version)) {
    throw new Error(`Unexpected Pyodide version: ${version}`);
  }
  return version;
}

// Pyodide asks for the NumPy wheel only once it has started, several seconds after its own files: until then it
// does not know the name. That name is in pyodide-lock.json, which Pyodide fetches anyway, so this reads the lock
// as well and puts the wheel into the HTTP cache while Pyodide is still coming up. Nothing is written down here:
// the version is the one resolved at run time, the file name comes from the lock. Anything unexpected (another
// shape of the lock, a CDN that says no) only means no head start, so every error is dropped.
async function prefetchNumpy(base: string) {
  try {
    const lock = await (await fetch(`${base}pyodide-lock.json`)).json();
    const name = lock.packages?.numpy?.file_name;
    if (typeof name !== "string" || !/^[A-Za-z0-9._+-]+\.whl$/.test(name)) {
      return;
    }
    const wheel = await fetch(base + name);
    // read it to the end so that the browser keeps it, and drop every chunk: this copy is never used
    await (wheel.body ? wheel.body.pipeTo(new WritableStream()) : wheel.arrayBuffer());
  } catch {
    // no head start
  }
}

// T.118: whether Pyodide's loading still moves. Its large files (pyodide.asm.wasm 3.4 MB, the standard library 2.5 MB,
// NumPy's wheel 2.9 MB) come by this worker's fetch, which, while Pyodide loads, hands out responses whose bodies are
// counted as they arrive; a file that is finished (the two imports too) counts as well. A step is given up when
// nothing has arrived for QUIET_SECONDS, not when it takes long: on a line of 128 kbps to 1 Mbps (a phone's plan past
// its limit) the 9 MB took longer than T.113's limits of 60 to 90 seconds, and a sound load fell back to no service
// worker, then timed out again. Nothing arriving for this long is a stop, not a slow line (128 kbps is 16 kB a second).
const QUIET_SECONDS = 30;
// (T.368.3) what watches a load's arrivals: the count, stop() (the fetch as it was), quiet(seconds) (below)
type Watch = { arrived: number; stop(): void; quiet(seconds: number): { promise: Promise<void>; cancel(): void } };
function watchArrivals() {
  const watch = { arrived: 0 } as Watch;
  const plain = self.fetch;
  // T.129 (6): a browser that would not make a Response of a stream piped through a TransformStream gets its responses
  // as they came, each counted once as it arrives (the three engines of the CI make them; a throw in the wrapper
  // failed every fetch of the load). Tried once on an empty stream, before a response's body is touched
  const wraps = (() => {
    try {
      new Response(new ReadableStream().pipeThrough(new TransformStream()));
      return true;
    } catch {
      return false;
    }
  })();
  self.fetch = async (...args: Parameters<typeof fetch>) => {
    const res = await plain(...args);
    watch.arrived += 1;
    if (!wraps || !res.body || [101, 204, 205, 304].includes(res.status)) {
      return res;
    }
    const counted = res.body.pipeThrough(new TransformStream({
      transform(chunk, out) {
        watch.arrived += chunk.byteLength;
        out.enqueue(chunk);
      },
    }));
    return new Response(counted, { status: res.status, statusText: res.statusText, headers: res.headers });
  };
  let observer: PerformanceObserver | undefined;
  try {
    observer = new PerformanceObserver((list) => { watch.arrived += list.getEntries().length; });
    observer.observe({ type: "resource" });
  } catch {
    observer = undefined;  // no PerformanceObserver here: the fetches alone
  }
  watch.stop = () => {
    self.fetch = plain;
    observer?.disconnect();
  };
  /** { promise, cancel }: the promise settles once nothing has arrived for seconds. The silence is counted in the
   * ticks that ran (one a second), not in the clock's seconds: a page that a phone froze while another app was in front,
   * or a worker busy for a long while, runs no tick, and a clock that jumped over it would call that a line that
   * stopped (T.129's review; the same lesson as the software threads', T.120) */
  watch.quiet = (seconds) => {
    let timer: ReturnType<typeof setInterval>;
    const promise = new Promise<void>((resolve) => {
      let seen = -1, silent = 0;
      timer = setInterval(() => {
        if (watch.arrived !== seen) {
          [seen, silent] = [watch.arrived, 0];
        } else if (++silent >= seconds) {
          clearInterval(timer);
          resolve();
        }
      }, 1000);
    });
    return { promise, cancel: () => clearInterval(timer) };
  };
  return watch;
}

// Loads Pyodide and NumPy of version in named steps. Each step says its name, and ends in an error rather than
// never: loadPyodide() does not fail when a fetch of its files fails, it waits for ever (AGENTS.md), and a phone that
// stopped at "Loading Pyodide" said nothing else. It ends when nothing has arrived for QUIET_SECONDS (T.118), however
// long it takes while bytes keep coming. importer(url) imports pyodide.mjs (tests/worker-check.ts gives its own).
// T.129 (2), the owner's choice (2026-09-28): loadPackage fetches NumPy's wheel without integrity (checkIntegrity:
// false). A fetch with integrity settles only once its whole body is in, so watchArrivals() counted none of the wheel
// until its end, and where the prefetch had not put it in the HTTP cache (no lock, another shape of it, a CDN that said
// no, a cache that did not keep it) a line below 0.8 Mbps gave NumPy up after QUIET_SECONDS while it was arriving. The
// check of the wheel's hash (SRI) goes; the wheel comes from the same CDN and version as the rest of Pyodide.
export async function pyodideSteps(version: string, importer: (url: string) => Promise<PyodideModule>) {
  const base = `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;
  const watch = watchArrivals();
  // (of the kind "runtime": the page may try again without the service worker, isolation made this hang on iOS)
  const stop = (name: string, why: string) => new Failure("runtime", `Pyodide ${version}: "${name}" ${why}`);
  const step = async <T>(name: string, promise: Promise<T>) => {
    postMessage({ type: "status", text: `Loading Pyodide ${version}: ${name}...` } satisfies Status);
    const quiet = watch.quiet(QUIET_SECONDS);
    const stalled = quiet.promise.then(() => { throw stop(name, `got nothing from the network for ${QUIET_SECONDS} seconds`); });
    try {
      return await Promise.race([promise, stalled]);
    } catch (error) {
      // T.242: loadPyodide() goes on without a standard library whose fetch failed (it writes that to the console), and
      // Python then ends as it starts: the promise rejects with Emscripten's ExitStatus, which is no Error and said
      // "[object Object]" (bench.yml's Windows WebKit, 2026-10-01: its fetches of jsDelivr failed together now and then).
      // Told as a step of Pyodide's that stopped, as one whose file never came is
      if (isError(error)) throw error;
      throw stop(name, `ended as it started (${told(error)}): one of its files may not have arrived`);
    } finally {
      quiet.cancel();
    }
  };
  try {
    // while Pyodide starts, not after: loadPackage("numpy") below finds the wheel in the HTTP cache
    const numpy = prefetchNumpy(base);
    const { loadPyodide } = await step("the loader", importer(`${base}pyodide.mjs`));
    const loaded = await step("the runtime", loadPyodide());
    // a prefetch that is still running would otherwise be raced by loadPackage, and the wheel fetched twice. One
    // that stopped is not waited for past a quiet spell (T.118): loadPackage then fetches the wheel itself
    const quiet = watch.quiet(QUIET_SECONDS);
    await Promise.race([numpy, quiet.promise]);
    quiet.cancel();
    await step("NumPy", loaded.loadPackage("numpy", { checkIntegrity: false }));
    return loaded;
  } finally {
    watch.stop();
  }
}

// T.367.2: the site's Python comes as one archive a module (python_archive.py builds them of src/python/): engine.zip, which
// every load imports, and converter.zip, which the first conversion asks for. One request each, where every .py was one
// (fifteen and nineteen). An archive is a built file (built.ts): the bundler gives it a name with a hash of its bytes,
// and the worker's own text names that file, so it is the Python of the worker's own build (T.367.3). It is fetched as
// every built file is (built.ts's built(): T.367.4).

/** Unpacks an archive (its bytes, or a promise of them) where `import` finds what is in it: Pyodide's working folder,
 * where the files were written one by one before. Nothing is placed unless the whole archive came.
 * An archive that came and cannot be read (cut short, or something else served in its place) is said so by its name
 * (T.432): what Python says of it names a temporary file. An archive that did not come fails as its fetch did. */
export async function placePython(pyodide: Pyodide, archive: ArrayBuffer | Promise<ArrayBuffer>, name: string) {
  const bytes = await archive;
  try {
    pyodide.unpackArchive(bytes, "zip");
  } catch (error) {
    throw new Error(`Could not read ${name}: the file did not arrive whole. Reload the page to try again.`, { cause: error });
  }
}

// T.397: jinja2, with which the converter renders a model's own chat template as transformers does (convert/template.py).
// Pyodide's package of the version that is loaded, by its lock (no version is written here), asked for when a
// conversion is first made: most visitors never convert anything. Whether it came: where it did not (the CDN said no,
// nothing arrived for QUIET_SECONDS), the conversion goes on and the converter's own reader reads what it can, so
// nothing is thrown and nothing is said to the visitor.
// One load at a time for a Pyodide: watchArrivals() wraps self.fetch and puts back the one it found, so two overlapping
// ones (a conversion that was cancelled while jinja2 came, and the next one) would leave the first's wrapper on every
// fetch for good (T.369 review). A load that did not succeed is forgotten, and the next conversion tries again.
const templatePackages = new WeakMap<Pyodide, Promise<boolean>>();
export function templatePackage(pyodide: Pyodide) {
  if (!templatePackages.has(pyodide)) {
    templatePackages.set(pyodide, loadTemplatePackage(pyodide).then((came) => {
      if (!came) templatePackages.delete(pyodide);
      return came;
    }));
  }
  return templatePackages.get(pyodide);
}

async function loadTemplatePackage(pyodide: Pyodide) {
  const watch = watchArrivals(), quiet = watch.quiet(QUIET_SECONDS);
  try {
    const stalled = quiet.promise.then(() => { throw new Error(`nothing arrived for ${QUIET_SECONDS} seconds`); });
    await Promise.race([pyodide.loadPackage("jinja2", { messageCallback: () => {} }), stalled]);
    pyodide.pyimport("jinja2").destroy();  // (a package that did not come is said by loadPackage in the console only)
    return true;
  } catch (error) {
    console.warn(`jinja2 did not load (${told(error)}): chat templates are read by the converter's own reader`);
    return false;
  } finally {
    quiet.cancel();
    watch.stop();
  }
}
