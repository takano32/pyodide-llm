// T462: the model's worker's start (src/runtime/worker/runtime.ts's init(), asked for by runtimeAsked()) run whole in
// Node, which no check did: the others set by hand what init() sets. The worker runs in tests/worker-harness.ts's vm on
// its made-up network: data.jsdelivr.com answers the version of Pyodide, engine.zip is made-up bytes (the made-up Pyodide
// reads none of it, and the deploy's light suite runs before the build makes the archives), this tree's kernels
// (src/runtime/built/, `make kernels`) come from the disk, and every other request is a 404 (pyodide-lock.json: no
// wheel read ahead). The two modules a
// browser imports from elsewhere are handed in (runtimeAsked(search, starts)): a made-up Pyodide, which writes down what
// it is asked (the archive it unpacks, the files written into it, the module imported, NumPy), and forward.ts as Node
// imports it, or a forward.ts that does not come.
//   - init() reads the URL's switches, resolves the version, starts Pyodide by its loader's URL, unpacks engine.zip,
//     imports llama2_numpy, writes the two kernels for Python beside it, and compiles the plain kernels of forward.ts
//     (no shared ones: the vm is not cross-origin isolated);
//   - where forward.ts does not come, the engine runs on NumPy: no kernels of forward.ts, and the console says why;
//   - runtimeAsked() starts it once: a second load is the same promise;
//   - T641.8: the two the worker began as its modules were imported, it begins in init(), with its first load: the
//     adapter asked for (none, one that cannot be asked for, one that is not what one is, one without a key: T380.2's,
//     out of tests/worker-parts-check.ts), and the Cache API of an earlier version of the page deleted.
// Not held here: STARTS's own catch of forward.ts (the failure handed in is the check's: tests/updated-check.ts holds that
// catch's text), the switches but hfParts (hfConnections, wide, gpuTest, bench), the shared kernels of a cross-origin
// isolated page, the kernels of a 64-bit memory as they are kept, and blockKernels(). Pyodide's real start is trodden
// by the browsers (tests/e2e.ts).
//   node tests/worker-init-check.ts
import assert from "node:assert/strict";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import { workerHarness } from "./worker-harness.ts";
import { runtimeUrl } from "./tree.ts";

const VERSION = "314.0.7";
// (engine.zip's bytes as this network has them)
const ARCHIVE = new Uint8Array(4096).map((_, i) => i % 251);
const forward = await import(runtimeUrl("forward.ts"));

// a made-up Pyodide: what the worker asks of it, written down
function madeUpPyodide() {
  const asked = { loader: [], packages: [], unpacked: [], written: [], imported: [] };
  const module = { marker: "llama2_numpy" };
  const loaded = {
    version: VERSION,
    loadPackage: async (names, options) => { asked.packages.push([names, options]); },
    unpackArchive: (bytes, format) => { asked.unpacked.push([bytes.byteLength, format]); },
    FS: { writeFile: (path, data) => { asked.written.push([path, data.length]); } },
    pyimport: (name) => { asked.imported.push(name); return module; },
  };
  const importer = async (url) => { asked.loader.push(url); return { loadPyodide: async () => loaded }; };
  return { asked, module, importer };
}

// a worker on the made-up network: the version, and the built files from the disk
function started(gpu = false) {
  const harness = workerHarness({ gpu });
  harness.fresh((url) => {
    if (url === "https://data.jsdelivr.com/v1/packages/npm/pyodide/resolved?specifier=latest") return Response.json({ version: VERSION });
    if (url.endsWith("/built/engine.zip")) return new Response(ARCHIVE);
    if (url.startsWith("file:")) {
      const file = fileURLToPath(url);
      return fs.existsSync(file) ? new Response(fs.readFileSync(file)) : new Response("", { status: 404 });
    }
    return new Response("", { status: 404 });
  });
  return harness;
}

// ---- the whole start, with forward.ts
{
  const { run, requests } = started();
  const { asked, module, importer } = madeUpPyodide();
  const starts = { pyodide: importer, forward: async () => forward };
  const first = run("runtimeAsked")("?hfParts=16", starts);
  assert.equal(run("runtimeAsked")("?hfParts=16", starts), first, "runtimeAsked() starts the runtime once: a second load waits on the same promise");
  await first;
  assert.deepEqual(asked.loader, [`https://cdn.jsdelivr.net/pyodide/v${VERSION}/full/pyodide.mjs`], "Pyodide's loader by its URL, of the version jsDelivr said");
  // (the options are the vm's object: compared as text)
  assert.equal(JSON.stringify(asked.packages), JSON.stringify([["numpy", { checkIntegrity: false }]]), "NumPy, fetched without integrity (the owner's choice, 2026-09-28)");
  assert.deepEqual(asked.unpacked, [[ARCHIVE.length, "zip"]], "engine.zip, unpacked whole");
  assert.deepEqual(asked.imported, ["llama2_numpy"], "the engine's window imported");
  assert.deepEqual(asked.written.map(([path]) => path), ["/home/pyodide/simdkernel.so"], "the kernels for Python, beside it (T644.2: no relaxed module)");
  assert.equal(run("llama2_numpy"), module);
  assert.equal(run("pythonKernels"), "/home/pyodide/simdkernel.so");
  assert.equal(run("forwardModule"), forward, "forward.ts as it was handed in");
  assert.ok(run("jsKernels")?.plain, "the plain kernels of forward.ts compiled");
  assert.ok(!run("sharedKernels"), "and no shared ones where the page is not cross-origin isolated");
  assert.equal(run("hfPartBytes"), 16 * 1024 * 1024, "the URL's switches read");
  assert.ok(typeof run("pyodideAt") === "number" && run("pyodide").version === VERSION);
  assert.ok(!requests.some(({ url }) => url.endsWith("forward.ts")), "forward.ts is not fetched: it is the one handed in");
}
console.log("worker-init-check: init() resolves Pyodide's version, starts it by its loader's URL, unpacks engine.zip, writes the kernels for Python and compiles forward.ts's");

// ---- forward.ts that does not come: the engine runs on NumPy
{
  const { run } = started();
  const { importer } = madeUpPyodide();
  const warned = [], warn = console.warn;
  console.warn = (line) => warned.push(String(line));
  try {
    await run("runtimeAsked")("", { pyodide: importer, forward: async () => { throw new TypeError("Failed to fetch dynamically imported module"); } });
  } finally {
    console.warn = warn;
  }
  assert.equal(run("jsKernels"), null, "no kernels of forward.ts");
  assert.ok(run("llama2_numpy"), "and the engine's Python there, which runs on NumPy");
  assert.ok(warned.some((line) => line.startsWith("kernels: forward.js and its kernels did not start") && line.endsWith("the engine runs NumPy")),
    `the console says why: ${JSON.stringify(warned)}`);
}
console.log("worker-init-check: where forward.ts does not come, the engine runs on NumPy and the console says why");

// ---- T641.8: the adapter and the old Cache API, asked for by init() and not as the modules are imported
{
  const asked = [], deleted = [];
  const { run, context } = started({ requestAdapter: async () => { asked.push("adapter"); return null; } });
  context.caches = { delete: async (name) => { deleted.push(name); return true; } };
  // (the worker's modules have run: nothing of the two yet)
  assert.equal(run("adapterAsked"), undefined, "no adapter is asked for as the worker's modules are imported (T641.8)");
  assert.deepEqual([asked, deleted], [[], []], "nothing asked of the GPU or the Cache API before the first load");
  const { importer } = madeUpPyodide();
  const first = run("runtimeAsked")("", { pyodide: importer, forward: async () => forward });
  // (init()'s first statements, before it awaits anything: the load that follows waits on adapterAsked)
  assert.deepEqual(asked, ["adapter"], "init() asks for the adapter before it awaits anything");
  assert.deepEqual(deleted, ["models-v1"], "and deletes what an earlier version of the page stored");
  await first;
  assert.deepEqual([asked, deleted], [["adapter"], ["models-v1"]], "once, whatever the loads after");
}

// T380.2 (out of tests/worker-parts-check.ts): what fails as the adapter is asked for was silent: an adapter that cannot
// be asked for, and a device with no key (in this vm no module can be imported later, which is such a failure)
{
  const begun = async (gpu) => {
    const warn = console.warn, said = [];
    console.warn = (...words) => said.push(words.join(" "));
    try {
      const { run } = started(gpu);
      const { importer } = madeUpPyodide();
      await run("runtimeAsked")("", { pyodide: importer, forward: async () => forward });
      await run("adapterAsked");
      return { said, adapter: JSON.parse(JSON.stringify(run("gpuAdapter"))) };
    } finally {
      console.warn = warn;
    }
  };
  // no adapter here: nothing failed, and nothing is said
  assert.deepEqual(await begun({ requestAdapter: async () => null }), { said: [], adapter: null });
  // the asking itself fails
  assert.deepEqual(await begun({ requestAdapter: () => Promise.reject(new Error("the GPU process is gone")) }),
    { said: ["gpu: the adapter could not be asked for (Error: the GPU process is gone): no GPU on this visit"], adapter: null });
  // an adapter that is not what one is
  const odd = await begun({ requestAdapter: async () => ({}) });
  assert.equal(odd.adapter, null);
  assert.match(odd.said.join("|"), /^gpu: the adapter could not be asked for \(TypeError: [^|]+\): no GPU on this visit$/);
  // an adapter, and no key of it: the adapter stands, without one
  const keyless = await begun({ requestAdapter: async () => ({ limits: { maxStorageBufferBindingSize: 1 << 27, maxBufferSize: 1 << 28 }, info: { isFallbackAdapter: false } }) });
  assert.deepEqual(keyless.adapter, { fallback: false, limits: { maxStorageBufferBindingSize: 1 << 27, maxBufferSize: 1 << 28 }, packed: false });
  assert.match(keyless.said.join("|"), /^gpu: no key of this device \([^|]+\): what the page kept of it is not used$/);
  // a fallback adapter: said to be one
  const fallback = await begun({ requestAdapter: async () => ({ limits: { maxStorageBufferBindingSize: 1, maxBufferSize: 2 }, info: { isFallbackAdapter: true } }) });
  assert.equal(fallback.adapter.fallback, true, "a fallback adapter is taken for one");
}
console.log("worker-init-check: init() asks for the adapter and deletes the old Cache API before it awaits anything (none as the modules are imported); an adapter that cannot be asked for and a device without a key are said on the console");
