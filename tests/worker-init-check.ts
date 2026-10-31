// T.462: the model's worker's start (src/runtime/worker/runtime.ts's init(), asked for by runtimeAsked()) run whole in
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
//   - runtimeAsked() starts it once: a second load is the same promise.
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
function started() {
  const harness = workerHarness();
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
  assert.deepEqual(asked.loader, [`https://cdn.jsdelivr.net/pyodide/v${VERSION}/full/pyodide.ts`], "Pyodide's loader by its URL, of the version jsDelivr said");
  // (the options are the vm's object: compared as text)
  assert.equal(JSON.stringify(asked.packages), JSON.stringify([["numpy", { checkIntegrity: false }]]), "NumPy, fetched without integrity (the owner's choice, 2026-09-28)");
  assert.deepEqual(asked.unpacked, [[ARCHIVE.length, "zip"]], "engine.zip, unpacked whole");
  assert.deepEqual(asked.imported, ["llama2_numpy"], "the engine's window imported");
  assert.deepEqual(asked.written.map(([path]) => path), ["/home/pyodide/simdkernel.so", "/home/pyodide/simdkernel_relaxed.wasmlib"], "the two kernels for Python, beside it");
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
