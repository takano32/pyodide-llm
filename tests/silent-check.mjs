// tests/silent-check.mjs (T.380.2): every place of the sources that catches a failure and tells nobody, held to a table
// of why it may. Node only, the syntax trees alone (nothing of the site runs), under a second:
//
//   node tests/silent-check.mjs
//
// A catch that neither reads what it caught nor throws (tests/silent.mjs has the rule: `catch {}`, `catch { x = null; }`,
// `.catch(() => {})`, `.catch(() => [])`, …) loses the failure: nobody learns that it happened, nor why. Some may:
// a browser without a storage, a copy that is fetched again on the next visit, a promise that is awaited elsewhere.
// Each of those is a row of SILENT below, with why. One that is written and is no row fails this check, which then
// says the three ways out:
//   - say it: read the caught value and tell the console, in one line, what failed and what stands in for it
//     (console.warn(`… (${told(error)}): …`), src/runtime/failure.ts's told(); a part of forward.ts and of the page
//     says String(error)), or hand it to what already tells the visitor (the reason a conversion was not kept);
//   - throw it on, where going on would give a wrong result;
//   - keep it silent, with a row here that says what may fail and why nobody needs to know.
// A sentence a visitor reads is the owner's: none is made up for a catch. Never console.error (the tests of a real
// browser fail on one), and nothing on the console on a visit where nothing failed.
//
// What is held: the lister's rule on made-up sources; the table against the tree, both ways (a silent catch that is no
// row; a row whose catch is gone); every row with a reason; the catches that say (SAID) with their very line, and not
// silent; public/coi.js counted and not changed by this (tests/coi-js-check.mjs holds its text).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { silentIn, silentOf, sourcesOf } from "./silent.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));

// ---- the rule, on made-up sources
{
  const of = (text, name = "made-up.ts") => silentIn(text, name).map(({ where, what }) => `${where}: ${what}`);
  // silent: nothing reads what was caught, and nothing is thrown
  assert.deepEqual(of("function a() { try { b(); } catch {} }"), ["a: catch {}"]);
  assert.deepEqual(of("const a = () => { try { b(); } catch { kernels = null; } };"), ["a: catch { kernels = null; }"]);
  assert.deepEqual(of("function a() { try { return b(); } catch (error) { return false; } }"), ["a: catch (error) { return false; }"], "named and left unread");
  assert.deepEqual(of("cache.delete(url).catch(() => {});"), ["(the module): cache.delete(url).catch(() => {})"]);
  assert.deepEqual(of("const kept = await models().catch(() => []);"), ["(the module): models().catch(() => [])"]);
  assert.deepEqual(of("async function f() { await caches?.open(NAME).catch(() => undefined); }"), ["f: caches?.open(NAME).catch(() => undefined)"], "an optional call");
  assert.deepEqual(of("a().then(taken, () => undefined);"), ["(the module): a().then(taken, () => undefined)"], "the second handler of a then");
  assert.deepEqual(of("a().catch(function (error) { count += 1; });"), ["(the module): a().catch(function (error) { count += 1; })"]);
  assert.deepEqual(of("x.onclick = async () => { try { await copy(); } catch { button.textContent = 'could not copy'; } };"),
    ["x.onclick: catch { button.textContent = 'could not copy'; }"], "a fixed word to the visitor still loses why");
  assert.deepEqual(of("const o = { run() { try { a(); } catch {} } };"), ["run: catch {}"], "a method's name");
  // (a type is no part of it, and a line is the source's)
  assert.deepEqual(silentIn("let m: Memory | undefined;\ntry {\n  m = make() as Memory;\n} catch {\n  m = undefined;\n}\n", "made-up.ts").map(({ what, line }) => [what, line]),
    [["catch { m = undefined; }", 4]]);
  // (a function thrown from inside the handler throws for its own caller, not for this one)
  assert.deepEqual(of("try { a(); } catch { later = () => { throw new Error('x'); }; }").length, 1);
  // a long call is said by its beginning and its handler
  const long = of(`navigator.gpu.requestAdapter().then(async (adapter) => { ${"await something(adapter); ".repeat(8)} }).catch(() => {});`);
  assert.equal(long.length, 1);
  assert.equal(long[0], "(the module): navigator.gpu.requestAdapter().then(async (adapter) => { awa … .catch(() => {})");
  // not silent: the caught value is read, or something is thrown
  for (const said of [
    "try { a(); } catch (error) { console.warn(`a failed (${error}): b instead`); }",
    "try { a(); } catch (error) { throw new Error('a', { cause: error }); }",
    "try { a(); } catch { throw new Error('a did not come'); }",
    "try { a(); } catch (error) { void wentWithout(error); }",
    "a().catch((error) => void wentWithout(error));",
    "a().catch((error) => unlessUpdated(error) ?? null);",
    "a().catch(refused);",
    "a().then(done, done);",
    "a().then((value) => use(value));",
    "try { a(); } catch ({ message }) { postMessage({ error: message }); }",
    "try { a(); } finally { b(); }",
    "for (const one of list.catch) one();",
  ]) assert.deepEqual(of(said), [], said);
  // and a page's script is read as the modules are
  assert.ok(sourcesOf(root).includes("src/pages/index.astro") && sourcesOf(root).includes("public/coi.js") && sourcesOf(root).includes("src/runtime/worker/runtime.ts"));
  assert.ok(!sourcesOf(root).some((name) => name.startsWith("src/runtime/built/") || name.endsWith(".d.ts")));
}
console.log("silent-check: a catch is silent when it neither reads what it caught nor throws (made-up sources: 13 that are, 12 that are not)");

// ---- the table: file → [where, what, why]. A row is a catch that stays silent, and why it may.
const STORAGE = "no storage here (a private window, a policy): ";
const SILENT = {
  "public/coi.js": [
    // (the owner's devices come first with this file: it is counted here and never changed for this table)
    ["quietly", "promise.catch(() => undefined)", "T.117: a Cache API that will not open or read leaves the network to answer, as before there was a copy; a Service Worker has no page to tell"],
  ],
  "src/benchmark/isolation.ts": [
    ["isolating", "catch {}", `${STORAGE}the copies are kept, as by default`],
    ["isolating", "catch { return false; }", "the Service Worker's registration was refused: the page goes on unisolated, as where there is none, and the report's features say isolated: no"],
  ],
  "src/benchmark/memory.ts": [
    ["(the module)", "catch { return null; }", `${STORAGE}the page-memory section does not run, and its place on the page says so`],
  ],
  "src/benchmark/model.ts": [
    ["modelSection", "catch {}", `${STORAGE}no count of threads is remembered, and the page path searches`],
  ],
  "src/benchmark/run.ts": [
    ["runSections", "catch {}", `${STORAGE}the CPU's speed is not kept for the model page, where a model on the GPU alone then stays there`],
    ["toClipboard", "catch { refused(); }", "the clipboard refused: refused() selects the report and says to copy it by hand, which is all a visitor can do about any reason"],
  ],
  "src/benchmark/page-memory.ts": [
    ["forgetMark", "catch {}", `${STORAGE}there is no mark to remove`],
    ["readMark", "catch { return undefined; }", `${STORAGE}no mark was written either`],
    ["readMark", "catch { return undefined; }", "what stands under the mark's name is not a mark (something else wrote there): as none"],
  ],
  "src/page/draw.ts": [
    ["copy.onclick", "catch { copy.textContent = \"could not copy\"; }", "the clipboard refused: the button says so, and the table is on the page to select"],
  ],
  "src/page/isolation.ts": [
    ["isolating", "catch {}", `${STORAGE}the address alone says whether the copies are kept`],
    ["isolating", "catch { return false; }", "the Service Worker's registration was refused: the page goes on unisolated and on one core, as where there is none (T.371 is to say why there is one thread)"],
  ],
  "src/page/receiver.ts": [
    ["worker.onmessage", "catch {}", `${STORAGE}the next visit weighs the GPU and the CPU again`],
    ["worker.onmessage", "catch {}", `${STORAGE}the next visit times the shaders again`],
    ["worker.onmessage", "catch {}", `${STORAGE}the next visit searches the threads again`],
    ["worker.onmessage", "catch {}", `${STORAGE}the tokens are weighed as many`],
    ["worker.onmessage", "catch {}", "no session storage here: no second try without the Service Worker, and the error is shown as it is"],
  ],
  "src/page/remembered.ts": [
    ["threadsFor", "catch {}", `${STORAGE}search again`],
    ["gpuFor", "catch { return {}; }", `${STORAGE}or something else under the name: the GPU's forms are chosen again`],
    ["remembered", "catch { return undefined; }", `${STORAGE}or something else under the name: the first model of the list opens`],
    ["remember", "catch {}", `${STORAGE}the next visit opens the first model of the list`],
  ],
  "src/runtime/benchmark/gpu/device.ts": [
    ["(the module)", "shaders.catch(() => {})", "T.442: the shaders' module, started before any step awaits it: a failure before that awaiter would be unhandled. Every step awaits the same promise and says the failure (an update, by updated.ts)"],
  ],
  "src/runtime/benchmark/memory.ts": [
    ["grow", "catch { memory = new WebAssembly.Memory({ initial: 0 }); maximum = null; }", "a browser that takes no maximum that large: a memory with the browser's own, and the result says maximum: null"],
  ],
  "src/runtime/benchmark/sections.ts": [
    ["kernelsOf", "catch { return null; }", "the kernels came and do not compile (no WebAssembly SIMD here): null is the answer, which the section says in its own row. A file that did not come is thrown, not caught here (T.440)"],
    ["device", "catch {}", "no navigator.storage here: the features' row of OPFS says it"],
    ["ceilingsOf", "catch {}", "no relaxed SIMD here (Safari): the ceilings' row says so"],
    ["measure", "reader.cancel().catch(() => {})", "a body that is read no further: whether the cancel took is nothing the measurement reads"],
  ],
  "src/runtime/benchmark/storage.ts": [
    ["onmessage", "catch {}", "the cleaning after a run that failed, whose own failure is what is posted: nothing was made, or it stays until the next run removes it"],
  ],
  "src/runtime/failure.ts": [
    ["told", "catch { fields = undefined; }", "a thrown object that refers to itself has no JSON: it is told by its name alone"],
  ],
  "src/runtime/forward.ts": [
    ["compileKernels", "catch { relaxedModule = null; }", "no relaxed SIMD here (Safari), an ordinary visit: int8 runs on matmul_q8, and the status line leaves out \"relaxed SIMD\""],
  ],
  "src/runtime/forward/memory.ts": [
    ["memory64", "catch { return false; }", "the question itself: a browser that makes no 64-bit memory throws, and false is the answer"],
    ["weightsMemory", "catch {}", "too much address space for this browser: less is asked for, and the last refusal is thrown by the loop's end"],
  ],
  "src/runtime/kept.ts": [
    ["folders", "catch { return undefined; }", "no origin private file system here (or its folder not made yet): the Cache API is looked in, and keep() says why nothing was kept"],
    ["openCache", "globalThis.caches?.open(CACHE_NAME).catch(() => undefined)", "a Cache API that will not open (storage turned off, a private window): as none, and keep() says why nothing was kept"],
    ["keptModels", "catch {}", "a folder without a manifest is a conversion that did not finish: it is none of the kept models"],
    ["openNamed", "directory?.getDirectoryHandle(name).catch(() => undefined)", "no folder of that name is the ordinary answer for a model that was never kept: NotFoundError"],
    ["openNamed", "catch {}", "half there (a conversion that was cut off): the Cache API is looked in, then the model is converted again, which the visitor sees as a download"],
    ["keep", "directory.removeEntry(name, { recursive: true }).catch(() => {})", "the cleaning after a write that failed: that failure is what keep() returns, and the page shows it"],
    ["keeper", "directory.removeEntry(name, { recursive: true }).catch(() => {})", "nothing kept under the name before is the ordinary case: NotFoundError"],
    ["drop", "directory.removeEntry(name, { recursive: true }).catch(() => {})", "letting go of a file that is not kept: what is left has no manifest and is none of the kept models, and the next keeper() removes it"],
    ["finish", "directory.removeEntry(name, { recursive: true }).catch(() => {})", "T.329.2.2: a conversion not kept whose table is read on from its file: the folder, open still, may refuse to go now; close() removes it once the file is let go of, and what is left has no manifest meanwhile"],
    ["close", "directory.removeEntry(name, { recursive: true }).catch(() => {})", "T.329.2.2: the file of a conversion that was not kept, let go of: what is left has no manifest and is none of the kept models, and the next keeper() removes it"],
    ["openRows", "catch { return undefined; }", "T.329.2.2: no handle of the kept file here (another tab of this site holds it, or it is gone): the worker says so and reads the table into the memory"],
    ["forget", "(await folders())?.removeEntry(name, { recursive: true }).catch(() => {})", "a kept model that is already gone, or held open by a load: the list is read again after it and shows what is there"],
  ],
  "src/runtime/updated.ts": [
    ["ofThisBuild", "catch { return false; }", "the question itself: what is no address is no file of this build"],
    ["gone", "catch { return false; }", "T.367.4's line: a HEAD that fails (offline, a dropped line, blocked) does not say the file is gone, and the failure that asked is told as it was"],
  ],
  "src/runtime/worker.ts": [
    ["self.onmessage", "generating?.catch(() => {})", "only waited for here: the generation's own failure is posted by the message that started it"],
    ["self.onmessage", "current.catch(() => {})", "unloaded is what the next load waits for: this load's failure is thrown by `await current` on the next line"],
    ["self.onmessage", "generating?.catch(() => {})", "only waited for here: the generation's own failure is posted by the message that started it"],
    ["self.onmessage", "current.catch(() => {})", "unloaded is what the next load waits for: this load's failure is thrown by `await current` on the next line"],
  ],
  "src/runtime/worker/convert.ts": [
    ["keepConverted", "keptModule.replaced(model).catch(() => [])", "T.136: what was kept of the model's old source could not be listed: it stays and takes room, and the new one is kept or keep() says why not"],
    ["keepConverted", "keptModule.forget(old).catch(() => {})", "T.136: an old copy that could not be deleted stays and takes room: keep() says so if the new one then does not fit"],
  ],
  "src/runtime/worker/load.ts": [
    ["load", "tokenizerBytes.catch(() => {})", "awaited below: a load that ends before that (cancelled, the runtime never came) leaves no unhandled rejection"],
  ],
  "src/runtime/worker/pyodide.ts": [
    ["prefetchNumpy", "catch {}", "a head start only: Pyodide's own loadPackage fetches the wheel next, and its failure is told"],
    ["watchArrivals", "catch { return false; }", "the question itself (T.129): a browser that makes no Response of a piped stream gets its responses as they came"],
    ["watchArrivals", "catch { observer = undefined; }", "no PerformanceObserver here: the fetches alone are counted"],
  ],
  "src/runtime/worker/ranges.ts": [
    ["bodyBetween", "reader.cancel().catch(() => {})", "a body that is read no further (T.112: the whole file came for a range): whether the cancel took is nothing the load reads"],
  ],
  "src/runtime/worker/runtime.ts": [
    ["compileKernelsHere", "catch { wideKernels = undefined; }", "T.101: a browser that says it makes 64-bit memories and cannot compile their kernels (Playwright's WebKit did, on an ordinary visit) " +
      "keeps the 32-bit ones; a model past 4 GB is then refused with its own sentence"],
    ["init", "engine.catch(() => {})", "awaited below (placePython): a load that ends before that leaves no unhandled rejection"],
  ],
  "src/runtime/worker/sources.ts": [
    ["(the module)", "globalThis.caches?.delete(\"models-v1\").catch(() => {})", "what an earlier version of the page stored: if it cannot be deleted it stays, and nothing reads it"],
    ["fetchPart", "globalThis.caches?.open(MODEL_CACHE).catch(() => undefined)", "T.117: a Cache API that will not open (some private modes) leaves a plain fetch"],
    ["fetchPart", "cache?.match(key).catch(() => undefined)", "T.117: a copy that cannot be read leaves the network to answer"],
    ["fetchPart", "cache.put(key, new Response(res.clone().body.pipeThrough(whole))).catch(() => {})", "a copy that is not stored (no room, a part that broke off or was cancelled: " +
      "put() rejects for each, by design for the last two): the next visit fetches the part again"],
    ["forgetPart", "globalThis.caches?.open(MODEL_CACHE).catch(() => undefined)", "T.117: no Cache API, so there is no copy to forget"],
    ["forgetPart", "cache?.delete(partKey(url, model)).catch(() => {})", "a broken part's copy that cannot be deleted is read by the next try, which counts its bytes again: three tries, and then the failure is told"],
    ["dropStaleParts", "globalThis.caches?.open(MODEL_CACHE).catch(() => undefined)", "T.117: no Cache API, so there are no stale parts"],
    ["dropStaleParts", "cache?.keys().catch(() => undefined)", "the copies cannot be listed: the stale ones stay and take room, and none is read (the size is in the key)"],
    ["dropStaleParts", "cache.delete(request).catch(() => {})", "a stale copy that cannot be deleted stays and takes room, and is never read (the size is in the key)"],
    ["fetchOnce", "reader.cancel().catch(() => {})", "after the read itself failed, which is thrown on the next line: the cancel's own failure adds nothing"],
    ["download", "finished.catch(() => {})", "awaited by into(): a load cancelled while Pyodide still loads never gets there, and that is no unhandled rejection"],
    ["download", "source.header.catch(() => {})", "awaited by the load: one that ends before it leaves no unhandled rejection"],
  ],
  "src/benchmark/wake.ts": [
    ["ask", "got.release?.().catch?.(() => {})", "a lock that is let go at once (the run ended while the browser answered): one already released is the same"],
    ["ask", "catch {}", "the wake lock was refused (a battery saver, a page not shown, a policy): the run goes on, and the report says how long the page was hidden"],
    ["stop", "held?.release?.().catch?.(() => {})", "the run is over: a lock the browser already took back is the same"],
  ],
};

// ---- the catches that say: [file, the very text of what they tell the console]. None of them is silent (the table
// has no row of theirs), and each line reads the caught value
// (THEN: the two that no check of Node's runs, init() behind Pyodide's CDN and the page's list beside its elements, with
// the statement after the line, which is the fallback they had before they said anything)
const THEN = {
  "console.warn(`kernels: forward.js and its kernels did not start (${told(error)}): the engine runs NumPy`);": "jsKernels = null;",
  "console.warn(`kept: what this browser kept could not be listed (${String(error)}): shown as nothing kept`);": "return [];",
};
const SAID = {
  "src/runtime/worker/convert.ts": ["console.warn(`kept: no file to keep this model in as it comes (${told(error)}): it is not kept, and a model's table of every layer's inputs is held in the memory`);"],
  "src/runtime/worker/runtime.ts": [
    "console.warn(`gpu: no key of this device (${told(error)}): what the page kept of it is not used`);",
    "console.warn(`gpu: the adapter could not be asked for (${told(error)}): no GPU on this visit`);",
    "console.warn(`kernels: forward.js and its kernels did not start (${told(error)}): the engine runs NumPy`);",
    "console.warn(`kernels: the kernels of a shared memory did not compile (${told(error)}): one thread`);",
  ],
  "src/runtime/worker/load.ts": ["console.warn(`threads: the software threads did not start (${told(error)}): one thread`);"],
  "src/runtime/worker/weights.ts": [
    "console.warn(`memory: no shared memory for this model (${told(error)}): a memory that is not shared, and one thread`);",
    "console.warn(`engine: the engine of a model that failed as it was built could not be let go of (${told(error)}): its threads and GPU's worker may stay`);",
  ],
  // (forward.ts's parts and the page's read no failure.ts: the caught value in its own words)
  "src/runtime/forward/threads.ts": ["console.warn(`threads: a software thread did not start (${String(error)}): the search ends at ${best}`);"],
  "src/page/list.ts": [
    "console.warn(`kept: what this browser kept could not be listed (${String(error)}): shown as nothing kept`);",
    "console.warn(`kept: ${one.name} could not be deleted (${String(error)}): it is still kept`);",
  ],
};

// ---- the tree against the table
const found = new Map();
for (const { file, where, what, line } of silentOf(root)) found.set(file, [...(found.get(file) ?? []), { where, what, line }]);
const key = ({ where, what }) => `${where}: ${what}`;
const WAYS = "say it (read the caught value: console.warn(`what failed (${told(error)}): what stands in`), never console.error), throw it on, or keep it silent " +
  "with a row of SILENT in tests/silent-check.mjs that says what may fail and why nobody needs to know (AGENTS.md: 失敗を捕まえて黙るとき)";
let rows = 0;
for (const file of new Set([...found.keys(), ...Object.keys(SILENT)])) {
  const here = found.get(file) ?? [], listed = (SILENT[file] ?? []).map(([where, what, why]) => ({ where, what, why }));
  for (const one of here) {
    const at = listed.findIndex((row) => !row.taken && key(row) === key(one));
    assert.ok(at >= 0, `${file}:${one.line}: ${one.where} catches a failure and tells nobody (${one.what}), and the table has no row for it: ${WAYS}`);
    listed[at].taken = true;
  }
  for (const row of listed) {
    assert.ok(row.taken, `${file}: the table (SILENT, tests/silent-check.mjs) has a row for ${key(row)}, and no silent catch of ${row.where} reads so any more: take the row out, or write it as the catch is now`);
    assert.ok(typeof row.why === "string" && row.why.split(" ").length >= 6, `${file}: ${key(row)} is silent without a reason: what may fail there, and why nobody needs to know`);
    rows++;
  }
  // (the rows stand as the catches do in the file, so that a row is found by reading down)
  assert.deepEqual(listed.map(key), here.map(key), `${file}: the rows of the table are not in the order of the file`);
}
assert.equal(SILENT["public/coi.js"].length, 1, "public/coi.js is counted and not changed for this table");
const files = Object.keys(SILENT).length;
console.log(`silent-check: ${rows} catches in ${files} files tell nobody, each with a row that says why it may (${sourcesOf(root).length} sources read)`);

// ---- the ones that say
let saying = 0;
for (const [file, lines] of Object.entries(SAID)) {
  const text = fs.readFileSync(path.join(root, file), "utf8");
  for (const line of lines) {
    assert.ok(text.includes(line), `${file} no longer tells the console ${JSON.stringify(line)}: a catch that said its failure says it still, or its row here goes with it`);
    assert.ok(/^console\.warn\(`[^`]+ \(\$\{(?:told|String)\(error\)\}\): [^`]+`\);$/.test(line), `${file}: ${line} is one line of console.warn: what failed, (the caught value), what stands in`);
    if (THEN[line]) {
      const after = text.slice(text.indexOf(line) + line.length).trimStart().split("\n")[0];
      assert.equal(after, THEN[line], `${file}: after it says ${JSON.stringify(line.slice(0, 50))}… the catch goes on as it did before it said anything`);
    }
    saying++;
  }
}
for (const name of sourcesOf(root).filter((name) => name.startsWith("src/runtime/"))) {
  const text = fs.readFileSync(path.join(root, name), "utf8").replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");
  assert.ok(!/console\.error\(/.test(text), `${name} writes with console.error: the tests of a real browser fail on a line of it (tests/e2e.mjs). console.warn says a failure something stood in for`);
}
console.log(`silent-check: ${saying} catches tell the console what failed and what stood in for it; no file of the runtime writes with console.error`);
console.log("silent-check: ok");
