// compare-engines.mjs
// The engine of another commit (main by default) and of the working tree, in one Pyodide and one process, turn by turn:
// what a model of the site writes on the page's path, in tokens a second. The page's path: Python's generate() calls
// forward.js once a token (the weights in a WebAssembly memory of forward.js's own, shared as on an isolated page; no
// software threads) and samples on simdkernel.so. Each side is its tree's Python (llama2_numpy and engine/) and its
// tree's forward.js; the built kernels are the working tree's for both (a change of the kernels is not what this
// measures: tests/q8r-bench.mjs and the like do).
//
// Runs vary by several percent, so the two are measured in the same process (AGENTS.md), and the order is turned
// around: a round is old new new old, the next new old old new. Each side is measured twice a round, so the two of one
// side say how far a measurement moves by itself: read the ratio against that.
//
// Before timing, the two must write the same text from the same seed (exit 1 otherwise).
//
//   node tests/compare-engines.mjs [model ids of src/models.js = tiny-lm llm-jp-3-150m] [--ref origin/main]
//                                  [--rounds 5] [--tokens 128] [--greedy]
// --greedy: temperature 0 (the forward pass and the least of a sampler), not the entry's own sampling.
import fs from "node:fs";
import os from "node:os";
import { loadPyodide } from "pyodide";
import { MODELS } from "../src/models.js";
import { otherTree } from "./other-tree.mjs";
import { leave } from "./leave.mjs";
import { built, placeKernels, placePython, treeOf } from "./tree.mjs";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const VALUED = ["--ref", "--rounds", "--tokens"];
const option = (name, value) => (args.includes(name) ? args[args.indexOf(name) + 1] : value);
const ref = option("--ref", "origin/main"), rounds = Number(option("--rounds", 5)), tokens = Number(option("--tokens", 128));
const greedy = args.includes("--greedy");
const named = args.filter((arg, i) => !arg.startsWith("--") && !VALUED.includes(args[i - 1]));
const models = named.length ? named : ["tiny-lm", "llm-jp-3-150m"];

const { commit, folder: other } = otherTree(ref);
const trees = { old: other, new: root };

const py = await loadPyodide();
await py.loadPackage("numpy", { messageCallback: () => {} });
placeKernels(py, treeOf());
const outside = {};
for (const [which, tree] of Object.entries(trees)) {
  // each tree's engine in a folder of its own: the two have a package of the same name (engine)
  // (the engine's files by walking each tree: its window, and its package where the commit has one)
  placePython(py, treeOf(tree), ["llama2_numpy"], `/trees/${which}`);
  const forward = await import(treeOf(tree).runtimeUrl("forward.js"));
  const kernels = forward.compileKernels(fs.readFileSync(built("simdkernel_shared.wasm")), fs.readFileSync(built("simdkernel_relaxed_shared.wasm")));
  outside[which] = (file) => {
    const checkpoint = fs.readFileSync(file);
    const { memory, base } = forward.weightsMemory(checkpoint.length, { shared: true });
    new Uint8Array(memory.buffer, base, checkpoint.length).set(checkpoint);
    return forward.external({ memory, base, size: checkpoint.length, kernels });
  };
}
py.globals.set("outside_old", outside.old);
py.globals.set("outside_new", outside.new);
py.runPython(`
import gc, statistics, sys, time

def loaded(folder):
    """llama2_numpy of the tree in folder. Each tree is imported with none of the other's modules loaded and taken out
    again: a second would else get the first one's parts and be compared with itself (AGENTS.md, T347). The modules go
    on working: nothing in them imports a part later."""
    ours = lambda: [key for key in sys.modules if key.split(".")[0] in ("llama2_numpy", "engine")]
    aside = {key: sys.modules.pop(key) for key in ours()}
    sys.path.insert(0, folder)
    try:
        import llama2_numpy
        return llama2_numpy
    finally:
        sys.path.remove(folder)
        for key in ours():
            del sys.modules[key]
        sys.modules.update(aside)

ENGINES = {"old": loaded("/trees/old"), "new": loaded("/trees/new")}
assert ENGINES["old"].__file__.startswith("/trees/old/") and ENGINES["new"].__file__.startswith("/trees/new/")
assert ENGINES["old"].Llama is not ENGINES["new"].Llama and ENGINES["old"].Tokenizer is not ENGINES["new"].Tokenizer, "the two trees share a module"
OUTSIDE = {"old": outside_old, "new": outside_new}

def compare(checkpoint, tokenizer, options, prompt, settings, tokens, rounds):
    tokenizer = open(tokenizer, "rb").read()
    llamas = {which: ENGINES[which].Llama(None, tokenizer, kernels="simdkernel.so", external=OUTSIDE[which](checkpoint), **options)
              for which in ("old", "new")}
    prompt_tokens = len(llamas["new"].tokenizer.encode(prompt, llamas["new"].specials))
    steps = prompt_tokens + 1 + tokens

    def run(which, seed):
        llama = llamas[which]
        text = "".join(llama.generate(prompt, steps=steps, seed=seed, **settings))
        return text, dict(llama.stats)

    # the same text from the same seed, and a seed that writes the whole run (no stop token on the way)
    seed = None
    for candidate in range(1, 40):
        texts = {which: run(which, candidate) for which in ("old", "new")}
        if texts["old"][0] != texts["new"][0]:
            return {"differ": [candidate, texts["old"][0], texts["new"][0]]}
        if seed is None and texts["new"][1]["sampled"] == tokens + 1:
            seed = candidate
        if seed is not None and candidate >= 3:
            break
    if seed is None:
        return {"differ": [0, "no seed of 39 wrote the whole run", ""]}
    speeds, twice = {"old": [], "new": []}, []
    for round in range(rounds + 1):  # (the first is a warm-up)
        order = ("old", "new", "new", "old") if round % 2 == 0 else ("new", "old", "old", "new")
        measured = {"old": [], "new": []}
        for which in order:
            measured[which].append(run(which, seed)[1]["tokens_per_second"])
        if round:
            for which in measured:
                speeds[which] += measured[which]
                twice.append(max(measured[which]) / min(measured[which]))
    backend = llamas["new"].backend
    for llama in llamas.values():
        llama.release()
    del llamas
    gc.collect()
    return {"old": speeds["old"], "new": speeds["new"], "twice": twice, "seed": seed, "prompt": prompt_tokens, "backend": backend}
`);

const cpu = os.cpus()[0]?.model ?? "?";
console.log(`compare-engines: ${ref} (${commit.slice(0, 7)}, old) against the working tree (new); ${cpu} (${process.arch}, ${os.cpus().length} logical cores), ` +
  `Node ${process.version}, load ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}`);
console.log(`compare-engines: ${tokens} sampled tokens a run, ${rounds} rounds after a warm-up (old new new old, then new old old new), ` +
  `${greedy ? "greedy" : "each entry's own sampling"}; tok/s of the sampled tokens, as the page says them`);
const median = (values) => { const s = [...values].sort((a, b) => a - b); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
const range = (values, digits = 1) => `${Math.min(...values).toFixed(digits)}–${Math.max(...values).toFixed(digits)}`;
const rows = [];
let failed = false;
for (const id of models) {
  const entry = MODELS.find((m) => m.id === id);
  if (!entry) throw new Error(`${id} is no model of src/models.js`);
  if (entry.hf) throw new Error(`${id} is converted in the browser: this tool takes the site's own files (make models)`);
  const settings = { ...entry.generation, ...(greedy ? { temperature: 0.0 } : {}) };
  delete settings.steps;
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(root + entry.tokenizer));
  const result = py.globals.get("compare").callKwargs(root + entry.checkpoint, "tokenizer.bin", py.toPy(entry.options ?? {}), entry.prompt,
    py.toPy(settings), tokens, rounds, {}).toJs({ dict_converter: Object.fromEntries });
  if (result.differ) {
    failed = true;
    console.log(`${id}: FAILED: the two write another text from seed ${result.differ[0]}:\n  old: ${JSON.stringify(result.differ[1])}\n  new: ${JSON.stringify(result.differ[2])}`);
    continue;
  }
  const ratio = median(result.new) / median(result.old);
  // each round's own ratio (the medians of its two and two): what the order and the minute did to both is out of it
  const paired = [];
  for (let i = 0; i < result.new.length; i += 2) paired.push((result.new[i] + result.new[i + 1]) / (result.old[i] + result.old[i + 1]));
  const noise = median(result.twice);
  rows.push(`| ${entry.name} | ${median(result.old).toFixed(1)} | ${range(result.old)} | ${median(result.new).toFixed(1)} | ${range(result.new)} | ` +
    `${ratio.toFixed(3)} | ${median(paired).toFixed(3)} | ${range(paired, 3)} | ${noise.toFixed(3)} | ${Math.max(...result.twice).toFixed(3)} |`);
  console.log(`${id}: ${result.backend}; seed ${result.seed}, a prompt of ${result.prompt} tokens; the same text from both; ` +
    `old ${result.old.map((s) => s.toFixed(1)).join(" ")}; new ${result.new.map((s) => s.toFixed(1)).join(" ")}`);
}
console.log("tok/s: the medians of the rounds' runs and their range; new ÷ old of the medians, and round by round (median, range);");
console.log("one side twice in a round, the faster ÷ the slower (median, largest): what a measurement moves by itself");
console.log("| model | old | range | new | range | new ÷ old | by round | range | one side twice | largest |");
console.log("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const row of rows) console.log(row);
console.log(`compare-engines: load after ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}${failed ? "; FAILED" : ""}`);
await leave(failed ? 1 : 0);
