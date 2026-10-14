// plan-keys-check.mjs (T357)
// The two plans that cross a line as plain objects, held to both of their ends: a key one side writes and the other
// never reads, or reads and is never written, is said. Nothing else holds them: a key that is not there is undefined
// without a word (`plan.rotary` spelled otherwise turns whole heads). T375.2: the tests that run forward.js alone are
// handed Python's own plan (tests/plans.mjs's planOf(), with a check's few settings over it: held to Python's keys
// here), and the one plan that is written by hand, /benchmark/'s made-up model's (src/runtime/benchmark/madeup.ts,
// which runs without Pyodide), is held here to the plan Python makes for the same header, value by value.
//
//   1. Python -> forward.js: the plan Llama(external=) hands start() (src/python/engine/external.py's ExternalForward), as
//      Python builds it for one made-up model of every layout and dtype (tests/engine_plans.py, the native Python):
//      its keys, the names of its tensors, each tensor's keys, the names of its derived tables. Against what
//      src/runtime/forward.js and src/runtime/forward/*.js read, by their syntax trees: `plan.<key>` and what is taken out of
//      `plan`; the names handed to matrix() and floats() and read off `T` (plan.tensors); the properties read off a
//      tensor (`t.<key>`, `T.<name>.<key>`, `tensors[...]`).
//   2. forward.js -> the GPU's worker: the plan of { type: "start" } (src/runtime/forward/gpuside.js's literal) and of
//      { type: "open" } (src/runtime/forward/alone.js's gpuOnlyPlan()), with what they hold one level down where that is
//      written as a literal (tokens, words, vectors, tables; a vector is read by its name as a property or in a text). Against what src/runtime/gpu.js and src/runtime/gpu/*.js read of
//      `plan` and of anything's `.plan`. Both ends by their syntax trees: the worker runs only where there is a GPU.
//      (force and remembered come from the worker and the page, and matrices is keyed by the tensors' names: their
//      insides are not looked at.)
//
//   node tests/plan-keys-check.mjs          (PYTHON=.venv/bin/python; a few seconds, no Pyodide, no kernels)
//
// What it does not see: a key read under a computed name (`plan[name]`), a plan passed on under another name than
// `plan`, and whether a value is what the reader takes it for. Exit 1 on a key with one end only.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "@babel/parser";
import { plans as plansFor } from "./plans.mjs";
import { runtime, runtimeUrl } from "./tree.mjs";
import { contract, differing, posted, told } from "./contracts.mjs";
import { typed, untyped } from "./untyped.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
let failed = 0;
const ok = (line) => console.log(`ok: ${line}`);
const wrong = (line) => { failed++; console.log(`WRONG: ${line}`); };
const sorted = (names) => [...names].sort();
// both directions of one contract: written and read are sets; unread: { key: why nothing reads it } (said, not failed)
function held(what, written, read, unread = {}) {
  const neverWritten = sorted(read).filter((key) => !written.has(key));
  const neverRead = sorted(written).filter((key) => !read.has(key) && !(key in unread));
  const stale = Object.keys(unread).filter((key) => read.has(key) || !written.has(key));
  if (neverWritten.length) wrong(`${what}: read and never written: ${neverWritten.join(", ")}`);
  if (neverRead.length) wrong(`${what}: written and read nowhere: ${neverRead.join(", ")}`);
  if (stale.length) wrong(`${what}: listed as read nowhere, and it is read now or written no more: ${stale.join(", ")}`);
  for (const [key, why] of Object.entries(unread)) if (!stale.includes(key)) console.log(`     (${what}: ${key} is written and read nowhere: ${why})`);
  if (!neverWritten.length && !neverRead.length && !stale.length) ok(`${what}: ${written.size} written, ${read.size} read, each at both ends${Object.keys(unread).length ? ` but ${Object.keys(unread).join(", ")}` : ""}`);
}

// T368.2: a type of src/runtime/contracts/ against what is written across the line it is the contract of: the same names
// (more: { name: why the type has it and this writer does not })
function typedAs(what, declared, written, more = {}) {
  const lines = differing(what, declared, written, more);
  lines.forEach(wrong);
  if (!lines.length) ok(`${what}: the type's ${declared.size} are the ${written.size} written${Object.keys(more).length ? ` and ${Object.keys(more).join(", ")}` : ""}`);
}

// ---- the syntax trees
// (T368.3: a file that is TypeScript is parsed with its types blanked, by the name the tree has it under)
const treesOf = (files) => files.map((file) => ({ file, program: parse(untyped(fs.readFileSync(runtime(file), "utf8"), runtime(file)), { sourceType: "module" }).program }));
// (T368.4: a folder's files in either spelling: counting those that end in .js alone read none of forward/ once it was TypeScript)
const folder = (window) => [window, ...fs.readdirSync(runtime(window.replace(/\.js$/, ""))).filter((file) => file.endsWith(".js") || typed(file)).sort()
  .map((file) => `${window.replace(/\.js$/, "")}/${file}`)];
function walk(node, visit, parent = null) {
  if (!node || typeof node !== "object") return;
  if (Array.isArray(node)) { node.forEach((child) => walk(child, visit, parent)); return; }
  if (node.type) visit(node, parent);
  for (const key of Object.keys(node)) if (!["loc", "start", "end", "extra", "leadingComments", "trailingComments", "innerComments"].includes(key)) walk(node[key], visit, node.type ? node : parent);
}
const member = (node) => (node.type === "MemberExpression" || node.type === "OptionalMemberExpression") && !node.computed && node.property.type === "Identifier";
// the keys an object pattern takes ({ a, b: c, ...rest } takes a and b)
const taken = (pattern) => pattern.properties.filter((p) => p.type === "ObjectProperty" && !p.computed).map((p) => p.key.name ?? p.key.value);
// the keys of an object literal, through its spreads and the branches of a conditional or a logical expression
function literalKeys(node, into = new Set()) {
  if (!node) return into;
  if (node.type === "ObjectExpression") {
    for (const property of node.properties) {
      if (property.type === "SpreadElement") literalKeys(property.argument, into);
      else if (!property.computed) into.add(property.key.name ?? property.key.value);
    }
  } else if (node.type === "ConditionalExpression") { literalKeys(node.consequent, into); literalKeys(node.alternate, into); }
  else if (node.type === "LogicalExpression") { literalKeys(node.left, into); literalKeys(node.right, into); }
  return into;
}
const functionNamed = (trees, name) => {
  let found;
  for (const { program } of trees) walk(program, (node) => { if (node.type === "FunctionDeclaration" && node.id?.name === name) found = node; });
  assert.ok(found, `no function ${name}(): tests/plan-keys-check.mjs reads the plan it returns`);
  return found;
};
// the object literal a function returns (its one return of a literal)
const returned = (fn) => {
  const literals = [];
  walk(fn.body, (node) => { if (node.type === "ReturnStatement" && node.argument?.type === "ObjectExpression") literals.push(node.argument); });
  assert.equal(literals.length, 1, `${fn.id.name}() returns ${literals.length} object literals, not one`);
  return literals[0];
};

// ---- 1. Python -> forward.js
{
  const LINEAR = { every: 4, key_heads: 2, value_heads: 4, key_dim: 64, value_dim: 128, conv: 4 };
  const BLOCKS = ["Q2_K", "Q4_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "IQ4_XS"];
  const families = [
    { name: "llama", header: [256, 768, 4, 8, 8, -2048, 320], form: {}, more: { outliers: true, unturned: [1, 3] } },
    { name: "qwen2 and qwen3", header: [256, 768, 4, 4, 2, 2048, 320], form: { bias: true, qk_norm: true, head_dim: 128 }, more: {} },
    { name: "llama, rotated", header: [256, 768, 4, 8, 8, 2048, 320], form: { rotated: true }, more: {} },
    { name: "gpt2", header: [256, 1024, 3, 8, 8, 2048, 320], form: { arch: "gpt2" }, more: {} },
    { name: "neox", header: [256, 1024, 3, 8, 8, -2048, 320], form: { arch: "neox" }, more: { rotary: 8, parallel_residual: true } },
    { name: "qwen35", header: [256, 768, 8, 4, 2, 2048, 320], form: { arch: "qwen35", head_dim: 128, linear: LINEAR, rotated: true }, more: { rotary: 32 } },
    { name: "lfm2", header: [256, 768, 8, 8, 4, -2048, 320], form: { arch: "lfm2", convolution: { layers: "ccaccaca", taps: 3 } }, more: {} },
    { name: "gemma3", header: [256, 768, 6, 4, 1, 2048, 320], form: { arch: "gemma3", head_dim: 128 },
      more: { local: { layers: [0, 1, 2, 3, 4], window: 96, rope_theta: 10000 } } },
    // T363.5: a Qwen3.5 whose matrices are llama.cpp's blocks, a type for each layer of a stack (every type some
    // layer's), the embedding and the classifier each of one, and its output matrix in llama.cpp's order
    { name: "qwen35, of llama.cpp's blocks", header: [256, 768, 8, 4, 2, -2048, 320], more: { rotary: 32 },
      form: { arch: "qwen35", head_dim: 128, linear: LINEAR, tiled: true,
        kinds: { ...Object.fromEntries(Object.entries({ wq: 2, wg: 2, wk: 2, wv: 2, wo: 2, wqkv: 6, wz: 6, wout: 6, w1: 8, w2: 8, w3: 8 })
          .map(([stack, layers], at) => [stack, Array.from({ length: layers }, (_, l) => BLOCKS[(at + 3 * l) % BLOCKS.length])])), token_embedding_table: "IQ1_M", wcls: "IQ3_S" } } },
  ];
  // (the facts of these families a second time, below: [RoPE turns its layers, LayerNorms, a gated FFN]; a family that
  // is added without a line here is held to Python's plan alone)
  const SECOND = { llama: [true, false, true], "qwen2 and qwen3": [true, false, true], "llama, rotated": [true, false, true], gpt2: [false, true, false],
    neox: [true, true, false], qwen35: [true, false, true], lfm2: [true, false, true], gemma3: [true, false, true],
    "qwen35, of llama.cpp's blocks": [true, false, true] };
  const cases = families.flatMap((family) => ["float32", "float16", "int8", "int6", "ternary"].map((dtype) => ({ ...family, dtype })));
  const plans = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", [path.join(root, "tests/engine_plans.py"), root],
    { input: JSON.stringify(cases), maxBuffer: 1 << 28, env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).toString());
  assert.equal(plans.length, cases.length);

  // what Python writes: every model has the same keys (a key that only some had would be undefined for the others)
  const keys = new Set(Object.keys(plans[0])), tensors = new Set(), entryKeys = new Set(), derived = new Set(), partKeys = new Set(), stored = new Set();
  plans.forEach((plan, index) => {
    const which = `${cases[index].name}, ${cases[index].dtype}`;
    assert.deepEqual(sorted(Object.keys(plan)), sorted(keys), `the plan of ${which} has other keys than the plan of ${cases[0].name}, ${cases[0].dtype}`);
    for (const [name, entry] of Object.entries(plan.tensors)) {
      tensors.add(name);
      Object.keys(entry).forEach((key) => entryKeys.add(key));
      // (T363.5: "layers" is the key of a stack with a kind for each layer alone, which says "layers" for its kind)
      assert.deepEqual(sorted(Object.keys(entry).filter((key) => key !== "layers")), sorted(Object.keys(Object.values(plans[0].tensors)[0])), `the tensor ${name} of ${which} has other keys than the others`);
      assert.equal(entry.kind === "layers", "layers" in entry, `the tensor ${name} of ${which} says its layers where its kind is "layers", and only there`);
      for (const part of entry.layers ?? [entry]) stored.add(part.kind);
      stored.add(entry.kind);
      for (const part of entry.layers ?? []) Object.keys(part).forEach((key) => partKeys.add(key));
    }
    Object.keys(plan.derived).forEach((name) => derived.add(name.replace(/^signs\.\d+$/, "signs.<width>")));
  });
  ok(`Python's plan has the same ${keys.size} keys for ${plans.length} made-up models (${families.length} of the layouts' families in five dtypes), ${tensors.size} tensors among them`);

  // T368.2: and the types say the same (src/runtime/contracts/plan.ts, which whatever is TypeScript reads the plan by; the
  // form's two kinds of layers are form.ts's): the keys, the tensors' names, a tensor's keys and the ways it is stored,
  // the derived tables, a layer's facts and its kinds, the widths
  {
    const typed = contract("plan.ts"), form = contract("form.ts");
    const each = (pick) => new Set(plans.flatMap(pick));
    const inside = (pick) => each((plan) => Object.keys(pick(plan) ?? {}));
    typedAs("contracts/plan.ts's Plan, to Python's plan", typed.keysOf("Plan"), keys);
    typedAs("contracts/plan.ts's TensorName, to the tensors Python places", typed.literalsOf("TensorName"), tensors);
    typedAs("contracts/plan.ts's Tensor, to a tensor of Python's plan", typed.keysOf("Tensor"), entryKeys);
    typedAs("contracts/plan.ts's Stored, to the ways Python's tensors and their layers are stored", typed.literalsOf("Stored"), stored);
    typedAs("contracts/plan.ts's Part, to a layer of a stack of Python's plan", typed.keysOf("Part"), partKeys);
    typedAs("contracts/plan.ts's Derived, to the tables Python computes", typed.keysOf("Derived"), new Set([...derived].map((name) => name.replace("<width>", "<number>"))));
    typedAs("contracts/plan.ts's Layer, to a layer's facts", typed.keysOf("Layer"), each((plan) => plan.layers.flatMap(Object.keys)));
    typedAs("contracts/plan.ts's LayerKind, to the kinds of Python's layers", typed.literalsOf("LayerKind"), each((plan) => plan.layers.map((layer) => layer.kind)));
    typedAs("contracts/plan.ts's Widths, to the plan's widths", typed.keysOf("Widths"), inside((plan) => plan.widths));
    typedAs("contracts/plan.ts's LinearWidths, to the widths of a linear-attention layer", typed.keysOf("LinearWidths"), inside((plan) => plan.widths.linear));
    typedAs("contracts/form.ts's Linear, to the plan's linear", form.keysOf("Linear"), inside((plan) => plan.linear));
    typedAs("contracts/form.ts's Convolution, to the plan's convolution", form.keysOf("Convolution"), inside((plan) => plan.convolution));
  }

  // what forward.js reads
  const trees = treesOf(folder("forward.js"));
  const readKeys = new Set(), readTensors = new Set(), readEntry = new Set(), readDerived = new Set();
  // (toJs: the PyProxy's own method, where the plan is Python's object still)
  const NOT_KEYS = new Set(["toJs"]);
  // an expression that is a tensor's entry: `t` (the name these files give one), `T.<name>`, `tensors.<name>`, `tensors[...]`
  // and the two names an entry is kept under where it is read (the embedding's and the classifier's tables)
  const isTable = (node) => node.type === "Identifier" && ["T", "tensors"].includes(node.name);
  const isEntry = (node) => (node.type === "Identifier" && ["t", "embedding", "classifier"].includes(node.name)) ||
    ((node.type === "MemberExpression" || node.type === "OptionalMemberExpression") && isTable(node.object));
  for (const { program } of trees) {
    walk(program, (node) => {
      if (member(node) && node.object.type === "Identifier" && node.object.name === "plan" && !NOT_KEYS.has(node.property.name)) readKeys.add(node.property.name);
      if (node.type === "VariableDeclarator" && node.id.type === "ObjectPattern" && node.init?.type === "Identifier" && node.init.name === "plan") taken(node.id).forEach((key) => readKeys.add(key));
      // the tensors' names: matrix("wq"), floats("bq"), T.wo, tensors.wcls, and the lists a model on the GPU alone is cut by
      if (node.type === "CallExpression" && node.callee.type === "Identifier" && ["matrix", "floats"].includes(node.callee.name) && node.arguments[0]?.type === "StringLiteral") {
        readTensors.add(node.arguments[0].value);
        readDerived.add(node.arguments[0].value);
      }
      if (member(node) && isTable(node.object)) readTensors.add(node.property.name);
      if (node.type === "VariableDeclarator" && ["LAYER_MATRICES", "GPU_ALONE"].includes(node.id.name)) {
        walk(node.init, (inner) => { if (inner.type === "StringLiteral") readTensors.add(inner.value); });
      }
      if (member(node) && isEntry(node.object)) readEntry.add(node.property.name);
      if (node.type === "TemplateLiteral" && node.quasis[0].value.cooked === "signs.") readDerived.add("signs.<width>");
    });
  }
  // T375.2: forward.ts reads every layer's facts and the widths (T359.5's keys) and the two facts of the architecture
  // from the plan. The architecture's name is read by nothing of the engine from then on: it stays in the plan for a
  // person who looks at one
  const unread = { arch: "no engine reads an architecture's name (T375.2): its facts are layer_norm, gated_ffn, the layers' and the tensors'" };
  held("the plan's keys, Python to forward.js", keys, readKeys, unread);
  // (a name forward.js asks for that no model of these has is a tensor of no architecture: floats() answers 0 for it)
  // (T261.1: a Gemma 3's two norms behind are placed and read by no engine yet: Python lets go of the engine it made
  // of such a model's plan and refuses the model (engine/model.py), until T261.2 gives forward.ts its layers)
  const later = "the page's engine has not a Gemma 3's layers until T261.2, and Python refuses the model until then";
  held("the tensors' names, Python to forward.js", tensors, readTensors, { rms_att_post_weight: later, rms_ffn_post_weight: later });
  // (what else is read off those names: a table kept under `embedding` is also the GPU's plan's, { rows, n, ... })
  const entryRead = new Set([...readEntry].filter((key) => entryKeys.has(key) || !["rows", "n", "six", "ternary", "at", "slice", "length"].includes(key)));
  held("a tensor's keys, Python to forward.js", entryKeys, entryRead);
  held("the derived tables' names, Python to forward.js", derived, new Set([...readDerived].filter((name) => derived.has(name) || !tensors.has(name))));

  // T359.5, T375.2, T375.5: the engine reads the layers' facts and the widths from the plan; what sizes a model before
  // there is a plan (footprint(), gpuBytes(), gpuOnlyUnfit(): the worker as a conversion begins, /benchmark/, the tools)
  // has only its header and its form, and works the same out there: which layer is of which kind and a kind's widths
  // by the kinds of forward/kinds.ts (layers(), widths()), and what the architecture's name says by its line of
  // forward/architectures.ts, a copy of Python's table (engine/layout.py's LAYOUTS) that a page reads before Pyodide
  // is there. Both are held to Python's plans here: sizes.ts's Sizes of the form is the Sizes of the plan, and the
  // table's line is what the plan's tensors, layers and facts say. PYTHON IS THE OWNER: where one of these fails, the
  // TypeScript side is put right (the kind's layers() or widths(), or the architecture's line).
  // And whether RoPE turns a layer, a second time: of the ones that attend, none of a model without RoPE's tables, and
  // not the ones left alone
  const { sizesOfForm, sizesOfPlan } = await import(runtimeUrl("forward/sizes.js"));
  const { ARCHITECTURES } = await import(runtimeUrl("forward/architectures.js"));
  const { LAYER_KINDS } = await import(runtimeUrl("forward/kinds.js"));
  const lines = new Set();
  plans.forEach((plan, index) => {
    const { name, dtype, header, form } = cases[index], which = `${name}, ${dtype}`;
    const formed = sizesOfForm(header, form, LAYER_KINDS), planned = sizesOfPlan(plan);
    const fix = "Python's plan is the owner: put src/runtime/forward/ right";
    assert.equal(plan.layers.length, plan.n_layers, `${which}: a fact for every layer`);
    assert.deepEqual(formed.kinds, planned.kinds, `${which}: the kinds of the layers, as the kinds of forward/kinds.ts say of the form (layers()) and as the plan says (${fix})`);
    assert.deepEqual(formed.widths, planned.widths, `${which}: the widths, as worked out of the form (the kinds' widths(), sizes.ts) and as the plan says (${fix})`);
    assert.deepEqual({ ...formed, numbers: null }, { ...planned, numbers: null }, `${which}: what sizes the model, of the form and of the plan (${fix})`);
    for (const kind of Object.keys(LAYER_KINDS)) {
      assert.deepEqual(formed.numbers[kind] ?? null, planned.numbers[kind] ?? null, `${which}: the numbers of the layers of the kind ${kind}, under the kind's name in the form and in the plan`);
    }
    // a layer's place is its place among the layers of its kind
    const places = {};
    // T375.5: the architecture's line of the copy, against what this plan says of each of its facts
    const said = { positions: "positions" in plan.tensors, gate: "wg" in plan.tensors, rope: plan.layers.some(({ rope }) => rope),
      gatedFfn: plan.gated_ffn, layerNorm: plan.layer_norm };
    plan.layers.forEach((layer, l) => {
      assert.equal(layer.place, (places[layer.kind] = (places[layer.kind] ?? -1) + 1), `${which}: the place of layer ${l} among the layers of the kind ${layer.kind}`);
      assert.equal(layer.rope, layer.kind === "attention" && said.rope && !plan.unturned.includes(l), `${which}: whether RoPE turns layer ${l}`);
    });
    // (T375.2: the facts of the families here, a second time: a GPT-2 has no RoPE, and LayerNorms and GPT-2's FFN are a
    // GPT-2's and a GPT-NeoX's)
    if (Object.hasOwn(SECOND, name)) assert.deepEqual([said.rope, plan.layer_norm, plan.gated_ffn], SECOND[name], `${which}: RoPE, the kind of the norms and of the FFN`);
    assert.deepEqual(ARCHITECTURES[plan.arch], said, `${which}: src/runtime/forward/architectures.ts's line of ${JSON.stringify(plan.arch)} is not what Python's plan of that architecture says ` +
      "(a table of positions and a gate of the attention's output by the tensors, RoPE by the layers, gated_ffn, layer_norm). engine/layout.py's LAYOUTS is the owner: put architectures.ts right");
    // (and what footprint() makes of rope: a float32 file of a model with RoPE has the tables, and Python makes them otherwise)
    assert.equal("freq_cis_real" in plan.tensors, said.rope && dtype === "float32", `${which}: the RoPE tables are the file's`);
    assert.equal("freq_cis_real" in plan.derived, !(said.rope && dtype === "float32"), `${which}: the RoPE tables are Python's`);
    lines.add(plan.arch);
  });
  assert.deepEqual(sorted(lines), sorted(Object.keys(ARCHITECTURES)), "every line of forward/architectures.ts is held to a plan of Python's (a new architecture is a family of this check's too)");
  ok(`what sizes ${plans.length} models is the same of the form (sizes.ts, the kinds' layers() and widths()) as of Python's plan, and the ${lines.size} lines of forward/architectures.ts are what Python's plans say of each architecture`);

  // tests/plans.mjs's planOf(): the plan memory-check, gpu-alone-check and the net's calls hand createForward(). T375.2:
  // Python's own (forward_plan(), by the native Python) with a check's settings over it, so its keys are Python's
  const { plansOf, planOf, FORM } = plansFor(root);
  const [placed] = plansOf([{ header: cases[0].header, form: FORM, dtype: "int8" }]);
  held("the plan's keys, Python to tests/plans.mjs's planOf()", keys, new Set(Object.keys(planOf(placed))));

  // T375.2 (T482): the one plan written by hand, /benchmark/'s made-up model's (src/runtime/benchmark/madeup.ts: the CPU
  // section runs without Pyodide), against the plan Python makes for its header as an int8 Llama: every key, and every
  // value but what the section chooses for itself (kv_start), the bytes of the tables (their names are compared) and
  // the classifier under its second name (Python has the shared table under "wcls" too)
  {
    const { madeUpModel } = await import(runtimeUrl("benchmark/madeup.ts"));
    const made = madeUpModel();
    const [theirs] = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", [path.join(root, "tests/engine_plans.py"), root],
      { input: JSON.stringify([{ header: made.header, dtype: "int8", form: {}, more: {} }]), maxBuffer: 1 << 28,
        env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).toString());
    const { wcls, ...tensors } = theirs.tensors;
    assert.deepEqual(wcls, tensors.token_embedding_table, "Python's plan of a shared classifier has the embedding under both names");
    const ours = { ...made.plan, derived: Object.keys(made.plan.derived).sort(), kv_start: theirs.kv_start };
    const expected = { ...theirs, tensors, derived: Object.keys(theirs.derived).sort() };
    let differing = 0;
    for (const key of sorted(new Set([...Object.keys(ours), ...Object.keys(expected)]))) {
      if (JSON.stringify(ours[key]) !== JSON.stringify(expected[key])) {
        differing++;
        wrong(`benchmark/madeup.ts's plan, ${key}: ${JSON.stringify(ours[key])?.slice(0, 200)}, and Python's plan of the same header says ${JSON.stringify(expected[key])?.slice(0, 200)}`);
      }
    }
    if (!differing) ok(`benchmark/madeup.ts's plan is the plan Python makes for its header: ${Object.keys(expected).length} keys, ${Object.keys(tensors).length} tensors (but kv_start, the tables' bytes and "wcls")`);
  }
}

// ---- 2. forward.js -> the GPU's worker
{
  const writers = treesOf(folder("forward.js"));
  // the plan of { type: "start", memory, plan: { ... } }
  let start;
  for (const { program } of writers) {
    walk(program, (node) => {
      if (node.type !== "ObjectExpression") return;
      const property = (name) => node.properties.find((p) => p.type === "ObjectProperty" && p.key.name === name);
      if (property("type")?.value.value === "start" && property("plan")?.value.type === "ObjectExpression") {
        assert.ok(!start, "two messages of type \"start\" carry a plan");
        start = property("plan").value;
      }
    });
  }
  assert.ok(start, "no { type: \"start\", plan: { ... } } in the files of forward/: tests/plan-keys-check.mjs reads the GPU's plan there");
  const open = returned(functionNamed(writers, "gpuOnlyPlan"));
  const written = new Set([...literalKeys(start), ...literalKeys(open)]);
  const inside = {
    tokens: literalKeys(returned(functionNamed(writers, "gpuTokensPlan"))),
    vectors: literalKeys(returned(functionNamed(writers, "gpuVectors"))),
    words: literalKeys(start.properties.find((p) => p.key?.name === "words").value),
    tables: new Set(),
  };
  // (gpuOnlyPlan()'s tables: `const tables = { classifier, embedding }`)
  walk(functionNamed(writers, "gpuOnlyPlan").body, (node) => { if (node.type === "VariableDeclarator" && node.id.name === "tables") literalKeys(node.init, inside.tables); });
  for (const [name, keys] of Object.entries(inside)) assert.ok(keys.size, `nothing found of what the GPU's plan holds in ${name}`);

  const readers = treesOf(folder("gpu.js"));
  const read = new Set(), readInside = Object.fromEntries(Object.keys(inside).map((name) => [name, new Set()]));
  // a plan: `plan`, or anything's `.plan` (m.plan, model.plan)
  const isPlan = (node) => (node.type === "Identifier" && node.name === "plan") || (member(node) && node.property.name === "plan");
  // one level down: `<plan>.tokens`, `<plan>.vectors`, ... (which of them, or undefined)
  const insideOf = (node) => (member(node) && isPlan(node.object) && node.property.name in inside ? node.property.name : undefined);
  // (`plan.tables ?? plan.tokens`: either)
  const insidesOf = (node) => (node.type === "LogicalExpression" ? [...insidesOf(node.left), ...insidesOf(node.right)] : insideOf(node) ? [insideOf(node)] : []);
  for (const { program } of readers) {
    walk(program, (node) => {
      if (member(node) && isPlan(node.object)) read.add(node.property.name);
      if (member(node) && insideOf(node.object)) readInside[insideOf(node.object)].add(node.property.name);
      // the vectors go up as a list (m.vectors, by the plan's own names) and are read back as `V.<name>`, `m.vectors.<name>`
      // or by a name in a text (added(to, "bq")): a property is a read, and so is a text that is one of the written names
      if (member(node) && ((node.object.type === "Identifier" && node.object.name === "V") || (member(node.object) && node.object.property.name === "vectors"))) readInside.vectors.add(node.property.name);
      if (node.type === "StringLiteral" && inside.vectors.has(node.value)) readInside.vectors.add(node.value);
      if (node.type === "VariableDeclarator" && node.id.type === "ObjectPattern" && node.init) {
        if (isPlan(node.init)) taken(node.id).forEach((key) => read.add(key));
        for (const name of insidesOf(node.init)) taken(node.id).forEach((key) => readInside[name].add(key));
      }
    });
  }
  held("the GPU's plan, forward.js to the GPU's worker", written, read);
  // (tables and tokens are read through one expression, `plan.tables ?? plan.tokens`: a key of either is a key of both to the reader)
  const tables = new Set([...readInside.tables].filter((key) => inside.tables.has(key)));
  const tokens = new Set([...readInside.tokens].filter((key) => inside.tokens.has(key) || !inside.tables.has(key)));
  held("the GPU's plan's tokens", inside.tokens, tokens);
  held("the GPU's plan's tables (a model on the GPU alone)", inside.tables, tables);
  held("the GPU's plan's words", inside.words, readInside.words);
  // (what is read off a vector itself, { at, size }, is no name of one)
  held("the GPU's plan's vectors", inside.vectors, new Set([...readInside.vectors].filter((key) => !["at", "size"].includes(key))));

  // T368.2: and the types say the same (src/runtime/contracts/gpu/protocol.ts): the two plans, what they hold one level
  // down, and the messages either side posts, by their kinds and by the keys of each
  const typed = contract("gpu/protocol.ts");
  typedAs("contracts/gpu/protocol.ts's Started, to the plan of { type: \"start\" }", typed.keysOf("Started"), literalKeys(start));
  typedAs("contracts/gpu/protocol.ts's Opened, to gpuOnlyPlan()'s plan", typed.keysOf("Opened"), literalKeys(open));
  for (const [name, type] of [["tokens", "Tokens"], ["vectors", "Vectors"], ["words", "Words"], ["tables", "Tables"]]) {
    typedAs(`contracts/gpu/protocol.ts's ${type}, to the GPU's plan's ${name}`, typed.keysOf(type), inside[name]);
  }
  // (a message's keys that are not in its literal: where they come from)
  const messages = (what, type, writers, to, receivers, more = {}) => {
    const declared = typed.membersOf(type), sent = posted(writers, to);
    typedAs(`contracts/gpu/protocol.ts's ${type}, to the kinds ${what} posts`, new Set(declared.keys()), new Set(sent.keys()));
    typedAs(`contracts/gpu/protocol.ts's ${type}, to the kinds the other side tells apart`, new Set(declared.keys()), told(receivers));
    for (const [kind, keys] of declared) if (sent.has(kind)) typedAs(`contracts/gpu/protocol.ts's ${type}, the keys of "${kind}"`, keys, sent.get(kind).keys, more[kind]);
  };
  // (the model's worker opens the GPU's worker of a model on the GPU alone, with forward.js's plan: worker/weights.js)
  messages("forward.js (and worker/weights.js)", "ToGpu", [...writers, ...treesOf(["worker/weights.js"])], "worker", treesOf(["gpu.js"]));
  messages("the GPU's worker", "FromGpu", readers, "owner", treesOf(["forward/gpuside.js"]),
    { ready: Object.fromEntries(["form", "attention", "forms", "remembered"].map((key) => [key, "gpu/forms.js's formsSaid(), spread in"])) });
}

console.log(failed ? `plan-keys-check: FAILED (${failed})` : "plan-keys-check: ok");
process.exit(failed ? 1 : 0);
