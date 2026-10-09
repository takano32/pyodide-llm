// The GPU section of /benchmark/ (T134; T94's stage 0 until then, at /gpu-test/): what WebGPU gives this device,
// measured in a worker (where a GPU forward pass would run). The page (src/pages/benchmark.astro) asks for one step at
// a time and shows what comes back, and ends the worker after the section; nothing here touches the model page.
//
//   { step: "info" }                      the adapter, its limits and features, WGSL's language features
//   { step: "check" }                     the int8 shaders against JavaScript on small matrices (T146: the tiled ones
//                                         too, with their edges)
//   { step: "bandwidth", shape }          GB/s of one int8 matrix times a vector, by every shader of matVecShaders()
//                                         (T134's two and T149's from llama.cpp and ONNX Runtime), and the CPU's
//   { step: "token", model, kind, sample }
//                                         a whole token's work of a model's shapes (every layer's matrices, a few small
//                                         dispatches, the classifier, the logits read back), ms per token. sample: the
//                                         most likely token found on the GPU, and only its id read back instead of every
//                                         logit
//   { step: "layer" }                     T150: one layer of a token of Llama 3.2 1B's width, as its fourteen separate
//                                         steps and fused into five dispatches (shaders.js's fusedMatVec), ms a layer;
//                                         T175: the same on ORT's DP4A for small M (fusedDp4aMatVec), its vector
//                                         quantized before each matrix
//   { step: "layer steps" }               T202: where the time of a layer goes: each step of the fused forms a token
//                                         would run here alone, the matrices alone, a dispatch that does next to
//                                         nothing, and the whole layer, timed in turn, ms each (T208: the forms the
//                                         layer table's fastest, and the whole layer by timestamps where they are)
//   { step: "generate" }                  T151: tokens generated on the GPU (the sampling too), each read back as it
//                                         comes against 4, 8 and 16 a submission read back once, ms a token; T191:
//                                         each also with the sampling in chunks (many workgroups), and the sampling
//                                         alone both ways
//   { step: "overhead" }                  what a token costs besides the weights: 240 empty dispatches, a submission
//                                         with and without waiting for it, reading back 4 bytes and all the logits
//   { step: "prompt", counts }            the tokens of a prompt through the matrices all at once (matrix × matrix,
//                                         T135's first candidate), on the made-up model of the CPU section's shape,
//                                         by T135's batched shader and T146's tiled ones, with the GFLOPS of each
//   { step: "ceilings" }                  T168: the device's ceilings: f32 and f16 multiply-adds (GFLOPS), dot4I8Packed
//                                         (GOPS), reading the workgroup's memory and a storage buffer (GB/s), each a
//                                         loop of that alone (shaders.js), for the share of them the prompt's shaders reach
//   { step: "bridge", memory, rounds }    the round trip of a worker that waits with Atomics.wait and this one, which
//                                         answers with Atomics.waitAsync (stage 1's design), in microseconds
//
// The weights are random: only their size and layout matter. int8 in groups of 32 with a float32 scale each, as the
// checkpoints of this project (llama2_numpy's layout), 4 values to a u32.
// T353: this file is the window of the worker: the page starts it and its steps are asked of it, and the tests import the
// pure helpers from it. The steps are the modules of gpu/: device.js (the shaders' module, the device, buffers, a matrix
// bound and run, how a time is taken), halves.js (the float16 of the cache), layerparts.js, layercheck.js and layer.js
// (one layer of a token: its forms, its check, its times), matvec.js (a matrix times a vector, a token's work),
// generate.js and generatecheck.js (generated tokens and the sampling, and their check), check.js (the step "check"),
// ceilings.js and prompt.js. Each is asked for with this file's own ?v=<build>, as shaders.js is, so that all come from
// one deployment, and all at once: one after another's end would add a round trip for each.
const modules = Object.fromEntries(["device", "halves", "layerparts", "layercheck", "matvec", "layer", "generate",
  "generatecheck", "check", "ceilings", "prompt"].map((name) =>
  [name, import(new URL(`gpu/${name}.js${new URL(import.meta.url).search}`, import.meta.url))]));
// a module worker's port opens at its first await, and a message that comes before onmessage is set is lost (T109):
// what comes while the modules do is kept here, and handed to the steps in order at the end of this file
const early = [];
onmessage = (event) => early.push(event);
const { load, info } = await modules.device;
const { toHalf, fromHalf, heldHalves, halvesSaid, farthest } = await modules.halves;
const { layerReference, layerCheckData, judgeLayer } = await modules.layercheck;
const { bandwidth, token } = await modules.matvec;
const { layer, layerSteps } = await modules.layer;
const { generate } = await modules.generate;
const { check } = await modules.check;
const { overhead, ceilings } = await modules.ceilings;
const { prompt } = await modules.prompt;

// (these are pure: tests/gpu-choice-check.mjs holds the rounding of the cache to what a device may choose, and
// tests/layer-check.mjs holds the layer check's judgement to devices that are JavaScript too)
export { toHalf, fromHalf, heldHalves, halvesSaid, farthest, layerCheckData, judgeLayer, layerReference, load };

// ---- the bridge: a worker that waits (Atomics.wait, as the model's worker would while Python calls forward())
// and this one, which must not block (it awaits the GPU): Atomics.waitAsync where there is one
async function bridge(memory, rounds) {
  const words = new Int32Array(memory);
  const hasWaitAsync = typeof Atomics.waitAsync === "function";
  if (!hasWaitAsync) return { waitAsync: false };
  // words[0]: the request counter (the waiter adds 1), words[1]: the answer counter (this one sets it)
  let answered = 0;
  while (answered < rounds) {
    const seen = Atomics.load(words, 0);
    if (seen === answered) {
      const result = Atomics.waitAsync(words, 0, seen);
      if (result.async) await result.value;
      continue;
    }
    answered = seen;
    Atomics.store(words, 1, answered);
    Atomics.notify(words, 1);
  }
  return { waitAsync: true };
}

onmessage = async ({ data }) => {
  try {
    await load();
    let result;
    if (data.step === "info") result = await info();
    else if (data.step === "check") result = await check();
    else if (data.step === "bandwidth") result = await bandwidth(data.shape);
    else if (data.step === "token") result = await token(data.model, data.kind, data);
    else if (data.step === "layer") result = await layer();
    else if (data.step === "layer steps") result = await layerSteps();
    else if (data.step === "generate") result = await generate();
    else if (data.step === "overhead") result = await overhead();
    else if (data.step === "prompt") result = await prompt(data.counts);
    else if (data.step === "ceilings") result = await ceilings();
    else if (data.step === "bridge") result = await bridge(data.memory, data.rounds);
    postMessage({ step: data.step, result });
  } catch (error) {
    postMessage({ step: data.step, error: String(error?.message ?? error) });
  }
};
for (const event of early.splice(0)) onmessage(event);
