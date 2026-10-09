// gpu.js (T135, T147, T148): the model page's GPU, in a worker of its own. forward.js (in the model's worker) makes it
// wherever the worker has WebGPU and the model is one it takes (T148: by default, no option), and hands it the blocks
// of a prompt (up to plan.batch tokens) one at a time that it finds faster here than on the CPU, waiting in
// Atomics.wait for the answer: the model's worker cannot wait for a promise while Python calls forwardMany(), and this
// one can (T94's design; T134's bridge measured 12 to 95 µs a round trip).
//
//   { type: "start", memory, plan }  the layers of the model onto the GPU, read from the shared memory of forward.js
//                                    (plan: what forward.js says of them, addresses in that memory). Says
//                                    { type: "progress" } after every step (a layer's weights, a shader compiled,
//                                    checked, timed: STEP_MS each at most, T147), then { type: "ready",
//                                    adapter, key, bytes, seconds, form, attention, forms, remembered, blocks } or
//                                    { type: "unusable", reason }. T148: a fallback adapter (the CPU in the GPU's
//                                    place) is refused before anything is compiled, unless plan.force.fallback
//                                    (tests); plan.remembered ({ key, matrices, attention }: what the page kept of an
//                                    earlier visit) spares the timing of the matrices' shaders where key is this
//                                    adapter's and the shader is still right here; blocks: what a whole block of 16
//                                    and of 64 tokens takes here (forward.js weighs the GPU against the CPU by it)
//   { type: "prompt", serial, count, pos }
//                                    the rows of count tokens at positions pos, pos + 1, ... (embedded by forward.js,
//                                    at plan.rows) through every layer, and their keys and values of every layer into
//                                    plan.staging as float16. The answer is in the control area: words.failed, then
//                                    words.done = serial, while words.beat counts up meanwhile; a failure also says
//                                    { type: "failed", reason }. Nothing is written where words.wanted is no longer
//                                    serial: forward.js gave the request up (T147: a worker that went on late must not
//                                    write into a memory that may hold the next model by then)
//   { type: "tokens", serial, count, pos, from, token, history, length, cache, settings, randoms }
//                                    T152: count steps of a generation from token at pos (the forward pass and the
//                                    sampling of each, SAMPLE's), after the keys and values of positions from to pos - 1
//                                    went up from forward.js's cache; the ids into plan.tokens.ids and the keys and
//                                    values of their positions into plan.staging (see generate() below). The answer is
//                                    in the control area as a prompt's. Where plan.tokens asks for it, "ready" says
//                                    tokens ({ form, ms, forms, remembered, attention, attentions }: the layer of a
//                                    token chosen here and its ms a step, T224: the attention of a token chosen here
//                                    and what each came to) or tokensWhy (why they stay on the CPU)
//   { type: "stop" }                 every buffer and the device let go, and the worker ends; T205: it says
//                                    { type: "ended" } as it does (so does a start that ends as "unusable")
//
// A block goes as one submission: per layer the RMSNorm, the matrices of q, k and v (each weight read once for the
// block's tokens, by the tiled shader chosen on this device: see chooseMatrices), T153: their biases (Qwen2) and the
// norms of every head of q and k (Qwen3) where the model has them, RoPE with the keys and values into the
// GPU's own cache (float16), the attention (llama.cpp's flash attention with tiles: every token sees the positions up to
// its own), the output matrix added to the residual, the RMSNorm, the gate and the up matrices, SwiGLU, the down matrix
// added. T154: GPT-2 and GPT-NeoX (plan.layerNorm) have LayerNorm with a bias for either norm, a bias after every
// matrix, no gate (the up matrix is w1, then GELU), GPT-2 no RoPE at all (plan.turned 0: its learned positions are in
// the rows forward.js embeds) and GPT-NeoX RoPE on a part of every head; GPT-NeoX's parallel residual (plan.parallel)
// has the FFN's norm read the layer's input before the output matrix adds to it, into a buffer of its own (xn). The
// last layer stops at its keys and values: nothing of a prompt's token after them is used. Then the block's
// keys and values of every layer are copied out and read back. The activations are float32 (quantized to 8 bits first
// for the packed shaders, as the CPU's matmul_q8 takes them), the weights int8 widened or multiplied as int8. T155:
// int6 weights (T98) are widened to int8 once, as they go onto the GPU; a model in a 64-bit memory (T101) goes as
// one in a 32-bit memory (its addresses are Numbers); a matrix larger than a buffer the device binds, in pieces of rows.
// T232: ternary weights (T230) go up as the checkpoint holds them, two bits a weight and a scale a group of 128, and
// are multiplied by the packed shaders alone (shaders.js's TERNARY_PACKED: the codes unpacked to int8 where the
// shader loads them); a device without the packed int8 dot keeps such a model on the CPU.
//
// T352: this file is the window of the worker: forward.js starts it and its messages come here. What it does is the
// modules of gpu/: device.js (the shaders' module, what the modules share, the device, buffers), weights.js (the
// weights onto the GPU), forms.js (the prompt's shaders checked, timed and chosen), block.js (a block of a prompt),
// tokenattention.js, tokens.js and tokenforms.js (a generated token's attention, its steps, and their forms checked,
// timed and chosen), requests.js (what forward.js waits for in the control area) and start.js (the model going onto
// the GPU and leaving it). Each is asked for with this file's own ?v=<build>, as shaders.js is, so that all come from
// one deployment, and all at once: one after another's end would add a round trip for each.
const modules = Object.fromEntries(["device", "weights", "forms", "block", "tokenattention", "tokens", "tokenforms",
  "requests", "start"].map((name) =>
  [name, import(new URL(`gpu/${name}.js${new URL(import.meta.url).search}`, import.meta.url))]));
// (the shaders too, as this file asked for them where it began before it was divided: gpu/device.js's own import of
// them is this one, and does not wait for device.js to come)
import(new URL(`shaders.js${new URL(import.meta.url).search}`, import.meta.url));
// A module worker's port opens at its first await, and a message that comes before onmessage is set is lost (T109):
// what comes while the modules do is kept here, and handed to the receiver in order at the end of this file
const early = [];
onmessage = (event) => early.push(event);
const { heldFloats } = await modules.tokenforms;
const { generate, prompt, keysOf } = await modules.requests;
const { open, take, start, stop } = await modules.start;

// (pure: tests/gpu-choice-check.mjs holds the rounding of the keys and values to what a device may choose)
export { heldFloats };

onmessage = ({ data }) => {
  if (data.type === "open") open(data.plan, data.flow);
  else if (data.type === "weights") take(data);
  else if (data.type === "start") start(data.memory, data.plan);
  else if (data.type === "prompt") prompt(data);
  else if (data.type === "tokens") generate(data);
  else if (data.type === "keys") keysOf(data);
  else if (data.type === "stop") stop();
};
for (const event of early.splice(0)) onmessage(event);
