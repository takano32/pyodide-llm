// The model going onto the GPU and leaving it: open() and take() (T156: a model on the GPU alone, its layers written as
// the checkpoint comes), start() (the weights, the forms chosen, "ready" or "unusable"), stop().
// (T352: a module of the model's GPU worker, gpu.js, which a bundler links it into: T367.3)

import { model, stopping, lost, stopAsked, handed, within, openDevice, describe, unusable, end } from "./device.js";
import { upload, uploadLayers, uploadRest, tablesOn } from "./weights.js";
import { chooseAttention, chooseMatrices } from "./forms.js";
import { prepare, bindLayers, grow, timeBlocks } from "./block.js";
import { chooseTokens } from "./tokenforms.js";

let starting = false;

// ---- T156: a model on the GPU alone. open() makes the device and every buffer of the layers' matrices before a byte
// of them comes (plan: src/runtime/forward/alone.js's gpuOnlyPlan(): each matrix's rows and length, and where a layer's values and scales
// start in the checkpoint; the layers are joined for a token, as a model on the GPU alone always runs its steps here),
// and take() writes each stretch of them to its buffer as the worker posts it (routes: [start, end) in the checkpoint,
// the buffer and the offset in it). flow[0] counts the bytes on the GPU (the worker waits on it: a disk read faster
// than the GPU takes it would pile up in the messages). What comes before the buffers are made waits (backlog). A
// failure (no adapter, a buffer the device refused) takes the rest as if written, and start() says it
let opening = null, opened = false, failure = null, flow = null;
// the bytes that open() put up, for start() to say with the rest
let openedBytes = 0;
const backlog = [];
function open(plan, shared) {
  starting = true;
  flow = new BigInt64Array(shared);
  opening = (async () => {
    try {
      if (!(await openDevice({ ...plan, tokens: true }, (reason) => { failure = reason; }))) {
        failure ??= "the GPU's worker was stopped";
        return;
      }
      openedBytes = await uploadLayers(model, true);
      // T210: the tables too, which nothing holds in the shared memory either
      openedBytes += tablesOn(model);
      if (stopping) failure ??= "the GPU's worker was stopped";
    } catch (error) {
      failure = String(error?.message ?? error);
    } finally {
      opened = true;
      starting = startAsked;  // (a start() waiting on this is still starting)
      for (const data of backlog.splice(0)) take(data);
      if (stopping) end();
    }
  })();
}
// the bytes at offset of the checkpoint (bytes: a Uint8Array of their own) onto the buffers whose routes they fall in.
// writeBuffer takes whole words: the bytes of a word a stretch begins or ends in the middle of wait for the rest of it
// (partial), which another message brings
function take(data) {
  if (!opened) return void backlog.push(data);
  const { offset, bytes } = data;
  if (!failure) {
    try {
      const { routes } = model.weights, end = offset + bytes.length;
      let lo = 0, hi = routes.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (routes[mid][1] <= offset) lo = mid + 1;
        else hi = mid;
      }
      for (let i = lo; i < routes.length && routes[i][0] < end; i++) {
        const [start, stop, target, at] = routes[i];
        let a = Math.max(start, offset) - offset, here = at + (a + offset - start);
        const b = Math.min(stop, end) - offset;
        for (; a < b && here % 4; a++, here++) partialByte(target, here, bytes[a]);
        const whole = (b - a) & ~3;
        if (whole) model.device.queue.writeBuffer(target, here, bytes, a, whole);
        for (let k = a + whole; k < b; k++) partialByte(target, here + (k - a), bytes[k]);
      }
    } catch (error) {
      failure = String(error?.message ?? error);
    }
  }
  const count = BigInt(bytes.length), done = () => {
    Atomics.add(flow, 0, count);
    Atomics.notify(flow, 0);
  };
  if (failure) done();
  else model.device.queue.onSubmittedWorkDone().then(done, done);
}
// a byte of target at the byte offset at, the word it is in written once all four of its bytes are there
function partialByte(target, at, value) {
  const { partial } = model.weights, word = at - (at % 4);
  if (!partial.has(target)) partial.set(target, new Map());
  const words = partial.get(target);
  if (!words.has(word)) words.set(word, { bytes: new Uint8Array(4), count: 0 });
  const held = words.get(word);
  held.bytes[at % 4] = value;
  if (++held.count < 4) return;
  model.device.queue.writeBuffer(target, word, held.bytes);
  words.delete(word);
  if (!words.size) partial.delete(target);
}

let startAsked = false;
async function start(memory, plan) {
  const began = performance.now();
  starting = startAsked = true;
  try {
    let bytes;
    if (plan.direct) {
      // T156: opened before the checkpoint came, its layers written as they came (the worker waited for the last)
      await opening;
      if (failure && !stopping) return unusable(`the layers did not go up to the GPU (${failure})`);
      if (stopping || !model) return end();
      if (model.weights.partial.size) return unusable("the layers' bytes did not all come to the GPU");
      handed(memory, plan);
      bytes = openedBytes + (await uploadRest(model));
    } else {
      if (!(await openDevice(plan))) return;
      handed(memory, plan);
      bytes = await upload(model);
    }
    const { device, adapter, key } = model;
    if (stopping) return end();
    await prepare(model);
    await chooseAttention(model);
    if (stopping) return end();
    await chooseMatrices(model);
    if (stopping) return end();
    bindLayers(model);
    grow(model, Math.max(Math.min(plan.kvStart, plan.seqLen), Math.min(plan.batch, plan.seqLen)));
    await within(device.queue.onSubmittedWorkDone(), "the GPU's work");
    const invalid = await device.popErrorScope(), full = await device.popErrorScope();
    if (stopping) return end();
    if (invalid || full) return unusable(`the GPU did not take the layers (${(invalid ?? full).message})`);
    // (not for the page's tests: SwiftShader took more than STEP_MS to time Llama 3.2 1B's shaders, T147 in CI)
    const blocks = plan.force.quick ? [] : await within(timeBlocks(model), "timing a block");
    if (stopping) return end();
    // T152: a token and the ones after it, where the model's layers were put up for them (tokensLayout). What fails
    // here leaves the tokens on the CPU and the prompts on the GPU
    // (why not: the layers could not be joined for a token as they went up, or no form of a token is right here)
    let tokensWhy = model.weights.unjoined;
    if (plan.tokens && !tokensWhy) {
      try {
        await chooseTokens(model);
      } catch (error) {
        tokensWhy = String(error?.message ?? error);
      }
      if (stopping) return end();
    }
    if (lost) return unusable(lost);
    const g = model.tokens;
    postMessage({ type: "ready", adapter: describe(adapter), key, bytes, seconds: (performance.now() - began) / 1000,
      form: model.forms.form.name, attention: model.forms.attention.name, forms: model.forms.tried, remembered: Boolean(model.forms.form.remembered), blocks,
      tokens: g?.form ? { form: g.form.name, ms: g.ms, forms: g.forms, remembered: g.forms.some((f) => f.remembered), pieces: model.weights.tables.classifier.length,
        attention: g.attention.name, attentions: g.attentions } : null,
      tokensWhy: plan.tokens ? tokensWhy ?? null : undefined });
  } catch (error) {
    unusable(String(error?.message ?? error));
  } finally {
    starting = false;
  }
}

function stop() {
  stopAsked();
  if (!starting) end();  // else start() ends at its next step
}

export { open, take, start, stop };
