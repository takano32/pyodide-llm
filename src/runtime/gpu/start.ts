// The model going onto the GPU and leaving it: open() and take() (T.156: a model on the GPU alone, its layers written as
// the checkpoint comes), start() (the weights, the forms chosen, "ready" or "unusable"), stop().
// (T.352: a module of the model's GPU worker, gpu.ts, which a bundler links it into: T.367.3)

import { model, stopping, lost, stopAsked, handed, within, openDevice, describe, unusable, end, whyOf } from "./device.ts";
import { upload, uploadLayers, uploadRest, tablesOn, routed, unwritten } from "./weights.ts";
import { chooseAttention, chooseMatrices, formsSaid } from "./forms.ts";
import { prepare, bindLayers, grow, timeBlocks } from "./block.ts";
import { chooseTokens, tokensSaid } from "./tokenforms.ts";

import type { FromGpu, Opened, Started, ToGpu } from "../contracts/gpu/protocol.ts";

// T.368.6: a stretch of the checkpoint as it is posted (gpu.ts hands it on)
type Stretch = Extract<ToGpu, { type: "weights" }>;

let starting = false;

// ---- T.156: a model on the GPU alone. open() makes the device and every buffer of the layers' matrices before a byte
// of them comes (plan: src/runtime/forward/alone.ts's gpuOnlyPlan(): each matrix's rows and length, and where a layer's values and scales
// start in the checkpoint; the layers are joined for a token, as a model on the GPU alone always runs its steps here),
// and take() writes each stretch of them to its buffer as the worker posts it (routes: [start, end) in the checkpoint,
// the buffer and the offset in it). flow[0] counts the bytes on the GPU (the worker waits on it: a disk read faster
// than the GPU takes it would pile up in the messages). What comes before the buffers are made waits (backlog). A
// failure (no adapter, a buffer the device refused) takes the rest as if written, and start() says it
let opening: Promise<void> | null = null, opened = false, failure: string | null = null, flow: BigInt64Array | null = null;
// the bytes that open() put up, for start() to say with the rest
let openedBytes = 0;
const backlog: Stretch[] = [];
function open(plan: Opened, shared: SharedArrayBuffer) {
  starting = true;
  flow = new BigInt64Array(shared);
  opening = (async () => {
    try {
      if (!(await openDevice({ ...plan, tokens: true }, (reason) => { failure = reason; }))) {
        failure ??= "the GPU's worker was stopped";
        return;
      }
      openedBytes = await uploadLayers(model!);
      // T.210: the tables too, which nothing holds in the shared memory either (T.626: as this plan says, which start()'s has not)
      openedBytes += tablesOn(model!, plan.tables);
      if (stopping) failure ??= "the GPU's worker was stopped";
    } catch (error) {
      failure = whyOf(error);
    } finally {
      opened = true;
      starting = startAsked;  // (a start() waiting on this is still starting)
      for (const data of backlog.splice(0)) take(data);
      if (stopping) end();
    }
  })();
}
// the bytes at offset of the checkpoint (bytes: a Uint8Array of their own) onto the buffers whose routes they fall in
// (weights.ts's routed()), and flow[0] counted once the GPU has them
function take(data: Stretch) {
  if (!opened) return void backlog.push(data);
  const { offset, bytes } = data;
  if (!failure) {
    try {
      routed(model!, offset, bytes);
    } catch (error) {
      failure = whyOf(error);
    }
  }
  const count = BigInt(bytes.length), done = () => {
    Atomics.add(flow!, 0, count);
    Atomics.notify(flow!, 0);
  };
  if (failure) done();
  else model!.device.queue.onSubmittedWorkDone().then(done, done);
}
let startAsked = false;
async function start(memory: WebAssembly.Memory, plan: Started) {
  const began = performance.now();
  starting = startAsked = true;
  try {
    let bytes: number;
    if (plan.direct) {
      // T.156: opened before the checkpoint came, its layers written as they came (the worker waited for the last)
      await opening;
      if (failure && !stopping) return unusable(`the layers did not go up to the GPU (${failure})`);
      if (stopping || !model) return end();
      if (unwritten(model)) return unusable("the layers' bytes did not all come to the GPU");
      handed(memory, plan);
      bytes = openedBytes + (await uploadRest(model));
    } else {
      if (!(await openDevice(plan))) return;
      handed(memory, plan);
      bytes = await upload(model!);
    }
    // (T.479: the model's record, made by open() or openDevice() and handed its plan above: a stop in the middle of start()
    // ends it at its next step, not before (stop()), so it is the same record to the end)
    const m = model!, { device, adapter, key } = m;
    if (stopping) return end();
    await prepare(m);
    await chooseAttention(m);
    if (stopping) return end();
    await chooseMatrices(m);
    if (stopping) return end();
    bindLayers(m);
    grow(m, Math.max(Math.min(plan.kvStart, plan.seqLen), Math.min(plan.batch, plan.seqLen)));
    await within(device.queue.onSubmittedWorkDone(), "the GPU's work");
    const invalid = await device.popErrorScope(), full = await device.popErrorScope();
    if (stopping) return end();
    if (invalid || full) return unusable(`the GPU did not take the layers (${(invalid ?? full)!.message})`);
    // (not for the page's tests: SwiftShader took more than STEP_MS to time Llama 3.2 1B's shaders, T.147 in CI)
    const blocks = plan.force.quick ? [] : await within(timeBlocks(m), "timing a block");
    if (stopping) return end();
    // T.152: a token and the ones after it, where the model's layers were put up for them (tokensLayout). What fails
    // here leaves the tokens on the CPU and the prompts on the GPU
    // (why not: the layers could not be joined for a token as they went up, or no form of a token is right here)
    let tokensWhy = m.weights.unjoined;
    if (plan.tokens && !tokensWhy) {
      try {
        await chooseTokens(m);
      } catch (error) {
        tokensWhy = whyOf(error);
      }
      if (stopping) return end();
    }
    if (lost) return unusable(lost);
    postMessage({ type: "ready", adapter: describe(adapter), key, bytes, seconds: (performance.now() - began) / 1000,
      ...formsSaid(m), blocks, tokens: tokensSaid(m), tokensWhy: plan.tokens ? tokensWhy ?? null : undefined } satisfies FromGpu);
  } catch (error) {
    unusable(whyOf(error));
  } finally {
    starting = false;
  }
}

function stop() {
  stopAsked();
  if (!starting) end();  // else start() ends at its next step
}

export { open, take, start, stop };
