// The requests forward.ts waits for in the control area: a block of a prompt, the steps of a generation, the GPU's own
// keys and values (T.210), each answered by serve().
// (T.352: a module of the model's GPU worker, gpu.ts, which a bundler links it into: T.367.3)

import { model, lost } from "./device.ts";
import { roomFor, keysIn, block, keysOut, keysBack, stateIn } from "./block.ts";
import { attendingOf } from "./kinds.ts";
import { tokenStep, runTokens } from "./tokens.ts";
import { samplingState, samplingSettings } from "../shaders/run.ts";
import type { Thrown } from "./device.ts";
import type { FromGpu, ToGpu } from "../contracts/gpu/protocol.ts";

// T.368.6: a request by its kind (gpu.ts hands each on as it came)
type Asked<Kind extends ToGpu["type"]> = Extract<ToGpu, { type: Kind }>;

// A request for count steps from token at pos (src/runtime/forward/gpuside.ts's generateMany): the keys and values of positions from to
// pos - 1 up from forward.ts's cache first (cache: its addresses of the keys and the values, its capacity and the
// bytes of a position; half: float16 as here, else float32, narrowed on the way up: T.160, a grouped-query model's),
// the state (token, pos, the end of the history, its length and how many of its latest were sampled: T.546), the settings and a random number a step; the ids into
// plan.tokens.ids ([sampled, id, id, ...] and, after plan.tokens.most ids, the State's not_finite word: T.219, the step
// after the sampled ones was refused, its logits not finite; a stop token last where one came), and the keys and values of the positions
// sampled in float16 into plan.staging as a prompt's block's ([keys, values][layer][plan.batch positions]), which
// forward.ts puts into its cache (widened where it is float32), where it still wants the answer. T.210: a model on the
// GPU alone (cache null) has no cache there: nothing goes up, and nothing but the ids comes back.
// T.545.2: an LFM2's convolution layers go on from the CPU's rows (none at position 0, as a block), which the CPU holds
// while it waits; after the attending layers' keys and values (by their places) come the rows as the last step left
// them, as a block's come back (cpuSide.stateFromStaging takes them). T.271.2: and a Qwen3.5's linear-attention layers'
// states and rows; fresh, from the GPU's own (block.ts's stateIn). On the GPU alone (cache null) the state is its own
function generate({ serial, count, pos, fresh = false, from, token, history, length, written, cache, settings, randoms }: Asked<"tokens">) {
  serve(serial, async (wanted) => {
    const m = model!, { plan, tokens: g, tokenForm: chosen } = m, kvRow = plan.kvHeads * plan.headSize * 2, layers = attendingOf(plan).length;
    if (!chosen?.form) throw new Error("no tokens on this GPU");
    roomFor(m, pos + count);
    if (cache) keysIn(m, cache, from, pos);
    const encoder = m.device.createCommandEncoder();
    stateIn(m, encoder, pos, fresh || !cache);
    m.device.queue.submit([encoder.finish()]);
    const out = await runTokens(m, tokenStep(m, chosen.form, pos + count), { count, pos, keep: Boolean(cache),
      state: samplingState({ token, pos, history, length, written }), settings: samplingSettings({ vocab: g.vocab, ...settings }),
      randoms: Float32Array.from({ length: count }, (_, i) => randoms[i] ?? 0) });
    if (!wanted()) return;
    for (let side = 0; cache && side < 2; side++) {
      for (let l = 0; l < layers; l++) {
        new Uint8Array(m.memory.buffer, plan.staging + (side * layers + l) * plan.batch * kvRow, out.sampled * kvRow)
          .set(out.kv![side][l].subarray(0, out.sampled * kvRow));
      }
    }
    if (out.state && out.sampled === count) new Uint8Array(m.memory.buffer, plan.staging + 2 * layers * plan.batch * kvRow, out.state.length).set(out.state);
    const ids = new Int32Array(m.memory.buffer, plan.tokens!.ids, 2 + g.most);
    ids[0] = out.sampled;
    ids.set(out.ids.subarray(0, out.sampled), 1);
    ids[1 + g.most] = out.notFinite;
  });
}

// ---- a block of a prompt (T.210: tokens, the ids of a model on the GPU alone, which embeds them here)
function prompt({ serial, count, pos, fresh = false, tokens }: Asked<"prompt">) {
  serve(serial, (wanted) => block(model!, count, pos, wanted, false, tokens, fresh || Boolean(model!.plan.direct)));
}
// The answer to a request (a block of a prompt, T.152: the steps of a generation), in the control area: work(wanted)
// runs while words.beat counts up, then words.failed and words.done = serial, where forward.ts still waits for this
// request (wanted: T.147, it gave up, and the memory may soon be another model's)
async function serve(serial: number, work: (wanted: () => boolean) => Promise<void>) {
  const { memory, plan } = model!;
  const words = new Int32Array(memory.buffer, 0, Math.max(...Object.values(plan.words)) + 1);
  // the model's worker waits: this says that the work goes on, however long the GPU takes (a software adapter)
  const beat = setInterval(() => Atomics.add(words, plan.words.beat, 1), 250);
  const wanted = () => Atomics.load(words, plan.words.wanted) === serial;
  let failed = 1;
  try {
    if (lost) throw new Error(lost);
    await work(wanted);
    if (lost) throw new Error(lost);
    failed = 0;
  } catch (error) {
    postMessage({ type: "failed", reason: String((error as Thrown)?.message ?? error) } satisfies FromGpu);
  } finally {
    clearInterval(beat);
    if (wanted()) {
      Atomics.store(words, plan.words.failed, failed);
      Atomics.store(words, plan.words.done, serial);
      Atomics.notify(words, plan.words.done);
    }
  }
}

// T.210: the GPU's own keys and values of count positions (plan.batch at most) from pos into plan.staging, as a block
// writes them back: a model on the GPU alone keeps none in the shared memory, and the tests read them so
// (src/runtime/forward/engine.ts's keysAndValues)
function keysOf({ serial, count, pos }: Asked<"keys">) {
  serve(serial, async (wanted) => {
    const encoder = model!.device.createCommandEncoder();
    keysOut(model!, encoder, pos, count);
    model!.device.queue.submit([encoder.finish()]);
    await keysBack(model!, wanted);
  });
}

export { generate, prompt, keysOf };
