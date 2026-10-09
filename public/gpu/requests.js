// The requests forward.js waits for in the control area: a block of a prompt, the steps of a generation, the GPU's own
// keys and values (T210), each answered by serve().
// (T352: a module of the model's GPU worker, public/gpu.js, which asks for it with its own ?v=<build>)

const { common, copyIn, narrowIn } =
  await import(new URL(`device.js${new URL(import.meta.url).search}`, import.meta.url));
const { grow, block, keysOut, keysBack } =
  await import(new URL(`block.js${new URL(import.meta.url).search}`, import.meta.url));
const { tokenStep, runTokens } =
  await import(new URL(`tokens.js${new URL(import.meta.url).search}`, import.meta.url));

// A request for count steps from token at pos (public/forward/gpuside.js's generateMany): the keys and values of positions from to
// pos - 1 up from forward.js's cache first (cache: its addresses of the keys and the values, its capacity and the
// bytes of a position; half: float16 as here, else float32, narrowed on the way up: T160, a grouped-query model's),
// the state (token, pos, the end of the history and its length), the settings and a random number a step; the ids into
// plan.tokens.ids ([sampled, id, id, ...] and, after plan.tokens.most ids, the State's not_finite word: T219, the step
// after the sampled ones was refused, its logits not finite; a stop token last where one came), and the keys and values of the positions
// sampled in float16 into plan.staging as a prompt's block's ([keys, values][layer][plan.batch positions]), which
// forward.js puts into its cache (widened where it is float32), where it still wants the answer. T210: a model on the
// GPU alone (cache null) has no cache there: nothing goes up, and nothing but the ids comes back
function generate({ serial, count, pos, from, token, history, length, cache, settings, randoms }) {
  serve(serial, async (wanted) => {
    const m = common.model, { plan, wgsl, gen: g } = m, kvDim = plan.kvHeads * plan.headSize, kvRow = kvDim * 2;
    if (!g?.form) throw new Error("no tokens on this GPU");
    if (pos + count > m.cache.capacity) grow(m, pos + count);
    const at = (block, l, p) => block + l * cache.capacity * cache.row + p * cache.row;
    for (let l = 0; cache && l < plan.layers; l++) {
      for (const [block, target] of [[cache.keys, m.cache.keys[l]], [cache.values, m.cache.values[l]]]) {
        if (cache.half) copyIn(m, target, at(block, l, from), (pos - from) * kvRow, from * kvRow);
        else narrowIn(m, target, at(block, l, from), (pos - from) * kvDim, from * kvRow);
      }
    }
    const out = await runTokens(m, tokenStep(m, g.form, pos + count), { count, pos, keep: Boolean(cache),
      state: wgsl.samplingState({ token, pos, history, length }), settings: wgsl.samplingSettings({ vocab: g.vocab, ...settings }),
      randoms: Float32Array.from({ length: count }, (_, i) => randoms[i] ?? 0) });
    if (!wanted()) return;
    for (let side = 0; cache && side < 2; side++) {
      for (let l = 0; l < plan.layers; l++) {
        new Uint8Array(m.memory.buffer, plan.staging + (side * plan.layers + l) * plan.batch * kvRow, out.sampled * kvRow)
          .set(out.kv[side][l].subarray(0, out.sampled * kvRow));
      }
    }
    const ids = new Int32Array(m.memory.buffer, plan.tokens.ids, 2 + g.most);
    ids[0] = out.sampled;
    ids.set(out.ids.subarray(0, out.sampled), 1);
    ids[1 + g.most] = out.notFinite;
  });
}

// ---- a block of a prompt (T210: tokens, the ids of a model on the GPU alone, which embeds them here)
function prompt({ serial, count, pos, tokens }) {
  serve(serial, (wanted) => block(common.model, count, pos, wanted, false, tokens));
}
// The answer to a request (a block of a prompt, T152: the steps of a generation), in the control area: work(wanted)
// runs while words.beat counts up, then words.failed and words.done = serial, where forward.js still waits for this
// request (wanted: T147, it gave up, and the memory may soon be another model's)
async function serve(serial, work) {
  const { memory, plan } = common.model;
  const words = new Int32Array(memory.buffer, 0, Math.max(...Object.values(plan.words)) + 1);
  // the model's worker waits: this says that the work goes on, however long the GPU takes (a software adapter)
  const beat = setInterval(() => Atomics.add(words, plan.words.beat, 1), 250);
  const wanted = () => Atomics.load(words, plan.words.wanted) === serial;
  let failed = 1;
  try {
    if (common.lost) throw new Error(common.lost);
    await work(wanted);
    if (common.lost) throw new Error(common.lost);
    failed = 0;
  } catch (error) {
    postMessage({ type: "failed", reason: String(error?.message ?? error) });
  } finally {
    clearInterval(beat);
    if (wanted()) {
      Atomics.store(words, plan.words.failed, failed);
      Atomics.store(words, plan.words.done, serial);
      Atomics.notify(words, plan.words.done);
    }
  }
}

// T210: the GPU's own keys and values of count positions (plan.batch at most) from pos into plan.staging, as a block
// writes them back: a model on the GPU alone keeps none in the shared memory, and the tests read them so
// (public/forward/engine.js's keysAndValues)
function keysOf({ serial, count, pos }) {
  serve(serial, async (wanted) => {
    const encoder = common.model.device.createCommandEncoder();
    keysOut(common.model, encoder, pos, count);
    common.model.device.queue.submit([encoder.finish()]);
    await keysBack(common.model, wanted);
  });
}

export { generate, prompt, keysOf };
