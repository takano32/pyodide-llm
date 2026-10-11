// forward/gpuside.js (T349): the GPU's side of one engine: what the GPU would hold of the model and why it would not
// take it, the GPU's worker (public/gpu.js) started and stopped, a prompt's blocks and a generation's steps through
// it, and which of them go there (T135, T148, T152, T156, T210).
// A module of public/forward.js, asked for with its ?v=<build>; it reads jobs.js and its neighbours the same way.

const { BATCH, GPU_DONE, GPU_FAILED, GPU_BEAT, GPU_WANTED } = await import(new URL(`../jobs.js${new URL(import.meta.url).search}`, import.meta.url));
const { GPU_BLOCK, GPU_TOKENS, promptTimes, tokenTimes, PROMPTS_UNTIMED, PROMPTS_GPU, PROMPTS_CPU, gpuLine, STOPS_MOST } = await import(new URL(`choice.js${new URL(import.meta.url).search}`, import.meta.url));
const { GPU_WEIGHT_BYTES, GPU_TERNARY_BYTES, cpuReadBytes, aloneVerdict, OUTSIDE_VOCABULARY } = await import(new URL(`alone.js${new URL(import.meta.url).search}`, import.meta.url));

// T135: the GPU's worker says something at least this often while it puts a model on the GPU: after every step, each
// of which it gives up itself after 180 s (gpu.js's STEP_MS, T147: SwiftShader compiles a shader in up to 90 s). A
// worker quiet for longer than that is one the browser ended
const GPU_QUIET_MS = 200000;
// T147: the number of the GPU's requests, for this worker and every model it loads (the memory and its control area
// are kept from model to model, T96): a request of an engine let go is never one of the next engine's
let gpuRequests = 0;

/** T349: what the GPU would hold of one engine's model, and why it would not take it (createForward() asks before it
 * allocates the places the GPU writes to). */
export function gpuFit({ plan, gpuRoom, memoryUnsaid, direct, dim, layers, headSize, vocab, seqLen, hidden, kvDim, qDim,
  layerNorm, linear, convolution, rotated, sharedMemory, T, D, wq, wk, wv, wo, w1, w2, w3, wcls, attW, ffnW, attB, ffnB, bo,
  b1, b2, bq, bk, bv, qNorm, kNorm, positions }) {
  // why this model's prompt stays on the CPU, or null: the first stage (T135) takes Llama's layers of int8 weights
  // (T153: with Qwen2's biases, Qwen3's norms of the heads and heads of another size than dim / heads as well; T154:
  // GPT-2's and GPT-NeoX's as well, whose FFN has no gate: w3 is null; T155: int6 weights and 64-bit memories too)
  // (a function, not a const: gpuUnfit runs before this line, AGENTS.md)
  function gpuMatrices() {
    return Object.fromEntries(Object.entries({ wq, wk, wv, wo, w1, w2, w3 }).filter(([, m]) => m));
  }
  // (T155: int6 weights too, widened to int8 on the GPU, and a model in a 64-bit memory: gpu.js; T232: ternary weights
  // as they are, in their groups of 128. The 27B, T233, is ternary in a rotated basis with linear-attention layers:
  // either of the first two lines keeps it on the CPU)
  function gpuUnfit() {
    if (linear) return "linear-attention layers are not on the GPU yet";  // T229
    if (convolution) return "convolution layers are not on the GPU yet";  // T260
    if (rotated) return "a rotated basis is not on the GPU yet";  // T237
    if (!sharedMemory) return "the page is not cross-origin isolated";
    if (headSize % 4) return "heads of a size that is no multiple of 4 are not on the GPU";
    if (!Object.values(gpuMatrices()).every((m) => m.int8 && m.group === (m.ternary ? 128 : 32))) return "float32 weights are not on the GPU yet";
    // T148: the layers twice, in this memory and on the GPU (T156 will keep one): a device with too little memory
    // for both keeps the CPU's alone (a phone or an Apple shares its memory between the two)
    // (T153, the review: with the GPU's own keys and values, float16, as the prompt may fill the whole context: Qwen3
    // 0.6B's are 0.47 GB at 4096 positions, 95% of its layers')
    const onGpu = layersOnGpu();
    if (gpuRoom !== undefined && onGpu > gpuRoom && !direct) {
      return `the layers on the GPU as well (${Math.round(onGpu / 1e6)} MB) would not leave this device enough memory`;
    }
    return null;
  }
  // the bytes of the layers on the GPU, and its own keys and values
  function layersOnGpu() {
    return Object.values(gpuMatrices()).reduce((bytes, m) => bytes + layers * m.rows * m.n * (m.ternary ? GPU_TERNARY_BYTES : GPU_WEIGHT_BYTES), 0) +
      Object.values(gpuVectors()).reduce((bytes, { size }) => bytes + layers * size * 4, 0) + 2 * layers * seqLen * kvDim * 2;
  }
  // T152: why a generation's steps stay on the CPU, or null. A step on the GPU is T150's and T175's fused layer (gpu.js):
  // Llama's (RMSNorm, RoPE on whole heads, SwiGLU), and T226: with what the prompt's blocks take besides (T153: Qwen2's
  // biases of q, k and v, Qwen3's norms of the heads and heads of another size than dim / heads; T154: GPT-2's and
  // GPT-NeoX's LayerNorm, biases, GELU, learned positions, RoPE on a part of a head, parallel residual), and T92's
  // outlier channels (the GPU's classifier multiplies floats for such a model, and needs no columns apart). The keys
  // and values are float16 as the GPU's, or float32 where the CPU keeps them so (T160, widened on the way back and
  // narrowed on the way up). What is left: a classifier
  // and an embedding of int8 or int6 in groups of 32 (T232: or ternary in groups of 128; of a ternary classifier the
  // GPU takes the outlier channels apart as this does, shaders.js's TAKE_OUTLIERS); and the memory for the classifier, the embedding where it is
  // another table, RoPE's table and the vocabulary's three arrays of the sampling, besides the layers
  function tokensUnfit() {
    const embedding = T.token_embedding_table, group = wcls?.ternary ? 128 : 32;
    if (!wcls?.int8 || wcls.group !== group || !(wcls.ternary ? ["ternary"] : ["int8", "int6"]).includes(embedding.kind) || embedding.group !== group) {
      return "a classifier of float weights is not on the GPU's tokens";
    }
    // T205: the classifier and the embedding on the GPU as well (llm-jp-3 150M's 73 MB of layers came to 189 MB) where
    // the browser does not say what the device has: an iPhone's tab went down in /benchmark/'s model section. The
    // prompts' blocks still go (their layers alone)
    if (memoryUnsaid) return "this browser does not say how much memory the device has";
    const table = vocab * dim * (wcls.ternary ? GPU_TERNARY_BYTES : GPU_WEIGHT_BYTES);
    // (GPT-2's positions on the GPU as well, a row a position)
    const onGpu = layersOnGpu() + table * (plan.shared_classifier ? 1 : 2) + seqLen * headSize * 4 + 3 * vocab * 4 + (positions ? seqLen * D : 0);
    if (gpuRoom !== undefined && onGpu > gpuRoom) {
      return `the classifier on the GPU as well (${Math.round(onGpu / 1e6)} MB with the layers) would not leave this device enough memory`;
    }
    return null;
  }
  // the vectors of every layer the GPU reads (gpu.js's plan.vectors): the norms' weights; T153: Qwen2's biases of q,
  // k and v, Qwen3's norms of a head of q and of k; T154: GPT-2's and GPT-NeoX's biases of the two LayerNorms, of q,
  // k and v and of o, w1 and w2. Each one's address here and its floats a layer
  function gpuVectors() {
    return { attention: { at: attW, size: dim }, ffn: { at: ffnW, size: dim },
      ...(layerNorm ? { attentionBias: { at: attB, size: dim }, ffnBias: { at: ffnB, size: dim } } : {}),
      ...(bq ? { bq: { at: bq, size: qDim }, bk: { at: bk, size: kvDim }, bv: { at: bv, size: kvDim } } : {}),
      ...(qNorm ? { qNorm: { at: qNorm, size: headSize }, kNorm: { at: kNorm, size: headSize } } : {}),
      ...(bo ? { bo: { at: bo, size: dim }, b1: { at: b1, size: hidden }, b2: { at: b2, size: dim } } : {}) };
  }
  return { gpuMatrices, gpuUnfit, tokensUnfit, gpuVectors };
}

/** T349: the GPU's side of one engine (createForward(), which hands in what it reads of it): the GPU's worker started
 * and stopped, a prompt's blocks and a generation's steps through it, and which of them it takes (T135, T148, T152,
 * T156). Returns what the rest of the engine calls of it, and its variables that the rest reads or sets, each through
 * a getter and a setter. pool: the software threads' (threads.js); held: the engine's views and its cache as they are
 * now. */
export function gpuSide({ memory, base, plan, gpu, gpuRemembered, gpuForce, direct, stalledMs, always, dim, layers, heads,
  kvHeads, headSize, vocab, seqLen, hidden, kvDim, gpt2, layerNorm, parallel, unturned, views, T, halfKV, D, KV, wo, wcls,
  finalW, finalB, eps, cosTable, sinTable, positions, channels, gpuMatrices, gpuVectors, gpuWhyNot, staging, gpuRows,
  tokensWhyNot, gpuIds, grow, ctl, waitUntil, pool, embed, fromStaging, stagingFinite, notFiniteKV, run, held, halfToFloat }) {
  // ---- T135: a prompt on the GPU. gpu() makes the GPU's worker (gpu.js), which puts this model's layers on the GPU
  // once (a second copy of them: the CPU keeps its own, for the tokens after the prompt), and then runs each block of a
  // prompt that forwardMany() hands it through all the layers in one submission, while this thread waits for its
  // answer in the control area (Python calls forwardMany() and cannot wait for a promise; the GPU's answer is one). The
  // GPU multiplies a block's tokens by each weight it reads once (a matrix by a matrix: T134's "A prompt"). It writes the
  // keys and values of the block back (staging), and they go into the cache here as if the CPU had computed them: the
  // prompt's last token, with its logits, and every token after it stay the CPU's (T94: one token at a time the GPU was
  // slower on the owner's three devices). A model it does not take, or any failure, leaves the prompt on the CPU, said
  // once. gpuEnd: the positions up to which the GPU's own keys and values are the cache's (a block that begins after it
  // goes to the CPU; run() ends it where the CPU writes).
  // T148: by default (AGENTS.md's policy 9), and without waiting for it: the model is ready on the CPU at once, and the
  // GPU takes the blocks of a prompt from the first one after it is ready (a block past gpuEnd stays on the CPU, so a
  // GPU that is ready in the middle of a prompt changes nothing of it). Which blocks it takes: promptTimes (choice.js).
  // T205: gpuEnded, settled once the last GPU's worker made has let go of its buffers and its device (or never began)
  let gpuEnded = Promise.resolve(), gpuLast = null, quietTimer;
  let gpuWorker = null, gpuOn = false, gpuEnd = 0, gpuSerial = 0, gpuTokens = 0, settleGpu = null, gpuChosen = null;
  const gpuGettingReady = () => settleGpu !== null;
  // T148: what the status line says of the GPU now, and whether this generation checks the side not chosen again
  // recheck: the side not chosen that a part of the next long prompt goes to, to time it again ("cpu": the last
  // 2 × BATCH tokens of the prompt, "gpu": its first GPU_BLOCK; a GPU, then the CPU, is the order a prompt may take),
  // every GPU_RECHECK generations after the first verdict (sinceCheck), until it is timed (the review of T148: every
  // block on the other side put a GPU faster from 17 to 64 tokens on blocks of 16, 1.47 times as long, and never
  // timed the CPU). written: the positions this generation has filled
  let gpuStatus = gpu ? null : undefined, recheck = null, sinceCheck = 0, written = 0;
  // T184: why the prompts stay on the CPU (the model, the device, a failure), once it is known; and the side the
  // benchmark puts every block of a prompt on ("cpu", "gpu"), or null: the choice above
  let gpuReason = null, gpuSide = null;
  // T156: why the GPU stopped under a model on the GPU alone, once it has
  let directLost = null;
  const times = promptTimes();
  // T152: a generation's steps. tokensOn: the GPU takes them where it is faster (steps: their times); tokenStatus: what
  // the status line says of them (gpuLine's answers); tokenRecheck, cpuRecheck, sinceTokens: the side not chosen timed again now and then
  // (the CPU's first TOKEN_RECHECK steps of a generation, or the GPU's first run), every GPU_RECHECK generations after
  // the first verdict, as a prompt's; gpuSampled: the steps the GPU took since the generation began
  let tokensOn = false, tokenStatus = null, tokensReason = tokensWhyNot, tokenRecheck = null, cpuRecheck = 0, sinceTokens = 0, gpuSampled = 0;
  const steps = tokenTimes();
  // what the status line says of the GPU: of the prompts, and of the answers where there is something to say (gpuLine)
  const statusNow = () => gpuLine(gpuStatus, tokenStatus);
  const gpuNote = !gpu ? undefined : new Promise((resolve) => {
    settleGpu = (note) => {
      settleGpu = null;
      gpuStatus = note;
      resolve(statusNow());
    };
    // (T156: a model on the GPU alone that the GPU cannot take, or whose steps it cannot, is lost at once)
    const why = gpuWhyNot ?? (direct ? tokensWhyNot : null);
    if (why) stopGpu(why);
    else startGpu();
  });
  // T152: what gpu.js takes for a generation's steps: the classifier and (where it is another table) the embedding,
  // { rows, n, six, ternary (T232), at: [values, scales] }, the final norm's weights, where the ids go, and the steps a submission.
  // T226: the final LayerNorm's bias and GPT-2's positions (float32, a row a position), 0 where the model has none; and
  // whether its classifier has outlier channels (T92), and (T232) which they are
  function gpuTokensPlan() {
    const embedding = T.token_embedding_table;
    return { classifier: { rows: wcls.rows, n: wcls.n, six: wcls.six, ternary: Boolean(wcls.ternary), at: wcls.layer(0).slice(0, 2) },
      embedding: plan.shared_classifier ? null
        : { rows: vocab, n: dim, six: embedding.kind === "int6", ternary: embedding.kind === "ternary", at: [base + embedding.offset, base + embedding.scales] },
      final: finalW, finalBias: finalB, positions, outliers: channels.length > 0, channels, ids: gpuIds, most: GPU_TOKENS };
  }
  function startGpu() {
    // the values of a head RoPE turns: all of them, GPT-NeoX's first rotary (T154), none of GPT-2's
    const turned = gpt2 ? 0 : plan.rotary > 0 && plan.rotary < headSize ? plan.rotary : headSize;
    // (six, T155: the values at a layer's address are int6, packed as llama2_numpy.pack6 packs them; ternary, T232:
    // two bits a weight, as llama2_numpy.pack_ternary packs them, which the GPU takes as they are)
    const matrices = Object.fromEntries(Object.entries(gpuMatrices()).map(([name, m]) =>
      [name, { rows: m.rows, n: m.n, six: m.six, ternary: Boolean(m.ternary), layers: Array.from({ length: layers }, (_, l) => m.layer(l).slice(0, 2)) }]));
    const listen = () => {
      clearTimeout(quietTimer);
      quietTimer = setTimeout(() => stopGpu(`the GPU said nothing for ${GPU_QUIET_MS / 1000} s`), GPU_QUIET_MS);
    };
    const worker = gpuLast = gpuWorker = gpu();
    let ended;
    gpuEnded = new Promise((resolve) => { ended = resolve; });
    gpuStatus = "prompts on the CPU while the GPU gets ready";
    worker.onmessage = ({ data }) => {
      // T205: its buffers and its device let go (gpu.js's end()), and nothing more from it after a stop
      if (data.type === "ended") return ended();
      if (worker !== gpuWorker) return;
      if (data.type === "progress") return listen();
      clearTimeout(quietTimer);
      if (data.type === "ready") {
        gpuOn = true;
        if (data.blocks.length) times.started(data.blocks);
        gpuChosen = { matrices: data.form, attention: data.attention, key: data.key, remembered: data.remembered, seconds: data.seconds };
        console.info(`gpu: ${data.adapter}: ${Math.round(data.bytes / 1e6)} MB of layers on it in ${data.seconds.toFixed(1)} s`);
        // T147: the matrices' shader this device runs fastest of those that are right here, and what the others came to
        const forms = data.forms.map((f) => `${f.name} ${f.none ?? (f.remembered ? "remembered" : f.ms ? `${f.ms.toFixed(1)} ms` : "untimed")}`).join("; ");
        const blocks = data.blocks.map(({ count, ms }) => `${count} tokens ${ms.toFixed(1)} ms`).join(", ");
        console.info(`gpu: the matrices by ${data.form}, the attention by ${data.attention} (a pass of the first layer by ${GPU_BLOCK} tokens: ${forms}; a whole block: ${blocks})`);
        // T152: a generation's steps, where gpu.js made them (else why not)
        if (data.tokens) {
          tokensOn = true;
          gpuChosen.tokens = data.tokens.form;
          gpuChosen.tokenAttention = data.tokens.attention;  // T224: the attention of a token chosen here (tests)
          gpuChosen.tablePieces = data.tokens.pieces;  // T209: the classifier in pieces past what the device binds
          if (data.tokens.ms !== undefined) steps.gpu(data.tokens.ms);
          tokenStatus = always ? "gpu" : "untimed";
          const kinds = data.tokens.forms.map((f) => `${f.name} ${f.none ?? (f.remembered ? "remembered" : f.ms ? `${f.ms.toFixed(2)} ms` : "untimed")}`).join("; ");
          const attentions = (data.tokens.attentions ?? []).map((a) => `${a.name} ${a.none ?? (a.ms ? `${a.ms.toFixed(2)} ms` : "untimed")}`).join("; ");
          console.info(`gpu: a token by ${data.tokens.form} (a step of a run of ${GPU_TOKENS}: ${kinds}), its attention by ${data.tokens.attention} (${attentions})`);
        } else if (data.tokensWhy) {
          tokensReason = data.tokensWhy;
          tokenStatus = "why";
          console.info(`gpu: answers on the CPU (${tokensReason})`);
        } else if (tokensReason) {
          // (T152's review: a model whose steps were not asked of the GPU says why too, in the console alone: the
          // owner's words have the status line say no reason for the answers)
          tokenStatus = "why";
          console.info(`gpu: answers on the CPU (${tokensReason})`);
        }
        // T156: a model on the GPU alone needs its steps there too; and where /benchmark/ measured the CPU here, its
        // step (the checkpoint read once at the CPU section's fastest, T157's estimate) must not be faster by more
        // than BETTER's margin (the owner's B: the CPU's side estimated, not measured, for a model that does not fit twice)
        if (direct && !data.tokens) return stopGpu(data.tokensWhy ?? tokensReason ?? "the GPU did not take the steps");
        // (the owner, 2026-09-27: the prompts as well, which the GPU runs about five times as fast, weighed by use)
        if (direct) {
          const block = data.blocks.find((b) => b.count === GPU_BLOCK);
          const verdict = aloneVerdict({ size: cpuReadBytes(direct.size, wo.ternary ? "ternary" : "int8"), layerWeights: direct.layerWeights, cpu: direct.cpu, usage: direct.usage,
            gpu: { stepMs: data.tokens.ms, promptMs: block ? block.ms / block.count : undefined } });
          if (verdict.cpuFaster) {
            direct.verdict = { key: data.key, cpu: direct.cpu };  // the page keeps it: the next load goes on the CPU at once
            return stopGpu(`the CPU as /benchmark/ measured it (${direct.cpu.GBps?.toFixed(1)} GB/s): ${verdict.cpu.toFixed(0)} ms ` +
              `against the GPU's ${verdict.gpu.toFixed(0)} ms for ${Number(direct.usage.prompt).toFixed(0)} tokens of prompts and ${Number(direct.usage.written).toFixed(0)} written`);
          }
        }
        settleGpu?.(always ? PROMPTS_GPU : PROMPTS_UNTIMED);
      } else if (data.type === "unusable") {
        stopGpu(data.reason);
      } else if (data.type === "failed") {
        console.warn(`gpu: ${data.reason}`);
      }
    };
    worker.onerror = (event) => {
      ended();  // a worker that did not start holds nothing
      if (worker === gpuWorker) stopGpu(`the GPU's worker did not start (${event.message ?? "an error"})`);
    };
    listen();
    // T154: LayerNorm's epsilon is the CPU's layernorm kernel's and NumPy's, 1e-5 (GPT-2's and GPT-NeoX's
    // layer_norm_epsilon); parallel: GPT-NeoX's parallel residual
    worker.postMessage({ type: "start", memory, plan: { dim, hidden, layers, heads, kvHeads, headSize, turned, unturned, seqLen,
      kvStart: plan.kv_start, eps: layerNorm ? 1e-5 : eps, layerNorm, parallel: Boolean(parallel), batch: GPU_BLOCK, matrices,
      vectors: gpuVectors(), rows: gpuRows, tokens: gpuIds ? gpuTokensPlan() : null,
      cos: cosTable, sin: sinTable, staging, force: gpuForce, remembered: gpuRemembered, direct: Boolean(direct),
      words: { done: GPU_DONE, failed: GPU_FAILED, beat: GPU_BEAT, wanted: GPU_WANTED } } });
  }
  // the prompt stays on the CPU from here on (why: what the console says, where it was on the GPU); the GPU's worker
  // lets go of the device and ends
  function stopGpu(why) {
    clearTimeout(quietTimer);
    if (gpuOn && why) console.warn(`gpu: ${why}: the prompts and the tokens go on on the CPU`);
    // T156: the layers were on the GPU alone: the worker loads the model again on the CPU (not when it is let go)
    if (direct && why && !directLost) {
      directLost = why;
      direct.onLost?.(why);
    }
    gpuOn = false;
    tokensOn = false;
    tokenStatus = null;
    if (ctl) Atomics.store(ctl, GPU_WANTED, 0);  // T147: a request still under way writes nothing now
    gpuWorker?.postMessage({ type: "stop" });
    gpuWorker = null;
    gpuReason = why ?? "the model was let go";
    const note = `prompts on the CPU (${gpuReason})`;
    if (settleGpu) settleGpu(note);
    else if (gpu) gpuStatus = note;
  }
  // T148: whether a block of count tokens at pos0 goes to the GPU: where the GPU holds the keys and values before it,
  // and is faster for count tokens than the CPU on the threads in use now (promptTimes); and the first whole block of
  // a prompt where the GPU is to be timed again (recheck)
  function gpuTakes(count, pos0) {
    if (!gpuOn || pos0 > gpuEnd || gpuSide === "cpu") return false;
    if (always || gpuSide === "gpu") return true;
    const known = times.of(count, pool.threads);
    if (!known) return false;  // the CPU is timed first, on this prompt
    return known.faster || (recheck === "gpu" && pos0 === 0 && count === GPU_BLOCK);
  }
  // T148: the verdict for a whole block, as a prompt begins: the status line has it (the fewest tokens the GPU takes,
  // rounded up to a block of the CPU's, and none said up to one: the line does not move with every prompt), and the
  // console says it where it changes. Until the CPU is timed, the line stays as the GPU left it
  function verdict() {
    const most = times.of(GPU_BLOCK, pool.threads), from = times.threshold(GPU_BLOCK, pool.threads);
    if (!most || always) return;
    const status = from > GPU_BLOCK ? PROMPTS_CPU
      : from > BATCH ? `prompts of ${Math.ceil(from / BATCH) * BATCH} tokens and more on WebGPU` : PROMPTS_GPU;
    if (status !== gpuStatus) {
      console.info(`gpu: a block of ${GPU_BLOCK} tokens: ${most.gpu.toFixed(1)} ms on the GPU, ${most.cpu.toFixed(1)} ms on the ` +
        `CPU (${pool.threads} thread${pool.threads > 1 ? "s" : ""}), the GPU from ${from > GPU_BLOCK ? "no count" : `${from} tokens`}: ${status}`);
    }
    gpuStatus = status;
  }
  // A block of a prompt (up to GPU_BLOCK tokens at pos0, pos0 + 1, ...) through the layers on the GPU: false where it
  // must go to the CPU instead (no GPU, keys and values the GPU does not have, a failure)
  function promptOnGpu(tokens, pos0) {
    const count = tokens.length, began = performance.now();
    if (!direct && pos0 + count - 1 >= held.capacity) grow(pos0 + count - 1);
    views();
    // the GPU reads the rows from there (T210: on the GPU alone, it embeds the tokens itself)
    if (!direct) embed(tokens, pos0, gpuRows, D);
    gpuSerial = ++gpuRequests;
    Atomics.store(ctl, GPU_WANTED, gpuSerial);
    gpuWorker.postMessage({ type: "prompt", serial: gpuSerial, count, pos: pos0, ...(direct ? { tokens } : {}) });
    if (!waitUntil(GPU_DONE, (seen) => seen === gpuSerial, GPU_BEAT)) {
      stopGpu(`the GPU's worker stopped answering for ${stalledMs / 1000} s`);
      return false;
    }
    if (Atomics.load(ctl, GPU_FAILED)) {
      stopGpu("the GPU failed on a block of the prompt");  // the GPU's worker said why in the console
      return false;
    }
    views();
    // T243: a key or a value that is no finite number: the block is refused as a step with an id outside the vocabulary
    // is (T219): nothing of it written, the GPU's positions not counted as the cache's, the GPU stopped, and the CPU
    // takes the block from its own keys and values (forwardMany). T210: a model on the GPU alone reads none back (its
    // keys and values stay on the GPU, where T219's flag on the logits of the steps is what guards them)
    if (!direct && !stagingFinite(count)) {
      stopGpu(notFiniteKV(`in a block of the prompt at position ${pos0}`));
      return false;
    }
    if (!direct) fromStaging(pos0, count);
    gpuEnd = pos0 + count;
    gpuTokens += count;
    times.gpu(count, performance.now() - began);
    if (recheck === "gpu") recheck = null;
    return true;
  }
  // T210: the GPU's own keys and values of count positions from pos (a model on the GPU alone keeps none here), read
  // back GPU_BLOCK positions a request through staging: in float32, [layers][count][kvDim] each, as keysAndValues
  function gpuKeysAndValues(pos, count) {
    const out = [0, 1].map(() => new Float32Array(layers * count * kvDim));
    for (let at = 0; at < count; at += GPU_BLOCK) {
      const part = Math.min(GPU_BLOCK, count - at);
      if (!gpuOn) throw new Error(`The GPU stopped (${directLost ?? gpuReason}), and this model's keys and values were on it alone`);
      gpuSerial = ++gpuRequests;
      Atomics.store(ctl, GPU_WANTED, gpuSerial);
      gpuWorker.postMessage({ type: "keys", serial: gpuSerial, count: part, pos: pos + at });
      if (!waitUntil(GPU_DONE, (seen) => seen === gpuSerial, GPU_BEAT) || Atomics.load(ctl, GPU_FAILED)) {
        throw new Error("The GPU did not read its keys and values back");
      }
      views();
      for (let side = 0; side < 2; side++) {
        for (let l = 0; l < layers; l++) {
          for (let t = 0; t < part; t++) {
            const from = (staging + ((side * layers + l) * GPU_BLOCK + t) * kvDim * 2) / 2, to = (l * count + at + t) * kvDim;
            for (let i = 0; i < kvDim; i++) out[side][to + i] = halfToFloat(held.H[from + i]);
          }
        }
      }
    }
    return { keys: out[0], values: out[1] };
  }
  // T148: a block of a prompt on the CPU, BATCH at a time (T108), each whole one timed where the GPU is there to
  // weigh against (not while it starts: its upload and compilation share the CPU and the memory)
  function promptOnCpu(tokens, pos0) {
    for (let i = 0; i < tokens.length; i += BATCH) {
      const piece = tokens.slice(i, i + BATCH), began = performance.now();
      run(piece, pos0 + i, false);
      if (gpuOn && piece.length === BATCH) {
        times.cpu(pool.threads, (performance.now() - began) / BATCH);
        if (recheck === "cpu") recheck = null;
      }
    }
  }

  // T152: the verdict for a generation's steps, as a generation begins: the status line has it (it changes only there),
  // and the console says it where it changes
  function tokenVerdict() {
    const known = tokensOn && !always ? steps.of(pool.threads) : null;
    if (!known) return;
    const status = known.faster ? "gpu" : "cpu";
    if (status !== tokenStatus) {
      console.info(`gpu: a step of a generation: ${known.gpu.toFixed(2)} ms on the GPU, ${known.cpu.toFixed(2)} ms on the CPU ` +
        `(${pool.threads} thread${pool.threads > 1 ? "s" : ""}): answers on ${known.faster ? "WebGPU" : "the CPU"}`);
    }
    tokenStatus = status;
  }
  // T152: whether the steps of a generation go to the GPU now (see tokenBlock)
  function gpuSteps() {
    if (!tokensOn || !gpuOn || gpuSide === "cpu") return false;
    if (always || gpuSide === "gpu") return true;
    if (pool.search) return false;
    if (tokenRecheck === "gpu") return true;
    if (tokenRecheck === "cpu") return false;
    return Boolean(steps.of(pool.threads)?.faster);
  }
  /** T152: count steps of generate() on the GPU from token at pos: the forward pass of each token and the sampling of
   * the next (the penalty over the last of history, whose length is length; temperature, topp; randoms: a number for
   * each step, the CPU's generator's, none where greedy; stops: the stop tokens). Returns the ids sampled (a stop
   * token last where one came), whose keys and values are in the cache then as the CPU would have written them; or
   * undefined where the GPU did not take them (the CPU takes the step instead: nothing of it was written). Not null:
   * Pyodide makes JavaScript's null jsnull, which is not None (T152's review), and undefined None */
  function generateMany(token, pos, history, length, count, temperature, topp, penalty, randoms, stops) {
    const list = (x) => (x?.toJs ? x.toJs() : [...(x ?? [])]);
    if (!gpuSteps() || count < 1 || count > GPU_TOKENS) return undefined;
    const stopList = list(stops);
    if (stopList.length > STOPS_MOST) {
      tokensOn = false;
      tokensReason = `more than ${STOPS_MOST} stop tokens`;
      tokenStatus = "why";
      console.info(`gpu: answers on the CPU (${tokensReason})`);
      return undefined;
    }
    const began = performance.now();
    if (!direct && pos + count - 1 >= held.capacity) grow(pos + count - 1);
    views();
    gpuSerial = ++gpuRequests;
    Atomics.store(ctl, GPU_WANTED, gpuSerial);
    // (T210: on the GPU alone, no cache here to go up from, nor any position the GPU does not hold)
    gpuWorker.postMessage({ type: "tokens", serial: gpuSerial, count, pos, from: direct ? pos : Math.min(gpuEnd, pos), token, history: list(history), length,
      cache: direct ? null : { keys: held.keys, values: held.values, capacity: held.capacity, row: KV, half: halfKV }, settings: { temperature, topp, penalty, stops: stopList }, randoms: list(randoms) });
    if (!waitUntil(GPU_DONE, (seen) => seen === gpuSerial, GPU_BEAT)) {
      stopGpu(`the GPU's worker stopped answering for ${stalledMs / 1000} s`);
      return undefined;
    }
    if (Atomics.load(ctl, GPU_FAILED)) {
      stopGpu("the GPU failed on a token");  // the GPU's worker said why in the console
      return undefined;
    }
    views();
    const words = new Int32Array(memory.buffer, gpuIds, 2 + GPU_TOKENS), sampled = words[0], notFinite = words[1 + GPU_TOKENS];
    if (notFinite) {
      // T219: the sampler refused the step after the `sampled` ones: its logits held a NaN or +inf, or none over
      // -3.4e38 (the State's not_finite word, set by the bits: WGSL lets a GPU take NaN and infinities as absent). The
      // whole request is refused as one with an id outside the vocabulary below is (nothing of it written, the ids
      // before it not taken), the GPU stops, and the CPU takes the step again and stops where its own logits are not
      // finite either (T195's NOT_FINITE). A model on the GPU alone cannot: it stops, said in words
      if (direct) throw new Error(OUTSIDE_VOCABULARY);
      stopGpu(`the GPU computed logits that are not finite numbers (NaN or infinity) at position ${pos + sampled}`);
      return undefined;
    }
    if (!(sampled >= 1 && sampled <= count)) {
      stopGpu(`the GPU sampled ${sampled} of ${count} tokens`);
      return undefined;
    }
    // T219: an id outside the vocabulary is what SAMPLE gives for logits that are not finite numbers (NONE, -1: T195,
    // WGSL lets a GPU take NaN and infinities as absent). The step is refused, nothing of it written: the CPU takes it
    // again and stops where its own logits are not finite either (T195's NOT_FINITE), and the GPU, whose numbers are
    // no longer trusted, stops here. A model on the GPU alone cannot take the step on the CPU: it stops, said in words
    const ids = Array.from(words.subarray(1, 1 + sampled));
    const outside = ids.find((id) => !(id >= 0 && id < vocab));
    if (outside !== undefined) {
      if (direct) throw new Error(OUTSIDE_VOCABULARY);
      stopGpu(`the GPU sampled ${outside}, outside the vocabulary of ${vocab}`);
      return undefined;
    }
    // the keys and values of the positions sampled, float16 in the staging place as a prompt's block's (T147), into the
    // cache (T160's review of T152: a float32 cache, a grouped-query model's, widens them as a prompt's). T210: none
    // on the GPU alone. T243: where one of them is no finite number, the whole request is refused as above (nothing
    // written, none of its ids taken): the CPU takes the step, from keys and values that are its own
    if (!direct && !stagingFinite(sampled)) {
      stopGpu(notFiniteKV(`in a step at position ${pos}`));
      return undefined;
    }
    if (!direct) fromStaging(pos, sampled);
    gpuEnd = pos + sampled;
    gpuSampled += sampled;
    if (sampled === count) steps.gpu((performance.now() - began) / count);
    if (tokenRecheck === "gpu") tokenRecheck = null;
    return ids;
  }
  // T349: what the rest of the engine calls of the GPU's side, and reads and sets of where it is
  return {
    gpuGettingReady, times, steps, statusNow, gpuNote, stopGpu, gpuTakes, verdict, promptOnGpu, gpuKeysAndValues,
    promptOnCpu, tokenVerdict, gpuSteps, generateMany,
    get gpuEnded() { return gpuEnded; },
    get gpuLast() { return gpuLast; },
    get gpuWorker() { return gpuWorker; },
    get gpuOn() { return gpuOn; },
    get gpuEnd() { return gpuEnd; }, set gpuEnd(to) { gpuEnd = to; },
    get gpuTokens() { return gpuTokens; }, set gpuTokens(to) { gpuTokens = to; },
    get gpuChosen() { return gpuChosen; },
    get recheck() { return recheck; }, set recheck(to) { recheck = to; },
    get sinceCheck() { return sinceCheck; }, set sinceCheck(to) { sinceCheck = to; },
    get written() { return written; }, set written(to) { written = to; },
    get gpuReason() { return gpuReason; },
    get gpuSide() { return gpuSide; }, set gpuSide(to) { gpuSide = to; },
    get directLost() { return directLost; },
    get tokensOn() { return tokensOn; },
    get tokensReason() { return tokensReason; },
    get tokenRecheck() { return tokenRecheck; }, set tokenRecheck(to) { tokenRecheck = to; },
    get cpuRecheck() { return cpuRecheck; }, set cpuRecheck(to) { cpuRecheck = to; },
    get sinceTokens() { return sinceTokens; }, set sinceTokens(to) { sinceTokens = to; },
    get gpuSampled() { return gpuSampled; }, set gpuSampled(to) { gpuSampled = to; },
  };
}
