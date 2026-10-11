// forward/choice.js (T349): how a prompt's blocks and a generation's steps are given to the GPU or kept on the CPU:
// the sizes of what the GPU takes at once, the times of either side as they are measured (promptTimes, tokenTimes),
// the margin one must be faster by (BETTER), and what the status line says of it (gpuLine).
// A module of public/forward.js, which asks for it with its own ?v=<build> and exports its names as before.

// T205: the most release() waits for the GPU's worker to say it let go of its buffers and its device ("ended"), before
// the next model is read; one that says nothing by then (a compilation that does not return) is terminated (worker/weights.js
// waits as long for the GPU's worker of a model on the GPU alone let go before its engine was built, T156)
export const GPU_END_MS = 5000;
// T147: the most tokens of a prompt the GPU takes at once: the tokens of the largest tile (T146's 64 × 64), whose
// sixteen blocks left three quarters of it idle. Python hands a prompt over this many at a time where the GPU is on
// (promptBlock), BATCH where it is not: the worker answers nothing while one call runs (T108)
export const GPU_BLOCK = 64;
// T148: every this many generations after the first verdict, a part of a prompt goes to the side not chosen
const GPU_RECHECK = 8;
// T152: the steps of a generation the GPU takes a submission (T151's review, (k): a step takes W + F / N, the work and
// the submission's wait over the steps; with the owner's Android's F of about 8 ms and W of about 40 ms, 4 leaves 2 ms
// of the wait a step, 16 would leave 0.5 but show the text in pieces of 16 and run up to 15 passes past a stop token);
// the CPU's steps timed again, where the GPU is chosen, now and then; the most stop tokens the GPU's sampling holds
// (shaders.js's STOPS_MOST)
export const GPU_TOKENS = 4;
const TOKEN_RECHECK = 4, STOPS_MOST = 8;

// T148 (the review): the lower of the two middles. A block is slowed by what else runs (the first after a pause, a
// page in the background), never sped up: of two, the faster says the device, and one slow first block does not
// move a verdict
const lowerMedian = (xs) => [...xs].sort((a, b) => a - b)[(xs.length - 1) >> 1];

// T148: how long a block of a prompt takes on either side, from which forward.js gives each block to the GPU or keeps
// it on the CPU (AGENTS.md's policy 9: the GPU by default, the CPU where this device runs it faster). Measured, never
// written down (the development machine's numbers are no visitor's):
//   - the CPU: ms a token of the blocks of BATCH it runs of real prompts (no work only to time it), per number of
//     threads (the search may change it), the lower median of the last KEEP; none until TIMED of them (the first
//     block after a pause is the slowest: a single one would favour the GPU);
//   - the GPU: gpu.js times whole blocks of 16 and 64 tokens as it starts (in turn, after one of each to warm up),
//     and the line through them says a block of any count (fixed + a token: the tiles make a block of 16 cost nearly
//     as much as one of 64); every block it then runs for real scales that line by the lower median of the last KEEP
//     ratios (what the page adds around a block, a device that heats up or is loaded).
// A block of count tokens goes to the GPU where it takes less than BETTER of the CPU's time: a short prompt, whose
// block is small, stays on the CPU (T147's estimate: the GPU wins from about 64 tokens on a 1B model, a tiny-lm
// never), and so does a device whose GPU is slower. The threshold is the smallest count the GPU is faster for.
// (BETTER is also the margin of the threads' search: faster by more than the noise of a run)
const KEEP = 5, TIMED = 2, BETTER = 0.95;
export function promptTimes() {
  const cpu = new Map(), ratios = [];
  let line = null;
  const keep = (list, value) => {
    list.push(value);
    if (list.length > KEEP) list.shift();
  };
  const onLine = (count) => line.fixed + line.perToken * count;
  return {
    /** gpu.js's blocks timed as it started: [{ count, ms }], the smaller first */
    started(blocks) {
      const [a, b = a] = blocks;
      const perToken = b.count > a.count ? Math.max(0, (b.ms - a.ms) / (b.count - a.count)) : 0;
      line = { fixed: Math.max(0, a.ms - perToken * a.count), perToken };
    },
    /** a block of BATCH the CPU ran on threads threads: ms a token */
    cpu(threads, msPerToken) {
      if (!cpu.has(threads)) cpu.set(threads, []);
      keep(cpu.get(threads), msPerToken);
    },
    /** a block of count tokens the GPU ran, in ms */
    gpu(count, ms) {
      if (line && onLine(count) > 0) keep(ratios, ms / onLine(count));
    },
    /** { cpu, gpu, faster }: the ms of count tokens on either, on threads threads, and whether the GPU takes them;
     * null where the CPU is not timed yet (or the GPU has not started) */
    of(count, threads) {
      const times = cpu.get(threads);
      if (!line || !times || times.length < TIMED) return null;
      const onCpu = lowerMedian(times) * count, onGpu = onLine(count) * (ratios.length ? lowerMedian(ratios) : 1);
      return { cpu: onCpu, gpu: onGpu, faster: onGpu < BETTER * onCpu };
    },
    /** the fewest tokens of a block (up to most) the GPU takes, most + 1 where none; null where of() is */
    threshold(most, threads) {
      if (!this.of(most, threads)) return null;
      for (let count = 1; count <= most; count++) if (this.of(count, threads).faster) return count;
      return most + 1;
    },
  };
}

// T152: how long a step of a generation (the forward pass of the token fed, and the sampling of the next) takes on
// either side, from which forward.js gives the steps to the GPU or keeps them on the CPU, as promptTimes does a
// prompt's blocks: measured, never written down.
//   - the CPU: ms of the forward pass of a token with its logits, per number of threads, the lower median of the last
//     KEEP, none until TIMED (the sampling after it is Python's, on the kernels: not in it, so the CPU looks a little
//     faster than it is, the side the choice errs to);
//   - the GPU: ms a step of a run of GPU_TOKENS (gpu.js times runs as it starts, then every whole run is timed here,
//     from the request to its answer), the lower median of the last KEEP.
// The steps go to the GPU where a step takes less than BETTER of the CPU's.
export function tokenTimes() {
  const cpu = new Map(), gpu = [];
  const keep = (list, value) => {
    list.push(value);
    if (list.length > KEEP) list.shift();
  };
  return {
    /** a token's forward pass on the CPU on threads threads, in ms */
    cpu(threads, ms) {
      if (!cpu.has(threads)) cpu.set(threads, []);
      keep(cpu.get(threads), ms);
    },
    /** ms a step of a run on the GPU */
    gpu(ms) {
      keep(gpu, ms);
    },
    /** { cpu, gpu, faster }: ms a step on either, on threads threads, and whether the GPU takes the steps; null where
     * either is not timed yet */
    of(threads) {
      const times = cpu.get(threads);
      if (!gpu.length || !times || times.length < TIMED) return null;
      const onCpu = lowerMedian(times), onGpu = lowerMedian(gpu);
      return { cpu: onCpu, gpu: onGpu, faster: onGpu < BETTER * onCpu };
    },
  };
}

// T152: what the status line says of the GPU (the owner's words, 2026-09-27): prompts, the prompt's side as the prompts'
// verdict has it (PROMPTS_* below, "prompts of N tokens and more on WebGPU", or on the CPU and why), and answers, the
// generation's steps: "gpu", "cpu" (faster here), "why" (the GPU does not take them: why is in the console alone),
// "untimed", or null where there is nothing to say of them (no GPU, or the prompts are on the CPU for a reason)
export const PROMPTS_UNTIMED = "prompts on WebGPU where it is faster than the CPU";
export const PROMPTS_GPU = "prompts on WebGPU";
export const PROMPTS_CPU = "prompts on the CPU (faster here than WebGPU)";
const ANSWERS = { gpu: "answers on WebGPU", cpu: "answers on the CPU (faster here)", why: "answers on the CPU",
  untimed: "answers on WebGPU where it is faster than the CPU" };
export function gpuLine(prompts, answers) {
  if (!prompts || !answers) return prompts;
  if (prompts === PROMPTS_UNTIMED && answers === "untimed") return "WebGPU where it is faster than the CPU";
  if (prompts === PROMPTS_GPU && answers === "gpu") return "prompts and answers on WebGPU";
  if (prompts === PROMPTS_CPU && answers === "cpu") return "prompts and answers on the CPU (faster here than WebGPU)";
  return `${prompts}, ${ANSWERS[answers]}`;
}

// (T349) what forward.js and the other modules read besides, which was not exported where it was one file
export { GPU_RECHECK, TOKEN_RECHECK, STOPS_MOST, lowerMedian, BETTER };
