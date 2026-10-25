// shaders/likecpu.ts (T.351): no shader: the CPU's sampling in JavaScript (the kernel's and NumPy's order of
// operations), which the checks of the GPU's SAMPLE are held to.
// A part of src/runtime/shaders.ts, which is the window: everything outside src/runtime/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part takes its neighbours' names by static imports (T.367.3:
// a bundler links them).
import { REPETITION_WINDOW } from "./run.ts";

// (T.368.5) the tokens so far, of which the penalty reads the latest: an array or a typed one
type History = { slice(start: number): Iterable<number> };

// The CPU's sampling in JavaScript (kernels/kernel.ts's penalize() and sample(), as the engine's generate() calls
// them): what SAMPLE is held to (/benchmark/'s check), and itself held to the kernel (tests/smoke.mjs). logits: a
// Float32Array, changed in place by the penalty as the kernel changes them. history: BOS, the prompt and the sampled
// tokens, the token fed last. T.274: presence, the kernel's presence penalty (taken off each after the penalty). T.546:
// written, how many of the history's latest were sampled, of which the presence penalty's window is (the engine's
// generate() hands the kernel its written tokens apart: the prompt's are not counted); all of them where not given.
export function penalizeLikeCpu(logits: Float32Array, history: History, penalty: number, presence = 0, written = Infinity) {
  if (penalty === 1 && !(presence > 0)) return;
  const f = Math.fround, p = f(penalty), less = presence > 0 ? f(presence) : 0;
  const window = [...history.slice(-REPETITION_WINDOW)], sampled = new Set(written > 0 ? window.slice(-Math.min(written, REPETITION_WINDOW)) : []);
  for (const token of new Set(window)) {
    const value = logits[token];
    // (the kernel's two calls: the penalty's, then a penalty of 1 and the presence's, x / 1 being x)
    logits[token] = f((penalty === 1 ? value : value > 0 ? f(value / p) : f(value * p)) - (sampled.has(token) ? less : 0));
  }
}
/** The token kernel.ts's sample() draws for random (in [0, 1)): float32 probabilities, float64 sums; the nucleus
 * sorted from the most probable, equal ones in the order of their index (T.561: as the kernel's sort and NumPy's). temperature 0: the first index of the largest logit (NumPy's argmax). */
export function sampleLikeCpu(logits: ArrayLike<number>, temperature: number, topp: number, random: number, topk = 0, minp = 0) {
  if (temperature === 0) return argmaxLikeCpu(finiteLikeCpu(logits));
  const { tokens, cumulative, mass } = walkLikeCpu(logits, temperature, topp, topk, minp);
  const goal = random * mass;
  for (let k = 0; k < tokens.length; k++) if (cumulative[k] > goal) return tokens[k];
  return tokens[tokens.length - 1];
}
export function argmaxLikeCpu(logits: ArrayLike<number>) {
  let best = -Infinity, first = 0;
  for (let i = 0; i < logits.length; i++) if (logits[i] > best) [best, first] = [logits[i], i];
  return first;
}
/** T.195: the engine draws no token when the largest logit is no finite number (a NaN anywhere, +inf anywhere, or
 * all -inf): the kernel returns -1 and NumPy's sample() raises, and the engine stops with an error. So does this: it
 * throws, and returns the logits otherwise. (argmaxLikeCpu alone passes over a NaN, as `>` is false for it.) SAMPLE
 * is held to it by T.219: it sees such logits by their bits (not_finite: WGSL lets an implementation assume that no NaN
 * nor infinity occurs, §15.7, so a comparison would not do), refuses the step with the State's not_finite word, and the
 * CPU takes the step and decides by this rule. */
export function finiteLikeCpu<Logits extends ArrayLike<number>>(logits: Logits): Logits {
  let best = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (Number.isNaN(logits[i])) best = NaN;
    else if (logits[i] > best) best = logits[i];
  }
  if (!Number.isFinite(best)) throw new Error(`the largest logit is ${best}: no token is drawn from logits that are not finite (T195)`);
  return logits;
}
/** The tokens kernel.ts's sample() walks for a random number, in its order (the nucleus's, sorted; else the index's),
 * the running sum after each (float64) and the mass the random number is a share of. T.274: the kernel's top-k (the
 * topk most probable, whose nucleus it is then, walked from the most probable also without a nucleus) and min-p (what
 * is less than minp times as probable as the most probable is left out, last). */
export function walkLikeCpu(logits: ArrayLike<number>, temperature: number, topp: number, topk = 0, minp = 0) {
  const f = Math.fround, n = logits.length, best = logits[argmaxLikeCpu(finiteLikeCpu(logits))];
  const nucleus = topp > 0 && topp < 1;
  // without a nucleus the kernel's floor is -f32.MAX_VALUE (and SAMPLE's -3.4e38): a -inf logit is left out, not
  // given the exp(-87) of vexp() below (T.195: with a random number of 0 that drew a token at -inf)
  const lowest = nucleus ? f(best - f(f(temperature) * f(16.118095))) : -3.4028234663852886e38, inverse = f(1 / f(temperature));
  const probs: number[] = [], index: number[] = [];
  for (let i = 0; i < n; i++) {
    if (logits[i] >= lowest) {
      probs.push(f(f(logits[i] - best) * inverse));
      index.push(i);
    }
  }
  // kernel.ts's exp(): four at a time (vexp) holds x at -87 at least, the last count % 4 one by one (fexp) are 0 under
  // it. Only a random number of 0 draws such a token (tests/smoke.mjs found one: T.151's review round)
  const simd = probs.length - (probs.length % 4);
  probs.forEach((x, k) => (probs[k] = x < -87 ? (k < simd ? f(Math.exp(-87)) : 0) : f(Math.exp(x))));
  let total = 0, top = 0;
  for (const p of probs) (total += p), (top = Math.max(top, p));
  const fromTheMost = (a: number, b: number) => probs[b] - probs[a] || index[a] - index[b];
  let order = probs.map((_, k) => k);
  const narrowed = topk > 0 && topk < probs.length;
  if (narrowed) {
    order = order.sort(fromTheMost).slice(0, topk);
    total = 0;
    for (const k of order) total += probs[k];
  }
  const least = minp > 0 ? f(Math.min(f(minp), 1) * top) : 0;
  let last = order.length - 1;
  if (nucleus || narrowed) {
    // the most probable token always stays (kernel.ts, T.178)
    const cutoff = nucleus ? Math.min(((1 - f(topp)) / (order.length > 1 ? order.length - 1 : 1)) * total, top) : 0;
    order = order.filter((k) => probs[k] >= cutoff && probs[k] >= least).sort(fromTheMost);
    const limit = nucleus ? f(topp) * total : Infinity;
    let sum = 0;
    last = order.length - 1;
    for (let k = 0; k < order.length; k++) {
      sum += probs[order[k]];
      if (sum >= limit) {
        last = k;
        break;
      }
    }
  }
  if (!nucleus && !narrowed && minp > 0) {
    order = order.filter((k) => probs[k] >= least);
    last = order.length - 1;
  }
  const tokens: number[] = [], cumulative: number[] = [];
  let sum = 0;
  for (let k = 0; k <= last; k++) {
    sum += probs[order[k]];
    tokens.push(index[order[k]]);
    cumulative.push(sum);
  }
  return { tokens, cumulative, mass: sum };
}
