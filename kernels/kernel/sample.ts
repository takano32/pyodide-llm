// kernel/sample.ts (T356): from the logits to a token: the greedy choice, the penalties, and the sampling
// (temperature, top-k, min-p, top-p) as Llama.sample() does it in NumPy.

import { fexp, vexp, largest } from "./math";

export function argmax(x: usize, n: i32): i32 {
  let mi = 0; let mv = load<f32>(x);
  for (let j = 1; j < n; j++) { const v = load<f32>(x + (<usize>j << 2)); if (v > mv) { mv = v; mi = j; } }
  return mi;
}

// ---------------------------------------------------------------------------------------------- sampling
// What Llama.sample() does in NumPy, without walking a vocabulary of 50000 or 100000 tokens several times.

// tokens: the recently generated ones; each is made less likely once, however often it occurs. T274: presence is taken
// off the logit of each after that (a presence penalty: llama.cpp's and OpenAI's, the same for one occurrence and for
// ten); 0 or less takes nothing off
export function penalize(logits: usize, tokens: usize, count: i32, penalty: f32, presence: f32): void {
  const less: f32 = presence > 0 ? presence : 0;
  for (let i = 0; i < count; i++) {
    const token = load<i32>(tokens + (<usize>i << 2));
    let seen = false;
    for (let j = 0; j < i; j++) {
      if (load<i32>(tokens + (<usize>j << 2)) == token) { seen = true; break; }
    }
    if (seen) continue;
    const address = logits + (<usize>token << 2);
    const value = load<f32>(address);
    store<f32>(address, (value > 0 ? value / penalty : value * penalty) - less);
  }
}

// The state of sortNucleus(): globals are no static data
let nucleusMass: f64 = 0;
let nucleusLimit: f64 = 0;
let nucleusLast: i32 = -1;

// adds probs[lo..hi], already in their final order, to the nucleus: true when it is complete
// @ts-ignore: decorator
@inline function intoNucleus(probs: usize, lo: i32, hi: i32): bool {
  for (let k = lo; k <= hi; k++) {
    nucleusMass += load<f32>(probs + (<usize>k << 2));
    if (nucleusMass >= nucleusLimit) { nucleusLast = k; return true; }
  }
  return false;
}

// Sorts probs[lo..hi] in descending order, and index[] along with it, from the left and only as far as the nucleus
// reaches: usually a few dozen tokens out of thousands. Quicksort (median of three), insertion sort for short ranges.
function sortNucleus(probs: usize, index: usize, lo: i32, hi: i32): void {
  while (hi - lo > 12) {
    const mid = lo + ((hi - lo) >> 1);
    let a = load<f32>(probs + (<usize>lo << 2)), b = load<f32>(probs + (<usize>mid << 2));
    const c = load<f32>(probs + (<usize>hi << 2));
    if (a < b) { const t = a; a = b; b = t; }
    if (b < c) { b = c; if (a < b) b = a; }
    const pivot = b;
    let i = lo, j = hi;
    while (i <= j) {
      while (load<f32>(probs + (<usize>i << 2)) > pivot) i++;
      while (load<f32>(probs + (<usize>j << 2)) < pivot) j--;
      if (i <= j) {
        const pi = probs + (<usize>i << 2), pj = probs + (<usize>j << 2), xi = index + (<usize>i << 2), xj = index + (<usize>j << 2);
        const p = load<f32>(pi); store<f32>(pi, load<f32>(pj)); store<f32>(pj, p);
        const x = load<i32>(xi); store<i32>(xi, load<i32>(xj)); store<i32>(xj, x);
        i++; j--;
      }
    }
    sortNucleus(probs, index, lo, j);
    if (nucleusLast >= 0) return;
    if (intoNucleus(probs, j + 1, i - 1)) return; // equal to the pivot
    lo = i;
  }
  for (let i = lo + 1; i <= hi; i++) {
    const p = load<f32>(probs + (<usize>i << 2));
    const x = load<i32>(index + (<usize>i << 2));
    let j = i - 1;
    while (j >= lo && load<f32>(probs + (<usize>j << 2)) < p) {
      store<f32>(probs + (<usize>(j + 1) << 2), load<f32>(probs + (<usize>j << 2)));
      store<i32>(index + (<usize>(j + 1) << 2), load<i32>(index + (<usize>j << 2)));
      j--;
    }
    store<f32>(probs + (<usize>(j + 1) << 2), p);
    store<i32>(index + (<usize>(j + 1) << 2), x);
  }
  intoNucleus(probs, lo, hi);
}

// Writes value and token at probs[count] and index[count], and counts them when past is 1: a list that keeps some of
// a sequence in its order without a branch on each (T189: whether a token stays is not predictable). count must be
// at most the position read from, so that nothing not yet read is written over
// @ts-ignore: decorator
@inline function kept(probs: usize, index: usize, count: i32, value: f32, token: i32, past: i32): i32 {
  store<f32>(probs + (<usize>count << 2), value);
  store<i32>(index + (<usize>count << 2), token);
  return count + past;
}

// kept() for the four of d, the bits of past telling which of them are past the floor (T189)
// @ts-ignore: decorator
@inline function keptFour(probs: usize, index: usize, count: i32, d: v128, i: i32, past: i32): i32 {
  count = kept(probs, index, count, f32x4.extract_lane(d, 0), i, past & 1);
  count = kept(probs, index, count, f32x4.extract_lane(d, 1), i + 1, (past >> 1) & 1);
  count = kept(probs, index, count, f32x4.extract_lane(d, 2), i + 2, (past >> 2) & 1);
  return kept(probs, index, count, f32x4.extract_lane(d, 3), i + 3, past >> 3);
}

// T274: moves the k largest of probs[lo..hi] to its first k places (index[] along with them), in no set order: the
// partition of sortNucleus() without the sorting. Equal ones at the border: either
function selectTop(probs: usize, index: usize, lo: i32, hi: i32, k: i32): void {
  const until = lo + k;  // the first place that is not among the k
  while (lo < hi) {
    const mid = lo + ((hi - lo) >> 1);
    let a = load<f32>(probs + (<usize>lo << 2)), b = load<f32>(probs + (<usize>mid << 2));
    const c = load<f32>(probs + (<usize>hi << 2));
    if (a < b) { const t = a; a = b; b = t; }
    if (b < c) { b = c; if (a < b) b = a; }
    const pivot = b;
    let i = lo, j = hi;
    while (i <= j) {
      while (load<f32>(probs + (<usize>i << 2)) > pivot) i++;
      while (load<f32>(probs + (<usize>j << 2)) < pivot) j--;
      if (i <= j) {
        const pi = probs + (<usize>i << 2), pj = probs + (<usize>j << 2), xi = index + (<usize>i << 2), xj = index + (<usize>j << 2);
        const p = load<f32>(pi); store<f32>(pi, load<f32>(pj)); store<f32>(pj, p);
        const x = load<i32>(xi); store<i32>(xi, load<i32>(xj)); store<i32>(xj, x);
        i++; j--;
      }
    }
    // lo..j are the pivot at least, i..hi the pivot at most, and what is between them is the pivot
    if (until <= j) hi = j;
    else if (until > i) lo = i;
    else return;
  }
}

// Draws a token from softmax(logits / temperature), restricted to the nucleus when 0 < topp < 1.
// T274: topk > 0 keeps the topk most probable first, and the nucleus is then of those (their probabilities add up to
// one again); minp > 0 then leaves out what is less than minp times as probable as the most probable. The order of
// llama.cpp's samplers and of transformers' (top-k, top-p, min-p). With a top-k the tokens are walked from the most
// probable, as a nucleus's are.
// random: one number in [0, 1) from Python's generator, so that a seed reproduces. probs and index: scratch of n each.
// -1 when the largest logit is no finite number (T195): a NaN anywhere makes it NaN (f32x4.max and max keep a NaN),
// +inf makes it +inf, and all -inf make it -inf. Logits like these come of a broken model or an overflow; the engine
// stops with an error on -1 (NumPy's sample() and shaders.js's sampleLikeCpu() stop on the same logits)
export function sample(logits: usize, n: i32, temperature: f32, topp: f32, random: f64, probs: usize, index: usize, topk: i32, minp: f32): i32 {
  const best = largest(logits, n, load<f32>(logits));
  if (!isFinite<f32>(best)) return -1;
  const nucleus = topp > 0 && topp < 1;
  // With a nucleus, tokens less than a ten millionth as probable as the best one cannot matter (ln 1e-7 = -16.118):
  // they are left out before exp(), which is the expensive part
  const floor: f32 = nucleus ? best - temperature * <f32>16.118095 : -f32.MAX_VALUE;
  // T189: eight at a time. Eight that all stay below the floor (most of the vocabulary) cost two comparisons and one
  // branch; of the others each is written and only those past the floor are counted, in their order, without a
  // branch each (whether a token passes is not predictable). A branch for each four was slower (on the arm64 runner
  // the walk 1.23 times main's where this is 2.1), and none at all slower still (0.45 times)
  const floors = f32x4.splat(floor), shift = f32x4.splat(best);
  let count = 0;
  let i = 0;
  for (; i + 8 <= n; i += 8) {
    const at = logits + (<usize>i << 2);
    const a = v128.load(at), b = v128.load(at, 16);
    const pa = f32x4.ge(a, floors), pb = f32x4.ge(b, floors);
    if (!v128.any_true(v128.or(pa, pb))) continue;
    count = keptFour(probs, index, count, f32x4.sub(a, shift), i, i32x4.bitmask(pa));
    count = keptFour(probs, index, count, f32x4.sub(b, shift), i + 4, i32x4.bitmask(pb));
  }
  for (; i < n; i++) {
    const v = load<f32>(logits + (<usize>i << 2));
    count = kept(probs, index, count, v - best, i, <i32>(v >= floor));
  }
  const inverse = f32x4.splat(<f32>1.0 / temperature);
  for (i = 0; i + 4 <= count; i += 4) {
    const address = probs + (<usize>i << 2);
    v128.store(address, vexp(f32x4.mul(v128.load(address), inverse)));
  }
  for (; i < count; i++) {
    const address = probs + (<usize>i << 2);
    store<f32>(address, fexp(load<f32>(address) / temperature));
  }
  let total: f64 = 0;
  let top: f32 = 0;
  for (i = 0; i < count; i++) {
    const p = load<f32>(probs + (<usize>i << 2));
    total += p;
    top = max(top, p);
  }
  const narrowed = topk > 0 && topk < count;
  if (narrowed) {
    selectTop(probs, index, 0, count - 1, topk);
    count = topk;
    total = 0;
    for (i = 0; i < count; i++) total += load<f32>(probs + (<usize>i << 2));
  }
  // (the most probable token is never less than this: a minp past 1 is 1)
  const least: f32 = minp > 0 ? min(minp, <f32>1.0) * top : 0;
  let last = count - 1;
  let mass = total;
  if (nucleus || narrowed) {
    // Tokens below (1 - topp) / (n - 1) cannot be part of the nucleus (llama2.c), so they need not be sorted. That
    // holds while one token at least stays: when all are below it (n * topp < 1, which the floor above makes possible,
    // T178), the others add up to less than (1 - topp), so the nucleus is the most probable token alone. It always stays
    const cutoff: f64 = nucleus ? min((1.0 - <f64>topp) / <f64>(count > 1 ? count - 1 : 1) * total, <f64>top) : 0;
    let likely = 0;
    for (let k = 0; k < count; k++) {
      const p = load<f32>(probs + (<usize>k << 2));
      likely = kept(probs, index, likely, p, load<i32>(index + (<usize>k << 2)), <i32>(<f64>p >= cutoff) & <i32>(p >= least));
    }
    // the most probable tokens whose probabilities add up to topp (of what a top-k left; all of them without a nucleus)
    nucleusMass = 0;
    nucleusLimit = nucleus ? <f64>topp * total : Infinity;
    nucleusLast = -1;
    sortNucleus(probs, index, 0, likely - 1);
    last = nucleusLast >= 0 ? nucleusLast : likely - 1;
    mass = nucleusMass;
  } else if (minp > 0) {
    // in the order of the index, as without it
    let likely = 0;
    mass = 0;
    for (let k = 0; k < count; k++) {
      const p = load<f32>(probs + (<usize>k << 2));
      const stays = <i32>(p >= least);
      likely = kept(probs, index, likely, p, load<i32>(index + (<usize>k << 2)), stays);
      if (stays) mass += p;
    }
    last = likely - 1;
  }
  const target: f64 = random * mass;
  let cumulative: f64 = 0;
  for (let k = 0; k <= last; k++) {
    cumulative += load<f32>(probs + (<usize>k << 2));
    if (cumulative > target) return load<i32>(index + (<usize>k << 2));
  }
  return load<i32>(index + (<usize>last << 2));
}
