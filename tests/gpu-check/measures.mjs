// T357.3: the measures of tests/gpu-check.mjs, a part of that tool: how far two arrays of floats are from each other
// (the worst row, the scale, the logits' KL divergence). Arithmetic on what it is given, nothing else.
const floats = (b64) => {
  const bytes = Buffer.from(b64, "base64");
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
};
// the worst row (width values) of got against want: the largest difference over the row's largest value
function worstRow(got, want, width) {
  let worst = 0;
  for (let at = 0; at < want.length; at += width) {
    let largest = 0, difference = 0;
    for (let i = at; i < at + width; i++) {
      largest = Math.max(largest, Math.abs(want[i]));
      difference = Math.max(difference, Math.abs(got[i] - want[i]));
    }
    worst = Math.max(worst, difference / (largest || 1));
  }
  return worst;
}
// the worst layer of the ratios of its distances to its lines: the first that is NaN (a NaN is worse than any number),
// else the largest. A comparison with a NaN is false, so "x is not within the worst so far" alone let the layer after a
// NaN take its place and the NaN go unsaid unless it was the last (T357.3's review)
const worstAt = (ratios) => ratios.reduce((worst, x, l) => (Number.isNaN(ratios[worst]) ? worst : !(x <= ratios[worst]) ? l : worst), 0);
const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);
// T187: the log-probabilities of logits (float64)
function logSoftmax(logits) {
  const top = logits.reduce((m, x) => Math.max(m, x), -Infinity);
  let sum = 0;
  for (const x of logits) sum += Math.exp(x - top);
  const log = top + Math.log(sum);
  return Float64Array.from(logits, (x) => x - log);
}
// the KL divergence (nats) of got's softmax from want's (want's log-probabilities given)
function divergence(p, got) {
  const q = logSoftmax(got);
  let kl = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > -80) kl += Math.exp(p[i]) * (p[i] - q[i]);
  return Math.max(kl, 0);
}
// the least-squares factor of got against want, less 1: a scale that is wrong by a little shows here, where the
// rounding of every value (a noise) averages out
function scaleOff(got, want) {
  let gw = 0, ww = 0;
  for (let i = 0; i < want.length; i++) { gw += got[i] * want[i]; ww += want[i] * want[i]; }
  return gw / ww - 1;
}

export { floats, worstRow, worstAt, argmax, logSoftmax, divergence, scaleOff };
