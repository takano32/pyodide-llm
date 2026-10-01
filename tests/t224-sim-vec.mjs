// T224's review (pure Node, not for the repository): flash_attn_vec's split and reduce (subgroups form) simulated lane
// by lane with ideal subgroup operations, one subgroup of S lanes a workgroup, to see whether the algorithm is right at
// S = 32 (where lavapipe's shader fails). Written from public/shaders.js's flashVec and flashVecReduce, not from
// Fable's simulation.
import { pathToFileURL } from "node:url";
import path from "node:path";
const W = await import(pathToFileURL(path.resolve("public/shaders.js")).href);
const fromHalf = (h) => (h & 0x8000 ? -1 : 1) * ((h >> 10) & 31 ? 2 ** (((h >> 10) & 31) - 15) * (1 + (h & 1023) / 1024) : 2 ** -14 * ((h & 1023) / 1024));
const f32 = Math.fround;
const FLOAT_MIN = -1.0e9, KV_TILE = 32;

// ideal subgroup operations over arrays of S lanes (vec values are arrays too)
const shuffleDown = (a, delta) => a.map((_, l) => (l + delta < a.length ? a[l + delta] : NaN));
const shuffle = (a, src) => a.map((_, l) => a[src[l]]);
const lanesMax = (a) => a.map(() => a.reduce((m, x) => Math.max(m, x), -Infinity));
const lanesAdd = (a) => a.map(() => a.reduce((s, x) => f32(s + x), 0));

function split({ S, headSize, dSplit, q, K, V, positions, kvHeadOffset, nwg, iwg }) {
  const QC = headSize / 4, VC = headSize / 4, scale = f32(1 / Math.sqrt(headSize));
  const o = new Float32Array(headSize);
  let rowMax = FLOAT_MIN, expSum = 0;
  const seq = positions;
  const lane = [...Array(S).keys()];
  for (let kvTile = iwg * KV_TILE; kvTile < seq; kvTile += KV_TILE * nwg) {
    const inter = new Float32Array(KV_TILE);
    // q . k
    const tx = lane.map((l) => l % dSplit), ty = lane.map((l) => Math.floor(l / dSplit));
    for (let kvBase = 0; kvBase < KV_TILE; kvBase += S / dSplit) {
      const partial = lane.map((l) => {
        const kvIdx = kvBase + ty[l];
        const valid = kvIdx < KV_TILE && kvTile + kvIdx < seq;
        let sum = 0;
        if (valid) {
          for (let i = tx[l]; i < QC; i += dSplit) {
            let dot = 0;
            for (let c = 0; c < 4; c++) dot = f32(dot + f32(q[4 * i + c] * fromHalf(K[(kvTile + kvIdx) * (K.kvDim) + kvHeadOffset + 4 * i + c])));
            sum = f32(sum + dot);
          }
        }
        return sum;
      });
      let sum = partial.slice();
      for (let txDelta = dSplit >> 1; txDelta > 0; txDelta >>= 1) {
        const sh = shuffleDown(sum, txDelta);
        sum = sum.map((x, l) => (tx[l] < txDelta ? f32(x + sh[l]) : x));
      }
      const bcast = shuffle(sum, lane.map((l) => dSplit * ty[l]));
      lane.forEach((l) => {
        const kvIdx = kvBase + ty[l];
        if (tx[l] === 0 && kvIdx < KV_TILE && kvTile + kvIdx < seq) inter[kvIdx] = bcast[l];
      });
    }
    // softmax
    const term = (kvIdx) => (kvIdx < KV_TILE ? f32(inter[kvIdx] * scale) : FLOAT_MIN);
    let finalMax = lane.map(() => rowMax);
    for (let off = 0; off < KV_TILE; off += S) {
      finalMax = lanesMax(lane.map((l) => Math.max(finalMax[l], kvTile + off + l < seq && off + l < KV_TILE ? term(off + l) : FLOAT_MIN)));
    }
    let total = lane.map(() => 0);
    for (let off = 0; off < KV_TILE; off += S) {
      const p = lane.map((l) => (kvTile + off + l < seq && off + l < KV_TILE ? f32(Math.exp(term(off + l) - finalMax[l])) : 0));
      const add = lanesAdd(p);
      total = total.map((t, l) => f32(t + add[l]));
      lane.forEach((l) => {
        if (off + l < KV_TILE) inter[off + l] = p[l];
      });
    }
    const curExp = f32(Math.exp(rowMax - finalMax[0]));
    rowMax = finalMax[0];
    expSum = f32(f32(expSum * curExp) + total[0]);
    for (let e = 0; e < headSize; e++) o[e] = f32(o[e] * curExp);
    // P . V
    const neThreads = S / dSplit, nlThreads = Math.max(1, Math.floor(S / neThreads));
    const txPv = lane.map((l) => l % nlThreads), tyPv = lane.map((l) => Math.floor(l / nlThreads));
    for (let colBase = 0; colBase < VC; colBase += nlThreads) {
      let lo = lane.map((l) => {
        const acc = [0, 0, 0, 0];
        const vecCol = colBase + txPv[l];
        for (let cc = 0; cc * neThreads < KV_TILE; cc++) {
          const kvIdx = cc * neThreads + tyPv[l];
          if (kvIdx >= KV_TILE) continue;
          const row = kvTile + kvIdx;
          if (row >= seq) continue;
          for (let c = 0; c < 4; c++) acc[c] = f32(acc[c] + f32(inter[kvIdx] * fromHalf(V[row * V.kvDim + kvHeadOffset + vecCol * 4 + c])));
        }
        return acc;
      });
      for (let tyDelta = neThreads >> 1; tyDelta > 0; tyDelta >>= 1) {
        const sh = shuffleDown(lo, tyDelta * nlThreads);
        lo = lo.map((x, l) => (tyPv[l] < tyDelta ? x.map((v, c) => f32(v + sh[l][c])) : x));
      }
      lane.forEach((l) => {
        if (tyPv[l] === 0) {
          const base = (colBase + txPv[l]) * 4;
          for (let c = 0; c < 4; c++) o[base + c] = f32(o[base + c] + lo[l][c]);
        }
      });
    }
  }
  return { o, expSum, rowMax };
}

function attention({ S, min, headSize, q, K, V, positions, kvHeadOffset }) {
  const dSplit = Math.min(min, 4, Math.max((headSize & -headSize) / 4, 1));
  const shape = { splits: min, kvTile: KV_TILE };
  const nwg = W.flashVecSplits(shape, positions);
  const parts = [...Array(nwg).keys()].map((iwg) => split({ S, headSize, dSplit, q, K, V, positions, kvHeadOffset, nwg, iwg }));
  if (nwg === 1) {
    const p = parts[0];
    return { out: Float32Array.from(p.o, (x) => f32(x * (p.expSum !== 0 ? f32(1 / p.expSum) : 0))), nwg };
  }
  // the reduce: a lane a part
  const lane = [...Array(S).keys()];
  const mi = lane.map((l) => (l < nwg ? parts[l].rowMax : FLOAT_MIN));
  const si = lane.map((l) => (l < nwg ? parts[l].expSum : 0));
  const m = lanesMax(mi);
  const ms = lane.map((l) => (l < nwg ? f32(Math.exp(mi[l] - m[l])) : 0));
  const s = lanesAdd(lane.map((l) => f32(si[l] * ms[l])));
  const invS = s[0] !== 0 ? f32(1 / s[0]) : 0;
  const out = new Float32Array(headSize);
  for (let d = 0; d < headSize; d++) {
    let sum = 0;
    for (let l = 0; l < nwg; l++) sum = f32(sum + f32(parts[l].o[d] * ms[l]));
    out[d] = f32(sum * invS);
  }
  return { out, nwg };
}

let worst = 0, cases = 0;
for (const S of [4, 8, 16, 32]) {
  for (const headSize of [64, 128]) {
    for (const positions of [1, 31, 33, 64, 65, 129, 300, 1100]) {
      const heads = 2, kvHeads = 1;
      const data = W.tokenAttentionData({ heads, kvHeads, size: headSize, positions, steep: [1] });
      const K = Uint16Array.from(data.keys), Vv = Uint16Array.from(data.values);
      K.kvDim = kvHeads * headSize;
      Vv.kvDim = kvHeads * headSize;
      const outs = new Float32Array(heads * headSize);
      let nwg = 0;
      for (let h = 0; h < heads; h++) {
        const r = attention({ S, min: S, headSize, q: data.q.subarray(h * headSize, (h + 1) * headSize), K, V: Vv, positions, kvHeadOffset: 0 });
        outs.set(r.out, h * headSize);
        nwg = r.nwg;
      }
      const off = W.tokenAttentionOff(outs, data, { heads, kvHeads, size: headSize, positions });
      cases++;
      worst = Math.max(worst, off);
      if (!(off <= 1e-4)) console.log(`WRONG: S ${S} head ${headSize} positions ${positions} nwg ${nwg}: ${off}`);
    }
  }
  console.log(`S ${S}: done, worst so far ${worst.toExponential(2)}`);
}
console.log(`${cases} cases, worst ${worst.toExponential(2)}`);
