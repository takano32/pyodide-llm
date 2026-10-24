// gpu/linear.ts (T271.2, T271.3, T568.2): a Qwen3.5's linear-attention layers on the GPU, a kind of gpu/kinds.ts's
// table: its matrices of q, k and v (into mixed) and of z, its gates (DELTA_GATES), its convolution (DELTA_CONV into
// convolved) and its rows moved on (DELTA_CONV_STATE), the l2 norms of q and of k (HEAD_NORM), the delta rule over the
// place's state (DELTA_NET into out), the norm of a value head, z's gate (SWIGLU: into z), its matrix out; a token's
// step the same on a token's vectors (a block of one); the states and the rows they go on from, place after place, as the
// CPU keeps them (forward/linear.ts), and the check of a first layer of this kind.
// (a module of the model's GPU worker, gpu.ts, which a bundler links it into)
import { STORAGE, COPY_DST, COPY_SRC, MAP_READ, within, buffer, uniform, normShape as headNorm, pipelineOf, bind, dispatch } from "./device.ts";
import { GROUP } from "../shaders/common.ts";
import { ENGINE } from "../shaders/engine.ts";
import type { GpuKind, Quantizing } from "./kinds.ts";
import type { Pass } from "./device.ts";

export const linear: GpuKind = {
  // (the read, which z's gate leaves for the matrix out to quantize)
  widest: (plan) => plan.linear!.valueHeads * plan.linear!.valueDim,
  writes: () => ["write"],
  async block(m) {
    const { plan } = m, B = plan.batch, { keyHeads, valueHeads, keyDim, valueDim, taps } = plan.linear!, places = plan.kinds.filter((kind) => kind === "linear").length;
    const keys = keyHeads * keyDim, read = valueHeads * valueDim, mixed = 2 * keys + read, groups = Math.ceil(mixed / 64);
    const [gates, convolving, moving, rule, swiglu] = await within(Promise.all([ENGINE.DELTA_GATES, ENGINE.DELTA_CONV, ENGINE.DELTA_CONV_STATE, ENGINE.DELTA_NET,
      ENGINE.SWIGLU].map((code) => pipelineOf(m, code))), "the linear attention's shaders");
    const z = buffer(m, B * read * 4);
    // (the l2 norms' weights: 1 / key dim of q's, which norms it over the root of key dim more, and 1 / sqrt(key dim) of k's)
    const units = buffer(m, 2 * keyDim * 4, STORAGE | COPY_DST);
    m.device.queue.writeBuffer(units, 0, Float32Array.from({ length: 2 * keyDim }, (_, i) => (i < keyDim ? 1 / keyDim : 1 / Math.sqrt(keyDim))));
    const matricesBytes = places * valueHeads * keyDim * valueDim * 4, rowsBytes = places * taps * mixed * 4;
    const mixedOut = buffer(m, B * mixed * 4), convolved = buffer(m, B * mixed * 4), work = buffer(m, B * 2 * valueHeads * 4), out = buffer(m, B * read * 4);
    const states = buffer(m, matricesBytes, STORAGE | COPY_DST | COPY_SRC), rows = buffer(m, rowsBytes, STORAGE | COPY_DST | COPY_SRC);
    let quantizeZ: Quantizing | undefined;
    return {
      states: [{ buffer: states, bytes: matricesBytes }, { buffer: rows, bytes: rowsBytes }],
      // the bind groups of the layer at place among them (a block's: B tokens' room in its buffers)
      layer({ place, product, out: joins }) {
        const b = m.block, V = m.weights.vectors!;
        // (z quantized for the matrix out: one for every layer, as the block's own inputs' are)
        quantizeZ ??= { group: bind(m, b.quantize, [z, b.xq, b.xs, uniform(m, new Uint32Array([read, read, 0, 0])), b.step]), x: Math.ceil(read / GROUP / 64) };
        const conv = uniform(m, new Uint32Array([mixed, taps, place * taps * mixed, keys, B, 0, 0, 0]));
        const into = product("wqkv", b.xb, mixedOut), zOf = product("wz", b.xb, z);
        const gated = bind(m, gates, [b.xb, V.wb, V.wa, V.dtBias, V.decay, work, uniform(m, new Uint32Array([plan.dim, valueHeads, place * valueHeads, 0])), b.step]);
        const convolution = bind(m, convolving, [mixedOut, V.conv, rows, convolved, conv, b.step]), moved = bind(m, moving, [mixedOut, rows, conv, b.step]);
        const qNorm = bind(m, b.headNorm, [convolved, units, headNorm(m, keyDim, 0, 1e-6 / keyDim, 0), b.step]);
        const kNorm = bind(m, b.headNorm, [convolved, units, headNorm(m, keyDim, keyDim, 1e-6 / keyDim, B * keyHeads), b.step]);
        const ruled = bind(m, rule, [convolved, work, states, out, uniform(m, new Uint32Array([keyHeads, valueHeads, keyDim, valueDim, B, place * valueHeads * keyDim * valueDim, 0, 0])), b.step]);
        const valueNorm = bind(m, b.headNorm, [out, V.deltaNorm, headNorm(m, valueDim, place * valueDim, plan.eps, 0), b.step]);
        const gate = bind(m, swiglu, [z, out, uniform(m, new Uint32Array([read, 0, 0, 0])), b.step]), wout = joins("wout", z);
        // its two matrices of the normed stream, its gates, the convolution and its rows moved on, the l2 norms of q and
        // k, the delta rule over its state, the norm of every value head, z's gate, its matrix out (the last layer's too:
        // its state is what the next token goes on from)
        return ({ pass, count, last, multiplied, quantized }) => {
          multiplied(into);
          multiplied(zOf);
          dispatch(pass, gates, gated, valueHeads, count);
          dispatch(pass, convolving, convolution, groups, count);
          dispatch(pass, moving, moved, groups);
          dispatch(pass, b.headNorm, qNorm, keyHeads, count);
          dispatch(pass, b.headNorm, kNorm, keyHeads, count);
          dispatch(pass, rule, ruled, valueHeads);
          dispatch(pass, b.headNorm, valueNorm, valueHeads, count);
          dispatch(pass, swiglu, gate, Math.ceil(read / 64), count);
          if (last) return true;
          quantized(quantizeZ!);
          multiplied(wout);
          return false;
        };
      },
      // T271.3: a token's: the block's shaders on a token's vectors, the states and rows the block's (a token is a block
      // of one: its q, k and v apart in a room of one token each), and the matrices of the normed stream written as they are
      async tokens({ params }) {
        const t = { mixed: buffer(m, mixed * 4), convolved: buffer(m, mixed * 4), z: buffer(m, read * 4), work: buffer(m, 2 * valueHeads * 4), out: buffer(m, read * 4) };
        const u = { wqkv: params(mixed, plan.dim), wz: params(read, plan.dim), wout: params(plan.dim, read),
          gates: [...Array(places)].map((_, p) => uniform(m, new Uint32Array([plan.dim, valueHeads, p * valueHeads, 0]))),
          conv: [...Array(places)].map((_, p) => uniform(m, new Uint32Array([mixed, taps, p * taps * mixed, keys, 1, 0, 0, 0]))),
          rule: [...Array(places)].map((_, p) => uniform(m, new Uint32Array([keyHeads, valueHeads, keyDim, valueDim, 1, p * valueHeads * keyDim * valueDim, 0, 0]))),
          qNorm: headNorm(m, keyDim, 0, 1e-6 / keyDim, 0), kNorm: headNorm(m, keyDim, keyDim, 1e-6 / keyDim, keyHeads),
          norm: [...Array(places)].map((_, p) => headNorm(m, valueDim, p * valueDim, plan.eps, 0)), read: uniform(m, new Uint32Array([read, 0, 0, 0])),
          quantizeZ: uniform(m, new Uint32Array([read, read, 0, 0])) };
        return {
          // a layer l at its place among those layers: the stream normed into xb (a dispatch of its own: the gates read it
          // as floats), wqkv's and wz's of it, the gates, the convolution and its rows moved on by the token, the l2 norms
          // of q and k, the delta rule over the place's state, the norm of every value head, z's gate, the matrix out
          step({ l, place, form, matrix, plain, joining }) {
            const g = m.tokens, b = m.block, W = m.weights, V = W.vectors!, P = form.pipes;
            const normed = plain(g.xb, g.u.quantizeNormed, plan.dim), gatedZ = plain(t.z, u.quantizeZ, read);
            return [[b.norm, bind(m, b.norm, [g.h, V.attention, g.xb, g.u.norm[l], g.step]), 1, 1], ...normed.quantize,
              matrix(P.write, W.matrices.wqkv.pieces[0].layers[place], normed.input, u.wqkv, mixed, [[5, t.mixed]]),
              matrix(P.write, W.matrices.wz.pieces[0].layers[place], normed.input, u.wz, read, [[5, t.z]]),
              [gates, bind(m, gates, [g.xb, V.wb, V.wa, V.dtBias, V.decay, t.work, u.gates[place], g.step]), valueHeads, 1],
              [convolving, bind(m, convolving, [t.mixed, V.conv, rows, t.convolved, u.conv[place], g.step]), groups, 1],
              [moving, bind(m, moving, [t.mixed, rows, u.conv[place], g.step]), groups, 1],
              [b.headNorm, bind(m, b.headNorm, [t.convolved, units, u.qNorm, g.step]), keyHeads, 1],
              [b.headNorm, bind(m, b.headNorm, [t.convolved, units, u.kNorm, g.step]), keyHeads, 1],
              [rule, bind(m, rule, [t.convolved, t.work, states, t.out, u.rule[place], g.step]), valueHeads, 1],
              [b.headNorm, bind(m, b.headNorm, [t.out, V.deltaNorm, u.norm[place], g.step]), valueHeads, 1],
              [swiglu, bind(m, swiglu, [t.z, t.out, u.read, g.step]), Math.ceil(read / 64), 1],
              ...gatedZ.quantize, ...joining(V.attentionPost, W.matrices.wout.pieces[0].layers[place], gatedZ.input, u.wout)] satisfies Pass[];
          },
        };
      },
      // T271.3: a first layer of this kind goes on from rows of random numbers, ±1, and a state of random numbers, ±0.1,
      // both of its place 0; and it keeps no keys and values. As the CPU has it (linear.ts): mixed = wqkv · xn and z =
      // wz · xn; the gates beta = sigmoid(wb · xn) and decay = exp(decay × softplus(wa · xn + dt_bias)) of every value
      // head; the convolution of mixed over the rows (rows[1] to rows[taps - 1], then mixed) with SiLU; q and k each head
      // over the root of its sum of squares + 1e-6, q by sqrt(key dim) more; the delta rule over the state from the value
      // head's key head; the norm of every value head, times silu(z); wout of that. Its rows after: rows[1] to
      // rows[taps - 1], then mixed; its state after, the rule's
      first(owned) {
        const { dim } = plan, stateSize = valueHeads * keyDim * valueDim;
        const rows0 = new Float32Array(taps * mixed).map(() => Math.random() * 2 - 1), state0 = new Float32Array(stateSize).map(() => (Math.random() * 2 - 1) / 10);
        m.device.queue.writeBuffer(rows, 0, rows0);
        m.device.queue.writeBuffer(states, 0, state0);
        const readRows = buffer(m, taps * mixed * 4, MAP_READ | COPY_DST, owned), readState = buffer(m, stateSize * 4, MAP_READ | COPY_DST, owned);
        const back = async (from: GPUBuffer) => {
          await from.mapAsync(MAP_READ);
          const got = new Float32Array(from.getMappedRange().slice(0));
          from.unmap();
          return got;
        };
        return {
          copies(encoder) {
            encoder.copyBufferToBuffer(rows, 0, readRows, 0, taps * mixed * 4);
            encoder.copyBufferToBuffer(states, 0, readState, 0, stateSize * 4);
          },
          async then({ xn, matmul, vectorOf, rms, behind, off, largest, line }) {
            const rowsAfter = await back(readRows), stateAfter = await back(readState);
            const mixedOut = matmul("wqkv", xn), z = matmul("wz", xn), wb = vectorOf("wb")!, wa = vectorOf("wa")!, dt = vectorOf("dtBias")!, decays = vectorOf("decay")!;
            const weights = vectorOf("conv")!, norms = vectorOf("deltaNorm")!;
            const dot = (w: Float64Array, h: number) => xn.reduce((sum, value, i) => sum + w[h * dim + i] * value, 0);
            const beta = Array.from({ length: valueHeads }, (_, h) => 1 / (1 + Math.exp(-dot(wb, h))));
            const decay = Array.from({ length: valueHeads }, (_, h) => {
              const step = dot(wa, h) + dt[h];
              return Math.exp(decays[h] * (step > 20 ? step : Math.log1p(Math.exp(step))));
            });
            const convolvedValues = Float64Array.from({ length: mixed }, (_, c) => {
              let sum = 0;
              for (let j = 0; j < taps; j++) sum += weights[j * mixed + c] * (j < taps - 1 ? rows0[(j + 1) * mixed + c] : mixedOut[c]);
              return sum / (1 + Math.exp(-sum));
            });
            const unit = (at: number, scale: number) => {
              const part = convolvedValues.subarray(at, at + keyDim), s = 1 / Math.sqrt(part.reduce((sum, value) => sum + value * value, 0) + 1e-6);
              return part.map((value) => value * s * scale);
            };
            const state = Float64Array.from(state0), readOut = new Float64Array(read);
            for (let h = 0; h < valueHeads; h++) {
              const key = Math.floor(h / (valueHeads / keyHeads)), q = unit(key * keyDim, 1 / Math.sqrt(keyDim)), k = unit(keys + key * keyDim, 1);
              for (let j = 0; j < valueDim; j++) {
                const at = h * keyDim * valueDim + j;
                let kv = 0;
                for (let i = 0; i < keyDim; i++) kv += k[i] * (state[at + i * valueDim] * decay[h]);
                const d = (convolvedValues[2 * keys + h * valueDim + j] - kv) * beta[h];
                let sum = 0;
                for (let i = 0; i < keyDim; i++) {
                  state[at + i * valueDim] = state[at + i * valueDim] * decay[h] + k[i] * d;
                  sum += q[i] * state[at + i * valueDim];
                }
                readOut[h * valueDim + j] = sum;
              }
              readOut.set(rms(readOut.subarray(h * valueDim, (h + 1) * valueDim), norms), h * valueDim);
            }
            const gated = readOut.map((value, i) => value * (z[i] / (1 + Math.exp(-z[i]))));
            const o = behind(matmul("wout", gated), "attentionPost");
            const moved = rowsAfter.subarray(0, (taps - 1) * mixed).every((value, i) => value === rows0[mixed + i]);
            const rowOff = off(rowsAfter.subarray((taps - 1) * mixed), mixedOut) / largest(mixedOut);
            const stateOff = off(stateAfter, state) / largest(state);
            const wrong = !moved || !(rowOff <= line) || !(stateOff <= line)
              ? `the first layer's rows are ${moved ? `${rowOff.toExponential(2)} of their largest from JavaScript's` : "not moved on by one"} and its state ${stateOff.toExponential(2)} (line ${line})` : null;
            return { o, wrong };
          },
        };
      },
    };
  },
};
