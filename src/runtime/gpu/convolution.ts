// gpu/convolution.ts (T.545, T.568.2): an LFM2's convolution layers on the GPU, a kind of gpu/kinds.ts's table: in a block
// the matrix in (from the normed stream into mixed: B, C and z, 3 dim a token), SHORT_CONV (into xb: h = B × z over the
// rows the layer goes on from, times C) and CONV_STATE (the rows moved on by the block's tokens), the matrix out; a
// token's step the same on a token's vectors; the rows the layers go on from, taps rows of dim a layer, place after
// place, as the CPU keeps them (forward/convolution.ts), and the check of a first layer of this kind.
// (a module of the model's GPU worker, gpu.ts, which a bundler links it into)
import { STORAGE, COPY_DST, COPY_SRC, MAP_READ, within, buffer, uniform, pipelineOf, bind, dispatch } from "./device.ts";
import { ENGINE } from "../shaders/engine.ts";
import type { GpuKind } from "./kinds.ts";
import type { Pass } from "./device.ts";

export const convolution: GpuKind = {
  matrices: () => ({ mix: ["win"], join: ["wout"] }),
  async block(m) {
    const { plan } = m, B = plan.batch, { dim } = plan, { taps } = plan.convolution!, groups = Math.ceil(dim / 64);
    const [short, state] = await within(Promise.all([ENGINE.SHORT_CONV, ENGINE.CONV_STATE].map((code) => pipelineOf(m, code))), "the convolution's shaders");
    const bytes = plan.kinds.filter((kind) => kind === "convolution").length * taps * dim * 4;
    const mixed = buffer(m, B * 3 * dim * 4), rows = buffer(m, bytes, STORAGE | COPY_DST | COPY_SRC);
    return {
      states: [{ buffer: rows, bytes }],
      // a layer's matrix in, its two shaders on the layer's taps and rows (at its place), its matrix out
      layer({ place, product, out }) {
        const b = m.block, V = m.weights.vectors!, shape = uniform(m, new Uint32Array([dim, taps, place * taps * dim, 0]));
        const into = product("win", b.xb, mixed), convolved = bind(m, short, [mixed, V.conv!, rows, b.xb, shape, b.step]);
        const moved = bind(m, state, [mixed, rows, shape, b.step]), wout = out("wout", b.xb);
        // (the last layer leaves the convolution alone: the rows are all a prompt's token leaves there)
        return ({ pass, count, last, multiplied, quantized }) => {
          multiplied(into);
          if (!last) dispatch(pass, short, convolved, groups, count);
          dispatch(pass, state, moved, groups);
          if (last) return true;
          quantized(b.quantizeXb);
          multiplied(wout);
          return false;
        };
      },
      // T.545.2: a token's: the matrix in from the normed stream into a token's mixed, SHORT_CONV and CONV_STATE on the
      // place's taps and rows (as a prompt's block binds them: the rows a block left, its own after), the matrix out
      async tokens({ params, layers }) {
        const token = buffer(m, 3 * dim * 4), mix = layers.map((at) => params(3 * dim, dim, 0, at)), wout = params(dim, dim);
        const places = plan.kinds.filter((kind) => kind === "convolution").length;
        const conv = [...Array(places)].map((_, p) => uniform(m, new Uint32Array([dim, taps, p * taps * dim, 0])));
        return {
          step({ l, place, matrix, plain, joining, normed }) {
            const g = m.tokens!, W = m.weights, V = W.vectors!, convolved = plain(g.xb, g.u.quantizeNormed, dim);
            return [...normed.norm, ...normed.quantize,
              matrix("mix", W.matrices.win!.pieces[0].layers[place], normed.input, mix[l], 3 * dim, [[5, token]]),
              [short, bind(m, short, [token, V.conv!, rows, g.xb, conv[place], g.step]), groups, 1],
              [state, bind(m, state, [token, rows, conv[place], g.step]), groups, 1],
              ...convolved.quantize, ...joining(V.attentionPost, W.matrices.wout!.pieces[0].layers[place], convolved.input, wout)] satisfies Pass[];
          },
        };
      },
      // T.545.2: a first layer of this kind goes on from rows of random numbers, ±1, whose last taps - 1 it reads and moves
      // on by one, its own row last; and it keeps no keys and values. As the CPU has it (convolution.ts): mixed = win · xn,
      // of which B, C and z; h = B × z; C × (taps[0] × rows[1] + ... + taps[taps - 2] × rows[taps - 1] + taps[taps - 1] × h);
      // wout of that. Its rows after: rows[1] to rows[taps - 1], then h
      first(owned) {
        const rows0 = new Float32Array(taps * dim).map(() => Math.random() * 2 - 1);
        m.device.queue.writeBuffer(rows, 0, rows0);
        const read = buffer(m, taps * dim * 4, MAP_READ | COPY_DST, owned);
        return {
          copies: (encoder) => encoder.copyBufferToBuffer(rows, 0, read, 0, taps * dim * 4),
          async then({ xn, matmul, vectorOf, behind, off, largest, line }) {
            await read.mapAsync(MAP_READ);
            const after = new Float32Array(read.getMappedRange().slice(0));
            read.unmap();
            const mixedOut = matmul("win", xn), weights = vectorOf("conv")!, now = new Float64Array(dim), out = new Float64Array(dim);
            for (let c = 0; c < dim; c++) {
              now[c] = mixedOut[c] * mixedOut[2 * dim + c];
              let sum = 0;
              for (let j = 0; j < taps; j++) sum += weights[j * dim + c] * (j < taps - 1 ? rows0[(j + 1) * dim + c] : now[c]);
              out[c] = mixedOut[dim + c] * sum;
            }
            const o = behind(matmul("wout", out), "attentionPost");
            const moved = after.subarray(0, (taps - 1) * dim).every((value, i) => value === rows0[dim + i]), rowOff = off(after.subarray((taps - 1) * dim), now) / largest(now);
            const wrong = !moved || !(rowOff <= line) ? `the first layer's rows are ${moved ? `${rowOff.toExponential(2)} of their largest from JavaScript's (line ${line})` : "not moved on by one"}` : null;
            return { o, wrong };
          },
        };
      },
    };
  },
};
