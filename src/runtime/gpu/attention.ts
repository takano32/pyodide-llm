// gpu/attention.ts (T568.2): the attending layers on the GPU, a kind of gpu/kinds.ts's table: in a block q, k and v of
// the normed stream (T153: Qwen2's biases, Qwen3's norms of the heads; T271.2: a Qwen3.5's gate of the output, of the
// same stream), RoPE into the GPU's own keys and values (block.ts's cache, an attending layer's by its place), the flash
// attention, the gate, o; a token's step the same in a fused form (T152, T226), and the check of an attending first
// layer. What they keep is the keys and values, which go back and forth with the CPU's as they are (block.ts's keysIn,
// keysOut; tokens.ts's runTokens): no state is staged.
// (a module of the model's GPU worker, gpu.ts, which a bundler links it into)
import { COPY_DST, STORAGE, turnedAt, within, buffer, uniform, normShape, copyIn, validated, pipelineOf, bind, dispatch } from "./device.ts";
import { attentionPasses } from "./tokenattention.ts";
import { FLASH_Q_TILE } from "../shaders/attention.ts";
import { ENGINE } from "../shaders/engine.ts";
import type { GpuKind, Added } from "./kinds.ts";
import type { Entry } from "./tokens.ts";
import type { Pass } from "./device.ts";
import type { Vectors } from "../contracts/gpu/protocol.ts";

// (the norm of every head of q or of k, HEAD_NORM's bind group and the heads)
type HeadNorm = { group: GPUBindGroup; heads: number };

export const attention: GpuKind = {
  cached: true,
  // (T271.3: a Qwen3.5's gate of the output, a matrix of the normed stream written as it is)
  writes: (plan) => (plan.matrices.wg ? ["mix"] : []),
  async block(m) {
    const { plan } = m, B = plan.batch, qDim = plan.heads * plan.headSize;
    // T271.2: the gate of a Qwen3.5's full-attention layers' output
    const outputGate = plan.matrices.wg ? { pipeline: await within(pipelineOf(m, ENGINE.OUTPUT_GATE), "the output's gate"), gated: buffer(m, B * qDim * 4) } : null;
    return {
      states: [],
      // a layer's bind groups: q, k and v (T153: a bias of each, Qwen2's, or the norm of every head of q and of k,
      // Qwen3's, where the model has them; T154: GPT-2's biases), the gate of the output, o and its bias
      layer({ l, place, product, out, added }) {
        const b = m.block, V = m.weights.vectors!;
        const headNorm = (of: GPUBuffer, name: keyof Vectors, heads: number) =>
          (V[name] ? { group: bind(m, b.headNorm, [of, V[name], normShape(m, plan.headSize, (plan.vectors[name]!.of ? place : l) * plan.headSize, plan.eps), b.step]), heads } : null);
        const q = product("wq", b.xb, b.q), k = product("wk", b.xb, b.k), v = product("wv", b.xb, b.v);
        const biases = [added(b.q, "bq"), added(b.k, "bk"), added(b.v, "bv")].filter(Boolean) as Added[];
        const headNorms = [headNorm(b.q, "qNorm", plan.heads), headNorm(b.k, "kNorm", plan.kvHeads)].filter(Boolean) as HeadNorm[];
        const gate = outputGate ? { into: product("wg", b.xb, outputGate.gated),
          group: bind(m, outputGate.pipeline, [b.xb, outputGate.gated, uniform(m, new Uint32Array([qDim, 0, 0, 0])), b.step]) } : null;
        const o = out("wo", b.xb), oBias = added(b.x, "bo"), flash = m.forms.attention;
        // T153: as the CPU has it (forward.ts), the biases, then the norms of the heads, then RoPE; the attention over the
        // cache (block.ts's, as it is at the block: it grows), the gate, o
        return ({ pass, count, last, multiplied, quantized, biased, ffnNorm }) => {
          const cache = b.cache!;
          multiplied(q);
          multiplied(k);
          multiplied(v);
          if (gate) multiplied(gate.into);  // (T271.2: a Qwen3.5's gate of the output)
          biases.forEach(biased);
          for (const head of headNorms) dispatch(pass, b.headNorm, head.group, head.heads, count);
          dispatch(pass, b.rope, cache.rope[place], count);
          if (last) return true;  // the keys and values are all a prompt's token leaves
          // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, before o adds to it
          if (plan.parallel) ffnNorm();
          dispatch(pass, flash.pipeline, cache.attention[place], plan.heads * Math.ceil(count / FLASH_Q_TILE));
          if (gate) dispatch(pass, outputGate!.pipeline, gate.group, Math.ceil(qDim / 64), count);
          quantized(b.quantizeAttention);
          multiplied(o);
          biased(oBias);
          return false;
        };
      },
      // T226: a token's. A model whose q, k and v are not ready for RoPE as their matrix leaves them (T153: Qwen2's
      // biases, Qwen3's norms of the heads) has the matrix write the three as they are into qkv (q's rows, then k's, then
      // v's), the prompt's ADD and HEAD_NORM change them there, and TOKEN_ROPE turns them into q and the cache. The biases
      // of a layer are one vector then, in the rows' order (one ADD for the three); k's heads are the rows after q's
      async tokens({ params, norm, layers }) {
        const kvDim = plan.kvHeads * plan.headSize, qkvRows = qDim + 2 * kvDim, { bq, bk, bv, qNorm } = plan.vectors;
        // (T261.4.2: the tiles' Params of each window the layers attend: 0 for every position, a Gemma 3's local layers')
        const flashes = new Map<number, GPUBuffer>();
        const flashOf = (window: number) => {
          if (!flashes.has(window)) {
            const flash = new ArrayBuffer(16);
            new Uint32Array(flash, 0, 2).set([plan.heads, plan.kvHeads]);
            new Float32Array(flash, 8, 1)[0] = 1 / Math.sqrt(plan.headSize);
            new Uint32Array(flash, 12, 1)[0] = window;
            flashes.set(window, uniform(m, flash));
          }
          return flashes.get(window)!;
        };
        const u = { flash: layers.map((_, l) => flashOf(plan.windows[l] ?? 0)), o: params(plan.dim, qDim),
          qkv: layers.map((at, l) => params(qkvRows, plan.dim, 0, at, turnedAt(plan, l))),
          // QUANTIZE's (n, xStride) of the attention's output
          quantizeAttention: uniform(m, new Uint32Array([qDim, qDim, 0, 0])) };
        let rope: GPUComputePipeline | null = null, qkv: GPUBuffer | null = null, qkvBias: GPUBuffer | null = null, biasShapes: GPUBuffer[] | null = null;
        if (bq || qNorm) {
          rope = await within(validated(m, () => pipelineOf(m, ENGINE.TOKEN_ROPE)), "compiling a token's RoPE");
          qkv = buffer(m, qkvRows * 4);
        }
        if (bq) {
          qkvBias = buffer(m, plan.layers * qkvRows * 4, STORAGE | COPY_DST);
          for (let l = 0; l < plan.layers; l++) {
            let at = l * qkvRows * 4;
            for (const bias of [bq, bk!, bv!]) {
              copyIn(m, qkvBias, bias.at + l * bias.size * 4, bias.size * 4, at);
              at += bias.size * 4;
            }
          }
          biasShapes = layers.map((_, l) => uniform(m, new Uint32Array([qkvRows, l * qkvRows, 0, 0])));
        }
        // (T545.2: a layer's at its place among the attending layers, an LFM2's)
        const headNorms = qNorm ? layers.map((_, l) => [0, plan.heads].map((first) => norm((qNorm.of ? plan.places[l] : l) * plan.headSize, plan.headSize, first))) : null;
        // (T271.3: a Qwen3.5's gate of its attending layers' output, from the normed stream: wg's Params a layer with its
        // norm, OUTPUT_GATE's size)
        const gated = plan.matrices.wg ? { buffer: buffer(m, qDim * 4), params: layers.map((at) => params(qDim, plan.dim, 0, at)), size: uniform(m, new Uint32Array([qDim, 0, 0, 0])) } : null;
        return {
          // an attending layer l, at its place among the attending layers: q, k and v, RoPE into the cache, the attention, o
          step({ l, place, form, matrix, plain, joining, normed, ffnNorm, biased, bindAt, positions }) {
            const g = m.tokens, b = m.block, W = m.weights, V = W.vectors!, P = form.pipes, cache = b.cache!;
            const [o] = W.matrices.wo.pieces, attended = plain(g.att, u.quantizeAttention, qDim);
            const added = (to: GPUBuffer, bias: GPUBuffer, shape: GPUBuffer, n: number): Pass => [b.add, bind(m, b.add, [to, bias, shape, g.step]), Math.ceil(n / 64), 1];
            // where q, k and v go turned: q, and the cache at the Step's position (T261.4.2: a local layer by its own table)
            const turned: Entry[] = [[5, g.q], [6, cache.keys[place]], [7, cache.values[place]], [8, plan.local.includes(l) ? W.localAngleTable! : W.angleTable!], [9, g.step]];
            return [...normed.norm, ...normed.quantize,
              // T226: turned on the matrix's write (Llama: one dispatch), or written as they are where something comes before
              // RoPE: as the CPU has it and a prompt's block (T153), the biases, the norms of the heads of q and of k, RoPE
              matrix(P.qkv, W.joined[l].qkv, normed.input, u.qkv[l], qkvRows, qkv ? [[5, qkv]] : turned),
              // (T271.3: a Qwen3.5's gate of the output, of the same normed stream)
              ...(gated ? [matrix(P.mix, W.matrices.wg.pieces[0].layers[place], normed.input, gated.params[l], qDim, [[5, gated.buffer]])] : []),
              ...(qkvBias ? [added(qkv!, qkvBias, biasShapes![l], qkvRows)] : []),
              ...(headNorms ? ([[V.qNorm, plan.heads], [V.kNorm, plan.kvHeads]] satisfies [GPUBuffer, number][]).map(([weights, heads], i): Pass =>
                [b.headNorm, bind(m, b.headNorm, [qkv!, weights, headNorms[l][i], g.step]), heads, 1]) : []),
              ...(qkv ? [[rope!, bindAt(rope!, [[2, qkv], [3, u.qkv[l]], ...turned]), 1, 1]] satisfies Pass[] : []),
              // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, before o adds to it (into xb, which
              // q, k and v have read by now; a norm on the matrix's read would come too late: chooseTokens refuses the pair)
              ...(plan.parallel ? ffnNorm : []),
              ...attentionPasses(m, g.attention, { q: g.q, keys: cache.keys[place], values: cache.values[place], out: g.att, parts: g.parts, params: g.params,
                flash: u.flash[l], window: plan.windows[l] ?? 0, step: g.step }, plan.heads, positions),
              ...(gated ? [[outputGate!.pipeline, bind(m, outputGate!.pipeline, [g.att, gated.buffer, gated.size, g.step]), Math.ceil(qDim / 64), 1]] satisfies Pass[] : []),
              ...attended.quantize, ...joining(V.attentionPost, o.layers[place], attended.input, u.o), ...biased(g.h, "bo", l)];
          },
        };
      },
      // T545.2: an attending first layer: its q, k and v, its attention over position 0 (the cache's row 0, random) and
      // its own, o. As the CPU has it (forward.ts), the biases (Qwen2's; T154's), then the norms of the heads of q and k
      // (Qwen3's), then RoPE (all of a head, T154: a part of it, or none); the first layer's of each, where the model has them
      first() {
        return {
          copies() {},
          async then({ xn, pos, matmul, vectorOf, plus, rms, behind, row0, got, held }) {
            const { headSize } = plan, half = headSize / 2;
            const floats = (address: number, n: number) => Float64Array.from(new Float32Array(m.memory.buffer, address, n));
            const [q, k, v] = ([["wq", "bq"], ["wk", "bk"], ["wv", "bv"]] satisfies [string, keyof Vectors][]).map(([name, bias]) => plus(matmul(name, xn), bias));
            for (const [name, x] of [["qNorm", q], ["kNorm", k]] satisfies [keyof Vectors, Float64Array][]) {
              const weights = vectorOf(name);
              for (let at = 0; weights && at < x.length; at += headSize) x.set(rms(x.subarray(at, at + headSize), weights), at);
            }
            // (T261.4.2: a local layer's angles are of its own tables)
            const [cosAt, sinAt] = plan.local.includes(0) ? [plan.localCos, plan.localSin] : [plan.cos, plan.sin];
            const cos = floats(cosAt + pos * half * 4, half), sin = floats(sinAt + pos * half * 4, half);
            const turn = (vector: Float64Array) => {
              for (let at = 0; at < vector.length; at += 2) {
                const i = (at % headSize) / 2;
                if (at % headSize >= turnedAt(plan, 0)) continue;
                const [x, y] = [vector[at], vector[at + 1]];
                vector[at] = x * cos[i] - y * sin[i];
                vector[at + 1] = x * sin[i] + y * cos[i];
              }
            };
            turn(q);
            turn(k);
            const keys = [row0[0], held(k, got[0])], values = [row0[1], held(v, got[1])];
            const att = new Float64Array(qDim), group = plan.heads / plan.kvHeads;
            for (let head = 0; head < plan.heads; head++) {
              const kvAt = Math.floor(head / group) * headSize, at = head * headSize;
              const scores = keys.map((key) => {
                let sum = 0;
                for (let d = 0; d < headSize; d++) sum += q[at + d] * key[kvAt + d];
                return sum / Math.sqrt(headSize);
              });
              const top = Math.max(...scores), weights = scores.map((score) => Math.exp(score - top)), total = weights[0] + weights[1];
              for (let d = 0; d < headSize; d++) att[at + d] = (weights[0] * values[0][kvAt + d] + weights[1] * values[1][kvAt + d]) / total;
            }
            // (T271.3: a Qwen3.5's gate of the output, of the normed stream)
            if (plan.matrices.wg) {
              const gate = matmul("wg", xn);
              att.forEach((value, i) => { att[i] = value / (1 + Math.exp(-gate[i])); });
            }
            return { o: behind(plus(matmul("wo", att), "bo"), "attentionPost"), k, v, wrong: null };
          },
        };
      },
    };
  },
};
