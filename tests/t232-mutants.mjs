// T232 (the throwaway branch t232-broken only): the ternary GPU path broken on purpose, one way at a time, to see
// that tests/gpu-check.mjs fails each where it should. `node tests/t232-mutants.mjs <name>` changes the files of the
// working tree for that mutant (git checkout restores them); `node tests/t232-mutants.mjs` lists the names.
// Every mutant's text must be found exactly once, or this fails (a mutant that changes nothing would pass for the
// wrong reason).
import fs from "node:fs";

const SHADERS = "public/shaders.js", GPU = "public/gpu.js", FORWARD = "public/forward.js";
const MUTANTS = {
  // the code's mapping swapped (+1 for -1) where a matrix unpacks its codes
  "mapping-swapped": [[SHADERS, "  return (bits + vec4<u32>(0x7f7f7f7fu)) ^ vec4<u32>(0x80808080u);", "  return (vec4<u32>(0x81818181u) - bits) ^ vec4<u32>(0x80808080u);"]],
  // the codes left unsigned (the fork's form without its sum taken off: the analogue of the dropped -Σa)
  "codes-unsigned": [[SHADERS, "  return (bits + vec4<u32>(0x7f7f7f7fu)) ^ vec4<u32>(0x80808080u);", "  return bits;"]],
  // the tiles' scale of the wrong group (the int8 index: the activations' group of 32)
  "tile-scale": [[SHADERS, "    scale_B[row] = scales_b[b_global * (shape.perRow / 4u) + kidx_v / 8u];", "    scale_B[row] = scales_b[b_global * (shape.perRow / 4u) + kidx_v / 2u];"]],
  // a token's matrix: the scale of the group before (but the first)
  "token-scale": [[SHADERS, "                let own_scale_b = scales_b[b_global * K128 + k_offset / 4u];", "                let own_scale_b = scales_b[b_global * K128 + max(k_offset / 4u, 1u) - 1u];"]],
  // up's rows take gate's scales (SwiGLU's second matrix)
  "up-scale": [[SHADERS, "                let up_scale_b = scales_b[up_global * K128 + k_offset / 4u];", "                let up_scale_b = scales_b[b_global * K128 + k_offset / 4u];"]],
  // the last group of 128 of a row left out of a token's matrix
  "token-last-group": [[SHADERS, "            if (b_global < params.rows && k_offset < K32)\n            {\n                let b_offset = b_global * K32 + k_offset;\n                let own_scale_b = scales_b[b_global * K128",
    "            if (b_global < params.rows && k_offset < K32 - 4u)\n            {\n                let b_offset = b_global * K32 + k_offset;\n                let own_scale_b = scales_b[b_global * K128"]],
  // the second word of a group of 32 read from the first (weights 16 to 31 are 0 to 15 again)
  "token-second-word": [[SHADERS, "                let own_b1 = ternary_packed(b[b_offset * 2 + 1]);", "                let own_b1 = ternary_packed(b[b_offset * 2]);"]],
  // a step's q, k and v of layer 0 in every layer (the wrong layer's weights)
  "step-layer-0-qkv": [[GPU, "      matrix(P.qkv, m.joined[l].qkv, attention.input, g.u.qkv[l], qkvRows, g.qkv ? [[5, g.qkv]] : turned),", "      matrix(P.qkv, m.joined[0].qkv, attention.input, g.u.qkv[l], qkvRows, g.qkv ? [[5, g.qkv]] : turned),"]],
  // a step's down matrix of layer 0 in every layer
  "step-layer-0-down": [[GPU, "      ...activated.quantize, matrix(P.add, down.layers[l], activated.input, g.u.down, plan.dim, [[5, g.h]]), ...biased(g.h, \"b2\", l));",
    "      ...activated.quantize, matrix(P.add, down.layers[0], activated.input, g.u.down, plan.dim, [[5, g.h]]), ...biased(g.h, \"b2\", l));"]],
  // a prompt's block: the gate matrix of layer 0 in every layer
  "block-layer-0-gate": [[GPU, "      gate: product(\"w1\", l, m.ffnInput, m.gate), gateBias: added(m.gate, \"b1\"),", "      gate: product(\"w1\", 0, m.ffnInput, m.gate), gateBias: added(m.gate, \"b1\"),"]],
  // a layer's bytes on the GPU from where an int8 layer would be (a matrix's second layer on, in memory)
  "layer-offset": [[FORWARD, "      const layer = (l) => [values + l * rows * rowBytes, scales + l * rows * (n / t.group) * 4, corrections + l * rows * (n / t.group) * 4];",
    "      const layer = (l) => [values + l * rows * (ternary ? n / 8 : rowBytes), scales + l * rows * (n / t.group) * 4, corrections + l * rows * (n / t.group) * 4];"]],
  // the classifier's pieces all written at the logits' start
  "classifier-piece-offset": [[GPU, "    const logitsOf = (piece) => ({ buffer: g.logits, offset: piece.first * 4, size: piece.rows * 4 });", "    const logitsOf = (piece) => ({ buffer: g.logits, offset: 0, size: piece.rows * 4 });"]],
  // a table's pieces after the first read from where an int8 table's would be
  "table-piece-source": [[GPU, "    else copyIn(m, values, valuesAt + first * rowBytes(spec), valueBytes);", "    else copyIn(m, values, valuesAt + first * spec.n, valueBytes);"]],
  // the embedding's row: the scale of a group of 32
  "embed-scale": [[SHADERS, "        let d = scales[(row * n + word * 16u) / 128u];", "        let d = scales[(row * n + word * 16u) / 32u];"]],
  // the embedding's row: the code's mapping swapped
  "embed-mapping": [[SHADERS, "            dst[out + word * 16u + k] = f32(i32((codes >> (2u * k)) & 3u) - 1) * d;", "            dst[out + word * 16u + k] = f32(1 - i32((codes >> (2u * k)) & 3u)) * d;"]],
  // the outlier channels' columns not added
  "columns-dropped": [[SHADERS, "    dst[row] = dst[row] + sum;\n}`;", "    dst[row] = dst[row] + 0.0 * sum;\n}`;"]],
  // the outlier channels left in the vector that is quantized (counted twice)
  "outliers-not-taken": [[SHADERS, "    picked[k] = x[channel];\n    x[channel] = 0.0;", "    picked[k] = x[channel];"]],
  // the outlier channels' columns: the scale of the group of 128 before
  "columns-word": [[SHADERS, "        let code = (table[row * words + channel / 16u] >> (2u * (channel % 16u))) & 3u;", "        let code = (table[row * words + channel / 16u] >> (2u * ((channel + 1u) % 16u))) & 3u;"]],
  // on the GPU alone: a layer's codes from where an int8 layer would be
  "alone-layer-offset": [[FORWARD, "      [t.offset + l * (ternary ? perLayer / 4 : perLayer), t.scales + (l * perLayer / t.group) * 4]) }];", "      [t.offset + l * (ternary ? perLayer / 8 : perLayer), t.scales + (l * perLayer / t.group) * 4]) }];"]],
  // on the GPU alone: a table's piece's bytes routed from where an int8 table's would be
  "alone-table-route": [[GPU, "    const [valuesAt, scalesAt] = spec.at, from = [valuesAt + first * rowBytes(spec), scalesAt + (first * rowBytes(spec) / group) * 4];",
    "    const [valuesAt, scalesAt] = spec.at, from = [valuesAt + first * rowBytes(spec), scalesAt + (first * spec.n / group) * 4];"]],
};

const name = process.argv[2];
if (!name) {
  console.log(Object.keys(MUTANTS).join(" "));
  process.exit(0);
}
if (!MUTANTS[name]) throw new Error(`no mutant ${name}`);
for (const [file, find, replace] of MUTANTS[name]) {
  const text = fs.readFileSync(file, "utf8"), count = text.split(find).length - 1;
  if (count !== 1) throw new Error(`${name}: its text is in ${file} ${count} times, not once`);
  fs.writeFileSync(file, text.replace(find, () => replace));
}
console.log(`mutant ${name} applied`);
