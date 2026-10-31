// tests/gpu-tokens-check.ts (T.465.3): the model's GPU worker choosing the forms of a generated token's steps, on the
// stand-in device whose checks' dispatches are answered as the shaders would (tests/fakes/gpu-model.ts: the answers'
// arithmetic is tests/gpu_answers.py's, in NumPy). As gpu-forms-check.ts does for a prompt's forms.
//   - T.465.3, a token's attention (src/runtime/gpu/tokenattention.ts's chooseTokenAttention()): llama.cpp's decode form
//     with subgroups (where the device has them) and without, and the prompt's tiles, each checked right at 40, 70, 300
//     and 1100 positions (one part, two, and as many as the decode form takes, the reduce over them), timed, one taken
//     (the decode form's parts and Params made with it); a model with a window (a Gemma 3) through it too. Each check
//     bites: the decode form off a little in one head, or its reduce writing nothing (a token of more than one part reads
//     nothing then), is wrong and the others are taken; off inside the line is right.
// Not held here: that the fastest is taken (the timing's clock is the real one), a token's layer (T.465.4).
//   node tests/gpu-tokens-check.ts
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";
import { opened } from "./fakes/gpu-model.ts";
import { gpuAnswers } from "./fakes/gpu-answers.ts";

const forms_ = await import(runtimeUrl("gpu/forms.ts"));
const tokenAttention_ = await import(runtimeUrl("gpu/tokenattention.ts"));
const { flashVec } = await gpuAnswers();

// the decode form's output (binding 4) off where spoiled() says, after the right answer wrote it
const off = (model, spoiled) => (entries, x, y) => {
  model(entries, x, y);
  const { resource } = entries.find((entry) => entry.binding === 4), buffer = resource.buffer ?? resource;
  spoiled(new Float32Array(buffer.data.buffer, buffer.data.byteOffset, buffer.size / 4));
};
// a model opened with its prompt's attention chosen (the tiles a token's attention may take), then a token's
async function chosen(options = {}) {
  const { m } = await opened(options);
  await forms_.chooseAttention(m);
  await tokenAttention_.chooseTokenAttention(m);
  return m;
}
const VEC = "llama.cpp flash_attn_vec", TILES = "the prompt's attention tiles";

for (const [model, features] of [["llama", []], ["llama", ["subgroups"]], ["gemma3", []]]) {
  const where = `${model}${features.length ? ", subgroups" : ""}`;
  const m = await chosen({ model, features });
  const { attention, attentions, parts, params } = m.tokenAttention;
  const names = [...(features.includes("subgroups") ? [`${VEC}, subgroups`] : []), VEC, TILES];
  assert.deepEqual(attentions.filter(({ none }) => none), [], `${where}: no attention of a token wrong`);
  assert.deepEqual(attentions.map(({ name }) => name), names, `${where}: every candidate right and timed`);
  assert.ok(names.includes(attention.name), `${where}: one taken`);
  assert.equal(Boolean(parts && params), !attention.tiles, `${where}: the decode form's parts and Params made with it, none with the tiles`);
  console.log(`ok: ${where}: ${names.join(", ")} checked right${model === "gemma3" ? " (through a window too)" : ""}, ${attention.name} taken`);
}

// each check bites, and takes what is off inside its line
{
  // the decode form off by 1% of a value in its second head (its line 0.4% of the largest)
  const m = await chosen({ spoil: { vec: off(flashVec(16), (out) => { out[16] += 0.05; }) } });
  const vec = m.tokenAttention.attentions.find(({ name }) => name === VEC);
  assert.match(vec.none ?? "", /^wrong: its output is .* of the largest value from JavaScript's at 40 positions/, "the decode form a little off in one head");
  assert.equal(m.tokenAttention.attention.name, TILES, "the tiles taken");
  // its reduce writing nothing: a token of one part is right, of two it is not
  const idle = await chosen({ spoil: { reduce: () => {} } });
  assert.match(idle.tokenAttention.attentions.find(({ name }) => name === VEC).none ?? "", /^wrong: .* at 70 positions/, "the reduce writing nothing");
  // off by 0.1%: inside the line
  const near = await chosen({ spoil: { vec: off(flashVec(16), (out) => { out[16] += 0.001; }) } });
  assert.deepEqual(near.tokenAttention.attentions.filter(({ none }) => none), [], "off inside the line: right");
}
console.log("ok: each check bites: the decode form a little off in a head, or its reduce writing nothing, is wrong and the tiles taken; off inside the line is right");
console.log("gpu-tokens-check: ok");
