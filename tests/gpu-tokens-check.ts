// tests/gpu-tokens-check.ts (T.465.3): the model's GPU worker choosing the forms of a generated token's steps, on the
// stand-in device whose checks' dispatches are answered as the shaders would (tests/fakes/gpu-model.ts: the answers'
// arithmetic is tests/gpu_answers.py's, in NumPy). As gpu-forms-check.ts does for a prompt's forms.
//   - T.465.3, a token's attention (src/runtime/gpu/tokenattention.ts's chooseTokenAttention()): llama.cpp's decode form
//     with subgroups (where the device has them) and without, and the prompt's tiles, each checked right at 40, 70, 300
//     and 1100 positions (one part, two, and as many as the decode form takes, the reduce over them), timed, one taken
//     (the decode form's parts and Params made with it); a model with a window (a Gemma 3) through it too. Each check
//     bites: the decode form off a little in one head, its reduce writing nothing (a token of more than one part reads
//     nothing then), or the window left out (a Gemma 3's check through its window), is wrong and the others are taken;
//     off inside the line is right. The answers take the workgroups as many as the work's, no more, and the reduce reads
//     the parts the decode wrote, of the nwg and sums' start it had.
//   - T.465.4, a token's layer (tokenforms.ts's chooseTokens()) of a Llama: the float forms (with subgroups and without)
//     and the DP4A ones (the norm with the quantizer, and apart) each checked on the model's own first layer and head,
//     greedy and sampled with the penalty, timed, one taken (the made-up memory's numbers ones a model could have:
//     gpu-plans.ts's plausible). Each check bites: the stream off in one value (a float form's text: the DP4A forms stay
//     right), a greedy token not the largest logit; off inside the line is right.
// Not held here: that the fastest is taken (the timing's clock is the real one), the other models' layers (T.465.6).
//   node tests/gpu-tokens-check.ts
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";
import { opened, blockMade } from "./fakes/gpu-model.ts";
import { gpuAnswers } from "./fakes/gpu-answers.ts";

const forms_ = await import(runtimeUrl("gpu/forms.ts"));
const tokenAttention_ = await import(runtimeUrl("gpu/tokenattention.ts"));
const tokenForms_ = await import(runtimeUrl("gpu/tokenforms.ts"));
const { flashVec, fusedMatVec, dp4a, sample, quantize } = await gpuAnswers();

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
  // (T.465.3's review) a decode form that reads every position, its window left out: wrong through the window of a
  // Gemma 3's check, right for a Llama's, which has none
  const blind = { spoil: { vec: flashVec(16, { ignoring: true }) } };
  const gemma = await chosen({ model: "gemma3", ...blind }), llama = await chosen(blind);
  assert.match(gemma.tokenAttention.attentions.find(({ name }) => name === VEC).none ?? "", /^wrong: .* at 300 positions through a window of 37/, "a window left out");
  assert.deepEqual(llama.tokenAttention.attentions.filter(({ none }) => none), [], "no window: right");
}
console.log("ok: each check bites: the decode form a little off in a head, its reduce writing nothing, or its window left out, is wrong and the tiles taken; off inside the line is right");
// ---- T.465.4: a token's layer (tokenforms.ts's chooseTokens()): each form checked on the model's own first layer and
// head (the memory's numbers a model could have: opened's sane), greedy and sampled
// a Llama's prompt forms chosen and its block made, then a token's forms
async function tokens(options = {}) {
  const { m } = await opened({ sane: true, ...options });
  await forms_.chooseAttention(m);
  await forms_.chooseMatrices(m);
  blockMade(m);
  await tokenForms_.chooseTokens(m).catch((error) => { m.thrown = error.message; });
  return m;
}
// a model's output (binding at, a float32 array) off where spoiled() says, after the right one wrote
const outputOff = (model, at, spoiled) => (entries, x, y) => {
  model(entries, x, y);
  const { resource } = entries.find((entry) => entry.binding === at), buffer = resource.buffer ?? resource;
  spoiled(new Float32Array(buffer.data.buffer, buffer.data.byteOffset, buffer.size / 4));
};
const FLOAT = "llama.cpp, fused (T150)", DP4A = "DP4A, fused (T175)", FORMS = [FLOAT, `${FLOAT}, subgroups`, DP4A, `${DP4A}, the norms apart`];
{
  const m = await tokens({ features: ["subgroups"] });
  assert.equal(m.thrown, undefined);
  const { form, forms } = m.tokenForm;
  assert.deepEqual(forms.filter(({ none }) => none), [], "a Llama's token: no form wrong");
  assert.deepEqual(forms.map(({ name }) => name).sort(), [...FORMS].sort(), "every form right and timed");
  assert.ok(FORMS.includes(form.name));
  console.log(`ok: a Llama's token: ${FORMS.join(", ")} checked right on its first layer and head (greedy and sampled), ${form.name} taken`);
}
// each check bites
{
  // the down projection's add off by 1% in one value (the float forms' line 2e-3 of the stream's change)
  // (the float forms' text: the DP4A forms read their own, and stay right)
  const off1 = await tokens({ spoil: { fused: (right, output) => (output === "add" ? outputOff(right, 5, (out) => { out[3] *= 1.01; }) : right) } });
  const verdict = (name) => off1.tokenForm.forms.find((tried) => tried.name === name);
  assert.match(verdict(FLOAT).none ?? "", /^wrong: the first layer's stream is .* of its change from JavaScript's/, "a stream off in one value");
  assert.equal(verdict(DP4A).none, undefined, "the DP4A forms right");
  // the sampler drawing the second most likely token where the most likely is asked: wrong
  const greedy = await tokens({ spoil: { sample: (entries, x, y) => {
    sample(entries, x, y);
    const chosen = entries.find((entry) => entry.binding === 4).resource, words = new Uint32Array((chosen.buffer ?? chosen).data.buffer);
    words[0] = (words[0] + 1) % 96;
  } } });
  assert.match(greedy.thrown ?? "", /the greedy token is \d+, the largest logit's \d+/, "a greedy token not the largest logit");
  // off by 1e-5 of a value: inside the line
  const near = await tokens({ spoil: { fused: (right, output) => (output === "add" ? outputOff(right, 5, (out) => { out[3] *= 1 + 1e-5; }) : right) } });
  assert.deepEqual(near.tokenForm.forms.filter(({ none }) => none), [], "off inside the line: right");
}
console.log("ok: each check bites: a token's stream off in one value, a greedy token not the largest logit; off inside the line is right");
console.log("gpu-tokens-check: ok");
