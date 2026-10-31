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
//     right), a greedy token not the largest logit, a sampled token the CPU's walk does not reach, the penalty left off the
//     logits; off inside the line is right.
//   - T.465.6.1: a Qwen3's and a Gemma 3's token (biases, the norms of the heads and RoPE apart; GEGLU, the norms behind,
//     a window and a local layer's RoPE): every form right and every dispatch answered (answers.missed), and a head's
//     norm, a bias or a norm behind off is wrong. T.465.6.2: a GPT-2's and a GPT-NeoX's (LayerNorms with their biases,
//     GELU, learned positions, the parallel residual) the same, and a LayerNorm or a GELU off is wrong.
//   - T.465.6.3: a ternary model's token (matrices and table ternary: its own forms) and a token of each of the ten types
//     of llama.cpp's blocks (two on w2's layers a model): every form right, every dispatch answered, a product off wrong.
//   - T.465.6.4: an LFM2's token (its convolution layers) and a Qwen3.5's (its linear-attention layers and the gate of
//     its attending ones' output; and with its states in bundles): every form right, every dispatch answered, and the
//     convolution, its moved rows or the delta rule's state off is wrong.
//   - T.465.6.5: classifiers with outlier channels (int8 and ternary), and a wide Llama of blocks on every matrix and its
//     table (q, k and v apart, and joined): every form right, every dispatch answered, and their parts off wrong.
// Not held here: that the fastest is taken (the timing's clock is the real one); a model on the GPU alone is
// gpu-start-check.ts's.
//   node tests/gpu-tokens-check.ts
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";
import { opened, blockMade } from "./fakes/gpu-model.ts";
import { gpuAnswers } from "./fakes/gpu-answers.ts";

const forms_ = await import(runtimeUrl("gpu/forms.ts"));
const tokenAttention_ = await import(runtimeUrl("gpu/tokenattention.ts"));
const tokenForms_ = await import(runtimeUrl("gpu/tokenforms.ts"));
const { flashVec, sample, embed, normQuantize, rmsnorm, bias, layerNorm, gelu, shortConv, convState, deltaNet, deltaGates, deltaConv, deltaConvState, swiglu } = await gpuAnswers();

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
// (the largest |value| of an output)
const peak = (out) => out.reduce((most, value) => Math.max(most, Math.abs(value)), 0);
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
// T.465.6.1: a Qwen3 (biases of q, k and v, norms of the heads: q, k and v written, then ADD, HEAD_NORM and TOKEN_ROPE)
// and a Gemma 3 (GEGLU, the norms behind the attention and the FFN, a window and a local layer's own RoPE): every form
// right, every dispatch of theirs answered
for (const model of ["llama", "qwen3", "gemma3", "gpt2", "neox", "lfm2", "qwen35", "qwen35 (states in bundles)"]) {
  const { m, answers } = await (async () => {
    const opening = await opened({ model, sane: true });
    await forms_.chooseAttention(opening.m);
    await forms_.chooseMatrices(opening.m);
    blockMade(opening.m);
    await tokenForms_.chooseTokens(opening.m);
    return opening;
  })();
  assert.deepEqual(m.tokenForm.forms.filter(({ none }) => none), [], `${model}: no form wrong`);
  assert.deepEqual(answers.missed, [], `${model}: every dispatch answered`);
  if (model === "llama") continue;
  const said = { qwen3: "Qwen3", gemma3: "Gemma 3", gpt2: "GPT-2", neox: "GPT-NeoX", lfm2: "LFM2", qwen35: "Qwen3.5", "qwen35 (states in bundles)": "Qwen3.5 (its states in bundles)" }[model];
  console.log(`ok: a ${said}'s token: ${m.tokenForm.forms.map(({ name }) => name).join(", ")} checked right, every dispatch answered`);
}
{
  // a Qwen3's head norm off in one value, a bias off, a Gemma 3's norm behind off: the forms wrong
  // (value i of the output at binding at: of q, k and v written one after the other, 33 is k's second, which the check
  // reads as it is; q's error is spread by the attention)
  // (T.465.6.1's review: an error of q is spread by the attention, o and the FFN, and fell under a form's line one run in
  // five: the head norm's and the bias's go on a k row. The norm behind by 20% of the stream's largest: 5% left little
  // over the DP4A forms' line)
  const bump = (model, at, i = 1, share = 0.05) => (entries, x, y) => {
    model(entries, x, y);
    const r = entries.find((entry) => entry.binding === at).resource, buffer = r.buffer ?? r;
    const out = new Float32Array(buffer.data.buffer, buffer.data.byteOffset, buffer.size / 4);
    out[i] += share * peak(out);
  };
  for (const [model, spoil, what] of [["qwen3", { headNorm: bump(rmsnorm({ inPlace: true }), 0, 33) }, "a head's norm"], ["qwen3", { bias: bump(bias, 0, 33) }, "a bias"],
    ["gemma3", { normAdd: bump(rmsnorm({ add: true }), 2, 1, 0.2) }, "a norm behind"],
    // (T.465.6.2: a GPT-2's LayerNorm off in one value, a GPT-NeoX's GELU)
    ["gpt2", { layerNorm: (entries, x, y) => {
      // (the whole of it 10% off: one value off by 20% of the largest fell under a form's line one run in a few)
      layerNorm(entries, x, y);
      const r = entries.find((entry) => entry.binding === 3).resource, buffer = r.buffer ?? r;
      new Float32Array(buffer.data.buffer, buffer.data.byteOffset, buffer.size / 4).forEach((v, i, all) => { all[i] = v * 1.1; });
    } }, "a LayerNorm"], ["neox", { gelu: (entries, x, y) => {
      // (the whole of it 10% off: one value of the 256 the down projection reads stayed under the DP4A forms' line)
      gelu(entries, x, y);
      const r = entries.find((entry) => entry.binding === 0).resource, buffer = r.buffer ?? r;
      new Float32Array(buffer.data.buffer, buffer.data.byteOffset, buffer.size / 4).forEach((v, i, all) => { all[i] = v * 1.1; });
    } }, "a GELU"]]) {
    const m = await tokens({ model, spoil });
    assert.match(m.thrown ?? "", /no layer of a token is right on this GPU/, `${model}: ${what} off`);
  }
  console.log("ok: a Qwen3's head norm or bias off, a Gemma 3's norm behind, a GPT-2's LayerNorm, a GPT-NeoX's GELU: no form right");
}
// each check bites
{
  // the down projection's add off in one value by 1% of the stream's largest (the float forms' line 2e-3 of its change;
  // one value scaled by 1% fell under it where the value was small)
  // (the float forms' text: the DP4A forms read their own, and stay right)
  const off1 = await tokens({ spoil: { fused: (right, output) => (output === "add" ? outputOff(right, 5, (out) => { out[3] += 0.01 * peak(out); }) : right) } });
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
  // sampled (temperature over 0): a token the CPU's walk does not reach, or the penalty left off the logits
  const temperature = (entries) => { const r = entries.find((entry) => entry.binding === 6).resource; return new Float32Array((r.buffer ?? r).data.buffer)[4]; };
  const walked = await tokens({ spoil: { sample: (entries, x, y) => {
    sample(entries, x, y);
    if (!temperature(entries)) return;
    const chosen = entries.find((entry) => entry.binding === 4).resource, words = new Uint32Array((chosen.buffer ?? chosen).data.buffer);
    words[0] = (words[0] + 48) % 96;
  } } });
  assert.match(walked.thrown ?? "", /the sampled token is \d+, the CPU's walk reaches/, "a sampled token the walk does not reach");
  const unpenalized = await tokens({ spoil: { sample: (entries, x, y) => {
    const logits = entries.find((entry) => entry.binding === 0).resource, buffer = logits.buffer ?? logits, before = buffer.data.slice();
    sample(entries, x, y);
    if (temperature(entries)) buffer.data.set(before);
  } } });
  assert.match(unpenalized.thrown ?? "", /the penalized logits are .* from JavaScript's/, "the penalty left off");
  // (T.465.4's review) the head's logits off in row 0 (the check reads rows 0 and 95 of a vocabulary of 96): by 1% of the
  // largest logit in a float form's classifier (its line 1e-3 of it), by 10% in a DP4A form's (2e-2); the DP4A forms' stream off by 30% in an add;
  // a code of the norm and quantizer's turned; a key's exponent bit at position 1 turned; the embedding's row twice
  const verdicts = (m) => Object.fromEntries(m.tokenForm?.forms?.map(({ name, none }) => [name, none]) ?? []);
  const said = (m) => `${m.thrown ?? ""} ${JSON.stringify(verdicts(m))}`;
  // (by a share of the largest logit, which the check's line is of: a row of a small logit scaled by 1% fell under it)
  const headOff = await tokens({ spoil: { fused: (right, output) => (output === "write" ? outputOff(right, 5, (out) => { out[0] += 0.01 * peak(out); }) : right),
    dp4a: (right, output) => (output === "write" ? outputOff(right, 5, (out) => { out[0] += 0.1 * peak(out); }) : right) } });
  assert.match(said(headOff), /the logits are .* \(line 0\.001\)/, "a float classifier off in row 0");
  assert.match(said(headOff), /the logits are .* \(line 0\.02\)/, "a DP4A classifier off in row 0");
  const dp4aOff = await tokens({ spoil: { dp4a: (right, output) => (output === "add" ? outputOff(right, 5, (out) => { out[3] += 0.3 * peak(out); }) : right) } });
  assert.match(verdicts(dp4aOff)[DP4A] ?? "", /^wrong: the first layer's stream is .* \(line 0\.02\)/, "a DP4A form's stream off");
  assert.equal(verdicts(dp4aOff)[FLOAT], undefined, "the float forms right");
  const codes = await tokens({ spoil: { normQuantize: (entries, x, y) => {
    normQuantize(entries, x, y);
    const r = entries.find((entry) => entry.binding === 2).resource, data = (r.buffer ?? r).data;
    data[0] = (256 - data[0]) & 255;
  } } });
  assert.match(verdicts(codes)[DP4A] ?? "", /^wrong:/, "a code of the norm and quantizer's turned");
  assert.equal(verdicts(codes)[`${DP4A}, the norms apart`], undefined, "the norms apart: right (no NORM_QUANTIZE)");
  const keyBit = await tokens({ spoil: { fused: (right, output) => (output === "rope" ? (entries, x, y) => {
    right(entries, x, y);
    const r = entries.find((entry) => entry.binding === 6).resource, halves = new Uint16Array((r.buffer ?? r).data.buffer);
    halves[16 + 3] ^= 0x0400;
  } : right) } });
  assert.match(verdicts(keyBit)[FLOAT] ?? "", /^wrong: the first layer's keys are/, "a key's exponent bit turned");
  const twice = await tokens({ spoil: { embed: (entries, x, y) => {
    embed(entries, x, y);
    const r = entries.find((entry) => entry.binding === 3).resource, buffer = r.buffer ?? r;
    const row = new Float32Array(buffer.data.buffer, buffer.data.byteOffset, buffer.size / 4);
    for (let i = 0; i < row.length; i++) row[i] *= 2;
  } } });
  // (the norm takes the row's scale out: what is left is the residual's, which the float forms' line sees)
  assert.match(verdicts(twice)[FLOAT] ?? "", /^wrong: the first layer's stream is/, "the embedding's row twice");
  // off by 1e-5 of a value: inside the line
  const near = await tokens({ spoil: { fused: (right, output) => (output === "add" ? outputOff(right, 5, (out) => { out[3] *= 1 + 1e-5; }) : right) } });
  assert.deepEqual(near.tokenForm.forms.filter(({ none }) => none), [], "off inside the line: right");
}
console.log("ok: each check bites: a token's stream off in one value (float and DP4A each), a greedy token not the largest logit, a sampled token the walk does not reach, the penalty left off, the logits off in a row (float and DP4A), a code of the quantizer, a key's bit, the embedding twice (a float form); off inside the line is right");
// T.465.6.3: a ternary model's token (its matrices and table ternary: EMBED_TERNARY, the DP4A forms' ternary product), and
// each of the ten types of llama.cpp's blocks on w2's two layers (the DP4A product of each type: checkTokenBlocks' random
// blocks, then the first layer): every form right, every dispatch answered, and the product off is wrong
async function tokensOf(options) {
  const opening = await opened({ sane: true, ...options });
  await forms_.chooseAttention(opening.m);
  await forms_.chooseMatrices(opening.m);
  blockMade(opening.m);
  await tokenForms_.chooseTokens(opening.m).catch((error) => { opening.m.thrown = error.message; });
  return opening;
}
{
  const { m, answers } = await tokensOf({ model: "llama (8 heads)", weights: "ternary" });
  assert.equal(m.thrown, undefined, "a ternary model's token");
  assert.deepEqual(m.tokenForm.forms.filter(({ none }) => none), [], "ternary: no form wrong");
  assert.ok(m.tokenForm.forms.every(({ name }) => name.includes("ternary")), "its own forms");
  assert.deepEqual(answers.missed, [], "ternary: every dispatch answered");
  const off = await tokensOf({ model: "llama (8 heads)", weights: "ternary", spoil: { dp4a: (right, output) => (output === "add" ? outputOff(right, 5, (out) => { out[3] += 0.3 * peak(out); }) : right) } });
  assert.match(off.m.thrown ?? "", /no layer of a token is right on this GPU/, "a ternary product off");
  console.log(`ok: a ternary model's token: ${m.tokenForm.forms.map(({ name }) => name).join(", ")} checked right, every dispatch answered; its product off is wrong`);
}
for (const types of [["Q2_K", "Q4_K"], ["IQ2_XXS", "IQ2_XS"], ["IQ2_S", "IQ3_XXS"], ["IQ3_S", "IQ1_S"], ["IQ1_M", "IQ4_XS"]]) {
  const { m, answers } = await tokensOf({ weights: types });
  assert.equal(m.thrown, undefined, `${types.join(" and ")}: a token`);
  assert.deepEqual(m.tokenForm.forms.filter(({ none }) => none), [], `${types.join(" and ")}: no form wrong`);
  assert.deepEqual(answers.missed, [], `${types.join(" and ")}: every dispatch answered`);
}
{
  // (w2's product, the down projection, is the one matrix of blocks: its add off)
  const off = await tokensOf({ weights: ["Q4_K", "IQ2_XXS"], spoil: { dp4a: (right, output) => (output === "add" ? outputOff(right, 5, (out) => { out[3] += 0.3 * peak(out); }) : right) } });
  assert.match(off.m.thrown ?? "", /no layer of a token is right on this GPU/, "a product of blocks off");
}
console.log("ok: llama.cpp's blocks: a token of each of the ten types checked right, every dispatch answered; a product off is wrong");
// (a model's output at binding at, all of it 10% off)
const tenth = (model, at) => (entries, x, y) => {
  model(entries, x, y);
  const r = entries.find((entry) => entry.binding === at).resource, buffer = r.buffer ?? r;
  new Float32Array(buffer.data.buffer, buffer.data.byteOffset, buffer.size / 4).forEach((v, i, all) => { all[i] = v * 1.1; });
};
// T.465.6.4: an LFM2's convolution off (its output 10%), its rows not moved on (CONV_STATE writing nothing), a Qwen3.5's
// delta rule's state off (10%), its gates or its convolution off (10%), its rows not moved on: no form right
{

  for (const [model, spoil, what, said] of [["lfm2", { SHORT_CONV: tenth(shortConv, 3) }, "a convolution off", /stream|keys/],
    ["lfm2", { CONV_STATE: () => {} }, "its rows not moved on", /rows are not moved on by one/],
    // (its state: a scale of its output the norm of the value heads after it takes out)
    ["qwen35", { DELTA_NET: tenth(deltaNet, 2) }, "a delta rule's state off", /state/],
    // (T.465.6.4's review: and each other shader of the kinds: the gates, the convolution, its rows moved on, SwiGLU apart)
    ["qwen35", { DELTA_GATES: tenth(deltaGates, 5) }, "the gates off", /state|stream/], ["qwen35", { DELTA_CONV: tenth(deltaConv, 3) }, "the convolution off", /state|stream/],
    ["qwen35", { DELTA_CONV_STATE: () => {} }, "its rows not moved on", /rows/]]) {
    const m = await tokens({ model, spoil });
    assert.match(m.thrown ?? "", /no layer of a token is right on this GPU/, `${model}: ${what}`);
    assert.match(m.thrown, said, `${model}: ${what}, said`);
  }
}
// (SwiGLU of z and the delta rule's output, 10% off: the float forms' line sees it in the stream, 8e-3 of its change
// against 2e-3; the DP4A forms', 2e-2, does not)
{
  const m = await tokens({ model: "qwen35", spoil: { SWIGLU: tenth(swiglu, 0) } });
  assert.match(Object.fromEntries(m.tokenForm.forms.map(({ name, none }) => [name, none]))[FLOAT] ?? "", /^wrong: the first layer's stream is/, "qwen35: SwiGLU off");
}
console.log("ok: an LFM2's convolution off or its rows not moved on, a Qwen3.5's delta rule's state, gates, convolution or rows off: no form right; SwiGLU off: a float form wrong");
// T.465.6.5: a classifier with the final norm's outlier channels (T.92): an int8 one multiplied in floats on every form
// (floatHead), a ternary one with the channels taken out before it is quantized (TAKE_OUTLIERS) and their columns added
// after (TERNARY_COLUMNS): every form right, every dispatch answered; the columns not added, the channels taken but left
// in the stream: wrong
for (const [said, options] of [["int8", { outliers: true }], ["ternary", { model: "llama (8 heads)", weights: "ternary", outliers: true }]]) {
  const { m, answers } = await tokensOf(options);
  assert.equal(m.thrown, undefined, `${said} with outlier channels`);
  assert.deepEqual(m.tokenForm.forms.filter(({ none }) => none), [], `${said} with outlier channels: no form wrong`);
  assert.deepEqual(answers.missed, [], `${said} with outlier channels: every dispatch answered`);
}
// (the channels taken but left in the stream, counted twice: left out of both, they still come in through the product,
// and only its rounding of their groups differs, under the line in some runs)
const { takeOutliers } = await gpuAnswers();
const leftIn = (entries, x, y) => {
  const r = entries.find((entry) => entry.binding === 0).resource, buffer = r.buffer ?? r, before = buffer.data.slice();
  takeOutliers(entries, x, y);
  buffer.data.set(before);
};
for (const [spoil, what] of [[{ TERNARY_COLUMNS: () => {} }, "the columns not added"], [{ TAKE_OUTLIERS: leftIn }, "the channels left in the stream"]]) {
  const { m } = await tokensOf({ model: "llama (8 heads)", weights: "ternary", outliers: true, spoil });
  assert.match(m.thrown ?? "", /the logits are/, `a ternary classifier: ${what}`);
}
console.log("ok: classifiers with outlier channels, int8 and ternary: every form right, every dispatch answered; the columns not added or the channels left in the stream: wrong");
// T.465.6.5: a wide Llama (dim 256), of int8, and of llama.cpp's blocks on every matrix and its table: q, k and v each of
// its own type (written apart, T.547.3.2), gate and up of two, o and down of others, the table of Q4_K (a token's row of
// it, embedBlocks, and the classifier's text of its type); and q and k of one type with v of another, gate and up of one
// (q and k joined, v apart). Every form right, every dispatch answered; the classifier of blocks off, its row off: wrong
const APART = { wq: "Q4_K", wk: "IQ2_XXS", wv: "Q2_K", w1: "IQ3_S", w3: "IQ4_XS", w2: "Q4_K", wo: "IQ1_M", classifier: "Q4_K" };
const JOINED = { wq: "Q4_K", wk: "Q4_K", wv: "IQ2_XXS", w1: "Q2_K", w3: "Q2_K", w2: "IQ2_S", wo: "IQ3_XXS", classifier: "IQ1_S" };
for (const [said, blocks] of [["int8", {}], ["each matrix its own type", APART], ["q and k joined, v apart", JOINED]]) {
  const { m, answers } = await tokensOf({ model: "llama (wide)", blocks });
  assert.equal(m.thrown, undefined, `the wide Llama, ${said}`);
  assert.deepEqual(m.tokenForm.forms.filter(({ none }) => none), [], `the wide Llama, ${said}: no form wrong`);
  assert.deepEqual(answers.missed, [], `the wide Llama, ${said}: every dispatch answered`);
}
{
  const head = await tokensOf({ model: "llama (wide)", blocks: APART, spoil: { dp4a: (right, output) => (output === "write" ? outputOff(right, 5, (out) => { out[0] += 0.1 * peak(out); }) : right) } });
  // (the type's write text off: the random blocks' check of a token's matrices, checkTokenBlocks, refuses it first)
  assert.match(head.m.thrown ?? "", /no layer of a token is right on this GPU \((the Q4_K text of a token's matrix is|.*the logits are)/, "the classifier of blocks off");
  const { embedBlocks } = await gpuAnswers();
  // (one value of the row off by the row's largest: a scale of all of it the norm takes out, and a model of blocks has only
  // DP4A forms, whose line is wide)
  const row = await tokensOf({ model: "llama (wide)", blocks: APART, spoil: { embedBlocks: outputOff(embedBlocks({ weights: "Q4_K" }), 3, (out) => { out[3] += peak(out); }) } });
  assert.match(row.m.thrown ?? "", /no layer of a token is right on this GPU/, "a token's row of a table of blocks off");
}
console.log("ok: a wide Llama of int8 and of llama.cpp's blocks on every matrix and its table (q, k and v apart, and q and k joined): every form right, every dispatch answered; the classifier of blocks or its row off: wrong");
console.log("gpu-tokens-check: ok");
