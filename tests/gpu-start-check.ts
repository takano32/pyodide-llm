// tests/gpu-start-check.ts (T.465.5): the model's GPU worker's start (src/runtime/gpu/start.ts's start()) whole, on the
// stand-in device whose checks' dispatches are answered as the shaders would (tests/fakes/gpu-model.ts's made(): the
// answers' arithmetic is tests/gpu_answers.py's, in NumPy). Nothing ran it without a GPU: every check of a form stood in
// its way. A made-up Llama (its memory's numbers ones a model could have), the device a fallback adapter that the plan
// allows (each right form timed by a pair of one pass there):
//   - start(): the weights up, the prompt's attention and tiled form chosen, the block bound and its cache made, a block
//     timed, a token's forms chosen ("progress" posted after the steps), and "ready" posted with what each came to (the
//     forms, the attention, the blocks' times, a token's form and attention), no reason a token stays on the CPU;
//   - a token's forms all wrong (the sampler's greedy token not the largest logit): "ready" still, the prompts on the GPU,
//     and why a token is not (tokensWhy);
//   - no attention right: "unusable", with why, and the worker ended (its close()).
// Not held here: a model on the GPU alone (open(), take() and the stretches of its checkpoint), a stop in the middle, a
// device lost (T.465.6 and after, where they are wanted).
//   node tests/gpu-start-check.ts
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";
import { made } from "./fakes/gpu-model.ts";

const start_ = await import(runtimeUrl("gpu/start.ts"));
// (what the worker posts to the page: the worker's global's postMessage, as gpu-model.ts set it)
const posted = [];
globalThis.postMessage = (message) => posted.push(message);
// (and its close(), which the worker calls as it ends: unusable(), stop())
let closed = 0;
globalThis.close = () => { closed += 1; };

// a model's start on the stand-in, and what it posted
async function started(options = {}) {
  posted.length = 0;
  const { plan, memory } = await made({ sane: true, ...options });
  await start_.start(memory, plan);
  // (a "progress" after each step, which the page's watch reads as the worker being alive: T.135's GPU_QUIET_MS)
  const steps = posted.filter(({ type }) => type === "progress").length;
  assert.ok(steps > 0, "a start says it is going on");
  return posted.filter(({ type }) => type !== "progress");
}

{
  const [ready, ...more] = await started({ features: ["subgroups"] });
  assert.deepEqual(more, [], "one message");
  assert.equal(ready.type, "ready", `ready, not ${JSON.stringify(ready).slice(0, 300)}`);
  assert.ok(ready.form && ready.attention && Array.isArray(ready.forms) && ready.forms.every(({ none }) => !none?.startsWith("wrong")), "the prompt's forms, none wrong");
  assert.ok(Array.isArray(ready.blocks) && ready.blocks.length > 0, "a block timed");
  assert.ok(ready.tokens?.form && ready.tokens.attention, `a token's form and attention: ${JSON.stringify(ready.tokens)?.slice(0, 300)}`);
  assert.equal(ready.tokensWhy, null, "no reason a token stays on the CPU");
  assert.ok(ready.bytes > 0 && ready.seconds >= 0);
  console.log(`ok: start(): ready, the prompt's ${ready.form} and ${ready.attention}, ${ready.blocks.length} block sizes timed, a token by ${ready.tokens.form}`);
}
{
  // a token's forms all wrong: the sampler's greedy token one off
  const { gpuAnswers } = await import("./fakes/gpu-answers.ts");
  const { sample } = await gpuAnswers();
  const [ready] = await started({ spoil: { sample: (entries, x, y) => {
    sample(entries, x, y);
    const chosen = entries.find((entry) => entry.binding === 4).resource, words = new Uint32Array((chosen.buffer ?? chosen).data.buffer);
    words[0] = (words[0] + 1) % 96;
  } } });
  assert.equal(ready.type, "ready", "ready still: the prompts on the GPU");
  assert.equal(ready.tokens, null, "no token's form");
  assert.match(ready.tokensWhy ?? "", /no layer of a token is right on this GPU .*the greedy token is/, "why a token stays on the CPU");
  console.log("ok: a token's forms all wrong: ready, the prompts on the GPU, and why a token is not");
}
{
  // no attention right: unusable, with why
  const [said] = await started({ spoil: { attention: () => {} } });
  assert.equal(said?.type, "unusable", `unusable, not ${said?.type ?? "nothing said"}`);
  assert.match(JSON.stringify(said), /no attention is right on this GPU/, "with why");
  assert.equal(closed, 1, "and the worker ended");
  console.log("ok: no attention right: unusable, with why");
}
console.log("gpu-start-check: ok");
