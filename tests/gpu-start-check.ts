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
//   - no attention right: "unusable", with why, and the worker ended (its close()); the layers out of memory (the error
//     scope start() pushes first) and the device lost: "unusable", with why. The ready message's fields (the adapter, the
//     device's key, nothing remembered, a form timed and none wrong or failed) and the error scopes all popped.
//   - T.465.6.5: a model on the GPU alone (open(), take() of its checkpoint, start()): ready, every dispatch answered.
// Not held here: a stop in the middle, the forms a page remembered and force.quick, the order of the steps, timeBlocks'
// rounds of a device that is no fallback; of a model on the GPU alone, the layers after the first where the checkpoint
// routed them (the check compares the first layer and the tables only), the prompt's rows and RoPE (EMBED_ROWS, ROPE:
// answered, and timed, never compared), a checkpoint in stretches, and a ternary or blocks model on the GPU alone.
//   node tests/gpu-start-check.ts
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.ts";
import { made } from "./fakes/gpu-model.ts";
import { fresh } from "./fresh.ts";

const start_ = await import(runtimeUrl("gpu/start.ts"));
// (what the worker posts to the page: the worker's global's postMessage, as gpu-model.ts set it)
const posted = [];
globalThis.postMessage = (message) => posted.push(message);
// (and its close(), which the worker calls as it ends: unusable(), stop())
let closed = 0;
globalThis.close = () => { closed += 1; };

// a model's start on the stand-in, and what it posted
// (on: what to do to the device before start(): its error scopes held as a stack, which each case reads after)
let scopes = [];
async function started(options = {}, on = (device, lose) => {}) {
  posted.length = 0;
  const { plan, memory, device, lose } = await made({ sane: true, ...options });
  scopes = [];
  device.pushErrorScope = (filter) => { scopes.push(filter); };
  device.popErrorScope = async () => (scopes.pop(), null);
  on(device, lose);
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
  // (T.465.5's review: a form the device does not make is said so, and every other is timed: none wrong, none failed)
  assert.ok(ready.form && ready.attention && Array.isArray(ready.forms), "the prompt's forms");
  assert.ok(ready.forms.some(({ ms }) => Number.isFinite(ms)), "a form timed");
  assert.deepEqual(ready.forms.filter(({ none }) => none && !/shader-f16|subgroups|packed int8 dot|threads|bytes of workgroup memory/.test(none)), [], "no form wrong or failed");
  assert.match(ready.adapter, /fallback adapter/, "the adapter, said as one in the CPU's place");
  assert.ok(ready.key && ready.remembered === false && ready.blocks.every(({ ms }) => Number.isFinite(ms ?? 0)), "the device's key, nothing remembered, the blocks' times");
  assert.deepEqual(scopes, [], "the error scopes pushed all popped");
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
  assert.ok(ready.form && ready.attention, "the prompt's forms kept");
  assert.equal(ready.tokens, null, "no token's form");
  assert.match(ready.tokensWhy ?? "", /no layer of a token is right on this GPU .*the greedy token is/, "why a token stays on the CPU");
  console.log("ok: a token's forms all wrong: ready, the prompts on the GPU, and why a token is not");
}
{
  // no attention right: unusable, with why, and the worker ended
  const before = closed;
  const [said] = await started({ spoil: { attention: () => {} } });
  assert.equal(said?.type, "unusable", `unusable, not ${said?.type ?? "nothing said"}`);
  assert.match(JSON.stringify(said), /no attention is right on this GPU/, "with why");
  assert.equal(closed - before, 1, "and the worker ended");
  console.log("ok: no attention right: unusable, with why");
}
{
  // T.465.6.5: a model on the GPU alone: open() before the checkpoint comes, take() of its stretches (the whole of it at
  // once here), then start(): ready, every form chosen, a token's too, every dispatch answered (a prompt's rows from the
  // ids, EMBED_ROWS, and its RoPE among them)
  // (a start.ts of its own each: what open() and take() keep is a worker's, which a load has anew, tests/fresh.ts)
  const alone = async () => {
    posted.length = 0;
    const { plan, memory, answers } = await made({ sane: true, direct: true });
    const own = await fresh(runtimeUrl("gpu/start.ts"));
    own.open(plan, new SharedArrayBuffer(8));
    own.take({ type: "weights", offset: 0, bytes: new Uint8Array(memory.buffer).slice() });
    await own.start(memory, plan);
    return { said: posted.filter(({ type }) => type !== "progress"), answers };
  };
  const { said: [ready], answers } = await alone();
  assert.equal(ready?.type, "ready", `a model on the GPU alone: ready, not ${JSON.stringify(ready).slice(0, 200)}`);
  assert.ok(ready.tokens?.form && ready.tokensWhy === null, "its token's form");
  assert.deepEqual(answers.missed, [], "every dispatch answered");
  console.log("ok: a model on the GPU alone: ready with every form chosen (its first layer and tables read back), every dispatch answered");
}
{
  // (T.465.5's review) the device out of memory for the layers (the scope start() pushes first, opened with the device):
  // unusable, the GPU not taking them
  const [said] = await started({}, (device) => {
    device.popErrorScope = async () => (scopes.pop() === "out-of-memory" ? { message: "out of memory" } : null);
  });
  assert.equal(said?.type, "unusable");
  assert.match(JSON.stringify(said), /the GPU did not take the layers \(out of memory\)/, "the layers not taken");
}
{
  // the device lost as it starts (last: device.ts keeps that it was lost): unusable, with why
  const [said] = await started({}, (device, lose) => { lose({ reason: "unknown", message: "gone" }); });
  assert.equal(said?.type, "unusable");
  assert.match(JSON.stringify(said), /the GPU was lost \(unknown: gone\)/, "the device lost");
  console.log("ok: the layers out of memory, and the device lost: unusable, with why");
}
console.log("gpu-start-check: ok");
