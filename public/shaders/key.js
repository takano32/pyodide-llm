// shaders/key.js (T351): the key of what the page remembers of a device (T148, T366): the adapter, the browser and a
// hash of the WGSL of the engine's shaders. It takes every part the engine's GPU worker runs a shader of, and none of
// the benchmark's own (bench.js, stages.js) or likecpu.js.
// A part of public/shaders.js, which is the window: everything outside public/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part asks for its neighbours with its own ?v=<build>
// (GitHub Pages keeps a file for ten minutes: all must come from one deployment).
const { TERNARY_PACKED, promptForms } = await import(new URL(`prompt.js${new URL(import.meta.url).search}`, import.meta.url));
const { QUANTIZE, WIDEN_SIX, RMSNORM, HEAD_NORM, LAYER_NORM, ADD, ROPE, SWIGLU, GELU } = await import(new URL(`steps.js${new URL(import.meta.url).search}`, import.meta.url));
const { flashTile, flashVec, flashVecReduce } = await import(new URL(`attention.js${new URL(import.meta.url).search}`, import.meta.url));
const { TOKEN_ROPE, fusedMatVec, NORM_QUANTIZE, fusedDp4aMatVec, ternaryMatVec } = await import(new URL(`fused.js${new URL(import.meta.url).search}`, import.meta.url));
const { EMBED, EMBED_ROWS, EMBED_TERNARY, EMBED_ROWS_TERNARY, TAKE_OUTLIERS, TERNARY_COLUMNS } = await import(new URL(`run.js${new URL(import.meta.url).search}`, import.meta.url));
const { SAMPLE } = await import(new URL(`sample.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- T148: the key of what the page remembers of a device (the shaders it chose, T156: that the CPU was faster than a
// model on the GPU alone): the adapter and the browser, and the text of the engine's shaders as a short hash (FNV-1a):
// a deployment whose shaders changed chooses anew. adapter's info; device: the device made of it, or the adapter itself
// (the worker, before any device: gpu.js asks the device for the adapter's features and these limits, so the two give
// the same key).
// T366: what is hashed is WGSL as a device is given it, and nothing of this file's JavaScript: a maker counts by the
// texts it makes, never by its own source (String(maker), until T366: a formatter, a bundler or a new parameter changed
// every device's key). A maker's texts are those of every choice its text branches on (a form, a feature), with one
// set of numbers: a number is written into the text and chooses nothing of it. tests/device-key-check.mjs holds that
// every piece of text of the shaders gpu.js runs reaches the hash, and that the key does not move with the source.
const EITHER = [false, true], OUTPUTS = ["rope", "add", "swiglu", "write"];
/** the WGSL of every model's shaders in gpu.js, but for the prompt's tiles (devicePromptForms: those the device can make) */
const engineShaders = () => [RMSNORM, HEAD_NORM, ADD, ROPE, SWIGLU, QUANTIZE, LAYER_NORM, GELU, EMBED, EMBED_ROWS, SAMPLE, NORM_QUANTIZE, TOKEN_ROPE, WIDEN_SIX,
  ...EITHER.flatMap((subgroups) => [
    ...EITHER.map((half) => flashTile({ headSize: 64, half, subgroups, wgSize: 32, kvTile: 8, minSubgroup: 4 })),
    flashVec({ headSize: 64, subgroups, wgSize: 32, kvTile: 32, dSplit: 16 }), flashVecReduce({ headSize: 64, subgroups, reduceSize: 32 }),
    ...["norm", "plain"].flatMap((input) => OUTPUTS.map((output) => fusedMatVec({ input, output, subgroups })))]),
  ...OUTPUTS.map((output) => fusedDp4aMatVec({ output }))];
/** T232: and those of a model of ternary weights alone (its tiles: devicePromptForms' ternary) */
const ternaryShaders = () => [TERNARY_PACKED, ...OUTPUTS.map((output) => ternaryMatVec({ output })), EMBED_TERNARY, EMBED_ROWS_TERNARY, TAKE_OUTLIERS, TERNARY_COLUMNS];
/** the tiled shaders of T146 a device can make (promptForms; T232, ternary: those of a model of ternary weights) */
export const devicePromptForms = (device, ternary = false) => promptForms({ half: device.features.has("shader-f16"), subgroups: device.features.has("subgroups"),
  packed: Boolean(globalThis.navigator?.gpu?.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product")),
  memory: device.limits.maxComputeWorkgroupStorageSize,
  threads: Math.min(device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX), ternary });
// T232, ternary: the key of a model of ternary weights, the same with the text of its own shaders hashed after the
// others': the key of every other model does not hold them (a device keeps what it remembers of those), and a
// deployment that changes a ternary shader has the ternary models choose anew.
// T366's review: the browser's two WGSL language features that choose which of the shaders a device makes (packed int8
// dot: the DP4A forms; subgroup_id: the flash attention and the fused matrices with subgroups) are named in the key as the
// user agent is (a flag turned on under the same user agent chooses anew), and the texts are hashed with a mark between
// them, so that a character moved from the end of one text to the start of the next, or one shader cut in two, moves it.
// The device's limits are not in it: they size the tiles (none) and nothing in the text, and the worker's key (the adapter's)
// must be gpu.js's (the device's)
export function deviceKey(adapter, device = adapter, ternary = false) {
  const info = adapter.info ?? {};
  const language = globalThis.navigator?.gpu?.wgslLanguageFeatures;
  const named = [info.vendor, info.architecture, info.device, info.description, globalThis.navigator?.userAgent,
    language?.has("packed_4x8_integer_dot_product") ? "packed" : "", language?.has("subgroup_id") ? "subgroup_id" : ""].map((part) => part ?? "").join("|");
  const tiles = (forms) => forms.flatMap((form) => [form.name, form.code]);
  let hash = 0x811c9dc5;
  for (const text of [...tiles(devicePromptForms(device)), ...engineShaders(),
    ...(ternary ? [...tiles(devicePromptForms(device, true)), ...ternaryShaders()] : [])]) {
    for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193);
    hash = Math.imul(hash ^ 0x1f, 0x01000193);  // (the unit separator: in no WGSL)
  }
  return `${named}|${(hash >>> 0).toString(16)}`;
}
