// shaders/key.ts (T.351): the key of what the page remembers of a device (T.148, T.366): the adapter, the browser and a
// hash of the WGSL of the engine's shaders: the catalogue's (engine.ts, T.379.1), which is every text the engine's GPU
// worker may run and none of the benchmark's own, and the prompt's tiles a device can make.
// A part of src/runtime/shaders.ts, which is the window: everything outside src/runtime/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part takes its neighbours' names by static imports (T.367.3:
// a bundler links them).
import { promptForms } from "./prompt.ts";
import { ENGINE_SHADERS, TERNARY_SHADERS, BLOCK_SHADERS } from "./engine.ts";
import type { Own, PromptForm } from "./prompt.ts";

// (T.368.5) what the key reads of a device, or of the adapter it is made of (a GPUDevice and a GPUAdapter both have
// them): the features that choose its shaders and the limits that size its tiles. And of the adapter alone, its info
// (T.148: not every browser gives one)
export type Made = { features: { has(name: string): boolean };
  limits: Pick<GPUSupportedLimits, "maxComputeWorkgroupStorageSize" | "maxComputeInvocationsPerWorkgroup" | "maxComputeWorkgroupSizeX"> };
type Info = Pick<GPUAdapterInfo, "vendor" | "architecture" | "device" | "description">;
export type Adapter = Made & { info?: Info };

// ---- T.148: the key of what the page remembers of a device (the shaders it chose, T.156: that the CPU was faster than a
// model on the GPU alone): the adapter and the browser, and the text of the engine's shaders as a short hash (FNV-1a):
// a deployment whose shaders changed chooses anew. adapter's info; device: the device made of it, or the adapter itself
// (the worker, before any device: gpu.ts asks the device for the adapter's features and these limits, so the two give
// the same key).
// T.366: what is hashed is WGSL as a device is given it, and nothing of this file's JavaScript: a maker counts by the
// texts it makes, never by its own source (String(maker), until T.366: a formatter, a bundler or a new parameter changed
// every device's key). A maker's texts are those of every choice its text branches on (a form, a feature), with one
// set of numbers: a number is written into the text and chooses nothing of it. tests/device-key-check.mjs holds that
// every piece of text of the shaders gpu.ts runs reaches the hash, and that the key does not move with the source.
// T.379.1: which texts those are, each maker's choices and their order is the catalogue's (engine.ts: the rows of every
// model's shaders, and those of a model of ternary weights alone), which the worker takes its shaders from as well.
/** the tiled shaders of T.146 a device can make (promptForms; T.232, T.547.2.3, own: those of a model of ternary weights,
 * or of Q4_K blocks besides int8) */
export const devicePromptForms = (device: Made, own: Own = null) => promptForms({ half: device.features.has("shader-f16"), subgroups: device.features.has("subgroups"),
  packed: Boolean(globalThis.navigator?.gpu?.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product")),
  memory: device.limits.maxComputeWorkgroupStorageSize,
  threads: Math.min(device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX), own });
// T.232, ternary: the key of a model of ternary weights, the same with the text of its own shaders hashed after the
// others': the key of every other model does not hold them (a device keeps what it remembers of those), and a
// deployment that changes a ternary shader has the ternary models choose anew. T.547.2.3: own, a model's own shaders
// (ternary's, or a model of Q4_K blocks: its tiles, each with its Q4_K text), whose key is the others' and their text
// T.366's review: the browser's two WGSL language features that choose which of the shaders a device makes (packed int8
// dot: the DP4A forms; subgroup_id: the flash attention and the fused matrices with subgroups) are named in the key as the
// user agent is (a flag turned on under the same user agent chooses anew), and the texts are hashed with a mark between
// them, so that a character moved from the end of one text to the start of the next, or one shader cut in two, moves it.
// The device's limits are not in it: they size the tiles (none) and nothing in the text, and the worker's key (the adapter's)
// must be gpu.ts's (the device's)
export function deviceKey(adapter: Adapter, device: Made = adapter, own: Own = null): string {
  const info: Partial<Info> = adapter.info ?? {};
  const language = globalThis.navigator?.gpu?.wgslLanguageFeatures;
  const named = [info.vendor, info.architecture, info.device, info.description, globalThis.navigator?.userAgent,
    language?.has("packed_4x8_integer_dot_product") ? "packed" : "", language?.has("subgroup_id") ? "subgroup_id" : ""].map((part) => part ?? "").join("|");
  const tiles = (forms: PromptForm[]) => forms.flatMap((form) => [form.name, form.code, ...(form.blocks ? [form.blocks] : [])]);
  let hash = 0x811c9dc5;
  const written = (rows: { text: () => string }[]) => rows.map((row) => row.text());
  for (const text of [...tiles(devicePromptForms(device)), ...written(ENGINE_SHADERS),
    ...(own === "ternary" ? [...tiles(devicePromptForms(device, "ternary")), ...written(TERNARY_SHADERS)] : own === "Q4_K" ? [...tiles(devicePromptForms(device, "Q4_K")), ...written(BLOCK_SHADERS)] : [])]) {
    for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193);
    hash = Math.imul(hash ^ 0x1f, 0x01000193);  // (the unit separator: in no WGSL)
  }
  return `${named}|${(hash >>> 0).toString(16)}`;
}
