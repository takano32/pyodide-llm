// shaders/engine.ts (T379.1): the catalogue of the engine's shaders: every WGSL text the model's GPU worker
// (src/runtime/gpu.ts) may run, but for a prompt's tiles (key.ts's devicePromptForms: those a device can make). One
// list, which three read: the device's key hashes its texts in its order (key.ts's deviceKey), the worker takes each
// shader it compiles from it by name (ENGINE.SAMPLE, ENGINE.flashTile(shape): gpu/*.ts import no shader of a part),
// and the checks walk it (tests/shaders-catalogue-check.mjs, tests/device-key-check.mjs, the net's shaders).
// A part of src/runtime/shaders.ts, which is the window. It holds no WGSL of its own, and no line of another project.
//
// A row is a text, or a maker with one set of arguments: one of the choices its text branches on (a form, a feature,
// the packing of the weights), with one set of numbers (a number is written into the text and chooses nothing of it:
// T366). Every choice of every maker has a row, so that the key holds every text a device can be given.
//
// A shader more for the worker: its text or maker in its part, a row here (a maker: one for each choice), and
// ENGINE.<name> where the worker compiles it. A worker cannot take what has no row (the type says so), and what has a
// row is in the key. A row is a whole shader the worker runs: a piece of text that shaders share (prompt.ts's
// TERNARY_PACKED, which the key held as a row of its own from T232 to T379.2) is in the key by the texts it is in.
//
// THE ORDER IS THE KEY'S. A row moved, added or taken away changes the key of every device, which then chooses its
// forms again and weighs a model on the GPU alone again (T148, T156), as a changed character of a text does. The order
// is: the texts, then each maker's rows together (T379.2, the deployment that changed the texts: until then the makers
// of each value of subgroups stood together, as the key had walked them since T366).
import { QUANTIZE, WIDEN_SIX, RMSNORM, HEAD_NORM, LAYER_NORM, ADD, ROPE, SWIGLU, GELU, RMSNORM_ADD, GEGLU } from "./steps.ts";
import { flashTile, flashVec, flashVecReduce } from "./attention.ts";
import { TOKEN_ROPE, fusedMatVec, NORM_QUANTIZE, fusedDp4aMatVec, FUSED_INPUTS, FUSED_OUTPUTS } from "./fused.ts";
import { EMBED, EMBED_ROWS, EMBED_TERNARY, EMBED_ROWS_TERNARY, TAKE_OUTLIERS, TERNARY_COLUMNS } from "./run.ts";
import { SAMPLE } from "./sample.ts";

// rows of texts, each under its own name ({ RMSNORM, ADD }: the names are the object's keys, in its order)
const texts = <Name extends string>(named: Record<Name, string>) =>
  (Object.keys(named) as Name[]).map((name) => ({ name, shader: named[name], text: () => named[name] }));
// a row of a maker under its own name ({ flashTile }), with the arguments of one of its texts. (T379.2) One maker a
// row: two names ({ flashVec, flashVecReduce }) would make a row of the first alone, so the type takes no arguments
// for more than one (One<Name> is never where Name is a union)
type One<Name extends string> = { [Each in Name]: [Name] extends [Each] ? Each : never }[Name];
const made = <Name extends string, Given extends object>(named: Record<Name, (given: Given) => string>, given: One<Name> extends never ? never : NoInfer<Given>) => {
  const [name] = Object.keys(named) as Name[];
  return { name, shader: named[name], given, text: () => named[name](given) };
};
// what the names of some rows are: each name's text or maker
type Named<Row extends { name: string; shader: unknown }> = { [Each in Row as Each["name"]]: Each["shader"] };

const EITHER = [false, true];
/** every model's shaders */
export const ENGINE_SHADERS = [
  ...texts({ RMSNORM, HEAD_NORM, ADD, ROPE, SWIGLU, QUANTIZE, LAYER_NORM, GELU, EMBED, EMBED_ROWS, SAMPLE, NORM_QUANTIZE, TOKEN_ROPE, WIDEN_SIX, RMSNORM_ADD, GEGLU }),
  ...EITHER.flatMap((subgroups) => EITHER.map((half) => made({ flashTile }, { headSize: 64, half, subgroups, wgSize: 32, kvTile: 8, minSubgroup: 4 }))),
  ...EITHER.map((subgroups) => made({ flashVec }, { headSize: 64, subgroups, wgSize: 32, kvTile: 32, dSplit: 16 })),
  ...EITHER.map((subgroups) => made({ flashVecReduce }, { headSize: 64, subgroups, reduceSize: 32 })),
  ...EITHER.flatMap((subgroups) => FUSED_INPUTS.flatMap((input) => FUSED_OUTPUTS.map((output) => made({ fusedMatVec }, { input, output, subgroups })))),
  ...FUSED_OUTPUTS.map((output) => made({ fusedDp4aMatVec }, { output, weights: "int8" }))];
/** T232: and those of a model of ternary weights alone, after them (its tiles: devicePromptForms' ternary) */
export const TERNARY_SHADERS = [
  ...FUSED_OUTPUTS.map((output) => made({ fusedDp4aMatVec }, { output, weights: "ternary" })),
  ...texts({ EMBED_TERNARY, EMBED_ROWS_TERNARY, TAKE_OUTLIERS, TERNARY_COLUMNS })];
/** the shaders by name, for the worker: a text, or a maker to call with a model's own numbers */
export const ENGINE = Object.fromEntries([...ENGINE_SHADERS, ...TERNARY_SHADERS].map(({ name, shader }) => [name, shader])) as
  Named<(typeof ENGINE_SHADERS)[number] | (typeof TERNARY_SHADERS)[number]>;
