// tests/shader-makers.ts (T379.2): the makers of WGSL among src/runtime/shaders.ts's exports that the engine's
// catalogue (shaders/engine.ts) has no rows of, with the arguments that walk every branch of each one's text (a number
// is written into the text and chooses nothing: T366): the makers of a prompt's tiles, which the device's key holds by
// promptForms(), and the benchmark's own. One list, which the net reads (tests/unchanged.ts compares the texts these
// arguments make in two trees) and the light suite's tests/shaders-catalogue-check.ts, which holds that every maker
// of WGSL the window exports is the catalogue's or is here: until T379.2 a maker left out of this list was compared
// by its syntax tree alone, and nothing said so.
// A BRANCH ADDED TO ONE OF THESE MAKERS IS TO BE ADDED HERE (nothing else walks them: the catalogue's makers are
// walked by its rows, which tests/shaders-catalogue-check.ts and tests/device-key-check.ts hold).
const EITHER = [false, true];
// (T642.9's review: the types of llama.cpp's blocks that the GPU reads, a second copy of shaders/common.ts's
// BLOCK_PACKINGS, held to it by tests/shaders-catalogue-check.ts: the tiles of a model with blocks are in no key the
// net hashes, so the net walked none of them until then)
export const BLOCK_PACKINGS = ["Q4_K", "Q2_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "IQ4_XS"];
export const MAKERS_APART = {
  regTile: EITHER.map((half) => [half]),
  dp4a: EITHER.flatMap((subgroups) => ["int8", "ternary", ...BLOCK_PACKINGS].map((weights) => [{ subgroups, weights }])),
  fmaCeiling: EITHER.flatMap((half) => ["square", "affine"].map((shape) => [half, shape])),
  mulMatVec: EITHER.map((subgroups) => [{ subgroups }]),
};
/** the exports of a shaders module that make WGSL: the functions whose own text holds a shader's entry point */
export const makersOfWgsl = (module) => Object.keys(module).filter((name) => typeof module[name] === "function" && String(module[name]).includes("@compute")).sort();
