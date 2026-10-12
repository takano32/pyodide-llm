// shaders/common.ts (T351): what the shaders of every family share: the size of a group of weights, and the Step
// struct that every step of a layer reads.
// A part of src/runtime/shaders.ts, which is the window: everything outside src/runtime/shaders/ imports that file and no part.
// The statements are those of the one file shaders.js was, as they were. A part takes its neighbours' names by static imports (T367.3:
// a bundler links them).

export const GROUP = 32;

// the tokens of a request and the position of the first, written once per request: every step of a layer reads it
const STEP = /* wgsl */ `struct Step { tokens: u32, pos: u32, unused0: u32, unused1: u32 }`;

export { STEP };
