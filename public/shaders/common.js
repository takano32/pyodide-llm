// shaders/common.js (T351): what the shaders of every family share: the size of a group of weights, and the Step
// struct that every step of a layer reads.
// A part of public/shaders.js, which is the window: everything outside public/shaders/ imports that file and no part.
// The lines are those of the one file shaders.js was, as they were. A part asks for its neighbours with its own ?v=<build>
// (GitHub Pages keeps a file for ten minutes: all must come from one deployment).

export const GROUP = 32;

// the tokens of a request and the position of the first, written once per request: every step of a layer reads it
const STEP = /* wgsl */ `struct Step { tokens: u32, pos: u32, unused0: u32, unused1: u32 }`;

export { STEP };
