// T130's review: forward.js's grow() (the moves of the cache's blocks, copied here statement for statement) on a byte
// array with a distinct byte pattern per (side, layer, position, byte): every layer count, every growth from a capacity
// to a larger one (the doubling, a context that is not a power of two times the start, a jump past twice), both key
// types' row widths, and each block's old stretch must be where attention reads it afterwards.
//   node .tmp/t130/grow-sim.mjs
const grow = (U, state, larger, KV, layers, order = "T130") => {
  const { capacity, keys, values } = state;
  const oldLayer = capacity * KV, newLayer = larger * KV;
  const newValues = keys + layers * newLayer;
  if (order === "T130") {
    for (let l = layers - 1; l >= 0; l--) U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);
    for (let l = layers - 1; l > 0; l--) U.copyWithin(keys + l * newLayer, keys + l * oldLayer, keys + (l + 1) * oldLayer);
  } else if (order === "keys-first") {
    for (let l = layers - 1; l > 0; l--) U.copyWithin(keys + l * newLayer, keys + l * oldLayer, keys + (l + 1) * oldLayer);
    for (let l = layers - 1; l >= 0; l--) U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);
  } else if (order === "ascending") {
    for (let l = 0; l < layers; l++) U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);
    for (let l = 1; l < layers; l++) U.copyWithin(keys + l * newLayer, keys + l * oldLayer, keys + (l + 1) * oldLayer);
  }
  return { capacity: larger, keys, values: newValues };
};
const byteOf = (side, l, pos, i) => (side * 131 + l * 31 + pos * 7 + i * 3 + 1) & 255;
let cases = 0, failures = 0;
const bad = [];
for (const order of ["T130", "keys-first", "ascending"]) {
  let failed = 0, ran = 0;
  for (let layers = 1; layers <= 9; layers++) {
    for (const KV of [4, 8, 12, 16]) {  // bytes of one position's keys: kvDim 1, 2, 3, 4 values of float16 or float32
      for (let capacity = 1; capacity <= 20; capacity++) {
        for (let larger = capacity; larger <= 3 * capacity + 1; larger++) {
          const start = 16;  // where the keys begin in the made-up memory (anything: the checkpoint is below)
          const end = start + 2 * layers * larger * KV + 8;  // room for the larger cache and a sentinel after it
          const U = new Uint8Array(end).fill(0xee);
          const state = { capacity, keys: start, values: start + layers * capacity * KV };
          // fill: every position of every layer of both sides
          for (const [side, at] of [[0, state.keys], [1, state.values]]) {
            for (let l = 0; l < layers; l++) for (let pos = 0; pos < capacity; pos++) for (let i = 0; i < KV; i++) {
              U[at + l * capacity * KV + pos * KV + i] = byteOf(side, l, pos, i);
            }
          }
          const next = grow(U, state, larger, KV, layers, order);
          let ok = true;
          for (const [side, at] of [[0, next.keys], [1, next.values]]) {
            for (let l = 0; l < layers && ok; l++) for (let pos = 0; pos < capacity && ok; pos++) for (let i = 0; i < KV; i++) {
              if (U[at + l * larger * KV + pos * KV + i] !== byteOf(side, l, pos, i)) { ok = false; break; }
            }
          }
          // nothing beyond the cache's end (the sentinel) was written
          for (let i = start + 2 * layers * larger * KV; i < end; i++) if (U[i] !== 0xee) ok = false;
          ran++;
          if (!ok) { failed++; if (bad.length < 3 && order === "T130") bad.push({ layers, KV, capacity, larger }); }
        }
      }
    }
  }
  console.log(`${order}: ${ran} growths (1 to 9 layers, 4 row widths, capacities 1 to 20, every larger size up to 3 times and one over), ${failed} lost or misplaced data`);
  cases += ran;
  if (order === "T130") failures += failed;
}
console.log(failures ? `FAILED: ${JSON.stringify(bad)}` : "T130's order loses nothing; the two wrong orders above are the tests' mutants (they must show losses)");
process.exit(failures ? 1 : 0);
