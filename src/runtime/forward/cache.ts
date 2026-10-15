// forward/cache.ts (T375.3): the cache of the keys and values of one engine, and its state's owner. The layers that
// attend write and read it (attention.ts), the GPU's side reads where it is and has it grown (gpuside.ts's cpuSide),
// and the tests read it back. A module of src/runtime/forward.ts.
//
// Per layer [positions][kvDim], one block for the keys and one for the values, last in the memory so that growing it
// (from the plan's kv_start, doubling) takes only the room it adds (T130): every layer's block moves up to where it
// starts at the larger size, the last first, so that none is written over before it has moved (a block starts no
// lower than it did, and past the end of the ones below it). Before, the larger blocks were made after the smaller
// and moved down onto them: 1.5 times the context at the step to the whole of it, which a 32-bit memory at the edge
// could not hold (Qwen2.5 3B's float32 keys and values on a memory that is not shared: 4.03 GiB, 3.89 now).
// So the engine makes it first and places it last: hold() is how a kind of layer says, as it is made, how many of its
// layers keep keys and values here, and place() takes the room once everything else has its own.
//
// T510.1: a layer that attends only the last window positions (a Gemma 3's local layers) keeps no more than those, in a
// ring of its own (ring()), where the context is long enough for that to save anything (memory.ts's ringPositions()).
// Position p is at p mod C of the ring and once more at p mod C + C (C, the ring's period: the window and a prompt's
// block, BATCH: the first token of a block still finds its window after the last has been written), so that every
// token's window is one run of positions, oldest first, as it is in a whole layer's block: the kernels that attend
// read it as they read any other, and the numbers are the same. The rings are taken first and never grow; the blocks
// of the layers that keep every position follow and grow as before (a layer's index among those is index()).
import { halfToFloat, ringPositions } from "./memory.ts";
import type { Kernels } from "../jobs.ts";
import type { Room } from "./kinds.ts";

// T510.1: a layer's ring: its period, and where its keys and its values are (place() says, 2 * size positions each)
export type Ring = { size: number; keys: number; values: number };

/** room, k: the engine's memory and kernels. kvDim: the values of one position's keys; half: kept in float16 (T110);
 * start: the positions there is room for at first (0: no cache, T210: a model on the GPU alone); seqLen: the most. */
export function keyValueCache({ room, k, kvDim, half, start, seqLen }: { room: Room; k: Kernels; kvDim: number; half: boolean; start: number; seqLen: number }) {
  // KV: the bytes of one position's keys in the cache; KF: in float32
  const KV = kvDim * (half ? 2 : 4), KF = kvDim * 4;
  // held: the places taken; layers: of those, the ones that keep every position (the blocks that grow); for every
  // place, its index among those (-1: a ring's), and the rings by place
  let held = 0, layers = 0, capacity = 0, keys = 0, values = 0, placed = false;
  const indices: number[] = [], rings: (Ring | null)[] = [];
  return {
    KV, half,
    /** count more layers keep their keys and values here: the first of their places (before place()). T510.1: windows,
     * how many of the last positions each attends (0 or none: all of them): one that a ring holds keeps no more */
    hold(count: number, windows: number[] = []) {
      if (placed) throw new Error("The cache of the keys and values has its room already: a kind of layer takes its places as it is made.");
      const first = held;
      for (let a = 0; a < count; a++) {
        const size = start ? ringPositions(windows[a] ?? 0, seqLen) / 2 : 0;
        rings.push(size ? { size, keys: 0, values: 0 } : null);
        indices.push(size ? -1 : layers++);
      }
      held += count;
      return first;
    },
    /** T510.1: the ring of a place (null: the place keeps every position), and a place's index among the blocks */
    ring: (place: number) => rings[place],
    index: (place: number) => indices[place],
    /** the cache takes its room: after everything else of the engine, for it grows where it is (the rings first, which
     * do not) */
    place() {
      if (placed) throw new Error("The cache of the keys and values has its room already.");
      placed = true;
      const ringed = rings.filter((ring) => ring !== null);
      let at = ringed.length ? room.alloc(ringed.reduce((bytes, { size }) => bytes + 4 * size * KV, 0)) : 0;
      for (const ring of ringed) {
        ring.keys = at;
        ring.values = at + 2 * ring.size * KV;
        at += 4 * ring.size * KV;
      }
      capacity = start;
      keys = room.alloc(2 * layers * capacity * KV);
      values = keys + layers * capacity * KV;
    },
    /** the positions there is room for, and where the keys and the values are now (they move as the cache grows):
     * layer a's are a * capacity * KV after */
    get capacity() { return capacity; },
    get keys() { return keys; },
    get values() { return values; },
    /** room for position pos: twice the positions, or up to pos, and no more than the context */
    grow(pos: number) {
      const larger = Math.min(Math.max(2 * capacity, pos + 1), seqLen);
      const oldLayer = capacity * KV, newLayer = larger * KV;
      room.alloc(2 * layers * (newLayer - oldLayer));
      const U = room.U, newValues = keys + layers * newLayer;
      for (let l = layers - 1; l >= 0; l--) U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);
      for (let l = layers - 1; l > 0; l--) U.copyWithin(keys + l * newLayer, keys + l * oldLayer, keys + (l + 1) * oldLayer);
      values = newValues;
      capacity = larger;
    },
    /** a token's key and value (float32, at key and value) into the cache at keyAt and valueAt */
    put(keyAt: number, valueAt: number, key: number, value: number) {
      if (half) {
        k.to_f16(keyAt, key, kvDim);
        k.to_f16(valueAt, value, kvDim);
      } else {
        const U = room.U;
        U.copyWithin(keyAt, key, key + KF);
        U.copyWithin(valueAt, value, value + KF);
      }
    },
    /** T147: a token's key and value in float16 (the GPU's) into the cache at keyAt and valueAt. T160: a float32 cache
     * (a grouped-query model's) widens them with a kernel, as attention_f16 reads them (halfToFloat one at a time took
     * 27 to 33 ns a value on CI's runners: 1.7 ms a token of Qwen3 0.6B's prompt) */
    putHalves(keyAt: number, valueAt: number, key: number, value: number) {
      if (half) {
        const U = room.U;
        U.copyWithin(keyAt, key, key + kvDim * 2);
        U.copyWithin(valueAt, value, value + kvDim * 2);
      } else {
        k.from_f16(keyAt, key, kvDim);
        k.from_f16(valueAt, value, kvDim);
      }
    },
    /** the keys and values at positions from .. from + count - 1, in float32, [layers][count][kvDim] each (tests; T510.1:
     * every place in turn, and a ring's where its position is now, which is a later one's past its window) */
    read(from: number, count: number) {
      const { F, H } = room;
      const read = (block: number, side: "keys" | "values") => {
        const out = new Float32Array(held * count * kvDim);
        for (let l = 0; l < held; l++) {
          const ring = rings[l];
          for (let t = 0; t < count; t++) {
            const at = ring ? ring[side] + ((from + t) % ring.size) * KV
              : block + indices[l] * capacity * KV + (from + t) * KV;
            for (let i = 0; i < kvDim; i++) out[(l * count + t) * kvDim + i] = half ? halfToFloat(H[at / 2 + i]) : F[at / 4 + i];
          }
        }
        return out;
      };
      return { keys: read(keys, "keys"), values: read(values, "values") };
    },
  };
}
export type Cache = ReturnType<typeof keyValueCache>;
