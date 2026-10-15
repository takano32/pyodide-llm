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
//
// T331: the keys and values of a place may be narrower than the widest (a Gemma 4's sliding layers have heads of 256
// where its full ones have 512): hold() says each place's width, and a block or a ring is of its own place's. A layer's
// block is offset(index) bytes a position after the start of the keys (and of the values): index times KV where every
// place is as wide as the widest, which is every model before a Gemma 4's.
import { halfToFloat, ringPositions } from "./memory.ts";
import type { Kernels } from "../jobs.ts";
import type { Room } from "./kinds.ts";

// T510.1: a layer's ring: its period, and where its keys and its values are (place() says, 2 * size positions each);
// T331: and the bytes of a position of it (its place's own)
export type Ring = { size: number; keys: number; values: number; row: number };

/** room, k: the engine's memory and kernels. kvDim: the values of one position's keys (T331: of the widest place's);
 * half: kept in float16 (T110);
 * start: the positions there is room for at first (0: no cache, T210: a model on the GPU alone); seqLen: the most. */
export function keyValueCache({ room, k, kvDim, half, start, seqLen }: { room: Room; k: Kernels; kvDim: number; half: boolean; start: number; seqLen: number }) {
  // KV: the bytes of one position's keys in the cache (of the widest place's)
  const KV = kvDim * (half ? 2 : 4);
  // held: the places taken; layers: of those, the ones that keep every position (the blocks that grow); for every
  // place, its index among those (-1: a ring's), and the rings by place. T331: every place's width, and of the
  // blocks, the bytes a position of each and of those before it (before; total, of them all)
  let held = 0, layers = 0, capacity = 0, keys = 0, values = 0, placed = false, total = 0;
  const indices: number[] = [], rings: (Ring | null)[] = [], widths: number[] = [], rows: number[] = [], before: number[] = [];
  // (the bytes a position of the block of index l)
  const rowOf = (l: number) => (l + 1 < layers ? before[l + 1] : total) - before[l];
  return {
    KV, half,
    /** count more layers keep their keys and values here: the first of their places (before place()). T510.1: windows,
     * how many of the last positions each attends (0 or none: all of them): one that a ring holds keeps no more. T331:
     * wide, the values of a position of each (none: kvDim; T330, 0: the place keeps none, its layer reads another's) */
    hold(count: number, windows: number[] = [], wide: number[] = []) {
      if (placed) throw new Error("The cache of the keys and values has its room already: a kind of layer takes its places as it is made.");
      const first = held;
      for (let a = 0; a < count; a++) {
        const width = wide[a] ?? kvDim, row = width * (half ? 2 : 4), size = start && width ? ringPositions(windows[a] ?? 0, seqLen) / 2 : 0;
        if (width > kvDim) throw new Error(`A layer keeps keys and values ${width} wide, and the cache's widest are ${kvDim}.`);
        widths.push(width);
        rows.push(row);
        rings.push(size ? { size, keys: 0, values: 0, row } : null);
        if (!size && width) {
          before.push(total);
          total += row;
        }
        indices.push(size || !width ? -1 : layers++);
      }
      held += count;
      return first;
    },
    /** T510.1: the ring of a place (null: the place keeps every position), and a place's index among the blocks */
    ring: (place: number) => rings[place],
    index: (place: number) => indices[place],
    /** T331: the values and the bytes of a position of a place, and of a block (by its index) the bytes a position of
     * those before it: the block's keys are capacity times that after keys, and its values after values */
    width: (place: number) => widths[place],
    row: (place: number) => rows[place],
    offset: (index: number) => before[index],
    /** the cache takes its room: after everything else of the engine, for it grows where it is (the rings first, which
     * do not) */
    place() {
      if (placed) throw new Error("The cache of the keys and values has its room already.");
      placed = true;
      const ringed = rings.filter((ring) => ring !== null);
      let at = ringed.length ? room.alloc(ringed.reduce((bytes, { size, row }) => bytes + 4 * size * row, 0)) : 0;
      for (const ring of ringed) {
        ring.keys = at;
        ring.values = at + 2 * ring.size * ring.row;
        at += 4 * ring.size * ring.row;
      }
      capacity = start;
      keys = room.alloc(2 * capacity * total);
      values = keys + capacity * total;
    },
    /** the positions there is room for, and where the keys and the values are now (they move as the cache grows):
     * the block of index i's are capacity * offset(i) after (i * capacity * KV, where the places are as wide) */
    get capacity() { return capacity; },
    get keys() { return keys; },
    get values() { return values; },
    /** room for position pos: twice the positions, or up to pos, and no more than the context */
    grow(pos: number) {
      const larger = Math.min(Math.max(2 * capacity, pos + 1), seqLen);
      room.alloc(2 * total * (larger - capacity));
      const U = room.U, newValues = keys + larger * total;
      // (T331: a block of each one's own bytes; where they are alike, before[l] is l * KV and these are the moves they
      // always were)
      for (let l = layers - 1; l >= 0; l--) U.copyWithin(newValues + larger * before[l], values + capacity * before[l], values + capacity * (before[l] + rowOf(l)));
      for (let l = layers - 1; l > 0; l--) U.copyWithin(keys + larger * before[l], keys + capacity * before[l], keys + capacity * (before[l] + rowOf(l)));
      values = newValues;
      capacity = larger;
    },
    /** a token's key and value (float32, at key and value) into the cache at keyAt and valueAt (T331: width values
     * of each, a place's own; kvDim where it is not said) */
    put(keyAt: number, valueAt: number, key: number, value: number, width = kvDim) {
      if (half) {
        k.to_f16(keyAt, key, width);
        k.to_f16(valueAt, value, width);
      } else {
        const U = room.U;
        U.copyWithin(keyAt, key, key + width * 4);
        U.copyWithin(valueAt, value, value + width * 4);
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
     * every place in turn, and a ring's where its position is now, which is a later one's past its window). T331: a
     * place's are its own width, one place's after another's */
    read(from: number, count: number) {
      const { F, H } = room;
      const read = (block: number, side: "keys" | "values") => {
        const out = new Float32Array(widths.reduce((sum, width) => sum + width, 0) * count);
        let to = 0;
        for (let l = 0; l < held; l++) {
          const ring = rings[l], width = widths[l], row = rows[l];
          for (let t = 0; t < count; t++, to += width) {
            const at = ring ? ring[side] + ((from + t) % ring.size) * row
              : block + capacity * before[indices[l]] + (from + t) * row;
            for (let i = 0; i < width; i++) out[to + i] = half ? halfToFloat(H[at / 2 + i]) : F[at / 4 + i];
          }
        }
        return out;
      };
      return { keys: read(keys, "keys"), values: read(values, "values") };
    },
  };
}
export type Cache = ReturnType<typeof keyValueCache>;
