// forward/staging.ts (T.502.2): the keys and values the GPU's worker put into the staging place of the shared memory
// (T.147: float16, [keys, values][layer][GPU_BLOCK positions]), looked at (T.243) and written into the cache (cache.ts).
// createForward() (engine.ts) makes it once, where the GPU may take its blocks. A module of src/runtime/forward.ts.

import { GPU_BLOCK } from "./choice.ts";
import type { Kernels } from "../jobs.ts";
import type { Cache } from "./cache.ts";

/** staging: the place's address (0: none); attending: the layers that keep keys and values in the cache (in the cache's
 * order); kvDim: a position's keys of a layer */
export function stagingReader({ k, cache, staging, attending, kvDim }: { k: Kernels; cache: Cache; staging: number; attending: number; kvDim: number }) {
  const { KV } = cache;
  // T.147: the keys and values of count positions from pos that the GPU's worker put into staging (float16, [keys,
  // values][layer][GPU_BLOCK positions]), into the cache
  // (T.545: by the cache's places, the attending layers in order: every layer where every layer attends; and the
  // staging place lays out as many)
  function fromStaging(pos: number, count: number) {
    const { capacity } = cache;
    cache.written(pos);  // (T.244)
    for (let l = 0; l < attending; l++) {
      // T.261.4: a layer whose window a ring holds (T.510.1): each position at its two places there, and the ring told of
      // the positions written, as the layer itself tells it (attention.ts: a run that would go back past what the ring
      // still holds is refused there)
      const ring = cache.ring(l);
      if (ring) {
        const twice = ring.size * ring.row;
        for (let t = 0; t < count; t++) {
          const slot = ((pos + t) % ring.size) * ring.row, key = staging + (l * GPU_BLOCK + t) * kvDim * 2, value = staging + ((attending + l) * GPU_BLOCK + t) * kvDim * 2;
          cache.putHalves(ring.keys + slot, ring.values + slot, key, value);
          cache.putHalves(ring.keys + slot + twice, ring.values + slot + twice, key, value);
        }
        ring.latest = pos === 0 ? count - 1 : Math.max(ring.latest, pos + count - 1);
        continue;
      }
      // (a layer's block among those that keep every position: l * KV a position in, where no ring is)
      const offset = capacity * cache.offset(cache.index(l)), layerKeys = cache.keys + offset, layerValues = cache.values + offset;
      for (let t = 0; t < count; t++) {
        cache.putHalves(layerKeys + (pos + t) * KV, layerValues + (pos + t) * KV,
          staging + (l * GPU_BLOCK + t) * kvDim * 2, staging + ((attending + l) * GPU_BLOCK + t) * kvDim * 2);
      }
    }
  }
  // T.243: whether the keys and values of count positions in staging are all finite numbers. The CPU's readers of a
  // float16 (the kernels' halves4, and so from_f16 and attention_f16) make a finite number of a NaN and of an infinity
  // (65536 and more), so that what a GPU computed wrong would be read from the cache as numbers ever after, by the CPU's
  // steps and its logits (T.195 sees nothing then). They are looked at by their bits (finite_f16: the exponent's five)
  // before anything of them is written: one pass of a kernel over what fromStaging reads, none in the attention's loops
  function stagingFinite(count: number) {
    for (let part = 0; part < 2 * attending; part++) {
      if (!k.finite_f16(staging + part * GPU_BLOCK * kvDim * 2, count * kvDim)) return false;
    }
    return true;
  }
  return { fromStaging, stagingFinite };
}
