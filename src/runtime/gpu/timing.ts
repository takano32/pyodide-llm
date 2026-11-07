// The two ways the model's GPU worker times its forms at a load (T642.5: one loop each, which forms.ts,
// tokenattention.ts, block.ts and tokenforms.ts wrote out twice), read on the model's clock m.now (T642.4).
// (a module of the model's GPU worker, gpu.ts, which a bundler links it into)

import { stopping } from "./device.ts";
import type { Model } from "./device.ts";

// pairedMs: what a submission costs besides its work drops out of the difference of a submission of n passes and one
// of 2n (2.6 to 8.4 ms waited for on the owner's Android, T134, where llm-jp-3 150M's gate by 64 tokens is 0.6 ms of
// work). Each item warmed by a submission of one, n doubled from 1 until a submission of n takes TIMED_MS (at most
// most(MOST_PASSES)), then PAIRS rounds of a pair an item, the items in turn within each round (a device that warms up
// or is loaded meanwhile falls on all of them alike: T168's review), the median of each item's pairs, per pass. On a
// fallback adapter (the CPU in the GPU's place: its times are no GPU's) one pair of one pass. submit(item, n) records
// n passes of the item and returns them encoded; a stop ends the rounds after the one under way.
const TIMED_MS = 20, MOST_PASSES = 256, PAIRS = 5;
async function pairedMs<T>(m: Model, items: T[], record: (item: T, passes: number) => GPUCommandBuffer, most = MOST_PASSES) {
  const { device } = m;
  const submission = async (item: T, passes: number) => {
    const commands = record(item, passes), began = m.now();
    device.queue.submit([commands]);
    await device.queue.onSubmittedWorkDone();
    return m.now() - began;
  };
  const counts: number[] = [], differences: number[][] = items.map(() => []);
  for (const item of items) {
    await submission(item, 1);  // warm
    let n = 1;
    while (!m.fallback && n < most && (await submission(item, n)) < TIMED_MS) n *= 2;
    counts.push(n);
  }
  for (let round = 0; round < (m.fallback ? 1 : PAIRS); round++) {
    for (const [i, item] of items.entries()) {
      const once = await submission(item, counts[i]), twice = await submission(item, 2 * counts[i]);
      differences[i].push((twice - once) / counts[i]);
    }
    if (stopping) break;
  }
  return differences.map(median);
}

// roundsMs: ms of run(item), each item run once to warm up (the pipelines' first dispatches, the weights' first
// reads), then ROUNDS rounds of the items in turn (one on a fallback adapter), the median of each. A stop ends the
// rounds after the one under way.
const ROUNDS = 3;
async function roundsMs<T>(m: Model, items: T[], run: (item: T) => Promise<unknown>) {
  const timed = async (item: T) => {
    const began = m.now();
    await run(item);
    return m.now() - began;
  };
  for (const item of items) await timed(item);
  const times: number[][] = items.map(() => []);
  for (let round = 0; round < (m.fallback ? 1 : ROUNDS); round++) {
    for (const [i, item] of items.entries()) times[i].push(await timed(item));
    if (stopping) break;
  }
  return times.map(median);
}

function median(list: number[]) {
  return list.sort((a, b) => a - b)[list.length >> 1];
}

export { TIMED_MS, MOST_PASSES, PAIRS, ROUNDS, pairedMs, roundsMs };
