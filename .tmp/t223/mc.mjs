// T223's review: does the search settle on a device like the owner's Android's, with noise? The real search of forward.js
// (sim.mjs's visit()), many visits that remember what the last one found, as the page does.
//   node .tmp/t223/mc.mjs [visits]
import { DEVICES, visit } from "./sim.mjs";

const visits = Number(process.argv[2] ?? 400);
const noises = {
  "no noise": {},
  "sigma .05": { sigma: 0.05 },
  "sigma .10": { sigma: 0.10 },
  "sigma .20": { sigma: 0.20 },
  "sigma .10, 20% of blocks x3": { sigma: 0.10, pBlock: 0.2, blockFactor: 3 },
  "sigma .20, 30% of blocks x2, 2% of tokens x4": { sigma: 0.20, pBlock: 0.3, blockFactor: 2, pSpike: 0.02, spike: 4 },
  "sigma .10, heat 30%": { sigma: 0.10, drift: 0.3 },
  "sigma .30": { sigma: 0.30 },
};
const fmt = (counts, n) => Object.entries(counts).sort((a, b) => Number(a[0]) - Number(b[0])).map(([c, k]) => `${c}:${(100 * k / n).toFixed(1)}%`).join(" ");

for (const [deviceName, device] of Object.entries(DEVICES)) {
  console.log(`\n=== ${deviceName}: ${Object.entries(device).map(([n, ms]) => `${n} threads ${ms.toFixed(2)} ms`).join(", ")}`);
  for (const [noiseName, noise] of Object.entries(noises)) {
    // (a) a chain of visits, each remembering what the last found (the first from the logical cores: 8)
    let remembered = 0;
    const final = {}, transitions = {};
    let silent = 0;
    for (let v = 0; v < visits; v++) {
      const r = await visit(device, { remembered, hint: 8, ...noise });
      const found = r.told.at(-1);
      if (found === undefined) silent++;
      const next = found ?? remembered;
      final[next] = (final[next] ?? 0) + 1;
      if (v > 0) { const key = `${remembered}->${next}`; transitions[key] = (transitions[key] ?? 0) + 1; }
      remembered = next;
    }
    const moves = Object.entries(transitions).filter(([k]) => k.split("->")[0] !== k.split("->")[1]).reduce((s, [, c]) => s + c, 0);
    // (b) from each remembered count alone: where does one visit end?
    const from = {};
    for (const start of [1, 2, 4, 8]) {
      const ends = {};
      for (let v = 0; v < Math.min(visits, 60); v++) {
        const r = await visit(device, { remembered: start, hint: 8, ...noise });
        const found = r.told.at(-1) ?? start;
        ends[found] = (ends[found] ?? 0) + 1;
      }
      from[start] = fmt(ends, Math.min(visits, 60));
    }
    console.log(`${noiseName.padEnd(46)} chain: ${fmt(final, visits).padEnd(40)} changes between visits ${(100 * moves / (visits - 1)).toFixed(1)}%${silent ? `, ${silent} visits with no verdict` : ""}`);
    console.log(`${"".padEnd(46)} one visit from 1 -> ${from[1]} | from 2 -> ${from[2]} | from 4 -> ${from[4]} | from 8 -> ${from[8]}`);
  }
}
