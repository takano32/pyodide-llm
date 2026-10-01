// the cases of footprints.json closest to the edges: 4 GiB of a 32-bit memory, 16 GiB of a 64-bit one, and a shared
// memory's lowered maximum (checkpoint + 1 GiB, checkpoint + 256 MiB)
import fs from "node:fs";
const out = JSON.parse(fs.readFileSync(new URL("footprints.json", import.meta.url), "utf8"));
const GiB = 2 ** 30, MiB = 2 ** 20;
const show = (o) => `${o.id.padEnd(34)} ${o.dtype.padEnd(5)} ${o.browser.padEnd(26)} size ${(o.size / GiB).toFixed(3)} shared after ${(o.sharedAfter / GiB).toFixed(3)} (${o.halfShared ? "f16" : "f32"}, ${o.wideShared ? "64" : "32"}-bit) plain after ${(o.plainAfter / GiB).toFixed(3)} (${o.halfPlain ? "f16" : "f32"}) shared total ${(o.sharedTotal / GiB).toFixed(3)} plain total ${(o.refusedTotal / GiB).toFixed(3)}`;
console.log("--- 32-bit shared cases by the room left under 4 GiB on the plain memory (smallest first)");
for (const o of out.filter((o) => !o.wideShared).map((o) => ({ ...o, room: 4 * GiB - o.refusedTotal })).sort((a, b) => a.room - b.room).slice(0, 12)) {
  console.log(`room ${(o.room / MiB).toFixed(0).padStart(6)} MiB  ${show(o)}`);
}
console.log("--- 32-bit shared cases by the room left under 4 GiB on the shared memory (smallest first)");
for (const o of out.filter((o) => !o.wideShared).map((o) => ({ ...o, room: 4 * GiB - o.sharedTotal })).sort((a, b) => a.room - b.room).slice(0, 8)) {
  console.log(`room ${(o.room / MiB).toFixed(0).padStart(6)} MiB  ${show(o)}`);
}
console.log("--- 64-bit cases by the room left under 16 GiB on the plain memory (smallest first)");
for (const o of out.filter((o) => o.wideShared).map((o) => ({ ...o, room: 16 * GiB - o.refusedTotal })).sort((a, b) => a.room - b.room).slice(0, 6)) {
  console.log(`room ${(o.room / MiB).toFixed(0).padStart(6)} MiB  ${show(o)}`);
}
// the lowered maxima of weightsMemory(): checkpoint + 16384 pages (1 GiB), + 4096 pages (256 MiB), against what the
// shared engine puts after the checkpoint
console.log("--- shared memory at a lowered maximum: after > 1 GiB (the second try) or > 256 MiB (the third)");
const lowered = out.filter((o) => o.browser === "Chrome/Firefox" && o.dtype === "int8");
console.log(`  models whose shared after exceeds 1 GiB:   ${lowered.filter((o) => o.sharedAfter > GiB).map((o) => o.id).join(", ")}`);
console.log(`  models whose shared after exceeds 256 MiB: ${lowered.filter((o) => o.sharedAfter > 256 * MiB).length} of ${lowered.length}`);
