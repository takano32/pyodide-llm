// T130's review: footprint() and the type of the keys and values for every listed model, on a shared memory, a plain
// one and a shared one that was asked for and refused (sized as shared, engine on a plain memory).
import fs from "node:fs";
import { footprint, keysInHalf, needsWide, pastWide } from "../../public/forward.js";
import { CONTROL_BYTES } from "../../public/jobs.js";

const rows = JSON.parse(fs.readFileSync(new URL("sizes.json", import.meta.url), "utf8"));
const GiB = 2 ** 30, MiB = 2 ** 20;
const total = (size, after, base) => base + size + after;
const lines = [];
let violations = 0;
const note = (text) => { lines.push(text); };

// scenarios: the browser (relaxed SIMD or not: Safari has none; WebGPU or not), the bits
const browsers = { "Chrome/Firefox": { relaxed: true, gpu: true }, "Safari (no relaxed, GPU)": { relaxed: false, gpu: true }, "no GPU": { relaxed: true, gpu: false } };
const out = [];
for (const row of rows) {
  const dtypes = row.id.endsWith("-f16") ? ["float16"] : row.id.endsWith("-f32") ? ["float32"] : row.id === "stories260K" || row.id === "stories3_5M" ? ["float32"] : ["int8", "int6"];
  for (const dtype of dtypes) {
    const size = row[`size_${dtype}`], header = row.header;
    for (const [browser, { relaxed, gpu }] of Object.entries(browsers)) {
      const quantized = dtype === "int8" || dtype === "int6";
      const options = { ...row.form, dtype, int8: true, relaxed, halfKV: quantized, outliers: 8, gpu };
      const sharedAfter = footprint(header, size, { ...options, shared: true });
      const plainAfter = footprint(header, size, { ...options, shared: false });
      const halfShared = keysInHalf(header, size, { ...options, shared: true });
      const halfPlain = keysInHalf(header, size, { ...options, shared: false });
      const wideShared = needsWide(size, sharedAfter), widePlain = needsWide(size, plainAfter);
      // the worker sizes the memory by sharedAfter (wanted), asks pastWide with it, and on a refused shared memory the
      // engine is on a plain memory of the same width (pooledWeights(..., wide)) with halfPlain
      const refusedTotal = total(size, plainAfter, 64), sharedTotal = total(size, sharedAfter, CONTROL_BYTES);
      const checks = [];
      // I1: shared fits a 32-bit memory -> the plain one does
      if (!wideShared && widePlain) checks.push("I1 plain past 4 GiB where shared fits");
      // I2: a plain memory of the width the shared one asked for holds the plain engine (64-bit: 16 GiB; 32-bit: 4 GiB)
      const limit = wideShared ? 262144 * 65536 : 65536 * 65536;
      if (refusedTotal > limit) checks.push(`I2 plain total ${(refusedTotal / GiB).toFixed(2)} GiB past ${limit / GiB} GiB`);
      // I3: pastWide with the shared size agrees with the plain one
      if (pastWide(size, sharedAfter) !== pastWide(size, plainAfter) && !pastWide(size, sharedAfter)) checks.push("I3 plain past 16 GiB where shared fits");
      if (checks.length) violations++;
      out.push({ id: row.id, dtype, browser, size, sharedAfter, plainAfter, halfShared, halfPlain, wideShared, widePlain, sharedTotal, refusedTotal, checks });
    }
  }
}
fs.writeFileSync(new URL("footprints.json", import.meta.url), JSON.stringify(out, null, 1));

const fmt = (n) => (n / GiB).toFixed(3).padStart(7);
console.log("id".padEnd(36), "dtype", "browser".padEnd(26), "size   ", "sh.after", "pl.after", "half sh/pl", "wide sh/pl", "sh.total", "rf.total", "checks");
for (const o of out) {
  if (o.browser !== "Chrome/Firefox" || o.dtype === "int6") continue;
  console.log(o.id.padEnd(36), o.dtype.padEnd(5), o.browser.padEnd(26), fmt(o.size), fmt(o.sharedAfter), fmt(o.plainAfter),
    `${o.halfShared ? "f16" : "f32"}/${o.halfPlain ? "f16" : "f32"}`.padEnd(9), `${o.wideShared ? "64" : "32"}/${o.widePlain ? "64" : "32"}`.padEnd(6),
    fmt(o.sharedTotal), fmt(o.refusedTotal), o.checks.join("; "));
}
console.log(`\n${out.length} (model, dtype, browser) cases, ${violations} with a violated invariant`);
for (const o of out.filter((o) => o.checks.length)) console.log("VIOLATION", o.id, o.dtype, o.browser, o.checks);
