// which model a visitor could open with ?hf= sits in the window where pastWide() (asked with the shared size) lets it through and a
// plain memory of float32 keys and values would not hold it (16 GiB of a 64-bit memory)
import fs from "node:fs";
import { footprint, keysInHalf, needsWide, pastWide } from "../../public/forward.js";
import { CONTROL_BYTES } from "../../public/jobs.js";
const GiB = 2 ** 30;
const models = JSON.parse(fs.readFileSync(new URL("window.json", import.meta.url), "utf8"));
for (const m of models) {
  for (const dtype of ["int8", "int6"]) {
    const size = dtype === "int8" ? m.size : m.size6;
    const options = { ...m.form, dtype, int8: true, relaxed: true, halfKV: true, outliers: 8, gpu: true };
    const shared = footprint(m.header, size, { ...options, shared: true }), plain = footprint(m.header, size, { ...options, shared: false });
    const sTotal = CONTROL_BYTES + size + shared, pTotal = 64 + size + plain;
    const flag = pastWide(size, shared) ? "refused" : pTotal > 16 * GiB ? "IN THE WINDOW: passes, then a plain memory of float32 would not hold it" : "fits";
    console.log(`${m.name.padEnd(46)} ${dtype} size ${(size / GiB).toFixed(2)} GiB; shared ${(sTotal / GiB).toFixed(2)} (${keysInHalf(m.header, size, { ...options, shared: true }) ? "f16" : "f32"}), plain ${(pTotal / GiB).toFixed(2)} (${keysInHalf(m.header, size, { ...options, shared: false }) ? "f16" : "f32"}) -> ${flag}`);
  }
}
