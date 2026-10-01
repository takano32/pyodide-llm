import { DEVICES, visit } from "./sim.mjs";
const began = performance.now();
for (let i = 0; i < 40; i++) await visit(DEVICES.android, { remembered: 4, sigma: 0.1 });
console.log("per visit (ms):", ((performance.now() - began) / 40).toFixed(1));
