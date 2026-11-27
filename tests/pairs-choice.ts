// tests/pairs-choice.ts (T686): forward/pairs.ts's choice of the ternary products with the sums in int16 as an engine
// makes it, the kernels cold: each try is a process of its own (in one process the second engine's kernels are already
// optimized, which no visitor's first engine has), on a matrix of Ternary Bonsai 1.7B's FFN (6144 x 2048) of made-up
// bytes (the kernels take any as fast). Says the device's answer each time, and the ms of each kernel's rounds (read
// off the clock handed in: a round is two calls, timed in the order pairsChosen() makes them). Not in a suite: CI's
// x86-64 and arm64 are where it says anything (the development machine is arm64, where nothing is timed).
//
//   node tests/pairs-choice.ts [tries = 10]
// Exit 1 where the tries did not all choose alike (a choice that flips on one device is the timing's noise).
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { built, runtimeUrl } from "./tree.ts";
import { leave } from "./leave.ts";

const args = process.argv.slice(2);
if (args[0] === "--one") {
  const { pairsChosen } = await import(runtimeUrl("forward/pairs.ts"));
  const { addressed } = await import(runtimeUrl("jobs.ts"));
  const rows = 6144, n = 2048, bytes = (rows * n) / 4 + (rows * n) / 128 * 2;
  const memory = new WebAssembly.Memory({ initial: Math.ceil((bytes + (4 << 20)) / 65536) });
  const relaxed = addressed(new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(built("simdkernel_relaxed_plain.wasm"))), { env: { memory } }).exports, false);
  const U = new Uint8Array(memory.buffer);
  for (let i = 0; i < bytes; i += 4096) U[64 + i] = (i * 2654435761) >>> 24;
  let top = 64 + bytes + 64;
  const alloc = (size) => { const at = top; top += Math.ceil(size / 64) * 64; return at; };
  const stamps = [];
  const clock = () => { const now = performance.now(); stamps.push(now); return now; };
  const tensors = { w1: { kind: "ternary", offset: 0, shape: [rows, n], group: 128, scales: (rows * n) / 4 } };
  const pairs = pairsChosen({ relaxed, memory, alloc, base: 64, tensors, direct: false, clock, pairs: undefined });
  // the rounds: round r times t2r then t2x where r is even, t2x then t2r where it is odd
  const times = { t2r: [], t2x: [] };
  for (let i = 0; i + 1 < stamps.length; i += 2) {
    const round = (i / 4) | 0, second = (i / 2) % 2 === 1;
    times[(round % 2 === 0) === !second ? "t2r" : "t2x"].push(stamps[i + 1] - stamps[i]);
  }
  const said = (xs) => xs.map((x) => x.toFixed(2)).join(" ");
  console.log(JSON.stringify({ pairs, line: stamps.length ? `t2r [${said(times.t2r)}] t2x [${said(times.t2x)}] ms` : "nothing timed" }));
  await leave(0);
} else {
  const tries = Number(args[0] ?? 10), self = fileURLToPath(import.meta.url);
  console.log(`pairs-choice: ${os.cpus()[0]?.model ?? "unknown"} (${os.arch()}, ${os.cpus().length} logical cores), Node ${process.version}, ${tries} tries`);
  const chosen = [];
  for (let t = 0; t < tries; t++) {
    const run = spawnSync(process.execPath, [self, "--one"], { encoding: "utf8" });
    const said = run.stdout.trim().split("\n").pop() ?? "";
    if (run.status !== 0 || !said.startsWith("{")) { console.log(`pairs-choice: try ${t + 1} FAILED\n${run.stdout}${run.stderr}`); await leave(1); }
    const { pairs, line } = JSON.parse(said);
    chosen.push(pairs);
    console.log(`pairs-choice: try ${t + 1}: ${pairs ? "the int16 sums" : "the int32 sums"} (${line})`);
  }
  const alike = chosen.every((x) => x === chosen[0]);
  console.log(`pairs-choice: ${chosen.filter(Boolean).length} of ${tries} tries took the int16 sums${alike ? "" : ": the choice FLIPPED on one device"}`);
  await leave(alike ? 0 : 1);
}
