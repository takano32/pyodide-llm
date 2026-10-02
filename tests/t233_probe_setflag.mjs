// does v8.setFlagsFromString("--no-liftoff") change what a module compiled afterwards computes (the canary)?
import v8 from "node:v8";
import { Worker, isMainThread, parentPort } from "node:worker_threads";
const GiB = 2 ** 30, MiB = 2 ** 20;
const canary = Uint8Array.from([0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 126, 1, 125, 2, 15, 1, 3, 101, 110, 118, 6, 109, 101, 109, 111, 114, 121, 2, 4, 1,
  3, 2, 1, 0, 7, 9, 1, 5, 115, 112, 108, 97, 116, 0, 0, 10, 13, 1, 11, 0, 32, 0, 253, 9, 2, 0, 253, 31, 0, 11]);
function splatsRight() {
  const high = 4 * GiB + 2 * 65536, at = high + 4096;
  const memory = new WebAssembly.Memory({ initial: BigInt(Math.ceil((high + 4 * MiB) / 65536)), address: "i64" });
  const splat = new WebAssembly.Instance(new WebAssembly.Module(canary), { env: { memory } }).exports.splat;
  const F = new Float32Array(memory.buffer);
  F[at / 4] = 1.5;
  F[(at - 2 ** 32) / 4] = 2.5;
  return splat(BigInt(at)) === 1.5;
}
if (isMainThread) {
  console.log(`${process.arch} V8 ${process.versions.v8}`);
  console.log(`default: the canary reads ${splatsRight() ? "right" : "WRONG"}`);
  v8.setFlagsFromString("--no-liftoff");
  console.log(`after v8.setFlagsFromString("--no-liftoff"): the canary reads ${splatsRight() ? "right" : "WRONG"}`);
  const worker = new Worker(new URL(import.meta.url));
  worker.on("message", (message) => console.log(`in a worker created after it: ${message}`));
} else {
  parentPort.postMessage(`the canary reads ${splatsRight() ? "right" : "WRONG"}`);
}
