// wasm64.mjs (T101): an AssemblyScript transform that makes kernels for a 64-bit memory (Memory64), for models
// past the 4 GiB of a 32-bit one. asc itself always sets the 32-bit target, and the compiler's Wasm64 target is
// not finished there, so this finishes it, in the two places it stops (measured with AssemblyScript 0.28.20):
//   afterParse: the target, and the size type of the module the program made before any transform ran (the
//     module picks i32 or i64 operations on addresses by it: left at i32, every usize sum came out as i32.add
//     on i64 values and failed validation).
//   afterCompile: the memory, declared 32-bit whatever the target: declared again as 64-bit, imported as before,
//     and a shared one with a maximum of 16 GiB (what Chrome allows a 64-bit memory).
// Used by kernels/build.py (--transform). The kernels' source is the same; usize is 64 bits there, so JavaScript
// passes their addresses as BigInt.
import { Transform } from "assemblyscript/transform";
import * as assemblyscript from "assemblyscript";
import binaryen from "binaryen";

const WASM64 = 2;  // assemblyscript's Target.Wasm64
const MAXIMUM_PAGES = 262144;  // 16 GiB of 64 KiB pages

export default class Wasm64 extends Transform {
  afterParse() {
    assemblyscript.setTarget(this.program.options, WASM64);
    this.program.module.sizeType = binaryen.i64;
  }

  afterCompile(module) {
    const memory = module.getMemoryInfo("0");
    module.setMemory(memory.initial, memory.shared ? MAXIMUM_PAGES : -1, null, [], memory.shared, true, "0");
    if (memory.module) module.addMemoryImport("0", memory.module, memory.base, memory.shared);
  }
}
