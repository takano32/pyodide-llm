// The engine as the page runs it, for the tests and measurements that run in Node (T93): Pyodide with
// llama2_numpy.py and the sampling kernels, and the forward pass in public/forward.js on a WebAssembly memory of its
// own. Python gets one function for it:
//
//   kernel_llama(checkpoint, tokenizer, **options)
//
// which is Llama(checkpoint, tokenizer, kernels="simdkernel.so", **options) as it used to be: the checkpoint (bytes)
// is copied into the memory of forward.js. With "kernels" in disable it is the NumPy engine, as on the page. And
//
//   kernel_llama_file(path, tokenizer, **options)
//
// reads a checkpoint file of this machine straight into the memory of forward.js, never into Pyodide's: a model of
// gigabytes does not fit Pyodide's heap as a file and as bytes besides (T100).
import fs from "node:fs";
import { loadPyodide } from "pyodide";
import { compileKernels, external, weightsMemory } from "../public/forward.js";

const root = new URL("../", import.meta.url).pathname;

// shared: false gives the memory the page has where it is not cross-origin isolated (one thread, float32 keys)
export async function pyodideWithEngine({ shared = true } = {}) {
  const pyodide = await loadPyodide();
  await pyodide.loadPackage("numpy", { messageCallback: () => {} });
  for (const name of ["llama2_numpy.py", "llama2_convert.py", "simdkernel.so", "simdkernel_relaxed.wasmlib"]) {
    pyodide.FS.writeFile(name, fs.readFileSync(`${root}public/${name}`));
  }
  // a shared memory, as the page has where it is cross-origin isolated (since T93 stage 3, the usual case), so that
  // what these tests run is what the page runs: forward.js keeps an int8 model's keys and values in float16 there
  // (T110). No software threads are started: those are tests/threads-check.mjs's.
  const variant = shared ? "shared" : "plain";
  const kernels = compileKernels(fs.readFileSync(`${root}public/simdkernel_${variant}.wasm`), fs.readFileSync(`${root}public/simdkernel_relaxed_${variant}.wasm`));
  // a checkpoint (Python bytes) copied into a memory of forward.js, as what Llama(external=) takes
  pyodide.globals.set("outside", (data) => {
    const view = data.getBuffer("u8");
    const size = view.data.length;
    const { memory, base } = weightsMemory(size, { shared });
    new Uint8Array(memory.buffer, base, size).set(view.data);
    view.release();
    return external({ memory, base, size, kernels });
  });
  pyodide.globals.set("outside_file", (file) => {
    const size = fs.statSync(file).size;
    const { memory, base } = weightsMemory(size, { shared });
    const fd = fs.openSync(file, "r");
    for (let offset = 0; offset < size;) {
      offset += fs.readSync(fd, new Uint8Array(memory.buffer, base + offset, Math.min(64 << 20, size - offset)), 0, Math.min(64 << 20, size - offset), offset);
    }
    fs.closeSync(fd);
    return external({ memory, base, size, kernels });
  });
  pyodide.runPython(`
import llama2_numpy

def kernel_llama(checkpoint, tokenizer, **options):
    """The engine as the page runs it: the forward pass in forward.js, the sampling on simdkernel.so (T93)."""
    if "kernels" in tuple(options.get("disable", ())):
        return llama2_numpy.Llama(checkpoint, tokenizer, kernels="simdkernel.so", **options)
    return llama2_numpy.Llama(None, tokenizer, kernels="simdkernel.so", external=outside(checkpoint), **options)

def kernel_llama_file(path, tokenizer, **options):
    """The same with the checkpoint read from a file of this machine into the memory of forward.js directly."""
    return llama2_numpy.Llama(None, tokenizer, kernels="simdkernel.so", external=outside_file(path), **options)
`);
  return { pyodide, kernels };
}
