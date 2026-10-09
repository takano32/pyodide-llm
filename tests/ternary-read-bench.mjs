// T273: how fast the converter reads a GGUF's two ternary types (PQ2_0, PTQ1_0) in Pyodide, by NumPy and on the
// kernels, in turns in one process, after seeing that the two give the same float32. MB/s of the file's bytes.
//   node tests/ternary-read-bench.mjs [megabytes of blocks, default 32] [rounds, default 5]
import os from "node:os";
import { pyodideWithEngine } from "./engine.mjs";

const megabytes = Number(process.argv[2] ?? 32), rounds = Number(process.argv[3] ?? 5);
const { pyodide: py } = await pyodideWithEngine();
py.globals.set("megabytes", megabytes);
py.globals.set("rounds", rounds);
const lines = py.runPython(`
import time
import numpy as np
import llama2_convert, llama2_numpy

readers = llama2_numpy.kernel_ternary_readers("simdkernel.so")
rng = np.random.default_rng(3)
lines = []
for kind, size, by_numpy in (("PQ2_0", 34, llama2_convert.pq2_0), ("PTQ1_0", 28, llama2_convert.ptq1_0)):
    # the pieces the page's converter reads are rows; a megabyte at a time is near a wide row of the 27B
    piece = (1 << 20) // size * size
    pieces = [rng.integers(0, 243, piece, dtype=np.uint8).tobytes() for _ in range(megabytes)]
    assert all(np.array_equal(readers[kind](raw).view(np.uint32), by_numpy(raw).view(np.uint32)) for raw in pieces[:4])
    seconds = {"NumPy": [], "the kernel": []}
    for _ in range(rounds):
        for name, read in (("NumPy", by_numpy), ("the kernel", readers[kind])):
            began = time.perf_counter()
            for raw in pieces:
                read(raw)
            seconds[name].append(time.perf_counter() - began)
    rate = {name: megabytes * piece / (1 << 20) / sorted(times)[len(times) // 2] for name, times in seconds.items()}
    lines.append(f"ternary-read-bench: {kind}: NumPy {rate['NumPy']:.1f} MB/s, the kernel {rate['the kernel']:.1f} MB/s "
                 f"({rate['the kernel'] / rate['NumPy']:.1f} times), the median of {rounds} rounds of {megabytes} MB")
lines
`).toJs();
console.log(`ternary-read-bench: ${os.cpus()[0].model}, ${os.arch()}`);
for (const line of lines) console.log(line);
