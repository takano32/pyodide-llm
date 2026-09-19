# Build the kernels as Emscripten side modules that Pyodide's ctypes.CDLL can load, without Emscripten:
# compile with AssemblyScript, then prepend the custom section "dylink.0" that Emscripten's loader looks for
# (WASM_DYLINK_MEM_INFO with memory size, memory alignment, table size and table alignment all zero).
#
#   npm install --no-save assemblyscript && python3 build.py      ->  simdkernel.so, simdkernel_relaxed.wasmlib
#
# The relaxed module is NOT named *.so on purpose: Pyodide's package installer pre-loads every .so of a wheel,
# and a browser without relaxed SIMD fails to compile it.
import subprocess
from pathlib import Path

ASC = ["npx", "asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1"]


def side_module(source, out, features):
    subprocess.run(ASC + [source, "-o", "tmp.wasm", "--enable", features], check=True)
    wasm = Path("tmp.wasm").read_bytes()
    Path("tmp.wasm").unlink()
    # No data section (id 11) may exist: nothing would relocate it, and it would overwrite Pyodide's own memory
    position, section_ids = 8, []
    while position < len(wasm):
        section_ids.append(wasm[position])
        position += 1
        size = shift = 0
        while True:
            byte = wasm[position]
            position += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        position += size
    assert 11 not in section_ids, f"{source} has a data segment: avoid std math, strings and asserts"
    name = b"dylink.0"
    payload = bytes([len(name)]) + name + bytes([1, 4, 0, 0, 0, 0])
    Path(out).write_bytes(wasm[:8] + bytes([0, len(payload)]) + payload + wasm[8:])
    print(out, Path(out).stat().st_size, "bytes")


if __name__ == "__main__":
    side_module("kernel.ts", "simdkernel.so", "simd")
    side_module("kernel_relaxed.ts", "simdkernel_relaxed.wasmlib", "simd,relaxed-simd")
