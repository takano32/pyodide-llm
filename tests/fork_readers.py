"""The readers of Prism ML's two GGUF types (llama2_convert.pq2_0 and ptq1_0) against the fork's own C, at the commit the
repository pins (T230's review). Nothing is copied here: the fork's ggml-quants.c is fetched (MIT, ggml authors) and its
four functions, dequantize_row_pq2_0, dequantize_row_ptq1_0 and the two quantizers, are cut out of it by name and compiled
with a harness of a few lines of this file's own (the block structs, a float16 by _Float16), then run on

  - random bytes in every place of 40,000 blocks of each type (so every byte value in every place, the code 3 of PQ2_0
    that no quantizer writes, the bytes of PTQ1_0 above 242 that its dequantizer still reads as digits), with float16 scales
    of every kind (normal, subnormal, plus and minus zero, 65504, negative): the reader's float32 values and the fork's
    must be the same bits;
  - the fork's own quantizers' bytes of 24,000 ternary groups (a third of them with a zero or two): read back they must be
    the values, and ternary() of them must take back the scale.

  python tests/fork_readers.py [work directory = .tmp/fork-readers]   (needs gcc and the network; for CI's extra=)
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
import llama2_convert  # noqa: E402
from llama2_numpy import ternary  # noqa: E402

COMMIT = "88c4bc60b9c9578f134385be9535e853f2db9b9f"
SOURCE = f"https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/{COMMIT}/ggml/src/ggml-quants.c"
FUNCTIONS = ["void dequantize_row_pq2_0", "void dequantize_row_ptq1_0", "void quantize_row_pq2_0_ref", "void quantize_row_ptq1_0_ref"]

HARNESS = r"""
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define GGML_RESTRICT __restrict__
#define MAX(a, b) ((a) > (b) ? (a) : (b))
typedef uint16_t ggml_half;
static float half_to_float(uint16_t h) { _Float16 f; memcpy(&f, &h, 2); return (float) f; }
static uint16_t float_to_half(float x) { _Float16 f = (_Float16) x; uint16_t h; memcpy(&h, &f, 2); return h; }
#define GGML_FP16_TO_FP32(x) half_to_float(x)
#define GGML_FP32_TO_FP16(x) float_to_half(x)
typedef struct { ggml_half d; uint8_t qs[32]; } block_pq2_0;
typedef struct { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; } block_ptq1_0;
#define QK_PQ2_0 128
#define QK_PTQ1_0 128
static const size_t ptq1_0_stages[3] = {32, 16, 8};
#include "functions.c"
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  FILE *in = fopen(argv[2], "rb"), *out = fopen(argv[3], "wb");
  if (!in || !out) return 3;
  fseek(in, 0, SEEK_END); long size = ftell(in); fseek(in, 0, SEEK_SET);
  uint8_t *data = malloc(size);
  if (fread(data, 1, size, in) != (size_t) size) return 4;
  if (!strcmp(argv[1], "dequant-pq2")) {
    assert(sizeof(block_pq2_0) == 34);
    long blocks = size / 34; float *y = malloc(blocks * 128 * sizeof(float));
    dequantize_row_pq2_0((const block_pq2_0 *) data, y, blocks * 128); fwrite(y, sizeof(float), blocks * 128, out);
  } else if (!strcmp(argv[1], "dequant-ptq")) {
    assert(sizeof(block_ptq1_0) == 28);
    long blocks = size / 28; float *y = malloc(blocks * 128 * sizeof(float));
    dequantize_row_ptq1_0((const block_ptq1_0 *) data, y, blocks * 128); fwrite(y, sizeof(float), blocks * 128, out);
  } else if (!strcmp(argv[1], "quant-pq2")) {
    long blocks = size / sizeof(float) / 128; block_pq2_0 *y = malloc(blocks * 34);
    quantize_row_pq2_0_ref((const float *) data, y, blocks * 128); fwrite(y, 34, blocks, out);
  } else if (!strcmp(argv[1], "quant-ptq")) {
    long blocks = size / sizeof(float) / 128; block_ptq1_0 *y = malloc(blocks * 28);
    quantize_row_ptq1_0_ref((const float *) data, y, blocks * 128); fwrite(y, 28, blocks, out);
  } else return 5;
  fclose(out);
  return 0;
}
"""


def cut(text, start):
    """The function that begins at start, to its closing brace"""
    i = text.index(start)
    j = text.index("{", i)
    depth, k = 0, j
    while True:
        depth += text[k] == "{"
        depth -= text[k] == "}"
        if depth == 0:
            return text[i:k + 1]
        k += 1


def main():
    work = Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parent / ".tmp" / "fork-readers").resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = urllib.request.urlopen(SOURCE, timeout=120).read().decode()
    (work / "functions.c").write_text("\n".join(cut(source, name) for name in FUNCTIONS))
    (work / "harness.c").write_text(HARNESS)
    subprocess.run(["gcc", "-O2", "-o", str(work / "harness"), str(work / "harness.c"), "-lm"], check=True, cwd=work)
    rng = np.random.default_rng(20261002)

    def fork(mode, data):
        (work / "in.bin").write_bytes(np.ascontiguousarray(data).tobytes())
        subprocess.run([str(work / "harness"), mode, str(work / "in.bin"), str(work / "out.bin")], check=True)
        return (work / "out.bin").read_bytes()

    def scales(count):
        special = np.array([0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x2800, 0x5640], dtype=np.uint16)
        d = rng.integers(0, 0x7C00, count).astype(np.uint16) | (rng.integers(0, 2, count).astype(np.uint16) << 15)
        d[: len(special)] = special
        return d

    failed = False

    def check(ok, what):
        nonlocal failed
        print(f"fork-readers: {'ok' if ok else 'FAILED'}: {what}")
        failed = failed or not ok

    for name, size, mode, reader, front in (("PQ2_0", 34, "dequant-pq2", llama2_convert.pq2_0, True),
                                            ("PTQ1_0", 28, "dequant-ptq", llama2_convert.ptq1_0, False)):
        blocks = 40000
        raw = rng.integers(0, 256, size=(blocks, size), dtype=np.uint8)
        d = scales(blocks).view(np.uint8).reshape(-1, 2)
        raw[:, :2] = d if front else raw[:, :2]
        raw[:, size - 2:] = raw[:, size - 2:] if front else d
        mine = reader(raw.tobytes())
        theirs = np.frombuffer(fork(mode, raw), dtype=np.float32)
        check(np.array_equal(mine.view(np.uint32), theirs.view(np.uint32)),
              f"{name}: the reader and the fork's dequantize give the same bits on {blocks:,} blocks of random bytes ({blocks * 128:,} values)")
    for name, quant, size, reader in (("PQ2_0", "quant-pq2", 34, llama2_convert.pq2_0), ("PTQ1_0", "quant-ptq", 28, llama2_convert.ptq1_0)):
        groups = 24000
        d = np.abs(scales(groups).view(np.float16).astype(np.float32))
        d = np.where(np.isfinite(d), d, 1.0).astype(np.float32)
        signs = rng.integers(-1, 2, size=(groups, 128)).astype(np.float32)
        signs[::2, 5] = 1  # every group has a weight of its full size (or is all zeros)
        signs[1::5] = 0
        values = (signs * d[:, None]).astype(np.float32)
        blob = fork(quant, values)
        got = reader(blob).reshape(-1, 128)
        keep = np.abs(values).max(axis=1) > 0
        check(np.array_equal(got[keep].view(np.uint32), values[keep].view(np.uint32)) and not got[~keep].any(),
              f"{name}: the bytes of the fork's own quantizer for {int(keep.sum()):,} ternary groups read back as their values")
        packed, scale = ternary(got)
        check(np.array_equal(scale[keep], np.abs(got[keep]).max(axis=1)) and packed.shape == (groups, 32),
              f"{name}: ternary() of what was read takes back the groups' scales")
    sys.exit(1 if failed else 0)


main()
