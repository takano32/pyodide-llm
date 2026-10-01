# t246_scan.py (the review of T246, a probe for CI, not for main): why gguf_check.py found the 4B's PQ2_0 GGUF 8.53e-5
# off its original at the worst tensor and the 8B's exactly 0 (T235's 1.7B: 8.7e-5, "a block here and there has two
# magnitudes in the safetensors, 0.5% apart, and the larger one for all its values in the GGUF"). Every weight of the
# -unpacked float16 safetensors is a ternary value times its block's scale; this counts, per tensor, the blocks of 128
# (along the last axis, as PQ2_0 takes them) that hold more than one magnitude, and how far apart the magnitudes are.
#
#   python3 tests/t246_scan.py <label> <directory of the .safetensors files>
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

label, directory = sys.argv[1], Path(sys.argv[2])
started = time.time()
total_blocks = two = three = 0
worst = 0.0
per_tensor = []
for path in sorted(directory.glob("*.safetensors")):
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    base = 8 + length
    for name, info in header.items():
        if name == "__metadata__" or info["dtype"] != "F16" or len(info["shape"]) != 2:
            continue
        rows, cols = info["shape"]
        assert cols % 128 == 0, (name, cols)
        begin, end = info["data_offsets"]
        data = np.memmap(path, dtype=np.uint16, mode="r", offset=base + begin, shape=(rows, cols))
        blocks = differing = more = 0
        spread = 0.0
        step = max(1, (32 << 20) // cols)
        for start in range(0, rows, step):
            a = np.asarray(data[start:start + step]).reshape(-1, 128) & 0x7FFF
            mx = a.max(axis=1)
            mn = np.where(a != 0, a, np.uint16(0xFFFF)).min(axis=1).astype(np.uint16)
            diff = (mn != 0xFFFF) & (mn != mx)
            blocks += a.shape[0]
            differing += int(diff.sum())
            if diff.any():
                hi = mx[diff].view(np.float16).astype(np.float64)
                lo = mn[diff].view(np.float16).astype(np.float64)
                spread = max(spread, float(((hi - lo) / hi).max()))
                # more than two magnitudes: a value that is neither the largest nor the smallest of its block
                for block in np.flatnonzero(diff)[:50]:
                    kinds = np.unique(a[block][a[block] != 0])
                    more += int(len(kinds) > 2)
        total_blocks += blocks
        two += differing
        three += more
        worst = max(worst, spread)
        if differing:
            per_tensor.append((name, differing, blocks, spread))
        print(f"T246SCAN {label} {name} {rows}x{cols}: {blocks} blocks, {differing} with more than one magnitude, largest spread "
              f"{spread:.3e}", flush=True)
print(f"T246SCAN {label} TOTAL: {total_blocks} blocks, {two} hold more than one magnitude ({two / total_blocks:.2e}), "
      f"largest relative spread {worst:.3e}; in {len(per_tensor)} tensors ({time.time() - started:.0f} s)")
for name, differing, blocks, spread in sorted(per_tensor, key=lambda t: -t[3])[:12]:
    print(f"T246SCAN {label} widest: {name}: {differing} of {blocks} blocks, spread {spread:.3e}")
