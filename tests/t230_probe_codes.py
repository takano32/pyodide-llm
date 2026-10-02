"""T230's review (a probe, for CI): does a PQ2_0 GGUF of the list use the code 3 (+2 d) anywhere, and how many blocks are not
whole of the ternary values (-d, 0, d)? A reader that turns the file into ternary refuses a group whose values are not 0 or
plus or minus its largest, so one code 3 beside a code 0 or 2 would be a refusal; a group of codes 1 and 3 alone passes as a
group whose scale is 2 d. Counts every PQ2_0 tensor of the file.
    python tests/t230_probe_codes.py <the .gguf file> ..."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "public"))
from llama2_convert import gguf_read  # noqa: E402

THREES = np.array([sum(1 for k in range(4) if (b >> 2 * k) & 3 == 3) for b in range(256)], dtype=np.int64)
for name in sys.argv[1:]:
    data = np.memmap(name, dtype=np.uint8, mode="r")
    _, tensors, base = gguf_read(data[: 32 * 2 ** 20])
    total = blocks = threes = mixed = pure_zero = 0
    for tensor, info in tensors.items():
        if info["type"] != 142:
            continue
        count = int(np.prod(info["shape"], dtype=np.int64))
        nblocks = count // 128
        raw = np.asarray(data[base + info["offset"]: base + info["offset"] + nblocks * 34]).reshape(nblocks, 34)
        codes = raw[:, 2:]
        have3 = (THREES[codes].sum(axis=1) > 0)
        # a block with a code 3 and also a code 0 or 2 (the other codes of a ternary group): not representable
        counts = np.stack([sum(((codes >> (2 * k)) & 3 == c).sum(axis=1) for k in range(4)) for c in range(4)], axis=1)
        bad = have3 & ((counts[:, 0] > 0) | (counts[:, 2] > 0))
        total += count
        blocks += nblocks
        threes += int(counts[:, 3].sum())
        mixed += int(bad.sum())
        pure_zero += int((counts[:, 1] == 128).sum())
    print(f"codes: {Path(name).name}: {blocks:,} blocks of PQ2_0 ({total:,} weights), code 3 in {threes:,} weights, "
          f"{mixed:,} blocks with a 3 beside a 0 or a 2 (refused as not ternary), {pure_zero:,} blocks of zeros")
