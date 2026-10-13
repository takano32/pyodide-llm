# tests/saluki_blocks.py (T363.1): what tests/saluki-kernel-bench.mjs asks of the converter's own readers (T362), so
# that the forms it times are held to them and not to a second reading written in JavaScript.
#
#   python tests/saluki_blocks.py <folder>
#
# writes <folder>/grids.bin (the grids of convert/grids.py the forms look up: IQ2XXS's 256 entries of 8 values, IQ3S's
# 512 of 4, IQ1S's 2048 of 8, one int8 a value) and, for every <folder>/<TYPE>.blocks (whole blocks of the type, as a
# GGUF holds them), <folder>/<TYPE>.f32: their weights as convert/superblocks.py reads them, float32.
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tree import python_folder  # noqa: E402

sys.path.insert(0, python_folder(HERE.parent))
from convert.grids import grid  # noqa: E402
from convert.superblocks import SUPERBLOCKS  # noqa: E402

folder = Path(sys.argv[1])
(folder / "grids.bin").write_bytes(b"".join(grid(name).astype(np.int8).tobytes() for name in ("IQ2XXS", "IQ3S", "IQ1S")))
for blocks in sorted(folder.glob("*.blocks")):
    _, size, read = SUPERBLOCKS[blocks.stem]
    raw = blocks.read_bytes()
    assert len(raw) % size == 0, f"{blocks.name} is not whole blocks of {size} bytes"
    blocks.with_suffix(".f32").write_bytes(read(raw).astype("<f4").tobytes())
