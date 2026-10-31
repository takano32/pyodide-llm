# tests/superblock_cases.py (T.362): blocks of llama.cpp's K- and i-quantized types that are the same bytes on every
# machine, and one number for what a reader reads of them. tests/fixtures/superblocks.json has the numbers of a 64-bit
# NumPy, which pytest holds the readers to (tests/test_superblocks.py); tests/smoke.ts asks the same of the readers
# inside Pyodide, whose NumPy has integers of 32 bits. Neither pytest nor the converter is imported here.
import hashlib

import numpy as np

BLOCKS = 2000


def blocks(name, size, count=BLOCKS):
    """count blocks of size bytes for the type called name: SHAKE-256's bytes of the name (any bytes are a block)."""
    return hashlib.shake_256(name.encode()).digest(size * count)


def digest(values):
    """The SHA-256 of the float32 values a reader gave. A block whose float16 scale is an infinity or a NaN reads as
    infinities and NaNs, and a NaN's sign and payload are the machine's: every NaN counts as 0."""
    assert values.dtype == np.float32
    return hashlib.sha256(np.nan_to_num(values, nan=0.0, posinf=np.inf, neginf=-np.inf).astype("<f4").tobytes()).hexdigest()
