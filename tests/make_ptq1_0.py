# make_ptq1_0.py
# T230: a PQ2_0 GGUF written again with its ternary tensors as PTQ1_0 (Prism ML's other ternary type: five values a
# byte in base 3, 28 bytes a block of 128 where PQ2_0 takes 34). The only model published as PTQ1_0 is Ternary Bonsai 2
# 27B (5.9 GB), so this makes one of a model that is small enough to convert whole in CI (Ternary Bonsai 1.7B): the
# page's conversion of it must write the very checkpoint it writes of the PQ2_0 file, and its time is PTQ1_0's.
# The metadata is copied as it is; the tensors' types and offsets are written anew. The blocks are packed as
# tests/test_ternary.py's ptq1_0_blocks packs them (the fork's order of the values: docs/notes/t228-bonsai-2-2026-10-01.md),
# here on whole arrays.
#
#   python3 tests/make_ptq1_0.py <in: a PQ2_0 .gguf> <out: the same model, PTQ1_0>
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_check import BYTES, PQ2_0, PTQ1_0, Reader, read_gguf  # noqa: E402

BLOCKS = 1 << 16  # blocks a piece: 2.2 MB of PQ2_0


def base3(digits):
    """(blocks, 5, bytes) digits, the first of a byte's the most significant -> the bytes, ceil(256 v / 243)."""
    number = sum(digits[:, place].astype(np.int64) * 3 ** (4 - place) for place in range(5))
    return ((number * 256 + 242) // 243).astype(np.uint8)


def ptq1_0(raw):
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    digits = ((blocks[:, 2:, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3).reshape(-1, 128)
    if digits.max(initial=0) > 2:
        raise ValueError("this PQ2_0 tensor has the fourth code (+2 d): it is not ternary, and PTQ1_0 cannot hold it")
    out = np.empty((len(blocks), 28), dtype=np.uint8)
    out[:, :16] = base3(digits[:, :80].reshape(-1, 5, 16))
    out[:, 16:24] = base3(digits[:, 80:120].reshape(-1, 5, 8))
    last = np.zeros((len(blocks), 5, 2), dtype=np.uint8)
    last[:, :4] = digits[:, 120:].reshape(-1, 4, 2)
    out[:, 24:26] = base3(last)
    out[:, 26:] = blocks[:, :2]
    return out.tobytes()


def main():
    source, target = sys.argv[1], sys.argv[2]
    _, metadata, infos, data, base = read_gguf(source)
    alignment = metadata.get("general.alignment", 32)
    # where the tensors' infos begin: after the metadata, read again to find its end
    reader = Reader(data)
    reader.at = 8
    tensors, entries = reader.take("<Q"), reader.take("<Q")
    for _ in range(entries):
        reader.string()
        reader.value(reader.take("<I"))
    head = bytearray(bytes(data[:reader.at]))
    pad = lambda size: -size % alignment
    sizes, offset = {}, 0
    for name, info in infos.items():
        count = int(np.prod(info["shape"]))
        ternary = info["type"] == PQ2_0
        sizes[name] = (base + info["offset"], int(count * BYTES[info["type"]]), ternary)
        encoded = name.encode()
        head += struct.pack("<Q", len(encoded)) + encoded + struct.pack("<I", len(info["shape"]))
        head += struct.pack(f"<{len(info['shape'])}Q", *reversed(info["shape"]))
        head += struct.pack("<IQ", PTQ1_0 if ternary else info["type"], offset)
        written = count // 128 * 28 if ternary else sizes[name][1]
        offset += written + pad(written)
    head += b"\0" * pad(len(head))
    with open(target, "wb") as out:
        out.write(head)
        for name, (start, length, ternary) in sizes.items():
            written = 0
            for at in range(0, length, BLOCKS * 34 if ternary else 1 << 24):
                piece = bytes(data[start + at:start + min(at + (BLOCKS * 34 if ternary else 1 << 24), length)])
                piece = ptq1_0(piece) if ternary else piece
                out.write(piece)
                written += len(piece)
            out.write(b"\0" * pad(written))
    print(f"{target}: {Path(target).stat().st_size} bytes, {sum(1 for _, _, ternary in sizes.values() if ternary)} tensors as PTQ1_0")


if __name__ == "__main__":
    main()
