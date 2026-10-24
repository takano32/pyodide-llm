# saluki_rows.py
# T.362.2: the K- and i-quantized tensors of Underdog Saluki 27B's GGUF as the converter reads them, against the model the
# file was made from (Qwen/Qwen3.8-27B, 55.6 GB of bfloat16), without the 7.9 GB: the GGUF's head and a few rows of
# tensors by Range requests (34 MB all told in 213 requests, about four minutes; kept in the work directory, so that a
# second run asks the network nothing).
#
# What the readers are, llama.cpp's own C says to the bit (tests/fork_readers.py). This says the rest of the reading:
# that a tensor is where the converter's header puts it, of the type it says, under the Hugging Face name it gives it,
# its rows the rows of that name, and the value heads of a linear-attention layer where the converter puts them back
# (llama.cpp writes them in another order: along the rows of in_proj_qkv and in_proj_z, along the columns of out_proj).
# The file is not the original rounded (GSQ-RCO moves the weights: a row is 15% to 60% of its size away from the
# original's), so there is no line of 1e-3 to hold it to as tests/gguf_check.py holds a Q8_0; a row read right is
# highly correlated with the original's, the more the more bits its type has, and a row read any other way is not.
#
# For every one of the ten types and every role a matrix of that type has in the file (the first layer that has one):
# ROWS rows, and of each
#   - the correlation with the original's row, which must be above the type's line (LINES: what 1.56 to 4.5 bits
#     leave, with room below what was measured);
#   - its size against the original's (the ratio of the root mean squares), which must be near 1: a scale read as
#     twice or half what it is leaves the correlation as it was;
#   - the same values moved by 8 places, which must not correlate at all (the comparison can tell);
#   - where the value heads are put back: the rows or columns left as the file has them, which must not correlate.
# And the head: the tensors, the types and how many of each (COUNTS), which the converter's own reader of a GGUF takes
# with the original's config.json; and (T.363.3) what the conversion that keeps the blocks makes of that head, without
# the weights: the kinds of the rows (as many layers of each type to a row as tests/saluki_form.py counted), that the
# output matrices of the linear-attention layers stay tiled, which tensors' rows move, and the checkpoint's size.
#
#   python tests/saluki_rows.py [--work <directory = .tmp/saluki-rows>] [--rows 16] [--wrong <a reading>]
#
# --wrong: one of WRONG, a reading that is not the file's; the tool must then fail (exit 1), and says which types it
# failed for: what shows that the lines catch a reader's mistake. Exit 1 where a reading fails. For a development
# machine and for CI's extra= (NumPy and the network, nothing else).
import argparse
import collections
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
import llama2_convert  # noqa: E402
import saluki_form  # noqa: E402
from convert import superblocks  # noqa: E402
from fetching import download, ranged  # noqa: E402

ORIGINAL = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/"
SALUKI = ("https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0/resolve/1336c0b5d74dfe6ad7f793577092f5effd9f97c6/"
          "Underdog-Saluki-27B-1.0-IQ2-mix.gguf")
HEAD_BYTES = 12 * 1024 * 1024
PREFIX = "model.language_model."
# the file's tensors by type (docs/notes/t361-saluki-2026-10-13.md, section 2)
COUNTS = {"IQ2_XXS": 119, "IQ3_S": 44, "Q2_K": 46, "IQ2_S": 46, "IQ2_XS": 39, "IQ1_M": 31, "IQ3_XXS": 30, "IQ1_S": 36,
          "IQ4_XS": 7, "Q4_K": 4, "Q8_0": 96, "F32": 353}
# the least mean correlation of a tensor's sampled rows with the original's, by the type. Measured (2026-10-13, 16
# rows of each of the 73 tensors the tool reads; the least of a type's tensors): IQ1_S 0.829, IQ1_M 0.856, IQ2_XXS
# 0.916, Q2_K 0.933, IQ2_XS 0.943, IQ2_S 0.955, IQ3_XXS 0.967, IQ3_S 0.980, Q4_K 0.989, IQ4_XS 0.990
# (the file and the rows are fixed, so the lines stand close under them: a reading of IQ3_S's scales with the two halves
# of a byte the other way round left one tensor at 0.951)
LINES = {"IQ1_S": 0.80, "IQ1_M": 0.83, "IQ2_XXS": 0.90, "Q2_K": 0.915, "IQ2_XS": 0.93, "IQ2_S": 0.94, "IQ3_XXS": 0.955,
         "IQ3_S": 0.97, "Q4_K": 0.98, "IQ4_XS": 0.98}
# the size of a tensor's rows over the original's (measured: 0.87 to 1.24, the widest where the bits are fewest: the
# file's weights are not the nearest ones to the original's; twice or half is past either end), how much the values moved by 8 places may correlate (measured: under 0.01), and
# the value heads as the file has them (measured: under 0.05; two of the 48 heads are where Hugging Face has them, the
# first and the last)
SIZE, UNRELATED, AS_STORED = (0.75, 1.5), 0.03, 0.1


class Fetched:
    """Ranges of files, kept in a directory by what was asked: a second run reads them there."""

    def __init__(self, work):
        self.work, self.asked, self.came = work, 0, 0
        work.mkdir(parents=True, exist_ok=True)

    def __call__(self, url, start, length):
        kept = self.work / f"{hashlib.sha256(f'{url} {start} {length}'.encode()).hexdigest()[:24]}.bin"
        if not kept.exists() or kept.stat().st_size != length:
            try:
                data = ranged(url, start, length, tries=5, wait=2)
            except OSError as error:
                raise SystemExit(f"saluki-rows: {url} from {start}: {error}")
            kept.with_suffix(".part").write_bytes(data)
            kept.with_suffix(".part").replace(kept)
            self.asked, self.came = self.asked + 1, self.came + length
        return kept.read_bytes()

    def whole(self, url):
        """A small file, all of it."""
        try:
            return download(url, self.work / url.rpartition("/")[2]).read_bytes()
        except OSError as error:
            raise SystemExit(f"saluki-rows: {url}: {error}")


class Original:
    """The original model's tensors by name: the index, the header of the shard that holds one, and rows of it."""

    def __init__(self, fetch):
        self.fetch, self.headers = fetch, {}
        self.index = json.loads(fetch.whole(ORIGINAL + "model.safetensors.index.json"))["weight_map"]

    def rows(self, name, first, count):
        """count rows of the tensor called name from row first, float64, and the tensor's shape."""
        shard = self.index[name]
        if shard not in self.headers:
            (size,) = struct.unpack("<Q", self.fetch(ORIGINAL + shard, 0, 8))
            self.headers[shard] = (json.loads(self.fetch(ORIGINAL + shard, 8, size)), 8 + size)
        header, base = self.headers[shard]
        info = header[name]
        width = info["shape"][-1]
        if info["dtype"] != "BF16":
            raise SystemExit(f"saluki-rows: the original's {name} is {info['dtype']}, not BF16")
        raw = self.fetch(ORIGINAL + shard, base + info["data_offsets"][0] + first * width * 2, count * width * 2)
        return llama2_convert.bfloat16(raw).reshape(count, width).astype(np.float64), tuple(info["shape"])


class Sized:
    """The sink of a conversion nobody feeds: what it was opened with."""

    def open(self, size, header, dtype, form):
        self.size, self.header, self.dtype, self.form = size, list(header), dtype, form

    def write(self, offset, raw):
        pass


def correlation(a, b):
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def places(entry):
    """Where the converter puts the rows of a tensor: for each row of the file, the row of Hugging Face's tensor it
    is (the converter's own untiled() of the rows' numbers), or None where the rows are where they are."""
    tiled = entry.get("tiled")
    if not tiled or tiled[4] != 0:
        return None
    rows = entry["shape"][0]
    back = np.empty(rows, dtype=np.int64)
    back[llama2_convert.untiled(np.arange(rows), *tiled[:4])] = np.arange(rows)
    return back


def samples(entry, count):
    """Which rows of a tensor to read: [(what, the first row in the file)]. Where the rows of the value heads are
    moved, rows of the second head in the file, which is not Hugging Face's second (and of what stands before the
    heads, q and k of in_proj_qkv, rows that are not moved)."""
    rows, tiled = entry["shape"][0], entry.get("tiled")
    if tiled and tiled[4] == 0:
        first, size = tiled[0], tiled[3]
        return ([("the rows before the value heads", 100)] if first else []) + [("rows of the second value head", first + size)]
    return [("rows", 1000 if rows > 1000 + count else 0)]


def role(name):
    """The role of a matrix by its Hugging Face name: what is left of it without its layer."""
    parts = name.split(".")
    return ".".join(parts[3:-1]) if parts[:2] == ["model", "layers"] else ".".join(parts[:-1])


# ---- readings that are not the file's (--wrong): each must fail
def rolled_grids():
    for read in (superblocks.iq2_xxs, superblocks.iq2_xs, superblocks.iq2_s, superblocks.iq3_xxs, superblocks.iq3_s,
                 superblocks.iq1_s, superblocks.iq1_m):
        read.__defaults__ = (np.roll(read.__defaults__[0], 1, axis=0),)


def other_signs():
    signs = superblocks.signs
    superblocks.signs = lambda held: -signs(held)


def no_eighth():
    superblocks.SEVEN = np.arange(128, dtype=np.uint8)


def high_first():
    nibbles = superblocks.nibbles
    superblocks.nibbles = lambda held: nibbles(held).reshape(len(held), -1, 2)[:, :, ::-1].reshape(len(held), -1)


def plain_scales():
    superblocks.odd = lambda scale: (scale.astype(np.int32) + 1).astype(np.float32)


def other_values():
    superblocks.IQ4NL = np.roll(superblocks.IQ4NL, 1)


def minimum_added():
    """The two K types with their groups' minimum added, not subtracted (what is read without dmin, twice, less what
    is read with it)."""
    for name, at in (("Q2_K", 82), ("Q4_K", 2)):
        source = llama2_convert.SOURCES[name]

        def read(raw, read=source.read, size=source.size, at=at):
            without = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size).copy()
            without[:, at:at + 2] = 0
            return np.float32(2) * read(without.tobytes()) - read(raw)
        llama2_convert.SOURCES[name] = source._replace(read=read)


def wrapped(change):
    """Every one of the ten read by change(the reader's values as (blocks, 256))."""
    def patch():
        for name, source in list(llama2_convert.SOURCES.items()):
            if source.values == 256:
                llama2_convert.SOURCES[name] = source._replace(
                    read=lambda raw, read=source.read: change(read(raw).reshape(-1, 256)).reshape(-1))
    return patch


# the reading -> (what it does, the types it must fail for)
I_SIGNED = ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S")
EVERY = tuple(LINES)
WRONG = {
    "grids": (rolled_grids, I_SIGNED + ("IQ1_S", "IQ1_M"), "every index of a codebook read as the entry before it"),
    "signs": (other_signs, I_SIGNED, "a set bit of the signs read as a plus"),
    "eighth": (no_eighth, ("IQ2_XXS", "IQ2_XS", "IQ3_XXS"), "the eighth sign of a group, which is not stored, read as a plus"),
    "nibbles": (high_first, ("IQ2_XS", "IQ2_S", "IQ3_S", "IQ1_M", "IQ4_XS"), "the high four bits of a byte read before its low four"),
    "scales": (plain_scales, ("IQ3_S", "IQ1_S", "IQ1_M"), "a group's scale s read as s + 1, not 2 s + 1"),
    "values": (other_values, ("IQ4_XS",), "IQ4_XS's four bits read as the value before theirs"),
    "minimum": (minimum_added, ("Q2_K", "Q4_K"), "the minimum of a group of a K type added, not subtracted"),
    "places": (wrapped(lambda v: np.roll(v, 1, axis=1)), EVERY, "every weight of a block read one place on"),
    "halves": (wrapped(lambda v: np.roll(v, 128, axis=1)), EVERY, "the two halves of a block the other way round"),
    "minus": (wrapped(lambda v: -v), EVERY, "every weight with the other sign"),
    "twice": (wrapped(lambda v: v * np.float32(2)), EVERY, "every weight twice what it is"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=str(HERE.parent / ".tmp" / "saluki-rows"))
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--wrong", choices=sorted(WRONG))
    args = parser.parse_args()
    fetch = Fetched(Path(args.work).resolve())
    original = Original(fetch)
    failed = collections.defaultdict(list)

    def check(ok, what, kind="the head"):
        print(f"saluki-rows: {'ok' if ok else 'FAILED'}: {what}", flush=True)
        if not ok:
            failed[kind].append(what)

    head = fetch(SALUKI, 0, HEAD_BYTES)
    config = fetch.whole(ORIGINAL + "config.json").decode()
    _, tensors, _ = llama2_convert.gguf_read(head)
    found = collections.Counter(llama2_convert.GGUF_TENSORS.get(info["type"], f"ggml type {info['type']}") for info in tensors.values())
    check(dict(found) == COUNTS, f"the head has {len(tensors)} tensors of {len(found)} types: {dict(found.most_common())}")
    # the converter's own reader of a GGUF with the original's config.json (gguf_agrees() among it): the header, by
    # Hugging Face's names, with what llama.cpp stores otherwise marked
    header, base = llama2_convert.gguf_weights(head, config)
    whole = json.loads(header)
    header = {name: entry for name, entry in whole.items() if name != "__metadata__"}
    check(len(header) == len(tensors) and base == 10994816, f"the converter's reader takes all {len(header)} of them; the data begin at {base:,}")
    # T.363.3: the conversion that keeps the blocks, of this head
    sink = Sized()
    kept = llama2_convert.Kept(whole, base, json.loads(config), "int8", 4096, sink=sink)
    kinds = kept.form["kinds"] or {}
    counted = {row: dict(collections.Counter([types] if isinstance(types, str) else types)) for row, types in kinds.items()}
    check(counted == saluki_form.COUNTS, f"kept, {len(kinds)} rows are in the file's blocks, as many layers of each type as counted: "
                                         f"{', '.join(kinds)} ({len(json.dumps(kinds))} characters of JSON)")
    moved = collections.Counter(name.rsplit(".", 2)[-2] for name, orders in kept.stay.items() if orders is not None)
    check(len(kept.stay) == sum(saluki_form.TENSORS.values()) and dict(moved) == {"in_proj_qkv": 48, "in_proj_z": 48, "q_proj": 16, "k_proj": 16},
          f"all {len(kept.stay)} tensors in blocks stay bytes; the rows of {dict(moved)} are moved")
    check(kept.form["tiled"] is True and sink.size == saluki_form.BYTES + 33808 == 7956712492 and sink.dtype == "int8"
          and sink.header == saluki_form.HEADER and sink.form["linear"] == saluki_form.LINEAR,
          f"the checkpoint is {sink.size:,} bytes as {sink.dtype} with the header {sink.header}; the output matrices of "
          f"the linear-attention layers stay tiled: {kept.form['tiled']}")
    (fetch.work / "kept.json").write_text(json.dumps({"header": sink.header, "bytes": sink.size, "form": kept.form}, indent=1) + "\n")
    if args.wrong:
        WRONG[args.wrong][0]()
        print(f"saluki-rows: --wrong {args.wrong}: {WRONG[args.wrong][2]}")

    # the first matrix of every (type, role) there is, in the order of the layers
    chosen = {}
    for name, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        if entry["dtype"] in LINES:
            chosen.setdefault((entry["dtype"], role(name)), name)
    check({kind for kind, _ in chosen} == set(LINES), f"every one of the ten types has a matrix: {len(chosen)} pairs of a type and a role")
    lines, least = [], {}
    for (kind, _), name in sorted(chosen.items(), key=lambda item: (list(LINES).index(item[0][0]), item[1])):
        entry, source = header[name], llama2_convert.SOURCES[kind]
        rows, width = entry["shape"]
        row = source.bytes(width)
        if entry["data_offsets"][1] - entry["data_offsets"][0] != rows * row:
            raise SystemExit(f"saluki-rows: {name} is not {rows} rows of {row} bytes")
        moved, tiled = places(entry), entry.get("tiled")
        there = name if name in ("lm_head.weight",) else PREFIX + name[len("model."):]
        for what, first in samples(entry, args.rows):
            count = min(args.rows, rows - first)
            read = source.read(fetch(SALUKI, base + entry["data_offsets"][0] + first * row, count * row))
            stored = read.reshape(count, width).astype(np.float64)
            ours = stored
            if tiled and tiled[4] == 1:  # the columns of a linear-attention layer's output matrix: as the converter puts them back
                ours = llama2_convert.untiled(stored, *tiled[:4], axis=1)
            at = first if moved is None else int(moved[first])
            if moved is not None and not np.array_equal(moved[first:first + count], np.arange(at, at + count)):
                raise SystemExit(f"saluki-rows: {name}: the sampled rows are not one run of the original's")
            theirs, shape = original.rows(there, at, count)
            if list(shape) != list(entry["shape"]):
                raise SystemExit(f"saluki-rows: {name} is {entry['shape']}, the original's {shape}")
            each = [correlation(a, b) for a, b in zip(ours, theirs)]
            size = float(np.sqrt((ours * ours).mean() / (theirs * theirs).mean()))
            unrelated = correlation(np.roll(ours, 8, axis=1), theirs)
            # the other reading of the value heads: the file's rows or columns as they are (the original's rows of the
            # same numbers, for the rows)
            as_stored = None
            if tiled and (tiled[4] == 1 or at != first):
                as_stored = correlation(stored, theirs if tiled[4] == 1 else original.rows(there, first, count)[0])
            mean = float(np.mean(each))
            ok = (bool(np.isfinite(stored).all()) and mean > LINES[kind] and SIZE[0] < size < SIZE[1] and abs(unrelated) < UNRELATED
                  and (as_stored is None or abs(as_stored) < AS_STORED))
            least[kind] = min(least.get(kind, 1.0), mean)
            lines.append(dict(tensor=name, type=kind, what=what, rows=[first, first + count], original=[at, at + count],
                              correlation=round(mean, 4), least=round(min(each), 4), size=round(size, 4),
                              moved_by_8=round(unrelated, 4), as_stored=None if as_stored is None else round(as_stored, 4)))
            check(ok, f"{kind:8} {name[len('model.'):] if name.startswith('model.') else name}, {what} {first} to {first + count - 1}"
                      f"{'' if at == first else f' (the original{chr(39)}s {at} on)'}: correlation {mean:.4f} (least {min(each):.4f}, line "
                      f"{LINES[kind]}), size {size:.3f}, moved by 8: {unrelated:+.4f}"
                      f"{'' if as_stored is None else f', the value heads as the file has them: {as_stored:+.4f}'}", kind)
    print("saluki-rows: the least mean correlation of a tensor, by type: " + ", ".join(f"{kind} {least[kind]:.3f}" for kind in LINES if kind in least))
    print(f"saluki-rows: {fetch.asked} ranges were asked of the network, {fetch.came / 1e6:.1f} MB (the rest was in {fetch.work})")
    (fetch.work / f"rows{'-' + args.wrong if args.wrong else ''}.json").write_text(json.dumps(lines, indent=1) + "\n")
    if args.wrong:
        missed = [kind for kind in WRONG[args.wrong][1] if kind not in failed]
        print(f"saluki-rows: --wrong {args.wrong} failed for {sorted(kind for kind in failed if kind in LINES)}"
              + (f"; NOT for {missed}, which it reads otherwise: the lines do not catch it" if missed else ""))
    if failed:
        print(f"saluki-rows: FAILED: {sum(map(len, failed.values()))} of the readings")
    sys.exit(1 if failed else 0)


main()
