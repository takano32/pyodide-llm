# tests/reference_saluki.py (T361): what tests/reference_saluki.sh needs of Python, for the reference of Underdog Saluki
# 27B (stock llama.cpp's logits; docs/notes/t361-saluki-2026-10-13.md). NumPy and the standard library alone.
#
#   python tests/reference_saluki.py head <directory> [<the GGUF, where it was fetched>]
#       the head of the file (its first 12 MiB by a Range request, kept in the directory; or of the file itself): that
#       it is the file the notes describe. 851 tensors of the twelve types in the counts of TYPES, every tensor whole
#       blocks of its type and as long as the table's offsets say, the last one ending where the file ends, a qwen35
#       of the 27B's numbers, the chat template Qwen/Qwen3.8-27B's. Prints the table of the types.
#   python tests/reference_saluki.py fetch <directory>
#       the whole file (7.9 GB: on a CI runner, never on the development machine), by tests/fetching.py's download(),
#       held to its length and to the sha256 of the revision's file.
#   python tests/reference_saluki.py agree <directory> [<where to write summary.json>]
#       what the driver (tests/reference_27b_fork.cpp, built against stock llama.cpp) wrote for each prompt: that the
#       files are whole and finite, that the tokens written are the largest logits of their rows, and how far
#       llama.cpp's two paths are from each other (a prompt one token at a time and as one batch): the floor of any
#       line that holds another forward pass to these logits.
#   python tests/reference_saluki.py made-up <directory>
#       files of the driver's form with made-up logits, for the dry stage: one prompt that agrees, and one whose
#       written token is not its row's largest logit (agree must refuse it).
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetching import download, ranged, sized  # noqa: E402

REPOSITORY = "ConwayResearch/Underdog-Saluki-27B-1.0"
REVISION = "1336c0b5d74dfe6ad7f793577092f5effd9f97c6"
FILE = "Underdog-Saluki-27B-1.0-IQ2-mix.gguf"
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILE}"
BYTES = 7898369152
SHA256 = "4a673518b11b1c4445f9b9a9d3c40356f5ba39f6a6260dcb25f3fbfd475d9efb"
HEAD_BYTES = 12 << 20
# the sha256 of tokenizer.chat_template: that of chat_template.jinja of Qwen/Qwen3.8-27B at
# 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 (8,952 characters)
TEMPLATE = "c3cf9e34abf4f9e3"

# ggml's type: (its name, the values of a block, the bytes of a block, how many of the file's tensors are of it)
TYPES = {0: ("F32", 1, 4, 353), 8: ("Q8_0", 32, 34, 96), 10: ("Q2_K", 256, 84, 46), 12: ("Q4_K", 256, 144, 4),
         16: ("IQ2_XXS", 256, 66, 119), 17: ("IQ2_XS", 256, 74, 39), 18: ("IQ3_XXS", 256, 98, 30),
         19: ("IQ1_S", 256, 50, 36), 21: ("IQ3_S", 256, 110, 44), 22: ("IQ2_S", 256, 82, 46),
         23: ("IQ4_XS", 256, 136, 7), 29: ("IQ1_M", 256, 56, 31)}
NUMBERS = {"general.architecture": "qwen35", "qwen35.block_count": 64, "qwen35.embedding_length": 5120,
           "qwen35.feed_forward_length": 17408, "qwen35.attention.head_count": 24, "qwen35.attention.head_count_kv": 4,
           "qwen35.full_attention_interval": 4, "qwen35.ssm.time_step_rank": 48, "qwen35.ssm.group_count": 16,
           "qwen35.ssm.state_size": 128, "qwen35.rope.dimension_count": 64, "tokenizer.ggml.bos_token_id": 248044,
           "tokenizer.ggml.eos_token_id": 248046, "tokenizer.ggml.add_bos_token": False}


def gguf_head(data):
    """(metadata, [(name, dims as the file has them, type, offset)], base) of a GGUF's first bytes. The converter's own
    gguf_read() does the same; this tool reads nothing of src/ (it runs where only_extra installed NumPy alone)."""
    at = 0

    def take(fmt):
        nonlocal at
        (value,) = struct.unpack_from("<" + fmt, data, at)
        at += struct.calcsize("<" + fmt)
        return value

    def string():
        nonlocal at
        size = take("Q")
        at += size
        return bytes(data[at - size:at]).decode("utf-8", errors="replace")

    scalars = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            item, count = take("I"), take("Q")
            return [value(item) for _ in range(count)]
        return take(scalars[kind])

    if bytes(data[:4]) != b"GGUF":
        raise SystemExit("saluki: this is not a GGUF file")
    at = 4
    version, count, entries = take("I"), take("Q"), take("Q")
    metadata = {}
    for _ in range(entries):
        key = string()
        metadata[key] = value(take("I"))
    tensors = []
    for _ in range(count):
        name = string()
        dims = [take("Q") for _ in range(take("I"))]
        tensors.append((name, dims, take("I"), take("Q")))
    alignment = metadata.get("general.alignment", 32)
    return metadata, tensors, (at + alignment - 1) // alignment * alignment


def head(directory, file=None):
    directory.mkdir(parents=True, exist_ok=True)
    if file:
        with open(file, "rb") as source:
            data = source.read(HEAD_BYTES)
        size = Path(file).stat().st_size
    else:
        kept = directory / "saluki-head.bin"
        if not kept.exists() or kept.stat().st_size != HEAD_BYTES:
            kept.write_bytes(ranged(URL, 0, HEAD_BYTES, wait=2))
        data, size = kept.read_bytes(), sized(URL)
    wrong = []

    def held(what, got, want):
        if got != want:
            wrong.append(f"{what} is {got}, not {want}")

    metadata, tensors, base = gguf_head(data)
    held("the size of the file", size, BYTES)
    held("the count of tensors", len(tensors), sum(count for *_, count in TYPES.values()))
    for key, want in NUMBERS.items():
        held(key, metadata.get(key), want)
    template = hashlib.sha256(metadata.get("tokenizer.chat_template", "").encode()).hexdigest()[:16]
    held("the chat template's sha256", template, TEMPLATE)
    ends = sorted(offset for *_, offset in tensors) + [size - base]
    after = dict(zip(ends, ends[1:]))
    seen = {}
    for name, dims, kind, offset in tensors:
        if kind not in TYPES:
            wrong.append(f"{name} is of ggml's type {kind}, which the notes do not have")
            continue
        title, values, block, _ = TYPES[kind]
        count = int(np.prod(dims, dtype=object))
        took = count // values * block
        # (a row is dims[0] values: whole blocks of them, so that a row can be read by itself)
        if dims[0] % values or not took <= after[offset] - offset < took + 32:
            wrong.append(f"{name} ({title}, {dims}) takes {after[offset] - offset} bytes where {took} are its blocks")
        tally = seen.setdefault(title, [0, 0, 0])
        tally[0] += 1
        tally[1] += count
        tally[2] += took
    for kind, (title, _, _, count) in TYPES.items():
        held(f"the count of {title} tensors", seen.get(title, [0])[0], count)
    total = sum(tally[2] for tally in seen.values())
    print(f"saluki: {FILE} at {REVISION[:10]}: {size:,} bytes, {len(tensors)} tensors from byte {base:,}")
    for title, (count, values, took) in sorted(seen.items(), key=lambda item: -item[1][2]):
        print(f"saluki: {title:8} {count:4} tensors {values:15,} values {took:14,} bytes "
              f"{8 * took / values:7.4f} bits a value {took / total:6.1%}")
    for line in wrong:
        print(f"saluki: WRONG: {line}")
    print(f"saluki: the head is {'NOT ' if wrong else ''}the file the notes describe")
    return 1 if wrong else 0


def fetch(directory):
    target = directory / FILE
    download(URL, target, tries=4, timeout=120, said=lambda error: print(f"saluki: fetching again: {error}", flush=True))
    if target.stat().st_size != BYTES:
        print(f"saluki: {FILE} has {target.stat().st_size:,} bytes, not {BYTES:,}")
        return 1
    digest = hashlib.sha256()
    with open(target, "rb") as source:
        while block := source.read(16 << 20):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        print(f"saluki: {FILE} has the sha256 {digest.hexdigest()}, not {SHA256}")
        return 1
    print(f"saluki: {FILE} is whole: {BYTES:,} bytes, the sha256 of the revision's file")
    return 0


def ids_of(path):
    lines = path.read_text().split("\n")
    return [int(word) for word in lines[0].split()], [int(word) for word in (lines[1] if len(lines) > 1 else "").split()]


def agree(directory, out=None):
    """What the driver wrote in directory. The exit code: 1 where a file is not whole, a logit is not finite or a
    written token is not the largest logit of its row; how far the two paths are apart fails nothing (it is measured)."""
    summary, wrong = [], []
    names = sorted(directory.glob("fork-*.ids"), key=lambda path: int(path.stem.split("-")[1]) if path.stem.split("-")[1].isdigit() else -1)
    names = [path for path in names if path.stem.split("-")[1].isdigit()]
    if not names:
        print(f"saluki: no fork-<i>.ids in {directory}")
        return 1
    for path in names:
        index = int(path.stem.split("-")[1])
        prompt, wrote = ids_of(path)
        rows = len(prompt) + max(len(wrote) - 1, 0)
        logits = np.memmap(directory / f"fork-{index}.logits", dtype=np.float32, mode="r")
        if not wrote or logits.size % rows:
            wrong.append(f"text {index}: {logits.size} logits are not {rows} rows ({len(prompt)} tokens, {len(wrote)} written)")
            continue
        logits = logits.reshape(rows, -1)
        vocabulary = logits.shape[1]
        line = {"text": index, "prompt": prompt, "wrote": wrote, "vocabulary": vocabulary}
        if not all(np.isfinite(logits[row]).all() for row in range(rows)):
            wrong.append(f"text {index}: a logit is not finite")
        # greedy: the token written at step k is the largest logit of row len(prompt) - 1 + k, the first of equals
        chosen = [int(np.argmax(logits[len(prompt) - 1 + step])) for step in range(len(wrote))]
        if chosen != wrote:
            wrong.append(f"text {index}: wrote {wrote}, and the largest logits of its rows are {chosen}")
        top = np.sort(logits[len(prompt) - 1])[-2:]
        line["margin of the first written token"] = float(top[1] - top[0])
        single = directory / f"fork-{index}.single"
        if single.exists():
            alone = np.memmap(single, dtype=np.float32, mode="r")
            if alone.size != len(prompt) * vocabulary:
                wrong.append(f"text {index}: {alone.size} logits one at a time are not {len(prompt)} rows")
            else:
                alone = alone.reshape(len(prompt), vocabulary)
                apart = [float(np.abs(alone[row] - logits[row]).max()) for row in range(len(prompt))]
                same = sum(int(np.argmax(alone[row]) == np.argmax(logits[row])) for row in range(len(prompt)))
                spread = float(np.mean([logits[row].std() for row in range(len(prompt))]))
                line.update({"one at a time against one batch, the largest difference": max(apart),
                             "by row": apart,
                             "its mean over the rows": float(np.mean(apart)),
                             "rows whose largest logit is the same token": f"{same} of {len(prompt)}",
                             "the standard deviation of a row": spread})
                print(f"saluki: text {index}: {len(prompt)} tokens, one at a time against one batch: at most "
                      f"{max(apart):.3e} apart (mean {np.mean(apart):.3e}; a row's logits spread by {spread:.2f}), "
                      f"the same largest token in {same} of {len(prompt)} rows")
                # (row by row: whether the two paths part at the first token, which has nothing before it, or later)
                print(f"saluki: text {index}: apart by row: {' '.join(f'{value:.3f}' for value in apart[:12])}"
                      f"{' ...' if len(apart) > 12 else ''}; the most at row {int(np.argmax(apart))}")
        print(f"saluki: text {index}: wrote {wrote} (the first by a margin of {line['margin of the first written token']:.3f})")
        summary.append(line)
    floors = [line["one at a time against one batch, the largest difference"] for line in summary
              if "one at a time against one batch, the largest difference" in line]
    if floors:
        print(f"saluki: llama.cpp's two paths are at most {max(floors):.3e} apart: ten times that, {10 * max(floors):.3e}, "
              "is the least a line against these logits can be")
    for line in wrong:
        print(f"saluki: WRONG: {line}")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "summary.json").write_text(json.dumps({"file": FILE, "revision": REVISION, "texts": summary,
                                                            "wrong": wrong}, indent=1) + "\n")
    print(f"saluki: the reference's files are {'NOT ' if wrong else ''}whole")
    return 1 if wrong else 0


def made_up(directory):
    """A prompt in the driver's form with a vocabulary of 50, twice: in <directory>/right as the driver would write it,
    in <directory>/wrong with a last written token that is not its row's largest logit (agree must refuse it)."""
    rng = np.random.default_rng(361)
    for index, broken, where in ((0, False, directory / "right"), (0, True, directory / "wrong")):
        where.mkdir(parents=True, exist_ok=True)
        prompt = [int(token) for token in rng.integers(0, 50, 5)]
        logits = rng.standard_normal((5 + 3, 50)).astype(np.float32)
        wrote = [int(np.argmax(logits[4 + step])) for step in range(4)]
        if broken:
            wrote[-1] = (wrote[-1] + 1) % 50
        (where / f"fork-{index}.ids").write_text(" ".join(map(str, prompt)) + "\n" + " ".join(map(str, wrote)) + "\n")
        logits.tofile(where / f"fork-{index}.logits")
        (logits[:5] + rng.standard_normal((5, 50)).astype(np.float32) * np.float32(1e-3)).tofile(where / f"fork-{index}.single")
        (where / f"prompt-{index}.txt").write_text("made up")
    return 0


if __name__ == "__main__":
    command, directory, *more = sys.argv[1:]
    run = {"head": head, "fetch": fetch, "agree": agree, "made-up": made_up}[command]
    sys.exit(run(Path(directory), *more))
