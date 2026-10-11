# conducting.py (T374.4): what a tool that converts a model answers the conduct of a conversion with.
#
# Which files a model needs, in which order, which tokenizer is tried next and where the template comes from is the
# conduct's (src/python/convert/conduct.py: a generator that asks), as it is for the page's worker. A tool says where the
# files are and what the conversion is made with, and nothing else:
#
#   conversion = converted(Directory(folder), {"weights": "model.safetensors"}, dtype="float32", sink=Mapped(out))
#
#   Directory  the files of a folder (or of one folder a place: {"weights": ..., "vocabulary": ...})
#   Fetched    the files of a repository of huggingface.co, each fetched whole the first time it is asked for
#              (tests/fetching.py's download(): held to its length, asked for again) and kept in a folder
#   Heads      the same, but for the ranges: only the bytes a head is asked for are fetched (fetching.ranged()), and
#              kept as <name>.head. For the tools that want what the converter says of a model and none of its weights
#   converted  the loop: a conduct answered to its end (the conversion, finished), or, with weights=False, until it asks
#              for the first stream: the conversion as it is made, its options and its tokenizer, nothing of the weights
#              read but what came with a GGUF's head
#   listed     the model a path stands for, as tests/hf_fetch.py lays its files out, and the Directory that answers it
#   Mapped     the converter's sink into a file (a memory map: a checkpoint is never whole in memory); Nothing, the
#              sink of a tool that wants no checkpoint
#
# The tools: perplexity_prepare.py, fixed_outputs.py, format_check.py (and through it write_options.py, start_check.py
# and conversions_compare.py), hf_fetch.py, reference_llama.py, reference_qwen35.py, reference_lfm2.py, page_27b.py.
# The Node tools have the same in tests/conducting.mjs. Tested alone by tests/test_conducting.py.
import sys
import urllib.error
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
import fetching  # noqa: E402
from convert.conduct import conduct  # noqa: E402

MiB = 1 << 20


class Missing(Exception):
    """A file the conversion cannot do without is not there."""


class Files:
    """What answers a conduct from files on a disk: text, bytes, range, size and stream, each taking what its request
    has after the kind (src/python/convert/conduct.py says what each is answered with). path(where, name), the one thing
    a kind of answerer has of its own: the file, or None where there is none.
    instead: {(where, name): a text} to answer with in the place of that file (a config.json changed on purpose).
    part: how many bytes of a stream go to the conversion at a time. fed(): called after each of them."""

    def __init__(self, instead=None, part=8 * MiB, fed=None):
        self.instead, self.part, self.fed = dict(instead or {}), part, fed

    def path(self, where, name):
        raise NotImplementedError

    def place(self, where):
        """The place in words, for the sentence of a file that is missing."""
        raise NotImplementedError

    def bytes(self, where, name):
        if (where, name) in self.instead:
            return self.instead[where, name].encode()
        path = self.path(where, name)
        return None if path is None else path.read_bytes()

    def text(self, where, name):
        data = self.bytes(where, name)
        return None if data is None else data.decode()

    def range(self, where, name, begin, end):
        """(the bytes [begin, end), fewer where the file ends first; the size of the file)"""
        path = self.path(where, name)
        if path is None:
            return None
        with open(path, "rb") as file:
            size = file.seek(0, 2)
            file.seek(begin)
            return file.read(max(min(end, size) - begin, 0)), size

    def size(self, where, name):
        return self.path(where, name).stat().st_size

    def stream(self, where, name, begin, end, before, total, feed):
        """Every part of [begin, end) to the conversion's feed, in the order of the file. (Nothing of a stream with
        nothing in it: the file is not even opened.)"""
        if begin < end:
            with open(self.path(where, name), "rb") as file:
                file.seek(begin)
                for at in range(begin, end, self.part):
                    part = file.read(min(self.part, end - at))
                    if len(part) != min(self.part, end - at):
                        raise OSError(f"{name} ends at {at + len(part):,}, before the {end:,} bytes its head says it has")
                    feed(part)
                    if self.fed:
                        self.fed()
        return None


class Directory(Files):
    """places: a folder, or {where: a folder}."""

    def __init__(self, places, **more):
        super().__init__(**more)
        self.places = places

    def place(self, where):
        return str(self.places[where] if isinstance(self.places, dict) else self.places)

    def path(self, where, name):
        path = Path(self.place(where)) / name
        return path if path.is_file() else None


class Fetched(Files):
    """places: {where: (repository, revision, the folder its files are kept in)}. remember: a file the repository does
    not have is asked for once, and <name>.missing beside the others says so from then on."""

    def __init__(self, places, remember=False, **more):
        super().__init__(**more)
        self.places, self.remember = places, remember

    @classmethod
    def of(cls, hf, directory, between="--", **more):
        """For a model as the list has it (hf: repo, revision, and the same under vocabulary), the files of each
        repository in <directory>/<owner><between><name>/<revision>."""
        at = lambda place: (place["repo"], place["revision"], Path(directory) / place["repo"].replace("/", between) / place["revision"])
        return cls({"weights": at(hf), **({"vocabulary": at(hf["vocabulary"])} if hf.get("vocabulary") else {})}, **more)

    def place(self, where):
        repository, revision, _ = self.places[where]
        return f"{repository} at {revision}"

    def folder(self, where):
        return Path(self.places[where][2])

    def url(self, where, name):
        repository, revision, _ = self.places[where]
        return f"https://huggingface.co/{repository}/resolve/{revision}/{name}"

    def absent(self, where, name):
        return self.remember and (self.folder(where) / f"{name}.missing").exists()

    def gone(self, where, name):
        if self.remember:
            self.folder(where).mkdir(parents=True, exist_ok=True)
            (self.folder(where) / f"{name}.missing").touch()

    def path(self, where, name):
        if self.absent(where, name):
            return None
        try:
            return fetching.download(self.url(where, name), self.folder(where) / name)
        except urllib.error.HTTPError as error:
            # "not there" is a 404 alone: any other refusal (a gated repository, too many requests) is a failure
            if error.code != 404:
                raise
            self.gone(where, name)
            return None


class Heads(Fetched):
    """A range is answered from <name>.head, the beginning of the file as far as it was ever asked for (a conduct asks
    for a head from its first byte, and for each further piece from where the last ended), which grows by a Range
    request where it is too short; <name>.size holds the size of the file. No stream: nothing here has the weights."""

    def size(self, where, name):
        """The size of the file, or None where the repository has none."""
        kept = self.folder(where) / f"{name}.size"
        if kept.exists():
            return int(kept.read_text())
        if self.absent(where, name):
            return None
        size = fetching.sized(self.url(where, name))
        if size is None:
            self.gone(where, name)
            return None
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text(str(size))
        return size

    def range(self, where, name, begin, end):
        size = self.size(where, name)
        if size is None:
            return None
        head, end = self.folder(where) / f"{name}.head", min(end, size)
        have = head.stat().st_size if head.exists() else 0
        if have < end:
            more = fetching.ranged(self.url(where, name), have, end - have)
            partial = head.with_name(f"{head.name}.part")
            partial.write_bytes((head.read_bytes() if have else b"") + more)
            partial.rename(head)
        with open(head, "rb") as file:
            file.seek(begin)
            return file.read(max(end - begin, 0)), size

    def stream(self, where, name, *more):
        raise RuntimeError(f"Only the head of {name} is fetched: this conversion reads no weights (converted(weights=False)).")


def converted(answerer, hf, weights=True, **make):
    """The conversion of a model by its conduct, answered by answerer (a Files; anything with its five answers).
    hf: the model as src/python/convert/conduct.py takes it (names only: weights, config, tokenizer, vocabulary).
    make: what Conversion() takes besides the files (dtype, max_seq_len, sink, quantize_rows, readers).
    weights=False: stops where the first stream is asked for and returns the conversion as it is by then, made and
    unfinished (the request of a stream brings the conversion's own feed: the conversion is whose feed it is).
    Raises Missing for a file the conversion cannot do without, and whatever the converter refuses a file with."""
    steps = conduct(hf, **make)
    try:
        request = next(steps)
        while True:
            kind = request[0]
            if kind == "done":
                return request[1]
            if kind == "missing":
                raise Missing(f"{answerer.place(request[1])} has no {request[2]}.")
            if kind == "stream" and not weights:
                return request[7].__self__
            if kind not in ("text", "bytes", "range", "size", "stream"):
                raise RuntimeError(f"The conduct of the conversion asked for {kind}, which nothing here answers.")
            request = steps.send(getattr(answerer, kind)(*request[1:]))
    finally:
        steps.close()


def listed(path, **more):
    """(the model, the Directory that answers for it) of what tests/hf_fetch.py prints for a model: a .gguf file that
    says everything itself (T74); a folder with the original's config.json and tokenizer and a GGUF linked beside
    them (T136's second stage: the first .gguf by its name); or a folder with model.safetensors, or with the shards
    its index names. No tokenizer is named: the conduct's candidates, in its order. more: what Directory() takes."""
    path = Path(path)
    if path.suffix == ".gguf":
        return {"weights": path.name}, Directory(path.parent, **more)
    ggufs = sorted(file.name for file in path.glob("*.gguf"))
    if ggufs:
        # (any vocabulary that is not empty: the conduct then asks "vocabulary" for all but the weights, and the one
        # folder is both places)
        return {"weights": ggufs[0], "vocabulary": {"folder": path.name}}, Directory(path, **more)
    return {"weights": "model.safetensors"}, Directory(path, **more)


class Mapped:
    """The converter's sink (convert.checkpoint's Writer writes into it): the checkpoint goes straight into a file, a
    memory map, never whole into memory (a float32 Qwen3 0.6B is 2.4 GB, T124). size, header and dtype: what the
    converter opened it with."""

    def __init__(self, path):
        self.path = Path(path)

    def open(self, size, header, dtype, form):
        self.size, self.header, self.dtype = size, list(header), dtype
        self.data = np.memmap(self.path, dtype=np.uint8, mode="w+", shape=(size,))

    def write(self, offset, raw):
        self.data[offset:offset + raw.size] = raw


class Nothing:
    """The sink of a conversion whose checkpoint nobody wants: only its options and its tokenizer. (Without a sink a
    conversion keeps a bytearray of the checkpoint's size: 9.5 GB for a 9B, AGENTS.md.)"""

    def open(self, *args):
        pass

    def write(self, *args):
        pass
