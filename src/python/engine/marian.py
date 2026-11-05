# Marian's binary model (.bin) and its binary shortlist (lex.*.s2t.bin), read as they are (T648.1): the translation
# models of Firefox Translations (Bergamot) are these. Nothing is converted: an intgemm8 matrix stays the file's int8
# with its one quantMult, and a float32 item is a view of the file's bytes.
#
#     model = read_marian(data)          data: the file's bytes (bytes, bytearray or memoryview)
#     model.config                       special:model.yml as a dict ("dec-depth": 4, "dim-vocabs": [32000, 32000], ...)
#     model.items["encoder_l1_self_Wq"]  an Item: kind, shape (the head's), values (the file's order), quant_mult
#     model.alphas()                     a matrix's name -> the fixed quantMult of what is multiplied with it
#     lex = read_shortlist(data)         Shortlist: first, best, offsets, targets; lex.of(source_id)
#
# The layouts were read from marian-dev (browsermt's fork, MIT License: https://github.com/browsermt/marian-dev),
# src/common/binary.cpp (loadItems), src/common/types.h (the type numbers), src/tensors/cpu/integer_common.h (an
# intgemm8 matrix but an embedding is its transpose) and src/data/shortlist.h and shortlist.cpp
# (BinaryShortlistGenerator). Their form was taken and no line of them was copied. What each holds and how it was
# checked against the three models is docs/notes/t648-firefox-translations-format-2026-11-05.md.
#
# The model's file (all little-endian):
#     uint64 version (1), uint64 the number of items n
#     n heads of 32 bytes: uint64 the name's length (with its NUL), uint64 the type, uint64 the number of dimensions,
#                          uint64 the bytes of data
#     the n names (each ending in a NUL), then the n shapes (int32 for each dimension)
#     uint64 offset, then offset bytes of 0 (so that the data begins on a multiple of 256)
#     the n items' data, one after another: each item's bytes are what its shape needs, then 0s up to a multiple of
#     256 (the last item's are not filled: special:model.yml ends the three models with its 1339 bytes). An item's
#     length in its head is read as one of the two, and nothing else.
# An intgemm8 item is the int8 of its elements, then its float32 quantMult (4 bytes, on no word), then the 0s.
#
# The shortlist's file: uint64 magic, checksum, firstNum, bestNum, the length of wordToOffset (the source's vocabulary
# and one), the length of shortLists; then wordToOffset (uint64 each) and shortLists (uint32 each: target words).
import re
import struct
from typing import NamedTuple

import numpy as np

VERSION = 1  # src/common/binary.cpp's BINARY_FILE_VERSION
HEAD = 32  # the bytes of an item's head
SETTINGS = "special:model.yml"
ALPHA = "_QuantMultA"  # an alpha's name is its matrix's and this
# the type numbers of src/common/types.h that are read: a name -> its number
READ = {"float32": 0x0404, "int8": 0x0101, "intgemm8": 0x4101}
# (and those that are not, by name, for what is said when one is found)
NAMED = {0x0102: "int16", 0x0104: "int32", 0x0108: "int64", 0x0201: "uint8", 0x0202: "uint16", 0x0204: "uint32",
         0x0208: "uint64", 0x0402: "float16", 0x0408: "float64", 0x4102: "intgemm16"}
KINDS = {number: name for name, number in READ.items()}
MAGIC = 0xF11A48D5013417F5  # the binary shortlist's
SHORTLIST_HEAD = 48


class Item(NamedTuple):
    """One item of a model. kind is "float32", "int8" or "intgemm8"; shape is the head's. values are the file's, in
    its order: an intgemm8 matrix that is no embedding (a name without "Wemb") is held transposed, its head's [in, out]
    as out rows of in (transposed is then true and values.shape is the head's turned); every other item is values.shape
    = shape. quant_mult is an intgemm8's (the matrix is about values / quant_mult) and None for the rest."""
    name: str
    kind: str
    shape: tuple
    values: np.ndarray
    quant_mult: float | None
    transposed: bool


class Marian(NamedTuple):
    """A model: its binary version, its settings (special:model.yml) and its items by name, in the file's order."""
    version: int
    config: dict
    items: dict

    def alphas(self):
        """A matrix's name -> the quantMult of what is multiplied with it ("<name>_QuantMultA": one float32, fixed, for
        every token). Most are a float32 [1, 1]; the output layer's (decoder_Wemb_QuantMultA, Wemb_QuantMultA) is an
        intgemm8 [1, 1], which Marian turns back into a float32 as it does every item named Wemb: int8 × (1 / its
        quantMult), in float32, as here. "none" is ja-en's none_QuantMultA."""
        alphas = {}
        for name, item in self.items.items():
            if not name.endswith(ALPHA) or item.shape != (1, 1):
                continue
            if item.kind == "float32":
                value = item.values[0, 0]
            elif item.kind == "intgemm8":
                value = np.float32(item.values[0, 0]) * (np.float32(1) / np.float32(item.quant_mult))
            else:
                continue
            alphas[name[:-len(ALPHA)]] = float(value)
        return alphas


def needs(kind, count):
    """The bytes of data an item of this kind and so many elements holds, before its 0s."""
    return {"float32": 4 * count, "int8": count, "intgemm8": count + 4}[kind]


def read_marian(data):
    """A Marian model from its file's bytes; ValueError, saying what was not read, when they are not one."""
    view = memoryview(data).cast("B")
    size = len(view)
    at = 0

    def take(form, what):
        nonlocal at
        length = struct.calcsize(form)
        if at + length > size:
            raise ValueError(f"The Marian model ends inside {what}: it is {size:,} bytes long.")
        values = struct.unpack_from(form, view, at)
        at += length
        return values

    version, count = take("<2Q", "its first 16 bytes")
    if version != VERSION:
        raise ValueError(f"The Marian model is of binary version {version}, which is not read.")
    heads = [take("<4Q", "its items' heads") for _ in range(count)]
    names = []
    for length, *_ in heads:
        (raw,) = take(f"<{length}s", "its items' names")
        if length == 0 or raw[-1] != 0 or 0 in raw[:-1]:
            raise ValueError(f"The Marian model's item {len(names) + 1} has a name that does not end where its head says.")
        try:
            names.append(raw[:-1].decode("utf-8"))
        except UnicodeDecodeError:
            raise ValueError(f"The Marian model's item {len(names) + 1} has a name that is not UTF-8.") from None
    if len(set(names)) != len(names):
        twice = next(name for name in names if names.count(name) > 1)
        raise ValueError(f"The Marian model has two items named {twice}.")
    shapes = [take(f"<{dimensions}i", f"the shape of {name}") for name, (_, _, dimensions, _) in zip(names, heads)]
    (offset,) = take("<Q", "the length of the 0s before its data")
    take(f"<{offset}x", "the 0s before its data")

    items = {}
    for name, (_, number, _, length), shape in zip(names, heads, shapes):
        kind = KINDS.get(number)
        if kind is None:
            said = NAMED.get(number, f"type 0x{number:04x}")
            raise ValueError(f"The Marian model's item {name} is stored as {said}, which is not read.")
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"The Marian model's item {name} has the shape {list(shape)}.")
        elements = 1
        for dimension in shape:
            elements *= dimension  # (Python's integers: Pyodide's NumPy would wrap a product past 2**31)
        need = needs(kind, elements)
        if length not in (need, -(-need // 256) * 256):
            raise ValueError(f"The Marian model's item {name} has {length:,} bytes of data where its shape {list(shape)} "
                             f"takes {need:,}.")
        if at + length > size:
            raise ValueError(f"The Marian model ends inside the data of {name}: it is {size:,} bytes long where its "
                             f"heads say {at + length:,} or more.")
        if any(view[at + need:at + length]):
            raise ValueError(f"The Marian model's item {name} is followed by bytes that are not 0 before the next.")
        transposed = kind == "intgemm8" and "Wemb" not in name
        if kind == "intgemm8" and len(shape) != 2:
            raise ValueError(f"The Marian model's item {name} is an intgemm8 of the shape {list(shape)}, which is not "
                             "read: a matrix is.")
        stored = (shape[1], shape[0]) if transposed else tuple(shape)
        if kind == "float32":
            values, quant_mult = np.frombuffer(view, np.float32, elements, at).reshape(stored), None
        else:
            values = np.frombuffer(view, np.int8, elements, at).reshape(stored)
            quant_mult = struct.unpack_from("<f", view, at + elements)[0] if kind == "intgemm8" else None
        items[name] = Item(name, kind, tuple(shape), values, quant_mult, transposed)
        at += length
    if at != size:
        raise ValueError(f"The Marian model is {size:,} bytes long where its heads say {at:,}.")
    if SETTINGS not in items or items[SETTINGS].kind != "int8":
        raise ValueError(f"The Marian model has no {SETTINGS} to say what it is.")
    return Marian(version, settings(items[SETTINGS].values.tobytes().rstrip(b"\0").decode("utf-8")), items)  # (Marian ends it with a NUL)


SCALAR = re.compile(r"(?P<key>[A-Za-z0-9_.-]+):(?: (?P<value>.*))?$")
INTEGER = re.compile(r"-?[0-9]+$")
FLOAT = re.compile(r"-?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?$")


def scalar(text):
    """One of YAML's plain values as Marian writes them: true and false, integers, floats, "" and [], or text."""
    text = text.strip()
    if text in ("true", "false"):
        return text == "true"
    if text == "[]":
        return []
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if INTEGER.match(text):
        return int(text)
    if FLOAT.match(text):
        return float(text)
    return text


def settings(text):
    """special:model.yml as a dict. Only what Marian writes there is read: "key: value" at the left, and a key with
    nothing after it followed by its list as "  - value" lines; any other line is a ValueError that names it."""
    config, listing = {}, None
    for line in text.split("\n"):
        if not line.strip():
            continue
        if line.startswith("  - ") and listing is not None:
            config[listing].append(scalar(line[4:]))
            continue
        found = SCALAR.match(line)
        if not found or found["key"] in config:
            raise ValueError(f"The Marian model's {SETTINGS} has a line that is not read: {line!r}")
        if found["value"] is None or not found["value"].strip():
            config[found["key"]], listing = [], found["key"]
        else:
            config[found["key"]], listing = scalar(found["value"]), None
    return config


class Shortlist(NamedTuple):
    """A binary shortlist: the first target words every sentence may take (first: ids 0 to first − 1), the best that
    each source word adds at most (best), and for a source word s its target words targets[offsets[s]:offsets[s + 1]].
    checksum is the file's, not checked (its sum is util::hashMem's, which was not read)."""
    checksum: int
    first: int
    best: int
    offsets: np.ndarray
    targets: np.ndarray

    def of(self, word):
        """The target words a source word adds."""
        return self.targets[int(self.offsets[word]):int(self.offsets[word + 1])]


def read_shortlist(data):
    """A binary shortlist from its file's bytes; ValueError, saying what was not read, when they are not one."""
    view = memoryview(data).cast("B")
    size = len(view)
    if size < SHORTLIST_HEAD:
        raise ValueError(f"The shortlist ends inside its first {SHORTLIST_HEAD} bytes: it is {size:,} bytes long.")
    magic, checksum, first, best, words, listed = struct.unpack_from("<6Q", view, 0)
    if magic != MAGIC:
        raise ValueError(f"The shortlist begins with 0x{magic:016x}, which is no binary shortlist of Marian's.")
    expected = SHORTLIST_HEAD + 8 * words + 4 * listed
    if size != expected:
        raise ValueError(f"The shortlist is {size:,} bytes long where its head says {expected:,}.")
    if words == 0:
        raise ValueError("The shortlist has no offsets.")
    offsets = np.frombuffer(view, np.uint64, words, SHORTLIST_HEAD)
    targets = np.frombuffer(view, np.uint32, listed, SHORTLIST_HEAD + 8 * words)
    if int(offsets[0]) != 0 or int(offsets[-1]) != listed or bool(np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("The shortlist's offsets do not go up from 0 to the number of its target words.")
    return Shortlist(checksum, first, best, offsets, targets)
