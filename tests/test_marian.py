# T648.1: engine/marian.py, the reader of Marian's binary model and binary shortlist (Firefox Translations' files).
# A made-up model and shortlist are written here, as marian-dev's saveItems and BinaryShortlistGenerator lay them out
# (the layouts are said in engine/marian.py), and read back; each way of breaking one is refused, by name. The real
# files are read only where they are found (they are not in the repository, and CI has none): MARIAN_FILES names the
# folders to look in (os.pathsep between), and the owner's checkout of bergamot-vs-shisa is looked in by default.
import math
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from engine.marian import read_marian, read_shortlist

SETTINGS = """bert-train-type-embeddings: true
dec-depth: 2
dim-emb: 8
dim-vocabs:
  - 16
  - 16
input-types: []
lemma-dependency: ""
transformer-heads: 2
transformer-dropout: 0.1
type: transformer
version: v1.12.14 2d067af 2024-02-16 11:44:13 -0500
"""


def marian(items, version=1):
    """A Marian model's bytes: items are (name, type number, shape, data bytes); each item's data but the last is
    filled with 0s to a multiple of 256, and the data begins on one, as Marian writes them."""
    names = [name.encode() + b"\0" for name, *_ in items]
    datas = [data + (b"" if i == len(items) - 1 else b"\0" * (-len(data) % 256)) for i, (*_, data) in enumerate(items)]
    head = struct.pack("<2Q", version, len(items))
    head += b"".join(struct.pack("<4Q", len(name), number, len(shape), len(data))
                     for name, (_, number, shape, _), data in zip(names, items, datas))
    head += b"".join(names) + b"".join(struct.pack(f"<{len(shape)}i", *shape) for _, _, shape, _ in items)
    offset = -(len(head) + 8) % 256
    return head + struct.pack("<Q", offset) + b"\0" * offset + b"".join(datas)


def floats(values):
    return np.asarray(values, np.float32).tobytes()


def intgemm(values, quant_mult):
    return np.asarray(values, np.int8).tobytes() + struct.pack("<f", quant_mult)


RNG = np.random.default_rng(648)
WQ = RNG.integers(-127, 128, (3, 5), dtype=np.int8)  # the file's order: out rows of in (the head says [5, 3])
WEMB = RNG.integers(-127, 128, (4, 6), dtype=np.int8)  # an embedding, [V, D] as it is
BIAS = RNG.standard_normal((1, 3)).astype(np.float32)


def made_up():
    return [
        ("Wemb", 0x4101, (4, 6), intgemm(WEMB, 21.5)),
        ("Wemb_QuantMultA", 0x4101, (1, 1), intgemm([127], 19.1986)),
        ("encoder_l1_self_Wq", 0x4101, (5, 3), intgemm(WQ, 86.25)),
        ("encoder_l1_self_Wq_QuantMultA", 0x0404, (1, 1), floats([[7.5]])),
        ("encoder_l1_self_bq", 0x0404, (1, 3), floats(BIAS)),
        ("none_QuantMultA", 0x0404, (1, 1), floats([[6.8115]])),
        ("special:model.yml", 0x0101, (len(SETTINGS.encode()) + 1,), SETTINGS.encode() + b"\0"),  # (the real ones end with a NUL)
    ]


def test_a_made_up_model_reads_back():
    data = marian(made_up())
    model = read_marian(data)
    assert model.version == 1
    assert list(model.items) == [name for name, *_ in made_up()]
    wq = model.items["encoder_l1_self_Wq"]
    assert (wq.kind, wq.shape, wq.transposed, wq.values.shape, wq.values.dtype) == ("intgemm8", (5, 3), True, (3, 5), np.int8)
    assert np.array_equal(wq.values, WQ) and wq.quant_mult == np.float32(86.25)
    emb = model.items["Wemb"]
    assert (emb.transposed, emb.values.shape) == (False, (4, 6)) and np.array_equal(emb.values, WEMB)
    assert emb.quant_mult == np.float32(21.5)
    bias = model.items["encoder_l1_self_bq"]
    assert (bias.kind, bias.values.dtype, bias.quant_mult) == ("float32", np.float32, None)
    assert np.array_equal(bias.values, BIAS)
    # the values are the file's bytes, not a copy
    assert not bias.values.flags.owndata
    assert model.config == {
        "bert-train-type-embeddings": True, "dec-depth": 2, "dim-emb": 8, "dim-vocabs": [16, 16], "input-types": [],
        "lemma-dependency": "", "transformer-heads": 2, "transformer-dropout": 0.1, "type": "transformer",
        "version": "v1.12.14 2d067af 2024-02-16 11:44:13 -0500"}
    alphas = model.alphas()
    assert set(alphas) == {"Wemb", "encoder_l1_self_Wq", "none"}
    assert alphas["encoder_l1_self_Wq"] == 7.5
    # the output layer's, an intgemm8 [1, 1], is Marian's float of it: 127 × (1 / quantMult) in float32
    assert alphas["Wemb"] == float(np.float32(127) * (np.float32(1) / np.float32(19.1986)))
    assert math.isclose(alphas["Wemb"], 6.615, rel_tol=1e-4)
    # bytearray and memoryview read the same
    assert np.array_equal(read_marian(memoryview(bytearray(data))).items["encoder_l1_self_Wq"].values, WQ)


def heads_at(data):
    """Where each item's head is in a made-up model's bytes."""
    return [16 + 32 * i for i in range(struct.unpack_from("<Q", data, 8)[0])]


def broken(data, at, form, change):
    data = bytearray(data)
    (value,) = struct.unpack_from(form, data, at)
    struct.pack_into(form, data, at, change(value))
    return bytes(data)


def refused(data, said):
    with pytest.raises(ValueError, match=said):
        read_marian(data)


def test_a_broken_model_is_refused_by_name():
    data = marian(made_up())
    heads = heads_at(data)
    # an item's bytes of data one more or one fewer, or fewer than its shape takes
    refused(broken(data, heads[4] + 24, "<Q", lambda n: n + 1), "encoder_l1_self_bq has 257 bytes of data where its shape \\[1, 3\\] takes 12")
    refused(broken(data, heads[4] + 24, "<Q", lambda n: n - 1), "encoder_l1_self_bq has 255 bytes of data where its shape \\[1, 3\\] takes 12")
    refused(broken(data, heads[4] + 24, "<Q", lambda n: 4), "encoder_l1_self_bq has 4 bytes of data where its shape \\[1, 3\\] takes 12")
    # the last item's (unfilled): one more or one fewer, and the file does not end where its heads say
    refused(broken(data, heads[-1] + 24, "<Q", lambda n: n + 1), "special:model.yml has .* bytes of data where")
    refused(broken(data, heads[-1] + 24, "<Q", lambda n: n + 256 - n % 256), "is .* bytes long where its heads say")
    # a name's length one more or one fewer: the name does not end where the head says
    refused(broken(data, heads[2], "<Q", lambda n: n + 1), "item 3 has a name that does not end where its head says")
    refused(broken(data, heads[2], "<Q", lambda n: n - 1), "item 3 has a name that does not end where its head says")
    # a type that is not read, by its name or its number
    refused(broken(data, heads[4] + 8, "<Q", lambda n: 0x0402), "encoder_l1_self_bq is stored as float16, which is not read")
    refused(broken(data, heads[4] + 8, "<Q", lambda n: 0x1234), "encoder_l1_self_bq is stored as type 0x1234, which is not read")
    # a shape that takes more than the data (one that still fits in its 0s is not seen: [6, 3] is 22 bytes of 256)
    shapes = data.index(struct.pack("<2i", 5, 3))
    refused(broken(data, shapes, "<i", lambda n: 100), "encoder_l1_self_Wq has 256 bytes of data where its shape \\[100, 3\\] takes 304")
    # another version, a byte too few or too many, and 0s that are not
    refused(broken(data, 0, "<Q", lambda n: 2), "binary version 2, which is not read")
    refused(data[:-1], "is .* bytes long where its heads say")
    refused(data + b"\0", "is .* bytes long where its heads say")
    filled = bytearray(data)
    filled[len(data) - len(SETTINGS) - 1] = 1  # the last 0 after none_QuantMultA
    refused(bytes(filled), "none_QuantMultA is followed by bytes that are not 0")
    refused(data[:40], "ends inside its items' heads")
    # two items of a name, an intgemm8 that is no matrix, settings that are not there or not read
    items = made_up()
    refused(marian([items[4], items[4], items[-1]]), "two items named encoder_l1_self_bq")
    refused(marian([("x_W", 0x4101, (6,), intgemm(range(6), 1.0)), items[-1]]), "x_W is an intgemm8 of the shape \\[6\\]")
    refused(marian(items[:-1]), "has no special:model.yml")
    nested = "a:\n  b: 1\n".encode()
    refused(marian([items[0], ("special:model.yml", 0x0101, (len(nested),), nested)]), "a line that is not read: '  b: 1'")


def shortlist(first, best, lists, magic=0xF11A48D5013417F5, checksum=12345):
    offsets = np.cumsum([0] + [len(words) for words in lists]).astype(np.uint64)
    targets = np.asarray([word for words in lists for word in words], np.uint32)
    return struct.pack("<6Q", magic, checksum, first, best, len(offsets), len(targets)) + offsets.tobytes() + targets.tobytes()


def test_a_made_up_shortlist_reads_back():
    lists = [[3, 9], [], [1, 2, 5, 15], [7]]
    lex = read_shortlist(shortlist(2, 4, lists))
    assert (lex.checksum, lex.first, lex.best) == (12345, 2, 4)
    assert [lex.of(word).tolist() for word in range(4)] == lists
    assert lex.offsets.dtype == np.uint64 and lex.targets.dtype == np.uint32


def test_a_broken_shortlist_is_refused():
    data = shortlist(2, 4, [[3, 9], [], [1, 2, 5, 15], [7]])
    for bytes_, said in [
        (shortlist(2, 4, [[1]], magic=0xF11A48D5013417F4), "begins with 0xf11a48d5013417f4"),
        (data[:-1], "is 115 bytes long where its head says 116"),
        (data + b"\0\0\0\0", "is 120 bytes long where its head says 116"),
        (data[:47], "ends inside its first 48 bytes"),
        (broken(data, 48, "<Q", lambda n: 1), "offsets do not go up"),
        (broken(data, 48 + 8 * 2, "<Q", lambda n: 9), "offsets do not go up"),
        (broken(data, 48 + 8 * 4, "<Q", lambda n: 6), "offsets do not go up"),
    ]:
        with pytest.raises(ValueError, match=said):
            read_shortlist(bytes_)


# ---- the real files, where they are found
FOLDERS = [Path(folder) for folder in os.environ.get("MARIAN_FILES", "").split(os.pathsep) if folder]
FOLDERS.append(Path.home() / "GitHub" / "bergamot-vs-shisa" / "models")
# a file is known by the end of its name and its size (ja-en's 3.0 and 3.1 have one name): the size, the items, and
# what docs/notes/t648-firefox-translations-format-2026-11-05.md read of the settings
MODELS = {
    "en-ja 3.0": ("model.enja.intgemm.alphas.bin", 43_849_787, 253, 69,
                  {"enc-depth": 6, "dec-depth": 4, "dim-emb": 384, "transformer-dim-ffn": 1536, "transformer-heads": 8,
                   "dim-vocabs": [32000, 32000], "tied-embeddings": True, "tied-embeddings-all": False}),
    "ja-en 3.1": ("model.jaen.intgemm.alphas.bin", 43_977_787, 253, 70,
                  {"enc-depth": 6, "dec-depth": 4, "dim-emb": 384, "transformer-dim-ffn": 1536, "transformer-heads": 8,
                   "dim-vocabs": [64000, 64000], "tied-embeddings": False, "tied-embeddings-all": True}),
    "ja-en 3.0": ("model.jaen.intgemm.alphas.bin", 59_504_955, 195, 54,
                  {"enc-depth": 6, "dec-depth": 2, "dim-emb": 512, "transformer-dim-ffn": 2048, "transformer-heads": 8,
                   "dim-vocabs": [64000, 64000], "tied-embeddings": False, "tied-embeddings-all": True}),
}
SHORTLISTS = {
    "en-ja 3.0": ("lex.50.50.enja.s2t.bin", 4_128_360, 32001, 968_076),
    "ja-en 3.1": ("lex.50.50.jaen.s2t.bin", 9_348_172, 64001, 2_209_029),
    "ja-en 3.0": ("lex.50.50.jaen.s2t.bin", 9_346_816, 64001, 2_208_690),
}
SAME = {"type": "transformer", "transformer-decoder-autoreg": "rnn", "dec-cell": "ssru", "transformer-ffn-activation": "relu",
        "transformer-postprocess": "dan", "transformer-postprocess-emb": "d", "transformer-postprocess-top": "",
        "transformer-train-position-embeddings": False}


def found(ending, size):
    for folder in FOLDERS:
        if folder.is_dir():
            for path in sorted(folder.rglob(f"*{ending}")):
                if path.is_file() and path.stat().st_size == size:
                    return path
    pytest.skip(f"no {ending} of {size:,} bytes in {', '.join(map(str, FOLDERS))} (MARIAN_FILES)")


@pytest.mark.parametrize("which", MODELS)
def test_a_real_model(which):
    ending, size, count, alphas, settings = MODELS[which]
    model = read_marian(found(ending, size).read_bytes())
    assert len(model.items) == count
    assert {key: model.config[key] for key in settings} == settings
    assert {key: model.config[key] for key in SAME} == SAME
    assert {item.kind for item in model.items.values()} == {"float32", "intgemm8", "int8"}
    dim, ffn = settings["dim-emb"], settings["transformer-dim-ffn"]
    w1 = model.items["encoder_l1_ffn_W1"]
    assert (w1.shape, w1.values.shape, w1.transposed) == ((dim, ffn), (ffn, dim), True)
    embedding = model.items["decoder_Wemb" if which == "en-ja 3.0" else "Wemb"]
    assert (embedding.shape, embedding.transposed) == ((settings["dim-vocabs"][1], dim), False)
    # every intgemm8 is −127 to 127 (none is −128) and every alpha is the one of a matrix of the model
    assert all(item.values.min() >= -127 for item in model.items.values() if item.kind == "intgemm8")
    found_alphas = model.alphas()
    assert len(found_alphas) == alphas
    assert set(found_alphas) - {"none"} <= set(model.items)
    assert all(np.isfinite(value) and value > 0 for value in found_alphas.values())


@pytest.mark.parametrize("which", SHORTLISTS)
def test_a_real_shortlist(which):
    ending, size, words, listed = SHORTLISTS[which]
    lex = read_shortlist(found(ending, size).read_bytes())
    assert (lex.first, lex.best, len(lex.offsets), len(lex.targets)) == (50, 50, words, listed)
    assert int(np.diff(lex.offsets.astype(np.int64)).max()) <= lex.best
    assert int(lex.targets.max()) < MODELS[which][4]["dim-vocabs"][1]
