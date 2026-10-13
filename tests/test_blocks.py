# T363.2: a checkpoint whose matrices are held as the file has them, in llama.cpp's blocks (the K- and i-quantized types
# of a GGUF of Underdog Saluki 27B), a type to a tensor: to each layer of a stack. What the registry says of such a kind
# (engine/dtypes.py's BLOCKS), the form's "kinds" with a kind for each layer, where the rows then are and how large the
# file is, what checkpoint_dtype() answers, the codebooks such a file carries, and the engine in NumPy reading it: on a
# small model of the real one's shape, whose logits are those of the same values in a float32 file, to the bit. The
# blocks are random bytes (any bytes are blocks): what they read as, by the converter's readers
# (tests/test_superblocks.py holds those to the forms), are the model's weights.
import json
import math
import struct

import numpy as np
import pytest
import saluki_form
from made_up import BLOCK_SCALES, checkpoint, drawn_blocks, numbered, uniform
from test_external import Outside

from convert import grids
from convert.superblocks import SUPERBLOCKS
from llama2_convert import SOURCES, Writer, checkpoint_size
from llama2_numpy import (BLOCKS, CODEBOOKS, CODES, DTYPES, EITHER, FORM, KINDS, PACKED, QUANTIZED, SEVERAL_KINDS, Llama,
                          checkpoint_dtype, dtype_of, external_tensors, file_size, form_of, kind_of, placed, tensor_rows)

# the bytes of a block of 256 weights, written out a second time (docs/notes/t361-saluki-2026-10-13.md's table)
SIZES = {"Q2_K": 84, "Q4_K": 144, "IQ2_XXS": 66, "IQ2_XS": 74, "IQ2_S": 82, "IQ3_XXS": 98, "IQ3_S": 110, "IQ1_S": 50,
         "IQ1_M": 56, "IQ4_XS": 136}
TEN = list(SIZES)

# The real model's shape in small: a Qwen3.5 with a classifier of its own, a full-attention layer every fourth, three
# value heads to a key head, every row of a matrix whole blocks of 256
HEADER = [256, 512, 8, 4, 2, -40, 24]
LINEAR = {"every": 4, "key_heads": 2, "value_heads": 6, "key_dim": 64, "value_dim": 128, "conv": 4}
SHAPE = {"arch": "qwen35", "head_dim": 64, "linear": LINEAR}
OPTIONS = {**SHAPE, "rotary": 16}
STACKS = {"wq": 2, "wg": 2, "wk": 2, "wv": 2, "wo": 2, "wqkv": 6, "wz": 6, "wout": 6, "w1": 8, "w2": 8, "w3": 8}


def kinds_like_saluki():
    """A type to a tensor, so that the matrices of one role are of several types and every type is some matrix's; the
    embedding and the classifier the real file's; the two small matrices of the gates int8 in one and a kind to a layer
    in the other, of the kinds that are no blocks."""
    kinds, at = {"token_embedding_table": "IQ1_M", "wcls": "IQ3_S"}, 0
    for name, layers in STACKS.items():
        kinds[name] = [TEN[(at + 3 * layer) % len(TEN)] for layer in range(layers)]
        at += 1
    return {**kinds, "wb": "int8", "wa": ["int8", "float32", "float16", "int6", "int8", "float32"]}


KINDS_OF_IT = kinds_like_saluki()
MADE = {**SHAPE, "kinds": KINDS_OF_IT}
TOKENS = [1, 5, 7, 5, 2, 39]


def recorded(seed=4):
    """drawn_blocks(), and what it gave: [(the kind, the bytes)] in the order of the file."""
    given, blocks = [], drawn_blocks(seed)

    def recording(name, count):
        given.append((name, blocks(name, count)))
        return given[-1][1]
    return recording, given


@pytest.fixture(scope="module")
def made():
    blocks, given = recorded()
    data = checkpoint(HEADER, MADE, "int8", tensor=uniform(2), blocks=blocks)
    model = Llama(data, numbered(40), dtype="int8", kinds=KINDS_OF_IT, **OPTIONS)
    return data, model, given


# ---- the registry
def test_the_blocks_are_kinds_of_a_row_and_no_dtype_of_a_file():
    assert set(BLOCKS) == set(SIZES) == set(SUPERBLOCKS) == {name for name, source in SOURCES.items() if source.values == 256}
    assert list(DTYPES) == ["float32", "float16", "int8", "int6", "ternary"], "a file's dtypes are the five they were"
    assert not set(BLOCKS) & (set(DTYPES) | set(QUANTIZED) | set(PACKED) | set(EITHER))
    assert set(KINDS) == set(DTYPES) | set(BLOCKS) | {"uint8"} and all(KINDS[name] is DTYPES[name] for name in DTYPES)
    for name in BLOCKS:
        # (the words are those of before the blocks: nobody is told a file may be of one)
        with pytest.raises(ValueError, match=f"^dtype must be float32, float16, int8, int6 or ternary, not {name}.$"):
            dtype_of(name)


@pytest.mark.parametrize("name", TEN)
def test_a_block_is_256_weights_in_its_bytes_with_its_scales_inside(name):
    kind = BLOCKS[name]
    assert kind.name == name and kind.block == SIZES[name] == SOURCES[name].size == SUPERBLOCKS[name][1]
    assert kind.group(256) == kind.group(17408) == 256 and not kind.bits and not kind.itemsize and not kind.rounds
    assert kind.pack is None and kind.packer is None, "nothing is rounded into blocks"
    assert kind.stored_bytes(5 * 256, 256) == (5 * SIZES[name], 0)
    assert kind.suits(256) and kind.suits(5120) and not kind.suits(128) and not kind.suits(384) and "256" in kind.unsuited
    raw = drawn_blocks(1)(name, 7)
    assert len(raw) == 7 * SIZES[name]
    read = kind.floats(np.frombuffer(raw, dtype=np.uint8), np.zeros(0, dtype=np.float32), 256)
    assert read.dtype == np.float32 and read.shape == (7 * 256,) and np.isfinite(read).all() and np.ptp(read) > 0
    assert np.array_equal(read, SUPERBLOCKS[name][2](raw)) and np.array_equal(read, SOURCES[name].read(raw))


def test_the_quantized_dtypes_are_stored_as_they_were():
    """What the new columns say of the kinds that were there: a float32 scale behind every group, and bits a value."""
    for name, (values, scales) in {"int8": (1024, 128), "int6": (768, 128), "ternary": (256, 32)}.items():
        kind = DTYPES[name]
        assert kind.stored_bytes(1024, kind.group(1024)) == (values, scales) and kind.scale == 4 and not kind.block and kind.read is None
    assert not DTYPES["float32"].group and DTYPES["float32"].suits(7) and KINDS["uint8"].itemsize == 1


def test_made_up_blocks_have_finite_scales_of_either_sign():
    for name in TEN:
        raw = np.frombuffer(drawn_blocks(3)(name, 400), dtype=np.uint8).reshape(400, -1)
        if name == "IQ1_M":
            words = raw[:, 48:].copy().view("<u2").astype(np.uint16)
            scales = (words[:, 0] >> 12) | ((words[:, 1] >> 8) & 0xF0) | ((words[:, 2] >> 4) & 0xF00) | (words[:, 3] & 0xF000)
            scales = scales.astype("<u2").view("<f2")
        else:
            scales = np.concatenate([raw[:, at:at + 2].copy().view("<f2") for at in BLOCK_SCALES.get(name, (0,))], axis=1)
        assert ((np.abs(scales) >= 2.0 ** -12) & (np.abs(scales) < 2.0 ** -11)).all() and (scales < 0).any() and (scales > 0).any()


# ---- the form's kinds
def test_a_stack_is_given_a_kind_for_each_layer_and_json_carries_it():
    rows = {row.name: row for row in tensor_rows(HEADER, MADE)}
    assert rows["w1"].kind == tuple(KINDS_OF_IT["w1"]) and len(set(rows["w1"].kind)) > 4, "one role, several types"
    assert rows["wcls"].kind == "IQ3_S" and rows["wb"].kind == "int8" and rows["rms_att_weight"].kind is None
    assert {kind for name in STACKS for kind in rows[name].kind} == set(TEN), "every type is some matrix's"
    carried = json.loads(json.dumps(MADE))
    assert carried == MADE and tensor_rows(HEADER, carried) == tensor_rows(HEADER, MADE)
    assert form_of({**carried, "dtype": "int8", "bos": 1})["kinds"] == KINDS_OF_IT, "out of a model's options"
    # (tuples say the same)
    assert tensor_rows(HEADER, {**SHAPE, "kinds": {name: tuple(kind) if isinstance(kind, list) else kind
                                                   for name, kind in KINDS_OF_IT.items()}}) == tensor_rows(HEADER, MADE)
    # a kind for every layer that is the same kind is the stack in that kind, layer by layer the same bytes
    same = placed(tensor_rows(HEADER, {**SHAPE, "kinds": {"w1": ["int6"] * 8}}), "int8")
    one = placed(tensor_rows(HEADER, {**SHAPE, "kinds": {"w1": "int6"}}), "int8")
    assert [place.size for place in same] == [place.size for place in one]


@pytest.mark.parametrize("kinds, words", [
    ({"w1": ["IQ2_XXS"] * 7}, "is given 7 kinds, one for a layer, and it is a stack of 8"),
    ({"wq": ["IQ2_XXS"] * 8}, "is given 8 kinds, one for a layer, and it is a stack of 2"),
    ({"wcls": ["IQ3_S"]}, "is given 1 kinds, one for a layer, and it is no stack of layers"),
    ({"w1": ["IQ2_XXS"] * 7 + ["IQ5_XS"]}, "dtype must be"),
    ({"w1": "Q3_K"}, "dtype must be"),
    ({"w9": ["IQ2_XXS"]}, "no tensor called 'w9'"),
])
def test_kinds_that_are_not_the_models_are_refused(kinds, words):
    with pytest.raises(ValueError, match=words):
        tensor_rows(HEADER, {**SHAPE, "kinds": kinds})
    with pytest.raises(ValueError, match=words):
        checkpoint_size(HEADER, "int8", {**SHAPE, "kinds": kinds})


def test_a_row_that_is_no_whole_blocks_is_refused_in_the_blocks_words():
    """dim 128: the rows of w1 are half a block. Sizing does not refuse (as with the packed dtypes), reading and
    writing do."""
    header, kinds = [128, 256, 2, 4, 4, 320, 24], {"w1": ["int8", "IQ2_XXS"]}
    rows = tensor_rows(header, {"kinds": kinds})
    for refused in (lambda: checkpoint(header, {"kinds": kinds}, "int8", tensor=uniform(1)),
                    lambda: Llama(struct.pack("<7i", *header) + bytes(file_size(rows, "int8") - 28), numbered(320), dtype="int8", kinds=kinds),
                    lambda: Writer(bytearray(file_size(rows, "int8")), header, "int8", {"kinds": kinds})):
        with pytest.raises(ValueError, match="llama.cpp's blocks hold 256 weights of a row"):
            refused()
    # (w2's rows are 256 long)
    checkpoint(header, {"kinds": {"w2": ["int8", "IQ2_XXS"]}}, "int8", tensor=uniform(1))


# ---- where the rows are
def by_hand(row, kind):
    """The bytes of a tensor of one kind, by the arithmetic of the note and of quantize.py, with nothing of the registry."""
    count = math.prod(row.shape)
    if kind in SIZES:
        return count // 256 * SIZES[kind]
    return {"float32": 4 * count, "float16": 2 * count, "uint8": count, "int8": count + count // 32 * 4,
            "int6": count // 32 * 24 + count // 32 * 4, "ternary": count // 128 * 32 + count // 128 * 4}[kind]


def test_the_layers_of_a_stack_follow_one_another_each_in_its_own_bytes():
    rows = tensor_rows(HEADER, MADE)
    places, offset = placed(rows, "int8"), 28
    assert [place.row for place in places] == rows
    for place in places:
        assert place.offset == offset
        if not isinstance(place.kind, tuple):
            assert not place.layers and place.parts == (place,)
            assert place.size == (by_hand(place.row, place.kind) if place.kind else 0)
            offset += place.size
            continue
        assert place.kind == place.row.kind and len(place.layers) == place.row.shape[0] and place.parts == place.layers
        assert not place.group and not place.scales, "a stack of several kinds has no one group"
        for layer, kind in zip(place.layers, place.kind):
            assert layer.offset == offset and layer.kind == kind and layer.row.shape == place.row.shape[1:] and not layer.layers
            assert layer.size == by_hand(layer.row, kind) > 0
            assert layer.group == (256 if kind in SIZES else 32 if kind in QUANTIZED else 0)
            assert layer.scales == (layer.offset + layer.size if kind in SIZES else
                                    layer.offset + (layer.size - 4 * math.prod(layer.row.shape) // 32) if kind in QUANTIZED else 0)
            offset += layer.size
        assert place.size == offset - place.offset
    assert offset == file_size(rows, "int8") == checkpoint_size(HEADER, "int8", MADE)
    sizes = {place.row.name: place.size for place in places}
    # (two stacks of one shape whose layers have other types are other sizes: the layers' own kinds are counted)
    assert sizes["w1"] != sizes["w3"] and sizes["wq"] != sizes["wg"]


def test_the_rows_without_a_kind_of_their_own_are_where_the_files_dtype_puts_them():
    """A kind to a layer moves what lies behind the stack and changes nothing else of a row's place: its kind, its
    size, its group and where its scales are from its start."""
    several = placed(tensor_rows(HEADER, {**SHAPE, "kinds": {"wq": ["int6", "IQ2_XXS"], "w2": ["Q2_K"] * 8}}), "int8")
    plain = placed(tensor_rows(HEADER, SHAPE), "int8")
    assert [place.row.name for place in several] == [place.row.name for place in plain] + ["codebooks"]
    for one, other in zip(several, plain):
        if one.row.name not in ("wq", "w2"):
            assert (one.kind, one.size, one.group, one.group and one.scales - one.offset) == \
                (other.kind, other.size, other.group, other.group and other.scales - other.offset)
    assert several[-1].offset + CODEBOOKS == file_size(tensor_rows(HEADER, {**SHAPE, "kinds": {"wq": ["int6", "IQ2_XXS"], "w2": ["Q2_K"] * 8}}), "int8")


# ---- the codebooks
def test_a_file_with_blocks_carries_the_codebooks_last_and_no_other_file_has_the_row():
    for form in ({}, SHAPE, {**SHAPE, "kinds": {"w1": "int6", "wa": ["int8", "float32"] * 3}}, {"arch": "gpt2"}):
        header = HEADER if "linear" in form else [256, 512, 8, 4, 4, 320, 24]
        assert all(row.role != CODES for row in tensor_rows(header, form)), "a file without blocks has the rows it had"
    plain = tensor_rows(HEADER, SHAPE)
    for kinds in ({"wcls": "IQ3_S"}, {"wa": ["int8"] * 5 + ["Q4_K"]}, KINDS_OF_IT):
        rows = tensor_rows(HEADER, {**SHAPE, "kinds": kinds})
        assert [row.name for row in rows] == [row.name for row in plain] + ["codebooks"]
        assert (rows[-1].role, rows[-1].shape, rows[-1].kind, rows[-1].per) == (CODES, (CODEBOOKS,), "uint8", None)
        for dtype in DTYPES:
            assert kind_of(rows[-1], dtype) == "uint8" and placed(rows, dtype)[-1].size == CODEBOOKS


def test_the_codebooks_are_the_converters_grids_one_after_the_other():
    raw, places = grids.codebooks()
    assert len(raw) == CODEBOOKS == sum(grids.grid(name).size for name in grids.DIGITS) + 16
    assert list(places) == ["IQ2XXS", "IQ2XS", "IQ2S", "IQ3XXS", "IQ3S", "IQ1S", "IQ4NL"]
    held = np.frombuffer(raw, dtype=np.int8)
    for name in grids.DIGITS:
        table = grids.grid(name)
        assert np.array_equal(held[places[name]:places[name] + table.size].reshape(table.shape), table)
    assert np.array_equal(held[places["IQ4NL"]:], grids.IQ4NL) and places["IQ4NL"] + 16 == CODEBOOKS
    ends = sorted(places.values()) + [CODEBOOKS]
    assert ends[0] == 0 and ends == sorted(set(ends)), "one after the other"


def test_the_engine_holds_the_codebooks_of_the_file_as_bytes(made):
    data, model, _ = made
    assert model.codebooks.dtype == np.uint8 and model.codebooks.tobytes() == grids.codebooks()[0] == data[-CODEBOOKS:]


# ---- what the file's dtype is
def test_the_dtype_of_a_file_whose_matrices_all_have_kinds_is_int8():
    """Such a file is the same size whichever quantized dtype its other rows follow (they are vectors, float32 in each,
    and no RoPE tables): checkpoint_dtype() says int8, the first of them. As float32 or float16 it is another file."""
    rows = tensor_rows(HEADER, MADE)
    sizes = {dtype: file_size(rows, dtype) for dtype in DTYPES}
    assert sizes["int8"] == sizes["int6"] == sizes["ternary"] and len({sizes["float32"], sizes["float16"], sizes["int8"]}) == 3
    for dtype, said in (("int8", "int8"), ("int6", "int8"), ("ternary", "int8"), ("float32", "float32"), ("float16", "float16")):
        assert checkpoint_dtype(HEADER, sizes[dtype], MADE) == said
    with pytest.raises(ValueError, match="This is not a llama2.c checkpoint"):
        checkpoint_dtype(HEADER, sizes["int8"] - 1, MADE)
    # with one matrix left to the file's dtype the three are three sizes again
    partly = tensor_rows(HEADER, {**SHAPE, "kinds": {name: kind for name, kind in KINDS_OF_IT.items() if name != "w2"}})
    for dtype in DTYPES:
        assert checkpoint_dtype(HEADER, file_size(partly, dtype), {**SHAPE, "kinds": {name: kind for name, kind in KINDS_OF_IT.items() if name != "w2"}}) == dtype


def shapes():
    """Headers and forms whose matrices' rows are whole blocks, and for each the matrices' rows by name."""
    yield "the small one", HEADER, SHAPE
    yield "the real one", saluki_form.HEADER, {key: value for key, value in saluki_form.FORM.items() if key != "kinds"}
    yield "a llama", [256, 768, 6, 4, 2, 2048, 512], {}
    yield "a llama, its own classifier", [512, 1024, 3, 8, 8, -1024, 64], {}
    yield "a qwen3", [256, 512, 5, 4, 2, 4096, 128], {"qk_norm": True, "head_dim": 128, "bias": True}
    yield "an lfm2", [256, 768, 8, 8, 4, 2048, 320], {"arch": "lfm2", "convolution": {"layers": "ccaccaca", "taps": 3}}
    yield "a gpt2", [256, 1024, 3, 8, 8, -2304, 320], {"arch": "gpt2"}


@pytest.mark.parametrize("name, header, form", list(shapes()), ids=[name for name, _, _ in shapes()])
def test_a_file_of_blocks_is_no_size_of_a_file_of_one_dtype(name, header, form):
    """checkpoint_dtype() goes by the size, and a file of blocks whose form is not said must be none of the five
    files of its header (it would be read as that one). Every matrix in each of the ten types, and types dealt out
    to the layers in a few ways (for the real header: the real file's counts)."""
    rows = tensor_rows(header, form)
    plain = {file_size(rows, dtype) for dtype in DTYPES}
    matrices = [row for row in rows if kind_of(row, "int8") == "int8"]
    assert len(matrices) > 5 and all(row.shape[-1] % 256 == 0 for row in matrices)
    rng = np.random.default_rng(len(name))
    dealt = [{row.name: kind if row.per is None else [kind] * row.shape[0] for row in matrices} for kind in TEN]
    dealt += [{row.name: TEN[rng.integers(10)] if row.per is None else [TEN[i] for i in rng.integers(0, 10, row.shape[0])]
               for row in matrices} for _ in range(20)]
    if header == saluki_form.HEADER:
        dealt.append(saluki_form.KINDS)
    for kinds in dealt:
        said = {**form, "kinds": kinds}
        size = file_size(tensor_rows(header, said), "int8")
        assert size not in plain, kinds
        assert checkpoint_dtype(header, size, said) == "int8"
        with pytest.raises(ValueError, match="This is not a llama2.c checkpoint"):
            checkpoint_dtype(header, size, form)


def test_the_real_file_is_the_size_the_note_counted_and_its_codebooks():
    rows = tensor_rows(saluki_form.HEADER, saluki_form.FORM)
    places = placed(rows, "int8")
    assert file_size(rows, "int8") == saluki_form.BYTES + CODEBOOKS == 7956678684 + 33808
    matrices = sum(layer.size for place in places if place.kind for layer in place.parts if layer.kind in SIZES)
    gates = sum(place.size for place in places if place.row.name in ("wb", "wa"))
    assert (matrices, gates) == (saluki_form.MATRICES, saluki_form.GATES)
    assert all(type(place.offset) is int for place in places) and places[-1].offset > 2 ** 32


# ---- the engine in NumPy reads it
def test_the_engine_reads_every_tensor_of_blocks_as_the_readers_read_its_bytes(made):
    """The file is one tensor after the other with nobody's table of places (tests/made_up.py), and the engine finds
    each layer's blocks: what it holds of a matrix is what the converter's reader makes of the bytes that were put
    there, layer by layer."""
    data, model, given = made
    given, seen = iter(given), set()
    for row in tensor_rows(HEADER, MADE):
        kinds = row.kind if isinstance(row.kind, tuple) else (row.kind,)
        held = getattr(model, row.name)
        for layer, kind in enumerate(kinds):
            if kind not in SIZES:
                continue
            name, raw = next(given)
            shape = row.shape[1:] if isinstance(row.kind, tuple) else row.shape
            assert name == kind and len(raw) == math.prod(shape) // 256 * SIZES[kind]
            values = SUPERBLOCKS[kind][2](raw).reshape(shape)
            if row.name == "token_embedding_table":
                # (read a row at a time: the table stays the bytes of its blocks)
                assert not isinstance(held, np.ndarray) and held.raw.shape == (40, SIZES[kind]) and held.shape == (40, 256)
                assert all(np.array_equal(held[token], values[token]) and np.array_equal(model.embedding(token), values[token])
                           for token in range(40))
            else:
                got = held[layer] if isinstance(row.kind, tuple) else held
                assert got.dtype == np.float32 and np.array_equal(got, values), f"{row.name}, layer {layer}, {kind}"
            seen.add(kind)
    assert next(given, None) is None and seen == set(TEN)
    assert model.wq.shape == (2, 256, 256) and model.w1.shape == (8, 512, 256) and model.wout.shape == (6, 256, 768)


def test_its_logits_are_those_of_the_same_values_in_a_float32_file_to_the_bit(made):
    data, model, _ = made
    assert len(data) == file_size(tensor_rows(HEADER, MADE), "int8")
    held = {row.name: getattr(model, row.name) for row in tensor_rows(HEADER, SHAPE)}
    held["token_embedding_table"] = np.stack([held["token_embedding_table"][token] for token in range(40)])
    same = Llama(checkpoint(HEADER, SHAPE, "float32", tensor=lambda row: held[row.name]), numbered(40), **OPTIONS)
    other = Llama(checkpoint(HEADER, SHAPE, "float32", tensor=uniform(2)), numbered(40), **OPTIONS)
    for pos, token in enumerate(TOKENS):
        logits = model.forward(token, pos).copy()
        assert np.isfinite(logits).all() and np.ptp(logits) > 0
        assert np.array_equal(logits, same.forward(token, pos))
        assert not np.array_equal(logits, other.forward(token, pos)), "any model has these logits: this checks nothing"


def test_the_gates_small_matrices_are_read_in_the_kinds_the_form_gives_their_layers(made):
    """wa has a kind to a layer of the kinds that are no blocks: each layer is what a file of that kind holds of it."""
    _, model, _ = made
    values = uniform(2)
    for row in tensor_rows(HEADER, MADE):
        asked = None if row.role == CODES or all(kind in SIZES for kind in (row.kind if isinstance(row.kind, tuple) else (row.kind,))) else values(row)
        if row.name != "wa":
            continue
        for layer, kind in enumerate(row.kind):
            part = np.asarray(asked[layer], dtype=np.float32)
            if kind in QUANTIZED:
                packed, scales = DTYPES[kind].pack(part)
                part = (DTYPES[kind].unpack(np.ascontiguousarray(packed).reshape(-1)).reshape(-1, 32).astype(np.float32) * scales[:, None]).reshape(part.shape)
            elif kind == "float16":
                part = np.asarray(asked[layer]).astype(np.float16).astype(np.float32)  # (rounded once, from the float64)
            assert np.array_equal(model.wa[layer], part), f"layer {layer}, {kind}"
        return
    raise AssertionError("no wa")


def test_forward_js_is_not_given_a_file_of_blocks(made):
    data, _, _ = made
    with pytest.raises(ValueError, match="several kinds") as refusal:
        Llama(None, numbered(40), dtype="int8", kinds=KINDS_OF_IT, external=Outside(data), **OPTIONS)
    assert str(refusal.value) == SEVERAL_KINDS
    with pytest.raises(ValueError, match="several kinds"):
        external_tensors(HEADER, "int8", MADE)


# ---- the Writer
def test_the_writer_puts_the_rows_of_each_layer_where_the_layer_is():
    """A kind to a layer, of the kinds values are packed into (blocks are copied, which is T363.3's): rows written a few
    at a time, across the layers, make the file the maker makes."""
    header = [128, 256, 4, 4, 2, 320, 24]
    kinds = {"wq": ["int8", "int6", "float16", "int8"], "w2": ["int6", "int6", "float32", "ternary"], "w3": "int6",
             "rms_ffn_weight": ["float32", "float16", "float32", "float16"]}
    form = {"kinds": kinds}

    def ternary_where_asked(row, values=uniform(5)):
        made = values(row)
        if row.name == "w2":
            made[3] = np.sign(made[3]) * np.abs(made[3]).reshape(-1, 128).max(axis=1).repeat(128).reshape(made[3].shape)
        return made
    tensors = {}
    data = checkpoint(header, form, "int8", tensor=lambda row: tensors.setdefault(row.name, ternary_where_asked(row)))
    out = bytearray(len(data))
    writer = Writer(out, header, "int8", form)
    for index, place in enumerate(writer.places):
        if place.kind is None:
            continue
        values, width = tensors[place.row.name].reshape(-1), place.row.shape[-1]
        step = 5 * width  # (the layers of these stacks are no multiples of five rows: a piece lies across two)
        for first in reversed(range(0, values.size, step)):
            writer.write(index, first, values[first:first + step])
    assert bytes(out) == data
    assert Llama(bytes(out), numbered(320), dtype="int8", kinds=kinds).forward(1, 0) is not None


def test_the_forms_default_is_no_kinds_as_it_was():
    assert FORM["kinds"] is None and form_of({})["kinds"] is None
