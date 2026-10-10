# The two registries of T359's second step: the dtypes a checkpoint holds a tensor in (engine/dtypes.py's DTYPES) and
# the types a file stores one in (convert/readers.py's SOURCES), and what they are for: how a row is stored is decided
# for the row, not for the file. A checkpoint whose rows are of several kinds (the form's "kinds") is written by the
# Writer and read by the engine in NumPy; forward.js takes one dtype for a file and such a file is refused there.
import json
import struct

import numpy as np
import pytest
from conftest import pack_checkpoint, pack_tokenizer, synthetic_weights, tiny_vocab
from test_convert import hugging_face, safetensors_file
from test_external import Outside

import llama2_convert
from llama2_convert import GGUF_TENSORS, SOURCES, Conversion, Writer, checkpoint_size, read_types, source_of
from llama2_numpy import (DTYPES, EITHER, FORM, PACKED, QUANTIZED, SEVERAL_KINDS, Llama, checkpoint_dtype, dtype_of,
                          external_tensors, file_size, kind_of, placed, tensor_rows)

MATRICES = ["token_embedding_table", "wq", "wk", "wv", "wo", "w1", "w2", "w3"]


def header_of(config):
    return [config["dim"], config["hidden_dim"], config["n_layers"], config["n_heads"], config["n_kv_heads"],
            config["vocab_size"] if config["shared"] else -config["vocab_size"], config["seq_len"]]


def written(config, weights, dtype, kinds=None):
    """The checkpoint a Writer makes of these weights (llama2.c's, by the rows' names), row by row."""
    header, form = header_of(config), {"kinds": kinds}
    out = bytearray(checkpoint_size(header, dtype, form))
    writer = Writer(out, header, dtype, form)
    for index, row in enumerate(tensor_rows(header, form)):
        writer.write(index, 0, weights[row.name])
    return bytes(out)


def model(config, weights, dtype, kinds=None):
    return Llama(written(config, weights, dtype, kinds), pack_tokenizer(tiny_vocab(config["vocab_size"])), dtype=dtype, kinds=kinds)


def ternary_rows(shape, seed=3):
    """Ternary values: every group of 128 along the row is -s, 0 and s."""
    rng = np.random.default_rng(seed)
    signs = rng.integers(-1, 2, shape).astype(np.float32)
    signs[..., ::128] = 1.0  # (the largest of every group is its scale)
    scales = (0.05 + rng.random((*shape[:-1], shape[-1] // 128, 1))).astype(np.float32)
    return (signs.reshape(*shape[:-1], -1, 128) * scales).reshape(shape)


def close(got, expected, six):
    """tests/test_quantize.py's lines for the logits of an int8 model against its float32 original (random weights are
    a harsher test than trained ones). six: some rows are six bits, a step four times as coarse: the correlation
    alone (tests/test_int6.py holds six bits to the float32 of the same rounded values, as the test below does)."""
    got, expected = got.astype(np.float64), expected.astype(np.float64)
    assert six or np.abs(got - expected).max() / np.abs(expected).max() < 0.1
    assert np.corrcoef(got, expected)[0, 1] > 0.99


TOKENS = [1, 5, 7, 5, 2]


# ---- the dtypes of a checkpoint
def test_the_lists_of_dtypes_come_from_the_registry():
    assert list(DTYPES) == ["float32", "float16", "int8", "int6", "ternary"]
    assert (QUANTIZED, PACKED, EITHER) == (("int8", "int6", "ternary"), ("int6", "ternary"), ("int8", "int6"))
    assert [DTYPES[name].short for name in DTYPES] == ["f32", "f16", "int8", "int6", "ternary"]


def test_a_dtype_is_named_by_a_name_or_a_numpy_dtype_and_any_other_is_refused():
    for name in DTYPES:
        assert dtype_of(name) == name
    assert dtype_of(np.float32) == dtype_of(np.dtype("<f4")) == "float32"
    assert dtype_of(np.float16) == "float16" and dtype_of(np.int8) == "int8"
    for other in ("float64", "int4", np.int16, None):
        with pytest.raises(ValueError, match="dtype must be float32, float16, int8, int6 or ternary, not"):
            dtype_of(other)
    with pytest.raises(ValueError, match="dtype must be"):
        checkpoint_size([128, 256, 2, 4, 4, 320, 24], "float64")


@pytest.mark.parametrize("name", QUANTIZED)
def test_a_quantized_dtype_packs_what_it_says_it_stores_and_reads_it_back(name):
    dtype = DTYPES[name]
    for length in (128, 256) if name != "int8" else (128, 48, 20, 7):
        assert dtype.suits(length)
        group = dtype.group(length)
        values = ternary_rows((5, length)) if name == "ternary" else np.random.default_rng(length).standard_normal((5, length)).astype(np.float32)
        packed, scales = dtype.pack(values)
        stored, scaled = dtype.stored_bytes(values.size, group)
        assert (packed.nbytes, scales.nbytes) == (stored, scaled) and scales.dtype == np.float32
        assert packed.shape == (values.size // group, group * dtype.bits // 8), "rows of a group's bytes"
        back = (dtype.unpack(np.frombuffer(packed.tobytes(), dtype=np.uint8)).reshape(-1, group) * scales[:, None]).reshape(values.shape)
        # at most half a step of the kind away: a step is the group's largest over 127 (int8) or 31 (int6); none (ternary)
        steps = {"int8": 127.0, "int6": 31.0}
        largest = np.abs(values.reshape(-1, group)).max(axis=1, keepdims=True)
        line = 0 if name == "ternary" else largest / steps[name] * 0.5 + 1e-6
        assert (np.abs(back - values).reshape(-1, group) <= line).all()
    assert not DTYPES["int6"].suits(48) and not DTYPES["ternary"].suits(96) and DTYPES["float16"].suits(7)


def test_the_bytes_of_a_row_are_its_kinds():
    """placed() asks the registry: the values in groups and a float32 scale for each, or the values alone."""
    rows = tensor_rows([128, 256, 2, 4, 4, 320, 24])
    wq = next(row for row in rows if row.name == "wq")
    count = 2 * 128 * 128
    sizes = {"float32": 4 * count, "float16": 2 * count, "int8": count + 4 * count // 32, "int6": count * 24 // 32 + 4 * count // 32,
             "ternary": count * 32 // 128 + 4 * count // 128}
    for dtype, size in sizes.items():
        assert placed([wq], dtype)[0].size == size == llama2_convert.tensor_bytes(wq.shape, True, dtype)
    assert file_size(rows, "int8") == checkpoint_size([128, 256, 2, 4, 4, 320, 24], "int8")


# ---- a file of several kinds
MIXED = [
    ("one matrix in six bits, the rest int8", "int8", {"wq": "int6"}),
    ("two in six bits and one float16, the rest int8", "int8", {"wo": "int6", "w2": "int6", "w3": "float16"}),
    ("one ternary among int8", "int8", {"w1": "ternary"}),
    ("an int8 embedding and a six-bit matrix in a float32 file", "float32", {"token_embedding_table": "int8", "wv": "int6"}),
    ("int8 matrices in a ternary file", "ternary", {name: "int8" for name in MATRICES if name != "w1"} | {"wcls": "int8"}),
]


def weights_for(kinds, dtype, shared):
    config, weights = synthetic_weights(dim=128, hidden_dim=256, shared=shared)
    for name in MATRICES:
        if kinds.get(name, dtype) == "ternary":
            weights[name] = ternary_rows(weights[name].shape)
    return config, weights


@pytest.mark.parametrize("name,dtype,kinds", MIXED, ids=[name for name, _, _ in MIXED])
@pytest.mark.parametrize("shared", [True, False])
def test_a_file_of_several_kinds_is_written_and_read(name, dtype, kinds, shared):
    """Each row is written and read as its own kind: what the engine holds of it is what it holds of that row in a
    file that is of that kind throughout. The logits are those of the float32 file of the values each row was rounded
    to, to the bit (the line tests/test_int6.py and test_ternary.py hold a packed dtype to), and near the float32
    original's (the line tests/test_quantize.py holds int8 to)."""
    if not shared and dtype == "ternary":
        kinds = {**kinds, "token_embedding_table": "int8"}
    elif shared:
        kinds = {key: kind for key, kind in kinds.items() if key != "wcls"}
    config, weights = weights_for(kinds, dtype, shared)
    several = model(config, weights, dtype, kinds)
    header = header_of(config)
    rows = tensor_rows(header, {"kinds": kinds})
    assert {row.name: row.kind for row in rows if row.kind} == kinds
    # the size is the sum of the rows' own, and with the form the file is told from its size
    assert len(written(config, weights, dtype, kinds)) == file_size(rows, dtype) == 28 + sum(place.size for place in placed(rows, dtype))
    assert len({file_size(rows, dtype), file_size(tensor_rows(header), dtype)}) == 2, "the kinds change nothing: this checks nothing"
    assert checkpoint_dtype(header, file_size(rows, dtype), {"kinds": kinds}) == dtype
    with pytest.raises(ValueError):
        checkpoint_dtype(header, file_size(rows, dtype))
    # one model for every kind the file has, each row compared with the model of its kind
    kinds_used = {kind_of(row, dtype) for row in rows} - {None}
    whole = {}
    for kind in sorted(kinds_used - {"ternary"}):
        whole[kind] = Llama(written(config, weights, kind), pack_tokenizer(tiny_vocab(config["vocab_size"])), dtype=kind)
    def widened(held):  # (an embedding that stays as it is stored until a row is read)
        return (held[0] * held[1]).reshape(-1, config["dim"]) if isinstance(held, tuple) else np.asarray(held, dtype=np.float32)

    rounded = {"freq_cis_real": several.freq_cis_real, "freq_cis_imag": several.freq_cis_imag}
    for row in rows:
        kind = kind_of(row, dtype)
        if row.name.startswith("freq_cis"):
            continue
        got = rounded[row.name] = widened(getattr(several, row.name))
        if kind == "ternary":
            assert np.array_equal(got, weights[row.name]), f"{row.name}: a ternary row is its values as they are"
        else:
            assert np.array_equal(got, widened(getattr(whole[kind], row.name))), f"{row.name} is not read as {kind}"
        assert (kind in ("float32", "ternary")) == np.array_equal(got, weights[row.name]), f"{row.name}: rounded, unless its kind holds it as it is"
    vocabulary = pack_tokenizer(tiny_vocab(config["vocab_size"]))
    same, reference = Llama(pack_checkpoint(config, rounded), vocabulary), Llama(pack_checkpoint(config, weights), vocabulary)
    for pos, token in enumerate(TOKENS):
        logits = several.forward(token, pos).copy()
        assert np.array_equal(logits, same.forward(token, pos))
        close(logits, reference.forward(token, pos), "int6" in kinds.values())


def test_the_writer_goes_by_the_rows_kind_and_not_by_the_files():
    """The inherited flaw (T359's first step): a matrix the form says is six bits was written as int8 into the room of
    six bits. Its bytes are the six-bit file's bytes of that matrix, wherever the other rows put it."""
    config, weights = synthetic_weights(dim=128, hidden_dim=256)
    header, kinds = header_of(config), {"wq": "int6"}
    several, six, eight = (written(config, weights, dtype, k) for dtype, k in (("int8", kinds), ("int6", None), ("int8", None)))
    at = {name: {place.row.name: place for place in placed(tensor_rows(header, {"kinds": k}), dtype)}
          for name, dtype, k in (("several", "int8", kinds), ("six", "int6", None), ("eight", "int8", None))}
    for row in tensor_rows(header):
        mine, theirs, file = at["several"][row.name], at["six" if row.name == "wq" else "eight"][row.name], six if row.name == "wq" else eight
        assert mine.kind == theirs.kind and mine.size == theirs.size
        assert several[mine.offset:mine.offset + mine.size] == file[theirs.offset:theirs.offset + theirs.size], row.name


def test_a_kind_is_given_to_a_row_of_the_model_and_is_a_dtype_and_suits_the_row():
    config, weights = synthetic_weights(dim=48, hidden_dim=64)
    header = header_of(config)
    with pytest.raises(ValueError, match="no tensor called 'wx'"):
        tensor_rows(header, {"kinds": {"wx": "int6"}})
    with pytest.raises(ValueError, match="dtype must be"):
        tensor_rows(header, {"kinds": {"wq": "int4"}})
    # rows of 48 are no whole groups of 32: the Writer and the engine say so, in the kind's own words
    for build in (lambda: written(config, weights, "int8", {"wq": "int6"}),
                  lambda: Llama(pack_checkpoint(config, weights), b"", dtype="float32", kinds={"wq": "int6"})):
        with pytest.raises(ValueError, match="Six bits a weight needs rows of whole groups of 32"):
            build()
    assert tensor_rows(header, {"kinds": None}) == tensor_rows(header, {"kinds": {}}) == tensor_rows(header)
    assert FORM["kinds"] is None


def test_forward_js_is_not_given_a_file_of_several_kinds():
    """forward.js takes one dtype for a file (its kernels are chosen by the file's, T375): with the weights outside
    Python such a file is refused in words, as it is placed and as the model is built."""
    config, weights = synthetic_weights(dim=128, hidden_dim=256)
    kinds = {"wq": "int6"}
    data = written(config, weights, "int8", kinds)
    tokenizer = pack_tokenizer(tiny_vocab(config["vocab_size"]))
    with pytest.raises(ValueError, match="several kinds") as refusal:
        Llama(None, tokenizer, dtype="int8", kinds=kinds, external=Outside(data))
    assert str(refusal.value) == SEVERAL_KINDS
    with pytest.raises(ValueError, match="several kinds"):
        external_tensors(header_of(config), "int8", {"kinds": kinds})
    # (and without the kinds the same model goes there as ever)
    plain = Outside(written(config, weights, "int8"))
    Llama(None, tokenizer, dtype="int8", external=plain)
    assert plain.plan["tensors"]["wq"]["kind"] == "int8" and external_tensors(header_of(config), "int8", {"kinds": None})["wq"]["kind"] == "int8"


def test_the_options_of_a_conversion_carry_the_kinds_only_where_there_are_any(monkeypatch):
    """How the choice is recorded: the form's "kinds" goes into the options like the rest of the form, where it is not
    the default. No conversion of today has any, and none of their options has the key; a Stream whose form has them
    writes each row as its kind, and Llama(**options) reads the file."""
    config, weights = synthetic_weights(dim=128, hidden_dim=256)
    tensors, published = hugging_face(config, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocab = [["<unk>", 0.0], ["<s>", 0.0], ["</s>", 0.0]] + [[f"▁w{i}", -1.0 - i / 100] for i in range(317)]
    tokenizer = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0, "vocab": vocab}}).encode()

    def conversion(dtype="int8"):
        made = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), tokenizer, "tokenizer.json", dtype=dtype,
                          max_seq_len=config["seq_len"], start=8 + size)
        made.feed(file[8 + size:])
        made.finish()
        return made

    plain = conversion()
    assert "kinds" not in plain.options and plain.stream.writer.places[2].kind == "int8"
    form_of_today = llama2_convert.checkpoint_form
    monkeypatch.setattr("convert.stream.checkpoint_form", lambda *args: {**form_of_today(*args), "kinds": {"wq": "int6"}})
    several = conversion()
    assert several.options["kinds"] == {"wq": "int6"} and several.options["dtype"] == "int8"
    assert {key: value for key, value in several.options.items() if key != "kinds"} == plain.options
    assert len(several.checkpoint) < len(plain.checkpoint)
    read = Llama(bytes(several.checkpoint), bytes(several.tokenizer), **several.options)
    eight = Llama(bytes(plain.checkpoint), bytes(plain.tokenizer), **plain.options)
    assert not np.array_equal(read.wq, eight.wq) and np.array_equal(read.wk, eight.wk) and np.array_equal(read.w1, eight.w1)
    # (q as the conversion to six bits of the whole file holds it)
    monkeypatch.setattr("convert.stream.checkpoint_form", form_of_today)
    six = conversion("int6")
    assert np.array_equal(read.wq, Llama(bytes(six.checkpoint), bytes(six.tokenizer), **six.options).wq)


# ---- the types a file stores a tensor in
def test_the_stored_types_and_what_comes_from_their_table():
    assert list(SOURCES) == ["F32", "F16", "BF16", "Q8_0", "PQ2_0", "PTQ1_0"]
    assert GGUF_TENSORS == {0: "F32", 1: "F16", 30: "BF16", 8: "Q8_0", 142: "PQ2_0", 143: "PTQ1_0"}
    assert read_types() == "F32, F16, BF16, Q8_0, PQ2_0 and PTQ1_0"
    assert [(source.values, source.size) for source in SOURCES.values()] == [(1, 4), (1, 2), (1, 2), (32, 34), (128, 34), (128, 28)]
    assert {name: source.kernel for name, source in SOURCES.items()} == {
        "F32": None, "F16": None, "BF16": "widen_bf16", "Q8_0": "widen_q8_0", "PQ2_0": "widen_pq2_0", "PTQ1_0": "widen_ptq1_0"}
    assert source_of("a", "Q8_0") is SOURCES["Q8_0"]
    with pytest.raises(ValueError, match="w is stored as I64: only F32, F16, BF16, Q8_0, PQ2_0 and PTQ1_0 are read"):
        source_of("w", "I64")


@pytest.mark.parametrize("name", list(SOURCES))
def test_a_stored_type_reads_the_values_its_entry_says_a_block_holds(name):
    source = SOURCES[name]
    for blocks in (1, 3, 257):
        raw = np.random.default_rng(blocks).integers(0, 243, blocks * source.size, dtype=np.uint8).tobytes()
        assert source.bytes(blocks * source.values) == len(raw)
        with np.errstate(invalid="ignore"):  # (random bytes: some scales are NaNs)
            values = source.read(raw)
        assert values.shape == (blocks * source.values,) and values.dtype in (np.float32, np.float16)


def test_a_reader_given_for_a_type_reads_it_and_the_tables_own_reads_the_rest(monkeypatch):
    """Stream's readers (the kernels', in the page) are picked by the type's name; a type without one there is read by
    the table's own reader, and the checkpoint is the same either way."""
    config, weights = synthetic_weights(dim=128, hidden_dim=256)
    tensors, published = hugging_face(config, weights, True)
    file = safetensors_file(tensors, "BF16")
    size = struct.unpack("<Q", file[:8])[0]
    seen = []

    def bf16(raw):
        seen.append(len(raw))
        return llama2_convert.bfloat16(raw)

    outs = []
    for readers in (None, {"BF16": bf16}, {"Q8_0": lambda raw: 1 / 0}):
        stream = llama2_convert.Stream(json.loads(file[8:8 + size]), 8 + size, published, "int8", 1 << 20, readers=readers)
        stream.feed(file)
        stream.finish()
        outs.append(bytes(stream.out))
    assert seen and outs[0] == outs[1] == outs[2]
    with pytest.raises(TypeError):
        llama2_convert.Stream(json.loads(file[8:8 + size]), 8 + size, published, "int8", 1 << 20, bfloat16=bf16)
