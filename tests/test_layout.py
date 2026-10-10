"""engine/layout.py (T359): the rows of a checkpoint are written once, and what reads them agrees because it reads
them. These hold the rows to what the readers need of them (the net's kind "layouts" compares the answers themselves
between two trees)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from engine_plans import engine_plan  # noqa: E402

import llama2_convert  # noqa: E402
import llama2_numpy  # noqa: E402
from llama2_convert import Writer, checkpoint_size, conversion_plan, layout  # noqa: E402
from llama2_numpy import (ATTENDING, EVERY, FORM, MATRIX, QUANTIZED, STATEFUL, TABLE, VECTOR, Dims, Row, after,  # noqa: E402
                          checkpoint_dtype, external_tensors, file_size, kind_of, placed, suited, tensor_rows)

LINEAR = {"every": 4, "key_heads": 2, "value_heads": 4, "key_dim": 64, "value_dim": 128, "conv": 4}
# (name, header, form, what Llama takes besides the form): every dimension of a header differs from the others
MODELS = [
    ("llama", [256, 768, 4, 8, 4, 2048, 320], {}, {}),
    ("llama, its own classifier", [384, 1024, 6, 12, 4, -2304, 288], {}, {}),
    ("qwen2 and qwen3 at once", [256, 768, 4, 4, 2, 2048, 320], {"bias": True, "qk_norm": True, "head_dim": 128}, {}),
    ("gpt2", [256, 1024, 3, 8, 8, -2304, 320], {"arch": "gpt2"}, {}),
    ("neox", [256, 1024, 3, 8, 8, 2048, 320], {"arch": "neox"}, {"rotary": 8}),
    ("qwen35", [256, 768, 8, 4, 2, -2048, 320], {"arch": "qwen35", "head_dim": 128, "linear": LINEAR}, {"rotary": 32}),
    ("lfm2", [256, 768, 8, 8, 4, 2048, 320], {"arch": "lfm2", "convolution": {"layers": "ccaccaca", "taps": 3}}, {}),
]
DTYPES = ("float32", "float16", "int8", "int6", "ternary")
cases = pytest.mark.parametrize("name, header, form, more", MODELS, ids=[model[0] for model in MODELS])


@cases
def test_a_row_has_one_name_and_a_stack_one_tensor_for_each_of_its_layers(name, header, form, more):
    rows, d = tensor_rows(header, form), Dims(header, form)
    names = [row.name for row in rows]
    assert len(set(names)) == len(names)
    for row in rows:
        assert (row.per is None) or row.shape[0] == len(d.layers(row.per)), row
    assert d.layers(EVERY) == list(range(header[2]))
    assert sorted(d.layers(ATTENDING) + d.layers(STATEFUL)) == d.layers(EVERY)


@cases
def test_the_places_follow_one_another_to_the_size_of_the_file(name, header, form, more):
    rows = tensor_rows(header, form)
    for dtype in DTYPES:
        places = placed(rows, dtype)
        assert [place.row for place in places] == rows
        offset = 28
        for place in places:
            assert place.offset == offset and (place.kind is None) == (place.size == 0)
            assert not place.group or (place.offset < place.scales <= place.offset + place.size)
            offset += place.size
        assert offset == file_size(rows, dtype) == checkpoint_size(header, dtype, form)
        assert checkpoint_dtype(header, offset, form) == dtype


@cases
def test_a_quantized_file_keeps_its_vectors_in_float32_and_no_table(name, header, form, more):
    for row in tensor_rows(header, form):
        for dtype in DTYPES:
            kind = kind_of(row, dtype)
            if dtype not in QUANTIZED:
                assert kind == dtype
            else:
                assert kind == ("float32" if row.role == VECTOR else None if row.role == TABLE else dtype), row


@cases
def test_the_engine_holds_every_row_where_the_writer_puts_it(name, header, form, more):
    """Llama(external=), external_tensors() and the converter's Writer read the same rows: the same places."""
    rows = tensor_rows(header, form)
    for dtype in DTYPES:
        sink = type("Sink", (), {"open": lambda *_: None, "write": lambda *_: None})()
        written = {row.name: (offset, tuple(shape)) for row, (offset, shape, _) in zip(rows, Writer(None, header, dtype, form, sink=sink).tensors)}
        plan = engine_plan(llama2_numpy, header, dtype, form, more, checkpoint_size(header, dtype, form))
        held = {name: (tensor["offset"], tuple(tensor["shape"])) for name, tensor in plan["tensors"].items()}
        # the engine's tables are the file's in float32 alone; a model with no classifier of its own has its embedding twice
        expected = {row.name: written[row.name] for row in rows if kind_of(row, dtype) and (row.role != TABLE or dtype == "float32")}
        expected.setdefault("wcls", expected["token_embedding_table"])
        assert held == expected, dtype
        assert external_tensors(header, dtype, form) == plan["tensors"], dtype
        assert plan["shared_classifier"] == (header[5] > 0)


@cases
def test_the_converter_has_a_source_for_every_row_by_its_name(name, header, form, more):
    rows, d = tensor_rows(header, form), Dims(header, form)
    plan, shapes = conversion_plan(header, form, rotary=more.get("rotary", 0))
    assert shapes == [row.shape for row in rows] == [shape for shape, _ in layout(*header, **{**FORM, **form})]
    for row, parts in zip(rows, plan):
        assert (parts is None) == (row.role == TABLE)
        if parts is not None:
            assert len(parts) == (len(d.layers(row.per)) if row.per else 1), row
            assert len({source for source, _ in parts}) == len(parts), row


def test_a_layout_is_made_of_another_and_a_row_without_a_source_is_refused(monkeypatch):
    """A layout that extends a Llama's by a norm for each layer: every reader has the new row at once, and the
    converter says which row it cannot make."""
    header, form = [256, 768, 4, 8, 4, 2048, 320], {"arch": "twice normed"}

    def twice_normed(d):
        return after(llama2_numpy.LAYOUTS["llama"](d), "wo", d.stack("post_norm", VECTOR, EVERY, d.dim))

    monkeypatch.setitem(llama2_numpy.LAYOUTS, "twice normed", twice_normed)
    rows = tensor_rows(header, form)
    assert [row.name for row in rows][5:7] == ["wo", "post_norm"] and rows[6] == Row("post_norm", VECTOR, (4, 256), EVERY)
    assert [row for row in rows if row.name != "post_norm"] == tensor_rows(header)
    assert checkpoint_size(header, "int8", form) == checkpoint_size(header, "int8") + 4 * 4 * 256
    assert checkpoint_dtype(header, checkpoint_size(header, "float16", form), form) == "float16"
    tensors = external_tensors(header, "int8", form)
    assert tensors["post_norm"]["kind"] == "f32" and tensors["rms_ffn_weight"]["offset"] == tensors["post_norm"]["offset"] + 4 * 4 * 256
    with pytest.raises(ValueError, match="no source for the row post_norm"):
        conversion_plan(header, form)


def test_a_packed_dtype_needs_rows_of_whole_groups_wherever_it_is_asked():
    header = [200, 440, 3, 4, 2, 1000, 96]  # rows of 200 and 440 values: no groups of 32
    rows = tensor_rows(header)
    assert suited(rows, "int8") and not suited(rows, "int6") and not suited(rows, "ternary")
    sink = type("Sink", (), {"open": lambda *_: None, "write": lambda *_: None})()
    for dtype, words in (("int6", "Six bits"), ("ternary", "Ternary weights")):
        with pytest.raises(ValueError, match=words):
            Writer(None, header, dtype, sink=sink)
        with pytest.raises(ValueError, match=words):
            external_tensors(header, dtype)
        with pytest.raises(ValueError, match=words):
            engine_plan(llama2_numpy, header, dtype, {}, {}, checkpoint_size(header, dtype))
        with pytest.raises(ValueError, match="not a llama2.c checkpoint"):
            checkpoint_dtype(header, checkpoint_size(header, dtype))
    assert any(row.role == MATRIX for row in rows) and llama2_convert.tensor_bytes((4, 64), True, "int6") == 8 * 28
