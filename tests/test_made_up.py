"""T357.2: the one maker of the tests' made-up models (tests/made_up.py), on its own: that the file it writes is laid
out by the engine's rows and is the file the engine reads back, in every layout and dtype; that a Llama's is still
llama2.c's file (the one order the tests write out themselves, conftest.TENSOR_ORDER); what the values of each kind of
made-up model are; and that a tool can import it without the test suite."""
import struct
import subprocess
import sys

import numpy as np
import pytest
from conftest import ROOT, TENSOR_ORDER
from made_up import (BIASES, checkpoint, drawn, files, header_of, lfm2_model, named, numbered, pack_checkpoint, quantized,
                     settings_of, synthetic_weights, tensors_of, uniform, unigram)

from llama2_numpy import (DTYPES, FORM, LAYOUTS, MATRIX, STATEFUL, VECTOR, Layout, Llama, Stateful, Tokenizer, after,
                          checkpoint_dtype, file_size, kind_of, tensor_rows)

LINEAR = {"every": 2, "key_heads": 2, "value_heads": 4, "key_dim": 64, "value_dim": 32, "conv": 4}
# a header whose rows are whole groups of 128 (the ternary dtype's), and what each layout needs besides
GROUPED, FILLED = (128, 256, 4, 4, 2, 320, 48), (128, 256, 4, 4, 4, -320, 48)
MODELS = {
    "llama": (GROUPED, {}, {}),
    "llama, a classifier of its own": ((128, 256, 4, 4, 2, -320, 48), {}, {}),
    "qwen2 and qwen3": (GROUPED, {"bias": True, "qk_norm": True, "head_dim": 64}, {}),
    "gpt2": (FILLED, {"arch": "gpt2"}, {}),
    "neox": (FILLED, {"arch": "neox"}, {"rotary": 8}),
    "qwen35": (GROUPED, {"arch": "qwen35", "head_dim": 64, "linear": LINEAR}, {"rotary": 16}),
    "lfm2": (GROUPED, {"arch": "lfm2", "qk_norm": True, "convolution": {"layers": "ccac", "taps": 3}}, {}),
}
models = pytest.mark.parametrize("name", list(MODELS))
dtypes = pytest.mark.parametrize("dtype", list(DTYPES))


def by_hand(config, weights):
    """llama2.c's file written out with nothing of the engine: the header, then the tensors in llama2.c's order."""
    vocab_size = config["vocab_size"] if config["shared"] else -config["vocab_size"]
    out = [struct.pack("<7i", config["dim"], config["hidden_dim"], config["n_layers"], config["n_heads"],
                       config["n_kv_heads"], vocab_size, config["seq_len"])]
    names = TENSOR_ORDER + ([] if config["shared"] else ["wcls"])
    return b"".join(out + [np.ascontiguousarray(weights[name], dtype=np.float32).tobytes() for name in names])


@pytest.mark.parametrize("settings", [{}, {"n_kv_heads": 2}, {"n_kv_heads": 1, "shared": False}, {"hidden_dim": 48}])
def test_a_llamas_file_is_llama2cs(settings):
    """The engine's rows of a Llama are llama2.c's tensors in llama2.c's order: a row moved in engine/layout.py moves
    it for whoever writes and whoever reads at once, which no comparison of the two sees, and is another file."""
    config, weights = synthetic_weights(**settings)
    assert [row.name for row in tensor_rows(header_of(config))] == TENSOR_ORDER + ([] if config["shared"] else ["wcls"])
    assert pack_checkpoint(config, weights) == by_hand(config, weights)


@models
@dtypes
@pytest.mark.filterwarnings("ignore:overflow")  # (random weights of 0.3 N: a GPT-2's GELU of thousands)
def test_the_file_is_the_one_the_engine_reads(name, dtype):
    header, form, options = MODELS[name]
    rows, asked, given = tensor_rows(header, form), [], {}
    values = drawn(seed=3, ternary=dtype == "ternary")

    def tensor(row):
        asked.append(row.name)
        given[row.name] = values(row)
        return given[row.name]

    data = checkpoint(header, form, dtype, tensor=tensor)
    # every row the file holds was asked for once, in file order, and no other
    assert asked == [row.name for row in rows if kind_of(row, dtype) is not None]
    assert len(data) == file_size(rows, dtype) and data[:28] == struct.pack("<7i", *header)
    assert checkpoint_dtype(header, len(data), form) == dtype
    llama = Llama(data, numbered(320), dtype=dtype, **form, **options)
    for row in rows:
        if row.role not in (MATRIX, VECTOR) or kind_of(row, dtype) is None:
            continue
        read = getattr(llama, row.name)
        if isinstance(read, np.ndarray) and DTYPES[kind_of(row, dtype)].itemsize:
            # (float16 holds 11 bits of a value)
            assert np.allclose(read.reshape(row.shape), given[row.name], rtol=1e-3, atol=1e-4), row.name
    logits = [llama.forward(token, pos) for pos, token in enumerate([1, 7, 300])]
    assert all(np.isfinite(row).all() and row.shape == (320,) for row in logits)


@pytest.mark.parametrize("dtype", ["float32", "int8", "int6"])
def test_a_layout_nobody_has_written_a_maker_for_is_made(dtype, monkeypatch):
    """An architecture that is another's but for a fact, and one with a new kind of layer that keeps a state (the "rnn"
    of tests/test_architectures.py), are made with no line here: the maker has the rows of whatever the table has."""
    recurrent = Stateful("recurrent", lambda numbers, n_layers: numbers, lambda numbers, n_layers: [kind == "r" for kind in numbers["layers"]])
    rnn = lambda d: after(LAYOUTS["llama"].rows(d), "wo", d.stack("wr", MATRIX, STATEFUL, d.recurrent["width"], d.dim),
                          d.stack("br", VECTOR, STATEFUL, d.dim))
    monkeypatch.setitem(FORM, "recurrent", None)
    monkeypatch.setitem(LAYOUTS, "rnn", Layout(rnn, stateful=recurrent))
    monkeypatch.setitem(LAYOUTS, "rms gpt2", LAYOUTS["gpt2"]._replace(layer_norm=False))
    for header, form in ((GROUPED, {"arch": "rnn", "recurrent": {"layers": "rara", "width": 96}}), (FILLED, {"arch": "rms gpt2"})):
        rows, given = tensor_rows(header, form), {}

        def tensor(row, values=drawn(seed=5)):
            given[row.name] = values(row)
            return given[row.name]

        data = checkpoint(header, form, dtype, tensor=tensor)
        assert len(data) == file_size(rows, dtype) and checkpoint_dtype(header, len(data), form) == dtype
        if dtype == "float32":
            read = tensors_of(data, form)
            assert all(np.array_equal(read[row.name], given[row.name]) for row in rows)
    # (what the values are is by the rows' names: a bias of a new name is a norm's weight to drawn() until BIASES has it)
    assert abs(given["bq"].mean()) < 0.1 and abs(given["rms_att_weight"].mean() - 1) < 0.01


def test_a_float32_file_is_read_back_by_the_same_rows():
    header, form, _ = MODELS["qwen35"]
    given = {}

    def tensor(row, values=uniform(4)):
        given[row.name] = values(row).astype(np.float32)
        return given[row.name]

    read = tensors_of(checkpoint(header, form, tensor=tensor), form)
    assert list(read) == list(given) == [row.name for row in tensor_rows(header, form)]
    assert all(np.array_equal(read[name], given[name]) for name in given)
    with pytest.raises(AssertionError):
        tensors_of(checkpoint(header, form, tensor=tensor))  # (read as a Llama's: another file)


def test_values_of_another_shape_and_rows_the_dtype_cannot_hold_are_refused():
    header = (64, 96, 2, 4, 2, 320, 16)
    with pytest.raises(ValueError, match="The tensor wq of this model is"):
        checkpoint(header, tensor=lambda row: np.zeros(row.shape[::-1] if row.name == "wq" else row.shape))
    # (the engine's own words, the dtype's: engine/dtypes.py)
    with pytest.raises(ValueError, match=DTYPES["ternary"].unsuited):
        checkpoint(header, dtype="ternary", tensor=uniform(0))
    with pytest.raises(KeyError):
        checkpoint(header, tensor=named({}))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_uniform_is_one_stream_over_the_file(dtype):
    """(as the made-up files of tests/test_architectures.py and tests/plan_facts.py always were: one draw for the file)"""
    header, form, _ = MODELS["lfm2"]
    data = checkpoint(header, form, dtype, tensor=uniform(6))
    count = (len(data) - 28) // np.dtype(dtype).itemsize
    assert data[28:] == np.random.default_rng(6).uniform(-0.2, 0.2, count).astype(dtype).tobytes()


@pytest.mark.parametrize("name", ["llama", "qwen2 and qwen3", "gpt2", "neox"])
def test_what_is_drawn_for_a_row_goes_by_its_name(name):
    """A bias is around 0 and the weight of a norm around 1, whatever their numbers among the vectors are: the makers
    this replaced counted them (GPT-2's norms the vectors 0, 6 and 10, a Qwen2's biases 3 to 5)."""
    header, form, _ = MODELS[name]
    vectors = [row for row in tensor_rows(header, form) if row.role == VECTOR]
    assert {row.name for row in vectors} & set(BIASES) == {"gpt2": set(BIASES), "neox": set(BIASES),
                                                           "qwen2 and qwen3": {"bq", "bk", "bv"}, "llama": set()}[name]
    for calm, spread in ((False, 0.03), (True, 0.3)):
        tensor = drawn(seed=1, outliers=3, spread=37, calm=calm)
        for row in tensor_rows(header, form):
            values = tensor(row)
            assert values.dtype == np.float32 and values.shape == tuple(row.shape)
            if row.role != VECTOR:
                # a layer's matrix is 0.1 N where the model is calm, every other matrix 0.3 N
                assert abs(values.std() - (0.1 if calm and row.per else 0.3)) < 0.03, row.name
            elif row.name == "rms_final_weight":
                outliers = np.arange(3) * 37 % header[0]
                assert (values[outliers] == 12.0).all() and np.abs(np.delete(values, outliers) - 1).max() < 6 * spread
            elif row.name in BIASES:
                assert abs(values.mean()) < 0.1 and abs(values.std() - 0.3) < 0.05, row.name
            else:
                assert abs(values.mean() - 1) < 3 * spread / np.sqrt(values.size) + 1e-3 and abs(values.std() - spread) < 0.2 * spread, row.name


def test_ternary_values_are_ternary_as_they_are_drawn():
    header, form, _ = MODELS["qwen2 and qwen3"]
    tensor = drawn(seed=2, ternary=True)
    for row in tensor_rows(header, form):
        values = tensor(row)
        if row.role == VECTOR:
            continue
        groups = np.abs(values.reshape(-1, 128))
        scales = groups.max(axis=1, keepdims=True)
        assert ((groups == 0) | (groups == scales)).all() and (scales > 0).all()
        assert 0.25 < (groups == 0).mean() < 0.42  # (about a third)


def test_a_quantized_model_and_its_tokenizer():
    data, tokenizer = quantized(layers=3, vocab=-320, dtype="int6", qk_norm=True, head_dim=32)
    header = struct.unpack_from("<7i", data, 0)
    assert header == (64, 128, 3, 4, 2, -320, 256)
    assert checkpoint_dtype(header, len(data), {"qk_norm": True, "head_dim": 32}) == "int6"
    pieces = Tokenizer(tokenizer, 320)
    assert pieces.vocab[0] == b"<0>" and pieces.vocab[319] == b"<319>"
    # one stream: the same seed is the same file, another seed another
    assert quantized(layers=3, vocab=-320, dtype="int6", qk_norm=True, head_dim=32)[0] == data
    assert quantized(layers=3, vocab=-320, dtype="int6", qk_norm=True, head_dim=32, seed=1)[0] != data


def test_a_header_and_the_settings_of_a_llama_are_each_others():
    for settings in ({}, {"n_kv_heads": 2, "shared": False}):
        config, _ = synthetic_weights(**settings)
        assert settings_of(header_of(config)) == config


@pytest.mark.parametrize("dtype", ["int8", "float32"])
def test_a_model_as_files_is_one_the_engine_opens(dtype, tmp_path, capsys):
    import json
    tensors, config = lfm2_model(dim=64, block_ff_dim=192, kinds="ccacca", vocab_size=64, seq_len=32)
    files(tmp_path / "model", tensors, config, dtype, 32, 64)
    assert f"model.bin: {(tmp_path / 'model.bin').stat().st_size} bytes ({dtype})" in capsys.readouterr().out
    options = json.loads((tmp_path / "model.json").read_text())
    assert options["arch"] == "lfm2" and options["dtype"] == dtype and json.loads(unigram(64))["model"]["vocab"][63] == ["w63", -63.0]
    llama = Llama((tmp_path / "model.bin").read_bytes(), (tmp_path / "model.tokenizer.bin").read_bytes(), **options)
    assert np.isfinite(llama.forward(1, 0)).all()


def test_the_maker_needs_nothing_of_the_test_suite():
    """A tool (tests/make_*.py, tests/plan_facts.py) and a Pyodide (tests/gpu-check.mjs) import it with the engine's
    Python on the path and nothing else: no pytest, no conftest.py."""
    from tree import python_folder
    said = subprocess.run([sys.executable, "-c", f"""
import sys
sys.path[:0] = [{str(ROOT / "tests")!r}, {python_folder(ROOT)!r}]
import made_up
print(sorted(name for name in ("pytest", "conftest", "llama2_convert") if name in sys.modules))
data, tokenizer = made_up.quantized()
print(len(data), len(tokenizer))
"""], capture_output=True, text=True, check=True).stdout.split("\n")
    assert said[0] == "[]" and said[1].split() == [str(len(quantized()[0])), str(len(quantized()[1]))]
