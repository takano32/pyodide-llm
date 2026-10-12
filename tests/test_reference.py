"""tests/reference.py (T357.4): what the three tools that hold the engine to transformers share, a part at a time and
with no torch: how far logits are apart, the line and the verdict, the states of layers of several kinds, the long
text, the conversions, and the lines each says. And what the tools hand it: their kinds of states, read from a made-up
engine. The tools themselves run from end to end in tests/reference_dry.py, and against transformers in CI."""
import ast
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

import made_up
import reference as R
import reference_lfm2
import reference_llama
import reference_qwen35
from conducting import Directory
from llama2_convert import Conversion
from llama2_numpy import Llama

HERE = Path(__file__).resolve().parent
TOOLS = (reference_llama, reference_qwen35, reference_lfm2)
TOKENS = [1, 5, 7, 9, 5, 11, 3, 3, 8, 2, 14, 15]


class Said(list):
    """a tool's say(): the lines, kept"""

    def __call__(self, *parts):
        self.append(" ".join(str(part) for part in parts))


# ------------------------------------------------------------------------------------------------ logits and lines
def test_differences_are_the_largest_the_mean_the_agreement_and_the_gap_where_they_disagree():
    theirs = np.array([[0.0, 1.0, 3.0], [5.0, 4.0, 0.0], [1.0, 2.0, 2.5]])
    ours = theirs + np.array([[0.1, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, -1.0]])
    largest, mean, same, margin = R.differences(ours, theirs)
    assert largest == pytest.approx(1.5) and mean == pytest.approx(2.6 / 9) and same == 1
    assert margin == pytest.approx(1.0), "the largest gap of the reference's best two among the positions that differ (1.0 and 0.5)"
    assert R.differences(theirs, theirs) == (0.0, 0.0, 3, 0.0)
    assert R.relative([3.0, 4.0], [3.0, 0.0]) == pytest.approx(4 / 3) and R.relative([0.0], [0.0]) == 0.0


def test_the_line_is_ten_times_the_floor_and_no_less_than_the_least():
    assert R.line_of(0.0, 1e-3) == 1e-3 and R.line_of(5e-5, 1e-3) == 1e-3
    assert R.line_of(3e-4, 1e-3) == pytest.approx(3e-3) and R.line_of(3e-4, 2e-3) == pytest.approx(3e-3)


@pytest.mark.parametrize("largest, same, margin, held", [
    (5e-4, 10, 0.0, True),        # within the line, every token the same
    (2e-3, 10, 0.0, False),       # past the line
    (1e-3, 10, 0.0, True),        # on the line
    (5e-4, 9, 9e-4, True),        # another token where the reference's best two are closer than twice the difference
    (5e-4, 9, 2e-3, False),       # another token where they are not
    (float("nan"), 10, 0.0, False),  # a NaN is past every line
])
def test_within_the_line(largest, same, margin, held):
    assert R.within(largest, same, 10, margin, 1e-3) is held


def test_the_engines_logits_are_copies_token_by_token_from_position_zero():
    class Engine:
        def __init__(self):
            self.buffer, self.asked = np.zeros(3), []

        def forward(self, token, pos):
            self.asked.append((token, pos))
            self.buffer[:] = token
            return self.buffer

    engine = Engine()
    out = R.engine_logits(engine, [4, 6, 9])
    assert engine.asked == [(4, 0), (6, 1), (9, 2)] and [row[0] for row in out] == [4, 6, 9]


def test_the_line_about_logits_says_what_was_found_and_fails_past_the_line():
    theirs = np.array([[0.0, 1.0, 3.0], [5.0, 4.0, 0.0]])
    say = Said()
    assert R.logits_failed(say, "t: the engine against x", theirs + 5e-4, theirs, 1e-3) is False
    assert say == ["t: the engine against x: largest difference 5.00e-04, mean 5.00e-04, the same most likely token at 2 of 2 positions"]
    off = theirs.copy()
    off[1, 1] += 2.0
    say = Said()
    assert R.logits_failed(say, "t: x", off, theirs, 1e-3, gap=True) is True
    assert say == ["t: x: largest difference 2.00e+00, mean 3.33e-01, the same most likely token at 1 of 2 positions (the largest gap "
                   "between the reference's best two where it is not: 1.00e+00) — FAILED (the line: 1.0e-03)"]
    broken = theirs.copy()
    broken[0, 0] = np.nan
    assert R.logits_failed(Said(), "t", broken, theirs, 1e3) is True, "a NaN is past any line"


def test_the_small_lines(monkeypatch):
    say = Said()
    R.logits_said(say, "real", np.array([[1.0, 3.0]]), [np.array([2.0, 2.0])])
    assert say == ["real: logits of transformers: min 1.000, max 3.000, mean 2.0000, std 1.0000",
                   "real: logits of the engine: min 2.000, max 2.000, mean 2.0000, std 0.0000"]
    assert R.options_shown({"arch": "llama", "specials": ["a", "b"]}) == '{"arch": "llama", "specials": "2 of them"}'
    monkeypatch.setattr(sys, "argv", ["tool.py", "dir", "--only=made-up", "--model=350M,230M", "--weak"])
    assert R.option("only") == "made-up" and R.option("model") == "350M,230M" and R.option("positions", 96) == 96
    assert R.option("weak") is None, "a flag without a value is not an option"
    for failed, line, code in ((False, "the engine computes what transformers computes", 0), (True, "FAILED", 1)):
        say = Said()
        with pytest.raises(SystemExit) as left:
            R.ended(say, failed)
        assert left.value.code == code and say == [line]


def test_a_file_is_fetched_from_the_repository_at_the_revision(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(R, "download", lambda url, target, optional=False: asked.append((url, target, optional)) or "it")
    assert R.fetch("config.json", tmp_path, "owner/name", "abc") == "it" and R.fetch("x.bin", tmp_path, "o/n", "r", optional=True) == "it"
    assert asked == [("https://huggingface.co/owner/name/resolve/abc/config.json", tmp_path / "config.json", False),
                     ("https://huggingface.co/o/n/resolve/r/x.bin", tmp_path / "x.bin", True)]


# ------------------------------------------------------------------------------------------------------ the states
def states(least=1e-4):
    """two kinds of layers, and an engine that is its own states"""
    return R.States({"round": ("turns",), "square": ("sides", "corners")}, lambda llama, count: llama(count), None, least)


def held(scale=1.0, kinds=("round", "square", "square")):
    one = {"round": [np.arange(1.0, 5.0)], "square": [np.ones((2, 3)), np.arange(6.0).reshape(2, 3) + 1]}
    return {layer: (kind, [array * scale for array in one[kind]]) for layer, kind in enumerate(kinds)}


def test_states_are_the_largest_relative_distance_of_each_name_over_the_layers():
    kept = states()
    assert kept.names == ("turns", "sides", "corners")
    assert kept.distances(held(), held()) == {"turns": 0.0, "sides": 0.0, "corners": 0.0}
    off = held()
    off[2][1][1] = off[2][1][1] * 1.01  # the corners of the second square layer alone
    apart = kept.distances(off, held())
    assert apart["turns"] == 0 and apart["sides"] == 0 and apart["corners"] == pytest.approx(0.01)
    assert kept.distances(held(1.5), held()) == pytest.approx({"turns": 0.5, "sides": 0.5, "corners": 0.5})


def test_a_layer_of_another_kind_and_a_state_that_is_not_a_number_are_infinitely_far():
    kept = states()
    apart = kept.distances(held(kinds=("round", "square", "round")), held())
    assert apart == {"turns": float("inf"), "sides": 0.0, "corners": 0.0}, "the names of the kind the engine says the layer is"
    for where in (0, 2):  # in the first layer of its name and in the last: neither is let by
        broken = held()
        broken[where][1][0] = np.full_like(broken[where][1][0], np.nan)
        name = "turns" if where == 0 else "sides"
        apart = kept.distances(broken, held())
        assert apart[name] == float("inf") and kept.past(apart, kept.lines(dict.fromkeys(kept.names, 0.0))) == [name]


def test_each_state_has_its_own_line_from_its_own_floor():
    kept = states()
    lines = kept.lines({"turns": 0.0, "sides": 5e-5, "corners": 1e-6})
    assert lines == pytest.approx({"turns": 1e-4, "sides": 5e-4, "corners": 1e-4})
    assert kept.past({"turns": 1e-4, "sides": 6e-4, "corners": float("nan")}, lines) == ["sides", "corners"]
    assert states(least=1e-2).lines({"turns": 0.0, "sides": 5e-5, "corners": 0.0})["sides"] == 1e-2


def test_the_line_about_states_says_each_and_fails_where_one_is_past_its_line():
    kept = states()
    say = Said()
    asked = []
    engine = lambda count: asked.append(count) or held(1 + 2e-5)
    assert kept.failed(say, "real", engine, 7, held(1 + 1e-6), held()) is False and asked == [7]
    assert say == ["real: the states after 7 positions against transformers': turns 2.0e-05 (transformers' own 1.0e-06, line 1.0e-04); "
                   "sides 2.0e-05 (transformers' own 1.0e-06, line 1.0e-04); corners 2.0e-05 (transformers' own 1.0e-06, line 1.0e-04)"]
    say = Said()
    wrong = held()
    wrong[0][1][0] = wrong[0][1][0] * 1.01
    assert kept.failed(say, "real", lambda count: wrong, 7, held(1 + 1e-4), held()) is True
    assert say == ["real: the states after 7 positions against transformers': turns 1.0e-02 (transformers' own 1.0e-04, line 1.0e-03) — FAILED; "
                   "sides 0.0e+00 (transformers' own 1.0e-04, line 1.0e-03); corners 0.0e+00 (transformers' own 1.0e-04, line 1.0e-03)"]


# --------------------------------------------------------------------------------------------------- the long text
class Tokenized:
    def __init__(self, ids):
        self.ids = ids


class Words:
    """a tokenizer of words: an id a word, by its length"""

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        self.text = text
        return Tokenized([len(word) for word in text.split()])


class Counting:
    """an engine whose logits are its position's, and whose states are what it is told"""

    def __init__(self, off=0.0):
        self.off, self.ran = off, []

    def forward(self, token, pos):
        self.ran.append(pos)
        self.buffer = np.array([float(pos), 2.0 * pos + 1 + self.off, 0.5])
        return self.buffer


def test_the_long_text_is_twelve_sections_after_the_bos_cut_at_long(monkeypatch):
    tokenizer = Words()
    ids, at = R.long_ids(tokenizer, 99, (0, 3, 600, 1023, 5000))
    assert tokenizer.text.startswith("Section 0. Mount Fuji") and tokenizer.text.count(R.TEXT) == 12 and "Section 11. " in tokenizer.text
    assert ids[0] == 99 and ids[1:4] == [7, 2, 5] and len(ids) == len(tokenizer.text.split()) + 1 < R.LONG
    assert at == [0, 3, 600], "only the positions the text has"
    monkeypatch.setattr(R, "LONG", 40)
    ids, at = R.long_ids(Words(), 99, (0, 3, 39, 40))
    assert len(ids) == 40 and at == [0, 3, 39]


def test_the_long_run_holds_the_logits_at_some_positions_and_the_states_at_the_end():
    kept = states()
    ids, at = list(range(50)), [0, 3, 49]
    theirs = np.array([[float(pos), 2.0 * pos + 1, 0.5] for pos in at])
    engine, say = Counting(), Said()
    kept.of_engine = lambda llama, count: held(1 + 5e-5)
    assert R.long_failed(say, kept, engine, ids, at, theirs, held(), 1e-3) is False and engine.ran == ids
    assert len(say) == 1 and say[0].startswith("long: the engine ran 50 positions in ")
    assert say[0].endswith(" s; the logits at 3 of them against transformers': largest difference 0.00e+00, mean 0.00e+00, the same most "
                           "likely token at 3 of 3; the states after them: turns 5.0e-05; sides 5.0e-05; corners 5.0e-05")
    say = Said()
    assert R.long_failed(say, kept, Counting(off=0.5), ids, at, theirs, held(), 1e-3) is True
    assert "largest difference 5.00e-01" in say[0] and "of 3 — FAILED (the line: 1.0e-03); the states" in say[0] and say[0].endswith("5.0e-05")
    # the states are held to the least alone: there is no floor here
    kept.of_engine = lambda llama, count: held(1 + 5e-4, kinds=("round", "square", "square"))
    say = Said()
    assert R.long_failed(say, kept, Counting(), ids, at, theirs, held(), 1e-3) is True
    assert say[0].endswith("turns 5.0e-04; sides 5.0e-04; corners 5.0e-04 — FAILED (the line: 1e-04): turns, sides, corners")


# ----------------------------------------------------------------------------------------------------- conversions
def test_a_made_up_model_is_converted_as_the_page_converts_and_the_template_is_not_an_option():
    tensors, published = made_up.lfm2_model(seq_len=256)
    llama, options, checkpoint = R.made_up_engine(tensors, published)
    file = made_up.safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), made_up.unigram(320), "tokenizer.json",
                            dtype="float32", max_seq_len=256, start=8 + size)
    conversion.feed(file[8 + size:])
    conversion.finish()
    assert checkpoint == bytes(conversion.checkpoint) and isinstance(checkpoint, bytes)
    assert options == {key: value for key, value in conversion.options.items() if key != "template"} and options["arch"] == "lfm2"
    other = Llama(checkpoint, conversion.tokenizer, **options)
    assert np.array_equal(R.engine_logits(llama, TOKENS), R.engine_logits(other, TOKENS))
    assert llama.vocab_size == 320 and R.made_up_engine(*made_up.lfm2_model(seq_len=256, vocab_size=96), 96)[0].vocab_size == 96


def test_a_real_model_is_converted_into_the_file_and_the_file_is_let_go(tmp_path):
    tensors, published = made_up.lfm2_model(seq_len=256)
    (tmp_path / "config.json").write_text(json.dumps(published))
    (tmp_path / "tokenizer.json").write_bytes(made_up.unigram(320))
    (tmp_path / "model.safetensors").write_bytes(made_up.safetensors_file(tensors))
    path = tmp_path / "float32.bin"
    conversion, seconds = R.float32_file(Directory(tmp_path), {"weights": "model.safetensors", "tokenizer": "tokenizer.json"}, path)
    assert seconds >= 0 and conversion.options["arch"] == "lfm2"
    assert path.read_bytes() == R.made_up_engine(tensors, published)[2], "the checkpoint, whole, in the file"
    path.unlink()  # (nothing of the tool's holds it)


# ------------------------------------------------------------------------------------------ what the tools hand it
def test_no_tool_binds_a_name_it_takes_from_the_skeleton():
    """A local of the same name would stand in front of the function, silently (the tools say `line`, `shown`, `reference`,
    `theirs` for their own things): the names taken are never assigned, never an argument, never a loop's."""
    taken_by_all = set()
    for tool in TOOLS:
        tree = ast.parse(Path(tool.__file__).read_text())
        taken = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "reference"
                 for alias in node.names}
        assert taken and all(hasattr(R, name) for name in taken)
        bound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))}
        bound |= {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        bound |= {argument.arg for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.Lambda))
                  for argument in node.args.args + node.args.kwonlyargs}
        assert not taken & bound, f"{Path(tool.__file__).name} binds {sorted(taken & bound)}, which it takes from reference.py"
        assert not any(isinstance(node, ast.Import) and any(alias.name == "reference" for alias in node.names) for node in ast.walk(tree)), \
            "the module itself is not imported: the tools have locals called reference"
        taken_by_all |= taken
    public = {name for name, value in vars(R).items() if getattr(value, "__module__", None) == "reference"} | {"TEXT", "LONG"}
    assert public - taken_by_all <= {"relative", "within", "LONG"}, "what no tool takes is only what the skeleton itself uses"


def test_a_qwen35s_states_are_read_from_the_engine_kind_by_kind():
    tensors, published = made_up.qwen35_model(n_layers=4, every=2, seq_len=256)
    llama, _, _ = R.made_up_engine(tensors, published)
    R.engine_logits(llama, TOKENS)
    kept = reference_qwen35.HELD
    assert kept.names == ("recurrent state", "convolution rows", "keys", "values") and kept.least == 1e-4
    ours = kept.of_engine(llama, len(TOKENS))
    assert [kind for kind, _ in ours.values()] == ["linear", "full", "linear", "full"]
    assert all(len(arrays) == len(kept.kinds[kind]) for kind, arrays in ours.values())
    assert ours[1][1][0].shape[1] == len(TOKENS) and ours[1][1][1].shape[1] == len(TOKENS), "keys and values of the positions run"
    theirs = {layer: (kind, [np.array(array) for array in arrays]) for layer, (kind, arrays) in ours.items()}
    assert kept.apart(llama, len(TOKENS), theirs) == dict.fromkeys(kept.names, 0.0)
    llama.delta_state[1] *= 1.001  # the second linear layer's recurrent state alone
    apart = kept.apart(llama, len(TOKENS), theirs)
    assert apart["recurrent state"] == pytest.approx(1e-3, rel=1e-2) and apart["keys"] == 0 and apart["convolution rows"] == 0


def test_an_lfm2s_states_are_read_from_the_engine_kind_by_kind():
    tensors, published = made_up.lfm2_model(kinds="ccaca", seq_len=256)
    llama, _, _ = R.made_up_engine(tensors, published)
    R.engine_logits(llama, TOKENS)
    kept = reference_lfm2.HELD
    assert kept.names == ("convolution rows", "keys", "values") and kept.least == 1e-4
    ours = kept.of_engine(llama, len(TOKENS))
    assert [kind for kind, _ in ours.values()] == ["conv", "conv", "full", "conv", "full"]
    theirs = {layer: (kind, [np.array(array) for array in arrays]) for layer, (kind, arrays) in ours.items()}
    assert kept.apart(llama, len(TOKENS), theirs) == dict.fromkeys(kept.names, 0.0)
    llama.value_cache[1] *= 1.01  # the second attention layer's values alone
    apart = kept.apart(llama, len(TOKENS), theirs)
    assert apart["values"] == pytest.approx(1e-2, rel=1e-2) and apart["keys"] == 0 and apart["convolution rows"] == 0
    # the fault that changes the kinds of two layers is infinitely far, whatever its numbers
    other, _, _ = R.made_up_engine(*made_up.lfm2_model(kinds="cacca", seq_len=256))
    R.engine_logits(other, TOKENS)
    assert set(kept.apart(other, len(TOKENS), theirs).values()) == {float("inf")}


def smollm3():
    settings, weights = made_up.synthetic_weights(dim=64, hidden_dim=96, n_layers=8, n_heads=4, n_kv_heads=2, seq_len=64)
    tensors, published = made_up.hugging_face(settings, weights, True)
    return R.made_up_engine(tensors, {**published, "model_type": "smollm3", "no_rope_layer_interval": 4, "use_sliding_window": False})[0]


def test_the_faults_of_the_layers_rope_leaves_alone_are_all_seen_and_the_layers_are_put_back(capsys):
    llama = smollm3()
    assert llama.unturned == (3, 7)
    whole = np.array(R.engine_logits(llama, TOKENS))
    theirs = reference_llama.engine_cache(llama, len(TOKENS))
    assert reference_llama.unturned_faults("t", llama, TOKENS, whole, theirs, 1e-3, 1e-4) == []
    assert llama.unturned == (3, 7) and np.array_equal(np.array(R.engine_logits(llama, TOKENS)), whole)
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(": logits")[0] for line in lines] == ["reference: t: a fault, every layer turned (read as a Llama)",
                                                             "reference: t: a fault, layer 7 turned",
                                                             "reference: t: a fault, layer 0 left alone too"]
    assert all(" of the line), mean " in line and "seen by" in line and "FAILED" not in line for line in lines)
    # a line that lets everything by sees none of them, and says so of each: all three must be seen
    missed = reference_llama.unturned_faults("t", llama, TOKENS, whole, theirs, 1e3, 1e3)
    lines = capsys.readouterr().out.splitlines()
    assert len(missed) == 3 and all(line.endswith("seen by neither — FAILED") for line in lines)


def test_a_fault_the_check_need_not_see_is_said_and_not_counted(capsys):
    llama = smollm3()
    whole = np.array(R.engine_logits(llama, TOKENS))
    theirs = reference_llama.engine_cache(llama, len(TOKENS))
    nothing = lambda: (lambda: None)
    missed = reference_llama.faults_seen("t", llama, TOKENS, whole, theirs, 1e-3, 1e-4,
                                         [("nothing wrong, and it need not be seen", nothing, False), ("nothing wrong, and it must", nothing, True)])
    lines = capsys.readouterr().out.splitlines()
    assert missed == ["nothing wrong, and it must"]
    assert lines[0].endswith("seen by neither") and lines[1].endswith("seen by neither — FAILED")

    def broken():
        llama.forward = lambda token, pos: 1 / 0
        return lambda: delattr(llama, "forward")

    with pytest.raises(ZeroDivisionError):  # what takes the fault out runs whatever the engine does with the fault in it
        reference_llama.faults_seen("t", llama, TOKENS, whole, theirs, 1e-3, 1e-4, [("x", broken, True)])
    assert np.array_equal(np.array(R.engine_logits(llama, TOKENS)), whole)
