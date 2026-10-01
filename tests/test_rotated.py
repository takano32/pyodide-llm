"""The rotated basis (T237): Ternary Bonsai 2 27B's matrices are stored as W R^-1 with R = H S (signs, then the
normalized Walsh-Hadamard transform of every block), and the forward pass multiplies them by R x; the embedding's rows
are stored as R e and turned back. The defining property is that nothing else changes: a model folded into the basis
by hand computes what the model computed before.

The reference is the same model in its own basis (held to transformers' computation by test_qwen35.py and to the
naive loops by the others), and for the transform its matrix, entry by entry."""
import json
import struct

import numpy as np
import pytest
from conftest import FOLDED, basis, folded, naive_logits, naive_qwen35_logits, qwen35_model, synthetic_weights
from test_convert import converted, hugging_face, reader, safetensors_file
from test_gguf import fed, qwen35_gguf, safetensors_conversion, unigram, with_original
from test_qwen3 import qwen3

import llama2_convert
from llama2_convert import ROTATED, Conversion, Safetensors, checkpoint_form, gguf_rotated, normalize
from llama2_numpy import FORM, Llama, form_of, hadamard, rotate, rotated_form, rotated_widths, sign_bits, unrotate

TOKENS = [1, 5, 7, 9, 11, 5, 5, 300, 2]

def widths_of(config):
    config = normalize(config)
    form = checkpoint_form(config, {})
    q_dim = llama2_convert.head_size(config) * config["num_attention_heads"]
    return rotated_widths(config["hidden_size"], q_dim, config["intermediate_size"], form["linear"])


# ------------------------------------------------------------------------------------------- the transform
@pytest.mark.parametrize("block", [1, 2, 4, 8, 16, 64, 1024])
def test_the_transform_is_the_normalized_sylvester_matrix(block):
    """Entry (i, j) is (-1) ** popcount(i & j) / sqrt(block) (the fork's llama-model.cpp, 2054 to 2065), block after
    block; it is its own inverse and keeps a vector's length."""
    index = np.arange(block)
    parity = np.array([[bin(i & j).count("1") & 1 for j in index] for i in index])
    matrix = (1 - 2 * parity) / np.sqrt(block)
    rng = np.random.default_rng(block)
    x = rng.standard_normal((3, 2 * block)).astype(np.float32)
    want = np.concatenate([x[:, :block].astype(np.float64) @ matrix.T, x[:, block:].astype(np.float64) @ matrix.T], axis=1)
    got = hadamard(x, block)
    assert got.dtype == np.float32 and got.shape == x.shape
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)
    assert np.allclose(hadamard(got, block), x, rtol=1e-5, atol=1e-5)
    assert np.allclose((got * got).sum(axis=1), (x * x).sum(axis=1), rtol=1e-4)


def test_rotating_is_the_signs_then_the_transform_and_turning_back_the_other_way_round():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(48).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], 48).astype(np.float32)
    assert np.array_equal(rotate(x, signs, 16), hadamard(x * signs, 16))
    assert np.array_equal(unrotate(x, signs, 16), hadamard(x, 16) * signs)
    assert np.allclose(unrotate(rotate(x, signs, 16), signs, 16), x, rtol=1e-5, atol=1e-6)
    # the property everything rests on: a row stored as R w, times R x, is w . x
    w = rng.standard_normal(48).astype(np.float32)
    assert np.isclose(rotate(w, signs, 16) @ rotate(x, signs, 16), w @ x, rtol=1e-4)
    # and none of the three is a no-op here: other signs, no signs and another block are other numbers
    assert not np.allclose(rotate(x, signs, 16), rotate(x, np.ones(48, dtype=np.float32), 16), atol=1e-3)
    assert not np.allclose(rotate(x, signs, 16), rotate(x, signs, 8), atol=1e-3)
    assert not np.allclose(rotate(x, signs, 16), unrotate(x, signs, 16), atol=1e-3)


def test_the_signs_as_the_options_carry_them():
    said, signs = basis(8, [16, 24, 40])
    assert json.loads(json.dumps(said)) == said  # plain JSON: a manifest keeps it
    form = rotated_form(said, [16, 24, 40])
    assert form["block"] == 8 and sorted(form["signs"]) == [16, 24, 40]
    for width, values in signs.items():
        assert form["signs"][width].dtype == np.float32 and np.array_equal(form["signs"][width], values)
    assert sign_bits([1, -1, -1, 1, 1, 1, 1, -1, -1]) == "6180"
    assert rotated_form(None, [16]) is None


@pytest.mark.parametrize("change, what", [
    (lambda said: said.update(block=12), "power of two"),
    (lambda said: said.update(block=0), "power of two"),
    (lambda said: said.update(block=32), "whole blocks"),  # 16 and 24 are no multiples of 32
    (lambda said: said["signs"].pop("24"), "no signs for an input 24 wide"),
    (lambda said: said["signs"].update({"24": said["signs"]["24"] + "00"}), "whole blocks"),
    (lambda said: said["signs"].update({"24": said["signs"]["24"][:-2]}), "whole blocks"),
])
def test_a_basis_that_does_not_fit_the_model_is_refused(change, what):
    said, _ = basis(8, [16, 24, 40])
    change(said)
    with pytest.raises(ValueError, match=what):
        rotated_form(said, [16, 24, 40])


# ------------------------------------------------------------------------------------------- the forward pass
def conversion(tensors, config, said, dtype="float32", vocab_size=None, max_seq_len=1 << 20):
    """The page's way: a file that says its basis in its header, in its own order, and the options that come out."""
    file = safetensors_file(tensors, metadata={ROTATED: json.dumps(said)} if said else None)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(vocab_size)]}}).encode()
    made = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), vocabulary, "tokenizer.json", dtype=dtype,
                      max_seq_len=max_seq_len, start=8 + size)
    made.stream.feed(file[8 + size:])
    made.stream.finish()
    return made


def run(made, vocab_size, tokens=TOKENS):
    """The logits of the engine made of a conversion's own checkpoint, tokenizer and options, token after token."""
    options = {key: value for key, value in made.options.items() if key != "template"}
    llama = Llama(bytes(made.checkpoint), made.tokenizer, **options)
    return [llama.forward(token, pos).copy() for pos, token in enumerate(tokens)]


# a block that is the whole of the narrowest input and one of several blocks of every input; value heads on their own
# key heads and several to one; a classifier of its own and the embedding's; every second and every fourth layer
QWEN35 = {
    "blocks of 8": (dict(), 8),
    "blocks of 4, a classifier of its own": (dict(shared=False), 4),
    "blocks of 2, three value heads a key head": (dict(key_heads=1, value_heads=3, key_dim=4, value_dim=10, shared=False), 2),
    "blocks of 16, every fourth layer attends": (dict(n_layers=8, every=4, n_kv_heads=4, head_dim=8, rotary=0.5, value_dim=8), 16),
    "blocks of 1: the signs alone": (dict(), 1),
}


@pytest.mark.parametrize("model", QWEN35)
def test_a_qwen35_folded_into_the_basis_computes_what_it_computed(model):
    shape, block = QWEN35[model]
    tensors, config = qwen35_model(**shape)
    vocab_size = config["text_config"]["vocab_size"]
    said, signs = basis(block, widths_of(config))
    plain = conversion(tensors, config, None, vocab_size=vocab_size)
    turned = conversion(folded(tensors, block, signs), config, said, vocab_size=vocab_size)
    assert "rotated" not in plain.options and turned.options["rotated"] == said
    assert {key: value for key, value in turned.options.items() if key != "rotated"} == plain.options
    # the same tensors in the same places: only the numbers of the folded matrices differ
    assert len(turned.checkpoint) == len(plain.checkpoint) and bytes(turned.checkpoint) != bytes(plain.checkpoint)
    want = naive_qwen35_logits(tensors, config, TOKENS)
    got, before = run(turned, vocab_size), run(plain, vocab_size)
    for pos in range(len(TOKENS)):
        assert np.allclose(got[pos], want[pos], rtol=2e-4, atol=2e-4), pos
        assert np.allclose(got[pos], before[pos], rtol=2e-4, atol=2e-4), pos


LLAMAS = {"a llama": (dict(n_kv_heads=4), False, 8), "grouped keys, a classifier of its own": (dict(n_kv_heads=2, shared=False), False, 16),
          "a qwen3 with wide heads": (dict(n_kv_heads=2, head_size=16), True, 32), "a qwen3 with narrow heads": (dict(n_kv_heads=4, head_size=4, shared=False), True, 4)}


@pytest.mark.parametrize("model", LLAMAS)
def test_a_llama_or_a_qwen3_folded_into_the_basis_computes_what_it_computed(model):
    """The fork rotates a Llama's and a Qwen3's matrices the same way (its loader's list of architectures)."""
    shape, heads_normed, block = LLAMAS[model]
    shape = dict(shape)
    shared = shape.pop("shared", True)
    settings, weights = synthetic_weights(shared=shared, **shape)
    tensors, config = (qwen3 if heads_normed else hugging_face)(settings, weights, shared)
    said, signs = basis(block, widths_of(config))
    turned = conversion(folded(tensors, block, signs), config, said, vocab_size=settings["vocab_size"])
    assert turned.options["rotated"] == said
    want = naive_logits(settings, weights, TOKENS[:5])
    for pos, logits in enumerate(run(turned, settings["vocab_size"], TOKENS[:5])):
        assert np.allclose(logits, want[pos], rtol=2e-4, atol=2e-4), pos


def test_the_basis_is_needed_and_every_part_of_it():
    """The folded file read as if in its own basis, with other signs, with another block, and the file of the model's
    own basis read as rotated: all other numbers. (What a wrong reading would be; the engine has no way to tell.)"""
    tensors, config = qwen35_model()
    vocab_size = config["text_config"]["vocab_size"]
    said, signs = basis(8, widths_of(config))
    turned = conversion(folded(tensors, 8, signs), config, said, vocab_size=vocab_size)
    want = run(turned, vocab_size)[-1]
    other_signs, _ = basis(8, widths_of(config), seed=6)
    for name, options in (("no basis", {"rotated": None}), ("other signs", {"rotated": other_signs}),
                          ("blocks of 4", {"rotated": {**said, "block": 4}}), ("blocks of 1", {"rotated": {**said, "block": 1}})):
        wrong = Conversion.__new__(Conversion)
        wrong.checkpoint, wrong.tokenizer, wrong.options = turned.checkpoint, turned.tokenizer, {**turned.options, **options}
        assert not np.allclose(run(wrong, vocab_size)[-1], want, rtol=1e-2, atol=1e-2), name
    plain = conversion(tensors, config, None, vocab_size=vocab_size)
    plain.options["rotated"] = said
    assert not np.allclose(run(plain, vocab_size)[-1], want, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("kind", [kind for kind in FOLDED if kind != "lm_head.weight"])
def test_every_matrix_reads_a_rotated_input(kind):
    """One kind of matrix left in the model's own basis in a file that says it is rotated: other numbers, so the
    forward pass rotates what that matrix reads (and turns the embedding's row back)."""
    tensors, config = qwen35_model(shared=False)
    vocab_size = config["text_config"]["vocab_size"]
    said, signs = basis(8, widths_of(config))
    whole = folded(tensors, 8, signs)
    want = run(conversion(whole, config, said, vocab_size=vocab_size), vocab_size)[-1]
    partly = {name: tensors[name] if name.endswith(kind) else tensor for name, tensor in whole.items()}
    assert not np.allclose(run(conversion(partly, config, said, vocab_size=vocab_size), vocab_size)[-1], want, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("kind", ["linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"])
def test_the_gates_of_a_linear_layer_read_the_model_s_own_basis(kind):
    """The two small matrices of a linear-attention layer's gates are not rotated in the 27B's file (they are not in
    its prism.hadamard.weight_names): folding them too gives other numbers."""
    tensors, config = qwen35_model()
    vocab_size = config["text_config"]["vocab_size"]
    said, signs = basis(8, widths_of(config))
    whole = folded(tensors, 8, signs)
    want = run(conversion(whole, config, said, vocab_size=vocab_size), vocab_size)[-1]
    more = {name: rotate(tensor, signs[tensor.shape[-1]], 8) if name.endswith(kind) else tensor for name, tensor in whole.items()}
    assert not np.allclose(run(conversion(more, config, said, vocab_size=vocab_size), vocab_size)[-1], want, rtol=1e-2, atol=1e-2)


# ------------------------------------------------------------------------------------------- the converter
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8", "int6"])
def test_the_basis_moves_no_byte_of_the_file_and_no_other_option(dtype):
    """What the header's __metadata__ says of the basis goes into the options and nowhere else: the checkpoint is the
    one of the same tensors without it, in every dtype, and from the file in its own order as from its tensors."""
    tensors, config = qwen35_model(dim=64, hidden_dim=128, n_heads=4, n_kv_heads=2, head_dim=32, key_heads=2,
                                   value_heads=4, key_dim=16, value_dim=16)
    vocab_size = config["text_config"]["vocab_size"]
    said, signs = basis(32, widths_of(config))
    there = folded(tensors, 32, signs)
    with_it = conversion(there, config, said, dtype, vocab_size)
    without = conversion(there, config, None, dtype, vocab_size)
    assert bytes(with_it.checkpoint) == bytes(without.checkpoint)
    assert with_it.options == {**without.options, "rotated": said}
    assert form_of(with_it.options)["rotated"] == said and form_of(without.options) == {**FORM, **{
        key: without.options[key] for key in FORM if key in without.options}}
    source = Safetensors(reader(safetensors_file(there, metadata={ROTATED: json.dumps(said)})))
    assert source.rotated == said and checkpoint_form(config, source)["rotated"] == said
    assert converted(source, config, dtype) == bytes(with_it.checkpoint)


def test_a_file_that_says_no_basis_has_none_in_its_options():
    """A saved conversion is used again only while its options are what the converter makes now: no key for the models
    there were before."""
    tensors, config = qwen35_model()
    assert "rotated" not in conversion(tensors, config, None, vocab_size=config["text_config"]["vocab_size"]).options
    assert Safetensors(reader(safetensors_file(tensors))).rotated is None


@pytest.mark.parametrize("said, what", [
    ({"block": 8, "signs": {}}, "no signs for an input"),
    ({"block": 64, "signs": {"32": "00" * 4, "64": "00" * 8, "24": "00" * 3}}, "whole blocks"),
    ("not JSON", "not said in a way"), ({"blocks": 8}, "not said in a way"),
])
def test_a_file_whose_basis_does_not_fit_its_model_is_refused(said, what):
    tensors, config = qwen35_model()
    with pytest.raises(ValueError, match=what):
        file = safetensors_file(tensors, metadata={ROTATED: said if isinstance(said, str) else json.dumps(said)})
        size = struct.unpack("<Q", file[:8])[0]
        Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), unigram(config["text_config"]["vocab_size"]),
                   "tokenizer.json", dtype="float32", start=8 + size)


def test_a_gpt2_in_a_rotated_basis_is_refused():
    class Source(dict):
        rotated = {"block": 8, "signs": {"64": "00" * 8}}

    with pytest.raises(ValueError, match="GPT-2"):
        checkpoint_form({"model_type": "gpt2", "n_embd": 64, "n_layer": 2, "n_head": 4, "n_positions": 32, "vocab_size": 100}, Source())
    with pytest.raises(ValueError, match="GPT-2"):
        Llama(struct.pack("<7i", 64, 64, 2, 4, 4, 100, 32), None, arch="gpt2", rotated=Source.rotated)


# ------------------------------------------------------------------------------------------- a GGUF's metadata
def tensors_of(lines=(0, 1), full=(2,), own_classifier=True, dim=64, wide=96, hidden=128, vocab=40):
    """The table of a small Qwen3.5 GGUF (names, shapes outermost first), as gguf_read() gives it."""
    table = {"token_embd.weight": [vocab, dim], "output_norm.weight": [dim]}
    if own_classifier:
        table["output.weight"] = [vocab, dim]
    for layer in (*lines, *full):
        table.update({f"blk.{layer}.attn_norm.weight": [dim], f"blk.{layer}.post_attention_norm.weight": [dim],
                      f"blk.{layer}.ffn_gate.weight": [hidden, dim], f"blk.{layer}.ffn_up.weight": [hidden, dim],
                      f"blk.{layer}.ffn_down.weight": [dim, hidden]})
    for layer in lines:
        table.update({f"blk.{layer}.attn_qkv.weight": [wide + 64, dim], f"blk.{layer}.attn_gate.weight": [wide, dim],
                      f"blk.{layer}.ssm_out.weight": [dim, wide], f"blk.{layer}.ssm_alpha.weight": [6, dim],
                      f"blk.{layer}.ssm_beta.weight": [6, dim], f"blk.{layer}.ssm_conv1d.weight": [wide + 64, 4],
                      f"blk.{layer}.ssm_a": [6], f"blk.{layer}.ssm_dt.bias": [6], f"blk.{layer}.ssm_norm.weight": [16]})
    for layer in full:
        table.update({f"blk.{layer}.attn_q.weight": [2 * wide, dim], f"blk.{layer}.attn_k.weight": [32, dim],
                      f"blk.{layer}.attn_v.weight": [32, dim], f"blk.{layer}.attn_output.weight": [dim, wide],
                      f"blk.{layer}.attn_q_norm.weight": [16], f"blk.{layer}.attn_k_norm.weight": [16]})
    return {name: {"shape": shape, "type": 0, "offset": 0} for name, shape in table.items()}


def said_by(tensors, block=32, widths=(64, 96, 128), seed=9, **changes):
    """prism.hadamard.* as the 27B's GGUF says it, for these tensors: every matrix but the gates' small two rotated."""
    rng = np.random.default_rng(seed)
    kinds = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out", "ffn_gate", "ffn_up", "ffn_down")
    names = [name for name in tensors if name == "output.weight" or (name.startswith("blk.") and name.split(".")[2] in kinds)]
    values = {"version": 1, "block_size": block, "transform": "normalized-sylvester-walsh-hadamard",
              "axis": "input-last-dimension", "sign_mode": "explicit", "weight_names": names,
              "sign_widths": list(widths), "sign_values": [int(v) for v in rng.choice([-1, 1], sum(widths))],
              "inverse_weight_names": ["token_embd.weight"], "gdn_v_grouped": True, **changes}
    return {f"prism.hadamard.{key}": value for key, value in values.items() if value is not None}


def test_a_gguf_says_its_basis_as_the_27b_does():
    tensors = tensors_of()
    metadata = said_by(tensors)
    got = gguf_rotated(metadata, tensors, more_value_heads=True)
    values = np.array(metadata["prism.hadamard.sign_values"])
    assert got == {"block": 32, "signs": {"64": sign_bits(values[:64]), "96": sign_bits(values[64:160]),
                                          "128": sign_bits(values[160:])}}
    assert gguf_rotated({"general.architecture": "qwen35"}, tensors) is None
    # no signs: every one +1
    plain = gguf_rotated(said_by(tensors, sign_mode="identity", sign_widths=None, sign_values=None), tensors)
    assert plain == {"block": 32, "signs": {"64": "00" * 8, "96": "00" * 12, "128": "00" * 16}}
    # version 2: the embedding is the classifier too (no output.weight), and its name is in no list of matrices
    tied = tensors_of(own_classifier=False)
    assert gguf_rotated(said_by(tied, version=2, tied_output=True), tied)["block"] == 32


@pytest.mark.parametrize("changes, what", [
    (dict(version=3), "version 3"), (dict(version=2), "version 2"), (dict(tied_output=True), "tied output"),
    (dict(transform="fast-walsh"), "transform fast-walsh"), (dict(axis="output"), "along output"),
    (dict(block_size=24), "blocks of 24"), (dict(block_size=0), "blocks of 0"), (dict(block_size=64), "no signs in whole blocks of 64 for an input 96"),
    (dict(sign_mode="seeded"), "sign mode seeded"),
    (dict(sign_widths=[64, 96]), "signs are not"), (dict(sign_widths=[64, 128]), "signs are not"),
    (dict(sign_widths=[64, 96, 256], sign_values=[1] * 416), "no signs in whole blocks of 32 for an input 128"),
    (dict(sign_values=[0] * 288), "signs are not"), (dict(sign_values=[2] * 288), "signs are not"),
    (dict(inverse_weight_names=[]), "embedding"), (dict(inverse_weight_names=None), "embedding"),
    (dict(inverse_weight_names=["token_embd.weight", "blk.0.ffn_up.weight"]), "embedding"),
    (dict(gdn_v_grouped=False), "order of value heads"), (dict(gdn_v_grouped=None), "order of value heads"),
])
def test_a_basis_the_engine_does_not_compute_in_is_refused(changes, what):
    tensors = tensors_of()
    with pytest.raises(ValueError, match=what):
        gguf_rotated(said_by(tensors, **changes), tensors, more_value_heads=True)


def test_a_gguf_that_rotates_other_matrices_than_the_engine_does_is_refused():
    """The engine turns the input of every matrix it multiplies by: a file that leaves one in the model's own basis,
    or rotates one of the gates' small matrices, would run and write nonsense."""
    tensors = tensors_of()
    metadata = said_by(tensors)
    names = metadata["prism.hadamard.weight_names"]
    for left_out in ("output.weight", "blk.1.ssm_out.weight", "blk.2.attn_q.weight", "blk.0.ffn_down.weight"):
        with pytest.raises(ValueError, match=f"rotates not {left_out}"):
            gguf_rotated({**metadata, "prism.hadamard.weight_names": [name for name in names if name != left_out]}, tensors)
    for more in ("blk.0.ssm_alpha.weight", "token_embd.weight", "blk.9.ffn_up.weight"):
        with pytest.raises(ValueError, match=f"rotates {more}"):
            gguf_rotated({**metadata, "prism.hadamard.weight_names": [*names, more]}, tensors)
    with pytest.raises(ValueError, match="without its version"):
        gguf_rotated({key: value for key, value in metadata.items() if not key.endswith(".version")}, tensors)
    # a model whose value heads are its key heads' needs no word on their order
    assert gguf_rotated(said_by(tensors, gdn_v_grouped=None), tensors, more_value_heads=False)


def prism_metadata(config, block, signs, tensors):
    """The metadata of a rotated Qwen3.5 GGUF as the 27B's has it, for qwen35_gguf(more=)."""
    names = [gguf for gguf in tensors]
    widths = sorted(signs)
    return [("prism.hadamard.version", 4, 1), ("prism.hadamard.block_size", 4, block),
            ("prism.hadamard.transform", 8, "normalized-sylvester-walsh-hadamard"),
            ("prism.hadamard.axis", 8, "input-last-dimension"), ("prism.hadamard.sign_mode", 8, "explicit"),
            ("prism.hadamard.weight_names", 9, (8, names)), ("prism.hadamard.sign_widths", 9, (5, widths)),
            ("prism.hadamard.sign_values", 9, (5, [int(v) for width in widths for v in signs[width]])),
            ("prism.hadamard.inverse_weight_names", 9, (8, ["token_embd.weight"])), ("prism.hadamard.gdn_v_grouped", 7, True)]


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_a_rotated_gguf_converts_as_the_safetensors_of_the_same_values_that_says_the_same(dtype):
    """Both ways the page takes a GGUF (alone, and with its original's vocabulary and config.json): the basis of
    prism.hadamard.* is in the options, and the checkpoint is the one of the values the GGUF holds."""
    from test_gguf import QWEN35, QWEN35_LAYER  # the shape of test_gguf's small Qwen3.5: inputs 64, 128 and 32 wide
    tensors, config = qwen35_model(**QWEN35)
    said, signs = basis(16, widths_of(config))
    kinds = [QWEN35_LAYER[kind.rsplit(".", 1)[0]] for kind in FOLDED if kind.rsplit(".", 1)[0] in QWEN35_LAYER]
    layers = config["text_config"]["num_hidden_layers"]
    types = config["text_config"]["layer_types"]
    names = [f"blk.{layer}.{kind}.weight" for layer in range(layers) for kind in kinds
             if (kind in ("attn_qkv", "attn_gate", "ssm_out")) == (types[layer] == "linear_attention") or kind.startswith("ffn")]
    more = prism_metadata(config, 16, {width: values for width, values in signs.items()}, names)
    config_, file, same = qwen35_gguf(more=more, fold=lambda t: folded(t, 16, signs))
    vocabulary = unigram(config["text_config"]["vocab_size"])
    expected = safetensors_conversion(same, config, vocabulary, dtype)
    for got in (with_original(file, config, vocabulary, "tokenizer.json", dtype), fed(file, dtype)):
        assert got.options["rotated"] == said
        assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert "rotated" not in expected.options
    if dtype == "float32":
        # and it runs: the values the GGUF holds (Q8_0 rounds them) folded back are the model's
        got = with_original(file, config, vocabulary, "tokenizer.json", dtype)
        logits = run(got, config["text_config"]["vocab_size"], [1, 5, 7, 9, 11, 5])
        want = naive_qwen35_logits(tensors, config, [1, 5, 7, 9, 11, 5])
        assert np.abs(logits[-1] - want[-1]).max() < 0.05 * np.abs(want[-1]).max()
