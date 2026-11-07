"""Google's Gemma 2 (T612): a Gemma 3's layers without the norms of the heads of q and k, with the attention's scores
and the logits capped by a tanh, one local layer of every two (the first, then every other) turning by the one base of
RoPE, and the scores divided by the root of query_pre_attn_scalar (which the conversion folds into q).

The reference (conftest.naive_gemma3_logits, which reads a Gemma 2 too) is written from the Hugging Face tensors the way
transformers computes them, so it checks the converter and the engine together. tests/reference_gemma2.py holds that
reference, and the engine, to transformers itself (CI)."""
import json
import math
import struct

import numpy as np
import pytest
from conftest import naive_gemma3_logits, pack_tokenizer, tiny_vocab
from made_up import gemma2_model
from test_convert import converted, reader, safetensors_file
from test_gemma3 import TOKENS, close, safetensors_conversion
from test_gguf import metadata_value, q8_0_blocks, unigram, with_original

import llama2_convert
from llama2_convert import Safetensors, check_config, checkpoint_form, normalize
from llama2_numpy import LAYOUTS, Llama, tensor_rows
from convert.families import FAMILIES
from convert.families.gemma2 import GEMMA2_NORMS
from convert.families.gemma3 import GEMMA3

MODELS = {
    "small": dict(),
    "a classifier of its own": dict(shared=False),
    "the kinds unsaid (a config.json of 2024)": dict(listed=False),
    "the scores by another number than the head's size (the 27B's)": dict(scalar=40),
    "keys for every head": dict(n_kv_heads=4, kinds="slsl"),
    "a window past the text": dict(window=20),
    "a window of one": dict(window=1, kinds="sls"),
    "no caps": dict(scores_cap=None, logits_cap=None),
    "heads that fill the width": dict(head_dim=8),
}


def read(config):
    """config.json as transformers' Gemma2Config would have read it, for the reference: the kinds of the layers listed
    (one of every two from the first where it says none), the local layers' base its one base (by hand here, not by the
    converter's normalize())."""
    read = dict(config)
    read.setdefault("layer_types", ["sliding_attention" if layer % 2 == 0 else "full_attention"
                                    for layer in range(read["num_hidden_layers"])])
    read["rope_local_base_freq"] = read["rope_theta"]
    return read


def options_of(config):
    """What the caller passes to Llama() for a file of this config.json (the converter's options say the same)."""
    config = normalize(config)
    return {"arch": "gemma2", "head_dim": checkpoint_form(config, {})["head_dim"], "rope_theta": config["rope_theta"],
            "rms_norm_eps": config["rms_norm_eps"], **FAMILIES["gemma2"].options(config)}


def engine(tensors, config, dtype="float32"):
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), config, dtype)
    return Llama(checkpoint, pack_tokenizer(tiny_vocab(config["vocab_size"])), dtype=dtype, **options_of(config))


def conversion(tensors, config, dtype="float32"):
    """The page's way: the file in its own order, with a tokenizer.json, and the options that come out."""
    return safetensors_conversion(tensors, config, unigram(config["vocab_size"]), dtype)


@pytest.mark.parametrize("model", MODELS)
def test_a_gemma2_converts_and_runs_like_transformers(model):
    tensors, config = gemma2_model(**MODELS[model])
    llama = engine(tensors, config)
    want = naive_gemma3_logits(tensors, read(config), TOKENS)
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), pos


@pytest.mark.parametrize("model", MODELS)
def test_the_conversion_tells_the_engine_what_the_file_cannot(model):
    tensors, config = gemma2_model(**MODELS[model])
    made = conversion(tensors, config)
    kinds = read(config)["layer_types"]
    local = [layer for layer, kind in enumerate(kinds) if kind == "sliding_attention"]
    assert made.options["arch"] == "gemma2"
    assert made.options.get("local") == {"layers": local, "window": config["sliding_window"], "rope_theta": config["rope_theta"]}
    assert made.options.get("softcap") == config["final_logit_softcapping"]
    assert made.options.get("score_softcap") == config["attn_logit_softcapping"]
    llama = Llama(bytes(made.checkpoint), pack_tokenizer(tiny_vocab(config["vocab_size"])),
                  **{key: value for key, value in made.options.items() if key in options_of(config) or key in ("dtype",)})
    want = naive_gemma3_logits(tensors, read(config), TOKENS)
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), pos


KINDS = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
         "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
         "post_feedforward_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("which", [0, 1, -1])
def test_every_tensor_is_read_where_transformers_reads_it(kind, which):
    tensors, config = gemma2_model()
    name = f"model.layers.{which % config['num_hidden_layers']}.{kind}"  # a local layer, one that is not, the last
    changed = dict(tensors)
    changed[name] = (tensors[name] + np.random.default_rng(7).standard_normal(tensors[name].shape) * 0.3).astype(np.float32)
    before, after = engine(tensors, config), engine(changed, config)
    want = naive_gemma3_logits(changed, read(config), TOKENS)
    moved = False
    for pos, token in enumerate(TOKENS):
        was, got = before.forward(token, pos).copy(), after.forward(token, pos)
        assert close(got, want[pos]), pos
        moved |= not np.allclose(got, was, rtol=1e-3, atol=1e-3)
    assert moved, name


# the faults the reference is held to here as well (tests/reference_gemma2.py runs them against transformers): each
# changes the engine's numbers past the line close() holds them to
def capless(llama):
    llama.score_softcap = 0.0


def logits_capless(llama):
    llama.softcap = 0.0


def scores_by_the_head(llama):
    """the scores divided by the root of the head's size, not query_pre_attn_scalar's (q not multiplied)"""
    llama.wq = llama.wq / np.float32(math.sqrt(16 / 40))


def kinds_shifted(llama):
    """the local layers one later: the second, then every other"""
    llama.layers = [{**facts, "window": llama.layers[l - 1]["window"], "local": llama.layers[l - 1]["local"]}
                    for l, facts in enumerate(llama.layers)]


@pytest.mark.parametrize("fault, model", [(capless, {}), (logits_capless, {}), (scores_by_the_head, {"scalar": 40}),
                                          (kinds_shifted, {"window": 2})])
def test_the_reference_sees_each_fault(fault, model):
    tensors, config = gemma2_model(**model)
    llama = engine(tensors, config)
    fault(llama)
    want = naive_gemma3_logits(tensors, read(config), TOKENS)
    assert not all(close(llama.forward(token, pos), want[pos]) for pos, token in enumerate(TOKENS))


def test_q_is_multiplied_only_where_the_scalar_is_not_the_heads_size():
    """The 9B's query_pre_attn_scalar is a head's size: q is as it was; the 2B's is 224 of heads of 256, the 27B's 144
    of 128"""
    for scalar, factor in ((None, 1.0), (40, math.sqrt(16 / 40)), (4, 2.0)):
        config = normalize(gemma2_model(scalar=scalar)[1])
        assert FAMILIES["gemma2"].scale(config) == pytest.approx(factor)
    # (the real 2B's config.json, google/gemma-2-2b-jpn-it's and gemma-2-2b-it's alike)
    assert FAMILIES["gemma2"].scale(normalize({**gemma2_model()[1], "head_dim": 256, "query_pre_attn_scalar": 224})) == pytest.approx(math.sqrt(256 / 224))


def test_the_layout_is_a_gemma3s_without_the_norms_of_the_heads():
    header, form = (64, 128, 4, 4, 2, 320, 48), {"head_dim": 16}
    two = [row.name for row in tensor_rows(header, {**form, "arch": "gemma2"})]
    three = [row.name for row in tensor_rows(header, {**form, "arch": "gemma3"})]
    assert two == [name for name in three if name not in ("q_norm", "k_norm")]
    assert LAYOUTS["gemma2"]._replace(rows=None) == LAYOUTS["gemma3"]._replace(rows=None)


def test_the_options_are_the_engines_caps():
    llama = engine(*gemma2_model())
    assert llama.score_softcap == 2.0 and llama.softcap == 3.0
    assert engine(*gemma2_model(scores_cap=None, logits_cap=None)).score_softcap == 0.0


REFUSED = [
    ({"hidden_activation": "gelu"}, "its activation is gelu"),
    ({"attention_bias": True}, "its layers have biases"),
    ({"attn_logit_softcapping": -1.0}, "no usable attn_logit_softcapping"),
    ({"final_logit_softcapping": "30"}, "no usable final_logit_softcapping"),
    ({"query_pre_attn_scalar": 0}, "no usable query_pre_attn_scalar"),
    ({"layer_types": ["sliding_attention", "conv"] * 3}, "sliding-attention layers and full-attention layers"),
    ({"sliding_window": 0}, "no usable sliding_window"),
    ({"rope_scaling": {"rope_type": "linear", "factor": 2.0}}, "RoPE scaling of the linear kind"),
]


@pytest.mark.parametrize("change, reason", REFUSED)
def test_a_gemma2_the_engine_cannot_run_is_refused(change, reason):
    _, config = gemma2_model()
    check_config(normalize(config))
    with pytest.raises(ValueError, match=reason):
        check_config(normalize({**config, **change}))


def test_twice_normalized_is_once():
    for model in MODELS.values():
        config = normalize(gemma2_model(**model)[1])
        assert normalize(config) == config


# ---- a GGUF of a Gemma 2, the way llama.cpp writes one (bartowski's gemma-2-2b-jpn-it Q8_0)
GEMMA = dict(dim=64, hidden_dim=128, head_dim=32)
OUTSIDE, LAYER, NAMES = FAMILIES["gemma2"].gguf_names


def gemma2_gguf(more=(), leave=(), **shape):
    """Q8_0 matrices, F32 norms with the 1 added, q and k as Hugging Face holds them, the embedding as it is, the two
    caps and the window in the metadata, neither the kinds of the layers nor a base of the local ones (llama.cpp has
    one of every two and the one base). Returns the config.json, the file, and under the Hugging Face names the values
    it stands for."""
    tensors, config = gemma2_model(**{**GEMMA, **shape})
    inverse = {theirs: ours for ours, theirs in OUTSIDE.items()}
    layered = {theirs: ours for ours, theirs in NAMES.items()}
    stored, same = {}, {}
    for name in sorted(tensors, key=lambda name: name != "model.norm.weight"):
        tensor = tensors[name]
        if name in inverse:
            gguf = inverse[name]
        else:
            _, _, layer, *rest = name.split(".")
            gguf = f"blk.{layer}.{layered['.'.join(rest[:-1])]}.{rest[-1]}"
        if tensor.ndim == 2:
            blob, held = q8_0_blocks(tensor)
            stored[gguf], same[name] = [blob, 8, tensor.shape], held
        else:
            stored[gguf], same[name] = [(tensor + np.float32(1.0)).astype(np.float32).tobytes(), 0, tensor.shape], tensor
    arch = "gemma2"
    metadata = [("general.architecture", 8, arch), (f"{arch}.block_count", 4, config["num_hidden_layers"]),
                (f"{arch}.context_length", 4, config["max_position_embeddings"]),
                (f"{arch}.embedding_length", 4, config["hidden_size"]),
                (f"{arch}.feed_forward_length", 4, config["intermediate_size"]),
                (f"{arch}.attention.head_count", 4, config["num_attention_heads"]),
                (f"{arch}.attention.head_count_kv", 4, config["num_key_value_heads"]),
                (f"{arch}.attention.layer_norm_rms_epsilon", 6, config["rms_norm_eps"]),
                (f"{arch}.attention.key_length", 4, config["head_dim"]),
                (f"{arch}.attention.value_length", 4, config["head_dim"]),
                (f"{arch}.rope.freq_base", 6, float(config["rope_theta"])),
                (f"{arch}.attn_logit_softcapping", 6, config["attn_logit_softcapping"]),
                (f"{arch}.final_logit_softcapping", 6, config["final_logit_softcapping"]),
                (f"{arch}.attention.sliding_window", 4, config["sliding_window"]),
                ("tokenizer.ggml.model", 8, "llama"), ("tokenizer.ggml.pre", 8, "default"),
                ("tokenizer.ggml.bos_token_id", 4, 2), ("tokenizer.ggml.eos_token_id", 4, 1)]
    metadata = [entry for entry in metadata if entry[0] not in leave] + list(more)
    string = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()
    tokens = [f"w{i}" for i in range(config["vocab_size"])]
    out = [b"GGUF", struct.pack("<IQQ", 3, len(stored), len(metadata) + 3)]
    for key, kind, value in metadata:
        out.append(string(key) + struct.pack("<I", kind))
        out.append(metadata_value(kind, value))
    out.append(string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens)) + b"".join(map(string, tokens)))
    out.append(string("tokenizer.ggml.token_type") + struct.pack("<IIQ", 9, 5, len(tokens)) + struct.pack(f"<{len(tokens)}i", *[1] * len(tokens)))
    out.append(string("tokenizer.ggml.merges") + struct.pack("<IIQ", 9, 8, 0))
    blobs, offset = [], 0
    for gguf, (blob, type_, shape) in stored.items():
        out.append(string(gguf) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *reversed(shape))
                   + struct.pack("<IQ", type_, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    head = b"".join(out)
    head += b"\0" * (-len(head) % 32)
    return config, head + b"".join(blobs), same


# the real 2B's metadata: the caps of 50 and 30, a window, no kinds; its config.json of 2024 lists no kinds either
AS_PUBLISHED = dict(listed=False, scores_cap=50.0, logits_cap=30.0, kinds="slslsl", window=5, theta=10000.0)


# (the scores by another number than the head's size: the original's config.json says it, which llama.cpp's GGUF does not,
# and the conversion multiplies q by it all the same)
GGUF_SHAPES = {"as the published one says it": AS_PUBLISHED, "a classifier of its own, small caps": dict(shared=False, kinds="slsl"),
               "the scores by another number": {**AS_PUBLISHED, "scalar": 40}}


@pytest.mark.parametrize("dtype", ["int8", "float32"])
@pytest.mark.parametrize("shape", GGUF_SHAPES)
def test_a_gemma2_gguf_with_the_originals_files_is_the_safetensors_conversion(shape, dtype):
    config, file, same = gemma2_gguf(**GGUF_SHAPES[shape])
    vocabulary = unigram(config["vocab_size"])
    got = with_original(file, config, vocabulary, "tokenizer.json", dtype, piece=700)
    expected = safetensors_conversion(same, config, vocabulary, dtype, piece=700)
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert got.options == expected.options
    assert got.options["arch"] == "gemma2" and got.options["score_softcap"] == config["attn_logit_softcapping"]
    if dtype == "float32":
        llama = Llama(bytes(got.checkpoint), pack_tokenizer(tiny_vocab(config["vocab_size"])), **options_of(config))
        want = naive_gemma3_logits(same, read(config), TOKENS)
        for pos, token in enumerate(TOKENS):
            assert close(llama.forward(token, pos), want[pos]), pos


@pytest.mark.parametrize("change, what", [
    ({"attn_logit_softcapping": 30.0}, "cap of the scores"),
    ({"final_logit_softcapping": None}, "cap of the logits"),
    ({"sliding_window": 6}, "sliding-attention layers"),
    ({"layer_types": ["full_attention", "sliding_attention"] * 3}, "sliding-attention layers"),
    ({"rope_theta": 100.0}, "RoPE theta"),
])
def test_a_gemma2_gguf_that_is_not_the_originals_is_refused(change, what):
    """What no tensor shows: the caps, which layers are local and their window, the base."""
    config, file, _ = gemma2_gguf(**AS_PUBLISHED)
    llama2_convert.gguf_weights(file, json.dumps(config))  # its own config goes through
    with pytest.raises(ValueError, match=f"its {what} is "):
        llama2_convert.gguf_weights(file, json.dumps({**config, **change}))


def test_the_ggufs_names_are_a_gemma3s_but_for_the_norms_of_the_heads():
    three = GEMMA3.gguf_names
    assert OUTSIDE == three[0] and LAYER == three[1]
    assert NAMES == {ours: theirs for ours, theirs in three[2].items() if ours not in ("attn_q_norm", "attn_k_norm")}
    assert set(GEMMA2_NORMS) == {name + ".weight" for name in ("output_norm", "attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm")}
