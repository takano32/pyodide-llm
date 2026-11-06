"""Preferred Networks' PLaMo 3 (T649.2, model_type plamo3): a Gemma 3's layers (a norm in front of and another behind
the attention and the FFN, the norms of the heads of q and k, local layers that attend a window of the last positions)
with a SiLU in the FFN's gate, yarn in the RoPE of the layers that attend all the positions, q, k and v in one matrix and
the FFN's gate and up in another, and norms that multiply by c + weight with c = 1, 1/5 or 5^-1.5.

The reference (naive_plamo3_logits below) is written from the Hugging Face tensors the way the model's own
modeling_plamo.py computes them (pfnet/plamo-3-translate-0.8b@cc8e8816: Plamo3DecoderLayer, Attention, MLP,
Plamo3RMSNorm, RotaryEmbedding, swa_mask), so it checks the converter (the two matrices cut, q and k interleaved, the
offsets added to the norms, which layers are local ones, yarn) and the engine together. tests/reference_plamo3.py holds
that reference, and the engine, to transformers running the model's own code (CI: PyTorch is not on the development
machine)."""
import json
import math
import struct

import numpy as np
import pytest
from conftest import pack_tokenizer, tiny_vocab
from made_up import plamo3_model
from test_convert import converted, reader, safetensors_file
from test_gguf import unigram

import llama2_convert
from llama2_convert import Safetensors, check_config, checkpoint_form, checkpoint_size, normalize
from llama2_numpy import FORM, GELU, LAYOUTS, SILU, WHOLE, Llama, checkpoint_dtype, form_of, tensor_rows
from convert.families.local import local

# more than the window (4), and a token that comes again
TOKENS = [1, 5, 7, 9, 11, 5, 5, 300, 2, 17, 4, 40, 41, 42]
MODELS = {
    "small": dict(),
    "its classifier the embedding": dict(shared=True),
    "a classifier padded to 16 rows": dict(vocab_size=312, padded=True),
    "every layer attends all": dict(pattern=1),
    "every layer is local": dict(layers=3, pattern=8, window=3),
    "a window of one": dict(window=1),
    "a window past the text": dict(window=40),
    "keys for every head": dict(n_kv_heads=4),
    "no yarn": dict(factor=1.0),
    "another epsilon, other bases": dict(eps=1e-5, theta=20000.0, local_theta=1000.0),
}


def naive_plamo3_logits(tensors, config, tokens):
    """PLaMo 3 as its modeling_plamo.py computes it, written out from the Hugging Face tensors in float64, the whole
    sequence at once. config: config.json as Plamo3Config reads it (its flat keys: the code makes its layer_types and
    rope_parameters of these)."""
    wide = {name: np.asarray(tensor, dtype=np.float64) for name, tensor in tensors.items()}
    eps, heads, kv_heads, size = config["rms_norm_eps"], config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    window, pattern, ffn = config["window_size"], config["sliding_window_pattern"], config["intermediate_size"]
    norm = lambda v, weight, offset: (offset + weight) * (v / math.sqrt(float(v @ v) / len(v) + eps))
    silu = lambda v: v / (1.0 + np.exp(-v))

    def inverse(full):
        """RotaryEmbedding's inv_freq and concentration of a layer's kind"""
        base = config["rope_theta"] if full else config["rope_local_theta"]
        freq = base ** (np.arange(0, size, 2, dtype=np.float64) / size)
        factor = config["rope_scaling_factor"]
        if not full or factor == 1:
            return 1.0 / freq, 1.0
        half, original = size // 2, config["initial_context_length"]
        low = half * math.log(original / (32.0 * 2 * math.pi)) / math.log(base)
        high = half * math.log(original / (1.0 * 2 * math.pi)) / math.log(base)
        assert 0 < low < high < half - 1, (low, high)
        mask = 1 - np.clip((np.arange(half) - low) / (high - low), 0, 1)
        return 1.0 / (factor * freq) * (1.0 - mask) + 1.0 / freq * mask, 0.1 * math.log(factor) + 1.0

    def rotate(head, pos, full):
        """_rotary_pos_emb on one head: rotate_half, cos and sin times the concentration"""
        frequencies, concentration = inverse(full)
        angles = np.concatenate([pos * frequencies] * 2)
        cos, sin = np.cos(angles) * concentration, np.sin(angles) * concentration
        return head * cos + np.concatenate([-head[size // 2:], head[:size // 2]]) * sin

    x = [wide["model.embed_tokens.weight"][token] for token in tokens]
    q_dim, kv_dim = heads * size, kv_heads * size
    for layer in range(config["num_hidden_layers"]):
        full = (layer + 1) % pattern == 0
        p, a = f"model.layers.layers.{layer}.", f"model.layers.layers.{layer}.mixer."
        normed = [norm(v, wide[p + "pre_mixer_norm.weight"], 1.0) for v in x]
        queries, ks, vs = [], [], []
        for pos, v in enumerate(normed):
            qkv = wide[a + "qkv_proj.weight"] @ v
            q, k = qkv[:q_dim].reshape(heads, size), qkv[q_dim:q_dim + kv_dim].reshape(kv_heads, size)
            queries.append([rotate(norm(q[h], wide[a + "q_norm.weight"], 1.0), pos, full) for h in range(heads)])
            ks.append([rotate(norm(k[h], wide[a + "k_norm.weight"], 1.0), pos, full) for h in range(kv_heads)])
            vs.append(qkv[q_dim + kv_dim:].reshape(kv_heads, size))
        for pos in range(len(tokens)):
            # the causal mask, and over it the window of a local layer: position t where t > pos - window
            seen = [t for t in range(pos + 1) if full or t > pos - window]
            attended = np.zeros((heads, size))
            for h in range(heads):
                kv = h // (heads // kv_heads)
                scores = np.array([queries[pos][h] @ ks[t][kv] / math.sqrt(size) for t in seen])
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                attended[h] = sum(score * vs[t][kv] for score, t in zip(scores, seen))
            x[pos] = x[pos] + norm(wide[a + "o_proj.weight"] @ attended.reshape(-1), wide[p + "post_mixer_norm.weight"], 1.0 / 5)
            v = norm(x[pos], wide[p + "pre_mlp_norm.weight"], 1.0)
            h = wide[p + "mlp.gate_up_proj.weight"] @ v
            out = wide[p + "mlp.down_proj.weight"] @ (silu(h[:ffn]) * h[ffn:])
            x[pos] = x[pos] + norm(out, wide[p + "post_mlp_norm.weight"], 1.0 / 5 ** 1.5)
    classifier = wide["lm_head.weight"] if "lm_head.weight" in wide else wide["model.embed_tokens.weight"]
    return [(classifier @ norm(v, wide["model.norm.weight"], 1.0))[:config["vocab_size"]] for v in x]


def options_of(config):
    """What the caller passes to Llama() for a file of this config.json (the converter's options say the same)."""
    read = normalize(config)
    form = checkpoint_form(read, {})
    options = {"arch": "plamo3", "head_dim": form["head_dim"], "rope_theta": read["rope_theta"],
               "rms_norm_eps": read["rms_norm_eps"], "local": local(read)}
    return {**options, "rope_scaling": read["rope_scaling"]} if "rope_scaling" in read else options


def engine(tensors, config, dtype="float32"):
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), config, dtype)
    return Llama(checkpoint, pack_tokenizer(tiny_vocab(config["vocab_size"])), dtype=dtype, **options_of(config)), checkpoint


def conversion(tensors, config, dtype="float32"):
    """The page's way: the file in its own order, with a tokenizer.json, and the options that come out."""
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    made = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), unigram(config["vocab_size"]),
                                     "tokenizer.json", dtype=dtype, max_seq_len=config["max_position_embeddings"], start=8 + size)
    made.stream.feed(file[8 + size:])
    made.stream.finish()
    return made


def close(got, want):
    return np.allclose(got, want, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("model", MODELS)
def test_a_plamo3_converts_and_runs_like_its_own_code(model):
    tensors, config = plamo3_model(**MODELS[model])
    llama, checkpoint = engine(tensors, config)
    header = struct.unpack_from("<7i", checkpoint, 0)
    assert header[:5] == (32, 64, config["num_hidden_layers"], 4, config["num_key_value_heads"])
    assert checkpoint_dtype(header, len(checkpoint), options_of(config)) == "float32"
    want = naive_plamo3_logits(tensors, config, TOKENS)
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), pos


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_the_quantized_files_make_their_own_tables_with_yarn(dtype):
    """A quantized file has no RoPE tables: the engine makes them, yarn's magnitude in the full-attention layers' and
    none in the local layers'. Against the float32 file's, which the converter writes."""
    tensors, config = plamo3_model(pattern=2)
    exact, _ = engine(tensors, config)
    other, _ = engine(tensors, config, dtype)
    assert np.allclose(other.freq_cis_real, exact.freq_cis_real, atol=1e-6)
    assert np.allclose(other.freq_cis_imag, exact.freq_cis_imag, atol=1e-6)
    assert abs(float(np.abs(exact.freq_cis_real[0]).max()) - (0.1 * math.log(4.0) + 1.0)) < 1e-6
    assert float(np.abs(other.local_cis[0][0]).max()) == 1.0


# every tensor of a layer: changed in one layer, the engine must follow the reference to other numbers, or the tensor
# is not read where the model's code reads it
KINDS = ["mixer.qkv_proj.weight", "mixer.o_proj.weight", "mixer.q_norm.weight", "mixer.k_norm.weight",
         "pre_mixer_norm.weight", "post_mixer_norm.weight", "pre_mlp_norm.weight", "post_mlp_norm.weight",
         "mlp.gate_up_proj.weight", "mlp.down_proj.weight"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("which", [0, 3])
def test_every_tensor_is_read_where_its_own_code_reads_it(kind, which):
    tensors, config = plamo3_model()
    name = f"model.layers.layers.{which}.{kind}"
    changed = {**tensors, name: (tensors[name] * np.float32(1.5) + np.float32(0.05)).astype(np.float32)}
    llama, _ = engine(changed, config)
    want, before = naive_plamo3_logits(changed, config, TOKENS), naive_plamo3_logits(tensors, config, TOKENS)
    assert not close(want[-1], before[-1])
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), (name, pos)


@pytest.mark.parametrize("part, rows", [("q", (0, 128)), ("k", (128, 160)), ("v", (160, 192)),
                                        ("gate", (0, 64)), ("up", (64, 128))])
@pytest.mark.parametrize("layer", [0, 3])
def test_each_part_of_the_two_matrices_is_cut_where_its_own_code_cuts_it(part, rows, layer):
    """q, k and v of qkv_proj and the gate and the up of gate_up_proj, each changed alone, in a local layer (0) and in
    one that attends all (3)."""
    tensors, config = plamo3_model()
    name = f"model.layers.layers.{layer}." + ("mixer.qkv_proj.weight" if part in "qkv" else "mlp.gate_up_proj.weight")
    values = tensors[name].copy()
    start, stop = rows
    values[start:stop] = values[start:stop][::-1] * np.float32(1.3)
    changed = {**tensors, name: values}
    llama, _ = engine(changed, config)
    want = naive_plamo3_logits(changed, config, TOKENS)
    assert not close(want[-1], naive_plamo3_logits(tensors, config, TOKENS)[-1])
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), (part, layer, pos)


@pytest.mark.parametrize("window", [1, 2, 4])
def test_a_local_layer_attends_its_window_and_nothing_before_it(window):
    """A value changed at a position the window has left behind changes nothing of a local layer's later outputs."""
    tensors, config = plamo3_model(layers=2, pattern=8, window=window)
    llama, _ = engine(tensors, config)
    first = [llama.forward(token, pos) for pos, token in enumerate(TOKENS)]
    again, _ = engine(tensors, config)
    other = [again.forward(token if pos else 300, pos) for pos, token in enumerate(TOKENS)]
    # (two layers of a window w: position 0 reaches no later than position 2 (w - 1))
    for pos in range(len(TOKENS)):
        assert np.array_equal(first[pos], other[pos]) == (pos > 2 * (window - 1)), pos


def test_the_conversion_tells_the_engine_what_the_file_cannot():
    """The lesson of T72: the engine built from exactly the options the conversion gives runs like the reference."""
    for model in ("small", "its classifier the embedding", "every layer attends all", "no yarn",
                  "another epsilon, other bases", "a classifier padded to 16 rows"):
        shape = MODELS[model]
        tensors, config = plamo3_model(**shape)
        made = conversion(tensors, config)
        assert made.options["arch"] == "plamo3" and made.options["bias"] is False
        layers, pattern = config["num_hidden_layers"], config["sliding_window_pattern"]
        sliding = [layer for layer in range(layers) if (layer + 1) % pattern]
        if sliding:
            assert made.options["local"] == {"layers": sliding, "window": config["window_size"],
                                             "rope_theta": config["rope_local_theta"]}
        else:
            assert "local" not in made.options
        if config["rope_scaling_factor"] != 1:
            assert made.options["rope_scaling"] == {"rope_type": "yarn", "factor": 4.0, "beta_fast": 32.0,
                                                    "beta_slow": 1.0, "original_max_position_embeddings": 512,
                                                    "truncate": False}
        else:
            assert "rope_scaling" not in made.options
        assert made.options["rope_theta"] == config["rope_theta"]
        assert made.options.get("rms_norm_eps", 1e-5) == config["rms_norm_eps"]
        assert made.options.get("head_dim") == 32
        assert not {"linear", "convolution", "rotated", "unturned", "rotary"} & set(made.options)
        passed = {key: value for key, value in made.options.items()
                  if key in ("dtype", "arch", "bias", "head_dim", "local", "rope_theta", "rms_norm_eps",
                             "rope_scaling")}
        llama = Llama(bytes(made.stream.out), pack_tokenizer(tiny_vocab(config["vocab_size"])), **passed)
        want = naive_plamo3_logits(tensors, config, TOKENS)
        for pos, token in enumerate(TOKENS):
            assert close(llama.forward(token, pos), want[pos]), (model, pos)
        assert checkpoint_size(made.stream.header, "float32", made.options) == len(made.stream.out)
        assert form_of(made.options) == {**FORM, "arch": "plamo3", "head_dim": 32}
        assert (made.stream.header[5] < 0) is (not shape.get("shared", False))


def test_the_facts_of_the_architecture():
    """A Gemma 3's layers and rows, but a SiLU in the FFN's gate and yarn's magnitude in the tables."""
    assert LAYOUTS["plamo3"].rows is LAYOUTS["gemma3"].rows and LAYOUTS["plamo3"].local
    assert LAYOUTS["plamo3"].activation == SILU and LAYOUTS["gemma3"].activation == GELU
    assert LAYOUTS["plamo3"].yarn_magnitude and LAYOUTS["plamo3"].rope == WHOLE
    names = [row.name for row in tensor_rows((32, 64, 4, 4, 1, -320, 24), {"arch": "plamo3", "head_dim": 32})]
    assert names == [row.name for row in tensor_rows((32, 64, 4, 4, 1, -320, 24), {"arch": "gemma3", "head_dim": 32})]


# the published config.json of the 0.8B, as it is (the 2B's and the 8B's: its numbers, T649.3)
PUBLISHED = {"architectures": ["Plamo3ForCausalLM"], "auto_map": {}, "bos_token_id": 1, "dtype": "bfloat16",
             "eos_token_id": 1, "head_dim": 128, "hidden_size": 1536, "image_feature_size": None,
             "image_proj_type": "linear", "image_token_id": None, "initial_context_length": 4096,
             "intermediate_size": 6144, "linear_type": "normal", "max_position_embeddings": 262144,
             "model_type": "plamo3", "num_attention_heads": 16, "num_hidden_layers": 16, "num_key_value_heads": 1,
             "pad_token_id": 3, "rms_norm_eps": 1e-06, "rope_local_theta": 10000, "rope_scaling_factor": 256.0,
             "rope_theta": 1000000, "scale_embedding": False, "sliding_window_pattern": 8, "tie_word_embeddings": False,
             "tokenizer_class": "Plamo3Tokenizer", "transformers_version": "4.57.1", "use_cache": True,
             "vocab_size": 107520, "window_size": 2048,
             "rope_parameters": {"full_attention": {"rope_type": "yarn", "rope_theta": 1000000, "factor": 256.0,
                                                    "original_max_position_embeddings": 4096, "beta_fast": 32.0,
                                                    "beta_slow": 1.0, "truncate": False},
                                 "sliding_attention": {"rope_type": "default", "rope_theta": 10000}}}


def test_the_published_config_reads_as_the_models_code_reads_it():
    read = normalize(PUBLISHED)
    check_config(read)
    assert normalize(read) == read
    assert (read["rope_theta"], read["rope_local_base_freq"]) == (1000000, 10000)
    assert read["rope_scaling"] == {"rope_type": "yarn", "factor": 256.0, "beta_fast": 32.0, "beta_slow": 1.0,
                                    "original_max_position_embeddings": 4096, "truncate": False}
    assert local(read) == {"layers": [layer for layer in range(16) if layer not in (7, 15)], "window": 2048,
                           "rope_theta": 10000.0}
    form = checkpoint_form(read, {"model.layers.layers.0.mixer.q_norm.weight": None})
    assert form["arch"] == "plamo3" and form["head_dim"] == 128
    assert llama2_convert.checkpoint_header(read, {"lm_head.weight": None}, 4096) == (1536, 6144, 16, 16, 1, -107520, 4096)


REFUSED = [
    ({"scale_embedding": True}, "multiplies its embedding"),
    ({"linear_type": "fp8"}, "of the fp8 kind"),
    ({"window_size": None}, "no usable sliding_window"),
    ({"rope_local_theta": None}, "no usable base of RoPE"),
    ({"rope_scaling_factor": 4.0, "initial_context_length": None}, "names no factor or no original context"),
]


@pytest.mark.parametrize("change, reason", REFUSED)
def test_a_plamo3_the_engine_cannot_run_is_refused(change, reason):
    _, config = plamo3_model()
    check_config(normalize(config))
    with pytest.raises(ValueError, match=reason):
        check_config(normalize({**config, **change}))


def test_the_engine_reads_the_tensors_where_the_layout_puts_them():
    """Every row of the file as the engine holds it against the Hugging Face tensor it is made of, in every dtype's
    file: the sizes agree (checkpoint_dtype()), and the float32 file's rows are the tensors (the norms plus their c)."""
    tensors, config = plamo3_model()
    for dtype in ("float32", "float16", "int8", "int6"):
        llama, checkpoint = engine(tensors, config, dtype)
        header = struct.unpack_from("<7i", checkpoint, 0)
        assert checkpoint_dtype(header, len(checkpoint), options_of(config)) == dtype
        tolerance = {"float32": 0, "float16": 2e-3, "int8": 0.02, "int6": 0.05}[dtype]
        for layer in (0, 3):
            at = f"model.layers.layers.{layer}."
            qkv, gate_up = tensors[at + "mixer.qkv_proj.weight"], tensors[at + "mlp.gate_up_proj.weight"]
            assert np.allclose(llama.wv[layer], qkv[160:], atol=tolerance), dtype
            assert np.allclose(llama.w1[layer], gate_up[:64], atol=tolerance), dtype
            assert np.allclose(llama.w3[layer], gate_up[64:], atol=tolerance), dtype
            assert np.allclose(llama.wo[layer], tensors[at + "mixer.o_proj.weight"], atol=tolerance), dtype
            for ours, theirs, offset in (("rms_att_weight", "pre_mixer_norm", 1.0), ("rms_att_post_weight", "post_mixer_norm", 0.2),
                                         ("rms_ffn_weight", "pre_mlp_norm", 1.0),
                                         ("rms_ffn_post_weight", "post_mlp_norm", 5 ** -1.5)):
                assert np.allclose(getattr(llama, ours)[layer], tensors[at + theirs + ".weight"] + np.float32(offset),
                                   atol=max(tolerance, 1e-7) if dtype == "float16" else 1e-7), (dtype, ours)
        assert np.allclose(llama.rms_final_weight, tensors["model.norm.weight"] + 1, atol=1e-7 if dtype != "float16" else 2e-3)


# ---- a GGUF of a PLaMo 3 (T649.4)
# llama.cpp's names of a PLaMo 3's tensors (conversion/plamo.py's Plamo3Model, gguf-py's tensor_mapping.py and
# src/models/plamo3.cpp at 67107ac3: the norms behind the attention and the FFN without a ".weight"), written a second
# time on purpose, and what it adds to each norm (the model's offsets)
GGUF_NAMES = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
              "lm_head.weight": "output.weight"}
GGUF_LAYER = {"pre_mixer_norm.weight": "attn_norm.weight", "mixer.qkv_proj.weight": "attn_qkv.weight",
              "mixer.q_norm.weight": "attn_q_norm.weight", "mixer.k_norm.weight": "attn_k_norm.weight",
              "mixer.o_proj.weight": "attn_output.weight", "post_mixer_norm.weight": "post_attention_norm",
              "pre_mlp_norm.weight": "ffn_norm.weight", "mlp.gate_up_proj.weight": "ffn_up.weight",
              "mlp.down_proj.weight": "ffn_down.weight", "post_mlp_norm.weight": "post_ffw_norm"}
GGUF_ADDED = {"post_mixer_norm.weight": 1 / 5, "post_mlp_norm.weight": 1 / 5 ** 1.5}


def plamo3_gguf(tensors, config, matrices=None, more=()):
    """A GGUF v3 of a PLaMo 3's Hugging Face tensors the way llama.cpp writes one (pfnet/plamo-3-translate-0.8b-GGUF's
    header, read by Range): q, k and v one matrix and the gate and the up one, in Hugging Face's order; every norm with
    what the model adds to it, F32; the yarn, the window and its pattern as llama.cpp says them. matrices: what makes a
    matrix's blocks and their ggml type (None: F32). more: metadata (key, kind, value) over these. Returns the file
    and, under the Hugging Face names, the values it stands for."""
    string = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()
    arch = "plamo3"
    said = {f"{arch}.block_count": (4, config["num_hidden_layers"]), f"{arch}.context_length": (4, config["max_position_embeddings"]),
            f"{arch}.embedding_length": (4, config["hidden_size"]), f"{arch}.feed_forward_length": (4, config["intermediate_size"]),
            f"{arch}.attention.head_count": (4, config["num_attention_heads"]),
            f"{arch}.attention.head_count_kv": (4, config["num_key_value_heads"]),
            f"{arch}.attention.key_length": (4, config["head_dim"]), f"{arch}.attention.value_length": (4, config["head_dim"]),
            f"{arch}.attention.layer_norm_rms_epsilon": (6, config["rms_norm_eps"]),
            f"{arch}.rope.freq_base": (6, config["rope_theta"]), f"{arch}.rope.freq_base_swa": (6, config["rope_local_theta"]),
            f"{arch}.attention.sliding_window": (4, config["window_size"]),
            f"{arch}.attention.sliding_window_pattern": (4, config["sliding_window_pattern"])}
    if config["rope_scaling_factor"] != 1:
        said.update({f"{arch}.rope.scaling.type": (8, "yarn"), f"{arch}.rope.scaling.factor": (6, config["rope_scaling_factor"]),
                     f"{arch}.rope.scaling.original_context_length": (4, config["initial_context_length"]),
                     f"{arch}.rope.scaling.yarn_beta_fast": (6, 32.0), f"{arch}.rope.scaling.yarn_beta_slow": (6, 1.0),
                     f"{arch}.rope.scaling.yarn_truncate": (7, False)})
    said.update({key: (kind, value) for key, kind, value in more})
    metadata = [("general.architecture", 8, arch), *((key, kind, value) for key, (kind, value) in said.items() if value is not None)]
    stored = {}
    for name, tensor in tensors.items():
        if name in GGUF_NAMES:
            gguf = GGUF_NAMES[name]
        else:
            layer, rest = name.removeprefix("model.layers.layers.").split(".", 1)
            gguf = f"blk.{layer}.{GGUF_LAYER[rest]}"
        added = 0.0 if tensor.ndim != 1 else next((c for end, c in GGUF_ADDED.items() if name.endswith(end)), 1.0)
        stored[gguf] = (name, tensor.astype(np.float32) + np.float32(added), added)
    out = [b"GGUF", struct.pack("<IQQ", 3, len(stored), len(metadata))]
    for key, kind, value in metadata:
        out.append(string(key) + struct.pack("<I", kind) + metadata_value(kind, value))
    blobs, offset, same = [], 0, {}
    for gguf, (name, value, added) in stored.items():
        if value.ndim == 2 and matrices is not None:
            make, type_ = matrices
            blob, held = make(np.ascontiguousarray(value))
        else:
            blob, held, type_ = np.ascontiguousarray(value, np.float32).tobytes(), value, 0
        same[name] = (held - np.float32(added)).astype(np.float32) if added else held
        out.append(string(gguf) + struct.pack("<I", value.ndim) + struct.pack(f"<{value.ndim}Q", *reversed(value.shape))
                   + struct.pack("<IQ", type_, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    head = b"".join(out)
    head += b"\0" * (-len(head) % 32)
    return head + b"".join(blobs), same


def metadata_value(kind, value):
    from test_gguf import metadata_value as of
    return of(kind, value)


def gguf_converted(file, config, dtype, blocks=False):
    vocabulary = unigram(config["vocab_size"])
    header, base = llama2_convert.gguf_weights(file, json.dumps(config))
    made = llama2_convert.Conversion(header, base, json.dumps(config), vocabulary, "tokenizer.json", dtype=dtype,
                                     max_seq_len=config["max_position_embeddings"], start=base, blocks=blocks)
    for start in range(base, len(file), 5000):
        made.feed(file[start:start + 5000])
    made.finish()
    return made


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_a_gguf_with_the_originals_files_is_the_safetensors_conversion(dtype):
    """llama.cpp's names of a PLaMo 3, its norms with what the model adds (the plan leaves that step out: the "done"
    marks), q, k and v and the gate and the up as Hugging Face has them: the checkpoint, tokenizer.bin and options of
    the safetensors of the same values, to the byte."""
    tensors, config = plamo3_model()
    file, same = plamo3_gguf(tensors, config)
    got, expected = gguf_converted(file, config, dtype), conversion(same, config, dtype)
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options and got.options["local"]["layers"] == [0, 1, 2]


@pytest.mark.parametrize("more, what", [
    ([("plamo3.attention.sliding_window", 4, 8)], "sliding-attention layers"),
    ([("plamo3.attention.sliding_window_pattern", 4, 2)], "sliding-attention layers"),
    ([("plamo3.rope.freq_base_swa", 6, 1000.0)], "sliding-attention layers"),
    ([("plamo3.rope.scaling.factor", 6, 8.0)], "yarn RoPE scaling"),
    ([("plamo3.rope.scaling.yarn_beta_fast", 6, 16.0)], "otherwise than a PLaMo 3"),
    ([("plamo3.rope.scaling.yarn_truncate", 7, True)], "otherwise than a PLaMo 3"),
    ([("plamo3.attention.head_count_kv", 4, 2)], "number of key-value heads"),
])
def test_a_gguf_that_is_not_the_originals_is_refused(more, what):
    """What no tensor says: which layers attend a window, the window, its base, yarn (and the bounds a PLaMo 3's
    config.json does not say, which the converter takes as the model's own)."""
    tensors, config = plamo3_model()
    llama2_convert.gguf_weights(plamo3_gguf(tensors, config)[0], json.dumps(config))  # its own config goes through
    with pytest.raises(ValueError, match=what):
        llama2_convert.gguf_weights(plamo3_gguf(tensors, config, more=more)[0], json.dumps(config))


def test_its_blocks_are_kept_q_k_and_v_and_the_gate_and_the_up_cut_as_bytes():
    """Conversion(blocks=True): every matrix of a GGUF in llama.cpp's blocks stays as it is, q, k and v and the gate and
    the up of the FFN a run of the rows of one tensor each (the plan's "rows", moved as bytes), and the logits are those
    of the conversion that widens the same file, to the bit (float32 both). On rows of whole blocks of 256."""
    from test_superblocks import blocks_of
    tensors, config = plamo3_model(dim=256, hidden_dim=512, head_dim=64, n_heads=4, n_kv_heads=1, spread=0.05)
    file, _ = plamo3_gguf(tensors, config, matrices=blocks_of("IQ2_XS", scale=0x1400))
    kept, wide = gguf_converted(file, config, "float32", blocks=True), gguf_converted(file, config, "float32")
    assert kept.options["kinds"] == dict.fromkeys(
        ("token_embedding_table", "wq", "wk", "wv", "wo", "w1", "w2", "w3", "wcls"), "IQ2_XS")
    moved = {name.split(".")[-2] for name, orders in kept.stream.stay.items() if orders is not None}
    assert moved == {"qkv_proj", "gate_up_proj"}, moved
    read = lambda made: Llama(bytes(made.checkpoint), bytes(made.tokenizer), **made.options)
    ours, theirs = (np.stack([read(made).forward(token, at) for at, token in enumerate(TOKENS)]) for made in (kept, wide))
    assert np.isfinite(ours).all() and np.ptp(ours) > 0 and np.array_equal(ours.view(np.uint32), theirs.view(np.uint32))
