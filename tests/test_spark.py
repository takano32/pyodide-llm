"""XHToken's Spark-X2.5 (T.339 to T.342): a Llama's layers, three of every four attending a window of the last positions
and turning their whole heads by a base of their own, the others a quarter of each head; q, k and v in one matrix; a gate
of each head of the attention's output; and the exact GELU in the FFN's gate.

The reference (naive_spark_logits below) is written from the Hugging Face tensors the way the model's own
modeling_spark.py computes them, so it checks the converter (q, k and v cut out of one matrix, q and k interleaved over
the width each layer turns, which layers are local ones) and the engine together. tests/reference_spark.py holds that
reference, and the engine, to transformers running the model's own code (CI: PyTorch is not on the development
machine)."""
import json
import math
import struct

import numpy as np
import pytest
from conftest import pack_tokenizer, tiny_vocab
from made_up import spark_model
from test_convert import converted, reader, safetensors_file
from test_gguf import metadata_value, q8_0_blocks, unigram, with_original

import llama2_convert
from llama2_convert import Safetensors, check_config, checkpoint_form, checkpoint_size, normalize
from llama2_numpy import FORM, GELU_ERF, LAYOUTS, PARTLY, Llama, checkpoint_dtype, form_of, gelu_erf, tensor_rows
from convert.families.spark import local, turns

# more than the window (4), and a token that comes again
TOKENS = [0, 5, 7, 9, 11, 5, 5, 300, 2, 17, 4]
MODELS = {
    "small": dict(),
    "a classifier of its own": dict(shared=False),
    "every layer attends all": dict(kinds="lll"),
    "every layer is local": dict(kinds="sss", window=3),
    "a window of one": dict(window=1, kinds="sls"),
    "a window past the text": dict(window=20),
    "keys for every head": dict(n_kv_heads=4, kinds="lssl"),
    "heads that fill the width": dict(head_dim=8, n_kv_heads=2),
    "half of a head turned, the bases the other way round": dict(part=0.5, theta=50.0, local_theta=1000000.0),
    "another epsilon": dict(eps=1e-5),
}


def naive_spark_logits(tensors, config, tokens, states=None):
    """Spark-X2.5 as its modeling_spark.py computes it (at XHToken/Spark-X2.5-1.7B@14d6e83c: Spark2_5Model.forward,
    Spark2_5DecoderLayer, Spark2_5Attention with eager_attention_forward and the masks of the model's forward,
    Spark2_5MLP, Spark2_5RMSNorm, compute_rope_cos_sin and apply_rotary_pos_emb), written out from the Hugging Face
    tensors in float64, the whole sequence at once. config: config.json as it is published (its rope_parameters a
    dict for each kind of layer). states: a list that gets, for every layer, what its attention gave at every
    position (before out_proj)."""
    wide = {name: np.asarray(tensor, dtype=np.float64) for name, tensor in tensors.items()}
    eps, heads, kv_heads = config["rms_norm_eps"], config["num_attention_heads"], config["num_key_value_heads"]
    size, window = config["head_dim"], config["sliding_window"]
    norm = lambda v, weight: weight * (v / math.sqrt(float(v @ v) / len(v) + eps))
    gelu = lambda v: np.array([0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0))) for x in v])

    def rotate(head, pos, kind):
        """apply_rotary_pos_emb on one head: rotate_half over its first part, by the angles of that part's width"""
        said = config["rope_parameters"][kind]
        width = int(size * said["partial_rotary_factor"])
        inverse = 1.0 / said["rope_theta"] ** (np.arange(0, width, 2, dtype=np.float64) / width)
        cos, sin = np.cos(pos * inverse), np.sin(pos * inverse)
        cos, sin = np.concatenate([cos, cos]), np.concatenate([sin, sin])
        part = head[:width]
        turned = part * cos + np.concatenate([-part[width // 2:], part[:width // 2]]) * sin
        return np.concatenate([turned, head[width:]])

    x = [wide["model.embedding.weight"][token] for token in tokens]
    q_dim, kv_dim = heads * size, kv_heads * size
    for layer, kind in enumerate(config["layer_types"]):
        p, a = f"model.layers.{layer}.", f"model.layers.{layer}.self_attn."
        normed = [norm(v, wide[p + "input_layernorm.weight"]) for v in x]
        queries, ks, vs, gates = [], [], [], []
        for pos, v in enumerate(normed):
            qkv = wide[a + "q_k_v_proj.weight"] @ v
            q, k = qkv[:q_dim].reshape(heads, size), qkv[q_dim:q_dim + kv_dim].reshape(kv_heads, size)
            queries.append([rotate(q[h], pos, kind) for h in range(heads)])
            ks.append([rotate(k[h], pos, kind) for h in range(kv_heads)])
            vs.append(qkv[q_dim + kv_dim:].reshape(kv_heads, size))
            gates.append(1.0 / (1.0 + np.exp(-(wide[a + "g_proj.weight"] @ v))))
        mixed = []
        for pos in range(len(tokens)):
            # the causal mask, and over it the window of a sliding-attention layer: position t where t > pos - window
            seen = [t for t in range(pos + 1) if kind == "full_attention" or t > pos - window]
            attended = np.zeros((heads, size))
            for h in range(heads):
                kv = h // (heads // kv_heads)
                scores = np.array([queries[pos][h] @ ks[t][kv] / math.sqrt(size) for t in seen])
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                attended[h] = sum(score * vs[t][kv] for score, t in zip(scores, seen)) * gates[pos][h]
            mixed.append(attended.reshape(-1))
        if states is not None:
            states.append(mixed)
        for pos in range(len(tokens)):
            x[pos] = x[pos] + wide[a + "out_proj.weight"] @ mixed[pos]
            v = norm(x[pos], wide[p + "post_attention_layernorm.weight"])
            x[pos] = x[pos] + wide[p + "mlp.down_proj.weight"] @ (gelu(wide[p + "mlp.gate_proj.weight"] @ v) * (wide[p + "mlp.up_proj.weight"] @ v))
    classifier = wide["lm_head.weight"] if "lm_head.weight" in wide else wide["model.embedding.weight"]
    return [classifier @ norm(v, wide["model.norm.weight"]) for v in x]


def options_of(config):
    """What the caller passes to Llama() for a file of this config.json (the converter's options say the same)."""
    config = normalize(config)
    form = checkpoint_form(config, {})
    return {"arch": "spark", "head_dim": form["head_dim"], "rope_theta": config["rope_theta"],
            "rms_norm_eps": config["rms_norm_eps"], "local": local(config),
            "rotary": int(form["head_dim"] or config["hidden_size"] // config["num_attention_heads"]) * config["rotary_pct"]}


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
def test_a_spark_converts_and_runs_like_its_own_code(model):
    tensors, config = spark_model(**MODELS[model])
    llama, checkpoint = engine(tensors, config)
    header = struct.unpack_from("<7i", checkpoint, 0)
    assert header[:5] == (32, 64, config["num_hidden_layers"], 4, config["num_key_value_heads"])
    assert checkpoint_dtype(header, len(checkpoint), options_of(config)) == "float32"
    want = naive_spark_logits(tensors, config, TOKENS)
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), pos


# every tensor of a layer: changed in one layer, the engine must follow the reference to other numbers, or the tensor
# is not read where the model's code reads it (and one of q_k_v_proj's three parts at a time)
KINDS = ["self_attn.q_k_v_proj.weight", "self_attn.g_proj.weight", "self_attn.out_proj.weight", "input_layernorm.weight",
         "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("which", [0, 2, -1])
def test_every_tensor_is_read_where_its_own_code_reads_it(kind, which):
    tensors, config = spark_model()
    layer = range(config["num_hidden_layers"])[which]
    name = f"model.layers.{layer}.{kind}"
    changed = {**tensors, name: (tensors[name] * np.float32(1.5) + np.float32(0.05)).astype(np.float32)}
    llama, _ = engine(changed, config)
    want, before = naive_spark_logits(changed, config, TOKENS), naive_spark_logits(tensors, config, TOKENS)
    assert not close(want[-1], before[-1])
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), (name, pos)


@pytest.mark.parametrize("part", ["q", "k", "v"])
@pytest.mark.parametrize("kind", ["sliding", "full"])
def test_each_part_of_the_one_matrix_is_cut_where_its_own_code_cuts_it(part, kind):
    """q, k and v of a sliding-attention layer and of a full-attention one, each changed alone: q and k are turned
    over another width in each."""
    tensors, config = spark_model()
    layer = config["layer_types"].index(f"{kind}_attention")
    name = f"model.layers.{layer}.self_attn.q_k_v_proj.weight"
    q, kv = 4 * 16, 2 * 16
    start, stop = {"q": (0, q), "k": (q, q + kv), "v": (q + kv, q + 2 * kv)}[part]
    values = tensors[name].copy()
    values[start:stop] = values[start:stop][::-1] * np.float32(1.3)
    changed = {**tensors, name: values}
    llama, _ = engine(changed, config)
    want = naive_spark_logits(changed, config, TOKENS)
    assert not close(want[-1], naive_spark_logits(tensors, config, TOKENS)[-1])
    for pos, token in enumerate(TOKENS):
        assert close(llama.forward(token, pos), want[pos]), (part, kind, pos)


def test_a_head_is_gated_by_its_own_value():
    """The gate of the heads: the rows of g_proj swapped two at a time, each head's output follows its own row (an
    engine that gated every value by one number, or a head by another's, would not follow the reference)."""
    tensors, config = spark_model(kinds="ll")
    for first, second in ((0, 1), (0, 3), (1, 2)):
        gates = tensors["model.layers.0.self_attn.g_proj.weight"].copy()
        gates[[first, second]] = gates[[second, first]] * np.float32(3.0)
        changed = {**tensors, "model.layers.0.self_attn.g_proj.weight": gates}
        want = naive_spark_logits(changed, config, TOKENS)
        assert not close(want[-1], naive_spark_logits(tensors, config, TOKENS)[-1])
        llama, _ = engine(changed, config)
        for pos, token in enumerate(TOKENS):
            assert close(llama.forward(token, pos), want[pos]), (first, second, pos)


def test_the_gelu_is_the_exact_one():
    """gelu_erf() against erf itself, and how far the tanh form (a Gemma's) is from it: an engine that took the one for
    the other would be off by that much in every FFN."""
    x = np.linspace(-14, 14, 20001, dtype=np.float32)
    exact = np.array([0.5 * float(v) * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in x])
    assert np.abs(gelu_erf(x) - exact).max() < 5e-7
    from llama2_numpy import gelu
    assert np.abs(gelu(x) - exact).max() > 4e-4
    # the two kinds of numbers it is never handed, as the kernel reads them; and its sign at zero
    assert np.isnan(gelu_erf(np.float32(np.nan))) and gelu_erf(np.float32(np.inf)) == np.inf
    assert np.signbit(gelu_erf(np.float32(-0.0))) and gelu_erf(np.float32(-40.0)) <= 0


@pytest.mark.parametrize("window", [1, 2, 4, 5])
def test_a_local_layer_attends_its_window_and_nothing_before_it(window):
    """A value changed at a position the window has left behind changes nothing of a local layer's later outputs."""
    tensors, config = spark_model(kinds="ss", window=window)
    llama, _ = engine(tensors, config)
    first = [llama.forward(token, pos) for pos, token in enumerate(TOKENS)]
    again, _ = engine(tensors, config)
    other = [again.forward(token if pos else 300, pos) for pos, token in enumerate(TOKENS)]
    # (two layers of a window w: position 0 reaches no later than position 2 (w - 1))
    for pos in range(len(TOKENS)):
        assert np.array_equal(first[pos], other[pos]) == (pos > 2 * (window - 1)), pos


def test_the_conversion_tells_the_engine_what_the_file_cannot():
    """The lesson of T.72: the engine built from exactly the options the conversion gives runs like the reference."""
    for model in ("small", "a classifier of its own", "every layer attends all", "half of a head turned, the bases the other way round",
                  "heads that fill the width"):
        shape = MODELS[model]
        tensors, config = spark_model(**shape)
        made = conversion(tensors, config)
        kinds = shape.get("kinds", "sssl")
        assert made.options["arch"] == "spark" and made.options["bias"] is False
        if "s" in kinds:
            assert made.options["local"] == {"layers": [layer for layer, kind in enumerate(kinds) if kind == "s"],
                                             "window": 4, "rope_theta": shape.get("local_theta", 100.0)}
        else:
            assert "local" not in made.options
        head = shape.get("head_dim", 16)
        assert made.options["rotary"] == int(head * shape.get("part", 0.25))
        assert made.options["rope_theta"] == shape.get("theta", 5e6) and made.options["rms_norm_eps"] == 1e-6
        assert made.options.get("head_dim") == (None if head == 8 else 16)
        assert not {"qk_norm", "linear", "convolution", "rotated", "unturned", "rope_scaling"} & set(made.options)
        passed = {key: value for key, value in made.options.items()
                  if key in ("dtype", "arch", "bias", "head_dim", "local", "rope_theta", "rms_norm_eps", "rotary")}
        llama = Llama(bytes(made.stream.out), pack_tokenizer(tiny_vocab(config["vocab_size"])), **passed)
        want = naive_spark_logits(tensors, config, TOKENS)
        for pos, token in enumerate(TOKENS):
            assert close(llama.forward(token, pos), want[pos]), (model, pos)
        assert checkpoint_size(made.stream.header, "float32", made.options) == len(made.stream.out)
        assert form_of(made.options) == {**FORM, "arch": "spark", "head_dim": made.options.get("head_dim", 0)}
        assert (made.stream.header[5] < 0) is (not shape.get("shared", True))


def test_the_facts_of_the_architecture():
    """What no other architecture has: the exact GELU, and the gate of the heads as a row of its file, behind wo."""
    assert LAYOUTS["spark"].activation == GELU_ERF and LAYOUTS["spark"].rope == PARTLY and LAYOUTS["spark"].local
    assert all(layout.activation != GELU_ERF for arch, layout in LAYOUTS.items() if arch != "spark")
    names = [row.name for row in tensor_rows((32, 64, 4, 4, 2, 320, 24), {"arch": "spark", "head_dim": 16})]
    assert names == ["token_embedding_table", "rms_att_weight", "wq", "wk", "wv", "wo", "head_gate", "rms_ffn_weight",
                     "w1", "w2", "w3", "rms_final_weight", "freq_cis_real", "freq_cis_imag"]


def test_the_pages_engine_is_told_how_much_of_each_layers_heads_turns():
    """The plan's facts of each layer (engine/plan.py's layer_facts()): a local layer turns its whole head by the local
    tables and attends its window, the others turn the options' part of theirs. (A local layer turned by the others'
    part moves a made-up model's logits by a few 1e-4 alone, within plan-facts-check's line: held here by the facts.)"""
    from llama2_numpy import Dims, Settings, layer_facts
    local = {"layers": [0, 1, 2, 4], "window": 4, "rope_theta": 100.0}
    facts = layer_facts(Dims((32, 64, 6, 4, 2, 320, 24), {"arch": "spark", "head_dim": 16}),
                        Settings(int8=False, kv_start=16, rotary=4, local=local))
    assert [(fact["rotary"], fact["local"], fact["window"], fact["head"]) for fact in facts] == \
        [(16, True, 4, 16)] * 3 + [(4, False, 0, 16), (16, True, 4, 16), (4, False, 0, 16)]
    assert all(fact["rope"] and fact["kind"] == "attention" for fact in facts)


# the two published config.json, as they are (but the kinds of the layers, which the 1.7B lists 28 and the 4B 36 of)
PUBLISHED = {
    "1.7B": dict(hidden_size=2048, intermediate_size=6656, num_attention_heads=8, num_hidden_layers=28, num_key_value_heads=2),
    "4B": dict(hidden_size=2560, intermediate_size=10240, num_attention_heads=16, num_hidden_layers=36, num_key_value_heads=4),
}


@pytest.mark.parametrize("model", PUBLISHED)
def test_the_published_configs_read_as_the_models_code_reads_them(model):
    numbers = PUBLISHED[model]
    config = {"architectures": ["Spark2_5ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 0,
              "dtype": "bfloat16", "eos_token_id": 1, "gate_attn_act_mode": "sigmoid", "head_dim": 256,
              "headwise_attn_output_gate": True, "hidden_act": "gelu", "initializer_range": 0.02,
              "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * (numbers["num_hidden_layers"] // 4),
              "max_position_embeddings": 1048576, "mlp_bias": False, "model_type": "spark2_5", "pad_token_id": 2,
              "rms_norm_eps": 1e-06,
              "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
                                  "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000}},
              "sliding_window": 512, "tie_word_embeddings": True, "transformers_version": "4.57.1", "use_cache": True,
              "vocab_size": 131072, **numbers}
    read = normalize(config)
    check_config(read)
    assert normalize(read) == read
    assert (read["rope_theta"], read["rotary_pct"], read["rope_local_base_freq"]) == (5000000, 0.25, 10000)
    assert turns(read) == (256, 256, 256, 64) * (numbers["num_hidden_layers"] // 4)
    assert local(read) == {"layers": [l for l in range(numbers["num_hidden_layers"]) if l % 4 != 3], "window": 512, "rope_theta": 10000.0}
    form = checkpoint_form(read, {})
    assert form["arch"] == "spark" and form["head_dim"] == (0 if model == "1.7B" else 256)
    header = llama2_convert.checkpoint_header(read, {}, 4096)
    assert header == (numbers["hidden_size"], numbers["intermediate_size"], numbers["num_hidden_layers"],
                      numbers["num_attention_heads"], numbers["num_key_value_heads"], 131072, 4096)


REFUSED = [
    ({"hidden_act": "silu"}, "its activation is silu, not gelu"),
    ({"attention_bias": True}, "its layers have biases"),
    ({"mlp_bias": True}, "its layers have biases"),
    ({"headwise_attn_output_gate": False}, "otherwise than by a sigmoid"),
    ({"gate_attn_act_mode": "silu"}, "otherwise than by a sigmoid"),
    ({"layer_types": ["sliding_attention"] * 3}, "one kind for every layer"),
    ({"layer_types": ["sliding_attention", "chunked_attention", "sliding_attention", "full_attention"]}, "one kind for every layer"),
    ({"sliding_window": None}, "no usable sliding_window"),
    ({"rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5e6},
                          "sliding_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1e4}}}, "turns its heads"),
    ({"rope_parameters": {"full_attention": {"partial_rotary_factor": 0.0, "rope_theta": 5e6},
                          "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 1e4}}}, "turns its heads"),
    ({"rope_parameters": {"full_attention": {"rope_type": "yarn", "factor": 4.0, "partial_rotary_factor": 0.25, "rope_theta": 5e6},
                          "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 1e4}}}, "RoPE scaling of the yarn kind"),
]


@pytest.mark.parametrize("change, reason", REFUSED)
def test_a_spark_the_engine_cannot_run_is_refused(change, reason):
    _, config = spark_model()
    check_config(normalize(config))
    with pytest.raises(ValueError, match=reason):
        check_config(normalize({**config, **change}))


def test_the_engine_reads_the_tensors_where_the_layout_puts_them():
    """Every row of the file as the engine holds it against the Hugging Face tensor it is made of, in every dtype's
    file: the sizes agree (checkpoint_dtype()), and the float32 file's rows are the tensors."""
    tensors, config = spark_model(shared=False)
    for dtype in ("float32", "float16", "int8", "int6"):
        llama, checkpoint = engine(tensors, config, dtype)
        header = struct.unpack_from("<7i", checkpoint, 0)
        assert checkpoint_dtype(header, len(checkpoint), options_of(config)) == dtype
        assert llama.wq.shape == (4, 64, 32) and llama.wk.shape == (4, 32, 32) and llama.head_gate.shape == (4, 4, 32)
        tolerance = {"float32": 0, "float16": 1e-3, "int8": 0.02, "int6": 0.05}[dtype]
        for layer in (0, 3):
            qkv = tensors[f"model.layers.{layer}.self_attn.q_k_v_proj.weight"]
            assert np.allclose(llama.wv[layer], qkv[96:], atol=tolerance), dtype
            for ours, theirs in (("wo", "self_attn.out_proj"), ("head_gate", "self_attn.g_proj"), ("w1", "mlp.gate_proj"),
                                 ("w2", "mlp.down_proj"), ("w3", "mlp.up_proj")):
                assert np.allclose(getattr(llama, ours)[layer], tensors[f"model.layers.{layer}.{theirs}.weight"], atol=tolerance), (dtype, ours)


# ---- a GGUF
GGUF_NAMES = {"model.embedding.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
              "lm_head.weight": "output.weight"}
GGUF_LAYER = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_k_v_proj": "attn_qkv",
              "self_attn.g_proj": "attn_gate", "self_attn.out_proj": "attn_output", "mlp.gate_proj": "ffn_gate",
              "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
# rows of whole groups of 32 (Q8_0)
SPARK = dict(dim=64, hidden_dim=128, head_dim=32)


def spark_gguf(more=(), leave=(), **shape):
    """A GGUF v3 of a small Spark-X2.5 the way XHToken's Q8_0 of the 1.7B is: Q8_0 matrices (the gate of the heads
    too), F32 norms as they are, q, k and v one matrix as Hugging Face holds it, the last norm first, which layers have
    the window as a list, the two bases and the two widths RoPE turns. Returns the config.json, the file, and under the
    Hugging Face names the values it stands for."""
    tensors, config = spark_model(**{**SPARK, **shape})
    stored, same = {}, {}
    for name in sorted(tensors, key=lambda name: name != "model.norm.weight"):
        tensor = tensors[name]
        if name in GGUF_NAMES:
            gguf = GGUF_NAMES[name]
        else:
            _, _, layer, *rest = name.split(".")
            gguf = f"blk.{layer}.{GGUF_LAYER['.'.join(rest[:-1])]}.{rest[-1]}"
        if tensor.ndim == 2:
            blob, held = q8_0_blocks(tensor)
            stored[gguf], same[name] = [blob, 8, tensor.shape], held
        else:
            stored[gguf], same[name] = [tensor.astype(np.float32).tobytes(), 0, tensor.shape], tensor
    arch, rope = "spark2_5", config["rope_parameters"]
    metadata = [("general.architecture", 8, arch), (f"{arch}.block_count", 4, config["num_hidden_layers"]),
                (f"{arch}.context_length", 4, config["max_position_embeddings"]),
                (f"{arch}.embedding_length", 4, config["hidden_size"]),
                (f"{arch}.feed_forward_length", 4, config["intermediate_size"]),
                (f"{arch}.attention.head_count", 4, config["num_attention_heads"]),
                (f"{arch}.attention.head_count_kv", 4, config["num_key_value_heads"]),
                (f"{arch}.attention.layer_norm_rms_epsilon", 6, config["rms_norm_eps"]),
                (f"{arch}.attention.key_length", 4, config["head_dim"]),
                (f"{arch}.attention.value_length", 4, config["head_dim"]),
                (f"{arch}.rope.freq_base", 6, float(rope["full_attention"]["rope_theta"])),
                (f"{arch}.rope.freq_base_swa", 6, float(rope["sliding_attention"]["rope_theta"])),
                (f"{arch}.rope.dimension_count", 4, int(config["head_dim"] * rope["full_attention"]["partial_rotary_factor"])),
                (f"{arch}.rope.dimension_count_swa", 4, int(config["head_dim"] * rope["sliding_attention"]["partial_rotary_factor"])),
                (f"{arch}.attention.sliding_window", 4, config["sliding_window"]),
                (f"{arch}.attention.sliding_window_pattern", 9, (7, [kind == "sliding_attention" for kind in config["layer_types"]])),
                ("tokenizer.ggml.model", 8, "llama"), ("tokenizer.ggml.pre", 8, "default"),
                ("tokenizer.ggml.bos_token_id", 4, 0), ("tokenizer.ggml.eos_token_id", 4, 1)]
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


def safetensors_conversion(same, config, vocabulary, dtype):
    safetensors = safetensors_file(same)
    size = struct.unpack("<Q", safetensors[:8])[0]
    expected = llama2_convert.Conversion(safetensors[8:8 + size].decode(), 8 + size, json.dumps(config), vocabulary,
                                         "tokenizer.json", dtype=dtype, max_seq_len=1 << 20)
    expected.feed(safetensors)
    expected.finish()
    return expected


@pytest.mark.parametrize("dtype", ["int8", "float32", "float16", "int6"])
@pytest.mark.parametrize("shape", [dict(), dict(shared=False), dict(kinds="slssl", window=3)])
def test_a_spark_gguf_with_the_originals_files_is_the_safetensors_conversion(shape, dtype, monkeypatch):
    """The list's way in: the checkpoint, tokenizer.bin and options of the safetensors of the same values, to the byte."""
    monkeypatch.setattr("convert.stream.PIECE", 700)
    config, file, same = spark_gguf(**shape)
    vocabulary = unigram(config["vocab_size"])
    got = with_original(file, config, vocabulary, "tokenizer.json", dtype)
    expected = safetensors_conversion(same, config, vocabulary, dtype)
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options
    kinds = shape.get("kinds", "sssl")
    assert got.options["arch"] == "spark" and got.options["local"]["layers"] == [l for l, kind in enumerate(kinds) if kind == "s"]
    if dtype == "float32":
        options = {**options_of(config), "head_dim": 0 if config["head_dim"] * 4 == config["hidden_size"] else config["head_dim"]}
        llama = Llama(bytes(got.checkpoint), pack_tokenizer(tiny_vocab(config["vocab_size"])), **options)
        want = naive_spark_logits(same, config, TOKENS)
        for pos, token in enumerate(TOKENS):
            assert close(llama.forward(token, pos), want[pos]), pos


@pytest.mark.parametrize("change, what", [
    ({"layer_types": ["sliding_attention"] * 4}, "sliding-attention layers"),
    ({"sliding_window": 6}, "sliding-attention layers"),
    ({"rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5e6},
                          "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 1e3}}}, "sliding-attention layers"),
    ({"rope_parameters": {"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 5e6},
                          "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 100.0}}}, "part of a head RoPE turns"),
    ({"rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e4},
                          "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 100.0}}}, "RoPE theta"),
    ({"rms_norm_eps": 1e-5}, "RMSNorm epsilon"),
    ({"head_dim": 16}, "size of a head"),
    ({"model_type": "llama"}, "architecture"),
])
def test_a_spark_gguf_that_is_not_the_originals_is_refused(change, what):
    """What no tensor shows: which layers are local, their window, the two bases, how much of a head turns."""
    config, file, _ = spark_gguf()
    llama2_convert.gguf_weights(file, json.dumps(config))  # its own config goes through
    with pytest.raises(ValueError, match=f"its {what} is "):
        llama2_convert.gguf_weights(file, json.dumps({**config, **change}))
