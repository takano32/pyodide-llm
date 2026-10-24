# Made-up models for the tests and their tools, in one place (T.357.2). Nothing here reads pytest or conftest.py, so that a
# tool (tests/make_*.py, tests/plan_facts.py) and a Pyodide (tests/gpu-check.mjs, tests/forward-check.mjs) import it as
# it is: whoever imports it has put the engine's Python on sys.path first (tests/tree.py's python_folder()).
#
# The file of the engine: one maker, checkpoint(), which lays a model out by the engine's own rows (engine/layout.py)
# and stores each as the engine's registry of dtypes does (engine/dtypes.py): no order and no shape of a tensor is
# written down here, and what a made-up model's values are is a function of the row (named(), uniform(), drawn()).
#
# What Hugging Face would publish, family by family: the tensors by transformers' names and shapes and the config.json,
# written out by hand. These are the converter's input and the tests' own word of what a published model is: they do
# not read the converter's table of sources (convert/families/), or a mistake there would be in the input too.
import json
import struct
from pathlib import Path

import numpy as np

import llama2_numpy
from engine.dtypes import BLOCK, BLOCKS, KINDS
from engine.layout import CODES, VECTOR, check_suited, kind_of, tensor_rows


# ------------------------------------------------------------------------------------------- a Llama
def rope_tables(seq_len, head_size, rope_theta=10000.0):
    angles = np.arange(seq_len)[:, None] / rope_theta ** (np.arange(0, head_size, 2) / head_size)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)



def synthetic_weights(dim=32, hidden_dim=64, n_layers=2, n_heads=4, n_kv_heads=4,
                      vocab_size=320, seq_len=24, shared=True, seed=0, head_size=0):
    """Random weights for a tiny model, plus its configuration. Small enough for a naive reference.
    head_size: where a head is not dim / n_heads (T.124), q and the attention's output are n_heads * head_size wide."""
    rng = np.random.default_rng(seed)
    head_size = head_size or dim // n_heads
    q_dim, kv_dim = n_heads * head_size, n_kv_heads * head_size
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    cos, sin = rope_tables(seq_len, head_size)
    weights = {
        "token_embedding_table": normal(vocab_size, dim),
        "rms_att_weight": (1.0 + normal(n_layers, dim) * 0.1).astype(np.float32),
        "wq": normal(n_layers, q_dim, dim), "wk": normal(n_layers, kv_dim, dim),
        "wv": normal(n_layers, kv_dim, dim), "wo": normal(n_layers, dim, q_dim),
        "rms_ffn_weight": (1.0 + normal(n_layers, dim) * 0.1).astype(np.float32),
        "w1": normal(n_layers, hidden_dim, dim), "w2": normal(n_layers, dim, hidden_dim),
        "w3": normal(n_layers, hidden_dim, dim),
        "rms_final_weight": (1.0 + normal(dim) * 0.1).astype(np.float32),
        "freq_cis_real": cos, "freq_cis_imag": sin,
    }
    weights["wcls"] = weights["token_embedding_table"] if shared else normal(vocab_size, dim)
    config = dict(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
                  n_kv_heads=n_kv_heads, vocab_size=vocab_size, seq_len=seq_len,
                  head_size=head_size, q_dim=q_dim, kv_dim=kv_dim, shared=shared)
    return config, weights


def hugging_face(config, weights, shared):
    """The tensors and the config.json Hugging Face would publish for these llama2.c weights."""
    dim, n_heads, n_kv_heads = config["dim"], config["n_heads"], config["n_kv_heads"]
    head_size = config["head_size"]

    def permute(w, heads):  # llama2.c's adjacent pairs -> [first halves, second halves]: what convert undoes
        return w.reshape(heads, head_size // 2, 2, w.shape[1]).transpose(0, 2, 1, 3).reshape(w.shape)

    tensors = {"model.embed_tokens.weight": weights["token_embedding_table"], "model.norm.weight": weights["rms_final_weight"]}
    names = dict(rms_att_weight="input_layernorm", wq="self_attn.q_proj", wk="self_attn.k_proj", wv="self_attn.v_proj",
                 wo="self_attn.o_proj", rms_ffn_weight="post_attention_layernorm", w1="mlp.gate_proj", w2="mlp.down_proj",
                 w3="mlp.up_proj")
    for ours, theirs in names.items():
        for layer in range(config["n_layers"]):
            tensor = weights[ours][layer]
            if ours in ("wq", "wk"):
                tensor = permute(tensor, n_heads if ours == "wq" else n_kv_heads)
            tensors[f"model.layers.{layer}.{theirs}.weight"] = np.ascontiguousarray(tensor)
    if not shared:
        tensors["lm_head.weight"] = weights["wcls"]
    published = dict(model_type="llama", hidden_size=dim, intermediate_size=config["hidden_dim"],
                     num_hidden_layers=config["n_layers"], num_attention_heads=n_heads, num_key_value_heads=n_kv_heads,
                     vocab_size=config["vocab_size"], max_position_embeddings=config["seq_len"], rope_theta=10000.0,
                     tie_word_embeddings=shared)
    if head_size != dim // n_heads:
        published["head_dim"] = head_size
    return tensors, published


def safetensors_file(tensors, stored="F32", metadata=None):
    """metadata: more of the header's __metadata__ (T.237: a rotated basis, as gguf_model() says one)."""
    header, data = {"__metadata__": {"format": "pt", **(metadata or {})}}, b""
    for name, tensor in tensors.items():
        if stored == "BF16":  # the upper half of the float32: what bfloat16 is
            raw = (np.ascontiguousarray(tensor, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        else:
            raw = np.ascontiguousarray(tensor, dtype={"F32": np.float32, "F16": np.float16}[stored]).tobytes()
        header[name] = {"dtype": stored, "shape": list(tensor.shape), "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + data


# ------------------------------------------------------------------------------------------- Qwen2, Qwen3
def qwen2(config, weights, shared, seed=7):
    """What Hugging Face would publish for a Qwen2 with these weights: the biases too, permuted like wq and wk."""
    tensors, published = hugging_face(config, weights, shared)
    head_size = config["dim"] // config["n_heads"]
    rng = np.random.default_rng(seed)
    permute = lambda v, heads: v.reshape(heads, head_size // 2, 2).transpose(0, 2, 1).reshape(-1)
    for name, ours, heads in (("q_proj", "bq", config["n_heads"]), ("k_proj", "bk", config["n_kv_heads"]),
                              ("v_proj", "bv", None)):
        rows = config["dim"] if ours == "bq" else config["n_kv_heads"] * head_size
        weights[ours] = (rng.standard_normal((config["n_layers"], rows)) * 0.3).astype(np.float32)
        for layer in range(config["n_layers"]):
            vector = weights[ours][layer]
            tensors[f"model.layers.{layer}.self_attn.{name}.bias"] = \
                np.ascontiguousarray(permute(vector, heads) if heads else vector)
    return tensors, {**published, "model_type": "qwen2"}


def qwen3(config, weights, shared, seed=11):
    """What Hugging Face would publish for a Qwen3 with these weights: the norms of q and k, in Hugging Face's order
    of a head (first halves, then second halves), which the converter interleaves like the rows of wq and wk."""
    tensors, published = hugging_face(config, weights, shared)
    head_size = config["head_size"]
    rng = np.random.default_rng(seed)
    for name in ("q_norm", "k_norm"):
        weights[name] = (1.0 + rng.standard_normal((config["n_layers"], head_size)) * 0.3).astype(np.float32)
        for layer in range(config["n_layers"]):
            ours = weights[name][layer]
            tensors[f"model.layers.{layer}.self_attn.{name}.weight"] = \
                np.ascontiguousarray(ours.reshape(head_size // 2, 2).T.reshape(-1))
    return tensors, {**published, "model_type": "qwen3"}


# ------------------------------------------------------------------------------------- Qwen3.5 (T.229)
def qwen35_model(dim=32, hidden_dim=64, n_layers=4, every=2, n_heads=4, n_kv_heads=2, head_dim=16, rotary=0.25,
                 key_heads=2, value_heads=4, key_dim=8, value_dim=6, conv=4, vocab_size=320, seq_len=24, shared=True,
                 prefix="model.language_model.", seed=0, eps=1e-6):
    """Random Hugging Face tensors of a tiny Qwen3.5 (hybrid attention), with the names and shapes of
    Qwen/Qwen3.5-0.8B's model.safetensors, and its config.json (the language model's under text_config, as there)."""
    rng = np.random.default_rng(seed)
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    keys, values = key_heads * key_dim, value_heads * value_dim
    kinds = ["linear_attention" if (layer + 1) % every else "full_attention" for layer in range(n_layers)]
    tensors = {prefix + "embed_tokens.weight": normal(vocab_size, dim), prefix + "norm.weight": normal(dim)}
    for layer, kind in enumerate(kinds):
        p = f"{prefix}layers.{layer}."
        tensors[p + "input_layernorm.weight"] = normal(dim)  # around zero: the model multiplies by 1 + weight
        tensors[p + "post_attention_layernorm.weight"] = normal(dim)
        if kind == "full_attention":
            tensors[p + "self_attn.q_proj.weight"] = normal(2 * n_heads * head_dim, dim)
            tensors[p + "self_attn.k_proj.weight"] = normal(n_kv_heads * head_dim, dim)
            tensors[p + "self_attn.v_proj.weight"] = normal(n_kv_heads * head_dim, dim)
            tensors[p + "self_attn.o_proj.weight"] = normal(dim, n_heads * head_dim)
            tensors[p + "self_attn.q_norm.weight"] = normal(head_dim)
            tensors[p + "self_attn.k_norm.weight"] = normal(head_dim)
        else:
            tensors[p + "linear_attn.in_proj_qkv.weight"] = normal(2 * keys + values, dim)
            tensors[p + "linear_attn.in_proj_z.weight"] = normal(values, dim)
            tensors[p + "linear_attn.in_proj_b.weight"] = normal(value_heads, dim)
            tensors[p + "linear_attn.in_proj_a.weight"] = normal(value_heads, dim)
            tensors[p + "linear_attn.conv1d.weight"] = normal(2 * keys + values, 1, conv)
            tensors[p + "linear_attn.dt_bias"] = normal(value_heads)
            tensors[p + "linear_attn.A_log"] = normal(value_heads)
            tensors[p + "linear_attn.norm.weight"] = (1.0 + normal(value_dim)).astype(np.float32)
            tensors[p + "linear_attn.out_proj.weight"] = normal(dim, values)
        tensors[p + "mlp.gate_proj.weight"] = normal(hidden_dim, dim)
        tensors[p + "mlp.up_proj.weight"] = normal(hidden_dim, dim)
        tensors[p + "mlp.down_proj.weight"] = normal(dim, hidden_dim)
    if not shared:
        tensors["lm_head.weight"] = normal(vocab_size, dim)
    # what transformers leaves unread: the vision model and the look-ahead head
    tensors["model.visual.patch_embed.proj.bias"] = normal(8)
    tensors["mtp.norm.weight"] = normal(dim)
    text = dict(model_type="qwen3_5_text", hidden_size=dim, intermediate_size=hidden_dim, num_hidden_layers=n_layers,
                num_attention_heads=n_heads, num_key_value_heads=n_kv_heads, head_dim=head_dim, hidden_act="silu",
                layer_types=kinds, full_attention_interval=every, linear_conv_kernel_dim=conv,
                linear_key_head_dim=key_dim, linear_value_head_dim=value_dim, linear_num_key_heads=key_heads,
                linear_num_value_heads=value_heads, max_position_embeddings=seq_len, rms_norm_eps=eps,
                vocab_size=vocab_size, tie_word_embeddings=shared, eos_token_id=7, attn_output_gate=True,
                mlp_only_layers=[],
                rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": rotary,
                                 "mrope_interleaved": True, "mrope_section": [11, 11, 10]})
    return tensors, dict(model_type="qwen3_5", text_config=text, tie_word_embeddings=shared)


# T.237: a model folded into a rotated basis (tests/test_rotated.py, tests/make_qwen35.py)
# the matrices of a Hugging Face checkpoint that the forward pass multiplies an activation by, and the embedding
FOLDED = ("embed_tokens.weight", "lm_head.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
          "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
          "mlp.down_proj.weight", "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
          "linear_attn.out_proj.weight")


def basis(block, widths, seed=5):
    """A rotated basis with random signs for these widths, as the options say one, and the signs."""
    rng = np.random.default_rng(seed)
    signs = {width: rng.choice([-1.0, 1.0], width).astype(np.float32) for width in sorted(set(widths))}
    return {"block": block, "signs": {str(width): llama2_numpy.sign_bits(values) for width, values in signs.items()}}, signs


def folded(tensors, block, signs):
    """The tensors of a model as a file in the rotated basis holds them: every row of a matrix (and of the embedding)
    is R of the row, with the signs of the row's length."""
    return {name: llama2_numpy.rotate(tensor, signs[tensor.shape[-1]], block).astype(np.float32) if name.endswith(FOLDED) else tensor
            for name, tensor in tensors.items()}


# ------------------------------------------------------------------------------------- LFM2 (T.260)
def lfm2_model(dim=32, block_ff_dim=96, kinds="ccaccaca", n_heads=4, n_kv_heads=2, taps=3, vocab_size=320, seq_len=24,
               shared=True, seed=0, eps=1e-5, adjust=True):
    """Random Hugging Face tensors of a tiny LFM2 (convolution layers among attention layers), with the names and
    shapes of LiquidAI/LFM2.5-350M's model.safetensors, and its config.json. kinds: a letter a layer, c for a
    convolution layer and a for an attention layer. block_ff_dim is what the config says; the FFN's inside is what
    transformers' Lfm2MLP makes of it (two thirds, up to a multiple of block_multiple_of: 96 becomes 64)."""
    rng = np.random.default_rng(seed)
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    weight = lambda n: (1.0 + normal(n)).astype(np.float32)  # a norm's, as stored: the model multiplies by it
    multiple = 32
    hidden_dim = multiple * ((int(2 * block_ff_dim / 3) + multiple - 1) // multiple) if adjust else block_ff_dim
    head_dim = dim // n_heads
    tensors = {"model.embed_tokens.weight": normal(vocab_size, dim), "model.embedding_norm.weight": weight(dim)}
    for layer, kind in enumerate(kinds):
        p = f"model.layers.{layer}."
        tensors[p + "operator_norm.weight"] = weight(dim)
        tensors[p + "ffn_norm.weight"] = weight(dim)
        if kind == "a":
            tensors[p + "self_attn.q_proj.weight"] = normal(n_heads * head_dim, dim)
            tensors[p + "self_attn.k_proj.weight"] = normal(n_kv_heads * head_dim, dim)
            tensors[p + "self_attn.v_proj.weight"] = normal(n_kv_heads * head_dim, dim)
            tensors[p + "self_attn.out_proj.weight"] = normal(dim, n_heads * head_dim)
            tensors[p + "self_attn.q_layernorm.weight"] = weight(head_dim)
            tensors[p + "self_attn.k_layernorm.weight"] = weight(head_dim)
        else:
            tensors[p + "conv.in_proj.weight"] = normal(3 * dim, dim)
            tensors[p + "conv.conv.weight"] = normal(dim, 1, taps)
            tensors[p + "conv.out_proj.weight"] = normal(dim, dim)
        tensors[p + "feed_forward.w1.weight"] = normal(hidden_dim, dim)
        tensors[p + "feed_forward.w3.weight"] = normal(hidden_dim, dim)
        tensors[p + "feed_forward.w2.weight"] = normal(dim, hidden_dim)
    if not shared:
        tensors["lm_head.weight"] = normal(vocab_size, dim)
    config = dict(model_type="lfm2", architectures=["Lfm2ForCausalLM"], hidden_size=dim, block_dim=dim, conv_dim=dim,
                  block_ff_dim=block_ff_dim, intermediate_size=block_ff_dim, block_auto_adjust_ff_dim=adjust,
                  block_ffn_dim_multiplier=1.0, block_multiple_of=multiple, num_hidden_layers=len(kinds),
                  num_attention_heads=n_heads, num_heads=n_heads, num_key_value_heads=n_kv_heads,
                  layer_types=["conv" if kind == "c" else "full_attention" for kind in kinds], conv_L_cache=taps,
                  conv_bias=False, norm_eps=eps, block_norm_eps=eps, max_position_embeddings=seq_len, vocab_size=vocab_size,
                  bos_token_id=1, eos_token_id=7, pad_token_id=0, tie_embedding=shared, use_pos_enc=True,
                  rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"})
    return tensors, config


# ------------------------------------------------------------------------------------- Gemma 3 (T.261)
def gemma3_model(dim=32, hidden_dim=64, kinds="sslssl", n_heads=4, n_kv_heads=1, head_dim=16, window=4, vocab_size=320,
                 seq_len=24, shared=True, seed=0, eps=1e-6, theta=1000000.0, local_theta=100.0, listed=True, pattern=None,
                 parameters=False):
    """Random Hugging Face tensors of a tiny Gemma 3 (the text model), with the names and shapes of
    unsloth/gemma-3-270m-it's model.safetensors, and its config.json. kinds: a letter a layer, s for a
    sliding-attention layer and l for one that attends every position ("full_attention"). listed: the config.json
    lists the kinds (the 270M's); otherwise it says sliding_window_pattern (the 1B's: pattern, which then has to make
    kinds). parameters: the two bases of RoPE as transformers 5 writes them, one rope_parameters."""
    rng = np.random.default_rng(seed)
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    tensors = {"model.embed_tokens.weight": normal(vocab_size, dim), "model.norm.weight": normal(dim)}
    for layer in range(len(kinds)):
        p = f"model.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            tensors[f"{p}{norm}.weight"] = normal(dim)  # around zero: the model multiplies by 1 + weight
        tensors[p + "self_attn.q_proj.weight"] = normal(n_heads * head_dim, dim)
        tensors[p + "self_attn.k_proj.weight"] = normal(n_kv_heads * head_dim, dim)
        tensors[p + "self_attn.v_proj.weight"] = normal(n_kv_heads * head_dim, dim)
        tensors[p + "self_attn.o_proj.weight"] = normal(dim, n_heads * head_dim)
        tensors[p + "self_attn.q_norm.weight"] = normal(head_dim)
        tensors[p + "self_attn.k_norm.weight"] = normal(head_dim)
        tensors[p + "mlp.gate_proj.weight"] = normal(hidden_dim, dim)
        tensors[p + "mlp.up_proj.weight"] = normal(hidden_dim, dim)
        tensors[p + "mlp.down_proj.weight"] = normal(dim, hidden_dim)
    if not shared:
        tensors["lm_head.weight"] = normal(vocab_size, dim)
    config = dict(model_type="gemma3_text", architectures=["Gemma3ForCausalLM"], hidden_size=dim, intermediate_size=hidden_dim,
                  num_hidden_layers=len(kinds), num_attention_heads=n_heads, num_key_value_heads=n_kv_heads, head_dim=head_dim,
                  hidden_activation="gelu_pytorch_tanh", query_pre_attn_scalar=head_dim, sliding_window=window,
                  max_position_embeddings=seq_len, rms_norm_eps=eps, vocab_size=vocab_size, attention_bias=False,
                  attn_logit_softcapping=None, final_logit_softcapping=None, use_bidirectional_attention=False,
                  bos_token_id=2, eos_token_id=7, pad_token_id=0, rope_scaling=None)
    if not shared:
        config["tie_word_embeddings"] = False
    if listed:
        config["layer_types"] = ["sliding_attention" if kind == "s" else "full_attention" for kind in kinds]
    else:
        config["sliding_window_pattern"] = pattern
    if parameters:
        config["rope_parameters"] = {"full_attention": {"rope_type": "default", "rope_theta": theta},
                                     "sliding_attention": {"rope_type": "default", "rope_theta": local_theta}}
    else:
        config.update(rope_theta=theta, rope_local_base_freq=local_theta)
    return tensors, config


# ------------------------------------------------------------------------------------- Spark-X2.5 (T.342)
def spark_model(dim=32, hidden_dim=64, kinds="sssl", n_heads=4, n_kv_heads=2, head_dim=16, window=4, vocab_size=320,
                seq_len=24, shared=True, seed=0, eps=1e-6, theta=5000000.0, local_theta=100.0, part=0.25, local_part=1.0,
                spread=0.3):
    """Random Hugging Face tensors of a tiny Spark-X2.5, with the names and shapes of XHToken/Spark-X2.5-1.7B's
    safetensors (q, k and v in one matrix, q_k_v_proj; a value a head of the gate, g_proj), and its config.json (its
    rope_parameters a dict for each kind of layer, as published). kinds: a letter a layer, s for a sliding-attention
    layer and l for one that attends every position. part, local_part: how much of a head RoPE turns in each kind.
    spread: of every value drawn (the norms' weights are 1 and that around it), which a model of many layers, whose stream
    no norm behind its layers holds, wants small."""
    rng = np.random.default_rng(seed)
    normal = lambda *shape: (rng.standard_normal(shape) * spread).astype(np.float32)
    tensors = {"model.embedding.weight": normal(vocab_size, dim), "model.norm.weight": 1.0 + normal(dim)}
    for layer in range(len(kinds)):
        p = f"model.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"{p}{norm}.weight"] = 1.0 + normal(dim)  # the weight itself: no 1 is added
        tensors[p + "self_attn.q_k_v_proj.weight"] = normal((n_heads + 2 * n_kv_heads) * head_dim, dim)
        tensors[p + "self_attn.g_proj.weight"] = normal(n_heads, dim)
        tensors[p + "self_attn.out_proj.weight"] = normal(dim, n_heads * head_dim)
        tensors[p + "mlp.gate_proj.weight"] = normal(hidden_dim, dim)
        tensors[p + "mlp.up_proj.weight"] = normal(hidden_dim, dim)
        tensors[p + "mlp.down_proj.weight"] = normal(dim, hidden_dim)
    if not shared:
        tensors["lm_head.weight"] = normal(vocab_size, dim)
    config = dict(model_type="spark2_5", architectures=["Spark2_5ForCausalLM"], hidden_size=dim, intermediate_size=hidden_dim,
                  num_hidden_layers=len(kinds), num_attention_heads=n_heads, num_key_value_heads=n_kv_heads, head_dim=head_dim,
                  hidden_act="gelu", headwise_attn_output_gate=True, gate_attn_act_mode="sigmoid", sliding_window=window,
                  layer_types=["sliding_attention" if kind == "s" else "full_attention" for kind in kinds],
                  max_position_embeddings=seq_len, rms_norm_eps=eps, vocab_size=vocab_size, attention_bias=False,
                  mlp_bias=False, tie_word_embeddings=shared, bos_token_id=0, eos_token_id=1, pad_token_id=2,
                  rope_parameters={"full_attention": {"partial_rotary_factor": part, "rope_theta": theta},
                                   "sliding_attention": {"partial_rotary_factor": local_part, "rope_theta": local_theta}})
    return tensors, config


# ------------------------------------------------------------------------------------- Gemma 4 (T.331)
def gemma4_model(dim=32, hidden_dim=64, kinds="sslssl", n_heads=4, n_kv_heads=1, head_dim=16, wide_head=32, window=4,
                 vocab_size=320, seq_len=24, shared=True, seed=0, eps=1e-6, theta=1000000.0, local_theta=100.0,
                 proportion=0.25, softcap=30.0, whole=False, sharing=0, double=False, per_layer=0):
    """Random Hugging Face tensors of a tiny Gemma 4 (its language model), with the names and shapes of
    google/gemma-4-E2B-it's model.safetensors at 3e22461f (whole: "model.language_model." in front, as there; else
    "model.", the language model saved alone), and its config.json (whole: the whole model's, whose text_config it
    is). kinds: a letter a layer, s for a sliding-attention layer and l for a full-attention one, whose heads are
    wide_head (head_dim the others'). T.330: the last sharing layers read another's keys and values (their own k, v and
    norms are in the file all the same, as the published ones' are), and where double their FFN is twice as wide. There
    is an embedding of every layer of per_layer values (T.329: hidden_size_per_layer_input; 0, none). The norms are around
    1 (the model multiplies by the weight itself), and every layer's layer_scalar a number of its own."""
    rng = np.random.default_rng(seed)
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    around = lambda *shape: (1.0 + rng.standard_normal(shape) * 0.3).astype(np.float32)
    m = "model.language_model." if whole else "model."
    tensors = {m + "embed_tokens.weight": normal(vocab_size, dim), m + "norm.weight": around(dim)}
    if per_layer:
        tensors.update({m + "embed_tokens_per_layer.weight": normal(vocab_size, len(kinds) * per_layer),
                        m + "per_layer_model_projection.weight": normal(len(kinds) * per_layer, dim),
                        m + "per_layer_projection_norm.weight": around(per_layer)})
    for layer, kind in enumerate(kinds):
        p, size = f"{m}layers.{layer}.", wide_head if kind == "l" else head_dim
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            tensors[f"{p}{norm}.weight"] = around(dim)
        tensors[p + "self_attn.q_proj.weight"] = normal(n_heads * size, dim)
        tensors[p + "self_attn.k_proj.weight"] = normal(n_kv_heads * size, dim)
        tensors[p + "self_attn.v_proj.weight"] = normal(n_kv_heads * size, dim)
        tensors[p + "self_attn.o_proj.weight"] = normal(dim, n_heads * size)
        tensors[p + "self_attn.q_norm.weight"] = around(size)
        tensors[p + "self_attn.k_norm.weight"] = around(size)
        hidden = 2 * hidden_dim if double and layer >= len(kinds) - sharing else hidden_dim
        tensors[p + "mlp.gate_proj.weight"] = normal(hidden, dim)
        tensors[p + "mlp.up_proj.weight"] = normal(hidden, dim)
        tensors[p + "mlp.down_proj.weight"] = normal(dim, hidden)
        if per_layer:
            tensors[p + "per_layer_input_gate.weight"] = normal(per_layer, dim)
            tensors[p + "per_layer_projection.weight"] = normal(dim, per_layer)
            tensors[p + "post_per_layer_input_norm.weight"] = around(dim)
        tensors[p + "layer_scalar"] = (0.5 + rng.random(1)).astype(np.float32)
    if not shared:
        tensors["lm_head.weight"] = normal(vocab_size, dim)
    text = dict(model_type="gemma4_text", hidden_size=dim, intermediate_size=hidden_dim, num_hidden_layers=len(kinds),
                num_attention_heads=n_heads, num_key_value_heads=n_kv_heads, head_dim=head_dim, global_head_dim=wide_head,
                hidden_activation="gelu_pytorch_tanh", sliding_window=window, max_position_embeddings=seq_len,
                rms_norm_eps=eps, vocab_size=vocab_size, vocab_size_per_layer_input=vocab_size, attention_bias=False,
                final_logit_softcapping=softcap, use_bidirectional_attention=None, attention_k_eq_v=False,
                enable_moe_block=False, num_global_key_value_heads=None, num_kv_shared_layers=sharing,
                hidden_size_per_layer_input=per_layer, use_double_wide_mlp=double, tie_word_embeddings=shared,
                bos_token_id=2, eos_token_id=1, pad_token_id=0,
                layer_types=["sliding_attention" if kind == "s" else "full_attention" for kind in kinds],
                rope_parameters={"full_attention": {"rope_type": "proportional", "partial_rotary_factor": proportion,
                                                    "rope_theta": theta},
                                 "sliding_attention": {"rope_type": "default", "rope_theta": local_theta}})
    config = dict(model_type="gemma4", architectures=["Gemma4ForConditionalGeneration"], text_config=text) if whole else text
    return tensors, config


# ------------------------------------------------------------------------------------------- the engine's file
def checkpoint(header, form=None, dtype="float32", *, tensor, blocks=None):
    """The file of a model with this header (the 7 ints) and form (llama2_numpy.FORM's keys) in this dtype, as the
    engine reads it: the header, then every row of engine/layout.py's tensor_rows() in its order, each stored as its
    kind (kind_of(): the vectors of a quantized file are float32, and it has no RoPE tables; the form's "kinds", of a
    row or of each layer of a stack) by the registry of dtypes.
    tensor(row) -> the values of the row (its name, role, shape and layers: layout.py's Row), an array of the row's
    shape; it is asked for the rows the file holds, in file order, and for no other.
    T.363: a tensor of one of llama.cpp's blocks is not made of values (there is no packing them: the values are what
    its bytes read as, and any bytes are blocks). blocks(the kind's name, how many blocks) -> their bytes (drawn_blocks()
    where none is given); tensor is not asked for a row that is blocks throughout, and what it says of a layer that is
    blocks is left unread. The codebooks such a file carries are the converter's (convert/grids.py's codebooks()).
    One tensor after the other, by nobody's table of places: that the places of placed() are these is what the
    engine's reading of the file checks."""
    rows = tensor_rows(header, form)
    check_suited(rows, dtype)
    blocks = blocks or drawn_blocks()
    out = [struct.pack("<7i", *header)]
    for row in rows:
        kind = kind_of(row, dtype)
        if kind is None:
            continue
        if row.role == CODES:
            from convert.grids import codebooks  # (the converter's: only a file with blocks reads it)
            out.append(codebooks()[0])
            continue
        layers = kind if isinstance(kind, tuple) else None
        values = None
        if not all(one in BLOCKS for one in layers or (kind,)):
            values = np.asarray(tensor(row))
            if values.shape != tuple(row.shape):
                raise ValueError(f"The tensor {row.name} of this model is {tuple(row.shape)}, and the values made up for it are {values.shape}.")
        for layer, one in enumerate(layers or (kind,)):
            shape = row.shape[1:] if layers else row.shape
            stored = KINDS[one]
            if one in BLOCKS:
                out.append(bytes(blocks(one, int(np.prod(shape)) // BLOCK)))
            elif not stored.group:
                out.append(np.ascontiguousarray(values[layer] if layers else values, dtype=stored.name).tobytes())
            else:
                part = np.asarray(values[layer] if layers else values, dtype=np.float32).reshape(-1, row.shape[-1])
                packed, scales = stored.pack(part)
                out += [np.ascontiguousarray(packed).tobytes(), np.ascontiguousarray(scales, dtype=stored.scale_type).tobytes()]
    return b"".join(out)


# where the float16 scales of a block of llama.cpp's are (two bytes each; IQ1_M's is the top four bits of the last four
# words of 16 bits of its 56 bytes)
BLOCK_SCALES = {"Q2_K": (80, 82), "Q4_K": (0, 2), "IQ1_M": ()}


def drawn_blocks(seed=0):
    """blocks(name, count) -> the bytes of count blocks of llama.cpp's type called name: random bytes of one stream
    (any bytes are a block), with float16 scales of 2 ** -12 to 2 ** -11 and either sign in place of the random ones,
    which are an infinity or a NaN once in 32 and anything up to 65504 otherwise: a model's numbers stay finite and
    small."""
    rng = np.random.default_rng(seed)

    def blocks(name, count):
        raw = rng.integers(0, 256, (count, BLOCKS[name].block), dtype=np.uint8)
        scales = (rng.integers(0, 1 << 10, count) | 0x0C00 | (rng.integers(0, 2, count) << 15)).astype(np.uint16)
        if name == "IQ1_M":
            for word in range(4):
                raw[:, 49 + 2 * word] = (raw[:, 49 + 2 * word] & 15) | ((scales >> 4 * word & 15) << 4).astype(np.uint8)
        for at in BLOCK_SCALES.get(name, (0,)):
            raw[:, at], raw[:, at + 1] = scales & 255, scales >> 8
        return raw.tobytes()
    return blocks


def tensors_of(data, form=None):
    """{a row's name: its values} of a float32 checkpoint, read back by the same rows."""
    header, offset, tensors = struct.unpack_from("<7i", data, 0), 28, {}
    for row in tensor_rows(header, form):
        tensors[row.name] = np.frombuffer(data, dtype=np.float32, count=int(np.prod(row.shape)), offset=offset).reshape(row.shape)
        offset += tensors[row.name].nbytes
    assert offset == len(data), "the file is not a float32 checkpoint of this header and form"
    return tensors


# ---- what the values are
def named(weights):
    """The values a dict has under the rows' names."""
    return lambda row: weights[row.name]


def uniform(seed, low=-0.2, high=0.2):
    """Small random numbers for every tensor, of one stream in file order."""
    rng = np.random.default_rng(seed)
    return lambda row: rng.uniform(low, high, row.shape)


# the vectors of a file that are biases, by the rows' names: every other vector is the weight of a norm
BIASES = ("bq", "bk", "bv", "bo", "b1", "b2", "ln_att_bias", "ln_ffn_bias", "ln_final_bias")


def drawn(seed=0, outliers=0, spread=7, calm=False, ternary=False):
    """The values of the made-up models the GPU is checked on (tests/gpu-check.mjs), of one stream in file order: a
    matrix is 0.3 N; the weight of a norm is 1 + 0.03 N, a bias 0.3 N (T.154, T.187: around 0 as the matrices are, or
    the biases of every layer are nearly the same and 8 bits hide a GPU that reads layer 0's in every layer).
    outliers (T.226): that many weights of the final norm are 12 (GPT-2's are 12 to 17 times the others, T.92), so that
    the engine takes their channels apart in the classifier (llama2_numpy.outlier_channels), at channels 0, spread,
    2 spread, ... (mod dim: the T.232 review's spread of 37 puts some in every group of 128 of a dim of 256).
    calm (T.226's review): the layers' matrices at 0.1 and the norms' weights 1 +- 0.3, see the head of gpu-check/models.mjs.
    ternary (T.232): every group of 128 along a row of a matrix keeps the signs of its larger values (about two thirds)
    times one scale of its own, the rest are 0 (ternarized()'s, below), which the ternary dtype stores as they are."""
    rng = np.random.default_rng(seed)

    def tensor(row):
        values = (rng.standard_normal(row.shape) * 0.3).astype(np.float32)
        if row.role == VECTOR:
            values = values if row.name in BIASES else 1.0 + values * (1.0 if calm else 0.1)
            if outliers and row.name == "rms_final_weight":
                values[np.arange(outliers) * spread % row.shape[-1]] = 12.0
            # (T.271.2: a Qwen3.5's decay is -exp(A_log), below 0: one over 1 grows the state token after token, and
            # 149 tokens make it an infinity. The same numbers, below 0: no other value is drawn otherwise)
            if row.name == "decay":
                values = -values
            return values.astype(np.float32)
        if calm and row.per is not None:
            values = values / 3  # a layer's matrix (the embedding, the positions and the classifier: as they are)
        if ternary:
            groups = values.reshape(-1, 128)
            cut = np.quantile(np.abs(groups), 1 / 3, axis=1, keepdims=True)
            # (T.533: a float16, as PQ2_0 holds it: the ternary dtype keeps its scales so)
            d = (np.abs(groups).mean(axis=1, keepdims=True) * rng.uniform(0.5, 1.5, (len(groups), 1))).astype(np.float16).astype(np.float32)
            values = (np.sign(groups) * (np.abs(groups) > cut) * d).astype(np.float32).reshape(row.shape)
        return values
    return tensor


def quantized(dim=64, hidden=128, layers=2, heads=4, kv_heads=2, vocab=320, seq_len=256, seed=0, dtype="int8", outliers=0,
              calm=False, spread=7, **form):
    """A made-up checkpoint of drawn()'s values in a quantized dtype (int8 as quantize.py writes one, int6 of T.98,
    ternary of T.230) and its tokenizer.bin: grouped-query attention unless kv_heads says otherwise. form:
    llama2_numpy.FORM's keys (arch, bias, qk_norm, head_dim ...). A negative vocab (the header's own way of saying so):
    a classifier apart from the embedding."""
    tensor = drawn(seed, outliers, spread, calm, ternary=dtype == "ternary")
    return checkpoint((dim, hidden, layers, heads, kv_heads, vocab, seq_len), form, dtype, tensor=tensor), numbered(abs(vocab))


# ---- a Llama's, from synthetic_weights()
def header_of(config):
    """The 7 ints of synthetic_weights()'s configuration (a negative vocabulary: a classifier of its own)."""
    return (config["dim"], config["hidden_dim"], config["n_layers"], config["n_heads"], config["n_kv_heads"],
            config["vocab_size"] if config["shared"] else -config["vocab_size"], config["seq_len"])


def settings_of(header):
    """synthetic_weights()'s configuration of a Llama with this header, whose heads fill dim."""
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = header
    head_size = dim // n_heads
    return dict(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
                vocab_size=abs(vocab_size), seq_len=seq_len, head_size=head_size, q_dim=n_heads * head_size,
                kv_dim=n_kv_heads * head_size, shared=vocab_size > 0)


def pack_checkpoint(config, weights):
    """The float32 checkpoint of synthetic_weights()'s weights: llama2.c's file."""
    return checkpoint(header_of(config), {"head_dim": config["head_size"]}, "float32", tensor=named(weights))


# ------------------------------------------------------------------------------------------- tokenizers
def numbered(count):
    """A tokenizer.bin of count pieces <0>, <1>, ...: for a model nobody reads the text of."""
    return pack_tokenizer([(0.0, f"<{i}>".encode()) for i in range(count)])


def unigram(count):
    """A tokenizer.json of count pieces w0, w1, ..., the same."""
    return json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                       "vocab": [[f"w{i}", -float(i)] for i in range(count)]}}).encode()


def pack_tokenizer(pieces):
    """llama2.c's tokenizer.bin: max piece length, then (score, length, bytes) per piece."""
    out = [struct.pack("<i", max(len(text) for _, text in pieces))]
    out += [struct.pack("<fi", score, len(text)) + text for score, text in pieces]
    return b"".join(out)


def tiny_vocab(vocab_size=320):
    """A vocabulary of that many pieces: <unk>, byte fallbacks, then words and characters.

    Byte and control pieces get the unmatchable score convert_hf.py writes, so they never match text directly.
    """
    pieces = [(-1e9, b"<unk>"), (-1e9, b"<s>"), (-1e9, b"</s>")]
    pieces += [(-1e9, b"<0x%02X>" % byte) for byte in range(256)]
    words = [" ", " the", " cat", " s", "a", "t", "o", "n", "e", "h", "c", " a", " o",
             "流", "行", "、", " 流行り", "Ａ", "!", "?", " \n", "\n", " hello", " world",
             " he", "l", "w", "r", "d", " t", "i", "g", "猫", " 猫", "り"]
    for i, word in enumerate(words):
        pieces.append((-float(i) * 0.5, word.encode("utf-8")))
    assert len(pieces) <= vocab_size, "the vocabulary does not fit"
    while len(pieces) < vocab_size:  # padding rows, never matchable
        pieces.append((-1e9, b"<pad%d>" % len(pieces)))
    return pieces[:vocab_size]


# ------------------------------------------------------------------------------------------- a model as files
def files(out, tensors, config, dtype, seq_len, vocab, metadata=None):
    """<out>.bin, <out>.tokenizer.bin and <out>.json (as tests/perplexity_prepare.py writes them) of the model Hugging
    Face would publish as these tensors and this config.json, converted the way the page converts them
    (llama2_convert.Conversion), for the checks of forward.js that take a model by its files. vocab: the pieces of its
    vocabulary. metadata: more of the safetensors header's __metadata__ (T.237: a rotated basis)."""
    import llama2_convert  # (here and not above: everything else of this file needs the engine alone)
    file = safetensors_file(tensors, metadata=metadata)
    size = struct.unpack("<Q", file[:8])[0]
    conversion = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), unigram(vocab),
                                           "tokenizer.json", dtype=dtype, max_seq_len=seq_len, start=8 + size)
    conversion.feed(file[8 + size:])
    conversion.finish()
    Path(f"{out}.bin").write_bytes(bytes(conversion.checkpoint))
    Path(f"{out}.tokenizer.bin").write_bytes(conversion.tokenizer)
    Path(f"{out}.json").write_text(json.dumps(conversion.options))
    print(f"{out}.bin: {len(conversion.checkpoint)} bytes ({dtype}), {json.dumps(conversion.options)}")
