# Made-up models for the tests and their tools, in one place (T357.2). Nothing here reads pytest or conftest.py, so that a
# tool (tests/make_*.py, tests/plan_facts.py) and a Pyodide (tests/gpu-check.mjs, tests/forward-check.mjs) import it as
# it is: whoever imports it has put the engine's Python on sys.path first (tests/tree.py's python_folder()).
#
# What Hugging Face would publish, family by family: the tensors by transformers' names and shapes and the config.json,
# written out by hand. These are the converter's input and the tests' own word of what a published model is: they do
# not read the converter's table of sources (convert/families/), or a mistake there would be in the input too.
import json
import struct

import numpy as np

import llama2_numpy


# ------------------------------------------------------------------------------------------- a Llama
def rope_tables(seq_len, head_size, rope_theta=10000.0):
    angles = np.arange(seq_len)[:, None] / rope_theta ** (np.arange(0, head_size, 2) / head_size)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)



def synthetic_weights(dim=32, hidden_dim=64, n_layers=2, n_heads=4, n_kv_heads=4,
                      vocab_size=320, seq_len=24, shared=True, seed=0, head_size=0):
    """Random weights for a tiny model, plus its configuration. Small enough for a naive reference.
    head_size: where a head is not dim / n_heads (T124), q and the attention's output are n_heads * head_size wide."""
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
    """metadata: more of the header's __metadata__ (T237: a rotated basis, as gguf_model() says one)."""
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


# ------------------------------------------------------------------------------------- Qwen3.5 (T229)
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


# T237: a model folded into a rotated basis (tests/test_rotated.py, tests/make_qwen35.py)
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


# ------------------------------------------------------------------------------------- LFM2 (T260)
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
