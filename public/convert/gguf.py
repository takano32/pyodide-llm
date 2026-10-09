# GGUF as a source (T74): the header's metadata and tensor table shown as a safetensors-like header, what llama.cpp
# wrote differently put back, and the checks against the original's config.json.
import json
import math
import struct

import numpy as np

from llama2_numpy import rope_frequencies, sign_bits
from convert.readers import BLOCKS, READERS
from convert.sources import ROTATED
from convert.config import (GRANITE_ONES, architecture, convolution_layers, head_size, linear_layers, normalize,
                            rotary_dim, unturned_layers, yarn)
from convert.tokenizer import UNMATCHABLE, tokenizer_bin


# A GGUF file holds what config.json, tokenizer.json and model.safetensors hold, in one. Only what a Q8_0, PQ2_0 or F16
# Llama, Granite, Qwen2, Qwen3, Qwen3.5, GPT-2 or GPT-NeoX needs is read; tests/gguf_check.py is the separate reference this is
# held to.
class Incomplete(Exception):
    """The GGUF header goes on past the bytes given: fetch more and try again."""


GGUF_VALUES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
# ggml's types; the K-quants and the rest are refused. 142 and 143 are PQ2_0 and PTQ1_0 of Prism ML's fork of
# llama.cpp (T235's pq2_0(), T230's ptq1_0()); 30 is BF16 (Ternary Bonsai 2's two small matrices of the gates)
GGUF_TENSORS = {0: "F32", 1: "F16", 8: "Q8_0", 30: "BF16", 142: "PQ2_0", 143: "PTQ1_0"}
# llama.cpp's names of the pre-tokenizers, as the engine knows them (llama2_numpy.pretokenize)
# (granite-docling, T253: what llama.cpp calls a Granite 4.2's ByteLevel with its regex, and splits by GPT-2's pattern.
# minicpm5, T254: llama.cpp's two patterns of that name are tokenizer.json's but for the contractions, written out by
# case, which leaves a 's after U+017F unmatched; openbmb's own GGUFs of 2026-09 still say llama-bpe)
# (lfm2, T260: llama.cpp splits an LFM2's text as a Llama 3's, which is the pattern of its tokenizer.json; it also
    # takes a piece that is in the vocabulary whole, which that tokenizer.json does not: ignore_merges is false there,
    # and gguf_tokenizer() says it of llama-bpe alone)
GGUF_PRETOKENIZERS = {"gpt-2": "gpt2", "gpt2": "gpt2", "smollm": "gpt2-digits", "qwen2": "qwen", "llama-bpe": "llama3",
                      "qwen35": "qwen35", "granite-docling": "gpt2", "minicpm5": "minicpm5", "lfm2": "llama3"}
# the ones whose tokenizer.json normalizes to NFC, which a GGUF does not say (Qwen's)
GGUF_NFC = ("qwen2", "qwen35")
# T245: the tensors of a Qwen3.5's linear-attention layer that have the value heads along an axis, as (whether q and k
# stand before them, whether a head has value_dim entries there or one, the axis). llama.cpp's conversion/qwen.py at
# dcd387a4, _LinearAttentionVReorderBase.modify_tensors (lines 584 to 633), reorders these and no other (the norm of a
# value head, linear_attn.norm, is one for all the heads)
QWEN35_TILED = {"in_proj_qkv.weight": (True, True, 0), "in_proj_z.weight": (False, True, 0),
                "in_proj_a.weight": (False, False, 0), "in_proj_b.weight": (False, False, 0),
                "dt_bias": (False, False, 0), "A_log": (False, False, 0),
                "conv1d.weight": (True, True, 0), "out_proj.weight": (False, True, 1)}
GGUF_LAYER = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_q": "self_attn.q_proj",
              "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
              "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
GGUF_NAMES = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
              "output.weight": "lm_head.weight"}
# T136's third stage: GPT-2 and GPT-NeoX, by the names of their own safetensors (openai-community/gpt2's, without
# "transformer."). For each architecture: the tensors outside the layers, where a layer's go, and the layer's names
GGUF_ARCHITECTURES = {
    "llama": (GGUF_NAMES, "model.layers.{}.", GGUF_LAYER),
    "qwen2": (GGUF_NAMES, "model.layers.{}.", GGUF_LAYER),
    # T255: a SmolLM3, a Llama to the name of every tensor (llama.cpp's SmolLM3Model is its LlamaModel by another name,
    # and turns q and k as that does; conversion/llama.py at 71ad0590)
    "smollm3": (GGUF_NAMES, "model.layers.{}.", GGUF_LAYER),
    # T253: a Granite, a Llama to the name of every tensor (llama.cpp's GraniteModel is its LlamaModel with four numbers
    # more in the metadata, and turns q and k as that does)
    "granite": (GGUF_NAMES, "model.layers.{}.", GGUF_LAYER),
    # T203 (T136's fourth stage): a Qwen3 is a Qwen2 without the biases that normalizes each head of q and k (T124).
    # llama.cpp leaves q, k and the two norms in Hugging Face's order, as a Qwen2's; the head's size is key_length
    "qwen3": (GGUF_NAMES, "model.layers.{}.", {**GGUF_LAYER, "attn_q_norm": "self_attn.q_norm",
                                                 "attn_k_norm": "self_attn.k_norm"}),
    # T236: a Qwen3.5 (T229's hybrid attention), by the names of the language model saved alone ("model." in front).
    # llama.cpp calls the second norm post_attention_norm here, the linear-attention layer's q, k and v attn_qkv, its z
    # attn_gate, and the rest ssm_* after the state-space models it shares code with. A name with a dot is all of a
    # tensor's name after its layer (llama.cpp writes dt_bias as ssm_dt.bias, and A_log as ssm_a without a ".weight")
    "qwen35": (GGUF_NAMES, "model.layers.{}.",
               {**GGUF_LAYER, "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
                "post_attention_norm": "post_attention_layernorm", "attn_qkv": "linear_attn.in_proj_qkv",
                "attn_gate": "linear_attn.in_proj_z", "ssm_alpha": "linear_attn.in_proj_a",
                "ssm_beta": "linear_attn.in_proj_b", "ssm_conv1d": "linear_attn.conv1d", "ssm_norm": "linear_attn.norm",
                "ssm_out": "linear_attn.out_proj", "ssm_dt.bias": "linear_attn.dt_bias", "ssm_a": "linear_attn.A_log"}),
    # T260: an LFM2, by transformers' names of it. llama.cpp calls the last norm token_embd_norm (the model's own name
    # for it is embedding_norm), the convolution layer's three tensors shortconv.*, and writes q and k in Hugging
    # Face's order, as a Qwen3's (its converter is no child of the Llama's: conversion/lfm2.py at f1cee994)
    "lfm2": ({"token_embd.weight": "model.embed_tokens.weight", "token_embd_norm.weight": "model.embedding_norm.weight",
              "output.weight": "lm_head.weight"}, "model.layers.{}.",
             {"attn_norm": "operator_norm", "ffn_norm": "ffn_norm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
              "attn_v": "self_attn.v_proj", "attn_output": "self_attn.out_proj", "attn_q_norm": "self_attn.q_layernorm",
              "attn_k_norm": "self_attn.k_layernorm", "ffn_gate": "feed_forward.w1", "ffn_up": "feed_forward.w3",
              "ffn_down": "feed_forward.w2", "shortconv.in_proj.weight": "conv.in_proj.weight",
              "shortconv.conv.weight": "conv.conv.weight", "shortconv.out_proj.weight": "conv.out_proj.weight"}),
    "gpt2": ({"token_embd.weight": "wte.weight", "position_embd.weight": "wpe.weight", "output_norm.weight": "ln_f.weight",
              "output_norm.bias": "ln_f.bias", "output.weight": "lm_head.weight"}, "h.{}.",
             {"attn_norm": "ln_1", "attn_qkv": "attn.c_attn", "attn_output": "attn.c_proj", "ffn_norm": "ln_2",
              "ffn_up": "mlp.c_fc", "ffn_down": "mlp.c_proj"}),
    "gptneox": ({"token_embd.weight": "gpt_neox.embed_in.weight", "output_norm.weight": "gpt_neox.final_layer_norm.weight",
                 "output_norm.bias": "gpt_neox.final_layer_norm.bias", "output.weight": "embed_out.weight"},
                "gpt_neox.layers.{}.",
                {"attn_norm": "input_layernorm", "attn_qkv": "attention.query_key_value", "attn_output": "attention.dense",
                 "ffn_norm": "post_attention_layernorm", "ffn_up": "mlp.dense_h_to_4h", "ffn_down": "mlp.dense_4h_to_h"}),
}


def gguf_read(data):
    """(metadata, tensors, base) from the first bytes of a GGUF file: tensors maps each name to its ggml type,
    its shape (outermost first, as NumPy has it) and its offset from base, where the data begins."""
    data = memoryview(data.to_py() if hasattr(data, "to_py") else data).cast("B")
    at = 0

    def take(fmt):
        nonlocal at
        size = struct.calcsize(fmt)
        if at + size > len(data):
            raise Incomplete()
        (value,) = struct.unpack_from(fmt, data, at)
        at += size
        return value

    def string():
        nonlocal at
        size = take("<Q")
        if at + size > len(data):
            raise Incomplete()
        at += size
        return bytes(data[at - size:at]).decode("utf-8", errors="replace")

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            item, count = take("<I"), take("<Q")
            return [value(item) for _ in range(count)]
        if kind not in GGUF_VALUES:
            raise ValueError(f"This GGUF file has a value of type {kind}, which is not in the format.")
        return take(GGUF_VALUES[kind])

    if len(data) >= 4 and bytes(data[:4]) != b"GGUF":
        raise ValueError("This is not a GGUF file.")
    take("<I")
    version = take("<I")
    if version not in (2, 3):
        raise ValueError(f"This GGUF file is of version {version}; only 2 and 3 are supported.")
    count, entries = take("<Q"), take("<Q")
    metadata = {}
    for _ in range(entries):
        key = string()
        metadata[key] = value(take("<I"))
    tensors = {}
    for _ in range(count):
        name = string()
        dims = [take("<Q") for _ in range(take("<I"))]
        tensors[name] = {"type": take("<I"), "shape": list(reversed(dims)), "offset": take("<Q")}
    alignment = metadata.get("general.alignment", 32)
    return metadata, tensors, (at + alignment - 1) // alignment * alignment


def gguf_model(metadata, tensors, base, rope_freqs=False):
    """The safetensors-like header (Hugging Face's names, offsets from base) and the config.json of a GGUF.
    rope_freqs: keep llama.cpp's table of Llama 3's RoPE scaling in the header, to be checked against the original's
    rope_scaling as it streams past (gguf_weights, T136), instead of refusing it."""
    arch = metadata.get("general.architecture")
    if arch not in GGUF_ARCHITECTURES:
        raise ValueError(f"This GGUF holds a {arch}: only Llama, Granite, SmolLM3, Qwen2, Qwen3, Qwen3.5, LFM2, GPT-2 and GPT-NeoX ones are supported.")
    key = lambda name, default=None: metadata.get(f"{arch}.{name}", default)
    common = {"vocab_size": tensors["token_embd.weight"]["shape"][0] if "token_embd.weight" in tensors else None,
              "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id", 1),
              "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id", 2)}
    heads = key("attention.head_count")
    if arch == "gpt2":
        # T136's third stage: config.json's own spelling, which normalize() reads. GPT-2 always shares its classifier
        # with the embedding: llama.cpp writes a copy of it as output.weight, which the conversion leaves unread
        config = {"model_type": "gpt2", "n_embd": key("embedding_length"), "n_inner": key("feed_forward_length"),
                  "n_layer": key("block_count"), "n_head": heads, "n_positions": key("context_length"),
                  "layer_norm_epsilon": key("attention.layer_norm_epsilon"), "tie_word_embeddings": True, **common}
    elif arch == "gptneox":
        dim = key("embedding_length")
        config = {"model_type": "gpt_neox", "hidden_size": dim, "intermediate_size": key("feed_forward_length"),
                  "num_hidden_layers": key("block_count"), "num_attention_heads": heads,
                  "max_position_embeddings": key("context_length"),
                  "rotary_emb_base": float(key("rope.freq_base", 10000.0)),
                  # llama.cpp says the rotated part as a number of values, config.json as a share of the head
                  "rotary_pct": key("rope.dimension_count", 0) / (dim // heads) if dim and heads else None,
                  "use_parallel_residual": bool(key("use_parallel_residual", True)),
                  "layer_norm_eps": key("attention.layer_norm_epsilon"), "hidden_act": "gelu",
                  "tie_word_embeddings": "output.weight" not in tensors, **common}
    else:
        config = {"model_type": arch, "hidden_size": key("embedding_length"), "intermediate_size": key("feed_forward_length"),
                  "num_hidden_layers": key("block_count"), "num_attention_heads": heads,
                  "num_key_value_heads": key("attention.head_count_kv", heads),
                  "max_position_embeddings": key("context_length"), "rope_theta": float(key("rope.freq_base", 10000.0)),
                  "tie_word_embeddings": "output.weight" not in tensors, "hidden_act": "silu",
                  # a head of another size than dim / heads (T124): llama.cpp says it as the length of a key
                  "head_dim": key("attention.key_length"), "rms_norm_eps": key("attention.layer_norm_rms_epsilon"),
                  **common}
        if key("rope.scaling.type", "none") not in ("none", None):
            config["rope_scaling"] = {"type": key("rope.scaling.type"), "factor": key("rope.scaling.factor", 1.0)}
            if key("rope.scaling.type") == "yarn":
                # T235: yarn's other numbers, by config.json's names. llama.cpp takes the trained context where the
                # GGUF names no original one; what else a GGUF may say of yarn, check_config() refuses by these names
                # and gguf_agrees() where the original's config.json has it not. The keys are llama.cpp's own (the
                # fork's src/llama-arch.cpp: yarn_log_multiplier, which only a DeepSeek-V2 GGUF has; the review of
                # T235 found "yarn_log_mul" here, a name no GGUF has)
                config["rope_scaling"]["original_max_position_embeddings"] = \
                    key("rope.scaling.original_context_length", key("context_length"))
                for name, ours in (("attn_factor", "attention_factor"), ("yarn_log_multiplier", "mscale_all_dim")):
                    if key(f"rope.scaling.{name}") is not None:
                        config["rope_scaling"][ours] = key(f"rope.scaling.{name}")
        if arch == "smollm3":
            # T255: llama.cpp leaves every fourth layer's q and k unturned, whatever the GGUF says (it says nothing:
            # src/models/smollm3.cpp at 71ad0590 sets n_no_rope_layer_step to 4), which is what config.json's
            # interval of 4 says. gguf_agrees() compares the layers with the original's
            config["no_rope_layer_interval"] = 4
        if arch == "granite":
            # T253: a Granite's four multipliers by config.json's names. llama.cpp keeps the scores' in the metadata
            # (attention.scale) and multiplies at run time: q is not scaled in the file, and the conversion scales it
            # once, as it does a safetensors' (query_scale()). Where a GGUF names none llama.cpp divides by the root of
            # the head's size, a Llama's score; the other three it leaves out of the computation where they are
            # missing or 0 (logit_scale it requires)
            size = key("embedding_length") // heads if key("embedding_length") and heads else 0
            config["attention_multiplier"] = key("attention.scale") or (1.0 / math.sqrt(size) if size else None)
            for name, ours in (("embedding_scale", "embedding_multiplier"), ("residual_scale", "residual_multiplier"),
                               ("logit_scale", "logits_scaling")):
                config[ours] = key(name) or 1.0
        if arch == "lfm2":
            # T260: llama.cpp says an LFM2's layers by their key-value heads, one number a layer and 0 for a convolution
            # layer (conversion/lfm2.py's set_gguf_parameters at f1cee994), the taps as shortconv.l_cache, and the FFN's
            # inside as it is (no two thirds left to take). By config.json's names, for lfm2_config()
            groups = key("attention.head_count_kv")
            groups = groups if isinstance(groups, list) else [groups] * (key("block_count") or 0)
            attending = sorted({count for count in groups if count})
            if len(attending) != 1 or not all(isinstance(count, int) for count in groups):
                raise ValueError(f"This GGUF holds an LFM2 whose attention layers have {attending or 'no'} key-value "
                                 f"heads, not one number for all of them.")
            config.update(num_key_value_heads=attending[0], block_auto_adjust_ff_dim=False,
                          layer_types=["full_attention" if count else "conv" for count in groups],
                          norm_eps=config["rms_norm_eps"], rope_theta=float(key("rope.freq_base", 1000000.0)),
                          **({"conv_L_cache": key("shortconv.l_cache")} if key("shortconv.l_cache") is not None else {}))
        if arch == "qwen35":
            # T236: what config.json's text_config says of the linear-attention layers, by its names (llama.cpp's are a
            # state-space model's: the state is a key head, the groups the key heads, the rank the value heads), and
            # how much of a head turns, as GPT-NeoX's. One the GGUF leaves out is left out: linear_layers() has
            # transformers' defaults, and gguf_agrees() holds the whole to the original's
            head, values, inner = key("attention.key_length"), key("ssm.time_step_rank"), key("ssm.inner_size")
            said = {"full_attention_interval": key("full_attention_interval"), "linear_conv_kernel_dim": key("ssm.conv_kernel"),
                    "linear_key_head_dim": key("ssm.state_size"), "linear_num_key_heads": key("ssm.group_count"),
                    "linear_num_value_heads": values, "linear_value_head_dim": inner // values if inner and values else None,
                    "rotary_pct": key("rope.dimension_count", 0) / head if head else None}
            config.update({name: value for name, value in said.items() if value is not None}, model_type="qwen3_5_text")
            linear = linear_layers(config)
            if linear["value_heads"] % linear["key_heads"]:
                raise ValueError(f"This GGUF holds a Qwen3.5 of {linear['value_heads']} value heads to "
                                 f"{linear['key_heads']} key heads, which is not as many to each.")
    header = {}
    if "rope_freqs.weight" in tensors:
        # llama.cpp writes Llama 3's RoPE scaling as a table of divisors instead of the rope_scaling of config.json
        if not rope_freqs:
            raise ValueError("This GGUF scales its RoPE with a rope_freqs table, which the engine does not read.")
        info = tensors["rope_freqs.weight"]
        if info["type"] != 0:
            raise ValueError(f"rope_freqs.weight is stored as ggml type {info['type']}, not F32.")
        size = 4 * math.prod(info["shape"])
        header["rope_freqs.weight"] = {"dtype": "F32", "shape": info["shape"], "rope_freqs": True,
                                       "data_offsets": [info["offset"], info["offset"] + size]}
    names, layer, layers = GGUF_ARCHITECTURES[arch]
    turns = {"attn_q": heads, "attn_k": config.get("num_key_value_heads")}
    for name, info in tensors.items():
        if info["type"] not in GGUF_TENSORS:
            raise ValueError(f"{name} is stored as ggml type {info['type']}: only F32, F16, BF16, Q8_0, PQ2_0 and PTQ1_0 "
                             f"GGUF files are supported (not the K-quants).")
        parts = name.split(".")
        if name in names:
            target = names[name]
        elif len(parts) > 2 and parts[0] == "blk" and ".".join(parts[2:]) in layers:
            target = f"{layer.format(parts[1])}{layers['.'.join(parts[2:])]}"  # a whole name (a Qwen3.5's ssm_a)
        elif len(parts) == 4 and parts[0] == "blk" and parts[2] in layers:
            target = f"{layer.format(parts[1])}{layers[parts[2]]}.{parts[3]}"
        else:
            continue  # nothing the engine reads
        dtype = GGUF_TENSORS[info["type"]]
        if info["shape"][-1] % BLOCKS.get(dtype, 1):
            # ggml itself requires it; a file that breaks it would be read at the wrong offsets and write nonsense
            raise ValueError(f"{name} is {dtype} with rows of {info['shape'][-1]}, which is not a multiple of "
                             f"{BLOCKS[dtype]}.")
        size = int(math.prod(info["shape"]) * READERS[dtype][0])
        entry = {"dtype": dtype, "shape": info["shape"], "data_offsets": [info["offset"], info["offset"] + size]}
        kind = parts[2] if len(parts) == 4 else None
        if arch in ("llama", "granite", "smollm3") and kind in turns:
            # llama.cpp turns q and k of a Llama (and their biases) into llama2.c's order; a Qwen2 it leaves alone
            # (it rotates the other way at run time). tests/gguf_check.py found SmolLM2's turned. A Granite's as a
            # Llama's (T253: its converter is the Llama's), and a SmolLM3's (T255).
            entry["turned"] = turns[kind]
        if arch == "gpt2" and parts[-1] == "weight" and kind in ("attn_qkv", "attn_output", "ffn_up", "ffn_down"):
            # GPT-2's matrices are Conv1D, (in, out): llama.cpp stores them the other way round, as every other
            # model's. Back to Hugging Face's, so that the plan transposes them once, as it does a safetensors' own
            entry["shape"], entry["transposed"] = list(reversed(info["shape"])), True
        if arch == "gptneox" and kind == "attn_qkv":
            # GPT-NeoX's query_key_value holds q, k and v of every head in turn; llama.cpp stores all of q, then k,
            # then v (the matrix and its bias). Back to Hugging Face's order, like the turned q and k of a Llama
            entry["split"] = heads
        if arch == "qwen35":
            # T236: llama.cpp writes a Qwen3.5's norms with the 1 added that the model adds to them (all but the norm
            # of a linear-attention layer's value heads, which has none), A_log as -exp(A_log), and the convolution
            # (channels, 1, taps) without its axis of one. The first two are steps of the plan (transformed()'s "one"
            # and "decay") that are done already: no float32 comes back from them to the bit, so they are not undone
            # to be done again, as a turned q is
            if target.endswith("norm.weight") and not target.endswith("linear_attn.norm.weight"):
                entry["done"] = "one"
            if target.endswith("linear_attn.A_log"):
                entry["done"] = "decay"
            if target.endswith("linear_attn.conv1d.weight") and len(info["shape"]) == 2:
                entry["shape"] = [info["shape"][0], 1, info["shape"][1]]
            # T245: and where a key head has more value heads than one (the 4B and up: 2 or 3), it writes the value
            # heads in another order (untiled()) in every tensor that has them along an axis. Put back whole, as a
            # turned q is: it is a move of values, which comes back to the bit
            per, part = linear["value_heads"] // linear["key_heads"], target.rsplit("linear_attn.", 1)[-1]
            if per > 1 and "linear_attn." in target and part in QWEN35_TILED:
                after_keys, of_a_head, axis = QWEN35_TILED[part]
                entry["tiled"] = (2 * linear["key_heads"] * linear["key_dim"] if after_keys else 0, linear["key_heads"],
                                  per, linear["value_dim"] if of_a_head else 1, axis)
        if arch == "lfm2" and target.endswith("conv.conv.weight") and len(info["shape"]) == 2:
            # T260: an LFM2's convolution comes without its axis of one too, (channels, taps) for Hugging Face's
            # (channels, 1, taps); nothing else of an LFM2 is stored another way than the original has it
            entry["shape"] = [info["shape"][0], 1, info["shape"][1]]
        header[target] = entry
    # T237: a rotated basis is the header's to say (header_rotated), next to the tensors it is about
    more = arch == "qwen35" and linear["value_heads"] != linear["key_heads"]
    rotated = gguf_rotated(metadata, tensors, more)
    if rotated is not None:
        header["__metadata__"] = {ROTATED: json.dumps(rotated)}
        # A rotated GGUF holds the columns of a linear-attention layer's output matrix in Hugging Face's order of value
        # heads (gdn_v_grouped: columns moved after the fold would be another matrix), where llama.cpp otherwise
        # stores them in its own. So the reader of that order (T245's "tiled", once it is here) must leave them be
        for target, entry in header.items():
            if target.endswith("linear_attn.out_proj.weight"):
                entry.pop("tiled", None)
    return header, config


# T237: the kinds of a layer's matrices that the engine multiplies a rotated input by, by llama.cpp's names (and
# output.weight): all the matrices of a Llama, a Qwen and a Qwen3.5 but the two small ones of a linear-attention
# layer's gates (ssm_alpha, ssm_beta)
GGUF_ROTATED = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out", "ffn_gate", "ffn_up",
                "ffn_down")


def gguf_rotated(metadata, tensors, more_value_heads=False):
    """T237: the rotated basis a GGUF of Prism ML's says its matrices are in (prism.hadamard.*), as FORM's "rotated"
    ({"block", "signs": {width: sign_bits()}}), or None where it says none. The fork's loader is the definition
    (src/llama-model.cpp, lines 1196 to 1355 at 88c4bc60): version 1 (2 with tied_output: the embedding is the
    classifier too), the transform's name, the axis, a block that is a power of two and divides every width,
    sign_mode "explicit" (sign_widths, and sign_values one width after the other) or "identity" (no signs),
    weight_names (the matrices stored as W R^-1), inverse_weight_names (the embedding, whose rows are turned back) and
    gdn_v_grouped (a linear-attention layer's output matrix reads its heads in Hugging Face's order). The engine
    turns the input of every matrix and the embedding's row, so a file that rotates other tensors than those is
    refused: it would run without a word and write nonsense. more_value_heads: a linear-attention layer has more
    value heads than key heads, whose output matrix in llama.cpp's own order of heads could not be put back."""
    said = lambda name, default=None: metadata.get(f"prism.hadamard.{name}", default)
    version = said("version")
    if version is None:
        if any(key.startswith("prism.hadamard.") for key in metadata):
            raise ValueError("This GGUF says a rotated basis without its version.")
        return None
    refuse = lambda why: ValueError(f"This GGUF's rotated basis is not one the engine computes in: {why}.")
    tied = bool(said("tied_output", False))
    if version not in (1, 2) or (version == 2) != tied or (tied and "output.weight" in tensors):
        raise refuse(f"version {version}{' with a tied output' if tied else ''}")
    if said("transform") != "normalized-sylvester-walsh-hadamard" or said("axis") != "input-last-dimension":
        raise refuse(f"the transform {said('transform')} along {said('axis')}")
    block = said("block_size")
    if not isinstance(block, int) or block < 1 or block & (block - 1):
        raise refuse(f"blocks of {block}")
    parts = lambda name: name.split(".")
    folded = {name for name in tensors if name == "output.weight" or (
        len(parts(name)) == 4 and parts(name)[0] == "blk" and parts(name)[2] in GGUF_ROTATED and parts(name)[3] == "weight")}
    names = set(said("weight_names", []))
    if names != folded:
        odd = sorted(names ^ folded)
        raise refuse(f"it rotates {'' if odd[0] in names else 'not '}{odd[0]}")
    if list(said("inverse_weight_names", [])) != ["token_embd.weight"]:
        raise refuse("the embedding's rows are not the only ones to turn back")
    if more_value_heads and not said("gdn_v_grouped", False):
        raise refuse("a linear-attention layer's output matrix in llama.cpp's order of value heads")
    widths = sorted({tensors[name]["shape"][-1] for name in folded | {"token_embd.weight"}})
    mode = said("sign_mode")
    if mode == "identity":
        signs = {width: np.ones(width) for width in widths}
    elif mode == "explicit":
        values, signs, at = np.asarray(said("sign_values", []), dtype=np.int64), {}, 0
        for width in said("sign_widths", []):
            signs[int(width)] = values[at:at + int(width)]
            at += int(width)
        if at != values.size or np.any(np.abs(values) != 1):
            raise refuse("its signs are not +1 and -1 for the widths it names")
    else:
        raise refuse(f"sign mode {mode}")
    for width in widths:
        if width % block or width not in signs or signs[width].size != width:
            raise refuse(f"no signs in whole blocks of {block} for an input {width} wide")
    return {"block": block, "signs": {str(width): sign_bits(signs[width]) for width in widths}}


def gguf_weights(head, config):
    """T136's second stage: the weights of a GGUF with the vocabulary and config.json of the original repository,
    for the GGUF's own vocabulary is of no use there (a sentencepiece one says neither Unigram or BPE nor its
    normalization; llm-jp's scores are all -1000). head: the GGUF's beginning, as far as the tensors' data (Incomplete
    when it is not), config: the text of the original's config.json. Returns the safetensors-like header (JSON text)
    and where the tensors begin, which Conversion() then takes as it takes a safetensors file's."""
    metadata, tensors, base = gguf_read(head)
    header, own = gguf_model(metadata, tensors, base, rope_freqs=True)
    try:
        original = json.loads(config)
    except ValueError:
        raise ValueError("config.json is not JSON.") from None
    if not isinstance(original, dict):
        raise ValueError("config.json is not the configuration of a model.")
    gguf_agrees(normalize(own), normalize(original))
    return json.dumps(header), base


def gguf_agrees(own, config):
    """ValueError unless a GGUF (own: what gguf_model() read of it) holds the model config.json describes. The
    sizes of the tensors the conversion checks anyway (Stream); these are what the sizes do not show: heads and
    key-value heads of the same product, the number of layers (a GGUF of more layers than config.json says went
    through cut to that many: Stream reads the layers the header asks for), a classifier that would silently be the
    embedding (Stream shares it where lm_head is missing), and the numbers that are no tensor. The context is not
    compared: a sliding window cuts it (RakutenAI 2.0 mini: 131072 in the GGUF, 8192 as normalize() cuts it).
    Both are normalize()d. GPT-NeoX's (T136's third stage): also how much of each head turns and whether the two
    branches run in parallel, which the options say (no tensor does)."""
    f32 = lambda value: float(np.float32(value))
    scaled = lambda c: yarn(c) and {key: f32(value) for key, value in yarn(c).items()}
    heads = config.get("num_attention_heads")
    pairs = [("architecture", own["model_type"], config.get("model_type")),
             ("number of layers", own["num_hidden_layers"], config.get("num_hidden_layers")),
             ("number of heads", own["num_attention_heads"], heads),
             ("number of key-value heads", own.get("num_key_value_heads", own["num_attention_heads"]),
              config.get("num_key_value_heads", heads)),
             ("RoPE theta", f32(own.get("rope_theta", 10000.0)), f32(config.get("rope_theta", 10000.0))),
             # T235: what a yarn says (None: no yarn), which changes every angle and is no tensor
             ("yarn RoPE scaling", scaled(own), scaled(config))]
    if own.get("head_dim") and config.get("hidden_size") and heads:
        pairs.append(("size of a head", own["head_dim"], head_size(config)))
    if own.get("rms_norm_eps") is not None and config.get("rms_norm_eps") is not None:
        pairs.append(("RMSNorm epsilon", f32(own["rms_norm_eps"]), f32(config["rms_norm_eps"])))
    if "smollm3" in (own["model_type"], config.get("model_type")):
        # T255: the layers RoPE leaves alone, which are no tensor (llama.cpp's are every fourth, always)
        pairs.append(("layers without RoPE", unturned_layers(own), unturned_layers(config)))
    if "granite" in (own["model_type"], config.get("model_type")):
        # T253: a Granite's multipliers, which are no tensor: the scores' goes into q from config.json's (a GGUF that
        # says another would be scaled by the wrong one), and the three the engine has not must be 1 in both
        multipliers = lambda c: {key: f32(c.get(key, 1.0)) for key in ("attention_multiplier", *GRANITE_ONES)
                                 if isinstance(c.get(key, 1.0), (int, float))}
        pairs.append(("Granite's multipliers", multipliers(own), multipliers(config)))
    if architecture(own) in ("gpt2", "neox"):
        # transformers' default where config.json says none (the engine's LayerNorm takes 1e-5 whatever it says)
        layer_norm_eps = lambda c: c.get("layer_norm_eps", c.get("layer_norm_epsilon", 1e-5))
        pairs.append(("LayerNorm epsilon", f32(layer_norm_eps(own)), f32(layer_norm_eps(config))))
    if architecture(own) == "neox" and architecture(config) == "neox":
        pairs += [("number of rotated values of a head", rotary_dim(own), rotary_dim(config)),
                  ("parallel residual", own.get("use_parallel_residual", True), config.get("use_parallel_residual", True))]
    if architecture(own) == "qwen35" and architecture(config) == "qwen35":
        # T236: how much of a head turns, and the linear-attention layers: which layers they are and their heads
        # (the tensors show the products only: 16 key heads of 128 are 8 of 256 to them)
        pairs += [("number of rotated values of a head", rotary_dim(own), rotary_dim(config)),
                  ("linear-attention layers", linear_layers(own), linear_layers(config))]
    if architecture(own) == "lfm2" and architecture(config) == "lfm2":
        # T260: which layers are convolution layers and their taps, and the FFN's inside as the config's rule makes it
        # (the tensors' sizes would show it too, but only once they stream past)
        pairs += [("convolution layers", convolution_layers(own), convolution_layers(config)),
                  ("size of the FFN", own.get("intermediate_size"), config.get("intermediate_size"))]
    for what, here, there in pairs:
        if here != there:
            raise ValueError(f"This GGUF does not belong with the original's config.json: its {what} is {here} here "
                             f"and {there} there.")
    if own["tie_word_embeddings"] and not config.get("tie_word_embeddings", False):
        raise ValueError("This GGUF does not belong with the original's config.json: the original has a classifier of "
                         "its own, this GGUF has none.")


def rope_freqs_agree(table, config):
    """ValueError unless llama.cpp's rope_freqs (a divisor of each pair's angle) is what the original's rope_scaling
    makes: the engine makes its RoPE tables from rope_scaling (rope_frequencies), and the table is not used."""
    width = head_size(config)
    theta = config.get("rope_theta", 10000.0)
    expected = rope_frequencies(width, theta) / rope_frequencies(width, theta, config.get("rope_scaling"))
    table = np.asarray(table, dtype=np.float64)
    worst = float(np.max(np.abs(table - expected) / expected)) if table.shape == expected.shape else float("inf")
    if not worst <= 1e-5:
        raise ValueError(f"This GGUF scales its RoPE otherwise than the original's rope_scaling says (by {worst:.1e} "
                         f"at most).")


def gguf_tokenizer(metadata, vocab_size):
    """tokenizer.bin, the engine's options, the tokenizer_config and the special tokens of a GGUF's byte-level BPE
    vocabulary."""
    if metadata.get("tokenizer.ggml.model") != "gpt2":
        raise ValueError(f"This GGUF has a {metadata.get('tokenizer.ggml.model')} vocabulary: only byte-level BPE "
                         f"ones (gpt2) are supported.")
    pre = metadata.get("tokenizer.ggml.pre", "gpt-2")
    if pre not in GGUF_PRETOKENIZERS:
        raise ValueError(f"This GGUF splits text as {pre}, which the engine does not know.")
    tokens, kinds = metadata["tokenizer.ggml.tokens"], metadata.get("tokenizer.ggml.token_type", [])
    ranks = {}
    for rank, merge in enumerate(metadata.get("tokenizer.ggml.merges", [])):
        left, right = merge.split(" ")
        ranks.setdefault(left + right, -float(rank))
    # what tokenizer.json calls special: llama.cpp's control tokens (type 3). Its padding up to the vocabulary's size
    # ([PAD151665] ..., type 5, unused) is text no piece of tokenizer.json has: empty, as the safetensors path pads (T143)
    kind = lambda id: kinds[id] if id < len(kinds) else 1
    pieces = [("" if kind(id) == 5 else text, ranks.get(text, UNMATCHABLE), text in ranks and kind(id) != 3)
              for id, text in enumerate(tokens)]
    # Qwen's tokenizer.json normalizes to NFC, which a GGUF does not say: the page's safetensors path does it
    options = {"tokenizer_kind": "bytebpe", "nfkc": False, "nfc": pre in GGUF_NFC, "pretokenizer": GGUF_PRETOKENIZERS[pre],
               "ignore_merges": pre == "llama-bpe"}
    special = lambda key: tokens[metadata[key]] if isinstance(metadata.get(key), int) and metadata[key] < len(tokens) else ""
    config = {"chat_template": metadata.get("tokenizer.chat_template"), "bos_token": special("tokenizer.ggml.bos_token_id"),
              "eos_token": special("tokenizer.ggml.eos_token_id")}
    # the added tokens tokenizer.json does not call special are llama.cpp's user-defined ones (type 4)
    controls, added = ([text for id, text in enumerate(tokens) if id < len(kinds) and kinds[id] == type] for type in (3, 4))
    return tokenizer_bin(pieces, vocab_size, spaces=False), options, config, controls, added
