# unchanged_layouts.py (T357)
# What a tree says of a checkpoint's file, over a grid of made-up headers, forms and dtypes, as one JSON object: the
# record tests/unchanged.mjs compares between two trees (its kind "layouts"). The order of a file's tensors is written
# down in four places that must agree (the converter's layout(), the engine's *_tensors(), conversion_plan() and
# external_tensors(), which reads the engine's order), and its sizes in a fifth (checkpoint_dtype()); the unit tests
# walk the branches of the models they make and no others. Here every one of them answers for the same grid, through
# the two windows (llama2_convert, llama2_numpy) and Llama itself, so that a part moved behind them is asked the same:
#
#   layout    layout(): the shape and the kind of every tensor, in file order
#   places    Writer's: where each of them begins, and the checkpoint's size (checkpoint_size()); a refusal's words
#   dtype     checkpoint_dtype() of a file of that size
#   engine    the plan Llama(external=) hands forward.js (tests/engine_plans.py): where every tensor is by the engine's own order (the
#             *_tensors() of the architecture), and every other key of the plan (the derived tables as a hash)
#   external  external_tensors(): the places before the model is built (a Llama's; the others' refusal)
#   source    conversion_plan(): the names of the Hugging Face tensors each one is made of and what is done to them
# and, since the file's header and form come from a config.json and the unit tests read few of them:
#   config    normalize(), checkpoint_header(), checkpoint_form(), query_scale(), unturned_layers() and rotary_dim()
#             of made-up config.json files of every family (an LFM2's FFN by block_multiple_of among them)
#   window    which of a history's tokens penalize() reaches (the repetition penalty's window)
#
#   python tests/unchanged_layouts.py <the root of a tree>        -> one JSON object {name: what was said}
import hashlib
import json
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, root + "/public")
import llama2_convert as C  # noqa: E402
import llama2_numpy as L  # noqa: E402
from engine_plans import engine_plan  # noqa: E402  (tests/engine_plans.py: this script's neighbour, whichever tree is asked)

found = {}


def short(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def said(name, ask, whole=False):
    """What ask() answers, or the words of its refusal; a long answer as a hash and its length."""
    try:
        answer = ask()
    except Exception as error:  # a refusal is an answer too: which one, and in which words
        found[name] = f"raises {type(error).__name__}: {error}"
        return None
    text = json.dumps(answer, sort_keys=True, default=str)
    found[name] = text if whole or len(text) <= 200 else f"{short(answer)} ({len(text)} characters)"
    return answer


def signs(widths, block):
    """A rotated basis for these widths: every seventh sign minus."""
    return {"block": block, "signs": {str(width): L.sign_bits(np.where(np.arange(width) % 7 == 3, -1.0, 1.0)) for width in widths}}


# ---- the grid. Every dimension of a header differs from the others (a tensor taken for another is another shape),
# the rows are whole groups of 128 (so that all five dtypes can be written) but in the last header, which no packed
# dtype can hold; a negative vocabulary is a classifier of its own
HEADERS = {
    "llama": [[256, 768, 4, 8, 8, 2048, 320], [384, 1024, 6, 12, 4, -2304, 288], [200, 440, 3, 4, 2, 1000, 96]],
    "wide heads": [[256, 768, 4, 4, 2, 2048, 320], [384, 1024, 6, 6, 3, -2304, 288]],
    "gpt2": [[256, 1024, 3, 8, 8, 2048, 320], [384, 1536, 5, 6, 6, -2304, 288], [200, 800, 3, 4, 4, 1000, 96]],
    "qwen35": [[256, 768, 8, 4, 2, 2048, 320], [384, 1024, 12, 6, 3, -2304, 288]],
    "qwen35, every second": [[256, 768, 6, 4, 2, 2048, 320], [384, 1024, 5, 6, 3, -2304, 288]],
    "lfm2": [[256, 768, 8, 8, 4, 2048, 320], [384, 1024, 8, 12, 4, -2304, 288]],
}
LINEAR = {"every": 4, "key_heads": 2, "value_heads": 4, "key_dim": 64, "value_dim": 128, "conv": 4}
LINEAR_SECOND = {"every": 2, "key_heads": 4, "value_heads": 4, "key_dim": 32, "value_dim": 64, "conv": 3}
# (name, the headers, the form, what Llama takes besides the form, what conversion_plan() takes besides it)
FAMILIES = [
    ("llama", "llama", {}, {}, {}),
    ("llama, outliers", "llama", {}, {"outliers": True}, None),
    ("llama, q scaled (granite)", "llama", {}, None, {"scale": 0.125}),
    ("llama, unturned (smollm3)", "llama", {}, {"unturned": (1, 3)}, None),
    ("qwen2", "llama", {"bias": True}, {}, {}),
    ("qwen3", "llama", {"qk_norm": True}, {}, {}),
    ("qwen3, wide heads", "wide heads", {"qk_norm": True, "head_dim": 128}, {}, {}),
    ("bias and norms", "wide heads", {"bias": True, "qk_norm": True, "head_dim": 128}, {}, {}),
    ("llama, rotated", "llama", {"rotated": True}, {}, {}),
    ("gpt2", "gpt2", {"arch": "gpt2"}, {}, {"prefix": "transformer."}),
    ("gpt2, no prefix", "gpt2", {"arch": "gpt2"}, None, {"prefix": ""}),
    ("neox", "gpt2", {"arch": "neox"}, {"rotary": 8, "parallel_residual": True}, {"rotary": 8}),
    ("neox, whole heads", "gpt2", {"arch": "neox"}, {}, {"rotary": 32}),
    ("qwen35", "qwen35", {"arch": "qwen35", "head_dim": 128, "linear": LINEAR}, {"rotary": 32},
     {"prefix": "model.language_model.", "rotary": 32}),
    ("qwen35, rotated", "qwen35", {"arch": "qwen35", "head_dim": 128, "linear": LINEAR, "rotated": True}, {"rotary": 32},
     {"prefix": "model.", "rotary": 32}),
    ("qwen35, every second", "qwen35, every second", {"arch": "qwen35", "head_dim": 128, "linear": LINEAR_SECOND}, {"rotary": 64},
     {"prefix": "model.", "rotary": 64}),
    ("lfm2", "lfm2", {"arch": "lfm2", "convolution": {"layers": "ccaccaca", "taps": 3}}, {}, {}),
    ("lfm2, four taps", "lfm2", {"arch": "lfm2", "convolution": {"layers": "acaccacc", "taps": 4}}, {}, {}),
]
DTYPES = ("float32", "float16", "int8", "int6", "ternary")


for name, headers, form, engine, source in FAMILIES:
    for header in HEADERS[headers]:
        dim, hidden, n_layers, n_heads = header[:4]
        form = dict(form)
        if form.get("rotated"):
            q_dim = n_heads * (form.get("head_dim") or dim // n_heads)
            form["rotated"] = signs(L.rotated_widths(dim, q_dim, hidden, L.linear_form(form.get("linear"))), 8)
        if form.get("convolution") and header[2] != len(form["convolution"]["layers"]):
            continue
        case = f"{name} {header}"
        whole = {**L.FORM, **form}
        said(f"{case}: layout", lambda: [[list(shape), matrix] for shape, matrix in C.layout(*header, **whole)])
        if source is not None:
            said(f"{case}: source", lambda: C.conversion_plan(header, whole, **source))
        for dtype in DTYPES:
            size = said(f"{case}, {dtype}: size", lambda: C.checkpoint_size(header, dtype, whole))
            said(f"{case}, {dtype}: places", lambda: [list(tensor) for tensor in C.Writer(
                None, header, dtype, whole, sink=type("Sink", (), {"open": lambda *_: None, "write": lambda *_: None})()).tensors])
            said(f"{case}, {dtype}: dtype", lambda: L.checkpoint_dtype(header, size, whole))
            if engine is not None:
                said(f"{case}, {dtype}: engine", lambda: engine_plan(L, header, dtype, form, engine, size))
                if not engine:
                    said(f"{case}, {dtype}: external", lambda: L.external_tensors(header, dtype, whole))

# ---- config.json: what the converter reads of a made-up one of every family (a source is the names of its tensors)
BASE = {"hidden_size": 256, "intermediate_size": 768, "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 2048,
        "max_position_embeddings": 4096}
LFM2 = {"model_type": "lfm2", "hidden_size": 256, "num_hidden_layers": 6, "num_attention_heads": 8, "num_key_value_heads": 4,
        "vocab_size": 2048, "max_position_embeddings": 4096, "layer_types": ["conv", "conv", "full_attention"] * 2}
CONFIGS = {
    "llama": ({**BASE, "model_type": "llama", "num_key_value_heads": 4}, ["lm_head.weight"]),
    "llama, tied": ({**BASE, "model_type": "llama", "tie_word_embeddings": True, "head_dim": None}, ["lm_head.weight"]),
    "llama, rope_parameters": ({**BASE, "model_type": "llama", "rope_parameters": {"rope_theta": 500000.0, "rope_type": "llama3", "factor": 32.0}}, []),
    "mistral, a window": ({**BASE, "model_type": "mistral", "sliding_window": 1024}, []),
    "qwen2": ({**BASE, "model_type": "qwen2", "num_key_value_heads": 2}, ["model.layers.0.self_attn.q_proj.bias"]),
    "qwen3": ({**BASE, "model_type": "qwen3", "head_dim": 64}, ["model.layers.0.self_attn.q_norm.weight", "lm_head.weight"]),
    "granite": ({**BASE, "model_type": "granite", "attention_multiplier": 0.015625}, []),
    "smollm3, every fourth": ({**BASE, "model_type": "smollm3", "num_hidden_layers": 9}, []),
    "smollm3, every third": ({**BASE, "model_type": "smollm3", "num_hidden_layers": 9, "no_rope_layer_interval": 3}, []),
    "smollm3, a list": ({**BASE, "model_type": "smollm3", "no_rope_layers": [1, 0, 1, 0]}, []),
    "gpt2": ({"model_type": "gpt2", "n_embd": 256, "n_layer": 3, "n_head": 8, "n_positions": 512, "vocab_size": 2048}, ["wte.weight"]),
    "gpt2, n_inner and n_ctx": ({"model_type": "gpt2", "n_embd": 256, "n_inner": 640, "n_layer": 3, "n_head": 8, "n_ctx": 384,
                                 "vocab_size": 2048, "tie_word_embeddings": False}, ["transformer.wte.weight", "lm_head.weight"]),
    "neox": ({**BASE, "model_type": "gpt_neox", "rotary_pct": 0.25, "rotary_emb_base": 20000}, ["embed_out.weight"]),
    "neox, rope_parameters": ({**BASE, "model_type": "gpt_neox", "rope_parameters": {"rope_theta": 30000, "partial_rotary_factor": 0.5},
                               "tie_word_embeddings": True}, ["embed_out.weight"]),
    "qwen3.5": ({"model_type": "qwen3_5", "text_config": {**BASE, "num_hidden_layers": 8, "num_key_value_heads": 2, "eos_token_id": 7,
                                                           "linear_num_value_heads": 16, "linear_num_key_heads": 8}},
                ["model.language_model.embed_tokens.weight"]),
    "qwen3.5, text": ({**BASE, "model_type": "qwen3_5_text", "num_hidden_layers": 6, "head_dim": 128, "full_attention_interval": 3,
                       "partial_rotary_factor": 0.5, "linear_conv_kernel_dim": 3, "linear_key_head_dim": 64}, ["lm_head.weight"]),
    "qwen3.5, layer_types": ({**BASE, "model_type": "qwen3_5_text", "num_hidden_layers": 4,
                              "layer_types": ["linear_attention", "full_attention"] * 2}, []),
    # (a sixth of 1000 is no whole number and 666 no multiple of anything asked for: each key of the rule moves the size)
    "lfm2": ({**LFM2, "block_ff_dim": 1000}, []),
    "lfm2, a multiple of 64": ({**LFM2, "block_ff_dim": 1000, "block_multiple_of": 64}, []),
    "lfm2, a multiplier": ({**LFM2, "block_ff_dim": 1000, "block_ffn_dim_multiplier": 1.5, "block_multiple_of": 48}, []),
    "lfm2, no multiplier": ({**LFM2, "block_ff_dim": 1000, "block_ffn_dim_multiplier": None, "block_multiple_of": 64}, []),
    "lfm2, as it is": ({**LFM2, "intermediate_size": 1000, "block_auto_adjust_ff_dim": False, "block_multiple_of": 64}, ["lm_head.weight"]),
    "lfm2, full_attn_idxs": ({**{key: value for key, value in LFM2.items() if key != "layer_types"}, "intermediate_size": 960,
                              "full_attn_idxs": [1, 4, 5], "conv_L_cache": 4, "norm_eps": 1e-6, "tie_embedding": False,
                              "rope_parameters": {"rope_theta": 250000.0}}, ["lm_head.weight"]),
}
for name, (config, tensors) in CONFIGS.items():
    source = dict.fromkeys(tensors)
    normal = said(f"config {name}: normalized", lambda: C.normalize(config), whole=True)
    said(f"config {name}: check", lambda: C.check_config(normal))
    for context in (4096, 300):
        said(f"config {name}: header for a context of {context}", lambda: C.checkpoint_header(config, source, context))
    said(f"config {name}: form", lambda: C.checkpoint_form(config, source))
    said(f"config {name}: q's scale, the layers RoPE leaves, the values it turns",
         lambda: [C.query_scale(normal), C.unturned_layers(normal), C.rotary_dim(normal), C.head_size(normal)])

# ---- the repetition penalty's window: a history of 200 different tokens, and which of them the penalty reached
logits = np.ones(256, dtype=np.float32)
L.Llama.penalize(None, logits, list(range(200)), 2.0)
reached = np.flatnonzero(logits != 1.0)
said("window: the tokens of a history of 200 the penalty reaches", lambda: f"{len(reached)}, from {int(reached[0])} to {int(reached[-1])}")

print(json.dumps(found))
