# tests/saluki_form.py (T363.2): the header and the form of Underdog Saluki 27B's checkpoint as the page will hold it,
# its matrices in the blocks the GGUF has them in (llama.cpp's K- and i-quantized types, a type to a tensor: a layer of
# a stack, in the engine's rows), for what sizes such a file and lays it out. No weights, and nothing of pytest or of
# the converter is imported: tests/smoke.mjs reads this inside Pyodide, whose NumPy has integers of 32 bits.
#
# How many tensors of each role are of each type is the real file's (docs/notes/t361-saluki-2026-10-13.md, section 2;
# the roles the note sums up as "others" follow from its totals by type). Which layer has which is not: the types of
# a role are dealt out here in the order they are written, most of them first. The size of the file depends on the
# counts alone, the places of the tensors on the order too.
HEADER = [5120, 17408, 64, 24, 4, -248320, 4096]
LINEAR = {"every": 4, "key_heads": 16, "value_heads": 48, "key_dim": 128, "value_dim": 128, "conv": 4}
# the engine's row -> the types of its layers and how many layers have each
COUNTS = {
    "token_embedding_table": {"IQ1_M": 1},
    "wcls": {"IQ3_S": 1},
    # (the GGUF's attn_q is q and its gate in one tensor, so the two rows have a layer's type together)
    "wq": {"IQ2_XXS": 8, "Q2_K": 8},
    "wg": {"IQ2_XXS": 8, "Q2_K": 8},
    # (T363.3: k and v as the real head has them, which tests/saluki_rows.py holds these counts to: the note's "others"
    # had been dealt out with one IQ3_XXS and the IQ2_XS on the other of the two, which are the same shape)
    "wk": {"IQ3_S": 5, "IQ3_XXS": 3, "IQ4_XS": 2, "Q4_K": 1, "IQ2_S": 1, "IQ1_M": 1, "IQ2_XS": 1, "IQ2_XXS": 1, "Q2_K": 1},
    "wv": {"IQ3_S": 5, "IQ3_XXS": 4, "IQ4_XS": 2, "Q4_K": 2, "IQ2_S": 1, "IQ1_M": 1, "IQ1_S": 1},
    "wo": {"IQ3_S": 9, "IQ1_S": 2, "IQ1_M": 2, "IQ2_XXS": 1, "IQ2_XS": 1, "IQ3_XXS": 1},
    "wqkv": {"IQ2_XXS": 13, "IQ2_XS": 10, "IQ2_S": 8, "Q2_K": 6, "IQ1_M": 5, "IQ3_XXS": 3, "IQ1_S": 2, "IQ3_S": 1},
    "wz": {"IQ1_M": 11, "Q2_K": 11, "IQ2_XXS": 7, "IQ1_S": 6, "IQ2_XS": 5, "IQ2_S": 4, "IQ3_S": 2, "IQ3_XXS": 2},
    "wout": {"IQ2_S": 12, "IQ3_S": 12, "IQ2_XS": 9, "IQ3_XXS": 7, "Q2_K": 4, "IQ2_XXS": 3, "IQ4_XS": 1},
    "w1": {"IQ2_XXS": 17, "IQ1_S": 13, "Q2_K": 9, "IQ2_S": 7, "IQ1_M": 5, "IQ2_XS": 5, "IQ3_XXS": 5, "IQ3_S": 2, "Q4_K": 1},
    "w2": {"IQ2_XXS": 53, "IQ1_S": 5, "IQ3_S": 2, "IQ3_XXS": 2, "IQ1_M": 1, "IQ4_XS": 1},
    "w3": {"IQ2_XXS": 16, "IQ2_S": 13, "IQ2_XS": 8, "IQ1_S": 7, "Q2_K": 7, "IQ3_S": 5, "IQ1_M": 4, "IQ3_XXS": 3, "IQ4_XS": 1},
}
# a row of the model's alone has its type by itself, a stack one for each of its layers
ALONE = ("token_embedding_table", "wcls")
KINDS = {name: next(iter(counts)) if name in ALONE else [kind for kind, count in counts.items() for _ in range(count)]
         for name, counts in COUNTS.items()}
FORM = {"arch": "qwen35", "bias": False, "head_dim": 256, "linear": LINEAR, "kinds": KINDS}
# the note's: 28 + the matrices as the file's blocks + the two small matrices of the gates in float32 + the vectors
MATRICES, GATES, VECTORS = 7851724800, 94371840, 10582016
BYTES = 28 + MATRICES + GATES + VECTORS
assert BYTES == 7956678684
# the tensors of the GGUF by type (the same note): a row of the engine's is one of them, but that wq and wg are one
TENSORS = {"IQ2_XXS": 119, "IQ3_S": 44, "Q2_K": 46, "IQ2_S": 46, "IQ2_XS": 39, "IQ1_M": 31, "IQ3_XXS": 30, "IQ1_S": 36,
           "IQ4_XS": 7, "Q4_K": 4}


# ---- T363.3: the file as the converter is given it
# config.json of the model the file was made from, as far as the converter reads it (Qwen/Qwen3.8-27B's text_config)
CONFIG = {"model_type": "qwen3_5_text", "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
          "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
          "max_position_embeddings": 262144, "tie_word_embeddings": False, "full_attention_interval": 4,
          "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
          "linear_num_value_heads": 48, "linear_value_head_dim": 128, "partial_rotary_factor": 0.25,
          "rms_norm_eps": 1e-6, "rope_theta": 10000000.0, "bos_token_id": 248044, "eos_token_id": 248046}


def header(first="lm_head.weight"):
    """The safetensors-like header the converter makes of such a GGUF's head (gguf_weights()), without the file: every
    tensor the conversion's plan asks for, by its Hugging Face name, of the type KINDS gives its row (the two small
    matrices of the gates Q8_0 and the vectors F32, as the real file has them), marked as llama.cpp stores a Qwen3.5
    (the value heads tiled, the norms and A_log changed already), one after the other with the tensor called first
    first. For what a conversion that keeps the blocks lays out: no bytes of weights are behind it."""
    import llama2_convert
    import llama2_numpy
    from convert.families.qwen35 import QWEN35
    from convert.plan import source_shape
    plain = {key: value for key, value in FORM.items() if key != "kinds"}
    plan, shapes = llama2_convert.conversion_plan(HEADER, plain, "model.", 64)
    rows, stored, entries = llama2_numpy.tensor_rows(HEADER, plain), QWEN35.gguf_stored(CONFIG), {}
    for row, parts, shape in zip(rows, plan, shapes):
        for layer, (name, transform) in enumerate(parts or ()):
            kind = KINDS.get(row.name, "Q8_0" if row.name in ("wb", "wa") else "F32")
            found = source_shape(shape[1:] if len(parts) > 1 else shape, transform)
            entries[name] = {"dtype": kind if isinstance(kind, str) else kind[layer], "shape": list(found)}
            stored(entries[name], {"shape": list(found)}, name, None)
    at = 0
    for name in sorted(entries, key=lambda name: name != first):
        entry = entries[name]
        count = 1
        for size in entry["shape"]:
            count *= size
        size = llama2_convert.SOURCES[entry["dtype"]].bytes(count)
        entry["data_offsets"], at = [at, at + size], at + size
    return entries
