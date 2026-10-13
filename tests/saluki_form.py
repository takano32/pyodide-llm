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
    "wk": {"IQ3_S": 5, "IQ3_XXS": 4, "IQ4_XS": 2, "Q4_K": 1, "IQ2_S": 1, "IQ1_M": 1, "IQ2_XXS": 1, "Q2_K": 1},
    "wv": {"IQ3_S": 5, "IQ3_XXS": 3, "IQ4_XS": 2, "Q4_K": 2, "IQ2_S": 1, "IQ1_M": 1, "IQ2_XS": 1, "IQ1_S": 1},
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
