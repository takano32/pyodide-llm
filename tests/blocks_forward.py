# blocks_forward.py (T.363.5)
# Small made-up models whose matrices are kept as llama.cpp's blocks (the form's "kinds", a type for each layer of a
# stack), each as what an engine outside Python is handed (the checkpoint and the plan, as Llama(external=) makes it)
# and what the engine in NumPy answers for a few tokens, twice: as it is (float32 activations), and with the
# activations every block matrix reads rounded as the kernels round them (quantize_x: 8 bits, a scale for each 32),
# which is what forward.ts computes but for the order of its float32 sums. tests/blocks-forward-check.mjs runs
# forward.ts on them.
#   python tests/blocks_forward.py <the root of a tree> <a folder to write into>     -> the cases, as JSON; the files in the folder
# The blocks are random bytes (any bytes are blocks: tests/made_up.py's drawn_blocks()); what they read as, by the
# converter's readers, are the model's weights.
import json
import os
import sys

import numpy as np

from engine_plans import External, Handed

TEN = ["Q2_K", "Q4_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "IQ4_XS"]
# Underdog Saluki 27B's shape in small (tests/test_blocks.py's): a Qwen3.5 with a classifier of its own, a
# full-attention layer every fourth, three value heads to a key head, every row of a matrix whole blocks of 256
SALUKI = [256, 512, 8, 4, 2, -320, 24]
LINEAR = {"every": 4, "key_heads": 2, "value_heads": 6, "key_dim": 64, "value_dim": 128, "conv": 4}
QWEN35 = {"arch": "qwen35", "head_dim": 64, "linear": LINEAR}
STACKS = {"wq": 2, "wg": 2, "wk": 2, "wv": 2, "wo": 2, "wqkv": 6, "wz": 6, "wout": 6, "w1": 8, "w2": 8, "w3": 8}
# a Llama of the same widths, its embedding its classifier
LLAMA = [256, 512, 3, 4, 2, 320, 24]


def turns(stacks, start=0):
    """A type to a layer of every stack, so that the matrices of one role are of several types and every type is some
    matrix's."""
    return {name: [TEN[(start + at + 3 * layer) % len(TEN)] for layer in range(layers)] for at, (name, layers) in enumerate(stacks.items())}


def cases():
    saluki = {**turns(STACKS), "token_embedding_table": "IQ1_M", "wcls": "IQ3_S"}
    llama = {name: 3 for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3")}
    return [
        # (the form, what Llama takes besides, whether every matrix is blocks: then the rounded reference is the engine's)
        ("a Saluki in small: ten types by the layer, tiled", SALUKI, {**QWEN35, "kinds": saluki, "tiled": True}, {"rotary": 16}, True),
        ("the same in Hugging Face's order of the value heads", SALUKI, {**QWEN35, "kinds": saluki}, {"rotary": 16}, True),
        ("a Llama of blocks, its embedding its classifier", LLAMA, {"head_dim": 64, "kinds": {**turns(llama, 4), "token_embedding_table": "IQ2_XS"}}, {}, True),
        # (matrices of one input packed differently: w1 blocks and w3 int8, q six bits, k float32, and a stack of layers
        # that are blocks, int8, six bits and float16 in turn; the gates of a Qwen3.5 int8 and a kind to a layer)
        ("blocks beside int8, six bits and floats, by the row and by the layer", SALUKI,
         {**QWEN35, "tiled": True, "kinds": {"w1": turns({"w1": 8})["w1"], "wq": "int6", "wk": "float32", "wqkv": ["IQ3_S", "int8", "int6", "float16", "IQ1_S", "Q4_K"],
                                             "wb": "int8", "wa": ["int8", "float32", "float16", "int6", "int8", "float32"], "wcls": "IQ4_XS"}}, {"rotary": 16}, False),
    ]


def rounded(x):
    """x as the kernels' quantize_x(xq, xs, x, n, 0) leaves it for a matrix of blocks: for each 32, the largest
    magnitude over 127 as a float32 scale, and the nearest integers of the values over it; in float64."""
    groups = np.asarray(x, dtype=np.float32).reshape(-1, 32)
    scale = (np.abs(groups).max(axis=1) / np.float32(127.0)).astype(np.float32)
    inverse = np.where(scale > 0, np.float32(1.0) / np.where(scale > 0, scale, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
    return (np.rint((groups * inverse[:, None]).astype(np.float32)).astype(np.float64) * scale[:, None].astype(np.float64)).reshape(-1)


class Rounded:
    """A matrix (or a stack of them) that multiplies by the rounded activations where its layer is blocks."""

    def __init__(self, w, blocks):
        self.w, self.blocks = w, blocks

    def __getitem__(self, layer):
        return Rounded(self.w[layer], self.blocks[layer] if isinstance(self.blocks, list) else self.blocks)

    def __matmul__(self, x):
        return (self.w.astype(np.float64) @ rounded(x)).astype(np.float32) if self.blocks else self.w @ x


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tree import python_folder
    sys.path.insert(0, python_folder(sys.argv[1]))
    import llama2_numpy as L
    import made_up

    folder, said = sys.argv[2], []
    tokenizer = made_up.pack_tokenizer(made_up.tiny_vocab(320))
    tokens = [5, 17, 3, 250, 9, 9, 41, 2, 300, 77, 1, 12]
    for index, (name, header, form, options, whole) in enumerate(cases()):
        checkpoint = made_up.checkpoint(header, form, "int8", tensor=made_up.uniform(11 + index), blocks=made_up.drawn_blocks(3 + index))
        external = External(len(checkpoint), lambda offset, length: checkpoint[offset:offset + length])
        try:
            L.Llama(None, b"", dtype="int8", external=external, **form, **options)
        except Handed:
            pass
        plan = dict(external.plan)
        derived = {}
        for table, value in plan["derived"].items():
            derived[table] = f"{index}.{table}"
            open(os.path.join(folder, derived[table]), "wb").write(bytes(value))
        plan["derived"] = derived
        answers = {}
        for answer in ("floats", "rounded"):
            model = L.Llama(checkpoint, tokenizer, dtype="int8", **form, **options)
            if answer == "rounded":
                for row, kind in form["kinds"].items():
                    blocks = [one in L.BLOCKS for one in kind] if isinstance(kind, list) else kind in L.BLOCKS
                    if row != "token_embedding_table" or header[5] > 0:
                        setattr(model, "wcls" if row == "token_embedding_table" else row, Rounded(getattr(model, "wcls" if row == "token_embedding_table" else row), blocks))
            logits = np.stack([model.forward(token, pos) for pos, token in enumerate(tokens)]).astype(np.float32)
            answers[answer] = f"{index}.{answer}"
            open(os.path.join(folder, answers[answer]), "wb").write(logits.tobytes())
        open(os.path.join(folder, f"{index}.bin"), "wb").write(checkpoint)
        said.append({"name": name, "plan": plan, "tokens": tokens, "checkpoint": f"{index}.bin", "logits": answers, "whole": whole,
                     "header": header, "form": {key: value for key, value in form.items()}, "weights": L.dtype_weights(header, "int8", form)})
    print(json.dumps(said))
