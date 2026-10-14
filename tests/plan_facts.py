# plan_facts.py (T375.2)
# Small made-up models whose architecture is a record of facts (engine/layout.py's LAYOUTS), each as what an engine
# outside Python is handed (the checkpoint and the plan: engine/plan.py's forward_plan(), as Llama(external=) makes it)
# and what the reference in NumPy answers for a few tokens. tests/plan-facts-check.mjs runs forward.ts on them.
#   python tests/plan_facts.py <the root of a tree> <a folder to write into>     -> the cases, as JSON; the files in the folder
# Beside the five architectures there are, three whose facts are no architecture's pair: every one of the table has
# LayerNorms with GPT-2's FFN or RMSNorms with a gated one, so that an engine that read one fact for the other, or
# the name, would answer right for all five. Their names are none an engine outside Python has heard of.
# T261: and a Gemma 3 (a GELU in the gate of its FFN, norms behind the attention and the FFN, local layers), with three
# that have one of its three each and are Llamas otherwise: only a Gemma has any of them, and it has them all.
import json
import os
import sys

import numpy as np

from engine_plans import External, Handed

TOKENS = [5, 17, 3, 250, 9, 9, 41, 2, 300, 77]
# (a linear-attention layer's numbers: such that a frame sized by its key heads in place of its value heads is a line
# of the cache shorter, and the next token's frame begins inside it)
LINEAR = {"every": 2, "key_heads": 3, "value_heads": 9, "key_dim": 16, "value_dim": 16, "conv": 4}
# (dim, hidden, layers, heads, kv heads, vocab, context): a negative vocabulary is a classifier of its own
GROUPED, FILLED = [64, 128, 4, 4, 2, 320, 48], [64, 128, 3, 4, 4, -320, 48]


def cases(L):
    from engine.layout import gpt2
    layouts = L.LAYOUTS
    # RMSNorms with GPT-2's FFN (the file's biases of the norms are read by nobody), and LayerNorms with a gated FFN:
    # a GPT-2's rows and a third matrix. Both with GPT-NeoX's parallel residual, by which the FFN's norm, of either
    # kind, reads the layer's input
    layouts["rms and gelu"] = layouts["gpt2"]._replace(layer_norm=False)
    gated = lambda d: [*gpt2(d), d.stack("w3", L.MATRIX, L.EVERY, d.hidden_dim, d.dim)]
    layouts["layer norms and a gate"] = layouts["neox"]._replace(rows=gated, gated_ffn=True)
    # and a Llama that RoPE turns no layer of, which is no GPT-2 in anything else
    # (T237, a rotated basis: signs for every width the model's matrices read, every seventh of them minus)
    def rotated(header, form):
        q_dim = header[3] * (form.get("head_dim") or header[0] // header[3])
        widths = L.rotated_widths(header[0], q_dim, header[1], L.linear_form(form.get("linear")))
        return {**form, "rotated": {"block": 8, "signs": {str(width): L.sign_bits(np.where(np.arange(width) % 7 == 3, -1.0, 1.0)) for width in widths}}}
    qwen35 = {"arch": "qwen35", "head_dim": 16, "linear": LINEAR}
    # T261: a Llama with a GELU in its gate, one with local layers (a window of 3 of the 10 tokens, another base), and
    # one with the two norms behind, each of the three by itself
    layouts["a gelu in the gate"] = layouts["llama"]._replace(activation=L.GELU)
    layouts["local layers"] = layouts["llama"]._replace(local=True)
    behind = lambda d: [*layouts["llama"].rows(d), d.stack("rms_att_post_weight", L.VECTOR, L.EVERY, d.dim),
                        d.stack("rms_ffn_post_weight", L.VECTOR, L.EVERY, d.dim)]
    layouts["norms behind"] = layouts["llama"]._replace(rows=behind)
    local = {"layers": [0, 1, 3], "window": 3, "rope_theta": 100.0}
    return [
        ("llama", GROUPED, {}, {}),
        ("llama, RoPE leaving two layers alone", GROUPED, {}, {"unturned": (1, 3)}),
        ("qwen2 and qwen3", GROUPED, {"bias": True, "qk_norm": True, "head_dim": 32}, {"rms_norm_eps": 1e-6}),
        ("gpt2", FILLED, {"arch": "gpt2"}, {}),
        ("neox", FILLED, {"arch": "neox"}, {"rotary": 4, "parallel_residual": True}),
        ("qwen35", GROUPED, qwen35, {"rotary": 4}),
        ("lfm2", GROUPED, {"arch": "lfm2", "qk_norm": True, "convolution": {"layers": "ccac", "taps": 3}}, {}),
        ("llama, in a rotated basis", GROUPED, rotated(GROUPED, {"head_dim": 32}), {}),
        ("qwen35, in a rotated basis", GROUPED, rotated(GROUPED, qwen35), {"rotary": 4}),
        ("rms and gelu", FILLED, {"arch": "rms and gelu"}, {"parallel_residual": True}),
        ("layer norms and a gate", FILLED, {"arch": "layer norms and a gate"}, {"rotary": 4, "parallel_residual": True}),
        ("llama, RoPE leaving every layer alone", GROUPED, {}, {"unturned": (0, 1, 2, 3)}),
        ("gemma3", [64, 128, 6, 4, 1, 320, 48], {"arch": "gemma3", "head_dim": 32},
         {"local": {"layers": [0, 1, 2, 4, 5], "window": 4, "rope_theta": 100.0}, "rms_norm_eps": 1e-6}),
        ("gemma3, a classifier of its own and no local layer", [64, 128, 3, 4, 2, -320, 48], {"arch": "gemma3"}, {}),
        ("a gelu in the gate", GROUPED, {"arch": "a gelu in the gate"}, {}),
        ("local layers", GROUPED, {"arch": "local layers"}, {"local": local}),
        ("norms behind", GROUPED, {"arch": "norms behind"}, {}),
    ]


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tree import python_folder
    sys.path.insert(0, python_folder(sys.argv[1]))
    import llama2_numpy as L
    import made_up

    folder, said = sys.argv[2], []
    tokenizer = made_up.pack_tokenizer(made_up.tiny_vocab(320))
    for index, (name, header, form, options) in enumerate(cases(L)):
        checkpoint = made_up.checkpoint(header, form, "float32", tensor=made_up.uniform(6 + index))
        external = External(len(checkpoint), lambda offset, length: checkpoint[offset:offset + length])
        try:
            L.Llama(None, b"", dtype="float32", external=external, **form, **options)
        except Handed:
            pass
        plan = dict(external.plan)
        derived = {}
        for table, value in plan["derived"].items():
            derived[table] = f"{index}.{table}"
            open(os.path.join(folder, derived[table]), "wb").write(bytes(value))
        plan["derived"] = derived
        model = L.Llama(checkpoint, tokenizer, dtype="float32", **form, **options)
        logits = np.stack([model.forward(token, pos) for pos, token in enumerate(TOKENS)]).astype(np.float32)
        open(os.path.join(folder, f"{index}.bin"), "wb").write(checkpoint)
        open(os.path.join(folder, f"{index}.logits"), "wb").write(logits.tobytes())
        said.append({"name": name, "plan": plan, "tokens": TOKENS, "checkpoint": f"{index}.bin", "logits": f"{index}.logits",
                     "facts": {"layer_norm": model.layer_norm, "gated_ffn": model.gated_ffn,
                               "activation": L.layout_of(form.get("arch", "llama")).activation,
                               "local": list(model.local["layers"]) if model.local else [], "window": model.local["window"] if model.local else 0}})
    print(json.dumps(said))
