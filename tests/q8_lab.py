# q8_lab.py (the review of T245 and T247, a probe for CI, not for main): transformers' own forward pass of Qwen3.5 with
# the weights rounded the way llama.cpp's Q8_0 rounds them, in Hugging Face's order, to ask what the Q8_0 GGUF of the
# 4B costs without the page's reader in the way, and what a reader that left one of the eight value-head tensors
# in llama.cpp's order would cost (T245).
#
#   python tests/q8_lab.py <directory> --text <file> [--jobs all|v0,v1,only-,misread-...] [--positions 96] [--first 192]
#   python tests/q8_lab.py --tiny                    (a made-up model, in a few minutes: the script's own checks)
#
# The weights stay bfloat16 (what the checkpoint holds: 8.4 GB for the 4B; float32 is 17 GB on a runner of 16 GB), and a
# matrix is widened to float32, optionally rounded to Q8_0 and optionally put in llama.cpp's tiled order (what a reader
# that does not put it back would read), each time it is multiplied. The activations are float32 as in
# tests/reference_qwen35.py (the engine is 1.4e-4 from transformers on the original there).
#
# Every line of the log begins with "Q8LAB".
import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))

say = lambda *parts: print("Q8LAB", *parts, flush=True)

# ---------------------------------------------------------------------------------------------- the rounding
# Q8_0, llama.cpp's quantize_row_q8_0_ref: per 32 along the input axis, d = the largest magnitude / 127 (float32), the
# values times 1/d rounded half away from zero, d kept as float16 and the value read back as q * float(d). The same as
# tests/gguf_check.py's q8_0_of (the selftest holds the two to each other).


def q8_0(w, shift=0):
    if shift:  # blocks that start `shift` along (the last wraps around): another rounding of about the same size
        w = torch.roll(w, -shift, dims=-1)
    groups = w.reshape(-1, 32)
    d = groups.abs().amax(dim=1) / 127
    inverse = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    scaled = groups * inverse[:, None]
    scale = d.to(torch.float16).to(torch.float32)[:, None]
    rounded = torch.sign(scaled) * torch.floor(scaled.abs() + 0.5)
    out = torch.where(scale > 0, rounded * scale, torch.zeros_like(scaled)).reshape(w.shape)
    return torch.roll(out, shift, dims=-1) if shift else out


# ------------------------------------------------------------------------------------------ llama.cpp's order
# conversion/qwen.py's _LinearAttentionVReorderBase._reorder_v_heads at dcd387a4, written out here from the code (the
# selftest runs the file's own function against this when LLAMACPP_QWEN names the file): Hugging Face's value heads of
# key head 0, then those of key head 1, ... [keys, per, size] become [per, keys, size] along the axis, after what
# stands before them.


def tile(w, first, keys, per, size, axis):
    w = w.movedim(axis, 0)
    heads = w[first:].reshape(keys, per, size, *w.shape[1:]).transpose(0, 1).reshape(-1, *w.shape[1:])
    return torch.cat([w[:first], heads]).movedim(0, axis)


# the eight tensors of a linear-attention layer that llama.cpp reorders, by its names (as tests/gguf_check.py's
# VALUE_HEADS): (whether q and k stand before the heads, whether a head is value_dim entries or one, the axis)
EIGHT = {"attn_qkv": (True, True, 0), "attn_gate": (False, True, 0), "ssm_alpha": (False, False, 0),
         "ssm_beta": (False, False, 0), "ssm_a": (False, False, 0), "ssm_dt.bias": (False, False, 0),
         "ssm_conv1d": (True, True, 0), "ssm_out": (False, True, 1)}
# which parameter of transformers holds each
HF_NAME = {"attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z", "ssm_alpha": "linear_attn.in_proj_a",
           "ssm_beta": "linear_attn.in_proj_b", "ssm_out": "linear_attn.out_proj", "ssm_a": "linear_attn.A_log",
           "ssm_dt.bias": "linear_attn.dt_bias", "ssm_conv1d": "linear_attn.conv1d.weight"}
# the matrices Q8_0 rounds (everything llama.cpp keeps in float32 stays: the norms, A_log, dt_bias, the convolution)
KIND_OF = {"linear_attn.in_proj_qkv": "qkv", "linear_attn.in_proj_z": "z", "linear_attn.in_proj_a": "a",
           "linear_attn.in_proj_b": "b", "linear_attn.out_proj": "out", "self_attn.q_proj": "q", "self_attn.k_proj": "k",
           "self_attn.v_proj": "v", "self_attn.o_proj": "o", "mlp.gate_proj": "gate", "mlp.up_proj": "up",
           "mlp.down_proj": "down", "embed_tokens": "embed"}
GROUPS = {"all": set(KIND_OF.values()), "embedding": {"embed"}, "linear": {"qkv", "z", "a", "b", "out"},
          "gates": {"a", "b"}, "full": {"q", "k", "v", "o"}, "ffn": {"gate", "up", "down"}}


class Lab:
    """What the wrapped matrices do on each call: which kinds are rounded to Q8_0 (and by what shift), and which of the
    eight are read in llama.cpp's order (put in it from Hugging Face's, which is what reading a GGUF's tensor as
    Hugging Face's would give)."""

    def __init__(self, text_config):
        c = text_config
        self.keys, self.values = c.linear_num_key_heads, c.linear_num_value_heads
        self.key_dim, self.value_dim = c.linear_key_head_dim, c.linear_value_head_dim
        self.per = self.values // self.keys
        self.rounds, self.shift, self.misread = set(), 0, set()

    def spec(self, gguf_name):
        after, of_a_head, axis = EIGHT[gguf_name]
        return (2 * self.keys * self.key_dim if after else 0, self.keys, self.per, self.value_dim if of_a_head else 1, axis)

    def set(self, rounds=(), shift=0, misread=()):
        self.rounds, self.shift, self.misread = set(rounds), shift, set(misread)

    def matrix(self, kind, weight):
        w = weight.float()
        if kind in self.rounds:
            w = q8_0(w, self.shift)
        for gguf in self.misread:
            if KIND_OF.get(HF_NAME[gguf]) == kind:
                w = tile(w, *self.spec(gguf))
        return w


class QLinear(torch.nn.Module):
    def __init__(self, linear, kind, lab):
        super().__init__()
        self.weight, self.kind, self.lab = linear.weight, kind, lab

    def forward(self, x):
        return torch.nn.functional.linear(x, self.lab.matrix(self.kind, self.weight))


class QEmbedding(torch.nn.Module):
    def __init__(self, embedding, lab):
        super().__init__()
        self.weight, self.lab = embedding.weight, lab

    def forward(self, ids):
        rows = torch.nn.functional.embedding(ids, self.weight).float()
        return q8_0(rows, self.lab.shift) if "embed" in self.lab.rounds else rows


def install(text, lab, narrow=True):
    """The matrices of the language model behind the lab's wrappers, bfloat16 (exactly what the checkpoint holds)."""
    count = 0
    for parent_name, parent in list(text.named_modules()):
        for child_name, child in list(parent.named_children()):
            name = f"{parent_name}.{child_name}" if parent_name else child_name
            kind = next((kind for suffix, kind in KIND_OF.items() if name.endswith(suffix)), None)
            if kind is None or not isinstance(child, (torch.nn.Linear, torch.nn.Embedding)):
                continue
            if narrow and child.weight.dtype != torch.bfloat16:
                narrowed = child.weight.data.to(torch.bfloat16)
                if count < 3:  # the checkpoint's values are bfloat16's already: nothing is lost by holding them so
                    assert torch.equal(narrowed.float(), child.weight.data), f"{name} is not bfloat16's"
                child.weight.data = narrowed
            wrapper = QEmbedding(child, lab) if kind == "embed" else QLinear(child, kind, lab)
            setattr(parent, child_name, wrapper)
            count += 1
    return count


class Misreading:
    """The three tensors of the eight that are not matrices (A_log, dt_bias, the convolution's weight), put in
    llama.cpp's order in place while a variant runs, and back after it."""

    def __init__(self, text, lab, kinds):
        self.saved = []
        for gguf in kinds:
            if gguf in ("ssm_a", "ssm_dt.bias", "ssm_conv1d"):
                for name, parameter in text.named_parameters():
                    if name.endswith(HF_NAME[gguf]):
                        self.saved.append((parameter, parameter.data.clone()))
                        parameter.data = tile(parameter.data, *lab.spec(gguf))

    def undo(self):
        for parameter, data in self.saved:
            parameter.data = data


# ------------------------------------------------------------------------------------------------ the forward pass
@torch.no_grad()
def run(text, classifier, lab, rows, pad, positions, chunk=16384):
    """(the negative log likelihood of each next token, in float64, per row; the logits of the first `positions`
    positions of row 0). One batch: the language model once, then the classifier a chunk of the vocabulary at a time."""
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for at, row in enumerate(rows):
        ids[at, :len(row)] = torch.tensor(row)
        mask[at, :len(row)] = 1
    hidden = text(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    targets = torch.zeros((len(rows), width), dtype=torch.long)
    for at, row in enumerate(rows):
        targets[at, :len(row) - 1] = torch.tensor(row[1:])
    vocabulary = classifier.shape[0]
    lse = torch.full((len(rows), width), -math.inf, dtype=torch.float64)
    target_logit = torch.zeros((len(rows), width), dtype=torch.float64)
    first = np.zeros((positions, vocabulary), dtype=np.float32)
    for start in range(0, vocabulary, chunk):
        piece = lab.matrix("embed", classifier[start:start + chunk])
        logits = hidden @ piece.T
        lse = torch.logaddexp(lse, torch.logsumexp(logits.double(), dim=-1))
        inside = (targets >= start) & (targets < start + logits.shape[-1])
        chosen = logits.gather(-1, (targets - start).clamp(0, logits.shape[-1] - 1)[..., None])[..., 0].double()
        target_logit = torch.where(inside, chosen, target_logit)
        first[:, start:start + logits.shape[-1]] = logits[0, :positions].numpy()
    nll = lse - target_logit
    return [nll[at, :len(row) - 1].numpy() for at, row in enumerate(rows)], first


# ------------------------------------------------------------------------------------------------------ the report
def perplexity(values):
    return math.exp(float(np.mean(np.concatenate(values))))


def paired(label, base, other, tokenizer, ids, first):
    """The change from base to other, token by token (the same tokens): over the first `first` tokens of the first
    window and over all of them, by window and by quarter of the first, and where it is large."""
    delta = [o - b for b, o in zip(base, other)]  # below 0 is likelier

    def stats(values):
        values = np.concatenate(values)
        return float(values.mean()), float(values.std() / math.sqrt(len(values))), float((values < 0).mean())

    mean, se, better = stats([delta[0][:first]])
    all_mean, all_se, all_better = stats(delta)
    say(f"{label}: the first {first}: {100 * (math.exp(mean) - 1):+.2f}% perplexity ({mean:+.4f} nats a token, standard error "
        f"{se:.4f} as if the tokens were independent; {100 * better:.0f}% of the tokens likelier); all "
        f"{sum(len(d) for d in delta)}: {100 * (math.exp(all_mean) - 1):+.2f}% ({all_mean:+.4f}, {all_se:.4f}, "
        f"{100 * all_better:.0f}% likelier); by window " + ", ".join(f"{float(d.mean()):+.4f}" for d in delta)
        + "; by quarter of the first " + ", ".join(f"{float(delta[0][i:i + first // 4].mean()):+.4f}" for i in range(0, first, first // 4)))
    largest = np.argsort(-np.abs(delta[0][:first]))[:5]
    say(f"{label}: the largest changes among the first {first}: " + "; ".join(
        f"{int(i)} {tokenizer.decode([ids[0][int(i) + 1]])!r} {delta[0][i]:+.3f} (was {base[0][i]:.3f})" for i in largest))


def compare_logits(label, base, other):
    """The logits of the first positions against base's: the largest and the mean difference, how often the most likely
    token is the same, where it is not (and the gap between base's best two there), the KL of base's distribution from
    other's."""
    a, b = np.asarray(base, dtype=np.float64), np.asarray(other, dtype=np.float64)
    gap = np.abs(a - b)
    same = a.argmax(axis=1) == b.argmax(axis=1)
    top = np.sort(a, axis=1)[:, -2:]
    margins = top[:, 1] - top[:, 0]
    a0 = a - a.max(axis=1, keepdims=True)
    b0 = b - b.max(axis=1, keepdims=True)
    log_a = a0 - np.log(np.exp(a0).sum(axis=1, keepdims=True))
    log_b = b0 - np.log(np.exp(b0).sum(axis=1, keepdims=True))
    kl = (np.exp(log_a) * (log_a - log_b)).sum(axis=1)
    where = [f"{int(i)} (base's best two {margins[i]:.3f} apart)" for i in np.nonzero(~same)[0]]
    say(f"{label}: logits of {len(a)} positions: largest difference {gap.max():.3e}, mean {gap.mean():.3e}, the same most "
        f"likely token at {int(same.sum())} of {len(a)}{(' — differs at ' + ', '.join(where)) if where else ''}; KL "
        f"{kl.mean():.3e} on average, {kl.max():.3e} at most")


# ----------------------------------------------------------------------------------------------------- selftest
def selftest(lab):
    """What this script's own parts are held to: Q8_0 against tests/gguf_check.py's, tile against what the converter
    puts back (llama2_convert.untiled), and, where the file of llama.cpp's is at hand, against its own function."""
    import gguf_check
    import llama2_convert

    rng = np.random.default_rng(1)
    w = rng.standard_normal((6, 96)).astype(np.float32) * 0.02
    w[2, 5] = 0.3
    w[4, :32] = 0
    expected = gguf_check.q8_0_of(w)
    got = q8_0(torch.tensor(w)).numpy()
    assert np.array_equal(expected, got), f"Q8_0: {np.abs(expected - got).max()}"
    assert q8_0(torch.tensor(w), 16).shape == w.shape
    say("selftest: Q8_0 is tests/gguf_check.py's q8_0_of to the bit on random values")

    def sample(gguf):
        first, keys, per, size, axis = lab.spec(gguf)
        shape = [3, 5]
        shape.insert(axis, first + keys * per * size)
        return torch.tensor(rng.standard_normal(shape).astype(np.float32)), (first, keys, per, size, axis)

    for gguf in EIGHT:
        x, spec = sample(gguf)
        once = tile(x, *spec)
        back = llama2_convert.untiled(once.numpy(), *spec)
        assert np.array_equal(back, x.numpy()), f"{gguf}: the converter's untiled is not tile's inverse"
        assert not np.array_equal(once.numpy(), x.numpy())
    say("selftest: llama2_convert.untiled puts back what tile does, for the eight tensors")
    path = os.environ.get("LLAMACPP_QWEN")
    if path:
        import ast

        theirs = None
        for node in ast.walk(ast.parse(Path(path).read_text())):
            if isinstance(node, ast.ClassDef) and node.name == "_LinearAttentionVReorderBase":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "_reorder_v_heads":
                        item.decorator_list = []
                        namespace = {"torch": torch, "Tensor": torch.Tensor}
                        exec(compile(ast.Module([item], []), path, "exec"), namespace)
                        theirs = namespace["_reorder_v_heads"]
        assert theirs, "no _reorder_v_heads in the file"
        for gguf in EIGHT:
            x, (first, keys, per, size, axis) = sample(gguf)
            # llama.cpp applies it to the value part only (modify_tensors cuts q and k off first)
            part = x.narrow(axis, first, x.shape[axis] - first)
            llama_cpp = torch.cat([x.narrow(axis, 0, first), theirs(part, axis, keys, per, size)], dim=axis)
            assert torch.equal(llama_cpp, tile(x, first, keys, per, size, axis)), f"{gguf}: llama.cpp's own function differs"
        say(f"selftest: llama.cpp's own _reorder_v_heads ({path}) is tile() for the eight tensors")


# ------------------------------------------------------------------------------------------------------- the jobs
def jobs_of(names):
    """[(name, kinds rounded, shift, misread, row set)]: the variants. Row sets: full (3 windows under both BOSs),
    first (the first window under both), one (the first window under BOS 248044). v0 is always first, and v1 comes
    with the misreadings, which are held to it."""
    every = [("v0", (), 0, (), "full"), ("v1", GROUPS["all"], 0, (), "full")]
    for group in ("embedding", "linear", "gates", "full", "ffn"):
        every.append((f"only-{group}", GROUPS[group], 0, (), "first"))
    for group in ("embedding", "linear", "full", "ffn"):
        every.append((f"all-but-{group}", GROUPS["all"] - GROUPS[group], 0, (), "first"))
    for shift in (16, 8, 24):
        every.append((f"shifted-{shift}", GROUPS["all"], shift, (), "first"))
    for gguf in EIGHT:
        every.append((f"misread-{gguf}", GROUPS["all"], 0, (gguf,), "one"))
    if names == "all":
        return every
    wanted = names.split(",")
    picked = [job for job in every if any(job[0] == w or (w.endswith("-") and job[0].startswith(w)) for w in wanted)]
    names_of = {job[0] for job in picked}
    needed = {"v0"} | ({"v1"} if any(name.startswith("misread-") for name in names_of) else set())
    return [job for job in every if job[0] in names_of | needed]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?")
    parser.add_argument("--text")
    parser.add_argument("--jobs", default="all")
    parser.add_argument("--positions", type=int, default=96)
    parser.add_argument("--first", type=int, default=192)
    parser.add_argument("--model", default="4B")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--save")
    arguments = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    import transformers

    say(f"transformers {transformers.__version__}, torch {torch.__version__}, {os.cpu_count()} threads")
    if arguments.tiny:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        config = Qwen3_5TextConfig(
            vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=32,
            linear_value_head_dim=32, max_position_embeddings=256, eos_token_id=7, tie_word_embeddings=True,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                             "mrope_section": [11, 11, 10], "mrope_interleaved": True})
        torch.manual_seed(245)
        model = Qwen3_5ForCausalLM(config).to(torch.float32).eval()
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                if key.endswith("linear_attn.norm.weight"):
                    parameter.copy_(1.0 + 0.3 * torch.randn_like(parameter))
                elif key.endswith("A_log"):
                    parameter.copy_(torch.log(torch.rand_like(parameter) * 4 + 0.1))
                elif parameter.ndim > 1 or key.endswith("dt_bias"):
                    parameter.copy_(0.2 * torch.randn_like(parameter))
        text, text_config, classifier = model.model, config, model.lm_head.weight
        rng = np.random.default_rng(245)
        tokens = [int(t) for t in rng.integers(0, 320, 150)]
        tokenizer = type("T", (), {"decode": staticmethod(lambda ids: " ".join(f"w{i}" for i in ids))})
        bos_values = {"eot": 7, "ims": 8}
        windows = [tokens[:60], tokens[60:120], tokens[120:150]]
        first, positions, narrow = 24, 16, False
        with torch.no_grad():  # the plain forward pass the lab's v0 is held to
            direct = model(input_ids=torch.tensor([[7] + windows[0]])).logits[0].numpy()
    else:
        import tokenizers
        from transformers import Qwen3_5ForConditionalGeneration

        import reference_qwen35 as ref

        directory = Path(arguments.directory)
        entry = ref.LARGE[arguments.model]
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
            ref.fetch(name, directory, entry["repo"], entry["revision"])
        for shard in sorted(set(json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"].values())):
            ref.fetch(shard, directory, entry["repo"], entry["revision"])
        say(f"{entry['repo']}@{entry['revision']}: the files are here")
        tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
        tokens = tokenizer.encode(Path(arguments.text).read_text(), add_special_tokens=False).ids[:1500]
        windows = [tokens[start:start + 511] for start in range(0, len(tokens), 511)]
        bos_values = {"eot": 248044, "ims": 248045}
        first, positions, narrow, direct = arguments.first, arguments.positions, True, None
        began = time.perf_counter()
        model = Qwen3_5ForConditionalGeneration.from_pretrained(str(directory), dtype=torch.float32).eval()
        say(f"loaded as float32 in {time.perf_counter() - began:.0f} s")
        text, text_config, classifier = model.model.language_model, model.config.text_config, model.lm_head.weight
        assert classifier.data_ptr() == text.embed_tokens.weight.data_ptr(), "the classifier is not the embedding"

    lab = Lab(text_config)
    selftest(lab)
    count = install(text, lab, narrow)
    if narrow:
        model.model.visual = None  # (unused: text only)
        assert classifier.dtype == torch.bfloat16, "the classifier did not follow the embedding"
    gc.collect()
    say(f"{count} matrices are the lab's now (bfloat16 held, widened at each call)")

    rows_of = {
        "full": [[bos_values[b]] + w for b in ("eot", "ims") for w in windows],
        "first": [[bos_values[b]] + windows[0] for b in ("eot", "ims")],
        "one": [[bos_values["eot"]] + windows[0]],
    }
    pad = bos_values["eot"]
    sentence = None if arguments.tiny else ([pad] + tokenizer.encode(ref.TEXT, add_special_tokens=False).ids)[:positions]
    indices = lambda rows, b: [i for i, row in enumerate(rows) if row[0] == bos_values[b]]
    results = {}
    for name, rounds, shift, misread, rowset in jobs_of(arguments.jobs):
        began = time.perf_counter()
        lab.set(rounds, shift, misread)
        mis = Misreading(text, lab, misread)
        rows = rows_of[rowset]
        nlls, logits = run(text, classifier, lab, rows, pad, positions)
        mis.undo()
        results[name] = (rowset, nlls, logits)
        for b in ("eot", "ims"):
            picked = [nlls[i] for i in indices(rows, b)]
            if picked:
                say(f"{name}: BOS {b}: perplexity of the first {first} tokens {math.exp(float(picked[0][:first].mean())):.3f}, of "
                    f"all {sum(len(p) for p in picked)} {perplexity(picked):.3f}")
        say(f"{name}: {time.perf_counter() - began:.0f} s")
        if name == "v0" and not arguments.tiny:
            say("v0: the numbers of T245 and T247 for BOS 248044 are 7.873 (the first 192) and 14.834 (1500 tokens)")
        if name not in ("v0",) and "v0" in results and not name.startswith("misread-"):
            base_set, base_nlls, base_logits = results["v0"]
            base_rows = rows_of[base_set]
            for b in ("eot", "ims"):
                here = indices(rows, b)
                if here:
                    paired(f"{name} against v0, BOS {b}", [base_nlls[i] for i in indices(base_rows, b)][:len(here)],
                           [nlls[i] for i in here], tokenizer, [rows[i] for i in here], first)
            compare_logits(f"{name} against v0", results["v0"][2], logits)
        if name.startswith("misread-"):
            base_set, base_nlls, base_logits = results["v1"]
            paired(f"{name} against v1, BOS eot", [base_nlls[0]], [nlls[0]], tokenizer, [rows[0]], first)
            compare_logits(f"{name} against v1", base_logits, logits)
        if arguments.save and name in ("v0", "v1"):
            np.save(f"{arguments.save}-{name}-logits.npy", logits)
            np.save(f"{arguments.save}-{name}-nll.npy", nlls[0][:first])
        if sentence and name in ("v0", "v1"):
            # the 96 positions of tests/reference_qwen35.py's text (the engine's comparison with transformers there), for
            # the engine on the GGUF to be held to (tests/q8_compare.py)
            _, here = run(text, classifier, lab, [sentence], pad, len(sentence))
            results[f"{name}-sentence"] = here
            if arguments.save:
                np.save(f"{arguments.save}-{name}-sentence-logits.npy", here)
            if name == "v1":
                compare_logits("v1 against v0 on the engine's 96 positions", results["v0-sentence"], here)
        gc.collect()
    if arguments.tiny:
        a = results["v0"][2]
        difference = float(np.abs(a - direct[:len(a)]).max())
        say(f"selftest: the lab's v0 logits are the model's own to {difference:.2e}")
        assert difference < 1e-4


if __name__ == "__main__":
    main()
