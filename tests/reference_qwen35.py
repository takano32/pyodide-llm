# reference_qwen35.py
# T229: the engine's Qwen3.5 (hybrid attention: Gated DeltaNet layers between full-attention ones) against
# transformers itself. PyTorch is not on the development machine, so this runs in CI:
#
#   pip install numpy tokenizers torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
#   pip install safetensors "transformers @ git+https://github.com/huggingface/transformers@7fb5bcd1d4b8a5c225a2c33429b2e9e023dd61ae"
#   python tests/reference_qwen35.py <directory for the download> [--only=made-up|real|fetch] [--positions=96]
#   (--only=fetch: the real model's files into the directory and no more, for tests/page_qwen35.sh)
#   python tests/reference_qwen35.py <directory> --model=4B [--from=gguf] [--text=<file>] [--minutes=40]
#
#   node tests/ci.mjs run tests.yml extra="bash tests/reference_qwen35.sh" --ref <branch> --grep "qwen35"
#   node tests/ci.mjs run tests.yml extra="bash tests/reference_qwen35.sh --model=4B" --ref <branch> --grep "qwen35"
#
# Two parts, and a third by itself, every line of the log beginning with "qwen35:".
#   made-up: tiny random models of transformers' own class (Qwen3_5ForCausalLM), which hold what the real 0.8B does
#     not: three value heads to a key head (the 27B's), key and value heads of two sizes, a classifier of its own.
#     transformers' logits, over the whole text at once (its chunked delta rule) and token by token with its cache
#     (its recurrent rule), against the naive reference of the unit tests (conftest.naive_qwen35_logits) and against
#     the engine, converted the way the page converts (llama2_convert.Conversion). And what each holds after the text (the
#     review of T229): the recurrent state and the convolution's rows of every linear-attention layer, the keys and
#     values of every full-attention one, which a fault shows in long before the logits do.
#   real: Qwen/Qwen3.5-0.8B at a fixed revision, float32. The same token ids through transformers and through the
#     engine (NumPy): the largest difference of the logits, how often the most likely token is the same, the states
#     after them, 16 greedy tokens of each for the chat prompt, and then 1024 positions of a longer text (the cache of
#     the full-attention layers doubling twice, a thousand tokens for the state to drift in).
#   --model=2B (T247): the real part on Qwen/Qwen3.5-2B (7.5 GB as float32, which a runner holds), before its greedy
#     text is fixed in tests/fixed_outputs.py: the same lines, and what transformers writes for that prompt in the
#     list's format, from the format's own first token.
#   --model=4B (T245): Qwen/Qwen3.5-4B, the smallest real model with more value heads than key heads (two to one), and
#     17 GB as float32, more than a runner's 16 GB of memory holds: see large(). The engine (NumPy, float32, the
#     original's safetensors converted the way the page converts) against transformers over the whole text at once.
#     --from=gguf: the engine on the list's GGUF instead (unsloth's Q8_0, whose value heads llama.cpp tiled), its
#     float32 checkpoint held to the safetensors' tensor by tensor, and its logits to transformers' of the original.
#     --text: transformers' perplexity of the original on 1500 tokens of that text too, as tests/perplexity.py counts.
#     --first=192: and the engine's on the first so many of them, token by token, against transformers' on the same
#     (what the weights of a GGUF cost, which the 1500 tokens are hours of for the engine here).
# The lines it holds the engine to are at the end of each part; anything past them is exit 1.
import gc
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
import llama2_convert  # noqa: E402
from llama2_convert import Conversion  # noqa: E402
from llama2_numpy import Llama  # noqa: E402
from conducting import Directory, Mapped, converted  # noqa: E402
from fetching import download  # noqa: E402

REPO, REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json", WEIGHTS]
# <|endoftext|>: what the converter begins every text with (llama2_convert.normalize), and so what ?hf= opens. NOT what the
# list's entries begin with: <|im_start|>, 248045 (T236). The engine agrees with transformers on the same ids whichever
# the first token is, but a perplexity read here is of this other way: a Qwen3.5 reads a text 18% (2B) to 45% (4B) worse
# after it, and its perplexity then moves by about +-3% under a rounding of its weights, so a Q8_0 GGUF's can read LOWER
# than the original's (the review of T245: 7.574 against 7.873 on 192 tokens, 4B; with <|im_start|> it is 4.215 against
# 4.217). tests/perplexity_prepare.py --entry <id> measures the list's way.
BOS = 248044
# a text of more than 64 tokens (transformers' chunk of the delta rule), English and Japanese
TEXT = ("Mount Fuji is the highest mountain in Japan, standing 3,776 metres above sea level on the island of Honshu. "
        "It is an active stratovolcano that last erupted from 1707 to 1708, and on clear days it can be seen from "
        "Tokyo, about 100 kilometres to the north-east. 富士山は日本で最も高い山で、古くから信仰の対象とされ、"
        "多くの絵画や文学作品に描かれてきました。夏には多くの登山者が山頂を目指します。")
# the chat format of one turn without thinking, as chat_template.jinja writes it (the template calls a macro, which
# the converter's reader does not read: T236 writes it into the list by hand)
CHAT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPT = "What is the capital of Japan? Answer in one sentence."
SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
NEW_TOKENS = 16

say = lambda *parts: print("qwen35:", *parts, flush=True)


def fetch(name, directory, repo=REPO, revision=REVISION):
    # (tests/fetching.py, T357: a download that stopped short is asked for again, which this copy of the loop did not see,
    # and a file the repository does not have is refused at once: this copy asked three times)
    return download(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", directory / name)


def differences(ours, theirs):
    """(largest difference, mean difference, positions whose most likely token is the same, the smallest gap between
    the two most likely tokens of theirs where the most likely token differs)"""
    ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    gap = np.abs(ours - theirs)
    same = ours.argmax(axis=1) == theirs.argmax(axis=1)
    top = np.sort(theirs, axis=1)[:, -2:]
    margins = (top[:, 1] - top[:, 0])[~same]
    return float(gap.max()), float(gap.mean()), int(same.sum()), float(margins.max()) if margins.size else 0.0


def relative(a, b):
    """how far a is from b, in b's own size: the norm of the difference over the norm of b"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# What the engine and transformers hold after the same positions, which a fault of the engine shows in long before the
# logits do (the logits of the review's 34 faults of the real model were 5e-2 to 30, their states 8e-3 to 1e+29, against
# 3e-6 for the right computation): per layer a linear-attention layer's recurrent state (value heads, key_dim, value_dim)
# and the last conv - 1 tokens' q, k and v before the convolution; a full-attention layer's keys (as Hugging Face has them:
# the first rotary values of a head not interleaved) and values. STATES: what is compared.
STATES = ("recurrent state", "convolution rows", "keys", "values")
STATE_LINE = 1e-4  # at least: ten times what transformers' own two computations differ by where that is more


def engine_states(llama, count):
    """{layer: (kind, [two arrays])}"""
    out = {}
    rot = llama.rotary
    order = np.concatenate([np.arange(0, rot, 2), np.arange(1, rot, 2)])  # the engine's index of Hugging Face's 0..rot-1
    for l, (lines, a) in enumerate(llama.slots):
        if lines:
            out[l] = ("linear", [llama.delta_state[a], llama.conv_state[a].T])
        else:
            keys = llama.key_cache[a, :, :count].copy()
            keys[..., :rot] = keys[..., :rot][..., order]
            out[l] = ("full", [keys, llama.value_cache[a, :, :count]])
    return out


def model_states(past, count):
    out = {}
    for l, layer in enumerate(past.layers):
        if hasattr(layer, "recurrent_states"):
            out[l] = ("linear", [layer.recurrent_states[0][0].float().numpy(), layer.conv_states[0][0, :, 1:].float().numpy()])
        else:
            out[l] = ("full", [layer.keys[0].float().numpy()[:, :count], layer.values[0].float().numpy()[:, :count]])
    return out


def state_distances(ours, theirs):
    """the largest relative distance over the layers, for each of STATES"""
    worst = dict.fromkeys(STATES, 0.0)
    for l, (kind, mine) in ours.items():
        for name, a, b in zip(STATES[:2] if kind == "linear" else STATES[2:], mine, theirs[l][1]):
            worst[name] = max(worst[name], relative(a, b))
    return worst


def compare_states(label, llama, count, at_once, stepped):
    """the engine's states against transformers' after count positions: each within the line of 10 times what transformers'
    own two computations (at once, token by token) differ by, and 1e-4 at least. True where they are not"""
    apart = state_distances(engine_states(llama, count), stepped)
    floor = state_distances(at_once, stepped)
    failed = False
    parts = []
    for name in STATES:
        line = max(STATE_LINE, 10 * floor[name])
        bad = apart[name] > line
        failed |= bad
        parts.append(f"{name} {apart[name]:.1e} (transformers' own {floor[name]:.1e}, line {line:.1e}){' — FAILED' if bad else ''}")
    say(f"{label}: the states after {count} positions against transformers': " + "; ".join(parts))
    return failed


# Past the 96 positions of the main comparison: LONG positions of a text through transformers (at once, its chunks of 64) and
# through the engine, the logits at LONG_AT (the keys and values of the full-attention layers double at 256 and 512 here, the
# linear layers' state has had a thousand tokens to drift in) and the states at the end. The engine ran 1024 positions
# of English Wikipedia at 1.6e-4 and 1.7e-6 from transformers in the review (CI's x86-64, 100 s).
LONG = 1024
LONG_AT = (0, 3, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1023)


def long_check(directory, tokenizer, llama, line):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    text = " ".join(f"Section {i}. {TEXT}" for i in range(12))
    ids = ([BOS] + tokenizer.encode(text, add_special_tokens=False).ids)[:LONG]
    at = [position for position in LONG_AT if position < len(ids)]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(directory), dtype=torch.float32).eval()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=torch.tensor(at))
    theirs, states = out.logits[0].float().numpy(), model_states(out.past_key_values, len(ids))
    del model, out
    gc.collect()
    began = time.perf_counter()
    kept = {}
    for pos, token in enumerate(ids):
        logits = llama.forward(token, pos)
        if pos in at:
            kept[pos] = logits.copy()
    largest, mean, same, margin = differences(np.stack([kept[position] for position in at]), theirs)
    apart = state_distances(engine_states(llama, len(ids)), states)
    ok = largest <= line and (same == len(at) or margin <= 2 * largest)
    bad = [name for name in STATES if apart[name] > STATE_LINE]
    say(f"long: the engine ran {len(ids)} positions in {time.perf_counter() - began:.1f} s; the logits at {len(at)} of them against transformers': "
        f"largest difference {largest:.2e}, mean {mean:.2e}, the same most likely token at {same} of {len(at)}"
        f"{'' if ok else f' — FAILED (the line: {line:.1e})'}; the states after them: "
        + "; ".join(f"{name} {apart[name]:.1e}" for name in STATES) + (f" — FAILED (the line: {STATE_LINE:.0e}): {', '.join(bad)}" if bad else ""))
    return not ok or bool(bad)


# --------------------------------------------------------------------------------------------- made-up models
MADE_UP = {
    "three value heads a key head": dict(linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=8,
                                         linear_value_head_dim=12, num_hidden_layers=8),
    "one value head a key head, tied": dict(linear_num_key_heads=4, linear_num_value_heads=4, linear_key_head_dim=8,
                                            linear_value_head_dim=8, num_hidden_layers=6, tie_word_embeddings=True,
                                            full_attention_interval=3),
    "every second layer, whole heads turn": dict(linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=16,
                                                  linear_value_head_dim=4, num_hidden_layers=4, full_attention_interval=2,
                                                  partial_rotary_factor=1.0, linear_conv_kernel_dim=3),
    # (the review) heads of 256 with RoPE over 64 of them, as every real Qwen3.5 has (the others' heads are 16): the
    # rotated count and its interleaving on heads that wide, the attention kernels' scale of 1/16. Eight layers, two of them
    # full: the converter takes a stacked kind of exactly one layer for a tensor of its own (a 1-layer Llama has no better
    # luck: "(32,), not (1, 32)"), which no real model has
    "heads of 256, a quarter turned": dict(linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
                                           linear_value_head_dim=32, num_hidden_layers=8, head_dim=256, num_attention_heads=2,
                                           num_key_value_heads=1, intermediate_size=64),
}


def made_up(name, settings, positions=80):
    import torch
    from conftest import naive_qwen35_logits
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    settings = dict(settings)
    every, layers = settings.pop("full_attention_interval", 4), settings["num_hidden_layers"]
    rotary = settings.pop("partial_rotary_factor", 0.25)
    shape = dict(intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    shape.update({key: settings.pop(key) for key in tuple(shape) if key in settings})
    config = Qwen3_5TextConfig(
        vocab_size=320, hidden_size=32, **shape,
        max_position_embeddings=256, eos_token_id=7,
        layer_types=["linear_attention" if (layer + 1) % every else "full_attention" for layer in range(layers)],
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": rotary,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        **settings)
    torch.manual_seed(229)
    model = Qwen3_5ForCausalLM(config).to(torch.float32).eval()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key.endswith("linear_attn.norm.weight"):
                parameter.copy_(1.0 + 0.3 * torch.randn_like(parameter))
            elif not (key == "lm_head.weight" and config.tie_word_embeddings):
                parameter.copy_(0.3 * torch.randn_like(parameter))
    if config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight, "the classifier is not tied"
    rng = np.random.default_rng(229)
    tokens = [int(token) for token in rng.integers(0, 320, positions)]
    ids = torch.tensor([tokens])
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        whole, at_once = out.logits[0].numpy(), model_states(out.past_key_values, positions)
        past, stepped = None, []
        for at in range(positions):
            out = model(input_ids=ids[:, at:at + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            stepped.append(out.logits[0, -1].numpy())
    tensors = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    if config.tie_word_embeddings:
        tensors.pop("lm_head.weight", None)
    published = {"model_type": "qwen3_5", "text_config": json.loads(config.to_json_string(use_diff=False))}
    text = published["text_config"]
    if "rope_parameters" not in text:  # an older transformers writes them at the top
        text["rope_parameters"] = {"rope_theta": text.get("rope_theta", 1e7), "partial_rotary_factor": rotary}
    text["rope_parameters"].setdefault("partial_rotary_factor", rotary)
    naive = naive_qwen35_logits(tensors, published, tokens)

    from test_convert import safetensors_file
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(320)]}}).encode()
    conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary, "tokenizer.json",
                            dtype="float32", max_seq_len=256, start=8 + size)
    conversion.feed(file[8 + size:])
    conversion.finish()
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    llama = Llama(bytes(conversion.checkpoint), conversion.tokenizer, **options)
    ours = [llama.forward(token, pos).copy() for pos, token in enumerate(tokens)]

    failed = False
    # the line, as for the real model: ten times what transformers' own two computations differ by (it runs the
    # delta rule in float32 whatever the model's type, at once in chunks of 64 and token by token: what float32
    # leaves between two right computations of these random weights, 2.3e-3 for the first model in CI's first run),
    # and no less than 2e-3
    floor = differences(stepped, whole)[0]
    line = max(2e-3, 10 * floor)
    say(f"made-up ({name}), transformers token by token against transformers at once: largest difference {floor:.2e}")
    pairs = [("the naive reference against transformers at once", naive, whole),
             ("the engine against transformers at once", ours, whole),
             ("the engine against transformers token by token", ours, stepped)]
    for what, a, b in pairs:
        largest, mean, same, margin = differences(a, b)
        ok = largest <= line and (same == positions or margin <= 2 * largest)
        failed |= not ok
        say(f"made-up ({name}), {what}: largest difference {largest:.2e}, mean {mean:.2e}, the same most likely "
            f"token at {same} of {positions} positions{'' if ok else f' — FAILED (the line: {line:.1e})'}")
    failed |= compare_states(f"made-up ({name})", llama, positions, at_once, model_states(past, positions))
    say(f"made-up ({name}): options {json.dumps({key: options[key] for key in ('arch', 'linear', 'head_dim', 'rotary', 'rope_theta', 'rms_norm_eps') if key in options})}")
    return failed


# ------------------------------------------------------------------------------------------------- the real model
# the other models the real part runs on (--model=): one file of weights under a shard's name, as the 0.8B's
OTHERS = {"2B": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc")}
# tests/fixed_outputs.py's prompt in the list's format without thinking (src/models.js's QWEN35_AT_ONCE after its BOS,
# <|im_start|>): the ids the page sends, with no <|endoftext|> in front
LISTED = "<|im_start|>user\nこれからの流行りを3つ挙げてください。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def real(directory, positions, repo=REPO, revision=REVISION):
    import tokenizers
    import torch
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration

    say(f"real: {repo}@{revision}, transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}")
    for name in FILES:
        fetch(name, directory, repo, revision)
    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))

    # the conversion, the way the page does it (T374.4: by the conduct of a conversion, answered from the files fetched
    # above and no others), float32
    sink = Mapped(directory / "float32.bin")
    began = time.perf_counter()
    conversion = converted(Directory(directory), {"weights": WEIGHTS, "tokenizer": "tokenizer.json"}, dtype="float32", sink=sink)
    sink.data.flush()
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    shown = {key: (value if key != "specials" else f"{len(value)} of them") for key, value in options.items()}
    say(f"real: converted in {time.perf_counter() - began:.1f} s, {sink.path.stat().st_size} bytes, options {json.dumps(shown)}")
    if "template" in conversion.options:
        say(f"real: the converter read a template: {json.dumps(conversion.options['template'])}")
    llama = Llama(np.memmap(sink.path, dtype=np.uint8, mode="r"), conversion.tokenizer, kernels=None,
                  **{**options, "specials": SPECIALS, "stop_tokens": [248044, 248046]})

    # the same ids for both: the page's BOS, then the text as the real tokenizer splits it (and the engine's own
    # tokenizer has to split it the same)
    ids = [BOS] + tokenizer.encode(TEXT, add_special_tokens=False).ids
    mine = [BOS] + llama.tokenizer.encode(TEXT, llama.specials)
    chat = CHAT.format(prompt=PROMPT)
    chat_ids = tokenizer.encode(chat, add_special_tokens=False).ids
    chat_mine = llama.tokenizer.encode(chat, llama.specials)
    failed = mine != ids or chat_mine != chat_ids
    say(f"real: {len(ids)} tokens of text and {len(chat_ids)} of the chat prompt, the engine's tokenizer gives "
        f"{'the same ids' if not failed else 'OTHER IDS — FAILED'}")
    ids = ids[:positions]

    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(directory), dtype=torch.float32).eval()
    tensor = torch.tensor([ids])
    with torch.no_grad():
        began = time.perf_counter()
        out = model(input_ids=tensor, use_cache=True)
        whole, at_once = out.logits[0].float().numpy(), model_states(out.past_key_values, len(ids))
        say(f"real: transformers, {len(ids)} positions at once in {time.perf_counter() - began:.1f} s")
        past, stepped = None, []
        for at in range(len(ids)):
            out = model(input_ids=tensor[:, at:at + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            stepped.append(out.logits[0, -1].float().numpy())
        generated = model.generate(torch.tensor([[BOS] + chat_ids]), max_new_tokens=NEW_TOKENS, do_sample=False)[0].tolist()
        listed = tokenizer.encode(LISTED, add_special_tokens=False).ids
        as_listed = model.generate(torch.tensor([listed]), max_new_tokens=NEW_TOKENS, do_sample=False)[0].tolist()[len(listed):]
    theirs = generated[1 + len(chat_ids):]
    say(f"real: transformers wrote for the list's prompt, from the format's first token: "
        f"{json.dumps(tokenizer.decode(as_listed, skip_special_tokens=False), ensure_ascii=False)} {as_listed}")
    stepped_states = model_states(past, len(ids))
    del model, out
    gc.collect()

    began = time.perf_counter()
    ours = [llama.forward(token, pos).copy() for pos, token in enumerate(ids)]
    say(f"real: the engine (NumPy, float32), {len(ids)} positions in {time.perf_counter() - began:.1f} s")
    for what, logits in (("transformers", whole), ("the engine", ours)):
        logits = np.asarray(logits)
        say(f"real: logits of {what}: min {logits.min():.3f}, max {logits.max():.3f}, mean {logits.mean():.4f}, "
            f"std {logits.std():.4f}")
    floor = differences(stepped, whole)[0]
    say(f"real: transformers token by token against transformers at once: largest difference {floor:.2e} "
        f"(its two forms of the delta rule: what float32 leaves between two right computations)")
    # the line: ten times what transformers' own two computations differ by, and no less than 1e-3 (the engine's 1.6e-4 to
    # 2.0e-4 and transformers' 1.2e-4 to 1.3e-4 in CI's runs); the most likely token the same wherever the reference's best
    # two are further apart than twice the difference. The review of T229 held the engine to it with 34 faults one at a time
    # (tests/probe_qwen35.py, a throwaway branch): the weakest, RoPE over 32 values of a head and not 64, is 4.6e-2 away
    # (a line of 2e-2 let it by 2.3 times), and the states (below) are 3e-6 away from transformers' where the weakest
    # fault's are 8e-3. The line was 2e-2 before.
    line = max(1e-3, 10 * floor)
    for what, reference in (("transformers at once", whole), ("transformers token by token", stepped)):
        largest, mean, same, margin = differences(ours, reference)
        ok = largest <= line and (same == len(ids) or margin <= 2 * largest)
        failed |= not ok
        say(f"real: the engine against {what}: largest difference {largest:.2e}, mean {mean:.2e}, the same most "
            f"likely token at {same} of {len(ids)} positions (the largest gap between the reference's best two where "
            f"it is not: {margin:.2e}){'' if ok else f' — FAILED (the line: {line:.1e})'}")

    failed |= compare_states("real", llama, len(ids), at_once, stepped_states)

    # 16 greedy tokens for the chat prompt, by transformers' generate() and by the engine's
    text = "".join(llama.generate(chat, steps=len(chat_ids) + NEW_TOKENS, temperature=0.0, echo=False))
    encoded = llama.tokenizer.encode(text, llama.specials) if text else []
    wrote = tokenizer.decode(theirs, skip_special_tokens=False)
    say(f"real: transformers wrote {json.dumps(wrote, ensure_ascii=False)} {theirs}")
    say(f"real: the engine wrote {json.dumps(text, ensure_ascii=False)} {encoded}")
    stop = next((at for at, token in enumerate(theirs) if token in (248044, 248046)), len(theirs))
    same = tokenizer.decode(theirs[:stop], skip_special_tokens=False) == text
    failed |= not same
    say(f"real: the two wrote {'the same' if same else 'OTHER TEXTS — FAILED'}")
    failed |= long_check(directory, tokenizer, llama, line)
    sink.path.unlink()
    return failed


# ------------------------------------------------------------------------- a model past the runner's memory (T245)
LARGE = {"4B": {"repo": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                "gguf": ("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523", "Qwen3.5-4B-Q8_0.gguf")}}
WINDOW, TOKENS = 512, 1500  # tests/perplexity.mjs's


def prepared(source, out):
    """tests/perplexity_prepare.py in a process of its own: the page's conversion to float32 into <out>.bin (a memory
    map: 17 GB), and the options it gives Llama()."""
    began = time.perf_counter()
    run = subprocess.run([sys.executable, str(HERE / "perplexity_prepare.py"), str(source), str(out), "float32"],
                         capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(f"perplexity_prepare.py: {run.stderr[-2000:]}")
    options = json.loads(Path(f"{out}.json").read_text())
    shown = {key: (value if key != "specials" else f"{len(value)} of them") for key, value in options.items()}
    say(f"large: {source.name} converted in {time.perf_counter() - began:.0f} s, {Path(f'{out}.bin').stat().st_size} bytes, "
        f"options {json.dumps(shown)}")
    return options


def pinned(data, spare=3 << 29):
    """Keeps the end of a memory-mapped checkpoint in memory (mlock), all but spare bytes of what is free. The engine
    reads every weight for every token, and where the file is larger than the memory the page cache would drop each
    page just before it is wanted again (the whole file from the disk for every token); pinned, only the part that
    does not fit is read again. Returns the bytes pinned (0 where the system refuses)."""
    import ctypes
    import resource

    available = next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemAvailable:")) * 1024
    page = os.sysconf("SC_PAGE_SIZE")
    length = min(len(data), max(0, available - spare)) // page * page
    first = (len(data) - length + page - 1) // page * page
    try:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, OSError):
        pass  # as much as the limit allows (reference_qwen35.sh lifts it)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if first >= len(data) or libc.mlock(data.ctypes.data + first, len(data) - first):
        say(f"large: nothing pinned (errno {ctypes.get_errno()}, {available} bytes available)")
        return 0
    return len(data) - first


def theirs(directory, ids_file):
    """transformers on the float32 of the original, in a process of its own (so that what it took is given back
    before the engine runs): the logits of the text's positions, and with --text the negative log likelihoods of the
    windows of tests/perplexity.py, all rows in one forward pass (the weights are 18 GB of float32 in 16 GB of memory:
    a pass goes through the layers once, and what does not fit goes to the swap once)."""
    import torch
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration

    rows = json.loads(Path(ids_file).read_text())
    began = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(directory), dtype=torch.float32).eval()
    say(f"large: transformers {transformers.__version__}, torch {torch.__version__}, loaded as float32 in "
        f"{time.perf_counter() - began:.0f} s")
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), BOS, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for at, row in enumerate(rows):
        ids[at, :len(row)] = torch.tensor(row)
        mask[at, :len(row)] = 1
    began = time.perf_counter()
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    say(f"large: transformers, {len(rows)} rows of up to {width} positions at once in {time.perf_counter() - began:.0f} s")
    np.save(directory / "theirs.npy", logits[0, :len(rows[0])].float().numpy())
    total, count, first = 0.0, 0, []
    for at, row in enumerate(rows[1:], 1):
        logs = torch.log_softmax(logits[at, :len(row) - 1].double(), dim=-1)
        each = -logs.gather(1, torch.tensor(row[1:])[:, None])[:, 0]
        if at == 1:
            first = each.tolist()  # the first window, token by token: large() holds the engine's to it
        total += float(each.sum())
        count += len(row) - 1
    (directory / "theirs.json").write_text(json.dumps({"total": total, "count": count, "first": first}))


def tensors_apart(ours, other, options):
    """Two float32 checkpoints of one header, tensor by tensor of llama2_convert.layout(): [(index, shape, relative
    difference, the same to the bit)], read a piece at a time."""
    from llama2_numpy import form_of

    a, b = (np.memmap(path, dtype=np.uint8, mode="r") for path in (ours, other))
    header = struct.unpack("<7i", bytes(a[:28]))
    failed = bytes(a[:28]) != bytes(b[:28]) or len(a) != len(b)
    out, offset = [], 28
    for index, (shape, _) in enumerate(llama2_convert.layout(*header, **form_of(options))):
        size = 4 * math.prod(shape)
        difference = norm = 0.0
        same = True
        for start in range(offset, offset + size, 1 << 28):
            stop = min(start + (1 << 28), offset + size)
            x, y = (np.frombuffer(data[start:stop], dtype=np.float32) for data in (a, b))
            same &= bool(np.array_equal(x, y))
            wide = y.astype(np.float64)
            difference += float(((x - wide) ** 2).sum())
            norm += float((wide ** 2).sum())
        out.append((index, shape, math.sqrt(difference / max(norm, 1e-300)), same))
        offset += size
    return out, failed or offset != len(a)


def large(directory, positions, name, source, text_file, minutes, first=0):
    """A real model whose float32 is more than the runner's memory (T245: the 4B, 16.8 GB). What the 0.8B's part does
    not do here: transformers token by token and its generate() (each a pass over all the weights, through the swap),
    so the line is the 0.8B's 2e-2 with no floor measured, and the greedy text is left to tests/fixed_outputs.py's
    smaller models. The engine's checkpoint is a memory map with most of it pinned (pinned()), and stops after
    `minutes` if it has 64 positions by then."""
    import tokenizers

    model = LARGE[name]
    repo, revision = model["repo"], model["revision"]
    say(f"large: {repo}@{revision}, the engine on {'the GGUF ' + '@'.join(model['gguf'][:2]) if source == 'gguf' else 'its safetensors'}")
    for file in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        fetch(file, directory, repo, revision)
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        fetch(shard, directory, repo, revision)
    options = prepared(directory, directory / "float32")
    if source == "gguf":
        # the list's way in: the GGUF's weights with the original's config.json and vocabulary (a directory of its own
        # for perplexity_prepare.py, which takes a GGUF where it finds one)
        folder = directory / "gguf"
        folder.mkdir(exist_ok=True)
        for file in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if not (folder / file).exists():
                (folder / file).symlink_to((directory / file).resolve())
        fetch(model["gguf"][2], folder, model["gguf"][0], model["gguf"][1])
        from_gguf = prepared(folder, directory / "gguf-float32")
        failed = from_gguf != options
        say(f"large: the options of the GGUF's conversion are {'the same' if not failed else 'OTHER OPTIONS — FAILED'}")
        apart, wrong = tensors_apart(directory / "gguf-float32.bin", directory / "float32.bin", options)
        failed |= wrong
        # the line: Q8_0's rounding is 6e-3 of a matrix (T236: 5.7e-3 to 6.7e-3); value heads at other places are 1.4
        worst = max(apart, key=lambda entry: entry[2])
        for index, shape, relative, same in apart:
            say(f"large: tensor {index} {list(shape)}: {'the same to the bit' if same else f'{relative:.2e} apart'}"
                f"{' — FAILED' if relative > 2e-2 else ''}")
        failed |= worst[2] > 2e-2
        say(f"large: the float32 checkpoint from the GGUF against the one from the safetensors: {sum(e[3] for e in apart)} "
            f"of {len(apart)} tensors the same to the bit, the furthest {worst[2]:.2e} apart (tensor {worst[0]}; the line: "
            f"2e-2){' — FAILED' if failed else ''}")
        (directory / "float32.bin").unlink()
        checkpoint = directory / "gguf-float32"
    else:
        failed, checkpoint = False, directory / "float32"

    data = np.memmap(f"{checkpoint}.bin", dtype=np.uint8, mode="r")
    llama = Llama(data, Path(f"{checkpoint}.tokenizer.bin").read_bytes(), kernels=None,
                  **{**options, "specials": SPECIALS, "stop_tokens": [248044, 248046]})
    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
    ids = [BOS] + tokenizer.encode(TEXT, add_special_tokens=False).ids
    same = [BOS] + llama.tokenizer.encode(TEXT, llama.specials) == ids
    rows = [ids[:positions]]
    if text_file:
        text = Path(text_file).read_text()
        tokens = tokenizer.encode(text, add_special_tokens=False).ids[:TOKENS]
        same &= llama.tokenizer.encode(text)[:TOKENS] == tokens
        rows += [[llama.bos] + tokens[start:start + WINDOW - 1] for start in range(0, len(tokens), WINDOW - 1)]
    failed |= not same
    say(f"large: {len(ids)} tokens of text{f' and {len(tokens)} of {Path(text_file).name}' if text_file else ''}, the "
        f"engine's tokenizer gives {'the same ids' if same else 'OTHER IDS — FAILED'}")
    ids = ids[:positions]
    (directory / "ids.json").write_text(json.dumps(rows))
    subprocess.run([sys.executable, __file__, str(directory), "--only=theirs"], check=True)
    whole = np.load(directory / "theirs.npy")
    if text_file:
        nll = json.loads((directory / "theirs.json").read_text())
        say(f"large: perplexity of the float32 original, transformers: {math.exp(nll['total'] / nll['count']):.3f} "
            f"({nll['count']} tokens of {Path(text_file).name}, windows of {WINDOW} from the page's BOS)")

    began = time.perf_counter()
    held = pinned(data)
    say(f"large: {held} of the checkpoint's {len(data)} bytes pinned in {time.perf_counter() - began:.0f} s")
    began, ours = time.perf_counter(), []
    for pos, token in enumerate(ids):
        ours.append(llama.forward(token, pos).copy())
        elapsed = time.perf_counter() - began
        if pos % 8 == 7:
            say(f"large: the engine at position {pos + 1}, {elapsed:.0f} s")
        if elapsed > 60 * minutes and len(ours) >= 64:
            break
    say(f"large: the engine (NumPy, float32), {len(ours)} positions in {time.perf_counter() - began:.0f} s")
    if len(ours) < min(64, len(ids)):
        failed = True
        say(f"large: fewer than 64 positions — FAILED")
    whole = whole[:len(ours)]
    for what, logits in (("transformers", whole), ("the engine", ours)):
        logits = np.asarray(logits)
        say(f"large: logits of {what}: min {logits.min():.3f}, max {logits.max():.3f}, mean {logits.mean():.4f}, "
            f"std {logits.std():.4f}")
    largest, mean, agree, margin = differences(ours, whole)
    if source == "gguf":
        # the GGUF's values are Q8_0's rounding of the original's: no logits within a line of float32's, but the most
        # likely token of the original at 85% of the positions (forward-check's line for int8 against float32)
        ok = agree >= 0.85 * len(ours)
        line = "the same most likely token at 85% of the positions"
    else:
        ok = largest <= 2e-2 and (agree == len(ours) or margin <= 2 * largest)
        line = "2e-2"
    failed |= not ok
    say(f"large: the engine on {'the GGUF' if source == 'gguf' else 'the safetensors'} against transformers at once: "
        f"largest difference {largest:.2e}, mean {mean:.2e}, the same most likely token at {agree} of {len(ours)} "
        f"positions (the largest gap between the reference's best two where it is not: {margin:.2e})"
        f"{'' if ok else f' — FAILED (the line: {line})'}")
    if text_file and first:
        # the same on the first tokens of the perplexity's text: the negative log likelihood of each, as
        # tests/perplexity.py counts it (float64, from the page's BOS)
        row, began, mine = rows[1][:first + 1], time.perf_counter(), []
        for pos in range(len(row) - 1):
            logits = np.asarray(llama.forward(row[pos], pos), dtype=np.float64)
            logits -= logits.max()
            mine.append(-(logits[row[pos + 1]] - math.log(np.exp(logits).sum())))
        reference = nll["first"][:len(mine)]
        furthest = max(abs(a - b) for a, b in zip(mine, reference))
        ours_value, theirs_value = (math.exp(sum(values) / len(values)) for values in (mine, reference))
        say(f"large: perplexity of the first {len(mine)} tokens of {Path(text_file).name}: transformers on the float32 "
            f"original {theirs_value:.3f}, the engine (NumPy, float32) on {'the GGUF' if source == 'gguf' else 'the safetensors'} "
            f"{ours_value:.3f} ({(ours_value / theirs_value - 1) * 100:+.2f}%; a token's negative log likelihood at most "
            f"{furthest:.2e} apart; {time.perf_counter() - began:.0f} s)")
        if source != "gguf" and abs(ours_value / theirs_value - 1) > 1e-3:
            failed = True
            say("large: the engine's perplexity on the original's weights is not transformers' — FAILED")
    return failed


def main():
    directory = Path(sys.argv[1])
    only = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--only=")), None)
    positions = int(next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--positions=")), 96))
    option = lambda name, default=None: next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith(f"--{name}=")), default)
    failed = False
    if only == "theirs":
        theirs(directory, directory / "ids.json")
        return
    if option("model") in OTHERS:
        failed = real(directory / option("model"), positions, *OTHERS[option("model")])
        say("FAILED" if failed else "the engine computes what transformers computes")
        sys.exit(1 if failed else 0)
    if option("model") in LARGE:
        failed = large(directory, positions, option("model"), option("from", "safetensors"), option("text"),
                       float(option("minutes", 40)), int(option("first", 0)))
        say("FAILED" if failed else "the engine computes what transformers computes")
        sys.exit(1 if failed else 0)
    if only == "fetch":
        for name in FILES:
            fetch(name, directory)
        return
    if only in (None, "made-up"):
        for name, settings in MADE_UP.items():
            failed |= made_up(name, settings)
    if only in (None, "real"):
        failed |= real(directory, positions)
    say("FAILED" if failed else "the engine computes what transformers computes")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
