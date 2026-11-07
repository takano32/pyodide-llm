# reference_gemma2.py
# T612: the engine's Gemma 2 (Google's: a Gemma 3's layers without the norms of the heads of q and k, the attention's
# scores and the logits capped by a tanh, one local layer of every two) against transformers itself. PyTorch is not on
# the development machine, so this runs in CI:
#
#   python tests/reference_gemma2.py <directory for the download> [--only=made-up|real|fetch] [--model=2B] [--positions=96]
#   (--only=fetch: the real model's files into the directory and no more)
#
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_gemma2.sh" --ref <branch> --grep "gemma2:"
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_gemma2.sh --only=real" --ref <branch> --grep "gemma2:"
#
# Three parts, every line of the log beginning with "gemma2:".
#   made-up: tiny random models of transformers' own class (Gemma2ForCausalLM, eager attention: the one that caps the
#     scores) with a window of 8 positions, which the 80 positions of the text are ten times, small caps (2 of the scores,
#     3 of the logits: the tanh bites on these numbers as the real 50 and 30 do on the real ones), and what the real 2B
#     has not: a classifier of its own, query_pre_attn_scalar other than the head's size (the 27B's), keys for every head,
#     heads that fill the width. transformers' logits, over the whole text at once and token by token with its cache,
#     against the naive reference of the unit tests (conftest.naive_gemma3_logits, which reads a Gemma 2) and against the
#     engine, converted the way the page converts. And what each holds after the text: the keys and values of every
#     layer, of a sliding layer the last window - 1.
#   faults: the first made-up model's engine with one thing wrong at a time (FAULTS: no cap of the scores, no cap of the
#     logits, q not multiplied, the kinds of the layers shifted by one, ...), each against the same numbers of
#     transformers. Every fault has to be past the line the right engine is held to: the weakest is named.
#   real: a published model at a fixed revision (MODELS: unsloth's copy of gemma-2-2b-it; Google's are gated), float32.
#     The same token ids through transformers and through the engine (NumPy): the largest difference of the logits, how
#     often the most likely token is the same, the keys and values after them, and 16 greedy tokens of each for the chat
#     prompt. Its window (4096) is the page's whole context: no long text here. Its float32 is 10.5 GB on disk and in
#     transformers' memory: a runner of 16 GB, its /mnt for the files (reference_gemma2.sh).
# The lines it holds the engine to are at the end of each part; anything past them is exit 1.
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from llama2_numpy import Llama, silu  # noqa: E402
from conducting import Directory  # noqa: E402
from reference import (TEXT, both_ways, differences, ended, engine_logits, fetch, float32_file,  # noqa: E402
                       line_of, logits_failed, logits_said, made_up_engine, option, options_shown)
# (a Gemma 2's cache is a Gemma 3's: the keys and values of every layer, of a sliding one its last window - 1)
from reference_gemma3 import HELD, engine_states, model_states  # noqa: E402, F401

# the published models: (repository, revision). Google's own are gated; unsloth's copy of the 2B it is not
MODELS = {"2B": ("unsloth/gemma-2-2b-it", "457f2e15bf550c227ce6ad86e2ec108d3e42c106")}
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors"]
BOS = 2  # <bos>: what the real tokenizer puts first
# the chat format of one turn, as the model's template writes it after its bos_token
CHAT = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
BEGINS = "<bos>"
PROMPT = "What is the capital of Japan? Answer in one sentence."
SPECIALS = ["<start_of_turn>", "<end_of_turn>"]
STOPS = (1, 107)  # <eos> and <end_of_turn>: generation_config.json's eos_token_id
NEW_TOKENS = 16
LOGITS_LINE = 1e-3  # the least of the logits' line (the made-up models': 2e-3)

say = lambda *parts: print("gemma2:", *parts, flush=True)


# --------------------------------------------------------------------------------------------- made-up models
# s: a sliding-attention layer, l: one that attends every position (a Gemma 2's own: one of every two from the first)
MADE_UP = {
    "the 2B's order in small, the scores by another number": dict(kinds="slslslsl", query_pre_attn_scalar=40),
    "a classifier of its own, a window of 5": dict(kinds="slslsl", tie_word_embeddings=False, sliding_window=5),
    "the kinds unsaid, keys for every head": dict(layers=6, num_key_value_heads=4),
    "heads that fill the width, another epsilon": dict(kinds="slsl", head_dim=8, rms_norm_eps=1e-5),
    "the real caps (50 and 30)": dict(kinds="slsl", attn_logit_softcapping=50.0, final_logit_softcapping=30.0),
}


def made_up_model(settings):
    """transformers' Gemma2ForCausalLM of these settings, its parameters drawn (a norm's weight around zero), and its
    config.json as published"""
    import torch
    from transformers import Gemma2Config, Gemma2ForCausalLM

    settings = dict(settings)
    kinds = settings.pop("kinds", None)
    layers = settings.pop("layers", len(kinds or ""))
    said = dict(num_key_value_heads=2, head_dim=16, sliding_window=8, rms_norm_eps=1e-6, attn_logit_softcapping=2.0,
                final_logit_softcapping=3.0)
    said.update(settings)
    said.setdefault("query_pre_attn_scalar", said["head_dim"])
    if kinds:
        said["layer_types"] = ["sliding_attention" if kind == "s" else "full_attention" for kind in kinds]
    config = Gemma2Config(vocab_size=320, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                          num_attention_heads=4, max_position_embeddings=256, bos_token_id=2, eos_token_id=1,
                          pad_token_id=0, rope_theta=10000.0, hidden_activation="gelu_pytorch_tanh",
                          attn_implementation="eager", **said)
    torch.manual_seed(612)
    model = Gemma2ForCausalLM(config).to(torch.float32).eval()
    tied = bool(config.tie_word_embeddings)
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if not (key == "lm_head.weight" and tied):
                parameter.copy_(0.3 * torch.randn_like(parameter))
    published = json.loads(config.to_json_string(use_diff=False))
    if not kinds:
        published.pop("layer_types", None)  # (a config.json of 2024 says none: one of every two from the first)
    return model, config, published, tied


def made_up(name, settings, positions=80, with_faults=False):
    from conftest import naive_gemma3_logits

    model, config, published, tied = made_up_model(settings)
    rng = np.random.default_rng(612)
    tokens = [int(token) for token in rng.integers(0, 320, positions)]
    whole, at_once, stepped, stepped_states = both_ways(model, tokens, model_states)
    tensors = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    if tied:
        tensors.pop("lm_head.weight", None)
    kinds = list(config.layer_types)
    naive = naive_gemma3_logits(tensors, {**published, "layer_types": kinds, "rope_theta": 10000.0,
                                          "rope_local_base_freq": 10000.0}, tokens)

    llama, options, checkpoint = made_up_engine(tensors, published)
    ours = engine_logits(llama, tokens)

    failed = False
    floor = differences(stepped, whole)[0]
    line = line_of(floor, 2 * LOGITS_LINE)
    say(f"made-up ({name}), transformers token by token against transformers at once: largest difference {floor:.2e}")
    for what, a, b in (("the naive reference against transformers at once", naive, whole),
                       ("the engine against transformers at once", ours, whole),
                       ("the engine against transformers token by token", ours, stepped)):
        failed |= logits_failed(say, f"made-up ({name}), {what}", a, b, line)
    failed |= HELD.failed(say, f"made-up ({name})", llama, positions, at_once, stepped_states)
    sliding = [layer for layer, kind in enumerate(kinds) if kind == "sliding_attention"]
    theirs = {"layers": sliding, "window": config.sliding_window, "rope_theta": 10000.0} if sliding else None
    caps = (options.get("score_softcap"), options.get("softcap"))
    right = options.get("local") == theirs and caps == (config.attn_logit_softcapping, config.final_logit_softcapping)
    failed |= not right
    say(f"made-up ({name}): the local layers {json.dumps(options.get('local'))}, the caps of the scores and of the logits "
        f"{caps}{'' if right else f' — FAILED (transformers: {json.dumps(theirs)}, {config.attn_logit_softcapping}, {config.final_logit_softcapping})'}")
    if with_faults:
        from conftest import pack_tokenizer, tiny_vocab
        vocabulary = pack_tokenizer(tiny_vocab(320))  # (the forward pass reads none of it)
        make = lambda **other: Llama(checkpoint, vocabulary, **{**options, **other})
        failed |= faults(make, tokens, whole, stepped_states, line, HELD.distances(at_once, stepped_states))
    return failed


# ------------------------------------------------------------------------------------------------------- faults
# The engine with one thing wrong, each a function that takes a new engine of the made-up model (and what makes one)
# and returns the engine to run. What the right engine is held to must let none of them by.
def setting(name, value):
    def fault(llama, make):
        setattr(llama, name, value(llama))
        return llama
    return fault


def kinds_shifted(llama, make):
    """the local layers one later: the second, then every other (llama.cpp's set_swa_pattern counted the other way)"""
    layers = [layer for layer in range(llama.n_layers) if layer not in llama.local["layers"]]
    return make(local={**llama.local, "layers": layers})


def window(change):
    def fault(llama, make):
        llama.window_kept = llama.local["window"]
        llama.local = {**llama.local, "window": change(llama.local["window"])}
        llama.layers = [{**facts, "window": llama.local["window"]} if facts["local"] else facts for facts in llama.layers]
        return llama
    return fault


def gate(function):
    def fault(llama, make):
        llama.activation = function
        return llama
    return fault


FAULTS = {
    "no cap of the scores": setting("score_softcap", lambda llama: 0.0),
    "no cap of the logits": setting("softcap", lambda llama: 0.0),
    "the two caps in each other's place": lambda llama, make: make(score_softcap=llama.softcap, softcap=llama.score_softcap),
    "the cap of the scores twice as high": setting("score_softcap", lambda llama: 2 * llama.score_softcap),
    "q not multiplied: the scores by the root of the head's size": setting(
        "wq", lambda llama: llama.wq / np.float32(math.sqrt(llama.head_size / 40))),
    "the kinds of the layers shifted by one": kinds_shifted,
    "every layer attends all the positions": lambda llama, make: make(local=None),
    "the window a position too long": window(lambda size: size + 1),
    "the window a position too short": window(lambda size: size - 1),
    "no norm behind the attention": setting("rms_att_post_weight", lambda llama: None),
    "the 1 of the norm in front of the FFN not added": setting("rms_ffn_weight", lambda llama: llama.rms_ffn_weight - np.float32(1.0)),
    "the embedding not multiplied by the root of the width": lambda llama, make: _embedding_times(llama, 1.0 / math.sqrt(llama.dim)),
    "a SiLU in the FFN's gate (a Llama's)": gate(silu),
}


def _embedding_times(llama, factor):
    right = llama.embedding
    llama.embedding = lambda token: right(token) * np.float32(factor)
    return llama


def faults(make, ids, whole, states, line, state_floor):
    """Every fault's distance from transformers' numbers, and whether what the right engine is held to (the logits
    within line, the states within theirs) catches it. True where one gets by."""
    lines = HELD.lines(state_floor)
    slipped, weakest = [], None
    for name, fault in FAULTS.items():
        llama = fault(make(), make)
        ours = engine_logits(llama, ids)
        largest, _, agree, _ = differences(ours, whole)
        apart = HELD.apart(llama, len(ids), states)
        by_logits = largest / line if np.isfinite(largest) else float("inf")
        by_states = max((apart[state] / lines[state] if np.isfinite(apart[state]) else float("inf")) for state in HELD.names)
        caught = not largest <= line or bool(HELD.past(apart, lines))
        margin = max(by_logits, by_states)
        if not caught:
            slipped.append(name)
        if weakest is None or margin < weakest[1]:
            weakest = (name, margin)
        say(f"fault ({name}): logits {largest:.2e} from transformers' ({by_logits:.1f} times the line), the same most likely token "
            f"at {agree} of {len(ours)}; states {by_states:.1f} times their line{'' if caught else ' — NOT CAUGHT, FAILED'}")
    say(f"faults: {len(FAULTS) - len(slipped)} of {len(FAULTS)} are past the line (logits {line:.1e}); the weakest is "
        f"\"{weakest[0]}\", {weakest[1]:.1f} times past it{'' if not slipped else ' — FAILED: ' + '; '.join(slipped)}")
    return bool(slipped)


# ------------------------------------------------------------------------------------------------- the real model
def real(directory, positions, name):
    import tokenizers
    import torch
    import transformers
    from transformers import Gemma2ForCausalLM

    repo, revision = MODELS[name]
    say(f"real: {repo}@{revision}, transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}")
    for file in FILES:
        fetch(file, directory, repo, revision)
    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
    path = directory / "float32.bin"
    conversion, seconds = float32_file(Directory(directory), {"weights": "model.safetensors", "tokenizer": "tokenizer.json"}, path)
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    say(f"real: converted in {seconds:.1f} s, {path.stat().st_size} bytes, options {options_shown(options)}")
    specials = sorted(set(options.get("specials", [])) | set(SPECIALS), key=lambda token: (-len(token), token))
    checkpoint = np.memmap(path, dtype=np.uint8, mode="r")
    llama = Llama(checkpoint, conversion.tokenizer, kernels=None, **{**options, "specials": specials})
    failed = llama.bos != BOS or not set(llama.stop_tokens) & set(STOPS)
    say(f"real: the BOS is {llama.bos} and the answer stops at {sorted(llama.stop_tokens)} (transformers' generate() at "
        f"{list(STOPS)}){' — FAILED' if failed else ''}; the caps {llama.score_softcap} and {llama.softcap}")

    with_bos = tokenizer.encode(TEXT).ids
    ids = [BOS] + tokenizer.encode(TEXT, add_special_tokens=False).ids
    mine = [BOS] + llama.tokenizer.encode(TEXT, llama.specials)
    chat = CHAT.format(prompt=PROMPT)
    chat_ids = tokenizer.encode(chat, add_special_tokens=False).ids
    same = mine == ids and llama.tokenizer.encode(chat, llama.specials) == chat_ids and with_bos == ids
    failed |= not same
    say(f"real: {len(ids)} tokens of text and {len(chat_ids)} of the chat prompt, the engine's tokenizer gives "
        f"{'the same ids' if same else 'OTHER IDS — FAILED'}; the real tokenizer begins a text with {with_bos[0]}")
    template = transformers.AutoTokenizer.from_pretrained(str(directory)).apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    failed |= template != BEGINS + chat
    say(f"real: the model's own template writes {json.dumps(template)}: "
        f"{'the format here after its BOS' if template == BEGINS + chat else 'ANOTHER FORMAT — FAILED'}")
    ids = ids[:positions]

    model = Gemma2ForCausalLM.from_pretrained(str(directory), dtype=torch.float32, attn_implementation="eager").eval()
    began = time.perf_counter()
    whole, at_once, stepped, stepped_states = both_ways(model, ids, model_states)
    say(f"real: transformers, {len(ids)} positions at once and token by token in {time.perf_counter() - began:.1f} s")
    with torch.no_grad():
        generated = model.generate(torch.tensor([[BOS] + chat_ids]), max_new_tokens=NEW_TOKENS, do_sample=False,
                                   repetition_penalty=1.0)[0].tolist()
    theirs = generated[1 + len(chat_ids):]
    del model
    gc.collect()

    began = time.perf_counter()
    ours = engine_logits(llama, ids)
    say(f"real: the engine (NumPy, float32), {len(ids)} positions in {time.perf_counter() - began:.1f} s")
    logits_said(say, "real", whole, ours)
    floor = differences(stepped, whole)[0]
    say(f"real: transformers token by token against transformers at once: largest difference {floor:.2e}")
    line = line_of(floor, LOGITS_LINE)
    for what, reference in (("transformers at once", whole), ("transformers token by token", stepped)):
        failed |= logits_failed(say, f"real: the engine against {what}", ours, reference, line, gap=True)
    failed |= HELD.failed(say, "real", llama, len(ids), at_once, stepped_states)
    text = "".join(llama.generate(chat, steps=len(chat_ids) + NEW_TOKENS, temperature=0.0, echo=False))
    stop = next((at for at, token in enumerate(theirs) if token in llama.stop_tokens), len(theirs))
    same = tokenizer.decode(theirs[:stop], skip_special_tokens=False) == text
    failed |= not same
    say(f"real: for the chat prompt transformers wrote {json.dumps(tokenizer.decode(theirs, skip_special_tokens=False), ensure_ascii=False)} {theirs}")
    say(f"real: for the chat prompt the engine wrote {json.dumps(text, ensure_ascii=False)}: {'the two wrote the same' if same else 'OTHER TEXTS — FAILED'}")
    del llama, checkpoint
    path.unlink()
    return failed


def main():
    directory = Path(sys.argv[1])
    only, names, positions = option("only"), option("model", "2B").split(","), int(option("positions", 96))
    if only == "fetch":
        for name in names:
            for file in FILES:
                fetch(file, directory / name, *MODELS[name])
        return
    failed = False
    if only in (None, "made-up"):
        # the faults on the first made-up model
        for at, (label, settings) in enumerate(MADE_UP.items()):
            failed |= made_up(label, settings, with_faults=at == 0 and "--no-faults" not in sys.argv)
    if only == "real":
        # (not by default: the 2B's 10.5 GB of float32, twice; reference_gemma2.sh --only=real)
        for name in names:
            failed |= real(directory / name, positions, name)
    ended(say, failed)


if __name__ == "__main__":
    main()
