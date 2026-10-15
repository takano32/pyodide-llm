# reference_spark.py
# T342: the engine's Spark-X2.5 (XHToken's: three local layers of every four that turn their whole heads by a base of
# their own, the others a quarter of each head; a gate of each head's output; the exact GELU in the FFN's gate) against
# transformers running the model's own code (modeling_spark.py: transformers has no class of it). PyTorch is not on the
# development machine, so this runs in CI:
#
#   python tests/reference_spark.py <directory for the download> [--only=made-up|real|fetch] [--model=1.7B[,4B]] [--positions=96]
#   (--only=fetch: the real model's files into the directory and no more)
#
#   node tests/ci.mjs run tests.yml only_extra=true extra="bash tests/reference_spark.sh" --ref <branch> --grep "spark:"
#
# Three parts, every line of the log beginning with "spark:".
#   made-up: tiny random models of the model's own class (Spark2_5ForCausalLM, from its modeling_spark.py at the fixed
#     revision) with a window of 8 positions, which the 80 positions of the text are ten times, and with what the real
#     1.7B has not: a classifier of its own, keys for every head, heads that do not fill the width, half of a head turned.
#     Its logits, over the whole text at once (its masks) and token by token with its cache (which keeps the last
#     window - 1 keys of a sliding layer), against the naive reference of the unit tests (test_spark.naive_spark_logits)
#     and against the engine, converted the way the page converts (llama2_convert.Conversion). And what each holds after
#     the text: the keys and values of every layer, of a sliding layer the last window - 1.
#   real: a published model at a fixed revision (MODELS; the 1.7B unless --model= says another), float32. The same
#     token ids through the model's code and through the engine (NumPy): the largest difference of the logits, how
#     often the most likely token is the same, the keys and values after them, 16 greedy tokens of each, and then 1024
#     positions of a longer text: twice the window of 512, which the 96 positions before never reach.
#   faults: the real model's engine with one thing wrong at a time (FAULTS), each against the same numbers of the
#     model's code. Those of the window show past 512 positions alone and are run on the long text. Every fault has to
#     be past the line the right engine is held to, or the line would let it through: the weakest is named.
# The lines it holds the engine to are at the end of each part; anything past them is exit 1.
import gc
import importlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from llama2_numpy import Llama, gelu, rope_frequencies, silu  # noqa: E402
from conducting import Directory  # noqa: E402
from reference import (TEXT, States, both_ways, differences, ended, engine_logits, fetch, float32_file,  # noqa: E402
                       kept_logits, line_of, logits_failed, logits_said, long_ids, made_up_engine, option, options_shown,
                       within)

# the published models: (repository, revision, the shards of its safetensors)
MODELS = {"1.7B": ("XHToken/Spark-X2.5-1.7B", "14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8", 2),
          "4B": ("XHToken/Spark-X2.5-4B", "0bcb35678590218655dff3765b9e61c83b35e9c4", None)}
CODE = ["configuration_spark.py", "modeling_spark.py"]
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", *CODE]
BOS = 0  # <｜start▁of▁sentence｜>: what the model's template writes first
# the special tokens of its chat format, which the engine is handed as the page is (whatever the converter lists)
SPECIALS = ["<｜start▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<|System|>", "<|User|>", "<|Bot|>", "<think>", "</think>"]
PROMPT = "What is the capital of Japan? Answer in one sentence."
NEW_TOKENS = 16

say = lambda *parts: print("spark:", *parts, flush=True)
LOGITS_LINE = 1e-3  # the least of the logits' line, which is ten times that floor too (the made-up models': 2e-3)


def code(directory):
    """The model's own code (its configuration_spark.py and modeling_spark.py, fetched with it), as a package of the
    directory's: (Spark2_5Config, Spark2_5ForCausalLM). transformers has no class of this model; this is what
    trust_remote_code would run, read without the question."""
    package = Path(directory) / "spark_code"
    package.mkdir(parents=True, exist_ok=True)
    for name in CODE:
        shutil.copyfile(Path(directory) / name, package / name)
    (package / "__init__.py").write_text("")
    sys.path.insert(0, str(Path(directory)))
    importlib.invalidate_caches()
    modeling = importlib.import_module("spark_code.modeling_spark")
    return modeling.Spark2_5Config, modeling.Spark2_5ForCausalLM


def code_directory(directory):
    """where the model's code is fetched for the made-up models: the 1.7B's, at its revision"""
    repo, revision, _ = MODELS["1.7B"]
    for name in CODE:
        fetch(name, Path(directory), repo, revision)
    return Path(directory)


# What the engine and the model's code hold after the same positions: the keys (as Hugging Face has them: the pairs of
# the part of a head RoPE turns not interleaved) and values of every layer. The model's cache keeps the last window - 1
# positions of a sliding-attention layer and all of the others; the engine keeps them all (or its window's ring), and its
# last window - 1 are what is compared for a local layer.
def hf_order(size, turned):
    """the engine's index of Hugging Face's 0..size-1, where the first `turned` values of a head are interleaved"""
    return np.concatenate([np.arange(0, turned, 2), np.arange(1, turned, 2), np.arange(turned, size)])


def engine_states(llama, count):
    """{layer: (kind, [keys, values])}"""
    out, head = {}, llama.head_size
    local = set(llama.local["layers"]) if llama.local else ()
    # (the window the model's cache goes by: the model's, whatever a fault made of the engine's)
    size = getattr(llama, "window_kept", llama.local["window"] if llama.local else 0)
    for l, (_, a) in enumerate(llama.slots):
        first = max(0, count - (size - 1)) if l in local else 0
        order = hf_order(head, head if l in local else llama.rotary)
        out[l] = ("sliding" if l in local else "full",
                  [llama.key_cache[a, :, first:count][..., order], llama.value_cache[a, :, first:count]])
    return out


def model_states(past, count):
    out = {}
    for l, layer in enumerate(past.layers):
        kind = "sliding" if getattr(layer, "is_sliding", False) else "full"
        out[l] = (kind, [layer.keys[0].float().numpy()[:, :count], layer.values[0].float().numpy()[:, :count]])
    return out


HELD = States({"sliding": ("keys of the sliding layers", "values of the sliding layers"), "full": ("keys", "values")},
              engine_states, model_states)


# --------------------------------------------------------------------------------------------- made-up models
# s: a sliding-attention layer, l: one that attends every position
MADE_UP = {
    "the 1.7B's order in small": dict(kinds="ssslssslsssl"),
    "a classifier of its own, a window of 5": dict(kinds="sslssl", tie_word_embeddings=False, sliding_window=5),
    "keys for every head, heads wider than the stream": dict(kinds="sssl", num_key_value_heads=4, head_dim=16),
    "half of a head turned, another epsilon, other bases": dict(kinds="lsslss", part=0.5, rms_norm_eps=1e-5, theta=5000.0,
                                                               local_theta=300.0),
    "no sliding layer at all": dict(kinds="llll"),
}


def made_up(name, settings, positions=80, directory=None):
    import torch
    from test_spark import naive_spark_logits

    Config, ForCausalLM = code(code_directory(directory or HERE.parent / ".tmp" / "reference-spark"))
    settings = dict(settings)
    kinds = settings.pop("kinds")
    theta, local_theta, part = settings.pop("theta", 5000000.0), settings.pop("local_theta", 10000.0), settings.pop("part", 0.25)
    said = dict(num_key_value_heads=2, head_dim=8, sliding_window=8, rms_norm_eps=1e-6, tie_word_embeddings=True)
    said.update(settings)
    config = Config(vocab_size=320, hidden_size=32, intermediate_size=64, num_hidden_layers=len(kinds), num_attention_heads=4,
                    max_position_embeddings=256, bos_token_id=0, eos_token_id=1, pad_token_id=2, hidden_act="gelu",
                    headwise_attn_output_gate=True, gate_attn_act_mode="sigmoid",
                    layer_types=["sliding_attention" if kind == "s" else "full_attention" for kind in kinds],
                    rope_parameters={"full_attention": {"partial_rotary_factor": part, "rope_theta": theta},
                                     "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": local_theta}},
                    **said)
    torch.manual_seed(342)
    model = ForCausalLM(config).to(torch.float32).eval()
    tied = bool(config.tie_word_embeddings)
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key == "lm_head.weight" and tied:
                continue
            # (a norm's weight is the weight itself: around one)
            parameter.copy_((1.0 if key.endswith("norm.weight") else 0.0) + 0.3 * torch.randn_like(parameter))
    rng = np.random.default_rng(342)
    tokens = [int(token) for token in rng.integers(0, 320, positions)]
    whole, at_once, stepped, stepped_states = both_ways(model, tokens, model_states)
    tensors = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    if tied:
        tensors.pop("lm_head.weight", None)
    published = json.loads(config.to_json_string(use_diff=False))
    naive = naive_spark_logits(tensors, published, tokens)

    llama, options, _ = made_up_engine(tensors, published)
    ours = engine_logits(llama, tokens)

    failed = False
    floor = differences(stepped, whole)[0]
    line = line_of(floor, 2 * LOGITS_LINE)
    say(f"made-up ({name}), its code token by token against its code at once: largest difference {floor:.2e}")
    pairs = [("the naive reference against its code at once", naive, whole),
             ("the engine against its code at once", ours, whole),
             ("the engine against its code token by token", ours, stepped)]
    for what, a, b in pairs:
        failed |= logits_failed(say, f"made-up ({name}), {what}", a, b, line)
    failed |= HELD.failed(say, f"made-up ({name})", llama, positions, at_once, stepped_states)
    sliding = [layer for layer, kind in enumerate(published["layer_types"]) if kind == "sliding_attention"]
    theirs = {"layers": sliding, "window": said["sliding_window"], "rope_theta": local_theta} if sliding else None
    rotary = int(said["head_dim"] * part)
    right = options.get("local") == theirs and options.get("rotary") == rotary
    failed |= not right
    say(f"made-up ({name}): the local layers {json.dumps(options.get('local'))}, RoPE turns {options.get('rotary')} of a head "
        f"of the others{'' if right else f' — FAILED (its config: {json.dumps(theirs)}, {rotary})'}, options "
        f"{json.dumps({key: options[key] for key in ('arch', 'head_dim', 'rope_theta', 'rms_norm_eps') if key in options})}")
    return failed


# ------------------------------------------------------------------------------------------------- the real model
# the positions of the long text (reference.long_ids) whose logits are compared: around the window of 512, where a
# sliding layer first leaves a position out, and to the end
LONG_AT = (0, 1, 255, 256, 257, 510, 511, 512, 513, 514, 600, 767, 768, 1022, 1023)


def long_run(llama, ids, at):
    """the engine through ids: its logits at the positions `at`"""
    kept = {}
    for pos, token in enumerate(ids):
        logits = llama.forward(token, pos, need_logits=pos in at)
        if pos in at:
            kept[pos] = logits.copy()
    return np.stack([kept[position] for position in at])


def long_failed(llama, ids, at, theirs, their_states, line):
    """The engine through the long text, its logits at `at` against the model's code (within line) and its keys and
    values at the end against the code's (within the least of States). True where either is not."""
    began = time.perf_counter()
    largest, mean, same, margin = differences(long_run(llama, ids, at), theirs)
    apart = HELD.apart(llama, len(ids), their_states)
    ok = within(largest, same, len(at), margin, line)
    bad = HELD.past(apart, dict.fromkeys(HELD.names, HELD.least))
    say(f"long: the engine ran {len(ids)} positions in {time.perf_counter() - began:.1f} s; the logits at {len(at)} of them against "
        f"its code's: largest difference {largest:.2e}, mean {mean:.2e}, the same most likely token at {same} of {len(at)}"
        f"{'' if ok else f' — FAILED (the line: {line:.1e})'}; the states after them: "
        + "; ".join(f"{name} {apart[name]:.1e}" for name in HELD.names)
        + (f" — FAILED (the line: {HELD.least:.0e}): {', '.join(bad)}" if bad else ""))
    return not ok or bool(bad)


def greedy(llama, ids, count, stops):
    """count tokens the engine writes after ids, the most likely each time, until a stop token"""
    written, logits = [], None
    for pos, token in enumerate(ids):
        logits = llama.forward(token, pos)
    for pos in range(len(ids), len(ids) + count):
        token = int(np.argmax(logits))
        written.append(token)
        if token in stops:
            break
        logits = llama.forward(token, pos)
    return written


def files_of(name):
    repo, revision, shards = MODELS[name]
    weights = ["model.safetensors"] if not shards else \
        ["model.safetensors.index.json", *(f"model-{at:05d}-of-{shards:05d}.safetensors" for at in range(1, shards + 1))]
    return repo, revision, FILES + weights, weights


def real(directory, positions, name, with_faults):
    import tokenizers
    import torch
    import transformers

    repo, revision, files, weights = files_of(name)
    say(f"real: {repo}@{revision}, transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}")
    for file in files:
        fetch(file, directory, repo, revision)
    _, ForCausalLM = code(directory)
    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))

    # the conversion, the way the page does it (by the conduct of a conversion, answered from the files fetched above
    # and no others), float32
    path = directory / "float32.bin"
    conversion, seconds = float32_file(Directory(directory), {"weights": weights[0], "tokenizer": "tokenizer.json"}, path)
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    say(f"real: converted in {seconds:.1f} s, {path.stat().st_size} bytes, options {options_shown(options)}")
    say(f"real: the converter read {'no template' if 'template' not in conversion.options else 'a template: ' + json.dumps(conversion.options['template'], ensure_ascii=False)}")
    specials = sorted(set(options.get("specials", [])) | set(SPECIALS), key=lambda token: (-len(token), token))
    checkpoint = np.memmap(path, dtype=np.uint8, mode="r")

    def make(**other):
        return Llama(checkpoint, conversion.tokenizer, kernels=None, **{**options, "specials": specials, **other})

    llama = make()
    say(f"real: the BOS is {llama.bos} and the answer stops at {sorted(llama.stop_tokens)}; {len(llama.local['layers']) if llama.local else 0} "
        f"of {llama.n_layers} layers are local ones, a window of {llama.local['window'] if llama.local else 0}; RoPE turns {llama.rotary} "
        f"of the others' {llama.head_size}")

    # the same ids for both: the BOS, then the text as the real tokenizer splits it (and the engine's own tokenizer has
    # to split it the same)
    ids = [BOS] + tokenizer.encode(TEXT, add_special_tokens=False).ids
    mine = [BOS] + llama.tokenizer.encode(TEXT, llama.specials)
    template = transformers.AutoTokenizer.from_pretrained(str(directory)).apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    chat_ids = tokenizer.encode(template, add_special_tokens=False).ids
    chat_mine = llama.tokenizer.encode(template, llama.specials)
    same = mine == ids and chat_mine == chat_ids
    failed = not same
    say(f"real: {len(ids)} tokens of text and {len(chat_ids)} of the chat prompt (its own template: {json.dumps(template, ensure_ascii=False)}), "
        f"the engine's tokenizer gives {'the same ids' if same else 'OTHER IDS — FAILED'}")
    ids = ids[:positions]

    model = ForCausalLM.from_pretrained(str(directory), torch_dtype=torch.float32).eval()
    began = time.perf_counter()
    whole, at_once, stepped, stepped_states = both_ways(model, ids, model_states)
    say(f"real: its code, {len(ids)} positions at once and token by token in {time.perf_counter() - began:.1f} s")
    stops = sorted(llama.stop_tokens)
    with torch.no_grad():
        generated = model.generate(torch.tensor([chat_ids]), max_new_tokens=NEW_TOKENS, do_sample=False,
                                   repetition_penalty=1.0, eos_token_id=stops)[0].tolist()
    theirs = generated[len(chat_ids):]

    began = time.perf_counter()
    ours = engine_logits(llama, ids)
    say(f"real: the engine (NumPy, float32), {len(ids)} positions in {time.perf_counter() - began:.1f} s")
    logits_said(say, "real", whole, ours)
    floor = differences(stepped, whole)[0]
    say(f"real: its code token by token against its code at once: largest difference {floor:.2e}")
    # the line: ten times what the code's own two computations differ by, and no less than 1e-3; the most likely token
    # the same wherever the reference's best two are further apart than twice the difference. The faults below say what
    # it catches
    line = line_of(floor, LOGITS_LINE)
    for what, reference in (("its code at once", whole), ("its code token by token", stepped)):
        failed |= logits_failed(say, f"real: the engine against {what}", ours, reference, line, gap=True)
    failed |= HELD.failed(say, "real", llama, len(ids), at_once, stepped_states)
    state_floor = HELD.distances(at_once, stepped_states)

    # 16 greedy tokens for the chat prompt, by the code's generate() and by the engine's forward pass
    wrote = greedy(make(), chat_ids, NEW_TOKENS, set(stops))
    same = wrote == theirs[:len(wrote)] and (len(wrote) == len(theirs) or wrote[-1] in stops)
    failed |= not same
    say(f"real: for the chat prompt its code wrote {json.dumps(tokenizer.decode(theirs, skip_special_tokens=False), ensure_ascii=False)} {theirs}")
    say(f"real: for the chat prompt the engine wrote {json.dumps(tokenizer.decode(wrote, skip_special_tokens=False), ensure_ascii=False)}: "
        f"{'the two wrote the same' if same else 'OTHER TEXTS — FAILED'}")

    # the long text: past the window
    long, at = long_ids(tokenizer, BOS, LONG_AT)
    long_theirs, long_states = kept_logits(model, long, at, model_states)
    failed |= long_failed(llama, long, at, long_theirs, long_states, line)
    del model
    gc.collect()
    if with_faults:
        failed |= faults(make, ids, whole, stepped_states, line, state_floor, (long, at, long_theirs, long_states))
    del llama, checkpoint
    path.unlink()
    return failed


# ------------------------------------------------------------------------------------------------------- faults
# The engine with one thing wrong, each a function that takes a new engine of the real model (and what makes one) and
# returns the engine to run. What the right engine is held to must let none of them by. LONG: the ones that only a text
# longer than the window shows.
def window(change):
    def fault(llama, make):
        llama.window_kept = llama.local["window"]
        llama.local = {**llama.local, "window": change(llama.local["window"], llama.seq_len)}
        return llama
    return fault


def base(which, value):
    def fault(llama, make):
        if which == "local":
            angles = np.arange(llama.seq_len)[:, None] * rope_frequencies(llama.head_size, value)
            llama.local_cis = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        else:
            angles = np.arange(llama.seq_len)[:, None] * rope_frequencies(llama.rotary, value)
            tables = [np.zeros((llama.seq_len, llama.head_size // 2), dtype=np.float32) for _ in range(2)]
            for table, turn in zip(tables, (np.cos, np.sin)):
                table[:, :llama.rotary // 2] = turn(angles)
            llama.freq_cis_real, llama.freq_cis_imag = tables
        return llama
    return fault


def a_local_layer_attends_all(llama, make):
    """the first local layer among those that attend all (their base and their part of a head too)"""
    return make(local={**llama.local, "layers": list(llama.local["layers"][1:])})


def a_layer_that_attends_all_is_local(llama, make):
    """the first layer that attends all among the local ones (their base and their whole head too)"""
    other = next(layer for layer in range(llama.n_layers) if layer not in llama.local["layers"])
    return make(local={**llama.local, "layers": [*llama.local["layers"], other]})


def swapped(first, second):
    def fault(llama, make):
        a, b = getattr(llama, first), getattr(llama, second)
        setattr(llama, first, b)
        setattr(llama, second, a)
        return llama
    return fault


def gate(function):
    def fault(llama, make):
        llama.activation = function
        return llama
    return fault


def no_gate_of_the_heads(llama, make):
    llama.head_gate = None
    return llama


def the_heads_gated_by_each_others(llama, make):
    """the gate of head h by the value of the last head but h"""
    llama.head_gate = llama.head_gate[:, ::-1].copy()
    return llama


def the_gates_values_doubled(llama, make):
    """each head gated by sigmoid(2 g) in place of sigmoid(g)"""
    llama.head_gate = llama.head_gate * np.float32(2.0)
    return llama


def epsilon(value):
    def fault(llama, make):
        llama.rms_norm_eps = value
        return llama
    return fault


FAULTS = {
    "no gate of the heads": no_gate_of_the_heads,
    "each head gated by another's value": the_heads_gated_by_each_others,
    "the gate's values twice what they are": the_gates_values_doubled,
    "the tanh form of the GELU in the FFN's gate (a Gemma's)": gate(gelu),
    "a SiLU in the FFN's gate (a Llama's)": gate(silu),
    "the FFN's gate and up in each other's place": swapped("w1", "w3"),
    "the norms in front of the attention and of the FFN in each other's place": swapped("rms_att_weight", "rms_ffn_weight"),
    "the others turn the whole head": lambda llama, make: make(rotary=llama.head_size),
    "the others turn an eighth of the head": lambda llama, make: make(rotary=llama.rotary // 2),
    "the local layers' base for the others too (1e4)": lambda llama, make: base("global", llama.local["rope_theta"])(llama, make),
    "the others' base for the local layers too (5e6)": lambda llama, make: base("local", 5000000.0)(llama, make),
    "the first local layer attends all the positions, by the others' base and part": a_local_layer_attends_all,
    "the first layer that attends all is a local one": a_layer_that_attends_all_is_local,
}
# What is measured without being held to the line, to know how far it is: another epsilon of the norms
NOTED = {
    "the norms' epsilon 1e-5": epsilon(1e-5),
}
# (what only a text longer than the window shows: its logits at the positions of LONG_AT and the states at its end)
LONG_FAULTS = {
    "the window a position too long": window(lambda size, context: size + 1),
    "the window a position too short": window(lambda size, context: size - 1),
    "no window: the local layers attend all the positions": window(lambda size, context: context),
    "half the window": window(lambda size, context: size // 2),
}


def faults(make, ids, whole, states, line, state_floor, long):
    """Every fault's distance from the model's code's numbers, and whether what the right engine is held to (the logits
    within line, the states within theirs) catches it. True where one gets by."""
    lines = HELD.lines(state_floor)
    long_ids_, at, long_theirs, long_states = long
    long_lines = dict.fromkeys(HELD.names, HELD.least)
    slipped, weakest = [], None
    for name, fault in [*FAULTS.items(), *LONG_FAULTS.items(), *NOTED.items()]:
        llama = fault(make(), make)
        if name in LONG_FAULTS:
            ours, reference, count, held, their_states = long_run(llama, long_ids_, at), long_theirs, len(long_ids_), long_lines, long_states
        else:
            ours, reference, count, held, their_states = engine_logits(llama, ids), whole, len(ids), lines, states
        largest, _, agree, _ = differences(ours, reference)
        apart = HELD.apart(llama, count, their_states)
        # how many times past its line the fault is, by the logits and by the states: what catches it is the larger
        by_logits = largest / line if np.isfinite(largest) else float("inf")
        by_states = max((apart[state] / held[state] if np.isfinite(apart[state]) else float("inf")) for state in HELD.names)
        caught = not largest <= line or bool(HELD.past(apart, held))
        margin = max(by_logits, by_states)
        if name in NOTED:
            say(f"noted ({name}): logits {largest:.2e} from its code's ({by_logits:.1f} times the line), the same most likely "
                f"token at {agree} of {len(ours)}; states {by_states:.1f} times their line")
            continue
        if not caught:
            slipped.append(name)
        if weakest is None or margin < weakest[1]:
            weakest = (name, margin)
        say(f"fault ({name}): logits {largest:.2e} from its code's ({by_logits:.1f} times the line), the same most likely token "
            f"at {agree} of {len(ours)}; states " + ", ".join(f"{state} {apart[state]:.1e}" for state in HELD.names)
            + f" ({by_states:.1f} times their line){'' if caught else ' — NOT CAUGHT, FAILED'}")
    total = len(FAULTS) + len(LONG_FAULTS)
    say(f"faults: {total - len(slipped)} of {total} are past the line (logits {line:.1e}; states "
        + ", ".join(f"{state} {lines[state]:.1e}" for state in HELD.names)
        + f"); the weakest is \"{weakest[0]}\", {weakest[1]:.1f} times past it{'' if not slipped else ' — FAILED: ' + '; '.join(slipped)}")
    return bool(slipped)


def main():
    directory = Path(sys.argv[1])
    only, names, positions = option("only"), option("model", "1.7B").split(","), int(option("positions", 96))
    if only == "fetch":
        for name in names:
            repo, revision, files, _ = files_of(name)
            for file in files:
                fetch(file, directory / name, repo, revision)
        return
    failed = False
    if only in (None, "made-up"):
        for label, settings in MADE_UP.items():
            failed |= made_up(label, settings, directory=directory / "code")
    if only in (None, "real"):
        # several models, one after another (--model=1.7B,4B): the faults on the first alone
        for name in names:
            failed |= real(directory / name, positions, name, name == names[0] and "--no-faults" not in sys.argv)
    ended(say, failed)


if __name__ == "__main__":
    main()
