# reference_plamo3.py
# T649.2: the engine's PLaMo 3 (Preferred Networks', model_type plamo3: a Gemma 3's layers with a SiLU in the gate,
# seven local layers of every eight with a window of 2048, yarn in the others, norms of c + weight with c = 1, 1/5 or
# 5^-1.5) against transformers running the model's own code (modeling_plamo.py: transformers has no class of it).
# PyTorch is not on the development machine, so this runs in CI:
#
#   python tests/reference_plamo3.py <directory for the download> [--only=made-up|real|fetch] [--model=0.8B[,2B]] [--positions=96]
#   (--only=fetch: the real model's files into the directory and no more)
#
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_plamo3.sh" --ref <branch> --grep "plamo3:"
#
# Three parts, every line of the log beginning with "plamo3:".
#   made-up: tiny random models of the model's own class (Plamo3ForCausalLM, from its modeling_plamo.py at the fixed
#     revision) with a window of 8 positions, which the 80 positions of the text are ten times, yarn whose ramp falls
#     inside the heads, and what the real 0.8B has not: keys for every head, a classifier padded to 16 rows, a classifier
#     that is the embedding, no yarn. Its logits, over the whole text at once (its masks) and token by token with its
#     cache (which keeps the last window - 1 keys of a sliding layer), against the naive reference of the unit tests
#     (test_plamo3.naive_plamo3_logits) and against the engine, converted the way the page converts. And what each holds
#     after the text: the keys and values of every layer, of a sliding layer the last window - 1. The model caches its
#     keys before RoPE turns them (each by its place in the cache, after the window cropped it): the engine's keys are
#     turned back by the engine's own tables to be compared.
#   real: a published model at a fixed revision (MODELS; the 0.8B unless --model= says another), float32. The same
#     token ids through the model's code and through the engine (NumPy): the largest difference of the logits, how
#     often the most likely token is the same, the keys and values after them, 16 greedy tokens of each for a request
#     of a translation, and then LONG positions of a longer text: past the window of 2048, which the 96 positions
#     before never reach. The ids are the model's own tokenizer's (tokenization_plamo.py): until the converter reads
#     its tokenizer.jsonl (T649.1) the engine is converted with a made-up vocabulary of the same size, and the engine's
#     own tokenizer is not compared.
#   faults: the real model's engine with one thing wrong at a time (FAULTS), each against the same numbers of the
#     model's code. Those of the window show past 2048 positions alone and are run on the long text. Every fault has to
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
from llama2_numpy import Llama, gelu, rope_frequencies  # noqa: E402
from conducting import Directory, Missing  # noqa: E402
from made_up import unigram  # noqa: E402
from reference import (TEXT, States, both_ways, differences, ended, engine_logits, fetch, float32_file,  # noqa: E402
                       kept_logits, line_of, logits_failed, logits_said, made_up_engine, option, options_shown, within)

# the published models: (repository, revision, the shards of its safetensors)
MODELS = {"0.8B": ("pfnet/plamo-3-translate-0.8b", "cc8e88169807d5a272e8ff06db12eb0129b67f81", None),
          "2B": ("pfnet/plamo-3-translate-2b", "355637cc04db67195ba87a97da85eaf1d52a22e8", 2),
          "8B": ("pfnet/plamo-3-translate-8b", "681bd55cf01f5412aab8881e6a4a276c948fabc1", 4)}
CODE = ["modeling_plamo.py"]
FILES = ["config.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "chat_template.jinja", "tokenizer.jsonl",
         "tokenization_plamo.py", *CODE]
BOS = 1  # <|plamo:bos|>
STOPS = (1, 16)  # generation_config.json's: <|plamo:bos|> and <|plamo:tag|>
# a request of a translation, as the model's chat_template.jinja writes it
PROMPT = "The meeting has been rescheduled to next Monday because the room was not available."
LANGUAGES = {"source_lang": "en", "target_lang": "ja"}
NEW_TOKENS = 16
LONG = 2304  # the positions of the long text: past the window of 2048
LONG_SPLIT = 2040  # the code's second way through the long text: so many at once, the rest a token at a time
# the positions of the long text whose logits are compared: around the window, where a sliding layer first leaves a
# position out, and to the end
LONG_AT = (0, 1, 255, 1023, 2046, 2047, 2048, 2049, 2050, 2100, 2200, 2302, 2303)

say = lambda *parts: print("plamo3:", *parts, flush=True)
LOGITS_LINE = 1e-3  # the least of the logits' line, which is ten times that floor too (the made-up models': 2e-3)


def code(directory):
    """The model's own code (its modeling_plamo.py, fetched with it), as a package of the directory's: (Plamo3Config,
    Plamo3ForCausalLM). transformers has no class of this model; this is what trust_remote_code would run, read
    without the question."""
    package = Path(directory) / "plamo3_code"
    package.mkdir(parents=True, exist_ok=True)
    for name in CODE:
        shutil.copyfile(Path(directory) / name, package / name)
    (package / "__init__.py").write_text("")
    sys.path.insert(0, str(Path(directory)))
    importlib.invalidate_caches()
    modeling = importlib.import_module("plamo3_code.modeling_plamo")
    return modeling.Plamo3Config, modeling.Plamo3ForCausalLM


def code_directory(directory):
    """where the model's code is fetched for the made-up models: the 0.8B's, at its revision"""
    repo, revision, _ = MODELS["0.8B"]
    for name in CODE:
        fetch(name, Path(directory), repo, revision)
    return Path(directory)


def own_tokenizer(directory):
    """the model's own tokenizer (tokenization_plamo.py, which needs numba), by transformers"""
    import transformers
    return transformers.AutoTokenizer.from_pretrained(str(directory), trust_remote_code=True)


def ids_of(tokenizer, text):
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


# What the engine and the model's code hold after the same positions: the keys and values of every layer. The model's
# cache keeps the last window - 1 positions of a sliding-attention layer and all of the others, and its keys are those
# before RoPE (after the norms of the heads), in Hugging Face's order of a head; the engine keeps them all (or its
# window's ring) turned, in its interleaved order. The engine's are turned back by its own tables (whose cos and sin
# carry yarn's magnitude where it has one) and put in Hugging Face's order.
def hf_order(size):
    """the engine's index of Hugging Face's 0..size-1, the pairs of a head interleaved"""
    return np.concatenate([np.arange(0, size, 2), np.arange(1, size, 2)])


def unturned(keys, first, cos, sin):
    """keys (heads, positions, size) of positions first.. turned back by the tables (cos, sin: positions by pairs)"""
    c, s = cos[first:first + keys.shape[1]][None], sin[first:first + keys.shape[1]][None]
    y0, y1 = keys[..., 0::2], keys[..., 1::2]
    square = c * c + s * s
    out = np.empty_like(keys)
    out[..., 0::2], out[..., 1::2] = (y0 * c + y1 * s) / square, (y1 * c - y0 * s) / square
    return out


def engine_states(llama, count):
    """{layer: (kind, [keys, values])}"""
    out, head = {}, llama.head_size
    local = set(llama.local["layers"]) if llama.local else ()
    # (the window the model's cache goes by: the model's, whatever a fault made of the engine's)
    size = getattr(llama, "window_kept", llama.local["window"] if llama.local else 0)
    for l, (_, a) in enumerate(llama.slots):
        first = max(0, count - (size - 1)) if l in local else 0
        cos, sin = llama.local_cis if l in local else (llama.freq_cis_real, llama.freq_cis_imag)
        keys = unturned(np.asarray(llama.key_cache[a, :, first:count], dtype=np.float64), first, cos, sin)
        out[l] = ("sliding" if l in local else "full",
                  [keys[..., hf_order(head)], llama.value_cache[a, :, first:count]])
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
# (layers, pattern: every pattern-th layer, counted from 1, attends every position)
MADE_UP = {
    "the 0.8B's order in small": dict(layers=8, pattern=8),
    "keys for every head, a classifier padded to 16 rows": dict(layers=4, pattern=4, num_key_value_heads=4, vocab_size=312),
    "its classifier the embedding, no yarn": dict(layers=4, pattern=2, tie_word_embeddings=True, factor=1.0),
    "every layer attends all": dict(layers=3, pattern=1),
    "another epsilon, other bases": dict(layers=4, pattern=4, rms_norm_eps=1e-5, theta=20000.0, local_theta=1000.0),
}


def made_up(name, settings, positions=80, directory=None):
    import torch
    from test_plamo3 import naive_plamo3_logits

    Config, ForCausalLM = code(code_directory(directory or HERE.parent / ".tmp" / "reference-plamo3"))
    settings = dict(settings)
    layers, pattern = settings.pop("layers"), settings.pop("pattern")
    theta, local_theta, factor = settings.pop("theta", 10000.0), settings.pop("local_theta", 100.0), settings.pop("factor", 4.0)
    said = dict(num_key_value_heads=1, head_dim=32, window_size=8, rms_norm_eps=1e-6, tie_word_embeddings=False, vocab_size=320)
    said.update(settings)
    config = Config(hidden_size=32, intermediate_size=64, num_hidden_layers=layers, num_attention_heads=4,
                    max_position_embeddings=256, bos_token_id=1, eos_token_id=1, pad_token_id=3,
                    sliding_window_pattern=pattern, rope_theta=theta, rope_local_theta=local_theta,
                    rope_scaling_factor=factor, initial_context_length=512 if factor != 1 else None, **said)
    torch.manual_seed(649)
    model = ForCausalLM(config).to(torch.float32).eval()
    tied = bool(config.tie_word_embeddings)
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key == "lm_head.weight" and tied:
                continue
            # (a norm's weight is stored around zero: the model adds its c)
            parameter.copy_(0.3 * torch.randn_like(parameter))
    rng = np.random.default_rng(649)
    tokens = [int(token) for token in rng.integers(0, said["vocab_size"], positions)]
    whole, at_once, stepped, stepped_states = both_ways(model, tokens, model_states)
    tensors = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    if tied:
        tensors.pop("lm_head.weight", None)
    published = json.loads(config.to_json_string(use_diff=False))
    naive = naive_plamo3_logits(tensors, published, tokens)

    llama, options, _ = made_up_engine(tensors, published, said["vocab_size"])
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
    sliding = [layer for layer in range(layers) if (layer + 1) % pattern]
    theirs = {"layers": sliding, "window": said["window_size"], "rope_theta": local_theta} if sliding else None
    scaling = options.get("rope_scaling")
    right = options.get("local") == theirs and (scaling or {}).get("factor", 1.0) == factor
    failed |= not right
    say(f"made-up ({name}): the local layers {json.dumps(options.get('local'))}, yarn {json.dumps(scaling)}"
        f"{'' if right else f' — FAILED (its config: {json.dumps(theirs)}, a factor of {factor})'}, options "
        f"{json.dumps({key: options[key] for key in ('arch', 'head_dim', 'rope_theta', 'rms_norm_eps') if key in options})}")
    return failed


# ------------------------------------------------------------------------------------------------- the real model
def long_run(llama, ids, at):
    """the engine through ids: its logits at the positions `at`"""
    kept = {}
    for pos, token in enumerate(ids):
        logits = llama.forward(token, pos, need_logits=pos in at)
        if pos in at:
            kept[pos] = logits.copy()
    return np.stack([kept[position] for position in at])


def stepped_logits(model, ids, at, split):
    """the model's code on ids its second way: the first split positions at once, then one at a time on its cache (so
    the window's edge is crossed a token at a time): its logits at the positions `at`"""
    import torch

    kept = {}
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids[:split]]), use_cache=True,
                    logits_to_keep=torch.tensor([position for position in at if position < split] or [0]))
        early = [position for position in at if position < split]
        for i, position in enumerate(early):
            kept[position] = out.logits[0, i].float().numpy()
        cache = out.past_key_values
        for position in range(split, len(ids)):
            out = model(input_ids=torch.tensor([[ids[position]]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if position in at:
                kept[position] = out.logits[0, -1].float().numpy()
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
    # (the weights are "model.safetensors" to the conversion, which asks for its index where it is sharded)
    weights = ["model.safetensors"] if not shards else \
        ["model.safetensors.index.json", *(f"model-{at:05d}-of-{shards:05d}.safetensors" for at in range(1, shards + 1))]
    return repo, revision, FILES + weights, "model.safetensors"


def converted_file(directory, weights, vocabulary):
    """The conversion the way the page does it (by the conduct of a conversion, answered from the files fetched and
    no others), float32: with the model's own tokenizer.jsonl where the converter reads it, else (until T649.1) with a
    made-up vocabulary of its size, written beside the files. (the conversion, the seconds, which vocabulary)"""
    path = directory / "float32.bin"
    try:
        conversion, seconds = float32_file(Directory(directory), {"weights": weights, "tokenizer": "tokenizer.jsonl"}, path)
        return conversion, seconds, "its own tokenizer.jsonl"
    except (ValueError, Missing) as refused:
        said = str(refused).splitlines()[0][:200] if str(refused) else type(refused).__name__
    (directory / "made-up-tokenizer.json").write_bytes(unigram(vocabulary))
    conversion, seconds = float32_file(Directory(directory), {"weights": weights, "tokenizer": "made-up-tokenizer.json"}, path)
    return conversion, seconds, f"a made-up vocabulary (the converter does not read tokenizer.jsonl: {said})"


def real(directory, positions, name, with_faults):
    import torch
    import transformers

    repo, revision, files, weights = files_of(name)
    say(f"real: {repo}@{revision}, transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}")
    for file in files:
        fetch(file, directory, repo, revision)
    _, ForCausalLM = code(directory)
    tokenizer = own_tokenizer(directory)
    vocabulary = json.loads((directory / "config.json").read_text())["vocab_size"]

    conversion, seconds, read = converted_file(directory, weights, vocabulary)
    path = directory / "float32.bin"
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    say(f"real: converted in {seconds:.1f} s with {read}, {path.stat().st_size} bytes, options {options_shown(options)}")
    checkpoint = np.memmap(path, dtype=np.uint8, mode="r")
    own = read.startswith("its own")

    def make(**other):
        llama = Llama(checkpoint, conversion.tokenizer, kernels=None, **{**options, **other})
        # (what it was made with: the faults of RoPE make its tables again of these)
        llama.given = {**options, **other}
        return llama

    llama = make()
    say(f"real: {len(llama.local['layers']) if llama.local else 0} of {llama.n_layers} layers are local ones, a window of "
        f"{llama.local['window'] if llama.local else 0}; yarn {json.dumps(options.get('rope_scaling'))}")

    # the same ids for both: the BOS, then the text as the model's own tokenizer splits it
    ids = [BOS] + ids_of(tokenizer, TEXT)
    template = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False,
                                             **LANGUAGES)
    chat_ids = ids_of(tokenizer, template)
    failed = False
    if own:
        mine, chat_mine = [BOS] + llama.tokenizer.encode(TEXT, llama.specials), llama.tokenizer.encode(template, llama.specials)
        same = mine == ids and chat_mine == chat_ids
        failed |= not same
        said = "the same ids" if same else "OTHER IDS — FAILED"
    else:
        said = "not compared (a made-up vocabulary)"
    say(f"real: {len(ids)} tokens of text and {len(chat_ids)} of the request (its own template: {json.dumps(template, ensure_ascii=False)}), "
        f"the engine's tokenizer: {said}")
    ids = ids[:positions]

    model = ForCausalLM.from_pretrained(str(directory), torch_dtype=torch.float32).eval()
    began = time.perf_counter()
    whole, at_once, stepped, stepped_states = both_ways(model, ids, model_states)
    say(f"real: its code, {len(ids)} positions at once and token by token in {time.perf_counter() - began:.1f} s")
    with torch.no_grad():
        generated = model.generate(torch.tensor([chat_ids]), max_new_tokens=NEW_TOKENS, do_sample=False,
                                   repetition_penalty=1.0, eos_token_id=list(STOPS))[0].tolist()
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

    # 16 greedy tokens for the request, by the code's generate() and by the engine's forward pass
    wrote = greedy(make(), chat_ids, NEW_TOKENS, set(STOPS))
    same = wrote == theirs[:len(wrote)] and (len(wrote) == len(theirs) or wrote[-1] in STOPS)
    failed |= not same
    say(f"real: for the request its code wrote {json.dumps(tokenizer.decode(theirs, skip_special_tokens=False), ensure_ascii=False)} {theirs}")
    say(f"real: for the request the engine wrote {json.dumps(tokenizer.decode(wrote, skip_special_tokens=False), ensure_ascii=False)}: "
        f"{'the two wrote the same' if same else 'OTHER TEXTS — FAILED'}")

    # the long text: past the window
    text = " ".join(f"Section {i}. {TEXT}" for i in range(40))
    long = ([BOS] + ids_of(tokenizer, text))[:LONG]
    at = [position for position in LONG_AT if position < len(long)]
    long_theirs, long_states = kept_logits(model, long, at, model_states)
    # the long text's own line: ten times what the code's two computations of it differ by (over 2304 positions the
    # float32 sums of the two drift further apart than over 96), and no less than the short text's
    split = min(LONG_SPLIT, len(long) - 16)  # (a shorter text, the dry run's, still steps its last 16)
    long_floor = differences(stepped_logits(model, long, at, split), long_theirs)[0]
    long_line = line_of(long_floor, line)
    # (the code's two ways apart by more than the short text's line: the reference itself is not to be trusted, and a
    # line ten times that would let a wrong engine by)
    suspect = not long_floor <= line
    failed |= suspect
    say(f"long: its code with the first {split} positions at once and then token by token against its code at once: "
        f"largest difference {long_floor:.2e}; the line {long_line:.1e}"
        + (f" — FAILED (its two ways past the short text's line {line:.1e})" if suspect else ""))
    failed |= long_failed(llama, long, at, long_theirs, long_states, long_line)
    del model
    gc.collect()
    if with_faults:
        failed |= faults(make, ids, whole, stepped_states, line, state_floor, (long, at, long_theirs, long_states, long_line))
    del llama, checkpoint
    path.unlink()
    return failed


# ------------------------------------------------------------------------------------------------------- faults
# The engine with one thing wrong, each a function that takes a new engine of the real model (and what makes one) and
# returns the engine to run. What the right engine is held to must let none of them by. LONG_FAULTS: the ones that only
# a text longer than the window shows.
def window(change):
    def fault(llama, make):
        llama.window_kept = llama.local["window"]
        llama.local = {**llama.local, "window": change(llama.local["window"], llama.seq_len)}
        # (the forward pass reads a layer's window off its facts, T639.2)
        llama.layers = [{**facts, "window": llama.local["window"]} if facts["local"] else facts for facts in llama.layers]
        return llama
    return fault


def tables(scaling, magnitude, theta=None):
    """the tables of the layers that attend all made another way: yarn's scaling or none, a magnitude, a base"""
    def fault(llama, make):
        angles = np.arange(llama.seq_len)[:, None] * rope_frequencies(llama.head_size, theta or llama.given.get("rope_theta", 10000.0), scaling)
        llama.freq_cis_real, llama.freq_cis_imag = ((turn(angles) * magnitude).astype(np.float32) for turn in (np.cos, np.sin))
        return llama
    return fault


def yarn_of(llama):
    return dict(llama.given["rope_scaling"])


def local_base(value):
    def fault(llama, make):
        angles = np.arange(llama.seq_len)[:, None] * rope_frequencies(llama.head_size, value)
        llama.local_cis = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        return llama
    return fault


def plus(name, value):
    """a norm's weight with value more (a c of another number)"""
    def fault(llama, make):
        setattr(llama, name, np.asarray(getattr(llama, name), dtype=np.float32) + np.float32(value))
        return llama
    return fault


def a_local_layer_attends_all(llama, make):
    """the first local layer among those that attend all (their base and yarn too)"""
    return make(local={**llama.local, "layers": list(llama.local["layers"][1:])})


def a_layer_that_attends_all_is_local(llama, make):
    """the first layer that attends all among the local ones (their base and no yarn too)"""
    other = next(layer for layer in range(llama.n_layers) if layer not in llama.local["layers"])
    return make(local={**llama.local, "layers": sorted([*llama.local["layers"], other])})


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


def epsilon(value):
    def fault(llama, make):
        llama.rms_norm_eps = value
        return llama
    return fault


def magnitude_of(llama):
    return 0.1 * math.log(float(llama.given["rope_scaling"]["factor"])) + 1.0


FAULTS = {
    "the norm behind the attention by 1 + w (not 1/5 + w)": plus("rms_att_post_weight", 1.0 - 1.0 / 5),
    "the norm behind the FFN by 1 + w (not 5^-1.5 + w)": plus("rms_ffn_post_weight", 1.0 - 5 ** -1.5),
    "the norms behind the attention and the FFN by each other's c": lambda llama, make: plus("rms_ffn_post_weight", 1.0 / 5 - 5 ** -1.5)(
        plus("rms_att_post_weight", 5 ** -1.5 - 1.0 / 5)(llama, make), make),
    "the norms of the heads by w (no 1)": lambda llama, make: plus("k_norm", -1.0)(plus("q_norm", -1.0)(llama, make), make),
    "the last norm by w (no 1)": plus("rms_final_weight", -1.0),
    "a GELU in the FFN's gate (a Gemma's)": gate(gelu),
    "the FFN's gate and up in each other's place": swapped("w1", "w3"),
    "the norms in front of the attention and of the FFN in each other's place": swapped("rms_att_weight", "rms_ffn_weight"),
    "the norms behind the attention and the FFN in each other's place": swapped("rms_att_post_weight", "rms_ffn_post_weight"),
    "no yarn in the layers that attend all": lambda llama, make: tables(None, 1.0)(llama, make),
    "yarn without its magnitude": lambda llama, make: tables(yarn_of(llama), 1.0)(llama, make),
    "yarn's magnitude without its slowing": lambda llama, make: tables(None, magnitude_of(llama))(llama, make),
    "yarn's bounds rounded (truncate)": lambda llama, make: tables({**yarn_of(llama), "truncate": True}, magnitude_of(llama))(llama, make),
    "the local layers' base for the others too (1e4)": lambda llama, make: tables(yarn_of(llama), magnitude_of(llama),
                                                                                    llama.local["rope_theta"])(llama, make),
    "the others' base for the local layers too (1e6)": lambda llama, make: local_base(llama.given.get("rope_theta", 10000.0))(llama, make),
    "the first local layer attends all the positions, by the others' base and yarn": a_local_layer_attends_all,
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
    long_ids_, at, long_theirs, long_states, long_line = long
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
        logits_line = long_line if name in LONG_FAULTS else line
        by_logits = largest / logits_line if np.isfinite(largest) else float("inf")
        by_states = max((apart[state] / held[state] if np.isfinite(apart[state]) else float("inf")) for state in HELD.names)
        caught = not largest <= logits_line or bool(HELD.past(apart, held))
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
    only, names, positions = option("only"), option("model", "0.8B").split(","), int(option("positions", 96))
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
        # several models, one after another (--model=0.8B,2B): the faults on the first alone
        for name in names:
            failed |= real(directory / name, positions, name, name == names[0] and "--no-faults" not in sys.argv)
    ended(say, failed)


if __name__ == "__main__":
    main()
