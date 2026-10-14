# reference_gemma3.py
# T261: the engine's Gemma 3 (Google's text model: a norm behind the attention and the FFN as well as in front, a GELU
# in the FFN's gate, layers that attend a window of the last positions with a base of RoPE of their own) against
# transformers itself. PyTorch is not on the development machine, so this runs in CI:
#
#   pip install numpy tokenizers torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
#   pip install safetensors "transformers @ git+https://github.com/huggingface/transformers@28034545c35acac05793695bd5c878f66e6b544e"
#   python tests/reference_gemma3.py <directory for the download> [--only=made-up|real|fetch] [--model=270M[,1B]] [--positions=96]
#   (--only=fetch: the real model's files into the directory and no more)
#
#   node tests/ci.mjs run tests.yml only_extra=true extra="bash tests/reference_gemma3.sh" --ref <branch> --grep "gemma3:"
#
# Three parts, every line of the log beginning with "gemma3:".
#   made-up: tiny random models of transformers' own class (Gemma3ForCausalLM) with a window of 8 positions, which the 80
#     positions of the text are ten times, and with what the real 270M has not: a classifier of its own, the kinds of
#     the layers by a pattern, keys for every head, heads that fill the width. transformers' logits, over the whole text
#     at once (its masks) and token by token with its cache (which keeps the last window - 1 keys of a sliding layer),
#     against the naive reference of the unit tests (conftest.naive_gemma3_logits) and against the engine, converted the
#     way the page converts (llama2_convert.Conversion). And what each holds after the text: the keys and values of
#     every layer, of a sliding layer the last window - 1.
#   real: a published model at a fixed revision (MODELS; unsloth/gemma-3-270m-it unless --model= says another), float32.
#     The same token ids through transformers and through the engine (NumPy): the largest difference of the logits, how
#     often the most likely token is the same, the keys and values after them, 16 greedy tokens of each for the chat
#     prompt, and then 1024 positions of a longer text: twice the window of 512, which the 96 positions before never
#     reach.
#   faults: the real model's engine with one thing wrong at a time (FAULTS), each against the same numbers of
#     transformers. Those of the window (one position too long or too short, none at all) show past 512 positions
#     alone and are run on the long text; the others on the 96 positions. Every fault has to be past the line the right
#     engine is held to, or the line would let it through: the weakest is named.
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
from llama2_numpy import Llama, rope_frequencies, silu  # noqa: E402
from conducting import Directory  # noqa: E402
from reference import (TEXT, States, both_ways, differences, ended, engine_logits, fetch, float32_file,  # noqa: E402
                       kept_logits, line_of, logits_failed, logits_said, long_ids, made_up_engine, option, options_shown,
                       within)

# the published models: (repository, revision). Google's own are gated; unsloth's copies are not
MODELS = {"270M": ("unsloth/gemma-3-270m-it", "23cf460f6bb16954176b3ddcc8d4f250501458a9"),
          "1B": ("unsloth/gemma-3-1b-it", "5b11413a10db4e486ef16a20101fd028f8f2499c")}
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja",
         "model.safetensors"]
BOS = 2  # <bos>: what the real tokenizer puts first
# the chat format of one turn, as the model's template writes it after its bos_token
CHAT = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
BEGINS = "<bos>"
PROMPT = "What is the capital of Japan? Answer in one sentence."
LISTED = "これからの流行りを3つ挙げてください。"  # the list's prompt
SPECIALS = ["<start_of_turn>", "<end_of_turn>"]
STOPS = (1, 106)  # <eos> and <end_of_turn>: generation_config.json's eos_token_id, what transformers' generate() stops at
NEW_TOKENS = 16

say = lambda *parts: print("gemma3:", *parts, flush=True)

# What the engine and transformers hold after the same positions: the keys (as Hugging Face has them: the pairs of a
# head not interleaved) and values of every layer. transformers' cache keeps the last window - 1 positions of a
# sliding-attention layer (DynamicSlidingWindowLayer) and all of the others; the engine keeps them all, and its last
# window - 1 are what is compared for a local layer. A layer of the other kind on either side is infinitely far.
LOGITS_LINE = 1e-3  # the least of the logits' line, which is ten times that floor too (the made-up models': 2e-3)


def engine_states(llama, count):
    """{layer: (kind, [keys, values])}"""
    out, head = {}, llama.head_size
    order = np.concatenate([np.arange(0, head, 2), np.arange(1, head, 2)])  # the engine's index of Hugging Face's 0..head-1
    local = set(llama.local["layers"]) if llama.local else ()
    # (the window transformers' cache goes by: the model's, whatever a fault made of the engine's)
    size = getattr(llama, "window_kept", llama.local["window"] if llama.local else 0)
    for l, (_, a) in enumerate(llama.slots):
        first = max(0, count - (size - 1)) if l in local else 0
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
    "the 270M's order in small": dict(kinds="ssssslsssssl"),
    "a classifier of its own, a window of 5": dict(kinds="sslssl", tie_word_embeddings=False, sliding_window=5),
    "the kinds by a pattern of 3, keys for every head": dict(pattern=3, layers=7, num_key_value_heads=4),
    "heads that fill the width, another epsilon, other bases": dict(kinds="lsslss", head_dim=8, rms_norm_eps=1e-5,
                                                                    theta=5000.0, local_theta=300.0),
    "no sliding layer at all": dict(kinds="llll"),
}


def made_up(name, settings, positions=80):
    import torch
    from conftest import naive_gemma3_logits
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    settings = dict(settings)
    kinds, pattern = settings.pop("kinds", None), settings.pop("pattern", None)
    layers = settings.pop("layers", len(kinds or ""))
    theta, local_theta = settings.pop("theta", 1000000.0), settings.pop("local_theta", 10000.0)
    said = dict(num_key_value_heads=1, head_dim=16, sliding_window=8, rms_norm_eps=1e-6)
    said.update(settings)
    if kinds:
        said["layer_types"] = ["sliding_attention" if kind == "s" else "full_attention" for kind in kinds]
    else:
        said["sliding_window_pattern"] = pattern
    config = Gemma3TextConfig(vocab_size=320, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                              num_attention_heads=4, max_position_embeddings=256, bos_token_id=2, eos_token_id=7,
                              pad_token_id=0, query_pre_attn_scalar=said["head_dim"],
                              rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": theta},
                                               "sliding_attention": {"rope_type": "default", "rope_theta": local_theta}},
                              **said)
    torch.manual_seed(261)
    model = Gemma3ForCausalLM(config).to(torch.float32).eval()
    tied = bool(config.tie_word_embeddings)
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            # (a norm's weight is around zero here: the model multiplies by 1 + weight)
            if not (key == "lm_head.weight" and tied):
                parameter.copy_(0.3 * torch.randn_like(parameter))
    if tied:
        assert model.lm_head.weight is model.model.embed_tokens.weight, "the classifier is not tied"
    rng = np.random.default_rng(261)
    tokens = [int(token) for token in rng.integers(0, 320, positions)]
    whole, at_once, stepped, stepped_states = both_ways(model, tokens, model_states)
    tensors = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    if tied:
        tensors.pop("lm_head.weight", None)
    published = json.loads(config.to_json_string(use_diff=False))
    # the naive reference reads the config as the unit tests' own (made_up.gemma3_model) says it
    naive = naive_gemma3_logits(tensors, {**published, "layer_types": list(config.layer_types), "rope_theta": theta,
                                          "rope_local_base_freq": local_theta, "sliding_window": said["sliding_window"],
                                          "query_pre_attn_scalar": said["head_dim"], "head_dim": said["head_dim"],
                                          "rms_norm_eps": said["rms_norm_eps"]}, tokens)

    llama, options, _ = made_up_engine(tensors, published)
    ours = engine_logits(llama, tokens)

    failed = False
    floor = differences(stepped, whole)[0]
    line = line_of(floor, 2 * LOGITS_LINE)
    say(f"made-up ({name}), transformers token by token against transformers at once: largest difference {floor:.2e}")
    pairs = [("the naive reference against transformers at once", naive, whole),
             ("the engine against transformers at once", ours, whole),
             ("the engine against transformers token by token", ours, stepped)]
    for what, a, b in pairs:
        failed |= logits_failed(say, f"made-up ({name}), {what}", a, b, line)
    failed |= HELD.failed(say, f"made-up ({name})", llama, positions, at_once, stepped_states)
    sliding = [layer for layer, kind in enumerate(config.layer_types) if kind == "sliding_attention"]
    theirs = {"layers": sliding, "window": said["sliding_window"], "rope_theta": local_theta} if sliding else None
    failed |= options.get("local") != theirs
    say(f"made-up ({name}): the local layers {json.dumps(options.get('local'))}"
        f"{'' if options.get('local') == theirs else f' — FAILED (transformers: {json.dumps(theirs)})'}, options "
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
    """The engine through the long text, its logits at `at` against transformers' (within line) and its keys and values
    at the end against transformers' (within the least of States: no floor is measured here). True where either is not."""
    began = time.perf_counter()
    largest, mean, same, margin = differences(long_run(llama, ids, at), theirs)
    apart = HELD.apart(llama, len(ids), their_states)
    ok = within(largest, same, len(at), margin, line)
    bad = HELD.past(apart, dict.fromkeys(HELD.names, HELD.least))
    say(f"long: the engine ran {len(ids)} positions in {time.perf_counter() - began:.1f} s; the logits at {len(at)} of them against "
        f"transformers': largest difference {largest:.2e}, mean {mean:.2e}, the same most likely token at {same} of {len(at)}"
        f"{'' if ok else f' — FAILED (the line: {line:.1e})'}; the states after them: "
        + "; ".join(f"{name} {apart[name]:.1e}" for name in HELD.names)
        + (f" — FAILED (the line: {HELD.least:.0e}): {', '.join(bad)}" if bad else ""))
    return not ok or bool(bad)


def real(directory, positions, name, with_faults):
    import tokenizers
    import torch
    import transformers
    from transformers import Gemma3ForCausalLM

    repo, revision = MODELS[name]
    say(f"real: {repo}@{revision}, transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}")
    for file in FILES:
        fetch(file, directory, repo, revision)
    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))

    # the conversion, the way the page does it (by the conduct of a conversion, answered from the files fetched above
    # and no others), float32
    path = directory / "float32.bin"
    conversion, seconds = float32_file(Directory(directory), {"weights": "model.safetensors", "tokenizer": "tokenizer.json"}, path)
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    say(f"real: converted in {seconds:.1f} s, {path.stat().st_size} bytes, options {options_shown(options)}")
    say(f"real: the converter read {'no template' if 'template' not in conversion.options else 'a template: ' + json.dumps(conversion.options['template'])}")
    specials = sorted(set(options.get("specials", [])) | set(SPECIALS), key=lambda token: (-len(token), token))
    checkpoint = np.memmap(path, dtype=np.uint8, mode="r")

    def make(**other):
        return Llama(checkpoint, conversion.tokenizer, kernels=None, **{**options, "specials": specials, **other})

    llama = make()
    failed = llama.bos != BOS or not set(llama.stop_tokens) & set(STOPS)
    say(f"real: the BOS is {llama.bos} and the answer stops at {sorted(llama.stop_tokens)} (transformers' generate() at "
        f"{list(STOPS)}){' — FAILED' if failed else ''}")
    window = llama.local["window"] if llama.local else 0
    say(f"real: {len(llama.local['layers']) if llama.local else 0} of {llama.n_layers} layers are local ones, a window of {window}")

    # the same ids for both: the BOS the real tokenizer puts first, then the text as the real tokenizer splits it (and
    # the engine's own tokenizer has to split it the same)
    with_bos = tokenizer.encode(TEXT).ids
    ids = [BOS] + tokenizer.encode(TEXT, add_special_tokens=False).ids
    mine = [BOS] + llama.tokenizer.encode(TEXT, llama.specials)
    chat = CHAT.format(prompt=PROMPT)
    chat_ids = tokenizer.encode(chat, add_special_tokens=False).ids
    chat_mine = llama.tokenizer.encode(chat, llama.specials)
    same = mine == ids and chat_mine == chat_ids and with_bos == ids
    failed |= not same
    say(f"real: {len(ids)} tokens of text and {len(chat_ids)} of the chat prompt, the engine's tokenizer gives "
        f"{'the same ids' if same else 'OTHER IDS — FAILED'}; the real tokenizer begins a text with {with_bos[0]}")
    template = transformers.AutoTokenizer.from_pretrained(str(directory)).apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    failed |= template != BEGINS + chat
    say(f"real: the model's own template writes {json.dumps(template)}: "
        f"{'the format here after its BOS' if template == BEGINS + chat else 'ANOTHER FORMAT — FAILED'}")
    ids = ids[:positions]

    model = Gemma3ForCausalLM.from_pretrained(str(directory), dtype=torch.float32).eval()
    began = time.perf_counter()
    whole, at_once, stepped, stepped_states = both_ways(model, ids, model_states)
    say(f"real: transformers, {len(ids)} positions at once and token by token in {time.perf_counter() - began:.1f} s")
    listed = CHAT.format(prompt=LISTED)
    listed_ids = tokenizer.encode(listed, add_special_tokens=False).ids
    with torch.no_grad():
        generated = model.generate(torch.tensor([[BOS] + chat_ids]), max_new_tokens=NEW_TOKENS, do_sample=False,
                                   repetition_penalty=1.0)[0].tolist()
        as_listed = model.generate(torch.tensor([[BOS] + listed_ids]), max_new_tokens=NEW_TOKENS, do_sample=False,
                                   repetition_penalty=1.0)[0].tolist()[1 + len(listed_ids):]
    theirs = generated[1 + len(chat_ids):]

    began = time.perf_counter()
    ours = engine_logits(llama, ids)
    say(f"real: the engine (NumPy, float32), {len(ids)} positions in {time.perf_counter() - began:.1f} s")
    logits_said(say, "real", whole, ours)
    floor = differences(stepped, whole)[0]
    say(f"real: transformers token by token against transformers at once: largest difference {floor:.2e}")
    # the line: ten times what transformers' own two computations differ by, and no less than 1e-3; the most likely
    # token the same wherever the reference's best two are further apart than twice the difference. The faults below
    # say what it catches
    line = line_of(floor, LOGITS_LINE)
    for what, reference in (("transformers at once", whole), ("transformers token by token", stepped)):
        failed |= logits_failed(say, f"real: the engine against {what}", ours, reference, line, gap=True)
    failed |= HELD.failed(say, "real", llama, len(ids), at_once, stepped_states)
    state_floor = HELD.distances(at_once, stepped_states)

    # 16 greedy tokens for the chat prompt, by transformers' generate() and by the engine's
    for what, prompt, prompt_ids, wrote in (("the chat prompt", chat, chat_ids, theirs), ("the list's prompt", listed, listed_ids, as_listed)):
        text = "".join(llama.generate(prompt, steps=len(prompt_ids) + NEW_TOKENS, temperature=0.0, echo=False))
        stop = next((at for at, token in enumerate(wrote) if token in llama.stop_tokens), len(wrote))
        same = tokenizer.decode(wrote[:stop], skip_special_tokens=False) == text
        failed |= not same
        say(f"real: for {what} transformers wrote {json.dumps(tokenizer.decode(wrote, skip_special_tokens=False), ensure_ascii=False)} {wrote}")
        say(f"real: for {what} the engine wrote {json.dumps(text, ensure_ascii=False)}: {'the two wrote the same' if same else 'OTHER TEXTS — FAILED'}")

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
        angles = np.arange(llama.seq_len)[:, None] * rope_frequencies(llama.head_size, value)
        tables = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        if which == "local":
            llama.local_cis = tables
        else:
            llama.freq_cis_real, llama.freq_cis_imag = tables
        return llama
    return fault


def bases_in_each_others_place(llama, make):
    (llama.freq_cis_real, llama.freq_cis_imag), llama.local_cis = llama.local_cis, (llama.freq_cis_real, llama.freq_cis_imag)
    return llama


def a_local_layer_attends_all(llama, make):
    """the first local layer among those that attend all (their base too)"""
    return make(local={**llama.local, "layers": list(llama.local["layers"][1:])})


def a_layer_that_attends_all_is_local(llama, make):
    """the first layer that attends all among the local ones (their base too)"""
    other = next(layer for layer in range(llama.n_layers) if layer not in llama.local["layers"])
    return make(local={**llama.local, "layers": [*llama.local["layers"], other]})


def swapped(first, second):
    def fault(llama, make):
        a, b = getattr(llama, first), getattr(llama, second)
        setattr(llama, first, b)
        setattr(llama, second, a)
        return llama
    return fault


def without(name, value=1.0):
    """a norm that multiplies by this in place of its weights (1: no weights; the fault of a 1 not added is less(name))"""
    def fault(llama, make):
        setattr(llama, name, np.full_like(getattr(llama, name), value))
        return llama
    return fault


def less(name):
    """the 1 that the model adds to a norm's weight not added"""
    def fault(llama, make):
        setattr(llama, name, getattr(llama, name) - np.float32(1.0))
        return llama
    return fault


def not_normalized(name):
    """what the attention or the FFN gives joins the stream as it is: no norm behind"""
    def fault(llama, make):
        setattr(llama, name, None)
        return llama
    return fault


def embedding_times(factor):
    """the embedding's row times this (and the last norm, which the converter divided, as it is)"""
    def fault(llama, make):
        right = llama.embedding
        llama.embedding = lambda token: right(token) * np.float32(factor)
        return llama
    return fault


def last_norm_not_divided(llama, make):
    llama.rms_final_weight = llama.rms_final_weight * np.float32(math.sqrt(llama.dim))
    return llama


def gate(function):
    def fault(llama, make):
        llama.activation = function
        return llama
    return fault


def exact_gelu(x):
    """x times the normal distribution's function of x, where the model's is the tanh form of it"""
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return (x * 0.5 * (1.0 + erf(np.asarray(x, dtype=np.float64) / math.sqrt(2.0)))).astype(np.float32)


def epsilon(value):
    def fault(llama, make):
        llama.rms_norm_eps = value
        return llama
    return fault


def scores_by_the_width_over_the_heads(llama, make):
    """llama.cpp's reading of a Gemma 3 of 62 layers, on this one: the scores divided by the root of dim / heads"""
    llama.q_norm = llama.q_norm * np.float32(math.sqrt(llama.head_size / (llama.dim / llama.n_heads)))
    return llama


FAULTS = {
    "no norm behind the attention": not_normalized("rms_att_post_weight"),
    "no norm behind the FFN": not_normalized("rms_ffn_post_weight"),
    "the norms behind the attention and behind the FFN in each other's place": swapped("rms_att_post_weight", "rms_ffn_post_weight"),
    "the norms in front of the FFN and behind it in each other's place": swapped("rms_ffn_weight", "rms_ffn_post_weight"),
    "the norms in front of the attention and behind it in each other's place": swapped("rms_att_weight", "rms_att_post_weight"),
    "the 1 of the norm behind the attention not added": less("rms_att_post_weight"),
    "the 1 of the norm in front of the FFN not added": less("rms_ffn_weight"),
    "the 1 of the heads' norm of q not added": less("q_norm"),
    "the norms of q's heads and of k's in each other's place": swapped("q_norm", "k_norm"),
    "no weights in the norm of k's heads": without("k_norm"),
    "the embedding not multiplied by the root of the width": lambda llama, make: embedding_times(1.0 / math.sqrt(llama.dim))(llama, make),
    "the last norm not divided by the root of the width": last_norm_not_divided,
    "a SiLU in the FFN's gate (a Llama's)": gate(silu),
    "the exact GELU in the FFN's gate, not its tanh form": gate(exact_gelu),
    "the FFN's gate and up in each other's place": swapped("w1", "w3"),
    "the scores divided by the root of the width over the heads": scores_by_the_width_over_the_heads,
    "the local layers' base for every layer (1e4)": lambda llama, make: base("global", llama.local["rope_theta"])(llama, make),
    "the other layers' base for the local ones too (1e6)": lambda llama, make: base("local", 1000000.0)(llama, make),
    "the two bases in each other's place": bases_in_each_others_place,
    "the first local layer attends all the positions, by the others' base": a_local_layer_attends_all,
    "the first layer that attends all is a local one": a_layer_that_attends_all_is_local,
}
# What is measured without being held to the line, to know how far it is. Another epsilon of the norms: a Gemma's
# stream is large beside 1e-6 (the embedding is multiplied by the root of the width), and a made-up model's is larger
# still, so this may be within the line (an LFM2's was 17 times its own). And what is no fault of the engine's:
# transformers multiplies the embedding by the root of the width in the weights' own type, so a model run in bfloat16
# multiplies by the root as bfloat16 rounds it (25.25 for the 270M's 25.298), where float32 (this comparison, and
# llama.cpp) has the root itself
NOTED = {
    "the norms' epsilon 1e-5": epsilon(1e-5),
    "the norms' epsilon 0": epsilon(0.0),
    "the embedding multiplied by the root as bfloat16 rounds it": lambda llama, make: embedding_times(
        float(np.array([math.sqrt(llama.dim)], dtype=np.float32).view(np.uint32).__add__(np.uint32(0x8000)).__and__(np.uint32(0xFFFF0000)).view(np.float32)[0])
        / math.sqrt(llama.dim))(llama, make),
}
# (what only a text longer than the window shows: its logits at the positions of LONG_AT and the states at its end)
LONG_FAULTS = {
    "the window a position too long": window(lambda size, context: size + 1),
    "the window a position too short": window(lambda size, context: size - 1),
    "no window: the local layers attend all the positions": window(lambda size, context: context),
    "half the window": window(lambda size, context: size // 2),
}


def faults(make, ids, whole, states, line, state_floor, long):
    """Every fault's distance from transformers' numbers, and whether what the right engine is held to (the logits
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
            say(f"noted ({name}): logits {largest:.2e} from transformers' ({by_logits:.1f} times the line), the same most likely "
                f"token at {agree} of {len(ours)}; states {by_states:.1f} times their line")
            continue
        if not caught:
            slipped.append(name)
        if weakest is None or margin < weakest[1]:
            weakest = (name, margin)
        say(f"fault ({name}): logits {largest:.2e} from transformers' ({by_logits:.1f} times the line), the same most likely token "
            f"at {agree} of {len(ours)}; states " + ", ".join(f"{state} {apart[state]:.1e}" for state in HELD.names)
            + f" ({by_states:.1f} times their line){'' if caught else ' — NOT CAUGHT, FAILED'}")
    total = len(FAULTS) + len(LONG_FAULTS)
    say(f"faults: {total - len(slipped)} of {total} are past the line (logits {line:.1e}; states "
        + ", ".join(f"{state} {lines[state]:.1e}" for state in HELD.names)
        + f"); the weakest is \"{weakest[0]}\", {weakest[1]:.1f} times past it{'' if not slipped else ' — FAILED: ' + '; '.join(slipped)}")
    return bool(slipped)


def main():
    directory = Path(sys.argv[1])
    only, names, positions = option("only"), option("model", "270M").split(","), int(option("positions", 96))
    if only == "fetch":
        for name in names:
            for file in FILES:
                fetch(file, directory / name, *MODELS[name])
        return
    failed = False
    if only in (None, "made-up"):
        for label, settings in MADE_UP.items():
            failed |= made_up(label, settings)
    if only in (None, "real"):
        # several models, one after another (--model=270M,1B): the faults on the first alone
        for name in names:
            failed |= real(directory / name, positions, name, name == names[0] and "--no-faults" not in sys.argv)
    ended(say, failed)


if __name__ == "__main__":
    main()
