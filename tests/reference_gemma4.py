# reference_gemma4.py
# T.329.3: the engine's Gemma 4 (Google's E2B and E4B language models: a Gemma 3's layers, with full-attention layers of
# heads twice as wide that turn their first quarter by the angles of the whole head (proportional RoPE), v normalized
# head by head, a scale at the end of every layer, the logits capped, the last layers reading the keys and values of a
# layer before them (T.330), and an input of every layer's own from a table of every token (T.329)) against transformers
# itself. PyTorch is not on the development machine, so this runs in CI:
#
#   pip install numpy tokenizers torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
#   pip install safetensors jinja2 "transformers @ git+https://github.com/huggingface/transformers@2126f65b0ab267e113e78a002447fc7094de83d5"
#   python tests/reference_gemma4.py <directory for the download> [--only=made-up|real|fetch] [--model=E2B] [--positions=96]
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=150 extra="bash tests/reference_gemma4.sh" --ref <branch> --grep "gemma4:"
#
# Three parts, every line of the log beginning with "gemma4:".
#   made-up: tiny random models of transformers' own class (Gemma4ForCausalLM) with a window of 8 positions, which the
#     80 positions of the text are ten times: the E2B's kinds of layers in small (shared keys and values, a double FFN,
#     a table of every layer's input), the E4B's (a classifier of its own, two key heads, a wider table, no double FFN),
#     and one with none of those and other numbers (a half turned, another epsilon, other bases, no cap). transformers'
#     logits, over the whole text at once and token by token with its cache, against the naive reference of the unit
#     tests (conftest.naive_gemma4_logits) and against the engine, converted the way the page converts. And what each
#     holds after the text: the keys and values of every layer that makes its own.
#   real: google/gemma-4-E2B-it at a fixed revision (not gated: Gemma 4 is Apache-2.0), float32 for the engine. The
#     language model alone does not fit a runner of 16 GB as float32 (18.6 GB), so transformers holds its weights as
#     stored (bfloat16, 9.3 GB) and computes in float32 (every nn.Linear widens its weight: tests/reference_llama.py's
#     float32_arithmetic), and the two embeddings are taken from their tables and multiplied in float32 before they go
#     in (inputs_embeds and per_layer_inputs: transformers multiplies by the root of the width in the weights' own type,
#     which bfloat16 rounds). A bfloat16 is a float32 to the bit, so this is the float32 model's arithmetic. The same
#     token ids through transformers and through the engine (NumPy): the logits, the keys and values, 16 greedy tokens
#     of two chat prompts, and then a longer text past the window of 512.
#   faults: the real model's engine with one thing of a Gemma 4 wrong at a time (FAULTS), each held to transformers'
#     numbers as the right engine is. Those of the window show past 512 positions alone and run on the long text.
#     Every fault has to be past the line the right engine is held to: the weakest is named.
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
from llama2_numpy import Llama, rope_frequencies  # noqa: E402
from conducting import Directory  # noqa: E402
from reference import (TEXT, States, both_ways, differences, ended, engine_logits, fetch, float32_file,  # noqa: E402
                       line_of, logits_failed, logits_said, long_ids, made_up_engine, option, options_shown, within)

# the published models: (repository, revision). Google's own, which are not gated
MODELS = {"E2B": ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
          "E4B": ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2")}
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja",
         "model.safetensors"]
BOS = 2  # <bos>: what the model's template begins with (its tokenizer puts nothing first)
# the chat format of one turn, as the model's template writes it after its bos_token (without thinking, its default)
CHAT = "<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
BEGINS = "<bos>"
PROMPT = "What is the capital of Japan? Answer in one sentence."
LISTED = "日本でいちばん高い山はどこですか？"
SPECIALS = ["<|turn>", "<turn|>"]
STOPS = (1, 106, 50)  # generation_config.json's eos_token_id: <eos>, <turn|>, <|tool_response>
NEW_TOKENS = 16
LOGITS_LINE = 1e-3  # the least of the logits' line (the made-up models': 2e-3)
FAULT_POSITIONS = 48  # the positions a fault of the real model runs (the line is the 96 positions')
LONG_POSITIONS = 640  # of the long text (reference.long_ids): past the window of 512 by a quarter

say = lambda *parts: print("gemma4:", *parts, flush=True)


# What the engine and transformers hold after the same positions: the keys (as Hugging Face has them: the pairs of a
# head not interleaved) and values of every layer that makes its own (the shared layers read another's: T.330).
# transformers' cache keeps the last window - 1 positions of a sliding-attention layer; the engine keeps them all, and
# its last window - 1 are what is compared for a local layer.
def engine_states(llama, count):
    """{layer: (kind, [keys, values])}, of the layers that make their own"""
    out = {}
    local = set(llama.local["layers"]) if llama.local else ()
    size_kept = getattr(llama, "window_kept", llama.local["window"] if llama.local else 0)
    for l, (_, a) in enumerate(llama.slots):
        _, _, size, kept, held = llama.heads[a]
        if kept is None:
            continue
        order = np.concatenate([np.arange(0, size, 2), np.arange(1, size, 2)])  # the engine's index of Hugging Face's
        first = max(0, count - (size_kept - 1)) if l in local else 0
        out[l] = ("sliding" if l in local else "full",
                  [llama.key_cache[held][:, first:count][..., order], llama.value_cache[held][:, first:count]])
    return out


def model_states(past, count):
    """{layer: (kind, [keys, values])} of transformers' cache, of the layers that hold something (a shared layer
    writes nothing to it)"""
    out = {}
    for l, layer in enumerate(past.layers):
        keys = getattr(layer, "keys", None)
        if keys is None:
            continue
        keys = keys[0].float().numpy()
        if keys.size == 0:
            continue
        kind = "sliding" if getattr(layer, "is_sliding", False) else "full"
        out[l] = (kind, [keys[:, :count], layer.values[0].float().numpy()[:, :count]])
    return out


HELD = States({"sliding": ("keys of the sliding layers", "values of the sliding layers"), "full": ("keys", "values")},
              engine_states, model_states)


# --------------------------------------------------------------------------------------------- made-up models
# s: a sliding-attention layer, l: one that attends every position (transformers makes the last layer such)
MADE_UP = {
    "the E2B's kinds in small: shared keys, a double FFN, a table": dict(kinds="sslsslssl", sharing=3, double=True, per_layer=8),
    "the E4B's in small: a classifier of its own, two key heads, a wider table": dict(
        kinds="ssslssslssl", sharing=3, per_layer=16, num_key_value_heads=2, tie_word_embeddings=False),
    "none of those: a half turned, a window of 5, another epsilon, other bases, no cap": dict(
        kinds="slsl", proportion=0.5, sliding_window=5, rms_norm_eps=1e-5, theta=5000.0, local_theta=300.0, softcap=None),
}


def made_up(name, settings, positions=80):
    import torch
    from conftest import naive_gemma4_logits
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    settings = dict(settings)
    kinds = settings.pop("kinds")
    theta, local_theta = settings.pop("theta", 1000000.0), settings.pop("local_theta", 10000.0)
    proportion, softcap = settings.pop("proportion", 0.25), settings.pop("softcap", 30.0)
    said = dict(num_key_value_heads=1, head_dim=16, sliding_window=8, rms_norm_eps=1e-6, tie_word_embeddings=True,
                sharing=0, double=False, per_layer=0)
    said.update(settings)
    sharing, double, per_layer = said.pop("sharing"), said.pop("double"), said.pop("per_layer")
    wide = 2 * said["head_dim"]
    config = Gemma4TextConfig(vocab_size=320, hidden_size=32, intermediate_size=64, num_hidden_layers=len(kinds),
                              num_attention_heads=4, max_position_embeddings=256, bos_token_id=2, eos_token_id=7,
                              pad_token_id=0, global_head_dim=wide, final_logit_softcapping=softcap,
                              layer_types=["sliding_attention" if kind == "s" else "full_attention" for kind in kinds],
                              num_kv_shared_layers=sharing, use_double_wide_mlp=double,
                              hidden_size_per_layer_input=per_layer, vocab_size_per_layer_input=320,
                              hidden_activation="gelu_pytorch_tanh", attention_k_eq_v=False, enable_moe_block=False,
                              rope_parameters={"full_attention": {"rope_type": "proportional", "partial_rotary_factor": proportion,
                                                                  "rope_theta": theta},
                                               "sliding_attention": {"rope_type": "default", "rope_theta": local_theta}},
                              **said)
    torch.manual_seed(329)
    model = Gemma4ForCausalLM(config).to(torch.float32).eval()
    tied = bool(config.tie_word_embeddings)
    with torch.no_grad():
        # (the norms multiply by their weight itself: around 1; every layer's scale a number of its own)
        for key, tensor in model.state_dict().items():
            if key == "lm_head.weight" and tied:
                continue
            if key.endswith("layer_scalar"):
                tensor.copy_(0.5 + torch.rand_like(tensor))
            elif key.endswith("norm.weight"):
                tensor.copy_(1.0 + 0.3 * torch.randn_like(tensor))
            else:
                tensor.copy_(0.3 * torch.randn_like(tensor))
    rng = np.random.default_rng(329)
    tokens = [int(token) for token in rng.integers(0, 320, positions)]
    whole, at_once, stepped, stepped_states = both_ways(model, tokens, model_states)
    tensors = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    if tied:
        tensors.pop("lm_head.weight", None)
    # (global_head_dim is an argument of the config that it does not keep: the published config.json has it)
    published = {**json.loads(config.to_json_string(use_diff=False)), "global_head_dim": wide}
    naive = naive_gemma4_logits(tensors, published, tokens)

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
    # what the converter read of the config, against what transformers' own modules are
    layers = model.model.layers
    shared = [at for at, layer in enumerate(layers) if layer.self_attn.is_kv_shared_layer]
    sliding = [at for at, kind in enumerate(config.layer_types) if kind == "sliding_attention"]
    theirs = {"local": sorted(sliding), "kv_shared": shared,
              "double": [at for at, layer in enumerate(layers) if layer.mlp.intermediate_size != config.intermediate_size],
              "per_layer": per_layer}
    mine = {"local": sorted((options.get("local") or {}).get("layers", [])), "kv_shared": (options.get("kv_shared") or {}).get("layers", []),
            "double": options.get("double") or [], "per_layer": options.get("per_layer", 0)}
    failed |= mine != theirs
    say(f"made-up ({name}): the converter read {json.dumps(mine)}{'' if mine == theirs else f' — FAILED (transformers: {json.dumps(theirs)})'}; "
        f"options {json.dumps({key: options[key] for key in ('arch', 'head_dim', 'wide', 'rotary', 'softcap', 'rms_norm_eps') if key in options})}")
    return failed


# ------------------------------------------------------------------------------------------------- the real model
# the positions of the long text whose logits are compared: around 256 (where the keys and values double), around the
# window of 512 (where a sliding layer first leaves a position out), and to the end
LONG_AT = (0, 1, 255, 256, 257, 510, 511, 512, 513, 514, 575, 600, 639)


def text_model(directory):
    """transformers' Gemma4ForCausalLM of the published model's text_config, its weights as stored (bfloat16) from the
    language model's tensors of model.safetensors (model.language_model.*: the towers of images and sound are left
    out), the arithmetic float32 (float32_arithmetic) and RoPE's tables float32. The shared layers' k, v and norms of
    the file are not the model's (transformers leaves them unread too): they are counted, not loaded."""
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import Gemma4ForCausalLM

    config = transformers.AutoConfig.from_pretrained(str(directory)).text_config
    model = Gemma4ForCausalLM._from_config(config, dtype=torch.bfloat16).eval()
    wanted, loaded, unread, prefix = model.state_dict(), set(), 0, "model.language_model."
    with torch.no_grad(), safe_open(str(directory / "model.safetensors"), "pt") as file:
        for key in file.keys():
            if not key.startswith(prefix):
                continue
            name = "model." + key[len(prefix):]
            if name not in wanted:
                unread += 1
                continue
            wanted[name].copy_(file.get_tensor(key))
            loaded.add(name)
    missing = set(wanted) - loaded - {"lm_head.weight"}
    if missing:
        raise ValueError(f"the file has none of {sorted(missing)[:8]} ({len(missing)})")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.rotary_emb = type(model.model.rotary_emb)(config)  # (float32 tables, made outside the bfloat16 init)
    say(f"real: transformers' {type(model).__name__}, {len(loaded)} tensors of the language model as stored "
        f"({next(model.parameters()).dtype}), {unread} of the file it does not read, arithmetic in float32")
    return model


def embedded(model, ids):
    """The two inputs of these ids as transformers' float32 model makes them: the embedding's rows times the root of
    the width, and every layer's rows of the table times the root of their width (batch, positions, layers, width)."""
    import torch
    text = model.model
    config = model.config
    tokens = torch.tensor(ids)
    x = text.embed_tokens.weight[tokens].to(torch.float32) * torch.tensor(config.hidden_size ** 0.5, dtype=torch.float32)
    width = config.hidden_size_per_layer_input
    rows = text.embed_tokens_per_layer.weight[tokens].to(torch.float32) * torch.tensor(width ** 0.5, dtype=torch.float32)
    return x[None], rows.reshape(1, len(ids), config.num_hidden_layers, width)


def their_passes(model, ids):
    """transformers on ids, by embedded(): (logits at once, its states, logits token by token, its states)"""
    import torch
    x, rows = embedded(model, ids)
    with torch.no_grad():
        out = model(inputs_embeds=x, per_layer_inputs=rows, use_cache=True)
        whole, at_once = out.logits[0].float().numpy(), model_states(out.past_key_values, len(ids))
        past, stepped = None, []
        for at in range(len(ids)):
            out = model(inputs_embeds=x[:, at:at + 1], per_layer_inputs=None if rows is None else rows[:, at:at + 1],
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            stepped.append(out.logits[0, -1].float().numpy())
    return whole, at_once, np.stack(stepped), model_states(past, len(ids))


def their_kept(model, ids, at):
    """transformers on ids at once: (the logits at the positions `at`, its states after them)"""
    import torch
    x, rows = embedded(model, ids)
    with torch.no_grad():
        out = model(inputs_embeds=x, per_layer_inputs=rows, use_cache=True, logits_to_keep=torch.tensor(at))
    return out.logits[0].float().numpy(), model_states(out.past_key_values, len(ids))


def their_greedy(model, ids, count, stops):
    """count tokens after ids, the most likely one each time, by transformers with its cache; stops at a stop token"""
    import torch
    written = []
    with torch.no_grad():
        x, rows = embedded(model, ids)
        out = model(inputs_embeds=x, per_layer_inputs=rows, use_cache=True)
        for _ in range(count):
            token = int(np.argmax(out.logits[0, -1].float().numpy()))
            written.append(token)
            if token in stops:
                break
            x, rows = embedded(model, [token])
            out = model(inputs_embeds=x, per_layer_inputs=rows, past_key_values=out.past_key_values, use_cache=True)
    return written


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
    from reference_llama import float32_arithmetic

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
    template = conversion.options.get("template")
    say(f"real: converted in {seconds:.1f} s, {path.stat().st_size} bytes, options {options_shown(options)}")
    # (the converter writes {prompt:trim} where the template trims what was typed, as Gemma 4's does)
    failed = template is not None and template.replace("{prompt:trim}", "{prompt}") != CHAT.format(prompt="{prompt}")
    say(f"real: the converter read {'no template' if template is None else 'the template ' + json.dumps(template)}"
        f"{' — FAILED (the format here: ' + json.dumps(CHAT) + ')' if failed else ''}")
    specials = sorted(set(options.get("specials", [])) | set(SPECIALS), key=lambda token: (-len(token), token))
    checkpoint = np.memmap(path, dtype=np.uint8, mode="r")

    def make(**other):
        llama = Llama(checkpoint, conversion.tokenizer, kernels=None, **{**options, "specials": specials, **other})
        llama.theta = options["rope_theta"]
        return llama

    llama = make()
    wrong = llama.bos != BOS or not set(llama.stop_tokens) & set(STOPS)
    failed |= wrong
    say(f"real: the BOS is {llama.bos} and the answer stops at {sorted(llama.stop_tokens)} (generation_config.json's "
        f"{list(STOPS)}){' — FAILED' if wrong else ''}")
    say(f"real: {len(llama.local['layers']) if llama.local else 0} of {llama.n_layers} layers are local ones, a window of "
        f"{llama.local['window'] if llama.local else 0}; wide {json.dumps(options.get('wide'))}; shared "
        f"{json.dumps(options.get('kv_shared'))}; double {json.dumps(options.get('double'))}; per layer {options.get('per_layer')}")

    # the same ids for both: the BOS, then the text as the real tokenizer splits it (and the engine's own tokenizer has
    # to split it the same)
    plain = tokenizer.encode(TEXT).ids
    ids = [BOS] + tokenizer.encode(TEXT, add_special_tokens=False).ids
    mine = [BOS] + llama.tokenizer.encode(TEXT, llama.specials)
    chat = CHAT.format(prompt=PROMPT)
    chat_ids = tokenizer.encode(chat, add_special_tokens=False).ids
    chat_mine = llama.tokenizer.encode(chat, llama.specials)
    same = mine == ids and chat_mine == chat_ids
    failed |= not same
    say(f"real: {len(ids)} tokens of text and {len(chat_ids)} of the chat prompt, the engine's tokenizer gives "
        f"{'the same ids' if same else 'OTHER IDS — FAILED'}; the real tokenizer begins a text with {plain[:2]}")
    written = transformers.AutoTokenizer.from_pretrained(str(directory)).apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    failed |= written != BEGINS + chat
    say(f"real: the model's own template writes {json.dumps(written)}: "
        f"{'the format here after its BOS' if written == BEGINS + chat else 'ANOTHER FORMAT — FAILED'}")
    ids = ids[:positions]
    listed = CHAT.format(prompt=LISTED)
    listed_ids = tokenizer.encode(listed, add_special_tokens=False).ids
    long, at = long_ids(tokenizer, BOS, LONG_AT)
    long = long[:LONG_POSITIONS]
    at = [position for position in at if position < len(long)]

    # transformers first, all of it, and let go of it before the engine reads its 18.6 GB
    float32_arithmetic()
    model = text_model(directory)
    began = time.perf_counter()
    whole, at_once, stepped, stepped_states = their_passes(model, ids)
    say(f"real: transformers, {len(ids)} positions at once and token by token in {time.perf_counter() - began:.1f} s")
    began = time.perf_counter()
    theirs = {what: their_greedy(model, [BOS] + prompt_ids, NEW_TOKENS, set(STOPS))
              for what, prompt_ids in (("the chat prompt", chat_ids), ("the list's prompt", listed_ids))}
    long_theirs, long_states = their_kept(model, long, at)
    say(f"real: transformers, the greedy tokens and {len(long)} positions of the long text in {time.perf_counter() - began:.1f} s")
    del model
    gc.collect()

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

    # 16 greedy tokens for two chat prompts, by transformers and by the engine's generate(), both stopping where
    # generation_config.json says (as an entry of the list would give the engine: the converter reads no
    # generation_config.json, and its stops leave out <turn|>, which the answer ends with)
    writer = make(stop_tokens=list(STOPS))
    for what, prompt, prompt_ids in (("the chat prompt", chat, chat_ids), ("the list's prompt", listed, listed_ids)):
        text = "".join(writer.generate(prompt, steps=len(prompt_ids) + NEW_TOKENS, temperature=0.0, echo=False))
        wrote = theirs[what]
        stop = next((at for at, token in enumerate(wrote) if token in writer.stop_tokens), len(wrote))
        same = tokenizer.decode(wrote[:stop], skip_special_tokens=False) == text
        failed |= not same
        say(f"real: for {what} transformers wrote {json.dumps(tokenizer.decode(wrote, skip_special_tokens=False), ensure_ascii=False)} {wrote}")
        say(f"real: for {what} the engine wrote {json.dumps(text, ensure_ascii=False)}: {'the two wrote the same' if same else 'OTHER TEXTS — FAILED'}")

    del writer
    # the long text: past the window
    failed |= long_failed(llama, long, at, long_theirs, long_states, line)
    if with_faults:
        count = min(FAULT_POSITIONS, len(ids))
        failed |= faults(make, ids[:count], whole[:count], line, state_floor, (long, at, long_theirs, long_states))
    del llama, checkpoint
    gc.collect()
    path.unlink()
    return failed


# ------------------------------------------------------------------------------------------------------- faults
# The engine with one thing of a Gemma 4 wrong, each a function that takes a new engine of the real model (and what
# makes one) and returns the engine to run. What the right engine is held to must let none of them by. LONG: the ones
# that only a text longer than the window shows.
def window(change):
    def fault(llama, make):
        llama.window_kept = llama.local["window"]
        llama.local = {**llama.local, "window": change(llama.local["window"], llama.seq_len)}
        return llama
    return fault


def full_angles(width_of, theta_of):
    """the full-attention layers' angles of their turned part, by the frequencies of another width or base"""
    def fault(llama, make):
        rotary, wide = llama.rotary, llama.wide["head_dim"]
        angles = np.arange(llama.seq_len)[:, None] * rope_frequencies(width_of(rotary, wide), theta_of(llama))[:rotary // 2]
        for name, turn in (("freq_cis_real", np.cos), ("freq_cis_imag", np.sin)):
            table = np.array(getattr(llama, name), dtype=np.float32)
            table[:, :rotary // 2] = turn(angles)
            setattr(llama, name, table)
        return llama
    return fault


def local_base(theta):
    def fault(llama, make):
        angles = np.arange(llama.seq_len)[:, None] * rope_frequencies(llama.head_size, theta)
        llama.local_cis = (np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32))
        return llama
    return fault


def attribute(name, value):
    def fault(llama, make):
        setattr(llama, name, value(llama) if callable(value) else value)
        return llama
    return fault


def times(name, factor):
    """the named tensor of the engine times factor (a copy in its place)"""
    def fault(llama, make):
        setattr(llama, name, np.asarray(getattr(llama, name), dtype=np.float32) * np.float32(factor(llama)))
        return llama
    return fault


def table_times(factor):
    """every row of the table of every layer's input times factor"""
    def fault(llama, make):
        right, table = llama.embedding, llama.per_layer_table
        llama.embedding = lambda token, of=None: right(token, of) * np.float32(factor(llama)) if of is table else right(token, of)
        return llama
    return fault


def shared_read_earlier(llama, make):
    """every shared layer reads the keys and values of the layer of its size made before the right one"""
    made = [(held, size) for _, _, size, kept, held in llama.heads if kept is not None]
    def earlier(held, size):
        return max(at for at, of in made if of == size and at < held)
    llama.heads = [(named, at, size, kept, held if kept is not None else earlier(held, size))
                   for named, at, size, kept, held in llama.heads]
    return llama


def scores_by_the_root(llama, make):
    """the scores divided by the root of the head's size: the scale of 1 the converter put into q's norm left out"""
    llama.q_norm = np.asarray(llama.q_norm, dtype=np.float32) / np.float32(math.sqrt(llama.head_size))
    llama.q_norm_wide = np.asarray(llama.q_norm_wide, dtype=np.float32) / np.float32(math.sqrt(llama.wide["head_dim"]))
    return llama


def epsilon(value):
    def fault(llama, make):
        llama.rms_norm_eps = value
        return llama
    return fault


def embedding_times(factor):
    def fault(llama, make):
        right = llama.embedding
        llama.embedding = lambda token, of=None: right(token) * np.float32(factor(llama)) if of is None else right(token, of)
        return llama
    return fault


FAULTS = {
    "v not normalized": attribute("value_norm", False),
    "every layer's scale 1": attribute("layer_scale", lambda llama: np.ones_like(np.asarray(llama.layer_scale))),
    "the logits not capped": attribute("softcap", 0.0),
    "the shared layers read the layer of their size before the right one": shared_read_earlier,
    "the full-attention layers' angles of the turned part's width, not the whole head's": full_angles(
        lambda rotary, wide: rotary, lambda llama: llama.theta),
    "the full-attention layers turn by the local layers' base (1e4)": full_angles(
        lambda rotary, wide: wide, lambda llama: llama.local["rope_theta"]),
    "the local layers turn by the full-attention layers' base (1e6)": lambda llama, make: local_base(llama.theta)(llama, make),
    "the scores divided by the root of the head's size": scores_by_the_root,
    "the norm behind the attention by 1 + its weight (a Gemma 3's)": attribute(
        "rms_att_post_weight", lambda llama: np.asarray(llama.rms_att_post_weight, dtype=np.float32) + np.float32(1.0)),
    "the table's rows not multiplied by the root of half their width": table_times(lambda llama: 1.0 / math.sqrt(llama.per_layer / 2)),
    "the norm of the projection not multiplied by the root of a half": times("per_layer_norm", lambda llama: math.sqrt(2.0)),
    "no table: every layer's input the projection alone": table_times(lambda llama: 0.0),
}
# What is measured without being held to the line, to know how far it is: another epsilon of the norms; the projection of
# every layer's input without its factor (the root of the width over it), which the RMSNorm right after it takes away
# but for its epsilon (T.329.1: no comparison of a model whose projection is far larger than the epsilon sees it, the
# unit tests hold it on a made-up one whose projection is small); and what is no fault of the engine's: the embedding
# multiplied by the root of the width as bfloat16 rounds it (transformers in bfloat16: 39.25 for the E2B's 39.19)
NOTED = {
    "the norms' epsilon 1e-5": epsilon(1e-5),
    "the projection of every layer's input not multiplied by the root of the width": times(
        "per_layer_projection", lambda llama: math.sqrt(llama.dim)),
    "the embedding multiplied by the root as bfloat16 rounds it": embedding_times(lambda llama: float(
        np.array([math.sqrt(llama.dim)], dtype=np.float32).view(np.uint32).__add__(np.uint32(0x8000)).__and__(np.uint32(0xFFFF0000)).view(np.float32)[0])
        / math.sqrt(llama.dim)),
}
LONG_FAULTS = {
    "the window a position too long": window(lambda size, context: size + 1),
    "the window a position too short": window(lambda size, context: size - 1),
    "no window: the local layers attend all the positions": window(lambda size, context: context),
}


def faults(make, ids, whole, line, state_floor, long):
    """Every fault's distance from transformers' numbers, and whether what the right engine is held to (the logits
    within line, the states within theirs) catches it. True where one gets by. The states of a fault are not compared
    (transformers' were kept after all the positions, a fault runs fewer): its logits are what it is held to, and
    the long faults' states after the long text."""
    long_ids_, at, long_theirs, long_states = long
    long_lines = dict.fromkeys(HELD.names, HELD.least)
    slipped, weakest = [], None
    for name, fault in [*FAULTS.items(), *LONG_FAULTS.items(), *NOTED.items()]:
        began = time.perf_counter()
        llama = fault(make(), make)
        by_states, said = 0.0, ""
        if name in LONG_FAULTS:
            ours, reference = long_run(llama, long_ids_, at), long_theirs
            apart = HELD.apart(llama, len(long_ids_), long_states)
            by_states = max((apart[state] / long_lines[state] if np.isfinite(apart[state]) else float("inf")) for state in HELD.names)
            said = "; states " + ", ".join(f"{state} {apart[state]:.1e}" for state in HELD.names) + f" ({by_states:.1f} times their line)"
        else:
            ours, reference = engine_logits(llama, ids), whole
        largest, _, agree, _ = differences(ours, reference)
        by_logits = largest / line if np.isfinite(largest) else float("inf")
        margin = max(by_logits, by_states)
        caught = not largest <= line or by_states > 1.0
        seconds = time.perf_counter() - began
        if name in NOTED:
            say(f"noted ({name}): logits {largest:.2e} from transformers' ({by_logits:.1f} times the line), the same most likely "
                f"token at {agree} of {len(ours)} ({seconds:.0f} s)")
            continue
        if not caught:
            slipped.append(name)
        if weakest is None or margin < weakest[1]:
            weakest = (name, margin)
        say(f"fault ({name}): logits {largest:.2e} from transformers' ({by_logits:.1f} times the line), the same most likely token "
            f"at {agree} of {len(ours)}{said}{'' if caught else ' — NOT CAUGHT, FAILED'} ({seconds:.0f} s)")
    total = len(FAULTS) + len(LONG_FAULTS)
    say(f"faults: {total - len(slipped)} of {total} are past the line (logits {line:.1e}, of {len(ids)} positions and of the "
        f"long text's {len(at)}); the weakest is \"{weakest[0]}\", {weakest[1]:.1f} times past it"
        f"{'' if not slipped else ' — FAILED: ' + '; '.join(slipped)}")
    return bool(slipped)


def main():
    directory = Path(sys.argv[1])
    only, names, positions = option("only"), option("model", "E2B").split(","), int(option("positions", 96))
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
        for name in names:
            failed |= real(directory / name, positions, name, name == names[0] and "--no-faults" not in sys.argv)
    ended(say, failed)


if __name__ == "__main__":
    main()
