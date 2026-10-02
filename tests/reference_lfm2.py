# reference_lfm2.py
# T260: the engine's LFM2 (Liquid AI's LFM2 and LFM2.5: convolution layers among attention layers) against transformers
# itself. PyTorch is not on the development machine, so this runs in CI:
#
#   pip install numpy tokenizers torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
#   pip install safetensors "transformers @ git+https://github.com/huggingface/transformers@7cd73d9df0c14b151c684b708a9f27d8d0349dfe"
#   python tests/reference_lfm2.py <directory for the download> [--only=made-up|real|fetch] [--model=350M] [--positions=96]
#   (--only=fetch: the real model's files into the directory and no more)
#
#   node tests/ci.mjs run tests.yml extra="bash tests/reference_lfm2.sh" --ref <branch> --grep "lfm2:"
#
# Three parts, every line of the log beginning with "lfm2:".
#   made-up: tiny random models of transformers' own class (Lfm2ForCausalLM), which hold what the real 350M does not:
#     a classifier of its own, two taps and four, an FFN whose size the config says as it is, the layers said by
#     full_attn_idxs, another epsilon. transformers' logits, over the whole text at once (its padded convolution) and
#     token by token with its cache (the convolution's last tokens kept), against the naive reference of the unit tests
#     (conftest.naive_lfm2_logits) and against the engine, converted the way the page converts
#     (llama2_convert.Conversion). And what each holds after the text: the convolution's rows of every convolution
#     layer, the keys and values of every attention layer, which a fault shows in long before the logits do.
#   real: a published model at a fixed revision (MODELS; LiquidAI/LFM2.5-350M unless --model= says another), float32.
#     The same token ids through transformers and through the engine (NumPy): the largest difference of the logits, how
#     often the most likely token is the same, the states after them, 16 greedy tokens of each for the chat prompt, and
#     then 1024 positions of a longer text (the cache of the attention layers doubling twice).
#   faults: the real model's engine with one thing wrong at a time (FAULTS: the taps reversed, an activation where the
#     model has none, another epsilon, the convolution a token late, two layers of unlike kinds in each other's place,
#     ...), each against the same numbers of transformers: how far its logits and its states are from them. Every fault
#     has to be past the line the right engine is held to, or the line would let it through: the weakest is named.
# The lines it holds the engine to are at the end of each part; anything past them is exit 1.
import gc
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
from llama2_convert import Conversion  # noqa: E402
from llama2_numpy import Llama, rope_frequencies, silu  # noqa: E402
from reference_qwen35 import CHUNK, TEXT, File, differences, fetch, relative  # noqa: E402

# the published models: (repository, revision). The 350M is the one the list's smallest item is the GGUF of
MODELS = {"350M": ("LiquidAI/LFM2.5-350M", "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"),
          "230M": ("LiquidAI/LFM2.5-230M", "40cb2ad3b3044d5a41eee083a6103c8b523afa45"),
          "700M": ("LiquidAI/LFM2-700M", "86f49fc9a3800c3a325b7320bde179c318062583"),
          "1.2B": ("LiquidAI/LFM2.5-1.2B-Instruct", "0f604ada3f766f9f257460c4c9f0b5d6f69d431b"),
          "1.2B-JP": ("LiquidAI/LFM2.5-1.2B-JP-202606", "52b8b4475311a63bf839c6494f78f8ad59d13515")}
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja",
         "model.safetensors"]
BOS = 1  # <|startoftext|>: what the real tokenizer puts first, and the page
# the chat format of one turn, as chat_template.jinja writes it after its bos_token (the 350M's template calls macros,
# which the converter's reader does not read: the list has it by hand)
CHAT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
PROMPT = "What is the capital of Japan? Answer in one sentence."
SPECIALS = ["<|im_start|>", "<|im_end|>"]
STOPS = (BOS, 7)  # <|im_end|>
NEW_TOKENS = 16

say = lambda *parts: print("lfm2:", *parts, flush=True)

# What the engine and transformers hold after the same positions: per convolution layer the last taps - 1 tokens' values
# before the convolution; per attention layer its keys (as Hugging Face has them: the pairs of a head not interleaved)
# and values. STATES: what is compared.
STATES = ("convolution rows", "keys", "values")
STATE_LINE = 1e-4  # at least: ten times what transformers' own two computations differ by where that is more
LOGITS_LINE = 1e-3  # the same for the logits (the made-up models': 2e-3)


def engine_states(llama, count):
    """{layer: (kind, [arrays])}"""
    out = {}
    head = llama.head_size
    order = np.concatenate([np.arange(0, head, 2), np.arange(1, head, 2)])  # the engine's index of Hugging Face's 0..head-1
    for l, (short, a) in enumerate(llama.slots):
        if short:
            out[l] = ("conv", [llama.conv_state[a].T])
        else:
            out[l] = ("full", [llama.key_cache[a, :, :count][..., order], llama.value_cache[a, :, :count]])
    return out


def model_states(past, count):
    out = {}
    for l, layer in enumerate(past.layers):
        if getattr(layer, "conv_states", None) and layer.conv_states[0] is not None:
            out[l] = ("conv", [layer.conv_states[0][0, :, 1:].float().numpy()])
        else:
            out[l] = ("full", [layer.keys[0].float().numpy()[:, :count], layer.values[0].float().numpy()[:, :count]])
    return out


def state_distances(ours, theirs):
    """the largest relative distance over the layers, for each of STATES; a layer of another kind is infinitely far"""
    worst = dict.fromkeys(STATES, 0.0)
    for l, (kind, mine) in ours.items():
        names = STATES[:1] if kind == "conv" else STATES[1:]
        if theirs[l][0] != kind:
            for name in names:
                worst[name] = float("inf")
            continue
        for name, a, b in zip(names, mine, theirs[l][1]):
            worst[name] = max(worst[name], relative(a, b))
    return worst


def compare_states(label, llama, count, at_once, stepped):
    """the engine's states against transformers' after count positions: each within the line of 10 times what
    transformers' own two computations (at once, token by token) differ by, and 1e-4 at least. True where they are not"""
    apart = state_distances(engine_states(llama, count), stepped)
    floor = state_distances(at_once, stepped)
    failed, parts = False, []
    for name in STATES:
        line = max(STATE_LINE, 10 * floor[name])
        bad = not apart[name] <= line
        failed |= bad
        parts.append(f"{name} {apart[name]:.1e} (transformers' own {floor[name]:.1e}, line {line:.1e}){' — FAILED' if bad else ''}")
    say(f"{label}: the states after {count} positions against transformers': " + "; ".join(parts))
    return failed


def both_ways(model, ids):
    """transformers on ids: (logits at once, its states, logits token by token, its states)"""
    import torch

    tensor = torch.tensor([ids])
    with torch.no_grad():
        out = model(input_ids=tensor, use_cache=True)
        whole, at_once = out.logits[0].float().numpy(), model_states(out.past_key_values, len(ids))
        past, stepped = None, []
        for at in range(len(ids)):
            out = model(input_ids=tensor[:, at:at + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            stepped.append(out.logits[0, -1].float().numpy())
    return whole, at_once, np.stack(stepped), model_states(past, len(ids))


# --------------------------------------------------------------------------------------------- made-up models
# c: a convolution layer, a: an attention layer
MADE_UP = {
    "the 350M's order in small": dict(kinds="ccaccaca"),
    "a classifier of its own, four taps": dict(kinds="caccac", tie_word_embeddings=False, conv_L_cache=4),
    "the size of the FFN as it is, keys for every head": dict(kinds="ccacacac", block_auto_adjust_ff_dim=False,
                                                              intermediate_size=64, num_key_value_heads=4),
    "the layers by full_attn_idxs, two taps, another epsilon": dict(full_attn_idxs=[0, 3, 4], layers=7, conv_L_cache=2,
                                                                    norm_eps=1e-6),
}


def made_up(name, settings, positions=80):
    import torch
    from conftest import naive_lfm2_logits
    from test_convert import safetensors_file
    from transformers import Lfm2Config, Lfm2ForCausalLM

    settings = dict(settings)
    kinds = settings.pop("kinds", None)
    layers = settings.pop("layers", len(kinds or ""))
    said = dict(intermediate_size=96, num_key_value_heads=2, block_multiple_of=32)
    said.update(settings)
    if kinds:
        said["layer_types"] = ["conv" if kind == "c" else "full_attention" for kind in kinds]
    config = Lfm2Config(vocab_size=320, hidden_size=32, num_hidden_layers=layers, num_attention_heads=4,
                        max_position_embeddings=256, bos_token_id=1, eos_token_id=7, pad_token_id=0,
                        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0}, **said)
    torch.manual_seed(260)
    model = Lfm2ForCausalLM(config).to(torch.float32).eval()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key.endswith("norm.weight"):
                parameter.copy_(1.0 + 0.3 * torch.randn_like(parameter))
            elif not (key == "lm_head.weight" and config.tie_word_embeddings):
                parameter.copy_(0.3 * torch.randn_like(parameter))
    if config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight, "the classifier is not tied"
    rng = np.random.default_rng(260)
    tokens = [int(token) for token in rng.integers(0, 320, positions)]
    whole, at_once, stepped, stepped_states = both_ways(model, tokens)
    tensors = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    if config.tie_word_embeddings:
        tensors.pop("lm_head.weight", None)
    published = json.loads(config.to_json_string(use_diff=False))
    # the naive reference reads the config as the unit tests' own (conftest.lfm2_model) says it
    naive = naive_lfm2_logits(tensors, {**published, "norm_eps": config.norm_eps, "conv_L_cache": config.conv_L_cache,
                                        "layer_types": list(config.layer_types),
                                        "rope_parameters": {"rope_theta": 1000000.0}}, tokens)

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
    floor = differences(stepped, whole)[0]
    line = max(2 * LOGITS_LINE, 10 * floor)
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
    failed |= compare_states(f"made-up ({name})", llama, positions, at_once, stepped_states)
    hidden = int(struct.unpack_from("<7i", conversion.checkpoint, 0)[1])
    theirs = model.model.layers[0].feed_forward.w1.weight.shape[0]
    failed |= hidden != theirs
    say(f"made-up ({name}): the FFN's inside {hidden} (transformers' {theirs}{'' if hidden == theirs else ' — FAILED'}), options "
        f"{json.dumps({key: options[key] for key in ('arch', 'convolution', 'rope_theta', 'rms_norm_eps') if key in options})}")
    return failed


# ------------------------------------------------------------------------------------------------- the real model
LONG = 1024
LONG_AT = (0, 1, 2, 3, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1023)


def long_check(model, tokenizer, llama, line):
    import torch

    text = " ".join(f"Section {i}. {TEXT}" for i in range(12))
    ids = ([BOS] + tokenizer.encode(text, add_special_tokens=False).ids)[:LONG]
    at = [position for position in LONG_AT if position < len(ids)]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=torch.tensor(at))
    theirs, states = out.logits[0].float().numpy(), model_states(out.past_key_values, len(ids))
    began, kept = time.perf_counter(), {}
    for pos, token in enumerate(ids):
        logits = llama.forward(token, pos)
        if pos in at:
            kept[pos] = logits.copy()
    largest, mean, same, margin = differences(np.stack([kept[position] for position in at]), theirs)
    apart = state_distances(engine_states(llama, len(ids)), states)
    ok = largest <= line and (same == len(at) or margin <= 2 * largest)
    bad = [name for name in STATES if not apart[name] <= STATE_LINE]
    say(f"long: the engine ran {len(ids)} positions in {time.perf_counter() - began:.1f} s; the logits at {len(at)} of them against "
        f"transformers': largest difference {largest:.2e}, mean {mean:.2e}, the same most likely token at {same} of {len(at)}"
        f"{'' if ok else f' — FAILED (the line: {line:.1e})'}; the states after them: "
        + "; ".join(f"{name} {apart[name]:.1e}" for name in STATES) + (f" — FAILED (the line: {STATE_LINE:.0e}): {', '.join(bad)}" if bad else ""))
    return not ok or bool(bad)


def real(directory, positions, name, with_faults):
    import tokenizers
    import torch
    import transformers
    from transformers import Lfm2ForCausalLM

    repo, revision = MODELS[name]
    say(f"real: {repo}@{revision}, transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}")
    for file in FILES:
        fetch(file, directory, repo, revision)
    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))

    # the conversion, the way the page does it: the file in its own order, float32
    data = np.memmap(directory / "model.safetensors", dtype=np.uint8, mode="r")
    (length,) = np.frombuffer(bytes(data[:8]), dtype="<u8")
    first = 8 + int(length)
    sink = File(directory / "float32.bin")
    began = time.perf_counter()
    conversion = Conversion(bytes(data[8:first]).decode(), first, (directory / "config.json").read_text(),
                            (directory / "tokenizer.json").read_bytes(), "tokenizer.json", dtype="float32",
                            tokenizer_config=(directory / "tokenizer_config.json").read_text(), sink=sink, start=first,
                            chat_template=(directory / "chat_template.jinja").read_text())
    for start in range(first, len(data), CHUNK):
        conversion.feed(bytes(data[start:start + CHUNK]))
    conversion.finish()
    sink.data.flush()
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    shown = {key: (value if key != "specials" else f"{len(value)} of them") for key, value in options.items()}
    say(f"real: converted in {time.perf_counter() - began:.1f} s, {sink.path.stat().st_size} bytes, options {json.dumps(shown)}")
    say(f"real: the converter read {'no template' if 'template' not in conversion.options else 'a template: ' + json.dumps(conversion.options['template'])}")
    specials = sorted(set(options.get("specials", [])) | set(SPECIALS), key=lambda token: (-len(token), token))
    checkpoint = np.memmap(sink.path, dtype=np.uint8, mode="r")

    def make(**other):
        return Llama(checkpoint, conversion.tokenizer, kernels=None, **{**options, "specials": specials, **other})

    llama = make()
    failed = tuple(sorted(llama.stop_tokens)) != STOPS or llama.bos != BOS
    say(f"real: the BOS is {llama.bos} and the answer stops at {sorted(llama.stop_tokens)}{' — FAILED' if failed else ''}")

    # the same ids for both: the BOS the real tokenizer puts first (and the page), then the text as the real tokenizer
    # splits it (and the engine's own tokenizer has to split it the same)
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
    failed |= template != "<|startoftext|>" + chat
    say(f"real: the model's own template writes {json.dumps(template)}: "
        f"{'the format here after its BOS' if template == '<|startoftext|>' + chat else 'ANOTHER FORMAT — FAILED'}")
    ids = ids[:positions]

    model = Lfm2ForCausalLM.from_pretrained(str(directory), dtype=torch.float32).eval()
    began = time.perf_counter()
    whole, at_once, stepped, stepped_states = both_ways(model, ids)
    say(f"real: transformers, {len(ids)} positions at once and token by token in {time.perf_counter() - began:.1f} s")
    with torch.no_grad():
        generated = model.generate(torch.tensor([[BOS] + chat_ids]), max_new_tokens=NEW_TOKENS, do_sample=False,
                                   repetition_penalty=1.0)[0].tolist()
    theirs = generated[1 + len(chat_ids):]

    began = time.perf_counter()
    ours = [llama.forward(token, pos).copy() for pos, token in enumerate(ids)]
    say(f"real: the engine (NumPy, float32), {len(ids)} positions in {time.perf_counter() - began:.1f} s")
    for what, logits in (("transformers", whole), ("the engine", ours)):
        logits = np.asarray(logits)
        say(f"real: logits of {what}: min {logits.min():.3f}, max {logits.max():.3f}, mean {logits.mean():.4f}, "
            f"std {logits.std():.4f}")
    floor = differences(stepped, whole)[0]
    say(f"real: transformers token by token against transformers at once: largest difference {floor:.2e}")
    # the line: ten times what transformers' own two computations differ by, and no less than 1e-3; the most likely
    # token the same wherever the reference's best two are further apart than twice the difference. The faults below
    # say what it catches
    line = max(LOGITS_LINE, 10 * floor)
    for what, reference in (("transformers at once", whole), ("transformers token by token", stepped)):
        largest, mean, agree, margin = differences(ours, reference)
        ok = largest <= line and (agree == len(ids) or margin <= 2 * largest)
        failed |= not ok
        say(f"real: the engine against {what}: largest difference {largest:.2e}, mean {mean:.2e}, the same most "
            f"likely token at {agree} of {len(ids)} positions (the largest gap between the reference's best two where "
            f"it is not: {margin:.2e}){'' if ok else f' — FAILED (the line: {line:.1e})'}")
    failed |= compare_states("real", llama, len(ids), at_once, stepped_states)
    state_floor = state_distances(at_once, stepped_states)

    # 16 greedy tokens for the chat prompt, by transformers' generate() and by the engine's
    text = "".join(llama.generate(chat, steps=len(chat_ids) + NEW_TOKENS, temperature=0.0, echo=False))
    wrote = tokenizer.decode(theirs, skip_special_tokens=False)
    say(f"real: transformers wrote {json.dumps(wrote, ensure_ascii=False)} {theirs}")
    say(f"real: the engine wrote {json.dumps(text, ensure_ascii=False)}")
    stop = next((at for at, token in enumerate(theirs) if token in STOPS), len(theirs))
    same = tokenizer.decode(theirs[:stop], skip_special_tokens=False) == text
    failed |= not same
    say(f"real: the two wrote {'the same' if same else 'OTHER TEXTS — FAILED'}")
    failed |= long_check(model, tokenizer, llama, line)
    del model
    gc.collect()
    if with_faults:
        failed |= faults(make, ids, whole, stepped_states, line, state_floor)
    del llama, checkpoint
    sink.path.unlink()
    return failed


# ------------------------------------------------------------------------------------------------------- faults
# The engine with one thing wrong, each a function that takes a new engine of the real model (and what makes one) and
# returns the engine to run. What the right engine is held to must let none of them by.
def patched(llama, **methods):
    for name, method in methods.items():
        setattr(llama, name, method)
    return llama


def taps_reversed(llama, make):
    llama.conv = llama.conv[:, ::-1]
    return llama


def taps_of_the_next_layer(llama, make):
    llama.conv = np.roll(llama.conv, -1, axis=0)
    return llama


def oldest_tap_left_out(llama, make):
    llama.conv = np.array(llama.conv)
    llama.conv[:, 0] = 0
    return llama


def silu_after_the_convolution(llama, make):
    right = llama.convolved
    return patched(llama, convolved=lambda a, values: silu(right(a, values)))


def sigmoid_of_the_gate(llama, make):
    dim = llama.dim

    def short(a, xb):
        mixed = llama.win[a] @ xb
        return llama.wout[a] @ (mixed[dim:2 * dim] * llama.convolved(a, mixed[2 * dim:] / (1.0 + np.exp(-mixed[:dim]))))
    return patched(llama, short_convolution=short)


def gates_in_each_others_place(llama, make):
    dim = llama.dim

    def short(a, xb):
        mixed = llama.win[a] @ xb
        return llama.wout[a] @ (mixed[:dim] * llama.convolved(a, mixed[dim:2 * dim] * mixed[2 * dim:]))
    return patched(llama, short_convolution=short)


def a_token_late(llama, make):
    """the padding off by one: the convolution reads the taps - 1 tokens before this one and one more, not this one"""
    right, held = llama.convolved, {}

    def convolved(a, values):
        out = right(a, held.get(a, np.zeros_like(values)))
        held[a] = values.copy()
        return out
    return patched(llama, convolved=convolved)


def one_state_for_all_layers(llama, make):
    def convolved(a, values):
        taps, before = llama.conv[a], llama.conv_state[0]
        out = (taps[:-1] * before).sum(axis=0) + taps[-1] * values
        before[:-1] = before[1:]
        before[-1] = values
        return out
    return patched(llama, convolved=convolved)


def state_never_cleared(llama, make):
    """position 0 does not clear the state: the run begins on what another text left"""
    for pos, token in enumerate((BOS, 500, 600, 700)):
        llama.forward(token, pos, need_logits=False)
    return patched(llama, follow=lambda pos: None)


def epsilon(value):
    def fault(llama, make):
        llama.rms_norm_eps = value
        return llama
    return fault


def two_layers_change_kinds(llama, make):
    """a convolution layer and the attention layer after it in each other's place (the same tensors, as many of each)"""
    layers = llama.convolution["layers"]
    at = layers.index("ca")
    return make(convolution={**llama.convolution, "layers": layers[:at] + "ac" + layers[at + 2:]})


def head_norms_change_places(llama, make):
    llama.q_norm, llama.k_norm = llama.k_norm, llama.q_norm
    return llama


def no_norm_of_the_keys_heads(llama, make):
    llama.k_norm = np.ones_like(llama.k_norm)
    return llama


def layer_norms_change_places(llama, make):
    llama.rms_att_weight, llama.rms_ffn_weight = llama.rms_ffn_weight, llama.rms_att_weight
    return llama


def ffn_gate_and_up_change_places(llama, make):
    llama.w1, llama.w3 = llama.w3, llama.w1
    return llama


def theta(value):
    def fault(llama, make):
        angles = np.arange(llama.seq_len)[:, None] * rope_frequencies(llama.head_size, value)
        llama.freq_cis_real, llama.freq_cis_imag = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        return llama
    return fault


FAULTS = {
    "the taps reversed (the newest tap on the oldest token)": taps_reversed,
    "the taps of the next convolution layer": taps_of_the_next_layer,
    "the oldest tap left out": oldest_tap_left_out,
    "SiLU after the convolution (a Qwen3.5's)": silu_after_the_convolution,
    "the sigmoid of the gate B": sigmoid_of_the_gate,
    "the gates B and C in each other's place": gates_in_each_others_place,
    "the convolution a token late (its padding off by one)": a_token_late,
    "one state for all the convolution layers": one_state_for_all_layers,
    "the state not cleared at position 0": state_never_cleared,
    "the norms' epsilon 1e-6": epsilon(1e-6),
    "the norms' epsilon 1e-4": epsilon(1e-4),
    "the norms' epsilon 0": epsilon(0.0),
    "a convolution layer and the attention layer after it change kinds": two_layers_change_kinds,
    "the norms of q's heads and of k's in each other's place": head_norms_change_places,
    "no norm of k's heads": no_norm_of_the_keys_heads,
    "the operator's norm and the FFN's in each other's place": layer_norms_change_places,
    "the FFN's gate and up in each other's place": ffn_gate_and_up_change_places,
    "RoPE's theta 1e7": theta(1e7),
    "RoPE's theta 1e4": theta(1e4),
}


def faults(make, ids, whole, states, line, state_floor):
    """Every fault's distance from transformers' numbers, and whether what the right engine is held to (the logits
    within line, the states within theirs) catches it. True where one gets by."""
    lines = {name: max(STATE_LINE, 10 * state_floor[name]) for name in STATES}
    slipped, weakest = [], None
    for name, fault in FAULTS.items():
        llama = fault(make(), make)
        ours = [llama.forward(token, pos).copy() for pos, token in enumerate(ids)]
        largest, _, agree, _ = differences(ours, whole)
        apart = state_distances(engine_states(llama, len(ids)), states)
        # how many times past its line the fault is, by the logits and by the states: what catches it is the larger
        by_logits = largest / line if np.isfinite(largest) else float("inf")
        by_states = max((apart[state] / lines[state] if np.isfinite(apart[state]) else float("inf")) for state in STATES)
        caught = not largest <= line or any(not apart[state] <= lines[state] for state in STATES)
        if not caught:
            slipped.append(name)
        margin = max(by_logits, by_states)
        if weakest is None or margin < weakest[1]:
            weakest = (name, margin)
        say(f"fault ({name}): logits {largest:.2e} from transformers' ({by_logits:.1f} times the line), the same most likely token "
            f"at {agree} of {len(ids)}; states " + ", ".join(f"{state} {apart[state]:.1e}" for state in STATES)
            + f" ({by_states:.1f} times their line){'' if caught else ' — NOT CAUGHT, FAILED'}")
    say(f"faults: {len(FAULTS) - len(slipped)} of {len(FAULTS)} are past the line (logits {line:.1e}; states "
        + ", ".join(f"{state} {lines[state]:.1e}" for state in STATES)
        + f"); the weakest is \"{weakest[0]}\", {weakest[1]:.1f} times past it{'' if not slipped else ' — FAILED: ' + '; '.join(slipped)}")
    return bool(slipped)


def main():
    directory = Path(sys.argv[1])
    option = lambda name, default=None: next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith(f"--{name}=")), default)
    only, name, positions = option("only"), option("model", "350M"), int(option("positions", 96))
    if only == "fetch":
        for file in FILES:
            fetch(file, directory / name, *MODELS[name])
        return
    failed = False
    if only in (None, "made-up"):
        for label, settings in MADE_UP.items():
            failed |= made_up(label, settings)
    if only in (None, "real"):
        failed |= real(directory / name, positions, name, "--no-faults" not in sys.argv)
    say("FAILED" if failed else "the engine computes what transformers computes")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
