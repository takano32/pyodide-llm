# reference_llama.py
# T253 and T254: a model of the list that is a Llama to the engine, against transformers itself, on the original's
# safetensors. PyTorch is not on the development machine, so this runs in CI:
#
#   node tests/ci.mjs run tests.yml extra="bash tests/reference_llama.sh hf-granite-4.2-3b" --ref <branch> --grep "reference:"
#   python tests/reference_llama.py <directory for the downloads> [--only=made-up] [<id of the list> ...] [--positions=96]
#                                   [--weak] [--layers=N] [--gguf]
#
# --weak (the review of T253): after the right engine has been held to transformers, weak faults are put into its weights
# (q of every layer 0.1%, 0.01% and 0.001% off, one head's 1% and 10%, one layer's q left unmultiplied, one key head's 1%)
# and each is held the same way, to say what the check sees: by the logits (the line below) and by the states of every layer
# (the keys and values each layer wrote, relative to the layer's largest: the right engine is a few 1e-6 from transformers',
# a fault of one head of one layer is far more of them than of the logits, which come after 24 to 42 layers). --layers=N:
# only the first N layers of a real model, on both sides, for a model whose float32 a runner has not room for. --gguf: the
# engine's weights from the entry's GGUF (Q8_0, the original's vocabulary and config.json), as the page reads the list's
# models, against transformers on the original's weights: the route the visitors take, held to a line that Q8_0's rounding
# allows (GGUF_LINE below), which sees a fault that changes the model and none of the weak ones.
#
# Two parts, every line of the log beginning with "reference:".
#   made-up: tiny random Granites of transformers' own class (GraniteForCausalLM), whose attention multiplies its
#     scores by attention_multiplier (T253: the converter puts it into q, llama2_convert.query_scale). transformers'
#     logits, over the whole text at once and token by token with its cache, against the engine converted the way the
#     page converts. And the same tensors converted as a Llama's, which must be another model: the comparison sees the
#     multiplier. A Granite 4.1's other multipliers are refused.
#   an id of the list (src/models.js): its original repository at the list's revision. The text as the real
#     tokenizer splits it against the engine's tokenizer, the chat prompt as the list's format writes it against
#     apply_chat_template, the logits of the same ids through transformers and through the engine (NumPy, float32,
#     converted from the safetensors in the file's own order, shards joined as the page joins them), and 16 greedy
#     tokens of each for the chat prompt.
# transformers computes in float32 here whatever the weights are stored as: the model is loaded as stored (bfloat16:
# Granite 4.2 3B is 7.3 GB so, and 14.6 GB as float32, which a runner of 16 GB has not), every nn.Linear widens its
# weight for its own product, and the embeddings are widened before they go in. A bfloat16 is a float32 to the bit, so
# the arithmetic is the float32 one of the same weights; the modeling code is transformers' own.
# The lines the engine is held to are at the end of each part; anything past them is exit 1.
import gc
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
import llama2_convert  # noqa: E402
from llama2_convert import Conversion  # noqa: E402
from llama2_numpy import Llama  # noqa: E402
from conducting import Fetched, Mapped, converted  # noqa: E402
from convert.conduct import candidates_of  # noqa: E402
from fetching import download  # noqa: E402

# more than 64 tokens, English and Japanese, with numbers after one space and after several (T254: where MiniCPM5's
# two-stage split is not Llama 3's), a number of more than three digits and a contraction
TEXT = ("Mount Fuji is the highest mountain in Japan, standing 3,776 metres above sea level on the island of Honshu. "
        "It's an active stratovolcano that last erupted from 1707 to  1708, about   100 kilometres from Tokyo; in 2013 "
        "it became a World Heritage Site. 富士山は日本で最も高い山で、標高は 3776 メートルです。古くから信仰の対象とされ、"
        "多くの絵画や文学作品に描かれてきました。夏には多くの登山者が山頂を目指します。")
PROMPT = "What is the capital of Japan? Answer in one sentence."
NEW_TOKENS = 16
STEPPED = 32  # positions transformers also computes token by token with its cache, for the floor of the line
# --weak: faults put into the engine's weights, to see what the check sees (weak_errors); the made-up Granites' take seconds,
# the real model's about as long as its own forward pass, 96 positions, each
WEAK = "--weak" in sys.argv
# --gguf: the real model's weights from the list's GGUF, as the page reads them (Q8_0), against transformers on the original's.
# Q8_0 rounds every weight (about half a step of 1/127 of its group's largest: 0.3 to 0.7% of a typical weight), so the lines are
# of another kind than the float32 ones: a fault that changes the model (q scaled twice or not at all, q left turned) is a logit or
# several, the rounding some hundredths (a logit of 36 at most here), and the keys and values of a layer differ by about a percent
GGUF = "--gguf" in sys.argv
# (the logits' line was 1.0 until T255: SmolLM3 3B, 36 layers, is 1.37 at most and 9.4e-2 on average from a GGUF whose every
# tensor is the original's Q8_0 to the bit, where its float32 is 1.2e-4. What the line is to see is in unturned_faults())
GGUF_LINE, GGUF_STATE_LINE = 2.0, 0.1

say = lambda *parts: print("reference:", *parts, flush=True)


def differences(ours, theirs):
    """(largest difference, mean difference, positions whose most likely token is the same, the largest gap between
    the two most likely tokens of theirs where the most likely token differs)"""
    ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    gap = np.abs(ours - theirs)
    same = ours.argmax(axis=1) == theirs.argmax(axis=1)
    top = np.sort(theirs, axis=1)[:, -2:]
    margins = (top[:, 1] - top[:, 0])[~same]
    return float(gap.max()), float(gap.mean()), int(same.sum()), float(margins.max()) if margins.size else 0.0


def float32_arithmetic():
    """Every nn.Linear of transformers computes in float32 from here on (see the head of this file)."""
    import torch

    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(torch.float32)
        return torch.nn.functional.linear(x.to(torch.float32), self.weight.to(torch.float32), bias)

    torch.nn.Linear.forward = forward


def cache_arrays(cache):
    """[(keys, values)] of a transformers cache: float32 arrays (key-value heads, positions, head size), a pair for each
    layer"""
    import torch
    layers = getattr(cache, "layers", None)
    pairs = [(layer.keys, layer.values) for layer in layers] if layers is not None else list(zip(cache.key_cache, cache.value_cache))
    return [(k[0].to(torch.float32).numpy(), v[0].to(torch.float32).numpy()) for k, v in pairs]


def logits_of(model, ids, stepped=0, states=False):
    """transformers' logits for these ids at once, and for the first `stepped` of them token by token with its cache.
    states: and the keys and values of every layer, of the whole pass and of the stepped one (cache_arrays)"""
    import torch
    with torch.no_grad():
        embedded = model.get_input_embeddings()(torch.tensor([ids])).to(torch.float32)
        out = model(inputs_embeds=embedded, use_cache=states)
        whole = out.logits[0].to(torch.float32).numpy()
        kept = cache_arrays(out.past_key_values) if states else None
        past, steps = None, []
        for at in range(min(stepped, len(ids))):
            step = model(inputs_embeds=embedded[:, at:at + 1], past_key_values=past, use_cache=True)
            past = step.past_key_values
            steps.append(step.logits[0, -1].to(torch.float32).numpy())
    return (whole, steps, kept, cache_arrays(past) if states and past is not None else None) if states else (whole, steps)


# ------------------------------------------------------------------------------------------------ a layer at a time
class ByLayer:
    """transformers' model of the original in a directory, a layer at a time: for a model whose weights a runner has not
    room for even as stored (Granite 4.2 8B is 16.8 GB in bfloat16 and a runner of 16 GB cannot load it), as the review of
    T246 held a Ternary Bonsai 8B to transformers. A model of the same config with ONE layer, into which the weights of
    each layer of the original are read from the safetensors in turn (bfloat16 to float32 is exact: the arithmetic is the
    float32 model's); the embedding, the rotary embedding, the layer, the last norm and the classifier are transformers'
    own modules, and the loop over the layers, the causal mask and what a Granite's model does outside its layers (the
    embedding multiplier before the first, the logits' scaling after the classifier) are this class's. Only for a model
    whose layers are all of one kind: a hybrid (Qwen3.5's linear and full layers) fails at the strict load of a layer of
    the other kind, loudly."""

    def __init__(self, directory):
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM
        directory = Path(directory)
        config = AutoConfig.from_pretrained(directory)
        self.config, self.layers, config.num_hidden_layers = config, config.num_hidden_layers, 1
        if getattr(config, "layer_types", None):
            config.layer_types = config.layer_types[:1]
        try:
            self.model = AutoModelForCausalLM.from_config(config, dtype=torch.float32).eval()
        except TypeError:  # transformers before 4.56 calls it torch_dtype
            self.model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32).eval()
        index = directory / "model.safetensors.index.json"
        names = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
        self.where = {key: file for file in [safe_open(directory / name, "pt") for name in names] for key in file.keys()}
        self.embedding = float(getattr(config, "embedding_multiplier", 1.0))
        self.scaling = float(getattr(config, "logits_scaling", 1.0))
        self.put(self.model.model.embed_tokens, "model.embed_tokens.")
        self.put(self.model.model.norm, "model.norm.")
        if "lm_head.weight" in self.where:
            self.put(self.model.lm_head, "lm_head.")
        else:
            self.model.lm_head.weight = self.model.model.embed_tokens.weight
        say(f"by layer: {self.layers} layers from {len(names)} file(s), {len(self.where)} tensors; the classifier is "
            f"{'its own' if 'lm_head.weight' in self.where else 'the embedding'}")

    def put(self, module, prefix):
        """read the weights of `module` from the tensors named by `prefix` (as float32)"""
        import torch
        module.load_state_dict({key: self.where[prefix + key].get_tensor(prefix + key).to(torch.float32)
                                for key in module.state_dict()}, strict=True)

    def hidden(self, rows):
        """The hidden states after the last norm of each of these lists of ids, all through each layer together (one
        reading of the layer's weights): a list of tensors (positions, width). A shorter list is padded on the right with
        its own last token, which no earlier position sees."""
        import torch
        longest = max(len(row) for row in rows)
        ids = torch.tensor([row + [row[-1]] * (longest - len(row)) for row in rows])
        positions = torch.arange(longest)[None]
        model = self.model.model
        with torch.no_grad():
            hidden = model.embed_tokens(ids) * self.embedding
            embeddings = model.rotary_emb(hidden, positions)
            mask = torch.full((longest, longest), torch.finfo(hidden.dtype).min).triu(1)[None, None]
            layer = model.layers[0]
            for number in range(self.layers):
                self.put(layer, f"model.layers.{number}.")
                out = layer(hidden, attention_mask=mask, position_ids=positions, position_embeddings=embeddings)
                hidden = out[0] if isinstance(out, tuple) else out
            hidden = model.norm(hidden)
        return [hidden[at, :len(row)] for at, row in enumerate(rows)]

    def logits(self, hidden):
        """the logits of hidden states (positions, width) from `hidden()`"""
        import torch
        with torch.no_grad():
            return self.model.lm_head(hidden.to(torch.float32)) / self.scaling


# ------------------------------------------------------------------------------------------ the states of every layer
# The logits are the end of 24 to 42 layers' work, and a fault of one head of one layer is a few ten-thousandths of
# them (a 1% fault of one head's scale of Granite 4.2 3B: 3e-4 of a logit, the line is 1e-3). The keys and values every
# layer writes are what the next layer saw: a fault shows in them a layer sooner and not diluted by what comes after
# (the review of T253; T229's review found the same of the states of a Qwen3.5's layers).
def engine_cache(llama, count):
    """The engine's keys and values of the first `count` positions as transformers holds them: for each layer, (key heads,
    count, head size). The engine turns RoPE's pairs adjacent (the first of a pair is the first half of a head in
    transformers' order and the second the second half), and the values it holds as they are."""
    keys = llama.key_cache[:, :, :count]
    layers, heads, positions, size = keys.shape
    turned = keys.reshape(layers, heads, positions, size // 2, 2)
    keys = np.concatenate([turned[..., 0], turned[..., 1]], axis=-1)
    return [(keys[layer], llama.value_cache[layer, :, :count].copy()) for layer in range(layers)]


def state_gap(ours, theirs):
    """(the largest difference of the keys, of the values) over the layers, each layer's relative to its own largest
    value of theirs: with the layer it is in, and the largest key and value of all. NaN is as far as can be."""
    gaps = {"keys": [], "values": []}
    top = {"keys": 0.0, "values": 0.0}
    for (our_k, our_v), (their_k, their_v) in zip(ours, theirs):
        for name, a, b in (("keys", our_k, their_k), ("values", our_v, their_v)):
            largest = float(np.abs(b).max())
            top[name] = max(top[name], largest)
            gap = float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())
            gaps[name].append(float("inf") if gap != gap else gap / largest if largest > 0 else gap)
    worst = {name: max(range(len(values)), key=values.__getitem__) for name, values in gaps.items()}
    return {name: (gaps[name][worst[name]], worst[name]) for name in gaps}, top


def state_line(floor):
    """What the engine's states may be from transformers': ten times what transformers' own two computations differ by
    (relative, as state_gap), and no less than 1e-4"""
    return max(1e-4, 10 * max(floor["keys"], floor["values"]))


def scaled(llama, name, layers, rows, factor):
    """A fault in the engine's weights: the rows of matrix `name` (wq or wk) in these layers times factor, in a copy that
    stands in for the matrix. Returns what puts the matrix back."""
    original = getattr(llama, name)
    changed = np.array(original, dtype=np.float32)
    for layer in layers:
        changed[layer, rows] *= np.float32(factor)
    setattr(llama, name, changed)
    return lambda: setattr(llama, name, original)


def weak_errors(label, llama, tokens, whole, theirs, line, lined, scale):
    """The faults a conversion of a Granite could have that are weak (T253's review): what q was multiplied by 0.1%, 0.01%
    and 0.001% off, one head's, one layer's q left as it was, one key head's scale 1% off. Each put into the engine's
    weights and held to transformers' as the real engine is: by the logits and by the states of every layer. Returns the
    faults the check did not see among those it must (the first, one head 10%, a layer unscaled, a key head 1% off)."""
    layers, size = llama.wq.shape[0], llama.head_size
    middle, head, key_head = layers // 2, max(llama.n_heads // 3, 0), llama.n_kv_heads // 2
    rows = lambda at: slice(at * size, (at + 1) * size)
    every = range(layers)
    faults = [("q of every layer 0.1% off", lambda: scaled(llama, "wq", every, slice(None), 1.001), True),
              ("q of every layer 0.01% off", lambda: scaled(llama, "wq", every, slice(None), 1.0001), False),
              ("q of every layer 0.001% off", lambda: scaled(llama, "wq", every, slice(None), 1.00001), False),
              (f"q of one head (head {head}, layer {middle}) 1% off", lambda: scaled(llama, "wq", [middle], rows(head), 1.01), False),
              (f"q of one head (head {head}, layer {middle}) 10% off", lambda: scaled(llama, "wq", [middle], rows(head), 1.1), True),
              (f"k of one key head (head {key_head}, layer {middle}) 1% off", lambda: scaled(llama, "wk", [middle], rows(key_head), 1.01), True)]
    if scale != 1.0:
        faults.insert(4, (f"q of layer {middle} left as it was (not multiplied by {scale:.6g})",
                          lambda: scaled(llama, "wq", [middle], slice(None), 1 / scale), True))
    missed = []
    for what, put, must in faults:
        restore = put()
        try:
            ours = [llama.forward(token, pos).copy() for pos, token in enumerate(tokens)]
            states = engine_cache(llama, len(tokens))
        finally:
            restore()
        largest, mean, same, margin = differences(ours, whole)
        gaps, _ = state_gap(states, theirs)
        by_logits = not (largest <= line and (same == len(tokens) or margin <= 2 * largest))
        by_states = not (max(gaps["keys"][0], gaps["values"][0]) <= lined)
        seen = by_logits or by_states
        if must and not seen:
            missed.append(what)
        say(f"{label}: a fault, {what}: logits largest difference {largest:.2e} ({largest / line:.2g} of the line), the same most "
            f"likely token at {same} of {len(tokens)}; states: keys {gaps['keys'][0]:.2e} (layer {gaps['keys'][1]}), values "
            f"{gaps['values'][0]:.2e} (layer {gaps['values'][1]}), {max(gaps['keys'][0], gaps['values'][0]) / lined:.2g} of the line; "
            f"seen by {'both' if by_logits and by_states else 'the logits' if by_logits else 'the states' if by_states else 'neither'}"
            f"{'' if seen or not must else ' — FAILED'}")
    return missed


def unturned_faults(label, llama, tokens, whole, theirs, line, lined):
    """The faults of a model with layers RoPE leaves alone (T255, SmolLM3): every layer turned (the model read as a
    Llama), one of its layers turned, one layer more left alone. Each put into the engine and held to transformers' as the
    real engine is; all three are another model, so the check must see each. Returns those it did not see."""
    held, middle = llama.unturned, llama.unturned[len(llama.unturned) // 2]
    spare = next(layer for layer in range(llama.n_layers) if layer not in held)
    faults = [("every layer turned (read as a Llama)", ()),
              (f"layer {middle} turned", tuple(layer for layer in held if layer != middle)),
              (f"layer {spare} left alone too", tuple(sorted(held + (spare,))))]
    missed = []
    for what, unturned in faults:
        llama.unturned = unturned
        try:
            ours = [llama.forward(token, pos).copy() for pos, token in enumerate(tokens)]
            states = engine_cache(llama, len(tokens))
        finally:
            llama.unturned = held
        largest, mean, same, margin = differences(ours, whole)
        gaps, _ = state_gap(states, theirs)
        by_logits = not (largest <= line and (same == len(tokens) or margin <= 2 * largest))
        by_states = not (max(gaps["keys"][0], gaps["values"][0]) <= lined)
        if not (by_logits or by_states):
            missed.append(what)
        say(f"{label}: a fault, {what}: logits largest difference {largest:.2e} ({largest / line:.2g} of the line), mean "
            f"{mean:.2e}, the same most likely token at {same} of {len(tokens)}; states: keys {gaps['keys'][0]:.2e} (layer "
            f"{gaps['keys'][1]}), values {gaps['values'][0]:.2e} (layer {gaps['values'][1]}), "
            f"{max(gaps['keys'][0], gaps['values'][0]) / lined:.2g} of the line; seen by "
            f"{'both' if by_logits and by_states else 'the logits' if by_logits else 'the states' if by_states else 'neither — FAILED'}")
    return missed


def greedy(model, ids, count, stops):
    """count tokens after ids, the most likely one each time, by transformers with its cache; stops at a stop token"""
    import torch
    written = []
    with torch.no_grad():
        embed = lambda tokens: model.get_input_embeddings()(torch.tensor([tokens])).to(torch.float32)
        out = model(inputs_embeds=embed(ids), use_cache=True)
        for _ in range(count):
            token = int(out.logits[0, -1].argmax())
            if token in stops:
                break
            written.append(token)
            out = model(inputs_embeds=embed([token]), past_key_values=out.past_key_values, use_cache=True)
    return written


# --------------------------------------------------------------------------------------------- made-up Granites
# heads of 16 in a hidden size of 64: 0.3 * 4 and 0.0078125 * 4 are what q is multiplied by, the second a power of two
MADE_UP = {
    "grouped queries": dict(attention_multiplier=0.3, num_key_value_heads=2),
    "a power of two, tied": dict(attention_multiplier=0.0078125, num_key_value_heads=4, tie_word_embeddings=True),
    "one key head": dict(attention_multiplier=0.11, num_key_value_heads=1, num_hidden_layers=4),
}


def made_up_conversion(tensors, published):
    from test_convert import safetensors_file
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(published["vocab_size"])]}}).encode()
    conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary, "tokenizer.json",
                            dtype="float32", max_seq_len=256, start=8 + size)
    conversion.feed(file[8 + size:])
    conversion.finish()
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    return Llama(bytes(conversion.checkpoint), conversion.tokenizer, **options), options


def made_up(name, settings, positions=80):
    import torch
    from transformers import GraniteConfig, GraniteForCausalLM

    config = GraniteConfig(**{**dict(vocab_size=320, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                                     num_attention_heads=4, max_position_embeddings=256, rms_norm_eps=1e-5,
                                     rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                                     bos_token_id=1, eos_token_id=2), **settings})
    torch.manual_seed(253)
    model = GraniteForCausalLM(config).to(torch.float32).eval()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key == "lm_head.weight" and config.tie_word_embeddings:
                continue
            scale = 0.1 if key.endswith("norm.weight") else 0.3
            parameter.copy_((1.0 if key.endswith("norm.weight") else 0.0) + scale * torch.randn_like(parameter))
    rng = np.random.default_rng(253)
    tokens = [int(token) for token in rng.integers(0, 320, positions)]
    whole, stepped, cache_whole, cache_stepped = logits_of(model, tokens, positions, states=True)
    tensors = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    if config.tie_word_embeddings:
        tensors.pop("lm_head.weight", None)
    published = json.loads(config.to_json_string(use_diff=False))
    assert published["model_type"] == "granite" and published["attention_multiplier"] == settings["attention_multiplier"]

    llama, options = made_up_conversion(tensors, published)
    ours = [llama.forward(token, pos).copy() for pos, token in enumerate(tokens)]
    plain, _ = made_up_conversion(tensors, {**published, "model_type": "llama"})
    as_a_llama = [plain.forward(token, pos).copy() for pos, token in enumerate(tokens)]

    failed = False
    floor = differences(stepped, whole)[0]
    line = max(1e-3, 10 * floor)
    say(f"made-up ({name}), transformers token by token against transformers at once: largest difference {floor:.2e}")
    for what, reference in (("transformers at once", whole), ("transformers token by token", stepped)):
        largest, mean, same, margin = differences(ours, reference)
        ok = largest <= line and (same == positions or margin <= 2 * largest)
        failed |= not ok
        say(f"made-up ({name}), the engine against {what}: largest difference {largest:.2e}, mean {mean:.2e}, the same "
            f"most likely token at {same} of {positions} positions{'' if ok else f' — FAILED (the line: {line:.1e})'}")
    # the states of every layer (the keys and values), against the whole pass and against the floor of transformers' own two
    floors, _ = state_gap(cache_stepped, cache_whole)
    lined = state_line({kind: gap for kind, (gap, _) in floors.items()})
    theirs, top = state_gap(engine_cache(llama, positions), cache_whole)
    ok = max(theirs["keys"][0], theirs["values"][0]) <= lined
    failed |= not ok
    say(f"made-up ({name}), states of every layer, the engine against transformers: keys {theirs['keys'][0]:.2e} (layer "
        f"{theirs['keys'][1]}), values {theirs['values'][0]:.2e} (layer {theirs['values'][1]}), relative to each layer's largest "
        f"(keys up to {top['keys']:.3g}, values {top['values']:.3g}); transformers' own two: keys {floors['keys'][0]:.2e}, values "
        f"{floors['values'][0]:.2e}; the line {lined:.1e}{'' if ok else ' — FAILED'}")
    if WEAK:
        failed |= bool(weak_errors(f"made-up ({name})", llama, tokens, whole, cache_whole, line, lined, llama2_convert.query_scale(published)))
    largest, mean, same, _ = differences(as_a_llama, whole)
    seen = largest > 100 * line
    failed |= not seen
    say(f"made-up ({name}), the same tensors converted as a Llama's against transformers: largest difference "
        f"{largest:.2e}, the same most likely token at {same} of {positions}"
        f"{' (another model, as it must be)' if seen else ' — FAILED: the comparison does not see the multiplier'}")
    say(f"made-up ({name}): what q was multiplied by {llama2_convert.query_scale(published):.6g}, options "
        f"{json.dumps({key: options[key] for key in ('arch', 'bias', 'head_dim', 'rope_theta', 'rms_norm_eps') if key in options})}")
    # Granite 4.1's kind: the three multipliers the engine has not
    for key, value in (("embedding_multiplier", 12.0), ("residual_multiplier", 0.22), ("logits_scaling", 10.0)):
        try:
            made_up_conversion(tensors, {**published, key: value})
            failed = True
            say(f"made-up ({name}): {key} {value} was converted — FAILED")
        except ValueError:
            pass
    return failed


# ------------------------------------------------------------------------------------------------- a real model
def entries(ids):
    script = ("import('./src/models.js').then(({ MODELS }) => console.log(JSON.stringify("
              f"MODELS.filter((m) => {json.dumps(ids)}.includes(m.id)))))")
    found = {entry["id"]: entry for entry in json.loads(subprocess.check_output(["node", "-e", script], cwd=HERE.parent))}
    missing = [id for id in ids if id not in found]
    if missing:
        sys.exit(f"reference: the list has no {', '.join(missing)}")
    return [found[id] for id in ids]


def fetch(repo, revision, name, folder, optional=False):
    # (tests/fetching.py, T357: and a download that stopped short is asked for again, which this copy of the loop did not see)
    return download(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", folder / name, optional=optional)


def real(entry, directory, positions, layers=0):
    """layers: only the first of the model's layers, the same on both sides (the model's own embedding, the norm and the
    classifier; every weight of those layers as it is): for a model whose float32 does not fit a runner (Granite 4.2 8B is
    35 GB), the multiplier and the real weights of its layers through the whole conversion, in float32 to the floor."""
    import torch
    import transformers
    from format_check import filled
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    id = entry["id"]
    original = entry["hf"].get("vocabulary") or entry["hf"]
    repo, revision = original["repo"], original["revision"]
    folder = directory / repo.replace("/", "--") / revision
    say(f"{id}: {repo}@{revision}, transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}")
    # what transformers reads: the configuration, the tokenizer's files (the entry's, or every candidate there is), the
    # original's weights
    fetch(repo, revision, "config.json", folder)
    for name in dict.fromkeys([*candidates_of({"tokenizer": original.get("tokenizer")}), "tokenizer.json", "tokenizer_config.json",
                               "chat_template.jinja", "special_tokens_map.json", "generation_config.json"]):
        fetch(repo, revision, name, folder, optional=True)
    names = ["model.safetensors"]
    if fetch(repo, revision, "model.safetensors", folder, optional=True) is None:
        index = json.loads(fetch(repo, revision, "model.safetensors.index.json", folder).read_text())
        names = sorted(set(index["weight_map"].values()))
        for name in names:
            fetch(repo, revision, name, folder)
    config = (folder / "config.json").read_text()
    if layers:
        config = json.dumps({**json.loads(config), "num_hidden_layers": layers})
        say(f"{id}: only the first {layers} of the layers, on both sides")

    # the conversion, the way the page does it (T374.4: by the conduct of a conversion, answered from the files fetched
    # here; what it asks for besides them is fetched as it asks), float32
    if GGUF:
        # the route the list's entries take (T136's second stage): the entry's own GGUF (Q8_0) as the weights, the original's
        # vocabulary and config.json. transformers still has the original's weights, so the two differ by what Q8_0 rounds
        # (about 0.5% of a weight): a line of a different kind, below
        hf = entry["hf"]
        if not hf["weights"].endswith(".gguf"):
            sys.exit(f"reference: {id} is not a GGUF entry")
        places = {"weights": (hf["repo"], hf["revision"], directory / "gguf" / hf["revision"]), "vocabulary": (repo, revision, folder)}
        say(f"{id}: weights from {hf['repo']}'s {hf['weights']}, the vocabulary and config.json of the original")
    else:
        # the original's safetensors: the one file, or the shards its index names, joined as the page joins them
        hf = {"weights": "model.safetensors", "config": "config.json", "tokenizer": original.get("tokenizer")}
        places = {"weights": (repo, revision, folder)}
    where = "vocabulary" if hf.get("vocabulary") else "weights"
    sink = Mapped(directory / f"{id}.float32.bin")
    began = time.perf_counter()
    conversion = converted(Fetched(places, instead={(where, "config.json"): config}), hf, dtype="float32", sink=sink)
    sink.data.flush()
    del sink.data
    options = {**conversion.options, **entry.get("options", {})}
    template = entry.get("template") or options.get("template")
    options.pop("template", None)
    shown = {key: (value if key != "specials" else f"{len(value)} of them") for key, value in options.items()}
    say(f"{id}: converted in {time.perf_counter() - began:.1f} s, {sink.path.stat().st_size} bytes, what q was multiplied "
        f"by {llama2_convert.query_scale(llama2_convert.normalize(json.loads(config))):.6g}, options {json.dumps(shown)}")
    say(f"{id}: the format of {'the list' if entry.get('template') else 'the converter'}: {json.dumps(template)}")
    llama = Llama(np.memmap(sink.path, dtype=np.uint8, mode="r"), conversion.tokenizer, kernels=None, **options)
    specials = llama.specials

    # the ids: the text as the real tokenizer splits it, with whatever it puts in front, and the chat prompt as the
    # page sends it ([bos] + the list's format) against the real template's
    reference = AutoTokenizer.from_pretrained(folder)
    ids = list(reference(TEXT)["input_ids"])
    mine = llama.tokenizer.encode(TEXT, specials)
    # one token in front where the real tokenizer's first is one the text does not begin with (its BOS), else none
    front = ids[:1] if ids[:1] != mine[:1] and ids[1:2] == mine[:1] else []
    failed = ids[len(front):] != mine
    say(f"{id}: {len(ids)} tokens of text, the real tokenizer puts {front or 'nothing'} in front "
        f"({reference.convert_ids_to_tokens(front)}); after it the engine's tokenizer gives "
        f"{'the same ids' if not failed else 'OTHER IDS — FAILED'}")
    if failed:
        first = next((at for at, (a, b) in enumerate(zip(ids[len(front):], mine)) if a != b), min(len(mine), len(ids)))
        say(f"{id}: first at {first}: real {reference.convert_ids_to_tokens(ids[len(front):][first:first + 8])}, "
            f"the engine {reference.convert_ids_to_tokens(mine[first:first + 8])}")
    if front and front != [conversion.options["bos"]]:
        failed = True
        say(f"{id}: the converter's BOS is {conversion.options['bos']}, not what the real tokenizer begins with — FAILED")
    thinking = {"enable_thinking": False} if "(no thinking)" in entry["name"] else \
        {"enable_thinking": True} if "(thinking)" in entry["name"] else {}
    chat_real = reference.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                              tokenize=True, **thinking)
    chat_real = list(chat_real["input_ids"] if hasattr(chat_real, "keys") else chat_real)
    typed = filled(template, PROMPT) if template else PROMPT
    chat = [options["bos"]] + llama.tokenizer.encode(typed, specials)
    same_chat = chat == chat_real
    failed |= not same_chat
    say(f"{id}: the chat prompt, {len(chat)} ids of the page ([bos {options['bos']}] and the format) against "
        f"apply_chat_template({thinking or ''}): {'the same ids' if same_chat else 'OTHER IDS — FAILED'}")
    if not same_chat:
        say(f"{id}: page {reference.convert_ids_to_tokens(chat)[:40]}\nreference: {id}: real {reference.convert_ids_to_tokens(chat_real)[:40]}")
    ids = ids[:positions]

    float32_arithmetic()
    stored = json.loads(config).get("dtype") or json.loads(config).get("torch_dtype") or "float32"
    shorter = AutoConfig.from_pretrained(str(folder))
    if layers:
        shorter.num_hidden_layers = layers
        if getattr(shorter, "layer_types", None):
            shorter.layer_types = shorter.layer_types[:layers]
    model = AutoModelForCausalLM.from_pretrained(str(folder), config=shorter, dtype=getattr(torch, stored)).eval()
    say(f"{id}: transformers' {type(model).__name__}, weights held as {stored}, arithmetic in float32, "
        f"{model.config.num_hidden_layers} layers")
    began = time.perf_counter()
    whole, stepped, cache_whole, cache_stepped = logits_of(model, ids, STEPPED, states=True)
    say(f"{id}: transformers, {len(ids)} positions at once and {len(stepped)} token by token in {time.perf_counter() - began:.1f} s")
    theirs = greedy(model, chat, NEW_TOKENS, set(options["stop_tokens"]))
    del model
    gc.collect()

    began = time.perf_counter()
    ours = [llama.forward(token, pos).copy() for pos, token in enumerate(ids)]
    states = engine_cache(llama, len(ids))
    say(f"{id}: the engine (NumPy, float32), {len(ids)} positions in {time.perf_counter() - began:.1f} s")
    for what, logits in (("transformers", whole), ("the engine", ours)):
        logits = np.asarray(logits)
        say(f"{id}: logits of {what}: min {logits.min():.3f}, max {logits.max():.3f}, mean {logits.mean():.4f}, std {logits.std():.4f}")
    floor = differences(stepped, whole[:len(stepped)])[0]
    say(f"{id}: transformers token by token against transformers at once ({len(stepped)} positions): largest difference "
        f"{floor:.2e} (what float32 leaves between two right computations)")
    # the line: ten times what transformers' own two computations differ by (which changes with the CPU), and no less
    # than 1e-3 (tests/reference_qwen35.py's, after the review of T229 found 2e-2 let a fault by: the right computation
    # is 7e-5 off here); the most likely token the same wherever the reference's best two are further apart than twice
    # the difference. A NaN is past the line (largest <= line is false)
    line = max(1e-3, 10 * floor)
    if GGUF:
        line = GGUF_LINE  # (Q8_0 rounds every weight: the engine and transformers hold different weights)
    for what, reference_logits, mine_logits in (("transformers at once", whole, ours),
                                                ("transformers token by token", stepped, ours[:len(stepped)])):
        largest, mean, same, margin = differences(mine_logits, reference_logits)
        ok = largest <= line and (same == len(mine_logits) or margin <= 2 * largest)
        failed |= not ok
        say(f"{id}: the engine against {what}: largest difference {largest:.2e}, mean {mean:.2e}, the same most likely "
            f"token at {same} of {len(mine_logits)} positions (the largest gap between the reference's best two where "
            f"it is not: {margin:.2e}){'' if ok else f' — FAILED (the line: {line:.1e})'}")

    # the states: the keys and values of every layer (see state_gap), against transformers' whole pass and against the floor of
    # its own two (the stepped positions of the cache against the same of the whole pass)
    floors, _ = state_gap(cache_stepped, [(k[:, :len(stepped)], v[:, :len(stepped)]) for k, v in cache_whole])
    lined = GGUF_STATE_LINE if GGUF else state_line({kind: gap for kind, (gap, _) in floors.items()})
    gaps, top = state_gap(states, cache_whole)
    ok = max(gaps["keys"][0], gaps["values"][0]) <= lined
    failed |= not ok
    say(f"{id}: states of every layer, the engine against transformers' ({len(ids)} positions): keys {gaps['keys'][0]:.2e} "
        f"(layer {gaps['keys'][1]}), values {gaps['values'][0]:.2e} (layer {gaps['values'][1]}), each layer's relative to its largest; "
        f"transformers' own two: keys {floors['keys'][0]:.2e}, values {floors['values'][0]:.2e}; the line {lined:.1e}"
        f"{'' if ok else ' — FAILED'}")
    say(f"{id}: the largest key {top['keys']:.4g} and the largest value {top['values']:.4g} of any layer (a float16 holds 65504)")
    if WEAK and not GGUF:
        missed = weak_errors(id, llama, ids, whole, cache_whole, line, lined, llama2_convert.query_scale(llama2_convert.normalize(json.loads(config))))
        failed |= bool(missed)
        say(f"{id}: weak faults the check must see and did not: {missed or 'none'}")
    if llama.unturned:
        missed = unturned_faults(id, llama, ids, whole, cache_whole, line, lined)
        failed |= bool(missed)
        say(f"{id}: faults in the layers RoPE leaves alone the check did not see: {missed or 'none'}")

    # 16 greedy tokens after the page's ids of the chat prompt, by transformers and by the engine's generate()
    text = "".join(llama.generate(typed, steps=len(chat) - 1 + NEW_TOKENS, temperature=0.0, echo=False))
    wrote = reference.decode(theirs, skip_special_tokens=False)
    say(f"{id}: transformers wrote {json.dumps(wrote, ensure_ascii=False)} {theirs}")
    say(f"{id}: the engine wrote {json.dumps(text, ensure_ascii=False)}")
    # (from the GGUF, Q8_0 may turn a close second into the likeliest token a few tokens in: the texts are shown, not held)
    failed |= wrote != text and not GGUF
    say(f"{id}: the two wrote {'the same' if wrote == text else 'other texts' + ('' if GGUF else ' — FAILED')}")
    del llama
    gc.collect()
    sink.path.unlink()
    return failed


def main():
    directory = Path(sys.argv[1])
    only = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--only=")), None)
    positions = int(next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--positions=")), 96))
    layers = int(next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--layers=")), 0))
    ids = [arg for arg in sys.argv[2:] if not arg.startswith("--")]
    failed = False
    if only in (None, "made-up"):
        for name, settings in MADE_UP.items():
            failed |= made_up(name, settings)
    for entry in entries(ids) if only is None else []:
        failed |= real(entry, directory, positions, layers)
    say("FAILED" if failed else "the engine computes what transformers computes")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
