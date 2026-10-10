# GGUF as a source (T74): the header's metadata and tensor table shown as a safetensors-like header, what llama.cpp
# wrote differently put back, and the checks against the original's config.json. What a family's GGUF is called, says
# and holds otherwise than another's is the family's (convert/families/, T359), looked up by general.architecture.
import json
import math
import struct

import numpy as np

from engine.layers import rope_frequencies, sign_bits
from convert.readers import GGUF_TENSORS, SOURCES, read_types
from convert.sources import ROTATED
from convert.families import GGUF, family_of, named
from convert.families.family import f32
from convert.config import head_size, normalize, yarn
from convert.tokenizer import UNMATCHABLE, tokenizer_bin


# A GGUF file holds what config.json, tokenizer.json and model.safetensors hold, in one. Only what a Q8_0, PQ2_0 or F16
# Llama, Granite, Qwen2, Qwen3, Qwen3.5, GPT-2 or GPT-NeoX needs is read; tests/gguf_check.py is the separate reference this is
# held to.
class Incomplete(Exception):
    """The GGUF header goes on past the bytes given: fetch more and try again."""


GGUF_VALUES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
# (ggml's types of tensors, and which of them are read: convert/readers.py's SOURCES)
# llama.cpp's names of the pre-tokenizers, as the engine knows them (llama2_numpy.pretokenize)
# (granite-docling, T253: what llama.cpp calls a Granite 4.2's ByteLevel with its regex, and splits by GPT-2's pattern.
# minicpm5, T254: llama.cpp's two patterns of that name are tokenizer.json's but for the contractions, written out by
# case, which leaves a 's after U+017F unmatched; openbmb's own GGUFs of 2026-09 still say llama-bpe)
# (lfm2, T260: llama.cpp splits an LFM2's text as a Llama 3's, which is the pattern of its tokenizer.json; it also
    # takes a piece that is in the vocabulary whole, which that tokenizer.json does not: ignore_merges is false there,
    # and gguf_tokenizer() says it of llama-bpe alone)
GGUF_PRETOKENIZERS = {"gpt-2": "gpt2", "gpt2": "gpt2", "smollm": "gpt2-digits", "qwen2": "qwen", "llama-bpe": "llama3",
                      "qwen35": "qwen35", "granite-docling": "gpt2", "minicpm5": "minicpm5", "lfm2": "llama3"}
# the ones whose tokenizer.json normalizes to NFC, which a GGUF does not say (Qwen's)
GGUF_NFC = ("qwen2", "qwen35")


def gguf_read(data):
    """(metadata, tensors, base) from the first bytes of a GGUF file: tensors maps each name to its ggml type,
    its shape (outermost first, as NumPy has it) and its offset from base, where the data begins."""
    data = memoryview(data.to_py() if hasattr(data, "to_py") else data).cast("B")
    at = 0

    def take(fmt):
        nonlocal at
        size = struct.calcsize(fmt)
        if at + size > len(data):
            raise Incomplete()
        (value,) = struct.unpack_from(fmt, data, at)
        at += size
        return value

    def string():
        nonlocal at
        size = take("<Q")
        if at + size > len(data):
            raise Incomplete()
        at += size
        return bytes(data[at - size:at]).decode("utf-8", errors="replace")

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            item, count = take("<I"), take("<Q")
            return [value(item) for _ in range(count)]
        if kind not in GGUF_VALUES:
            raise ValueError(f"This GGUF file has a value of type {kind}, which is not in the format.")
        return take(GGUF_VALUES[kind])

    if len(data) >= 4 and bytes(data[:4]) != b"GGUF":
        raise ValueError("This is not a GGUF file.")
    take("<I")
    version = take("<I")
    if version not in (2, 3):
        raise ValueError(f"This GGUF file is of version {version}; only 2 and 3 are supported.")
    count, entries = take("<Q"), take("<Q")
    metadata = {}
    for _ in range(entries):
        key = string()
        metadata[key] = value(take("<I"))
    tensors = {}
    for _ in range(count):
        name = string()
        dims = [take("<Q") for _ in range(take("<I"))]
        tensors[name] = {"type": take("<I"), "shape": list(reversed(dims)), "offset": take("<Q")}
    alignment = metadata.get("general.alignment", 32)
    return metadata, tensors, (at + alignment - 1) // alignment * alignment


def gguf_model(metadata, tensors, base, rope_freqs=False):
    """The safetensors-like header (Hugging Face's names, offsets from base) and the config.json of a GGUF.
    rope_freqs: keep llama.cpp's table of Llama 3's RoPE scaling in the header, to be checked against the original's
    rope_scaling as it streams past (gguf_weights, T136), instead of refusing it."""
    arch = metadata.get("general.architecture")
    if arch not in GGUF:
        raise ValueError(f"This GGUF holds a {arch}: only {named(family for _, family in GGUF.values())} ones are supported.")
    model_type, family = GGUF[arch]
    key = lambda name, default=None: metadata.get(f"{arch}.{name}", default)
    # config.json by the names normalize() reads, as the family reads its metadata (convert/families/)
    config = {"model_type": model_type, **family.gguf_config(key, tensors),
              "vocab_size": tensors["token_embd.weight"]["shape"][0] if "token_embd.weight" in tensors else None,
              "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id", 1),
              "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id", 2)}
    header = {}
    if "rope_freqs.weight" in tensors:
        # llama.cpp writes Llama 3's RoPE scaling as a table of divisors instead of the rope_scaling of config.json
        if not rope_freqs:
            raise ValueError("This GGUF scales its RoPE with a rope_freqs table, which the engine does not read.")
        info = tensors["rope_freqs.weight"]
        if info["type"] != 0:
            raise ValueError(f"rope_freqs.weight is stored as ggml type {info['type']}, not F32.")
        size = 4 * math.prod(info["shape"])
        header["rope_freqs.weight"] = {"dtype": "F32", "shape": info["shape"], "rope_freqs": True,
                                       "data_offsets": [info["offset"], info["offset"] + size]}
    names, layer, layers = family.gguf_names
    # what llama.cpp stores of this family otherwise than Hugging Face has it, marked in the tensor's entry
    stored = family.gguf_stored(config)
    for name, info in tensors.items():
        if info["type"] not in GGUF_TENSORS:
            raise ValueError(f"{name} is stored as ggml type {info['type']}: only {read_types()} "
                             f"GGUF files are supported (not the K-quants).")
        parts = name.split(".")
        if name in names:
            target = names[name]
        elif len(parts) > 2 and parts[0] == "blk" and ".".join(parts[2:]) in layers:
            target = f"{layer.format(parts[1])}{layers['.'.join(parts[2:])]}"  # a whole name (a Qwen3.5's ssm_a)
        elif len(parts) == 4 and parts[0] == "blk" and parts[2] in layers:
            target = f"{layer.format(parts[1])}{layers[parts[2]]}.{parts[3]}"
        else:
            continue  # nothing the engine reads
        dtype = GGUF_TENSORS[info["type"]]
        if info["shape"][-1] % SOURCES[dtype].values:
            # ggml itself requires it; a file that breaks it would be read at the wrong offsets and write nonsense
            raise ValueError(f"{name} is {dtype} with rows of {info['shape'][-1]}, which is not a multiple of "
                             f"{SOURCES[dtype].values}.")
        size = SOURCES[dtype].bytes(math.prod(info["shape"]))
        entry = {"dtype": dtype, "shape": info["shape"], "data_offsets": [info["offset"], info["offset"] + size]}
        stored(entry, info, target, parts)
        header[target] = entry
    # T237: a rotated basis is the header's to say (header_rotated), next to the tensors it is about. Whether a
    # linear-attention layer has more value heads than key heads: whether T245's order of them is there to put back
    rotated = gguf_rotated(metadata, tensors, any("tiled" in entry for entry in header.values()))
    if rotated is not None:
        header["__metadata__"] = {ROTATED: json.dumps(rotated)}
        # A rotated GGUF holds the columns of a linear-attention layer's output matrix in Hugging Face's order of value
        # heads (gdn_v_grouped: columns moved after the fold would be another matrix), where llama.cpp otherwise
        # stores them in its own. So the reader of that order (T245's "tiled", once it is here) must leave them be
        for target, entry in header.items():
            if target.endswith("linear_attn.out_proj.weight"):
                entry.pop("tiled", None)
    return header, config


# T237: the kinds of a layer's matrices that the engine multiplies a rotated input by, by llama.cpp's names (and
# output.weight): all the matrices of a Llama, a Qwen and a Qwen3.5 but the two small ones of a linear-attention
# layer's gates (ssm_alpha, ssm_beta)
GGUF_ROTATED = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out", "ffn_gate", "ffn_up",
                "ffn_down")


def gguf_rotated(metadata, tensors, more_value_heads=False):
    """T237: the rotated basis a GGUF of Prism ML's says its matrices are in (prism.hadamard.*), as FORM's "rotated"
    ({"block", "signs": {width: sign_bits()}}), or None where it says none. The fork's loader is the definition
    (src/llama-model.cpp, lines 1196 to 1355 at 88c4bc60): version 1 (2 with tied_output: the embedding is the
    classifier too), the transform's name, the axis, a block that is a power of two and divides every width,
    sign_mode "explicit" (sign_widths, and sign_values one width after the other) or "identity" (no signs),
    weight_names (the matrices stored as W R^-1), inverse_weight_names (the embedding, whose rows are turned back) and
    gdn_v_grouped (a linear-attention layer's output matrix reads its heads in Hugging Face's order). The engine
    turns the input of every matrix and the embedding's row, so a file that rotates other tensors than those is
    refused: it would run without a word and write nonsense. more_value_heads: a linear-attention layer has more
    value heads than key heads, whose output matrix in llama.cpp's own order of heads could not be put back."""
    said = lambda name, default=None: metadata.get(f"prism.hadamard.{name}", default)
    version = said("version")
    if version is None:
        if any(key.startswith("prism.hadamard.") for key in metadata):
            raise ValueError("This GGUF says a rotated basis without its version.")
        return None
    refuse = lambda why: ValueError(f"This GGUF's rotated basis is not one the engine computes in: {why}.")
    tied = bool(said("tied_output", False))
    if version not in (1, 2) or (version == 2) != tied or (tied and "output.weight" in tensors):
        raise refuse(f"version {version}{' with a tied output' if tied else ''}")
    if said("transform") != "normalized-sylvester-walsh-hadamard" or said("axis") != "input-last-dimension":
        raise refuse(f"the transform {said('transform')} along {said('axis')}")
    block = said("block_size")
    if not isinstance(block, int) or block < 1 or block & (block - 1):
        raise refuse(f"blocks of {block}")
    parts = lambda name: name.split(".")
    folded = {name for name in tensors if name == "output.weight" or (
        len(parts(name)) == 4 and parts(name)[0] == "blk" and parts(name)[2] in GGUF_ROTATED and parts(name)[3] == "weight")}
    names = set(said("weight_names", []))
    if names != folded:
        odd = sorted(names ^ folded)
        raise refuse(f"it rotates {'' if odd[0] in names else 'not '}{odd[0]}")
    if list(said("inverse_weight_names", [])) != ["token_embd.weight"]:
        raise refuse("the embedding's rows are not the only ones to turn back")
    if more_value_heads and not said("gdn_v_grouped", False):
        raise refuse("a linear-attention layer's output matrix in llama.cpp's order of value heads")
    widths = sorted({tensors[name]["shape"][-1] for name in folded | {"token_embd.weight"}})
    mode = said("sign_mode")
    if mode == "identity":
        signs = {width: np.ones(width) for width in widths}
    elif mode == "explicit":
        values, signs, at = np.asarray(said("sign_values", []), dtype=np.int64), {}, 0
        for width in said("sign_widths", []):
            signs[int(width)] = values[at:at + int(width)]
            at += int(width)
        if at != values.size or np.any(np.abs(values) != 1):
            raise refuse("its signs are not +1 and -1 for the widths it names")
    else:
        raise refuse(f"sign mode {mode}")
    for width in widths:
        if width % block or width not in signs or signs[width].size != width:
            raise refuse(f"no signs in whole blocks of {block} for an input {width} wide")
    return {"block": block, "signs": {str(width): sign_bits(signs[width]) for width in widths}}


def gguf_weights(head, config):
    """T136's second stage: the weights of a GGUF with the vocabulary and config.json of the original repository,
    for the GGUF's own vocabulary is of no use there (a sentencepiece one says neither Unigram or BPE nor its
    normalization; llm-jp's scores are all -1000). head: the GGUF's beginning, as far as the tensors' data (Incomplete
    when it is not), config: the text of the original's config.json. Returns the safetensors-like header (JSON text)
    and where the tensors begin, which Conversion() then takes as it takes a safetensors file's."""
    metadata, tensors, base = gguf_read(head)
    header, own = gguf_model(metadata, tensors, base, rope_freqs=True)
    try:
        original = json.loads(config)
    except ValueError:
        raise ValueError("config.json is not JSON.") from None
    if not isinstance(original, dict):
        raise ValueError("config.json is not the configuration of a model.")
    gguf_agrees(normalize(own), normalize(original))
    return json.dumps(header), base


def gguf_agrees(own, config):
    """ValueError unless a GGUF (own: what gguf_model() read of it) holds the model config.json describes. The
    sizes of the tensors the conversion checks anyway (Stream); these are what the sizes do not show: heads and
    key-value heads of the same product, the number of layers (a GGUF of more layers than config.json says went
    through cut to that many: Stream reads the layers the header asks for), a classifier that would silently be the
    embedding (Stream shares it where lm_head is missing), and the numbers that are no tensor. The context is not
    compared: a sliding window cuts it (RakutenAI 2.0 mini: 131072 in the GGUF, 8192 as normalize() cuts it).
    Both are normalize()d. GPT-NeoX's (T136's third stage): also how much of each head turns and whether the two
    branches run in parallel, which the options say (no tensor does)."""
    scaled = lambda c: yarn(c) and {key: f32(value) for key, value in yarn(c).items()}
    heads = config.get("num_attention_heads")
    pairs = [("architecture", own["model_type"], config.get("model_type")),
             ("number of layers", own["num_hidden_layers"], config.get("num_hidden_layers")),
             ("number of heads", own["num_attention_heads"], heads),
             ("number of key-value heads", own.get("num_key_value_heads", own["num_attention_heads"]),
              config.get("num_key_value_heads", heads)),
             ("RoPE theta", f32(own.get("rope_theta", 10000.0)), f32(config.get("rope_theta", 10000.0))),
             # T235: what a yarn says (None: no yarn), which changes every angle and is no tensor
             ("yarn RoPE scaling", scaled(own), scaled(config))]
    if own.get("head_dim") and config.get("hidden_size") and heads:
        pairs.append(("size of a head", own["head_dim"], head_size(config)))
    if own.get("rms_norm_eps") is not None and config.get("rms_norm_eps") is not None:
        pairs.append(("RMSNorm epsilon", f32(own["rms_norm_eps"]), f32(config["rms_norm_eps"])))
    if own["model_type"] == config.get("model_type"):
        # and what the family's numbers are that no tensor shows (another family's fails as the first pair)
        pairs += family_of(own).agrees(own, config)
    for what, here, there in pairs:
        if here != there:
            raise ValueError(f"This GGUF does not belong with the original's config.json: its {what} is {here} here "
                             f"and {there} there.")
    if own["tie_word_embeddings"] and not config.get("tie_word_embeddings", False):
        raise ValueError("This GGUF does not belong with the original's config.json: the original has a classifier of "
                         "its own, this GGUF has none.")


def rope_freqs_agree(table, config):
    """ValueError unless llama.cpp's rope_freqs (a divisor of each pair's angle) is what the original's rope_scaling
    makes: the engine makes its RoPE tables from rope_scaling (rope_frequencies), and the table is not used."""
    width = head_size(config)
    theta = config.get("rope_theta", 10000.0)
    expected = rope_frequencies(width, theta) / rope_frequencies(width, theta, config.get("rope_scaling"))
    table = np.asarray(table, dtype=np.float64)
    worst = float(np.max(np.abs(table - expected) / expected)) if table.shape == expected.shape else float("inf")
    if not worst <= 1e-5:
        raise ValueError(f"This GGUF scales its RoPE otherwise than the original's rope_scaling says (by {worst:.1e} "
                         f"at most).")


def gguf_tokenizer(metadata, vocab_size):
    """tokenizer.bin, the engine's options, the tokenizer_config and the special tokens of a GGUF's byte-level BPE
    vocabulary."""
    if metadata.get("tokenizer.ggml.model") != "gpt2":
        raise ValueError(f"This GGUF has a {metadata.get('tokenizer.ggml.model')} vocabulary: only byte-level BPE "
                         f"ones (gpt2) are supported.")
    pre = metadata.get("tokenizer.ggml.pre", "gpt-2")
    if pre not in GGUF_PRETOKENIZERS:
        raise ValueError(f"This GGUF splits text as {pre}, which the engine does not know.")
    tokens, kinds = metadata["tokenizer.ggml.tokens"], metadata.get("tokenizer.ggml.token_type", [])
    ranks = {}
    for rank, merge in enumerate(metadata.get("tokenizer.ggml.merges", [])):
        left, right = merge.split(" ")
        ranks.setdefault(left + right, -float(rank))
    # what tokenizer.json calls special: llama.cpp's control tokens (type 3). Its padding up to the vocabulary's size
    # ([PAD151665] ..., type 5, unused) is text no piece of tokenizer.json has: empty, as the safetensors path pads (T143)
    kind = lambda id: kinds[id] if id < len(kinds) else 1
    pieces = [("" if kind(id) == 5 else text, ranks.get(text, UNMATCHABLE), text in ranks and kind(id) != 3)
              for id, text in enumerate(tokens)]
    # Qwen's tokenizer.json normalizes to NFC, which a GGUF does not say: the page's safetensors path does it
    options = {"tokenizer_kind": "bytebpe", "nfkc": False, "nfc": pre in GGUF_NFC, "pretokenizer": GGUF_PRETOKENIZERS[pre],
               "ignore_merges": pre == "llama-bpe"}
    special = lambda key: tokens[metadata[key]] if isinstance(metadata.get(key), int) and metadata[key] < len(tokens) else ""
    config = {"chat_template": metadata.get("tokenizer.chat_template"), "bos_token": special("tokenizer.ggml.bos_token_id"),
              "eos_token": special("tokenizer.ggml.eos_token_id")}
    # the added tokens tokenizer.json does not call special are llama.cpp's user-defined ones (type 4)
    controls, added = ([text for id, text in enumerate(tokens) if id < len(kinds) and kinds[id] == type] for type in (3, 4))
    return tokenizer_bin(pieces, vocab_size, spaces=False), options, config, controls, added
