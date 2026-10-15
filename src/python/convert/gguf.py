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
    # and GGUF_IGNORE_MERGES says it of llama-bpe and not of lfm2)
# (T505.2: smaug-bpe and deepseek-r1-qwen, which llama.cpp splits by the patterns of llama-bpe and of qwen2
# (src/llama-vocab.cpp at the commit convert/superblocks.py names), each held to a real GGUF and its original's
# tokenizer.json: a SmolLM3's and a Llama 3.1's say smaug-bpe, DeepSeek-R1's Qwen distills deepseek-r1-qwen. A name is
# added once a real file of it is held to its original so, and not for being on llama.cpp's list:
# docs/notes/t505-gguf-alone-2026-10-13.md has the names seen and the ones left out)
# (spark2_5, T340: XHToken's Q8_0 GGUF of Spark-X2.5 1.7B has the 131,072 tokens and 130,716 merges of the original's
# tokenizer.json, each the same, in the same order: docs/notes/t339-spark-x-2026-10-15.md)
GGUF_PRETOKENIZERS = {"gpt-2": "gpt2", "gpt2": "gpt2", "smollm": "gpt2-digits", "qwen2": "qwen", "llama-bpe": "llama3",
                      "qwen35": "qwen35", "granite-docling": "gpt2", "minicpm5": "minicpm5", "lfm2": "llama3",
                      "smaug-bpe": "llama3", "deepseek-r1-qwen": "qwen", "spark2_5": "spark2_5"}
# the ones whose tokenizer.json normalizes to NFC, which a GGUF does not say (Qwen's)
GGUF_NFC = ("qwen2", "qwen35", "deepseek-r1-qwen")
# the ones whose tokenizer.json takes a piece that is in the vocabulary whole (ignore_merges), which a GGUF does not
# say (a Llama 3's; llama.cpp does so for llama-bpe and for lfm2, and not for smaug-bpe, whose two originals do)
GGUF_IGNORE_MERGES = ("llama-bpe", "smaug-bpe")


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


def gguf_model(metadata, tensors, base, rope_freqs=False, table=None):
    """The safetensors-like header (Hugging Face's names, offsets from base) and the config.json of a GGUF.
    rope_freqs: keep llama.cpp's table of Llama 3's RoPE scaling in the header, to be checked against the original's
    rope_scaling as it streams past (gguf_weights, T136). table (T505.2): that table's values, where the GGUF is
    alone and its head held them (rope_freqs_values()): the config.json says the rope_scaling that makes it
    (llama3_scaling()). With neither, a GGUF that has the table is refused."""
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
    if "rope_freqs.weight" in tensors and family.gguf_rope_freqs:
        # T529: a table of the family's own RoPE (a Gemma 4's proportional one), which no rope_scaling says: held to
        # what the config.json makes as it streams past (rope_freqs_agree()), whichever config.json that is
        info = tensors["rope_freqs.weight"]
        if info["type"] != 0:
            raise ValueError(f"rope_freqs.weight is stored as ggml type {info['type']}, not F32.")
        header["rope_freqs.weight"] = {"dtype": "F32", "shape": info["shape"], "rope_freqs": True,
                                       "data_offsets": [info["offset"], info["offset"] + 4 * math.prod(info["shape"])]}
    elif "rope_freqs.weight" in tensors:
        # llama.cpp writes Llama 3's RoPE scaling as a table of divisors instead of the rope_scaling of config.json
        info = tensors["rope_freqs.weight"]
        if rope_freqs:
            if info["type"] != 0:
                raise ValueError(f"rope_freqs.weight is stored as ggml type {info['type']}, not F32.")
            size = 4 * math.prod(info["shape"])
            header["rope_freqs.weight"] = {"dtype": "F32", "shape": info["shape"], "rope_freqs": True,
                                           "data_offsets": [info["offset"], info["offset"] + size]}
        elif table is None:
            raise ValueError(NO_TABLE)
        else:
            scaling = llama3_scaling(table, config)
            if scaling:
                config["rope_scaling"] = scaling
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


NO_TABLE = "This GGUF scales its RoPE with a rope_freqs table, which the engine does not read."


def rope_freqs_values(head, tensors, base, metadata=None):
    """T505.2: llama.cpp's rope_freqs table of a GGUF alone, read off its head (the file from its beginning): float64,
    or None where the file has none. llama.cpp's converter writes the table before every other tensor, a few hundred
    bytes: Incomplete where the head ends before it does (whoever fetches the head fetches on). A table anywhere else
    (Swallow 8B's GGUF has it after every other tensor, gigabytes in) is refused as before: the RoPE tables of a float32
    checkpoint are made before the first tensor is read."""
    info = tensors.get("rope_freqs.weight")
    arch = (metadata or {}).get("general.architecture")
    if info is None or (arch in GGUF and GGUF[arch][1].gguf_rope_freqs):
        # (T529: a family's own table, a Gemma 4's, is checked as it streams past and says no rope_scaling)
        return None
    if info["type"] != 0 or info["offset"] != 0:
        raise ValueError(NO_TABLE)
    data = memoryview(head.to_py() if hasattr(head, "to_py") else head).cast("B")
    size = 4 * math.prod(info["shape"])
    if len(data) < base + size:
        raise Incomplete()
    return np.frombuffer(bytes(data[base:base + size]), dtype="<f4").astype(np.float64)


def llama3_scaling(table, config):
    """T505.2: the rope_scaling (config.json's, of the llama3 kind) that makes llama.cpp's rope_freqs table, which is
    all a GGUF says of it: a divisor of each pair's angle (rope_freqs_agree). None for a table of ones (no scaling).
    The divisor is 1 for the pairs that turn fast, factor for the slow ones, and between them 1 / ((1 - s) / factor + s)
    with s a line in a pair's frequency: s = (original / wavelength - low) / (high - low). So factor is the largest
    divisor, and the pairs between give the line; only original / low and original / high matter to the table, so
    low is taken as 1 (every Llama 3's), and where there are fewer than two pairs between, a Llama 3's 4 and 8192 are
    tried. Whole numbers where they make the table, as config.json has them. Whatever is found must make the table
    as rope_freqs_agree() holds one to an original's (1e-5), or the GGUF is refused as before: the engine makes its
    tables from a rope_scaling and would turn by another table without a word."""
    table = np.asarray(table, dtype=np.float64)
    width, theta = head_size(config), float(config.get("rope_theta", 10000.0))
    if table.shape != (width // 2,) or not np.all(np.isfinite(table)) or not np.all(table >= 1 - 1e-6):
        raise ValueError(NO_TABLE)
    factor = float(table.max())
    if factor - 1 < 1e-6:
        return None
    factor = float(round(factor)) if abs(round(factor) - factor) < 1e-5 * factor else factor
    tried = [(4.0, 8192)]
    between = (table > 1 + 1e-4) & (table < factor * (1 - 1e-4))
    if between.sum() >= 2:
        s = (1 / table[between] - 1 / factor) / (1 - 1 / factor)
        slope, at_zero = np.polyfit(rope_frequencies(width, theta)[between] / (2 * math.pi), s, 1)
        if at_zero < 0 and slope > 0:
            high, original = 1 - 1 / at_zero, -slope / at_zero
            tried = [(float(round(high)), int(round(original))), (float(high), float(original)), *tried]
    for high, original in tried:
        scaling = {"rope_type": "llama3", "factor": factor, "low_freq_factor": 1.0, "high_freq_factor": high,
                   "original_max_position_embeddings": original}
        try:
            if high > 1 and original > 0:
                rope_freqs_agree(table, {**config, "rope_scaling": scaling})
                return scaling
        except ValueError:
            pass
    raise ValueError(NO_TABLE)


def rope_freqs_agree(table, config):
    """ValueError unless llama.cpp's rope_freqs (a divisor of each pair's angle) is what the original's rope_scaling
    makes: the engine makes its RoPE tables from rope_scaling (rope_frequencies), and the table is not used. T529: or
    the family's own table, where it has one (a Gemma 4's: which pairs turn)."""
    own = family_of(normalize(config)).gguf_rope_freqs
    if own:
        expected = own(normalize(config))
    else:
        width = head_size(config)
        theta = config.get("rope_theta", 10000.0)
        expected = rope_frequencies(width, theta) / rope_frequencies(width, theta, config.get("rope_scaling"))
    table = np.asarray(table, dtype=np.float64)
    worst = float(np.max(np.abs(table - expected) / expected)) if table.shape == expected.shape else float("inf")
    if not worst <= 1e-5:
        raise ValueError(f"This GGUF scales its RoPE otherwise than the original's rope_scaling says (by {worst:.1e} "
                         f"at most).")


def gguf_tokenizer(metadata, vocab_size):
    """tokenizer.bin, the engine's options, the tokenizer_config and the special tokens of a GGUF's vocabulary, of the
    kinds of GGUF_VOCABULARIES (tokenizer.ggml.model: a byte-level BPE one, gpt2, or a sentencepiece BPE one, llama)."""
    model = metadata.get("tokenizer.ggml.model")
    tokens, kinds = metadata.get("tokenizer.ggml.tokens", []), metadata.get("tokenizer.ggml.token_type", [])
    kind = lambda id: kinds[id] if id < len(kinds) else 1
    read = GGUF_VOCABULARIES.get(model)
    found = read and read(metadata, tokens, kind)
    if not found:
        raise ValueError(f"This GGUF has a {model} vocabulary: only byte-level BPE ones (gpt2) are supported.")
    pieces, options, spaces = found
    special = lambda key: tokens[metadata[key]] if isinstance(metadata.get(key), int) and metadata[key] < len(tokens) else ""
    config = {"chat_template": metadata.get("tokenizer.chat_template"), "bos_token": special("tokenizer.ggml.bos_token_id"),
              "eos_token": special("tokenizer.ggml.eos_token_id")}
    # the added tokens tokenizer.json does not call special are llama.cpp's user-defined ones (type 4). (A
    # sentencepiece model has none to add: its user-defined pieces are matchable, as its own file's are)
    controls, added = ([text for id, text in enumerate(tokens) if id < len(kinds) and kinds[id] == type] for type in (3, 4))
    return tokenizer_bin(pieces, vocab_size, spaces=spaces), options, config, controls, [] if spaces else added


def gguf_bytebpe(metadata, tokens, kind):
    """(the pieces, the engine's options, False: a space is no U+2581) of a GGUF's byte-level BPE vocabulary, split as
    its tokenizer.ggml.pre says: one of GGUF_PRETOKENIZERS, or it is refused. T505.3: and so is one that does not say
    (the owner, 2026-10-13): llama.cpp reads such a file as "default", which splits as no model of these does (the
    GGUF of a Llama 3 or a Qwen made before the key was there would be split as a GPT-2's, and answer badly without a
    word)."""
    pre = metadata.get("tokenizer.ggml.pre", "default")
    if pre not in GGUF_PRETOKENIZERS:
        raise ValueError(f"This GGUF splits text as {pre}, which the engine does not know.")
    ranks = {}
    for rank, merge in enumerate(metadata.get("tokenizer.ggml.merges", [])):
        left, right = merge.split(" ")
        ranks.setdefault(left + right, -float(rank))
    # what tokenizer.json calls special: llama.cpp's control tokens (type 3). Its padding up to the vocabulary's size
    # ([PAD151665] ..., type 5, unused) is text no piece of tokenizer.json has: empty, as the safetensors path pads (T143)
    pieces = [("" if kind(id) == 5 else text, ranks.get(text, UNMATCHABLE), text in ranks and kind(id) != 3)
              for id, text in enumerate(tokens)]
    # Qwen's tokenizer.json normalizes to NFC, which a GGUF does not say: the page's safetensors path does it
    options = {"tokenizer_kind": "bytebpe", "nfkc": False, "nfc": pre in GGUF_NFC, "pretokenizer": GGUF_PRETOKENIZERS[pre],
               "ignore_merges": pre in GGUF_IGNORE_MERGES}
    return pieces, options, False


# llama.cpp's types of a token (tokenizer.ggml.token_type), which are sentencepiece's of a piece
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6


def sentencepiece_bpe(metadata):
    """T505.2: whether a GGUF's sentencepiece vocabulary (tokenizer.ggml.model "llama") is one the engine reads as
    llama2.c's BPE (merge the pair whose piece scores highest), which is how llama.cpp reads every such vocabulary.
    The GGUF does not say what the model was: a sentencepiece BPE model scores a piece by minus the number of its
    merge, so its normal pieces' scores are whole numbers and nearly all different (a Mistral 7B's 31,741: 31,727
    different ones). A Unigram model's are logarithms, and merged as a BPE's would be another tokenizer; a vocabulary
    converted from a tokenizer.json has no scores at all (TheBloke's TinyLlama: every score 0; llm-jp's: -1000), and
    its merges are anyone's guess. Those are left refused, and so is one that puts no space before the text
    (add_space_prefix false: the engine's sentencepiece reading always does)."""
    tokens, kinds = metadata.get("tokenizer.ggml.tokens", []), metadata.get("tokenizer.ggml.token_type", [])
    scores = metadata.get("tokenizer.ggml.scores", [])
    if not (len(tokens) == len(kinds) == len(scores)) or metadata.get("tokenizer.ggml.add_space_prefix", True) is not True:
        return False
    # T505.3: a vocabulary that comes with its normalizer (sentencepiece's precompiled_charsmap) normalizes otherwise
    # than by identity, and nothing here reads that map from a GGUF
    if metadata.get("tokenizer.ggml.precompiled_charsmap"):
        return False
    normal = [score for score, kind in zip(scores, kinds) if kind == NORMAL]
    return bool(normal) and all(float(score).is_integer() for score in normal) and 2 * len(set(normal)) > len(normal)


def gguf_sentencepiece(metadata, tokens, kind):
    """(the pieces, the engine's options, True: a space is written U+2581) of a GGUF's sentencepiece BPE vocabulary
    (None for one that sentencepiece_bpe() does not take for one), which holds the model's pieces, scores and types as
    they are: the same tokenizer.bin as the model's own file gives (convert/tokenizer.py's sentencepiece_pieces() and
    sentencepiece_options()). What a GGUF does not hold is the
    model's normalizer (its precompiled_charsmap): identity, as llama.cpp reads it and as a Llama's and a Mistral's
    is. collapse and unknown: sentencepiece_options()'s, from llama.cpp's key and from the types of the pieces.
    The dummy prefix stands before the first text alone, as for a model whose tokenizer_config.json says nothing else
    (T308's "prefixed", which a GGUF cannot say). llama.cpp puts it before the text after every special token too;
    that was held to two real files: a Mistral 7B v0.2's ids are the original's either way, and a v0.3's (whose
    [INST] is a token) are the original tokenizer's of the same text without it and not with it (a space twice)."""
    if not sentencepiece_bpe(metadata):
        return None
    scores = metadata["tokenizer.ggml.scores"]
    pieces = [(text, float(scores[id]), kind(id) in (NORMAL, USER_DEFINED)) for id, text in enumerate(tokens)]
    types = [kind(id) for id in range(len(tokens))]
    unknown = types.index(UNKNOWN) if UNKNOWN in types and BYTE not in types else None
    options = {"tokenizer_kind": "bpe", **({"collapse": True} if metadata.get("tokenizer.ggml.remove_extra_whitespaces") is True else {}),
               **({"unknown": unknown} if unknown is not None else {})}
    return pieces, options, True


# tokenizer.ggml.model -> what reads a vocabulary of that kind: (metadata, the tokens, the type of a token by its id) ->
# (the pieces as tokenizer_bin() takes them, the engine's options, whether a space is written U+2581), or None for a
# vocabulary it cannot read (refused as one of a kind that is not read)
GGUF_VOCABULARIES = {"gpt2": gguf_bytebpe, "llama": gguf_sentencepiece}
