# The tokenizer's conversion: tokenizer.json (Unigram, byte-level BPE) and sentencepiece models to tokenizer.bin and
# the options the engine needs.
import json
import struct

from llama2_numpy import CHARSMAP

UNMATCHABLE = -1e9  # control, unknown and byte pieces must never match user text: llama2_numpy.py skips such scores


def tokenizer_bin(pieces, vocab_size, spaces=True, charsmap=b""):
    """llama2.c's tokenizer.bin from (text, score, matchable) pieces. spaces: a sentencepiece vocabulary writes a space
    as U+2581, which the engine's pieces spell " ". A byte-level one writes it as its byte's character (Ġ), and a
    U+2581 there is an added token's own (DeepSeek's <｜begin▁of▁sentence｜>), kept as it is (T143). charsmap: a
    sentencepiece model's precompiled_charsmap, its normalizer, which goes after the pieces (T216; llama2_numpy's
    Charsmap reads it, and llama2.c's reader stops at the last piece)."""
    rows = [(score if matchable else UNMATCHABLE, (text.replace("▁", " ") if spaces else text).encode("utf-8"))
            for text, score, matchable in pieces]
    if len(rows) > vocab_size:
        raise ValueError(f"The tokenizer has {len(rows)} pieces, but the model has a vocabulary of {vocab_size}.")
    # A model can have a few more embedding rows than the tokenizer has pieces (padding to a round number). Many
    # more means the tokenizer of another model, which would convert fine and then write nonsense.
    if len(rows) < 0.9 * vocab_size:
        raise ValueError(f"The tokenizer has {len(rows)} pieces, but the model has a vocabulary of {vocab_size}: "
                         f"they do not belong together.")
    rows += [(UNMATCHABLE, b"")] * (vocab_size - len(rows))
    out = [struct.pack("<i", max(len(text) for _, text in rows))]
    out += [struct.pack("<fi", score, len(text)) + text for score, text in rows]
    if charsmap:
        out.append(CHARSMAP + struct.pack("<I", len(charsmap)) + bytes(charsmap))
    return b"".join(out)


def tokenizer_kind_of(tokenizer):
    """What kind of model this tokenizer.json holds. The oldest ones (GPT-2's own, version 1.0) have no "type",
    and are told apart by what they carry: merges for a BPE, a list of (piece, score) for a Unigram."""
    model = tokenizer["model"]
    if "type" in model:
        return model["type"]
    return "BPE" if "merges" in model else "Unigram"


def tokenizer_json_pieces(tokenizer):
    kind = tokenizer_kind_of(tokenizer)
    if kind == "BPE":
        yield from tokenizer_json_bpe_pieces(tokenizer)
        return
    if kind != "Unigram":
        raise ValueError(f"This tokenizer.json is a {kind} model: only Unigram and byte-level BPE ones are "
                         f"supported (or a sentencepiece tokenizer.model).")
    special = {token["content"] for token in tokenizer["added_tokens"] if token["special"]}
    for id, (text, score) in enumerate(tokenizer["model"]["vocab"]):
        is_byte = len(text) == 6 and text.startswith("<0x") and text.endswith(">")
        yield text, score, not (is_byte or text in special or id == tokenizer["model"].get("unk_id"))


def tokenizer_json_bpe_pieces(tokenizer):
    """Hugging Face's byte-level BPE (GPT-2, SmolLM2, Qwen). The pieces are written in the byte <-> character
    table, and the score is minus the rank of the merge that makes the piece: the engine merges the best-scoring
    pair, which is then the same as applying the merge with the lowest rank. A piece no merge makes (a single
    character, an added token) never starts a merge, so it is not matchable."""
    model = tokenizer["model"]
    ranks = {}
    for rank, merge in enumerate(model["merges"]):
        left, right = merge if isinstance(merge, list) else merge.split(" ")
        ranks.setdefault(left + right, -float(rank))
    texts = {id: text for text, id in model["vocab"].items()}
    for token in tokenizer["added_tokens"]:
        texts.setdefault(token["id"], token["content"])
    special = {token["content"] for token in tokenizer["added_tokens"] if token["special"]}
    for id in range(max(texts) + 1):
        text = texts.get(id)
        if text is None:
            raise ValueError(f"This tokenizer.json has no piece with id {id}.")
        yield text, ranks.get(text, UNMATCHABLE), text in ranks and text not in special


def tokenizer_json_options(tokenizer):
    """What the engine has to know about this tokenizer: Llama(tokenizer_kind=, nfkc=, nfc=, pretokenizer=). A
    "Precompiled" normalizer is sentencepiece's own map, which goes into tokenizer.bin (tokenizer_json_charsmap)."""
    normalizers = json.dumps(tokenizer.get("normalizer") or {})
    nfkc = '"NFKC"' in normalizers
    if tokenizer_kind_of(tokenizer) != "BPE":
        return {"tokenizer_kind": "unigram", "nfkc": nfkc}
    return {"tokenizer_kind": "bytebpe", "nfkc": nfkc, "nfc": '"NFC"' in normalizers,
            "pretokenizer": pretokenizer_name(tokenizer.get("pre_tokenizer")),
            "ignore_merges": bool(tokenizer["model"].get("ignore_merges"))}


def tokenizer_json_charsmap(tokenizer):
    """The precompiled_charsmap of a tokenizer.json's "Precompiled" normalizer (sentencepiece's, T216), or b""."""
    import base64
    steps = [tokenizer.get("normalizer") or {}]
    while steps:
        step = steps.pop()
        steps += step.get("normalizers") or []
        if step.get("type") == "Precompiled" and step.get("precompiled_charsmap"):
            return base64.b64decode(step["precompiled_charsmap"])
    return b""


# The engine writes these out by hand (llama2_numpy.pretokenize), so only the patterns it knows are accepted.
PRETOKENIZERS = {
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+": "llama3",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+": "qwen",
    # T229: Qwen3.5's takes the combining marks into the word
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+": "qwen35",
}


# T254: MiniCPM5's two Splits (openbmb/MiniCPM5-1B's tokenizer.json): the numbers cut off three at a time, then Llama
# 3's pattern with \p{N}+ on each piece, and a ByteLevel that splits no more. llama2_numpy.pretokenize's "minicpm5"
STAGED_PRETOKENIZERS = {
    (r"\p{N}{1,3}",
     r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"): "minicpm5",
}


def pretokenizer_name(spec):
    steps = spec.get("pretokenizers", [spec]) if spec else []
    kinds = [step["type"] for step in steps]
    patterns = [step["pattern"].get("Regex") for step in steps if step["type"] == "Split"]
    if tuple(patterns) in STAGED_PRETOKENIZERS:
        # every Split keeps what it matches as a piece of its own (Isolated) and matches what its pattern says (no
        # invert), and the ByteLevel after them does not split again or put a space in front
        plain = all(step.get("behavior") == "Isolated" and not step.get("invert") for step in steps if step["type"] == "Split")
        rest = [step for step in steps if step["type"] != "Split"]
        if plain and kinds[:len(patterns)] == ["Split"] * len(patterns) and len(rest) == 1 and rest[0]["type"] == "ByteLevel" \
                and rest[0].get("use_regex") is False and not rest[0].get("add_prefix_space"):
            return STAGED_PRETOKENIZERS[tuple(patterns)]
        raise ValueError(f"This tokenizer.json splits text in a way the engine does not know: {steps}")
    if patterns:
        if len(patterns) > 1 or patterns[0] not in PRETOKENIZERS:
            raise ValueError(f"This tokenizer.json splits text in a way the engine does not know: {patterns}")
        return PRETOKENIZERS[patterns[0]]
    if "ByteLevel" not in kinds or not all(kind in ("ByteLevel", "Digits") for kind in kinds):
        raise ValueError(f"This tokenizer.json splits text in a way the engine does not know: {kinds}")
    if not all(step.get("use_regex", True) for step in steps if step["type"] == "ByteLevel"):
        raise ValueError("This tokenizer.json has a ByteLevel pre-tokenizer without its regex, which the engine "
                         "does not know.")
    if any(step["type"] == "ByteLevel" and step.get("add_prefix_space") for step in steps):
        raise ValueError("This tokenizer.json adds a space in front of the text, which the engine does not do.")
    digits = [step for step in steps if step["type"] == "Digits"]
    if digits and not all(step.get("individual_digits") for step in digits):
        raise ValueError("This tokenizer.json groups digits in a way the engine does not know.")
    return "gpt2-digits" if digits else "gpt2"


def protobuf_fields(data):
    """Yield (field number, value) of one protobuf message; nested messages come back as bytes."""
    i = 0

    def varint():
        nonlocal i
        value = shift = 0
        while True:
            byte = data[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return value

    while i < len(data):
        key = varint()
        field, wire_type = key >> 3, key & 7
        if wire_type == 0:
            yield field, varint()
        elif wire_type == 1:
            yield field, data[i:i + 8]
            i += 8
        elif wire_type == 2:
            size = varint()
            yield field, data[i:i + size]
            i += size
        elif wire_type == 5:
            yield field, data[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")


def sentencepiece_pieces(model):
    """(text, score, matchable) of a sentencepiece model (spiece.model, tokenizer.model), given as bytes."""
    NORMAL, USER_DEFINED = 1, 4
    for field, value in protobuf_fields(model):
        if field == 1:  # ModelProto.pieces
            piece = dict(protobuf_fields(value))
            score = struct.unpack("<f", piece[2])[0] if 2 in piece else 0.0
            yield piece.get(1, b"").decode("utf-8"), score, piece.get(3, NORMAL) in (NORMAL, USER_DEFINED)


def sentencepiece_specials(model):
    """The control pieces of a sentencepiece model (<s>, </s>, sarashina's <|user|>): the special tokens a chat
    template writes between the turns. The engine's search of the vocabulary never finds them, so a template read
    from the model (T127) needs them named; without, "</s>" became four tokens of text."""
    CONTROL = 3
    specials = []
    for field, value in protobuf_fields(model):
        if field == 1:
            piece = dict(protobuf_fields(value))
            if piece.get(3) == CONTROL and piece.get(1):
                specials.append(piece[1].decode("utf-8"))
    return specials


def sentencepiece_charsmap(model):
    """A sentencepiece model's normalizer: its normalizer_spec's precompiled_charsmap (T216), b"" for none (identity:
    Llama's, Mistral's)."""
    for field, value in protobuf_fields(model):
        if field == 3:  # normalizer_spec
            return bytes(dict(protobuf_fields(value)).get(2, b""))
    return b""


def sentencepiece_options(model):
    """Llama(tokenizer_kind=, collapse=, unknown=) from the pieces and the trainer and normalizer specs of a
    sentencepiece model. Its normalizer is its own map, in tokenizer.bin (sentencepiece_charsmap, T216).

    collapse: remove_extra_whitespaces, runs of spaces made one and the ends trimmed. Said only where on: rinna's
    models are nmt_nfkc with it (and no byte pieces to spell a newline with), Llama's and Mistral's identity without
    it, tiny-lm's nfkc without it (the review of T126). unknown: the id of the unknown piece, for a model without byte
    pieces (rinna's): a character the vocabulary lacks is that piece, as in sentencepiece, and not bytes spelled with
    whatever pieces happen to be at byte + 3."""
    UNIGRAM, BPE = 1, 2
    UNKNOWN, BYTE = 2, 6
    kind, collapse = UNIGRAM, True  # sentencepiece's own defaults
    unknown, spelled, index = None, False, 0
    for field, value in protobuf_fields(model):
        if field == 1:  # a piece: its type
            piece_type = dict(protobuf_fields(value)).get(3, 1)
            unknown = index if piece_type == UNKNOWN and unknown is None else unknown
            spelled |= piece_type == BYTE
            index += 1
        elif field == 2:  # trainer_spec.model_type
            kind = dict(protobuf_fields(value)).get(3, UNIGRAM)
        elif field == 3:  # normalizer_spec: its remove_extra_whitespaces
            collapse = bool(dict(protobuf_fields(value)).get(4, 1))
    if kind not in (UNIGRAM, BPE):
        raise ValueError("This sentencepiece model is neither unigram nor BPE.")
    return {"tokenizer_kind": "unigram" if kind == UNIGRAM else "bpe", **({"collapse": True} if collapse else {}),
            **({"unknown": unknown} if unknown is not None and not spelled else {})}
