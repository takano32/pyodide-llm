# The tokenizer's conversion: tokenizer.json (Unigram, byte-level BPE, a sentencepiece BPE model written out) and
# sentencepiece models to tokenizer.bin and the options the engine needs.
import json
import struct

from engine.tokenizer import CHARSMAP

UNMATCHABLE = -1e9  # control, unknown and byte pieces must never match user text: llama2_numpy.py skips such scores


class Later(ValueError):
    """T257: what a tokenizer.json that writes out a sentencepiece BPE model with its dummy prefix (a Llama
    tokenizer's: a Mistral's, CAT-Translate 7B's) is put off with while the model's other tokenizers are still to be
    tried (Conversion(last=False), which is how convert.conduct asks for each candidate). Before which stretches of
    text the prefix stands is not the file's alone to say: transformers reads it off tokenizer_config.json's legacy
    (T308), a tokenizer.json written by an older transformers prepends to all of them whatever that says, and the
    model's own file, with its scores, is what the list's models of this kind were held to the real tokenizers with.
    Read where no other tokenizer is there, or none that can be read. One without the prefix (a Gemma's) has no such
    question, and is read at once: the model's own file read here would get a prefix it has not."""


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


def with_a_row(pieces, tokenizer, vocab_size):
    """T257: the pieces of a tokenizer.json (tokenizer_json_pieces()'s) that the model has a row for. An added token
    past the model's last row is no token of this model: a text-only Gemma 3 names <image_soft_token> one past its
    vocabulary of 262,144, the token of the models that see. Only where the tokenizer's own vocabulary fits: a model
    with fewer rows than that is another's (tokenizer_bin() refuses it)."""
    own = tokenizer["model"]["vocab"]
    fits = isinstance(own, dict) and max(own.values(), default=0) < vocab_size
    return pieces[:vocab_size] if fits else pieces


def tokenizer_json_bpe_pieces(tokenizer):
    """Hugging Face's byte-level BPE (GPT-2, SmolLM2, Qwen). The pieces are written in the byte <-> character
    table, and the score is minus the rank of the merge that makes the piece: the engine merges the best-scoring
    pair, which is then the same as applying the merge with the lowest rank. A piece no merge makes (a single
    character, an added token) never starts a merge, so it is not matchable.
    T257: and a sentencepiece BPE model written out (written_out()), whose pieces are sentencepiece's own text and
    get the same scores, once its merges are known to be ones the engine follows (merged_by_text())."""
    model = tokenizer["model"]
    ranks, merges = {}, [merge if isinstance(merge, list) else merge.split(" ") for merge in model["merges"]]
    for rank, (left, right) in enumerate(merges):
        ranks.setdefault(left + right, -float(rank))
    texts = {id: text for text, id in model["vocab"].items()}
    for token in tokenizer["added_tokens"]:
        texts.setdefault(token["id"], token["content"])
    if written_out(tokenizer):
        merged_by_text(set(texts.values()), merges)
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
        prefixed = prefixed_texts(tokenizer)
        return {"tokenizer_kind": "unigram", "nfkc": nfkc, **({"prefixed": prefixed} if prefixed else {})}
    if written_out(tokenizer):
        return written_out_options(tokenizer)
    return {"tokenizer_kind": "bytebpe", "nfkc": nfkc, "nfc": '"NFC"' in normalizers,
            "pretokenizer": pretokenizer_name(tokenizer.get("pre_tokenizer")),
            "ignore_merges": bool(tokenizer["model"].get("ignore_merges"))}


# T257: a sentencepiece BPE model written out as a tokenizer.json (a Gemma's, CAT-Translate 7B's, a Llama 2's): a "BPE"
# whose pieces are sentencepiece's, with U+2581 for the space, and whose merges are every way to join two pieces into a
# third, in the order of the model's scores. Nothing splits the text: the normalizer makes each space a U+2581 and the
# merges run over the whole of it, as sentencepiece's do.
SPACE_WRITTEN = {"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"}  # the normalizer's step
PREPENDED = {"type": "Prepend", "prepend": "\u2581"}  # and the dummy prefix, where the model has one
# a Gemma's pre_tokenizer: it would cut the text after each space, of which the normalizer has left none
SPLIT_AT_SPACES = {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": False}


def written_out(tokenizer):
    """Whether a BPE tokenizer.json is a sentencepiece model written out, by how it reads a text: a normalizer of the
    two steps above and nothing else (the space one of them), and no pre_tokenizer or a Gemma's. Any other is a
    byte-level BPE or one the engine does not know (pretokenizer_name() says which)."""
    normalizer = tokenizer.get("normalizer") or {}
    steps = normalizer.get("normalizers", [normalizer]) if normalizer.get("type") == "Sequence" else [normalizer]
    return SPACE_WRITTEN in steps and all(step in (SPACE_WRITTEN, PREPENDED) for step in steps) \
        and (tokenizer.get("pre_tokenizer") or SPLIT_AT_SPACES) == SPLIT_AT_SPACES


def written_out_options(tokenizer):
    """The engine's options for a sentencepiece BPE model written out: llama2.c's BPE ("bpe": the pieces' scores are
    minus the ranks of their merges, tokenizer_json_bpe_pieces()), with the dummy prefix before every stretch of text
    where the normalizer prepends it (a Llama tokenizer's file: CAT-Translate 7B's) and before none where it does not
    (a Gemma's). A character the vocabulary lacks is spelled with the byte pieces, as the engine's BPE spells it: the
    model must say so (byte_fallback) and have all 256. Its other ways (a piece taken whole before the merges, marks
    at the edges of a word, merges dropped at random) are none of sentencepiece's."""
    model = tokenizer["model"]
    if not model.get("byte_fallback") or not all("<0x%02X>" % byte in model["vocab"] for byte in range(256)):
        raise ValueError("This tokenizer.json spells a character it lacks in a way the engine does not know.")
    if model.get("ignore_merges") or model.get("dropout") or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        raise ValueError("This tokenizer.json merges its pieces in a way the engine does not know.")
    return {"tokenizer_kind": "bpe", "prefixed": prefixed_texts(tokenizer) or "never"}


def merged_by_text(pieces, merges):
    """T257: refuses the merges that the engine's BPE would not follow. The engine joins two pieces wherever what they
    spell together is a piece, the one with the best score first (llama2.c's, which is sentencepiece's); a
    tokenizer.json joins the pairs its merges name, the first-named first. The two are one where every way to spell
    a piece with two pieces is a merge, which is how a sentencepiece model is written out (transformers'
    SentencePieceExtractor lists them all, by the score of what they make): held to here, since a pair the merges
    lack would be joined by the engine all the same. And no piece is written with a space: the engine's pieces spell
    U+2581 so, and the two would be one piece. pieces: the texts of all of them (a set), merges: the pairs in order."""
    named = {(left, right) for left, right in merges}
    if any(" " in piece for piece in pieces) or not all(left in pieces and right in pieces and left + right in pieces for left, right in named) \
            or any(piece[:cut] in pieces and piece[cut:] in pieces and (piece[:cut], piece[cut:]) not in named
                   for piece in pieces for cut in range(1, len(piece))):
        raise ValueError("This tokenizer.json merges its pieces in a way the engine does not know.")


def prefixed_texts(tokenizer):
    """T308: which stretches of text this tokenizer.json puts sentencepiece's "▁" before, besides the first: the text
    after a special token too (the tokenizers library normalizes and pre-tokenizes each stretch between two added
    tokens on its own). "every": a normalizer that prepends it (Prepend) or replaces the start of the text with it (a
    Replace of a pattern that ends in ^: llm-jp's "(?<!\n)^"). "wanting": a Metaspace pre-tokenizer that always
    prepends, which leaves a stretch that begins with a space as it is. False: neither. Said to the engine as
    Llama(prefixed=), only where it is one of the two (each as the tokenizers library does it: tests/test_prefixed.py)."""
    steps, found = [tokenizer.get("normalizer") or {}, tokenizer.get("pre_tokenizer") or {}], set()
    while steps:
        step = steps.pop()
        steps += [*(step.get("normalizers") or []), *(step.get("pretokenizers") or [])]
        kind, pattern = step.get("type"), step.get("pattern") or {}
        if (kind == "Prepend" and step.get("prepend") == "▁") or \
                (kind == "Replace" and step.get("content") == "▁" and str(pattern.get("Regex", "")).endswith("^")):
            found.add("every")
        # (before prepend_scheme the key was add_prefix_space, which prepended to every stretch as "always" does)
        if kind == "Metaspace" and step.get("prepend_scheme", "always" if step.get("add_prefix_space", True) else "never") == "always":
            found.add("wanting")
    return "every" if "every" in found else "wanting" if found else False


def described_options(config):
    """What a sentencepiece model's tokenizer_config.json says of how its text is read, which the model's own file does
    not (config: read, a dict), as Llama()'s options and only where it is so.
    lowercase (T265): do_lower_case, which transformers' slow tokenizers apply before sentencepiece sees the text
    (rinna's japanese-gpt2: a vocabulary without capital Latin letters).
    prefixed (T308): a Llama tokenizer that is legacy (said, or not said: transformers' default) puts the dummy prefix
    before the text after a special token as well, unless that text begins with a space ("wanting": transformers
    makes a Metaspace that always prepends of it, and tests/format_check.py holds the list's two to that); with legacy
    false, before the first text only."""
    legacy = str(config.get("tokenizer_class", "")).startswith("LlamaTokenizer") and config.get("legacy", True) is not False
    return {**({"lowercase": True} if config.get("do_lower_case") is True else {}), **({"prefixed": "wanting"} if legacy else {})}


def piece_ids(tokenizer, vocab_size):
    """{a piece's text as the engine has it (bytes): its id, the first of equal pieces} of a tokenizer.bin: what
    Tokenizer.encode() looks a special token up in."""
    ids, offset = {}, 4
    for id in range(vocab_size):
        (length,) = struct.unpack_from("<i", tokenizer, offset + 4)
        ids.setdefault(bytes(tokenizer[offset + 8:offset + 8 + length]), id)
        offset += 8 + length
    return ids


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
