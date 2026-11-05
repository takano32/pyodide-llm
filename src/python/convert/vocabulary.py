# A model's vocabulary, read (T640.1): what each kind of tokenizer file gives the conversion, as one value. The
# conversion (convert/conversion.py) reads the vocabulary first, and the rules of how a text begins and where an
# answer stops (convert/beginning.py) read only this value.
import json
import struct
from typing import NamedTuple

from convert.template import config_token
from convert.gguf import gguf_tokenizer
from engine.tokenizer import PLAMO_SPECIAL
from convert.tokenizer import (Later, described_options, prefixed_texts, sentencepiece_charsmap, sentencepiece_options,
                               sentencepiece_pieces, sentencepiece_specials, tokenizer_bin, tokenizer_json_charsmap,
                               tokenizer_json_options, tokenizer_json_pieces, with_a_row, written_out)


def described(tokenizer_config):
    """A tokenizer_config.json, read: its text, its bytes or the dict itself. {} for none and for one that is no JSON
    or no object."""
    try:
        config = json.loads(tokenizer_config) if isinstance(tokenizer_config, (str, bytes)) else tokenizer_config
    except ValueError:
        config = None
    return config if isinstance(config, dict) else {}


class Vocabulary(NamedTuple):
    """tokenizer: the bytes of tokenizer.bin. options: the tokenizer's, as the engine gets them. described: the
    tokenizer_config.json, read (described()). specials: the tokenizer's special tokens, the ones a chat template
    writes between the turns. added: the added tokens that are not special, one token wherever they are written
    (T143). bos: the id of the BOS the tokenizer names, where it names one (T143). dropped (T509): {id: text} of the
    added tokens a tokenizer.json has past the model's last row, which convert.tokenizer's with_a_row() leaves out."""
    tokenizer: bytes
    options: dict
    described: dict
    specials: tuple = ()
    added: tuple = ()
    bos: object = None
    dropped: object = None


def named_bos(pieces, said):
    """T143: the BOS is the token the tokenizer names, which transformers begins a text with, where config.json says
    another: DeepSeek-R1's Distill says 151643 there, its end of a sentence, and <｜begin▁of▁sentence｜> (151646)
    in tokenizer_config.json (T138's review: perplexity 2.5 to 2.7 times higher with the former)"""
    named = config_token(said, "bos_token")
    return next((id for id, (text, _, _) in enumerate(pieces) if named and text == named), None)


def tokenizer_json(data, vocab_size, said, last):
    """The vocabulary of a tokenizer.json (its bytes). last (T257): see Conversion."""
    try:
        parsed = json.loads(data)
    except ValueError:
        raise ValueError("tokenizer.json is not JSON.") from None
    if not last and written_out(parsed) and prefixed_texts(parsed):
        raise Later("This tokenizer.json writes out a sentencepiece model, whose own file is read first.")
    options = tokenizer_json_options(parsed)
    every = list(tokenizer_json_pieces(parsed))
    pieces = with_a_row(every, parsed, vocab_size)
    # (T509: the added tokens past the model's last row, which beginning_and_stops() holds to what the model uses)
    dropped = {id: text for id, (text, _, _) in enumerate(every[len(pieces):], len(pieces))}
    tokenizer = tokenizer_bin(pieces, vocab_size, spaces=options["tokenizer_kind"] != "bytebpe",
                              charsmap=tokenizer_json_charsmap(parsed))
    added = [token for token in parsed.get("added_tokens", []) if token.get("id", 0) < vocab_size]
    specials = [token["content"] for token in added if token.get("special")]
    # T143: the added tokens it does not call special are read as one token wherever they are too, by the real
    # tokenizers before it splits the text: Qwen3's <think>, Pythia's runs of 2 to 24 spaces (T215's finding)
    added = [token["content"] for token in added if not token.get("special")]
    return Vocabulary(tokenizer, options, said, specials, added, named_bos(pieces, said), dropped)


def sentencepiece(data, vocab_size, said, last):
    """The vocabulary of a sentencepiece model (its bytes)."""
    pieces = list(sentencepiece_pieces(data))
    tokenizer = tokenizer_bin(pieces, vocab_size, charsmap=sentencepiece_charsmap(data))
    # (T265, T308) and what only tokenizer_config.json says of a sentencepiece model. A model with no dummy
    # prefix has none after a special token either (T400: its "never" over a legacy tokenizer's "wanting")
    own = sentencepiece_options(data)
    options = {**own, **described_options(said), **({"prefixed": "never"} if own.get("prefixed") else {})}
    return Vocabulary(tokenizer, options, said, sentencepiece_specials(data), [], named_bos(pieces, said), {})


def plamo_rows(data):
    """T649.1: the rows of PLaMo's tokenizer.jsonl (its bytes): [piece, score, kind] on each line, the kind NORMAL,
    BYTE (<0xNN>, one for each of the 256 bytes), CONTROL or UNKNOWN"""
    try:
        rows = [json.loads(line) for line in bytes(data).decode("utf-8").splitlines() if line.strip()]
    except ValueError:
        raise ValueError("tokenizer.jsonl is not a line of JSON on each line.") from None
    for row in rows:
        if not (isinstance(row, list) and len(row) >= 3 and isinstance(row[0], str) and row[2] in PLAMO_KINDS
                and isinstance(row[1], (int, float)) and not isinstance(row[1], bool)):
            raise ValueError(f"tokenizer.jsonl has a line that is not [piece, score, kind]: {json.dumps(row)[:80]}")
        # the engine adds the scores up as round(score * 1e4) (engine.tokenizer's plamo_pieces()), of what tokenizer.bin
        # holds: a float32
        if round(struct.unpack("<f", struct.pack("<f", row[1]))[0] * 1e4) != round(row[1] * 1e4):
            raise ValueError(f"tokenizer.jsonl has a score that a float32 does not hold to four decimals: {row[1]!r}")
    # (PLaMo's own takes the last of two rows of one piece, the engine the first)
    if len({row[0] for row in rows}) < len(rows):
        raise ValueError("tokenizer.jsonl has a piece twice.")
    found = {row[0] for row in rows if row[2] == "BYTE"}
    if found != {f"<0x{byte:02X}>" for byte in range(256)}:
        raise ValueError("tokenizer.jsonl does not have a piece for each of the 256 bytes.")
    return rows


PLAMO_KINDS = ("NORMAL", "BYTE", "CONTROL", "UNKNOWN")


def plamo(data, vocab_size, said, last):
    """T649.1: the vocabulary of PLaMo's tokenizer.jsonl (its bytes), read by the engine's "plamo" kind. A byte piece
    never matches text; every other piece does, a control one too (with a score of 0, as PLaMo's search has it).
    The specials are the control pieces PLaMo cuts a text around (<|plamo:…|>); the others its search finds as it
    finds any piece. breaks: what tokenizer_config.json says of the runs to cut around."""
    rows = plamo_rows(data)
    pieces = [(text, float(score), kind != "BYTE") for text, score, kind, *_ in rows]
    breaks = {name: said[key] for name, key in (("spaces", "break_around_consecutive_spaces_threshold"),
                                                ("repeats", "break_around_repeated_chars_threshold"))
              if isinstance(said.get(key), int) and not isinstance(said.get(key), bool)}
    options = {"tokenizer_kind": "plamo", **({"breaks": breaks} if breaks else {})}
    specials = [text for text, _, kind, *_ in rows if kind in ("CONTROL", "UNKNOWN") and PLAMO_SPECIAL.fullmatch(text)]
    return Vocabulary(tokenizer_bin(pieces, vocab_size, spaces=False), options, said, specials, [],
                      named_bos(pieces, said), {})


# the kind of a tokenizer file -> what reads it: (its bytes, config.json's vocab_size, the tokenizer_config.json read,
# last) -> a Vocabulary. A GGUF's vocabulary is gguf_vocabulary()'s (its kinds are convert/gguf.py's GGUF_VOCABULARIES)
VOCABULARIES = {"tokenizer.json": tokenizer_json, "sentencepiece": sentencepiece, "tokenizer.jsonl": plamo}


def vocabulary_kind(name):
    """The kind of a tokenizer file by its name: a .json is a tokenizer.json, a .jsonl PLaMo's tokenizer.jsonl (T649.1),
    anything else a sentencepiece model."""
    name = name.lower()
    return "tokenizer.json" if name.endswith(".json") else "tokenizer.jsonl" if name.endswith(".jsonl") else "sentencepiece"


def vocabulary_of(data, name, vocab_size, tokenizer_config=None, last=True):
    """The Vocabulary of a tokenizer file (its bytes and its name)."""
    return VOCABULARIES[vocabulary_kind(name)](data, vocab_size, described(tokenizer_config), last)


def gguf_vocabulary(metadata, vocab_size):
    """The Vocabulary a GGUF holds (gguf_tokenizer()'s): the BOS it names is the config's (gguf_model()), none here."""
    tokenizer, options, tokenizer_config, specials, added = gguf_tokenizer(metadata, vocab_size)
    return Vocabulary(tokenizer, options, described(tokenizer_config), specials, added)
