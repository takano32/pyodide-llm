# tests/spelled_bpe_cases.py (T.257): a small sentencepiece BPE model written out as a tokenizer.json, made by hand of
# a few words, the texts to read with it, the definition of what their ids are, and one number for all of them.
# pytest holds the engine, the tokenizers library and the definition to each other and to the number
# (tests/test_spelled_bpe.py); tests/smoke.ts asks the same of the engine inside Pyodide. Neither pytest nor the
# converter is imported here.
import hashlib

WORDS = ["the", "then", "there", "they", "this", "that", "abab", "aba", "bab", "baab", "dog", "dogs", "日本語", "本語学", "語学", "です。"]
TEXTS = ["the dog", " the dogs then", "they  bark", "ababab abba baabab", "日本語学です。", "日本 語 学", "théâtre 🙂", "this\nthat\n\nthere",
         "", " ", "   ", "a▁b", "▁▁the", "thethethe", "xyz", "<s>the</s>", "the<unk>dog", "The Dog"]
SPECIALS = ("<unk>", "</s>", "<s>")
# the SHA-256 of the ids of TEXTS as text (digest()): what a 64-bit Python's engine gives, and Pyodide's must
IDS = {False: "d3a4d84553d1c286c451bae06a899aab307216f6d436e1360a632ba9c3c0548f",
       True: "00830d95f989861c7346e642b0d19d0b8933cd8638a02a350412432e1ebd8876"}  # (by whether the file has the dummy prefix)


def spec(dummy):
    """The tokenizer.json: <unk>, <s>, </s>, the 256 byte pieces, the words' characters and U+2581, then every
    beginning of two characters or more of every word, with U+2581 in front and without; every way to join two pieces
    into a third is a merge, the shorter pieces' first. dummy: a Llama tokenizer's file, which prepends U+2581 to
    every stretch of text (and takes it off what it decodes); without, a Gemma's, with its pre_tokenizer."""
    singles = sorted({char for word in WORDS for char in word} | {"▁"})
    longer = sorted({lead + word[:end] for word in WORDS for end in range(2, len(word) + 1) for lead in ("", "▁")}
                    | {"▁" + char for char in singles}, key=lambda piece: (len(piece), piece))
    pieces = ["<unk>", "<s>", "</s>", *("<0x%02X>" % byte for byte in range(256)), *singles, *longer]
    vocab = {piece: id for id, piece in enumerate(pieces)}
    merges = [[piece[:cut], piece[cut:]] for piece in longer for cut in range(len(piece) - 1, 0, -1)
              if piece[:cut] in vocab and piece[cut:] in vocab]
    space = {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
    return {"version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [{"id": id, "content": pieces[id], "single_word": False, "lstrip": False, "rstrip": False,
                              "normalized": False, "special": True} for id in range(3)],
            "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, space]} if dummy else space,
            "pre_tokenizer": None if dummy else {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": False},
            "post_processor": None,
            "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                                                         {"type": "ByteFallback"}, {"type": "Fuse"},
                                                         *([{"type": "Strip", "content": " ", "start": 1, "stop": 0}] if dummy else [])]},
            "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
                      "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
                      "vocab": vocab, "merges": merges}}


def by_the_definition(spec, text, specials=()):
    """The ids of a text as the file defines them, the slow way: the text is cut at the special tokens (the first of
    them that is written at a place); each stretch between goes through the normalizer, is a symbol for each character
    (the byte pieces for one the vocabulary lacks), and then, as long as two neighbouring symbols are a pair the merges
    name, the pair named first is joined, the leftmost of them."""
    vocab, merges = spec["model"]["vocab"], {tuple(merge): rank for rank, merge in enumerate(spec["model"]["merges"])}
    at = min(((text.find(special), order) for order, special in enumerate(specials) if special in text), default=None)
    if at is not None:
        special = specials[at[1]]
        return by_the_definition(spec, text[:at[0]], specials) + [vocab[special]] + by_the_definition(spec, text[at[0] + len(special):], specials)
    text = ("▁" if spec["normalizer"].get("normalizers") else "") + text.replace(" ", "▁") if text else ""
    symbols = [symbol for char in text for symbol in ([char] if char in vocab else ["<0x%02X>" % byte for byte in char.encode()])]
    while True:
        ranked = [(merges[pair], place) for place, pair in enumerate(zip(symbols, symbols[1:])) if pair in merges]
        if not ranked:
            return [vocab[symbol] for symbol in symbols]
        _, place = min(ranked)
        symbols[place:place + 2] = [symbols[place] + symbols[place + 1]]


def digest(ids):
    """The SHA-256 of the ids of the texts ([[id, ...], ...]), written out as text."""
    return hashlib.sha256(repr([[int(id) for id in text] for text in ids]).encode()).hexdigest()
