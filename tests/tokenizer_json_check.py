"""T257: a real tokenizer.json, read by the converter and the engine, against the tokenizers library reading the same
file (and against sentencepiece reading the model's own file, where the folder has a tokenizer.model): the ids of
many texts, one by one. Run by hand (the files are megabytes and another's; pytest has made-up vocabularies of the
same shapes, tests/test_spelled_bpe.py):

    python tests/tokenizer_json_check.py <a folder with tokenizer.json> ... [--texts <a file> ...] [--planes 17]

The texts: the nine prompts of tests/format_check.py; the lines of the files named (the repository's own notes and
sources by default: Japanese, English, code, addresses, tables); the boundaries the pre-tokenizers' tests have
(conftest's TEXTS, runs of spaces, tabs and line breaks, emoji); every character of the Basic Multilingual Plane
between two letters, alone and after a space (--planes 17: every code point); random texts of the vocabulary's own
pieces and characters, with spaces and the added tokens thrown in; and texts with every added token in them. An
added token is one token wherever it is written (the engine is handed all of them, the longest first, as the
converter names them). What the engine decodes is held to what the library decodes for the texts without added
tokens.

Any text whose ids differ from the tokenizers library's fails (exit 1) and is printed. sentencepiece's differences are
counted, sorted by what is written in the text and printed, and fail nothing: its file knows no special token, and
spells one out where it is written.
"""
import json
import random
import sys
import time
from pathlib import Path

from tree import python_folder

sys.path.insert(0, python_folder())
from conftest import CORPUS, TEXTS, detokenize  # noqa: E402
from convert.tokenizer import tokenizer_bin, tokenizer_json_options, tokenizer_json_pieces  # noqa: E402
from engine.tokenizer import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ["これからの流行りを3つ挙げてください。", "What will be popular next? Name three things.",
           "  leading and trailing spaces  ", "trailing only \n", "改行\nを含む\n\n文", "A", "ＡＢＣ１２３ｶﾀｶﾅ①",
           "Hello  world\ttab", "<think> and <|im_end|> typed"]  # tests/format_check.py's
EDGES = ["", " ", "  ", "   ", " " * 40, "\n" * 40, "\t" * 40, "a" + " " * 33 + "b", "a" + "\n" * 33 + "b", "▁", "▁▁ a",
         "a▁▁▁b", " a", "a ", " a ", "a  b", "a\n b", "a \nb", "\n\n\n", "\r\n\r\n", "x\ty", "🙂", "👨‍👩‍👧‍👦 family", "🇯🇵🇺🇸",
         "é", "́", "﻿a", "\x00", "a\x00b", "\x7f", "퟿", "<0x0A>", "<0xE3><0x81>", "<unk>", "<s></s>",
         "<bos><eos><pad>", "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n", "<|im_start|>user\nhi<|im_end|>\n",
         "<unused0><unused12> <unused6241>x", "[multimodal]", "<mask>", "https://example.com/a/b?c=1&d=2#frag",
         "def f(x):\n    return x  # 日本語のコメント\n", "<div class=\"a\"><p>hello</p></div>\n", "１２３４５６７８９０", "3.14159 1,234,567",
         "ｶﾞｷﾞｸﾞ", "㍻㋿", "ǅ ǆ ß ẞ ſ", "ﬁﬂ", "…—–‐", "　全角の空白　", "ゟ", "𠮷野家", "\U0001f600\U00029E3D"]
DEFAULT_TEXTS = ["TASKS.md", "AGENTS.md", "README.md", "docs/performance.md", "src/python/engine/tokenizer.py",
                 "src/runtime/forward/engine.ts", "src/pages/index.astro"]


def read(folder):
    """(the engine's tokenizer and the specials to hand it, the tokenizers library's, sentencepiece's or None, the
    added tokens, the pieces' texts) of a folder's files."""
    from tokenizers import Tokenizer as Real
    spec = json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))
    options = tokenizer_json_options(spec)
    pieces = list(tokenizer_json_pieces(spec))
    started = time.perf_counter()
    engine = Tokenizer(tokenizer_bin(pieces, len(pieces), spaces=options["tokenizer_kind"] != "bytebpe"), len(pieces),
                       **{{"tokenizer_kind": "kind"}.get(key, key): value for key, value in options.items()})
    built = time.perf_counter() - started
    added = [token["content"] for token in spec["added_tokens"]]
    specials = tuple(sorted(added, key=lambda token: (-len(token), token)))
    model = None
    if (folder / "tokenizer.model").exists():
        import sentencepiece
        model = sentencepiece.SentencePieceProcessor(model_file=str(folder / "tokenizer.model"))
    print(f"{folder}: {options}, {len(pieces):,} pieces, {len(added):,} added tokens "
          f"({sum(not token['special'] for token in spec['added_tokens']):,} of them not special), the engine's tokenizer made in {built * 1e3:.0f} ms")
    return engine, specials, Real.from_file(str(folder / "tokenizer.json")), model, added, [text for text, _, _ in pieces]


def texts_of(files, pieces, added, planes, seed=257):
    """{a kind of text: the texts}"""
    lines = [line for file in files for line in Path(file).read_text(encoding="utf-8").split("\n") if line.strip()]
    whole = [Path(file).read_text(encoding="utf-8")[:20000] for file in files]
    letters = [chr(code) for code in range(planes << 16) if not 0xD800 <= code < 0xE000]
    rng = random.Random(seed)
    plain = [piece for piece in pieces if piece and not (piece.startswith("<") and piece.endswith(">"))]
    spaces = [" ", " ", "  ", "\n", "\t", "▁", "   ", "\n\n"]

    def drawn(tokens):
        return "".join(rng.choice(spaces) if rng.random() < 0.2 else rng.choice(tokens) if rng.random() < 0.1 and tokens
                       else rng.choice(letters[:0x3100]) if rng.random() < 0.2 else rng.choice(plain) for _ in range(rng.randint(1, 30)))

    some = rng.sample(added, min(len(added), 400))
    return {"the nine prompts": PROMPTS, "boundaries": [*TEXTS, *EDGES, CORPUS * 3], "lines": lines, "files, their first 20,000 characters": whole,
            "a character between two letters": ["a" + char + "b" for char in letters],
            "a character alone": letters, "a character after a space": [" " + char for char in letters[:0x10000]],
            "random texts": [drawn(()) for _ in range(20000)], "random texts with added tokens": [drawn(added) for _ in range(20000)],
            "an added token in a text": [f"a{token}b {token} {token}{token}\n{token}" for token in some]}


def check(folder, files, planes):
    engine, specials, real, model, added, pieces = read(folder)
    wrong, total, others = [], 0, []
    for kind, texts in texts_of(files, pieces, added, planes).items():
        started, differ, apart, back = time.perf_counter(), 0, 0, 0
        for text in texts:
            want = real.encode(text, add_special_tokens=False).ids
            got = engine.encode(text, specials)
            if got != want:
                differ += 1
                wrong.append((kind, text, got, want))
            elif not any(token in text for token in added) and "▁" not in text:
                # what the engine writes back is what the library does: the text itself, but for what the model's
                # normalizer changed and the bytes that are no text. (U+2581 is a space to both, and comes back one.)
                back += detokenize(engine, got, bos=-1) != real.decode(want, skip_special_tokens=False)
            if model is not None and model.encode(text) != want:
                apart += 1
                others.append((kind, text, model.encode(text), want))
        total += len(texts)
        print(f"  {kind}: {len(texts):,} texts, {differ} with other ids than the tokenizers library's, {back} decoded otherwise"
              + (f", sentencepiece's own differ for {apart}" if model is not None else "") + f" ({time.perf_counter() - started:.0f} s)")
        wrong += [(kind, "decoded otherwise", back, 0)] if back else []
    for kind, text, got, want in wrong[:40]:
        print(f"  WRONG ({kind}) {text[:100]!r}\n    the engine    {got[:30] if isinstance(got, list) else got}\n    the tokenizers {want[:30] if isinstance(want, list) else want}")
    if others:
        special = [token["content"] for token in json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))["added_tokens"] if token["special"]]
        kinds = {"with a special token written in it": lambda text: any(token in text for token in special),
                 "with U+2581 written in it": lambda text: "\u2581" in text, "another": lambda text: True}
        sorted_out = {}
        for other in others:
            sorted_out.setdefault(next(kind for kind, is_one in kinds.items() if is_one(other[1])), []).append(other)
        print(f"  sentencepiece: {len(others)} texts differ from the tokenizers library: "
              + ", ".join(f"{len(found)} {kind}" for kind, found in sorted_out.items()))
        for name, found in sorted_out.items():
            for kind, text, got, want in found[:4 if name != "another" else 40]:
                at = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
                print(f"    {name} ({kind}) {text[:60]!r} at {at}: sentencepiece {[model.id_to_piece(id) for id in got[max(0, at - 1):at + 4]]}, "
                      f"the tokenizers {[real.id_to_token(id) for id in want[max(0, at - 1):at + 4]]}")
    print(f"  {folder}: {total:,} texts, {len(wrong)} wrong")
    return not wrong


def main(arguments):
    planes = int(arguments[arguments.index("--planes") + 1]) if "--planes" in arguments else 1
    files = arguments[arguments.index("--texts") + 1:] if "--texts" in arguments else [ROOT / name for name in DEFAULT_TEXTS]
    files = [file for file in files if not str(file).startswith("--") and str(file) != str(planes)]
    folders = [Path(argument) for argument in (arguments[:arguments.index("--texts")] if "--texts" in arguments else arguments)
               if not argument.startswith("--") and argument != str(planes)]
    if not folders:
        raise SystemExit(__doc__)
    return 0 if all([check(folder, files, planes) for folder in folders]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
