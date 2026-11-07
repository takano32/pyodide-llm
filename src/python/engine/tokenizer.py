# The tokenizers: llama2.c's BPE, Unigram (Viterbi) and byte-level BPE with its pre-tokenizers, the normalization of
# a sentencepiece model's own charsmap, and the bytes and characters of GPT-2's vocabulary.
#
# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository), and it is
# derived from two works under the MIT License, whose notice follows: tairov/llama2.py
# (https://github.com/tairov/llama2.py; its LICENSE names no copyright holder) and karpathy/llama2.c
# (https://github.com/karpathy/llama2.c), Copyright (c) 2023 Andrej.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import heapq
import math
import re
import struct
import unicodedata
from typing import Callable, NamedTuple, Optional

from engine.proxy import plain

BOS = 1  # beginning-of-sequence token, also what the model emits when a story is over


def byte_chars():
    """GPT-2's byte <-> character table: every byte becomes one printable character, so that a byte-level BPE
    vocabulary is plain text. 0x20 is "\u0120", 0x0A is "\u010a"."""
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, extra = list(printable), 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {byte: chr(char) for byte, char in zip(printable, chars)}


BYTE_CHARS = byte_chars()
CHAR_BYTES = {char: byte for byte, char in BYTE_CHARS.items()}


def letter(char):
    return unicodedata.category(char)[0] == "L"


def number(char):
    return unicodedata.category(char)[0] == "N"


def mark(char):
    return unicodedata.category(char)[0] == "M"


class CharClasses(dict):
    r"""What the pre-tokenizer's patterns tell apart, one character for each character (T200): the ASCII letters,
    the apostrophe, the space, \r and \n as they are (the contractions and the line breaks name them) and ſ (U+017F,
    which (?i:'s) takes for s), any other whitespace "\t", any other letter (\p{L}) "a", a number (\p{N}) "0", a
    combining mark (\p{M}) "~" (T229: Qwen3.5's pattern takes one into a word, the others take it for anything else),
    anything else "!". A text goes through str.translate() with it, and the standard re module runs the patterns on
    what comes out: \p{L}, \p{N} and \p{M}, which re has not, become [A-Za-zſ], 0 and ~. A character is classed the
    first time it is seen.

    Whitespace is what the real tokenizers' regex (Oniguruma) calls \s (T206): str.isspace less \x1c to \x1f, which
    Oniguruma takes for neither whitespace nor a letter nor a number."""

    def __missing__(self, code):
        char = chr(code)
        if char.isascii() and char.isalpha() or char in "' \r\nſ":
            kind = char
        elif char.isspace() and not "\x1c" <= char <= "\x1f":
            kind = "\t"
        elif letter(char):
            kind = "a"
        elif number(char):
            kind = "0"
        elif mark(char):
            kind = "~"
        else:
            kind = "!"
        self[code] = kind
        return kind


CHAR_CLASSES = CharClasses()
# pretokenize()'s patterns on the classes: \p{L} is [A-Za-zſ], \p{N} is 0, \s is [ \t\r\n], \p{M} is ~, anything
# else ['!] (['!~] where the pattern does not name the marks).
# Under (?i) re takes ſ for s, as Oniguruma does; GPT-2's case-sensitive 's does not.
CONTRACTED = "'s|'t|'re|'ve|'m|'ll|'d"
SPACES = r"[ \t\r\n]+(?![^ \t\r\n])|[ \t\r\n]+"
PATTERNS = {
    "gpt2": re.compile(CONTRACTED + r"| ?[A-Za-zſ]+| ?0+| ?['!~]+|" + SPACES),
    "qwen": re.compile(f"(?i:{CONTRACTED})" + r"|[^\r\nA-Za-zſ0]?[A-Za-zſ]+|0| ?['!~]+[\r\n]*|[ \t\r\n]*[\r\n]+|" + SPACES),
    "llama3": re.compile(f"(?i:{CONTRACTED})" + r"|[^\r\nA-Za-zſ0]?[A-Za-zſ]+|0{1,3}| ?['!~]+[\r\n]*|[ \t\r\n]*[\r\n]+|" + SPACES),
    "qwen35": re.compile(f"(?i:{CONTRACTED})" + r"|[^\r\nA-Za-zſ0]?[A-Za-zſ~]+|0| ?['!]+[\r\n]*|[ \t\r\n]*[\r\n]+|" + SPACES),
}
DIGITS = re.compile("0|[^0]+")  # Digits(individual_digits) on the classes: every number a piece of its own
THREES = re.compile("0{1,3}|[^0]+")  # Split(\p{N}{1,3}) on the classes: numbers cut off, three at a time (T254)
# the pre-tokenizers of two stages: what cuts the text first, and the pattern that then runs on each piece alone
STAGED = {"gpt2-digits": (DIGITS, PATTERNS["gpt2"]), "minicpm5": (THREES, PATTERNS["llama3"])}


class SparkClasses(dict):
    r"""T340: the classes Spark-X2.5's pre-tokenizer tells apart, which are others than CharClasses' (its patterns name
    punctuation and symbols, \p{P} and \p{S}, apart from what is neither; the ASCII letters apart from the others; and a
    block of the CJK ideographs and the kana, whatever their categories): the ASCII letters "A", any other letter "a",
    the ASCII punctuation and symbols "!" (all 32 of them, which its first pattern names one by one), any other
    punctuation or symbol "p", a combining mark "~", a number "0", the space, \r and \n as they are, any other
    whitespace "\t" (Oniguruma's \s, as CharClasses has it), anything else "x". A character of KANJI_KANA is "k" (a
    letter), "q" (punctuation or a symbol), "m" (a mark) or "z" (unassigned) instead: it is what it was and of that
    block too. (None of the block is a number or whitespace.)"""

    def __missing__(self, code):
        char = chr(code)
        kind = unicodedata.category(char)[0]
        if char in " \r\n":
            found = char
        elif char.isspace() and not "\x1c" <= char <= "\x1f":
            found = "\t"
        elif kind == "N":
            found = "0"
        elif any(low <= code <= high for low, high in KANJI_KANA):
            found = {"L": "k", "P": "q", "S": "q", "M": "m"}.get(kind, "z")
        elif kind == "L":
            found = "A" if char.isascii() else "a"
        elif kind in "PS":
            found = "!" if char.isascii() else "p"
        else:
            found = "~" if kind == "M" else "x"
        self[code] = found
        return found


# T340: the block Spark-X2.5's second Split takes whole ([一-龥぀-ゟ゠-ヿ]+: the ideographs to U+9FA5, the hiragana
# and the katakana blocks)
KANJI_KANA = ((0x4E00, 0x9FA5), (0x3040, 0x309F), (0x30A0, 0x30FF))
SPARK_CLASSES = SparkClasses()


class TekkenClasses(dict):
    r"""T611: the classes Mistral's Tekken pre-tokenizer tells apart (its pattern names the letters by their cases): an
    upper or title case letter (Lu, Lt) "U", a lower case letter (Ll) "l", a letter of neither case (Lm, Lo, which its
    pattern takes on both sides) "o", a combining mark "~", a number "0", the space, \r, \n and "/" as they are (its
    punctuation takes "/" after it as it takes a line break), any other whitespace "\t" (Oniguruma's \s, as CharClasses
    has it), anything else "!"."""

    def __missing__(self, code):
        char = chr(code)
        kind = unicodedata.category(char)
        if char in " \r\n/":
            found = char
        elif char.isspace() and not "\x1c" <= char <= "\x1f":
            found = "\t"
        elif kind in ("Lu", "Lt"):
            found = "U"
        elif kind == "Ll":
            found = "l"
        elif kind in ("Lm", "Lo"):
            found = "o"
        else:
            found = {"M": "~", "N": "0"}.get(kind[0], "!")
        self[code] = found
        return found


TEKKEN_CLASSES = TekkenClasses()
# The pre-tokenizers of their own classes: the classes, and a pattern for each of their stages, each of which cuts
# every piece of the one before (Split, Isolated: what a pattern matches is a piece, and so is what lies between two
# matches). Spark-X2.5's (T340) on SparkClasses; Tekken's (T611), one Split, on TekkenClasses
STAGES = {"spark2_5": (SPARK_CLASSES, (re.compile("0|[^0]+"), re.compile("[kqmz]+|[^kqmz]+"),
                       re.compile("!A+|[^\r\nAak!pq]?[Aak~m]+| ?[!pq]+|[\r\n]|[ \t\r\n]+(?![^ \t\r\n])|[ \t\r\n]+"))),
          "tekken": (TEKKEN_CLASSES, (re.compile("[^\r\nUlo0]?[Uo~]*[lo~]+|[^\r\nUlo0]?[Uo~]+[lo~]*|0"
                                                 "| ?[^ \t\r\nUlo0]+[\r\n/]*|[ \t\r\n]*[\r\n]+|" + SPACES),))}


def isolated(pattern, classes, start, end):
    """The pieces a Split (Isolated) cuts classes[start:end] into, as (start, end): every match of the pattern and every
    stretch between two (none of the patterns matches nothing)."""
    at = start
    for match in pattern.finditer(classes, start, end):
        if match.start() > at:
            yield at, match.start()
        yield match.span()
        at = match.end()
    if at < end:
        yield at, end


def pretokenize(text, pattern):
    r"""Split text the way the tokenizer.json's pre_tokenizer does. Those patterns need \p{L} and \p{N}, which the
    standard re module has not, so they run on the text's character classes (CharClasses) and the pieces are cut
    from the text at the same places (T200: a loop of Python over the characters did it before, a tenth as fast).

    "gpt2" is what ByteLevel(use_regex) applies:
        's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    "gpt2-digits" is the same after Digits(individual_digits), which SmolLM2 puts in front of it: every number is a
    piece of its own, \p{N} as Rust's char::is_numeric says (², Ⅱ and ① too, not only \d: T206), and GPT-2's
    pattern runs on each piece as if the piece were the whole text.
    "qwen" is the pattern Qwen2 spells out (the contractions match whatever the case, digits come one by one,
    and a piece of anything-but-a-line-break may lead a word):
        (?i:'s|…)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    "llama3" is Llama 3's, which is Qwen's with the digits taken up to three at a time (\p{N}{1,3}).
    "qwen35" is Qwen3.5's (T229), which is Qwen's with the combining marks (\p{M}) taken into the word they follow:
        (?i:'s|…)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    "minicpm5" is MiniCPM5's (T254), two Splits one after the other: \p{N}{1,3} cuts the numbers off first, up to
    three at a time, and then Llama 3's pattern with \p{N}+ for its numbers runs on each piece as if the piece were
    the whole text. A number piece is one piece either way, so the second stage here is Llama 3's pattern itself. It is
    not Llama 3's on the whole text: before a number, \s+(?!\S) ends at the piece's end and takes all of a run of
    spaces, where Llama 3's leaves the last one of them to stand alone ("a  1": "a", "  ", "1", not "a", " ", " ", "1").
    "spark2_5" is Spark-X2.5's (T340), three Splits and then Digits(individual_digits), each on every piece of the one
    before: \p{N}{1,3}, which with the Digits after it is every number a piece of its own (and nothing else has a number
    then); [一-龥぀-ゟ゠-ヿ]+, the ideographs and the kana; and
        [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+|[\r\n]|\s+(?!\S)|\s+
    which leaves what it does not match (a character of none of those classes, alone) a piece too (SparkClasses).
    "tekken" is Mistral's Tekken (T611: Mistral Nemo's), one Split, whose letters are told apart by their cases
    (TekkenClasses):
        [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
    All eight are checked against the real patterns in tests/test_bytebpe.py and tests/test_llama3.py.
    """
    if pattern in STAGES:
        table, stages = STAGES[pattern]
        classes, spans = text.translate(table), [(0, len(text))]
        for stage in stages:
            spans = [piece for start, end in spans for piece in isolated(stage, classes, start, end)]
        return [text[start:end] for start, end in spans]
    classes = text.translate(CHAR_CLASSES)
    if pattern in STAGED:
        first, then = STAGED[pattern]
        return [text[match.start():match.end()] for piece in first.finditer(classes)
                for match in then.finditer(classes, piece.start(), piece.end())]
    return [text[match.start():match.end()] for match in PATTERNS.get(pattern, PATTERNS["gpt2"]).finditer(classes)]


# T216: what may follow the pieces of a tokenizer.bin: this, the length of a sentencepiece model's precompiled_charsmap
# (uint32) and the charsmap itself
CHARSMAP = b"charsmap"


def pieces_of(data):
    """The pieces of a tokenizer.bin, one (score, piece, end) at a time: its score, its bytes and where it ends, after
    the 4 bytes of max_token_length; as far as the file holds whole pieces (T639.7: the one walk Tokenizer and
    check_tokenizer() read). It does not know the vocabulary's size, so past the last piece it reads what follows
    (charsmap_after()'s) as pieces: the reader stops at the size it knows."""
    offset = 4
    while offset + 8 <= len(data):
        score, length = struct.unpack_from("<fi", data, offset)
        end = offset + 8 + length
        if length < 0 or end > len(data):
            return
        yield score, bytes(data[offset + 8:end]), end
        offset = end


def charsmap_after(data, offset):
    """(a sentencepiece model's precompiled_charsmap, where it ends) if the tokenizer.bin has one at offset, after its
    pieces (T216), else (None, offset)."""
    if bytes(data[offset:offset + len(CHARSMAP)]) != CHARSMAP or offset + len(CHARSMAP) + 4 > len(data):
        return None, offset
    (size,) = struct.unpack_from("<I", data, offset + len(CHARSMAP))
    start = offset + len(CHARSMAP) + 4
    return bytes(data[start:start + size]), start + size


class Charsmap:
    """sentencepiece's normalizer, from the model's own precompiled_charsmap (T216): the length of a Darts-clone double
    array (uint32), the array, and the normalized texts, each ended by a NUL. The array's keys are UTF-8 texts and its
    values where their normalized text begins. At each place the longest key that begins there is replaced, and where
    none does, one character is kept: sentencepiece's Normalizer::NormalizePrefix, which walks the array as Darts-clone's
    commonPrefixSearch does (a unit: bit 31 a value, bits 0-7 its label, bit 8 a leaf below, and its offset from bit 10,
    shifted by 8 more where bit 9 says). The model's own map: not Python's NFKC, whose Unicode is another version (the
    four letters T126 found in tiny-lm), and not a table of the nmt_ kinds, which differ between models (rinna's)."""

    def __init__(self, blob):
        (size,) = struct.unpack_from("<I", blob)
        self.units = struct.unpack_from(f"<{size // 4}I", blob, 4)
        self.texts = bytes(blob[4 + size:])
        self.found = {}  # the normalized text at each value, read once
        # T222: the bytes a key may begin with, the root's children (the first step of the walk below). The text
        # between two of them is kept as it is, without a walk at each byte. No continuation byte of UTF-8 begins a
        # character, so the text is cut between characters, as the walk cuts it
        units, root = self.units, self.offset(self.units[0])
        begins = bytes(byte for byte in [*range(0x80), *range(0xC0, 0x100)]
                       if root ^ byte < len(units) and units[root ^ byte] & 0x800000FF == byte)
        self.starts = frozenset(begins)  # a walk begins at once where one is, without a search
        self.begins = re.compile(b"[" + re.escape(begins) + b"]") if begins else None

    @staticmethod
    def offset(unit):
        return (unit >> 10) << ((unit & 0x200) >> 6)

    def replaced(self, text):
        data, units, out, i = text.encode("utf-8"), self.units, [], 0
        offset, begins, starts = self.offset, self.begins, self.starts
        root, size = offset(units[0]), len(units)
        while i < len(data):
            if data[i] not in starts:
                found = begins.search(data, i) if begins else None
                if not found:
                    out.append(data[i:])
                    break
                out.append(data[i:found.start()])
                i = found.start()
            node, longest, value = root, 0, 0
            for j in range(i, len(data)):
                node ^= data[j]
                if node >= size or units[node] & 0x800000FF != data[j]:
                    break
                unit = units[node]
                node ^= offset(unit)
                if unit & 0x100:
                    longest, value = j + 1 - i, units[node] & 0x7FFFFFFF
            if longest:
                if value not in self.found:
                    self.found[value] = self.texts[value:self.texts.index(b"\0", value)]
                out.append(self.found[value])
                i += longest
            else:  # one character as it is (Python's texts are always well-formed UTF-8)
                length = 1 if data[i] < 0xC0 else 2 if data[i] < 0xE0 else 3 if data[i] < 0xF0 else 4
                out.append(data[i:i + length])
                i += length
        return b"".join(out).decode("utf-8")


# T649.1: what PLaMo's tokenizer cuts a text around before its search: every <|plamo:…|> written in it (whether its
# vocabulary has that piece or not), the mark it writes at a cut (U+EE00, a character of the Private Use Area), and
# what a character no piece begins with costs
PLAMO_SPECIAL = re.compile(r"(<\|plamo:[^|\s]{,64}\|>)")
PLAMO_CUT = "\uee00"
PLAMO_UNKNOWN = 10_000_000


def plamo_breaks(breaks):
    """The pattern of the runs a "plamo" tokenizer cuts a text around (Tokenizer's breaks), or None: one character so
    many times or more, then so many spaces or more, in that order (tokenization_plamo.py's)."""
    unknown = set(breaks) - {"spaces", "repeats"}
    if unknown:
        raise ValueError(f"Unsupported breaks: {sorted(unknown)} (the runs a text is cut around: spaces, repeats)")
    patterns = ([rf"(.)\2{{{breaks['repeats'] - 1},}}"] if breaks.get("repeats") is not None else []) + \
               ([rf" {{{breaks['spaces']},}}"] if breaks.get("spaces") is not None else [])
    return re.compile(f"({'|'.join(patterns)})") if patterns else None


# T639.1: the settings of the reading a model is given with its options (a Llama's, the converter's), each under the
# name of the Tokenizer's argument it is: a setting that is added is a line here and an argument of the Tokenizer, whose
# default is the default
SETTINGS = {"tokenizer_kind": "kind", "nfkc": "nfkc", "nfc": "nfc", "pretokenizer": "pretokenizer",
            "ignore_merges": "ignore_merges", "collapse": "collapse", "unknown": "unknown", "lowercase": "lowercase",
            "prefixed": "prefixed", "spaced": "spaced", "breaks": "breaks"}


class Tokenizer:
    """llama2.c's tokenizer.bin: sentencepiece pieces with their scores.

    kind="bpe" merges the best-scoring adjacent pair, like llama2.c (Llama 2 vocabulary; and a sentencepiece BPE model
    written out as a tokenizer.json, whose converter writes minus the rank of a piece's merge as its score: T257);
    kind="unigram" picks the segmentation with the best total score (sentencepiece unigram models);
    kind="bytebpe" is Hugging Face's byte-level BPE (GPT-2, SmolLM2, Qwen). It merges by the rank of the merge
    that makes the piece, which is the same loop as "bpe" (merged()) because the converter writes minus the rank as
    the score; only it does not merge a pair into a piece that scores UNMATCHABLE or less. Its vocabulary is written
    in the byte <-> character table above, so the pieces are plain text.
    """

    UNMATCHABLE = -1e8  # convert_hf.py gives control and byte pieces a score below this
    UNMERGED = -1e10  # llama2.c's line: "bpe" merges a pair that scores above it (an unmatchable piece's -1e9 too)

    def __init__(self, data, vocab_size, kind="bpe", nfkc=False, nfc=False, pretokenizer="gpt2", ignore_merges=False,
                 collapse=False, unknown=None, lowercase=False, prefixed=False, spaced=False, breaks=None):
        if kind not in TOKENIZER_KINDS:
            raise ValueError(f"Unsupported tokenizer kind: {kind!r} (the engine reads {', '.join(TOKENIZER_KINDS)})")
        # T639.4: what the kind does, from its row of TOKENIZER_KINDS, taken apart once (encode() and decode() read it)
        self.kind, (self.encoding, self.decoding, self.prefixes, self.spaces, made) = kind, TOKENIZER_KINDS[kind]
        self.nfkc, self.nfc, self.pretokenizer = nfkc, nfc, pretokenizer
        # T265: the text is made lower case before anything else (rinna's japanese-gpt2: tokenizer_config.json's
        # do_lower_case, with a vocabulary that has no capital Latin letter, each of which was <unk> without it)
        self.lowercase = lowercase
        # T308: sentencepiece's dummy prefix stands before the text after a special token too, not only before the
        # first text. "every": before every such stretch (a tokenizer.json whose normalizer begins each with "▁":
        # llm-jp-4's). "wanting": before every one that does not begin with a space already (a Metaspace that always
        # prepends, which is what transformers makes of a legacy Llama tokenizer: zephyr's, EuroLLM's). False: none
        # besides the first. T257: "never": not before the first either (a Gemma's sentencepiece model has no dummy
        # prefix), and decode() takes no space off the beginning of what was written
        if prefixed not in (False, "every", "wanting", "never"):
            raise ValueError(f"Unsupported prefixed: {prefixed!r} (the stretches of text the dummy prefix stands before: every, wanting or never)")
        self.prefixed = prefixed
        # T400 (3): a Metaspace pre-tokenizer (a tokenizer.json's) puts no "▁" before the first text either where that
        # text begins with a space already (the space is its "▁"); sentencepiece itself puts it there whatever the text
        # begins with (a tokenizer.model's, the engine's way where spaced is not said)
        self.spaced = spaced
        # T649.1: where a "plamo" tokenizer cuts a text before its search, besides the <|plamo:…|> it writes: around a
        # run of so many spaces or more ({"spaces": n}) and of one character so many times or more ({"repeats": n}),
        # tokenizer_config.json's break_around_consecutive_spaces_threshold and break_around_repeated_chars_threshold
        # (T649.3: a dict of the page's options, a JsProxy in Pyodide: unwrapped as every other one, engine/proxy.py)
        self.breaks = plamo_breaks(plain(breaks) or {})
        self.cuts = {}  # cut()'s, of the specials it was last asked for
        self.collapse = collapse  # a sentencepiece model's remove_extra_whitespaces: see normalized()
        # a sentencepiece model without byte pieces (rinna's) writes a character it lacks as its unknown piece, a run
        # of them as one; the others spell it in bytes
        self.unknown = unknown
        self.ignore_merges = ignore_merges
        self.vocab, self.scores, offset = [], [], 4
        for _, (score, piece, offset) in zip(range(vocab_size), pieces_of(data)):
            self.vocab.append(piece)
            self.scores.append(score)
        if len(self.vocab) != vocab_size:
            raise ValueError(f"This tokenizer.bin holds {len(self.vocab)} whole pieces, not the {vocab_size} of the vocabulary.")
        # T216: the sentencepiece model's own normalizer, where the converter put it after the pieces
        charsmap, _ = charsmap_after(data, offset)
        self.charsmap = None if charsmap is None else Charsmap(charsmap)
        # first occurrence wins, like list.index() in llama2.py
        self.index = {}
        for i, piece in enumerate(self.vocab):
            self.index.setdefault(piece, i)
        # raw byte tokens look like b"<0x0A>"; they spell out whatever the vocabulary lacks
        self.byte_tokens = [self.index.get(b"<0x%02X>" % byte, byte + 3) for byte in range(256)]
        self.text_index, self.pieces, self.reach = {}, {}, {}
        if made:
            made(self)
        self.unknown_score = min(score for score in self.scores if score > self.UNMATCHABLE) - 10.0

    def encode(self, text, specials=()):
        """specials: pieces such as "</s>" that stand for their token wherever they are written (a chat template
        puts them between the turns). As plain text they would be spelled out letter by letter."""
        tokens, first = [], True
        pattern, pieces = self.cut(tuple(specials)) if specials else (None, ())
        for part in pattern.split(text) if specials else [text]:
            if part in pieces:
                tokens.append(self.index[pieces[part]])
            elif part and (part := self.normalized(part)):
                # sentencepiece's dummy prefix: the model saw every text start with a space (but not the text after a
                # special token, unless its tokenizer puts one there as well: prefixed, T308; and a model without the
                # prefix saw none, T257). A byte-level vocabulary has none: it spells the space as a character
                if self.prefixes and self.prefixed != "never" and ((first and not (self.spaced and part.startswith(" "))) or self.prefixed == "every"
                                                                   or (self.prefixed and not part.startswith(" "))):
                    part = " " + part
                tokens += self.encoding(self, part)
            first = False
        return tokens

    def cut(self, specials):
        """(the pattern that cuts a text at these specials, {a special: its piece as the vocabulary spells it}), made
        once for a tuple of them (T257: a Gemma 3 has 6,406 added tokens, and writing the pattern out for each text
        took 2 ms). The first of the alternatives that matches is taken, so the converter names the longest first. A
        sentencepiece vocabulary's pieces spell U+2581 as a space (the converter's tokenizer_bin()), a special's
        among them (a Gemma 3's runs of U+2581)."""
        if specials not in self.cuts:
            spelled = (lambda special: special.replace("\u2581", " ")) if self.spaces else (lambda special: special)
            self.cuts = {specials: (re.compile("(" + "|".join(re.escape(special) for special in specials) + ")"),
                                    {special: spelled(special).encode("utf-8") for special in specials})}
        return self.cuts[specials]

    def normalized(self, text):
        """The text as the model's normalizer makes it: a sentencepiece model's own map (charsmap, T216: nmt_nfkc makes
        tabs, newlines and a few more characters a space, and drops or keeps the other control characters, as each
        model's map says), NFKC or NFC (a tokenizer.json's), and collapse: sentencepiece's remove_extra_whitespaces,
        runs of spaces one and none at either end. Without them a newline was spelled with byte + 3 in a vocabulary
        with no byte pieces: rinna's った (the review of T126). lowercase (T265) comes first, as transformers' slow
        tokenizers lower the text before sentencepiece sees it."""
        if self.lowercase:
            text = text.lower()
        if self.charsmap:
            text = self.charsmap.replaced(text)
        if self.nfkc:
            text = unicodedata.normalize("NFKC", text)
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        if self.collapse:
            text = re.sub(" {2,}", " ", text).strip(" ")
        if self.spaces:
            # sentencepiece writes a space as U+2581, so one written in the text is a space too (after the collapse,
            # which sees spaces only; an nmt_ map made it one before). The pieces here spell it " " (T216)
            text = text.replace("\u2581", " ")
        return text

    def encode_bpe(self, text):
        # First encode every individual character; a character the vocabulary lacks becomes its UTF-8 bytes
        tokens = []
        for char in text:
            piece = char.encode("utf-8")
            if piece in self.index:
                tokens.append(self.index[piece])
            elif self.unknown is not None:
                tokens += [] if tokens[-1:] == [self.unknown] else [self.unknown]  # a run of them is one
            else:
                tokens.extend(self.byte_tokens[byte] for byte in piece)
        return self.merged(tokens, [self.vocab[token] for token in tokens], self.index, self.UNMERGED)

    def merged(self, tokens, pieces, index, floor):
        """The tokens with the best consecutive pair merged each iteration, according to the scores in vocab_scores
        (the first of equal scores), as long as a pair scores above floor. pieces: the tokens' pieces as index spells
        them (bytes for "bpe", the byte-level BPE's text for "bytebpe": T639.4, which merged in a loop of its own
        that looked at every pair after each merge)."""
        # The pairs wait in a heap by (minus the score, where the pair's left piece began), so the best and
        # the first of equal ones comes out first; a merge pushes the two pairs next to it, and the pairs it ended
        # are left in the heap and passed over when they come out (stamp: a piece's stamp changes when it merges).
        # T200 had the scores in a list and took the best with max() and index(): 0.17 s for 1000 tokens (T207).
        # The pieces are a list linked by after and before, the first one never goes (a merge keeps the left piece).
        count = len(tokens)
        after, before, stamp, waiting = list(range(1, count + 1)), list(range(-1, count - 1)), [0] * count, []

        def pair(k, m):
            """the pieces k and m joined, if they merge: (minus the score, k, m, their stamps, the token)"""
            id = index.get(pieces[k] + pieces[m])
            if id is not None and self.scores[id] > floor:
                return -self.scores[id], k, m, stamp[k], stamp[m], id

        for k in range(count - 1):
            if joined := pair(k, k + 1):
                waiting.append(joined)
        heapq.heapify(waiting)
        while waiting:
            _, k, m, stamp_k, stamp_m, id = heapq.heappop(waiting)
            if stamp[k] != stamp_k or stamp[m] != stamp_m:
                continue  # a merge has changed one of the two since
            tokens[k], pieces[k] = id, pieces[k] + pieces[m]
            stamp[k] += 1
            stamp[m] += 1
            after[k] = after[m]
            if after[k] < count:
                before[after[k]] = k
            for left, right in ((before[k], k), (k, after[k])):  # the pairs with the merged piece
                if left >= 0 and right < count and (joined := pair(left, right)):
                    heapq.heappush(waiting, joined)
        merged, k = [], 0
        while k < count:
            merged.append(tokens[k])
            k = after[k]
        return merged

    def encode_bytebpe(self, text):
        # The pre-tokenizer keeps merges inside a word: the pieces never cross from a word into the next.
        tokens = []
        for part in pretokenize(text, self.pretokenizer):
            word = part.encode("utf-8").decode("latin-1").translate(BYTE_CHARS)  # each byte as its character
            if self.ignore_merges and word in self.text_index:
                # Llama 3: a piece the vocabulary has is that one token, whatever the merges would make of it
                tokens.append(self.text_index[word])
                continue
            # a byte the vocabulary lacks (SmolLM2 has no piece for 21 of them: 0x04, 0xF1, ...) is left out, and its
            # neighbours may then merge, as Hugging Face's BPE does without byte_fallback or an unk_token (T215)
            symbols = [char for char in word if char in self.text_index]
            tokens += self.merged([self.text_index[symbol] for symbol in symbols], symbols, self.text_index,
                                  self.UNMATCHABLE)
        return tokens

    def text_pieces(self):
        """byte-level pieces are text, and the merging works on that text rather than on bytes"""
        for i, piece in enumerate(self.vocab):
            self.text_index.setdefault(piece.decode("utf-8", "replace"), i)

    def unigram_pieces(self):
        """the Viterbi search's pieces as text, each the token index finds for it (the first of equal pieces), those
        it can match only, and how long such a piece that begins with two given characters can be (T200)"""
        for piece, i in self.index.items():
            try:
                text = piece.decode("utf-8")
            except UnicodeDecodeError:
                continue  # no text encodes to it
            if self.scores[i] > self.UNMATCHABLE:
                self.pieces[text] = i
                if len(text) > 1 and self.reach.get(text[:2], 0) < len(text):
                    self.reach[text[:2]] = len(text)

    def encode_unigram(self, text):
        # Viterbi: best[j] is the best total score of any segmentation of text[:j]. The pieces that begin at i are
        # no longer than the reach of text[i:i + 2] (one character where no longer piece begins with those two) (T200)
        n, pieces, scores, reach = len(text), self.pieces, self.scores, self.reach
        best = [0.0] + [-math.inf] * n
        back = [None] * (n + 1)
        for i in range(n):
            here = best[i]
            if here == -math.inf:
                continue
            for j in range(i + 1, min(n, i + reach.get(text[i:i + 2], 1)) + 1):
                id = pieces.get(text[i:j])
                if id is not None and here + scores[id] > best[j]:
                    best[j], back[j] = here + scores[id], (i, [id])
            # a character the vocabulary lacks becomes its UTF-8 bytes, at a penalty
            if back[i + 1] is None or back[i + 1][0] != i:
                score = best[i] + self.unknown_score
                if score > best[i + 1]:
                    spelled = [self.unknown] if self.unknown is not None else [self.byte_tokens[byte] for byte in text[i].encode("utf-8")]
                    best[i + 1], back[i + 1] = score, (i, spelled)
        tokens, j = [], len(text)
        while j > 0:
            i, ids = back[j]
            if self.unknown is not None and ids == [self.unknown] and tokens[:1] == ids:
                ids = []  # sentencepiece makes a run of unknown characters one unknown piece
            tokens[:0] = ids
            j = i
        return tokens

    def plamo_pieces(self):
        """T649.1: the unigram search's pieces (unigram_pieces()), and each token's score as PLaMo's search adds it up:
        an integer, round(score * 1e4) (the scores are written with four decimals at most, and float32 holds them to
        less than half of that last decimal: the converter checks)"""
        self.unigram_pieces()
        self.points = [round(score * 1e4) for score in self.scores]

    def encode_plamo(self, text):
        """T649.1: PLaMo's own tokenizer (tokenization_plamo.py's, which this takes the form of; none of its lines):
        the text is cut around every <|plamo:…|> written in it, and then around the runs breaks says, and each stretch
        between two cuts is searched alone (PLaMo writes U+EE00 at the cuts, a piece of score 0 that it then leaves
        out: one written in the text is left out too). The search is a Viterbi of the integer scores (points) made from
        the end of the stretch: at each place, the pieces that begin there are tried from the longest down, and one is
        taken only where it is strictly better, so of equal totals the longest first piece wins (unigram's search,
        made from the beginning, takes the longest last piece instead). A character no piece begins with costs
        PLAMO_UNKNOWN, whatever it is, and is spelled in the bytes of its UTF-8."""
        text = PLAMO_SPECIAL.sub(PLAMO_CUT + r"\1" + PLAMO_CUT, text)
        if self.breaks is not None:
            text = self.breaks.sub(PLAMO_CUT + r"\1" + PLAMO_CUT, text)
        tokens = []
        for stretch in text.split(PLAMO_CUT):
            tokens += self.plamo_search(stretch)
        return tokens

    def plamo_search(self, text):
        n, pieces, points, reach = len(text), self.pieces, self.points, self.reach
        cost, taken = [0] * (n + 1), [None] * n  # the least cost from a place to the end, and (length, token) there
        for i in range(n - 1, -1, -1):
            best, take = None, None
            for j in range(min(n, i + reach.get(text[i:i + 2], 1)), i, -1):
                id = pieces.get(text[i:j])
                if id is not None and (best is None or cost[j] - points[id] < best):
                    best, take = cost[j] - points[id], (j - i, id)
            if best is None or cost[i + 1] + PLAMO_UNKNOWN < best:
                best, take = cost[i + 1] + PLAMO_UNKNOWN, (1, None)
            cost[i], taken[i] = best, take
        tokens, i = [], 0
        while i < n:
            length, id = taken[i]
            tokens += [id] if id is not None else [self.byte_tokens[byte] for byte in text[i].encode("utf-8")]
            i += length
        return tokens

    def decode(self, prev_token, token, bos=1):
        return self.decoding(self, prev_token, token, bos)

    def decode_bytes(self, prev_token, token, bos):
        # back through the byte <-> character table; a piece that is not written in it (an added token such as
        # <|im_end|>) is its own text
        piece = self.vocab[token]
        text = piece.decode("utf-8", "replace")
        return bytes(CHAR_BYTES[char] for char in text) if all(char in CHAR_BYTES for char in text) else piece

    def decode_pieces(self, prev_token, token, bos):
        piece = self.vocab[token]
        # Following the first token, sentencepiece decoder strips the leading whitespace (the dummy prefix)
        if self.prefixes and prev_token == bos and piece.startswith(b" ") and self.prefixed != "never":
            piece = piece[1:]
        # Some tokens designate raw bytes, and look like b"<0x0A>"
        if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
            piece = bytes([int(piece[3:5], 16)])
        return piece


class Kind(NamedTuple):
    """T639.4: one of the Tokenizer's kinds, which it takes apart once when it is made"""
    encode: Callable  # (tokenizer, a stretch of normalized text) -> its tokens
    decode: Callable  # (tokenizer, the token before, the token, bos) -> the token's bytes
    prefixes: bool  # sentencepiece's dummy prefix may stand before a stretch of text (prefixed and spaced say where)
    spaces: bool  # U+2581 is a space, in the text and in a special's spelling: the pieces spell it " " (T216)
    made: Optional[Callable]  # (tokenizer) makes the kind's own index of the pieces, once


# the kinds a tokenizer.bin is read as (the converter's tokenizer_kind): a kind that is added is a row here and the
# Tokenizer's methods it names
TOKENIZER_KINDS = {
    "bpe": Kind(Tokenizer.encode_bpe, Tokenizer.decode_pieces, True, True, None),
    "unigram": Kind(Tokenizer.encode_unigram, Tokenizer.decode_pieces, True, True, Tokenizer.unigram_pieces),
    "bytebpe": Kind(Tokenizer.encode_bytebpe, Tokenizer.decode_bytes, False, False, Tokenizer.text_pieces),
    # T649.1: PLaMo's tokenizer.jsonl (the converter's plamo()): no dummy prefix, a space is a space
    "plamo": Kind(Tokenizer.encode_plamo, Tokenizer.decode_pieces, False, False, Tokenizer.plamo_pieces)}
