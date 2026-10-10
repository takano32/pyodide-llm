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
    All six are checked against the real patterns in tests/test_bytebpe.py and tests/test_llama3.py.
    """
    classes = text.translate(CHAR_CLASSES)
    if pattern in STAGED:
        first, then = STAGED[pattern]
        return [text[match.start():match.end()] for piece in first.finditer(classes)
                for match in then.finditer(classes, piece.start(), piece.end())]
    return [text[match.start():match.end()] for match in PATTERNS.get(pattern, PATTERNS["gpt2"]).finditer(classes)]


# T216: what may follow the pieces of a tokenizer.bin: this, the length of a sentencepiece model's precompiled_charsmap
# (uint32) and the charsmap itself
CHARSMAP = b"charsmap"


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


class Tokenizer:
    """llama2.c's tokenizer.bin: sentencepiece pieces with their scores.

    kind="bpe" merges the best-scoring adjacent pair, like llama2.c (Llama 2 vocabulary);
    kind="unigram" picks the segmentation with the best total score (sentencepiece unigram models);
    kind="bytebpe" is Hugging Face's byte-level BPE (GPT-2, SmolLM2, Qwen). It merges by the rank of the merge
    that makes the piece, which is the same loop as "bpe" because the converter writes minus the rank as the
    score. Its vocabulary is written in the byte <-> character table above, so the pieces are plain text.
    """

    UNMATCHABLE = -1e8  # convert_hf.py gives control and byte pieces a score below this

    def __init__(self, data, vocab_size, kind="bpe", nfkc=False, nfc=False, pretokenizer="gpt2", ignore_merges=False,
                 collapse=False, unknown=None, lowercase=False):
        self.kind, self.nfkc, self.nfc, self.pretokenizer = kind, nfkc, nfc, pretokenizer
        # T265: the text is made lower case before anything else (rinna's japanese-gpt2: tokenizer_config.json's
        # do_lower_case, with a vocabulary that has no capital Latin letter, each of which was <unk> without it)
        self.lowercase = lowercase
        self.collapse = collapse  # a sentencepiece model's remove_extra_whitespaces: see normalized()
        # a sentencepiece model without byte pieces (rinna's) writes a character it lacks as its unknown piece, a run
        # of them as one; the others spell it in bytes
        self.unknown = unknown
        self.ignore_merges = ignore_merges
        self.vocab, self.scores = [], []
        offset = 4  # skip max_token_length
        for _ in range(vocab_size):
            score, length = struct.unpack_from("<fi", data, offset)
            offset += 8
            self.vocab.append(bytes(data[offset:offset + length]))
            self.scores.append(score)
            offset += length
        # T216: the sentencepiece model's own normalizer, where the converter put it after the pieces
        rest = bytes(data[offset:offset + len(CHARSMAP) + 4])
        self.charsmap = None
        if rest[:len(CHARSMAP)] == CHARSMAP:
            (size,) = struct.unpack_from("<I", rest, len(CHARSMAP))
            start = offset + len(CHARSMAP) + 4
            self.charsmap = Charsmap(bytes(data[start:start + size]))
        # first occurrence wins, like list.index() in llama2.py
        self.index = {}
        for i, piece in enumerate(self.vocab):
            self.index.setdefault(piece, i)
        # raw byte tokens look like b"<0x0A>"; they spell out whatever the vocabulary lacks
        self.byte_tokens = [self.index.get(b"<0x%02X>" % byte, byte + 3) for byte in range(256)]
        # byte-level pieces are text, and the merging works on that text rather than on bytes
        self.text_index = {}
        if self.kind == "bytebpe":
            for i, piece in enumerate(self.vocab):
                self.text_index.setdefault(piece.decode("utf-8", "replace"), i)
        # the Viterbi search's pieces as text, each the token index finds for it (the first of equal pieces), those
        # it can match only, and how long such a piece that begins with two given characters can be (T200)
        self.pieces, self.reach = {}, {}
        if self.kind == "unigram":
            for piece, i in self.index.items():
                try:
                    text = piece.decode("utf-8")
                except UnicodeDecodeError:
                    continue  # no text encodes to it
                if self.scores[i] > self.UNMATCHABLE:
                    self.pieces[text] = i
                    if len(text) > 1 and self.reach.get(text[:2], 0) < len(text):
                        self.reach[text[:2]] = len(text)
        self.unknown_score = min(score for score in self.scores if score > self.UNMATCHABLE) - 10.0

    def encode(self, text, specials=()):
        """specials: pieces such as "</s>" that stand for their token wherever they are written (a chat template
        puts them between the turns). As plain text they would be spelled out letter by letter."""
        tokens, first = [], True
        for part in re.split("(" + "|".join(re.escape(special) for special in specials) + ")", text) if specials else [text]:
            if part in specials:
                tokens.append(self.index[part.encode("utf-8")])
            elif part and (part := self.normalized(part)):
                if self.kind == "bytebpe":
                    # no dummy prefix: a byte-level vocabulary spells the space out as a character of its own
                    tokens += self.encode_bytebpe(part)
                else:
                    # sentencepiece's dummy prefix: the model saw every text start with a space (but not the
                    # text after a special token)
                    part = " " + part if first else part
                    tokens += self.encode_unigram(part) if self.kind == "unigram" else self.encode_bpe(part)
            first = False
        return tokens

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
        if self.kind != "bytebpe":
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

        # Merge the best consecutive pair each iteration, according to the scores in vocab_scores (the first of equal
        # scores). The pairs wait in a heap by (minus the score, where the pair's left piece began), so the best and
        # the first of equal ones comes out first; a merge pushes the two pairs next to it, and the pairs it ended
        # are left in the heap and passed over when they come out (stamp: a piece's stamp changes when it merges).
        # T200 had the scores in a list and took the best with max() and index(): 0.17 s for 1000 tokens (T207).
        # The pieces are a list linked by after and before, the first one never goes (a merge keeps the left piece).
        pieces, count = [self.vocab[token] for token in tokens], len(tokens)
        after, before, stamp, waiting = list(range(1, count + 1)), list(range(-1, count - 1)), [0] * count, []

        def pair(k, m):
            """the pieces k and m joined, if they merge: (minus the score, k, m, their stamps, the token)"""
            id = self.index.get(pieces[k] + pieces[m])
            if id is not None and self.scores[id] > -1e10:
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
            while len(symbols) > 1:
                best_score, best_id, best_idx = self.UNMATCHABLE, -1, -1
                for i in range(len(symbols) - 1):
                    id = self.text_index.get(symbols[i] + symbols[i + 1])
                    if id is not None and self.scores[id] > best_score:
                        best_score, best_id, best_idx = self.scores[id], id, i
                if best_idx == -1:
                    break
                symbols[best_idx:best_idx + 2] = [self.vocab[best_id].decode("utf-8", "replace")]
            tokens += [self.text_index[symbol] for symbol in symbols]
        return tokens

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

    def decode(self, prev_token, token, bos=1):
        piece = self.vocab[token]
        if self.kind == "bytebpe":
            # back through the byte <-> character table; a piece that is not written in it (an added token such
            # as <|im_end|>) is its own text
            text = piece.decode("utf-8", "replace")
            return bytes(CHAR_BYTES[char] for char in text) if all(char in CHAR_BYTES for char in text) else piece
        # Following the first token, sentencepiece decoder strips the leading whitespace (the dummy prefix)
        if prev_token == bos and piece.startswith(b" "):
            piece = piece[1:]
        # Some tokens designate raw bytes, and look like b"<0x0A>"
        if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
            piece = bytes([int(piece[3:5], 16)])
        return piece
