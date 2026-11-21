# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository), and its cutting
# of a text into chunks (split() and the ways it cuts: the chunks are the same as those of the Rust function it was
# read from, on 14,000 random texts) and (T667) its translating again of a chunk that ran away in smaller pieces
# (Document) are derived from tanakh/plamo-translate-desktop's engine::document
# (https://github.com/tanakh/plamo-translate-desktop at 3d69a116c45a8116159f1a405311a59039ec4b95), which is under the
# MIT License (it is under Apache-2.0 too; this is the MIT License's notice), Copyright (c) 2026 Hideyuki Tanaka.
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
"""engine/document.py (T666): a long text translated by PLaMo 3 Translate a piece at a time, each piece with the pieces
before it and their translations as its history (engine/pfml.py's). The text is cut into paragraphs at its empty
lines, and as many whole paragraphs as fit go into one piece (a chunk); a paragraph too long for one is cut at the
first of these that makes its parts fit: a line that ends a sentence, a sentence, a line, a clause (after a comma or
at a space), and last a number of characters. What lies between the chunks (the empty lines) and the whitespace the
text begins with are written out as they are, so that the chunks and what lies between them are the text again.

The form is tanakh/plamo-translate-desktop's engine::document (3d69a116c45a8116159f1a405311a59039ec4b95; the notice
above). Where it differs: a text of only whitespace is written out as it is (theirs writes nothing), and the numbers
of tokens are this page's (a context of 4096 at most, where theirs is 32768: CHUNK_TOKENS, HISTORY_TOKENS).

T667: a translation that runs away (falls into a loop: engine/repetition.py; runs out of room; or stops far too
short) is taken back (Withdrawn) and its text is translated again in smaller pieces: a chunk in pieces of RETRY_TOKENS,
such a piece a sentence at a time, and a sentence that still runs away is cut (Cut, a mark where the page writes its
words) after the first time of its loop. Theirs translates that sentence once more with DRY's penalty first (T668: not
here yet, PENALIZED). A translation that ran away is never history. Where theirs stops the document at a full context,
this one takes the run as one that ran out of room (its pieces' prompts are shorter).

A chunk is translated by the model's generate() from a fresh start (a new cache of keys and values): the loop of
generate() is not touched, it is called once for each chunk. The positions are of characters (Python's), the tokens
the model's tokenizer's, read literally (a mark typed in the text is its characters, as pfml_tokens() reads it)."""
from engine.pfml import pfml_tokens
from engine.tokenizer import WHITE_SPACE

# T666: the most tokens of a chunk and of the history: a sixth and a third of what a context of 4096 leaves, as
# tanakh's share theirs out (1024 and 4096, for a context of 32768), so that a full history, a chunk, the request's
# marks and a translation of LONGER times the chunk and MORE fit in 4096 (1280 + 640 + about 60 + 2048). Measured on
# PLaMo 3 Translate 0.8B (tests/document_measure.ts): TASKS.md's T666.
CHUNK_TOKENS = 640
HISTORY_TOKENS = 1280
# the share of what the context leaves (after the context, the glossary and ROOM) that a chunk and the history take at
# most, and the least that is left to share (a context and a glossary that take all of it still leave this)
CHUNK_SHARE, HISTORY_SHARE = 6, 3
ROOM, LEAST = 64, 384
# the tokens a glossary's pair costs beside its two texts (its two marks), and a pair of the history (its elements'
# marks and words)
GLOSSARY_PAIR, HISTORY_PAIR = 2, 16
# the characters that end a sentence wherever they are, the Latin ones that end one before whitespace (or at the end),
# what closes a quotation or a parenthesis after them (it belongs to the sentence), and what ends a clause
TERMINALS = "。！？．"
LATIN_TERMINALS = ".!?"
CLOSINGS = "」』）)\"'”’"
JAPANESE_QUOTES = "」』"
CLAUSES = "、，,;；:："
# what lies between paragraphs besides newlines (on the blank line), and what the separator after one takes in
BLANK = " \t\r"
ASCII_SPACE = " \t\n\x0c\r"
# a chunk's translation is no longer than this many times its tokens and MORE (tanakh's: a runaway translation stops
# there; T667 tells it from one that is only long)
LONGER, MORE = 3, 128
# T667 (tanakh's): the tokens of a piece a chunk that ran away is translated again in, and how short a translation of
# a text of SHORT_SOURCE tokens or more is where it stopped far too short (less than a SHORTER-th of them)
RETRY_TOKENS = 256
SHORT_SOURCE, SHORTER = 200, 4
# T667: the sampling settings a sentence that still runs away is translated again with before it is cut (T668's DRY;
# none: it is cut as it ran)
PENALIZED = None
# why a translation ran away: it fell into a loop, it ran out of room, it stopped far too short
LOOP, LONG, SHORT = "loop", "long", "short"


def blank(char):
    return char in WHITE_SPACE


def trimmed(text, start, end):
    """where text[start:end] ends without the whitespace at its end"""
    while end > start and blank(text[end - 1]):
        end -= 1
    return end


def paragraphs(text, start):
    """the paragraphs of text from start: (start, end, the end of the separator after it), the end without the
    whitespace at its end and the separator to the next paragraph (or to the end of the text). A paragraph ends at an
    empty line (one of only spaces, tabs and carriage returns)."""
    found = []
    at = start
    while at < len(text):
        if text[at] == "\n":
            after = at + 1
            while after < len(text) and text[after] in BLANK:
                after += 1
            if after < len(text) and text[after] == "\n":
                end = trimmed(text, start, at)
                separated = after
                while separated < len(text) and text[separated] in ASCII_SPACE:
                    separated += 1
                if end > start:
                    found.append((start, end, separated))
                start = at = separated
                continue
        at += 1
    if start < len(text):
        end = trimmed(text, start, len(text))
        if end > start:
            found.append((start, end, len(text)))
    return found


def ends_sentence(text, start, end):
    """whether the line text[start:end] ends a sentence: in a terminal (Latin or not) and what closes after it, or in a
    Japanese quote's closing (read from its end back, not cut out: a paragraph of many lines none of which ends a
    sentence would copy its growing front for each line)"""
    core = end
    while core > start and text[core - 1] in CLOSINGS:
        core -= 1
    return any(char in JAPANESE_QUOTES for char in text[core:end]) or (core > start and text[core - 1] in TERMINALS + LATIN_TERMINALS)


def lines_after_sentences(text, start, end, separated):
    """the parts of text[start:end] cut at the newlines after a line that ends a sentence"""
    parts, part = [], start
    at = start
    while (newline := text.find("\n", at, end)) >= 0:
        line_end = trimmed(text, part, newline)
        following = newline
        while following < end and blank(text[following]):
            following += 1
        if line_end > part and following < end and ends_sentence(text, part, line_end):
            parts.append((part, line_end, following))
            part = following
        at = max(following, newline + 1)
    return parts + [(part, end, separated)]


def sentences(text, start, end, separated, at_newlines=False):
    """the parts of text[start:end] cut after each sentence (with what closes after it; at_newlines: and after each
    newline), the whitespace after it the separator"""
    parts, sentence = [], start
    at = start
    while at < end:
        char = text[at]
        at += 1
        latin = char in LATIN_TERMINALS and (at >= end or blank(text[at]))
        if not (char in TERMINALS or latin or (at_newlines and char == "\n")):
            continue
        while at < end and text[at] in CLOSINGS:
            at += 1
        sentence_end = trimmed(text, sentence, at)
        while at < end and blank(text[at]):
            at += 1
        if at >= end:
            break
        if sentence_end > sentence:
            parts.append((sentence, sentence_end, at))
        sentence = at
    if sentence < end:
        parts.append((sentence, end, separated))
    elif parts:
        parts[-1] = (*parts[-1][:2], separated)
    return parts


def clauses(text, start, end, separated):
    """the parts of text[start:end] cut after each comma (and the like) and at each run of whitespace"""
    parts, part = [], start
    at = start
    while at < end:
        char = text[at]
        if blank(char):
            part_end, at = at, at + 1
        elif char in CLAUSES:
            part_end = at = at + 1
        else:
            at += 1
            continue
        while at < end and blank(text[at]):
            at += 1
        if part_end > part and at < end:
            parts.append((part, part_end, at))
            part = at
    return parts + [(part, end, separated)]


def by_characters(text, start, end, separated, most, count):
    """the parts of text[start:end] cut every so many characters (as many as seven eighths of most tokens take, at the
    text's own characters to the token), with nothing between them"""
    tokens = max(count(text[start:end]), 1)
    each = max((end - start) * most * 7 // 8 // tokens, 1)
    cuts = list(range(start, end, each))[1:]
    return [(a, b, b) for a, b in zip([start] + cuts, cuts)] + [((cuts or [start])[-1], end, separated)]


# the ways a paragraph too long is cut, the first first: each part that is still too long is cut the next way
CUTS = (lines_after_sentences, sentences, lambda *part: sentences(*part, at_newlines=True), clauses, by_characters)


def refined(text, part, most, count, way=0):
    """the part as it is where it fits in most tokens, else its parts cut the way-th way (CUTS), each refined"""
    start, end, separated = part
    if count(text[start:end]) <= most:
        return [part]
    cut = CUTS[way]
    if cut is by_characters:
        return by_characters(text, start, end, separated, most, count)
    return [each for piece in cut(text, start, end, separated) for each in refined(text, piece, most, count, way + 1)]


def split(text, most, count):
    """text cut into chunks of at most most tokens (count(text): its number of tokens) where they can be: (the
    whitespace the text begins with, [(a chunk, the separator after it) ...]), which joined are the text again. A
    chunk is as many paragraphs (or parts of one too long) as fit, with what lies between them. Only a chunk cut at
    characters may be more where counting a part alone is fewer than counting it in its place."""
    body = 0
    while body < len(text) and blank(text[body]):
        body += 1
    parts = [each for paragraph in paragraphs(text, body) for each in refined(text, paragraph, most, count)]
    counts = [count(text[start:end]) for start, end, _ in parts]
    chunks = []
    i = 0
    while i < len(parts):
        start, end, separated = parts[i]
        estimate = counts[i]
        j = i + 1
        # (the sum of the parts and what lies between them, counted again whole near most: a token may join two)
        while j < len(parts):
            following, following_end, following_separated = parts[j]
            guess = estimate + count(text[end:following]) + counts[j]
            if guess > most - most // 16:
                exact = count(text[start:following_end])
                if exact > most:
                    break
                guess = exact
            estimate, end, separated = guess, following_end, following_separated
            j += 1
        chunks.append((text[start:end], text[end:separated]))
        i = j
    return text[:body], chunks


def budgets(count, context_length, context=None, glossary=(), chunk_tokens=CHUNK_TOKENS, history_tokens=HISTORY_TOKENS):
    """the most tokens of a chunk and of the history: CHUNK_SHARE's and HISTORY_SHARE's share of what the context
    leaves after the context text, the glossary and ROOM (LEAST at the least), and at most chunk_tokens and
    history_tokens"""
    fixed = (count(context) if context else 0) + sum(count(term) + count(said) + GLOSSARY_PAIR for term, said in glossary or ()) + ROOM
    left = max(context_length - fixed, LEAST)
    return min(left // CHUNK_SHARE, chunk_tokens), min(left // HISTORY_SHARE, history_tokens)


def recent(pairs, costs, most):
    """the latest of the pairs whose costs add up to most at the most, the earliest first (as pfml_pieces() takes
    them): taken from the latest back, until the next would be too many"""
    used, first = 0, len(pairs)
    for at in range(len(pairs) - 1, -1, -1):
        used += costs[at]
        if used > most:
            break
        first = at
    return pairs[first:]


def sentences_of(text):
    """T667: text cut at its sentences (in its paragraphs: a newline in one is whitespace), as split() returns its
    chunks (theirs, for a piece that ran away again)"""
    body = 0
    while body < len(text) and blank(text[body]):
        body += 1
    return text[:body], [(text[start:end], text[end:separated]) for paragraph in paragraphs(text, body)
                         for start, end, separated in sentences(text, *paragraph)]


class Withdrawn:
    """T667: what translate_document() yields where it takes back the last characters it wrote (of a translation that
    ran away, to be translated again)"""
    __slots__ = ("characters",)

    def __init__(self, characters):
        self.characters = characters

    def __repr__(self):
        return f"Withdrawn({self.characters})"


class Cut:
    """T667: what translate_document() yields where it cut a translation that still ran away (after its loop's first
    time, or as far as it ran): the page writes a mark there, in its own words. index: the chunk's; reason: LOOP,
    LONG or SHORT; source: the text whose translation was cut"""
    __slots__ = ("index", "reason", "source")

    def __init__(self, index, reason, source):
        self.index, self.reason, self.source = index, reason, source

    def __repr__(self):
        return f"Cut({self.index}, {self.reason!r}, {self.source!r})"


def applied(pieces, mark=""):
    """T667: the text translate_document()'s pieces write: each text added, each Withdrawn's characters taken back from
    the end of the text (never a mark), each Cut written as mark"""
    text = ""
    for piece in pieces:
        if isinstance(piece, Withdrawn):
            text = text[:len(text) - piece.characters]
        else:
            text += mark if isinstance(piece, Cut) else piece
    return text


class Document:
    """T667: a document's translation as it goes (translate_document()'s): the history, the sums of the stats, how many
    were taken back and cut, and how many characters of text it wrote (where a run began, for its Withdrawn). Its
    generators yield what translate_document() yields and return what they say."""

    def __init__(self, model, count, history_most, request, settings, penalized):
        self.model, self.count, self.history_most = model, count, history_most
        self.request, self.settings, self.penalized = request, settings, penalized
        self.pairs, self.costs = [], []
        self.sums = {"tokens": 0, "sampled": 0, "prompt_tokens": 0, "seconds": 0.0}
        self.retries = self.cut = self.written = 0

    def out(self, text):
        if text:
            self.written += len(text)
            yield text

    def withdraw(self, start):
        """take back what was written after start"""
        if self.written > start:
            yield Withdrawn(self.written - start)
            self.written = start

    def run(self, source, penalized=None):
        """source translated once with the latest history, written as it comes. Returns its translation, why it ran
        away (LOOP, LONG, SHORT; None: it did not), and the characters of it worth keeping where it fell into a loop"""
        request = pfml_tokens(self.model.tokenizer, source, **self.request,
                              history=recent(self.pairs, self.costs, self.history_most))
        source_tokens = self.count(source)
        written = []
        for piece in self.model.generate(request, steps=len(request) + 1 + LONGER * source_tokens + MORE, echo=False,
                                         loops=True, **{**self.settings, **(penalized or {})}):
            written.append(piece)
            yield from self.out(piece)
        stats = self.model.stats
        for key in self.sums:
            self.sums[key] += stats[key]
        tokens = stats["tokens"] - stats["prompt_tokens"]
        if stats["loop"] is not None:
            reason = LOOP
        elif stats["sampled"] <= tokens:  # (a stop token ends a run with one more draw than there are tokens after the prompt)
            reason = LONG
        elif source_tokens >= SHORT_SOURCE and tokens * SHORTER < source_tokens:
            reason = SHORT
        else:
            reason = None
        return "".join(written), reason, stats["loop"]

    def remember(self, source, translation):
        translation = translation.strip()
        if translation:
            self.pairs.append((source, translation))
            self.costs.append(self.count(source) + self.count(translation) + HISTORY_PAIR)

    def piece(self, index, text, level=0):
        """text translated (of the chunk index); where it runs away, taken back and translated again in the pieces of
        the first finer way from level (FINER) that cuts it in more than one, or, where none does, cut (last())"""
        start = self.written
        translation, reason, keep = yield from self.run(text)
        if reason is None:
            self.remember(text, translation)
            return
        self.retries += 1
        for finer in range(level, len(FINER)):
            leading, pieces = FINER[finer](text, self.count)
            if len(pieces) > 1:
                break
        else:
            yield from self.last(index, text, start, reason, keep)
            return
        yield from self.withdraw(start)
        yield from self.out(leading)
        for piece, separator in pieces:
            yield from self.piece(index, piece, finer + 1)
            yield from self.out(separator)

    def last(self, index, text, start, reason, keep):
        """text that no finer way cuts, whose translation from start ran away for reason: translated once more with the
        penalized settings where there are, and what still ran away cut after its loop's first time (or kept as far as
        it ran) and marked"""
        if self.penalized is not None:
            yield from self.withdraw(start)
            translation, reason, keep = yield from self.run(text, self.penalized)
            if reason is None:
                self.remember(text, translation)
                return
        if reason == LOOP:
            yield from self.withdraw(start + keep)
        self.cut += 1
        yield Cut(index, reason, text)


# T667: the finer ways a text that ran away is cut, the first first (each returns what split() returns)
FINER = (lambda text, count: split(text, RETRY_TOKENS, count), lambda text, count: sentences_of(text))


def translate_document(model, text, target, source=None, form=None, context=None, glossary=(),
                       chunk_tokens=CHUNK_TOKENS, history_tokens=HISTORY_TOKENS, told=None, penalized=PENALIZED,
                       **settings):
    """Yield the translation of text, piece by piece as it is written: the whitespace it begins with, then each
    chunk's translation (split()'s) and the separator after the chunk as it is. model: a PLaMo 3 Translate (its
    tokenizer and generate()); target, source, form, context, glossary: pfml_pieces()'s; settings: generate()'s
    sampling (none: greedy). told(index, count, chunk): called before each chunk is translated (where a page would
    say how far it is).

    Each chunk is a new request (pfml_tokens()), its history the latest chunks and their translations that fit in
    the history's budget (budgets(), recent()), and generate() runs it from the start, its translation at most
    LONGER times the chunk's tokens and MORE, ended where it falls into a loop (loops=True). T667: a translation that
    runs away is taken back (a Withdrawn is yielded: the last so many characters written are not the translation) and
    its text translated again in smaller pieces (Document.piece()); one that no finer way cuts is cut where it ran
    away and a Cut is yielded, where the page writes its mark (penalized: the sampling settings of one more try before
    that, T668's). applied() puts the pieces together. A translation that ran away is not put in the history.
    model.stats is left with the sums over the runs (tokens, sampled, prompt_tokens, seconds) and the numbers of chunks
    (chunks), of translations taken back (retries) and of pieces cut (cut)."""
    tokenizer = model.tokenizer

    def count(said):
        return len(tokenizer.encode(said, literal=True))

    chunk_most, history_most = budgets(count, model.seq_len, context, glossary, chunk_tokens, history_tokens)
    leading, chunks = split(text, chunk_most, count)
    document = Document(model, count, history_most,
                        {"target": target, "source": source, "form": form, "context": context, "glossary": glossary},
                        settings, penalized)
    yield from document.out(leading)
    for index, (chunk, separator) in enumerate(chunks):
        if told is not None:
            told(index, len(chunks), chunk)
        yield from document.piece(index, chunk)
        yield from document.out(separator)
    model.stats = {**document.sums, "chunks": len(chunks), "retries": document.retries, "cut": document.cut}
