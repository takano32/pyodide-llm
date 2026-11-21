"""engine/document.py (T666): a long text translated by PLaMo 3 Translate a piece at a time, each piece with the pieces
before it and their translations as its history (engine/pfml.py's). The text is cut into paragraphs at its empty
lines, and as many whole paragraphs as fit go into one piece (a chunk); a paragraph too long for one is cut at the
first of these that makes its parts fit: a line that ends a sentence, a sentence, a line, a clause (after a comma or
at a space), and last a number of characters. What lies between the chunks (the empty lines) and the whitespace the
text begins with are written out as they are, so that the chunks and what lies between them are the text again.

The form is tanakh/plamo-translate-desktop's engine::document (3d69a116c45a8116159f1a405311a59039ec4b95: none of its
lines). Where it differs: a text of only whitespace is written out as it is (theirs writes nothing), and the numbers
of tokens are this page's (a context of 4096 at most, where theirs is 32768: CHUNK_TOKENS, HISTORY_TOKENS). What
theirs does with a chunk that runs away (repeats itself, or stops far too short) is T667's: here such a chunk is
written as the model wrote it and is not put in the history.

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


def ends_sentence(line):
    """whether a line ends a sentence: in a terminal (Latin or not) and what closes after it, or in a Japanese quote's
    closing"""
    core = line.rstrip(CLOSINGS)
    return any(char in JAPANESE_QUOTES for char in line[len(core):]) or (core != "" and core[-1] in TERMINALS + LATIN_TERMINALS)


def lines_after_sentences(text, start, end, separated):
    """the parts of text[start:end] cut at the newlines after a line that ends a sentence"""
    parts, part = [], start
    at = start
    while (newline := text.find("\n", at, end)) >= 0:
        line_end = trimmed(text, part, newline)
        following = newline
        while following < end and blank(text[following]):
            following += 1
        if line_end > part and following < end and ends_sentence(text[part:line_end]):
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


def translate_document(model, text, target, source=None, form=None, context=None, glossary=(),
                       chunk_tokens=CHUNK_TOKENS, history_tokens=HISTORY_TOKENS, told=None, **settings):
    """Yield the translation of text, piece by piece as it is written: the whitespace it begins with, then each
    chunk's translation (split()'s) and the separator after the chunk as it is. model: a PLaMo 3 Translate (its
    tokenizer and generate()); target, source, form, context, glossary: pfml_pieces()'s; settings: generate()'s
    sampling (none: greedy). told(index, count, chunk): called before each chunk is translated (where a page would
    say how far it is).

    Each chunk is a new request (pfml_tokens()), its history the latest chunks and their translations that fit in
    the history's budget (budgets(), recent()), and generate() runs it from the start, its translation at most
    LONGER times the chunk's tokens and MORE. A chunk whose translation did not end in a stop token (it ran out of
    room) is not put in the history. model.stats is left with the sums over the chunks (tokens, sampled,
    prompt_tokens, seconds) and their number (chunks), how many ran out of room (unfinished)."""
    tokenizer = model.tokenizer

    def count(said):
        return len(tokenizer.encode(said, literal=True))

    chunk_most, history_most = budgets(count, model.seq_len, context, glossary, chunk_tokens, history_tokens)
    leading, chunks = split(text, chunk_most, count)
    if leading:
        yield leading
    pairs, costs = [], []
    sums = {"tokens": 0, "sampled": 0, "prompt_tokens": 0, "seconds": 0.0}
    unfinished = 0
    for index, (chunk, separator) in enumerate(chunks):
        if told is not None:
            told(index, len(chunks), chunk)
        request = pfml_tokens(tokenizer, chunk, target, source=source, form=form, context=context, glossary=glossary,
                              history=recent(pairs, costs, history_most))
        chunk_count = count(chunk)
        written = []
        for piece in model.generate(request, steps=len(request) + 1 + LONGER * chunk_count + MORE, echo=False, **settings):
            written.append(piece)
            yield piece
        stats = model.stats
        for key in sums:
            sums[key] += stats[key]
        # (a stop token ends the run with one more sampling step than there are tokens after the prompt)
        if stats["sampled"] > stats["tokens"] - stats["prompt_tokens"]:
            translation = "".join(written).strip()
            if translation:
                pairs.append((chunk, translation))
                costs.append(chunk_count + count(translation) + HISTORY_PAIR)
        else:
            unfinished += 1
        if separator:
            yield separator
    model.stats = {**sums, "chunks": len(chunks), "unfinished": unfinished}
