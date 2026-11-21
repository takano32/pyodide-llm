"""tests/document_measure.py (T666): how a long text comes through engine/document.py's translate_document() on a
PLaMo 3 Translate at a size of chunk and of history: per chunk, its tokens and its translation's, whether a stop token
ended it, whether it repeats itself, whether it stopped far too short, and how many of its paragraphs it lost. The
measure of tanakh/plamo-translate-desktop's engine::document (0.8B on the head of an English book: the chunks that
repeat, the paragraphs dropped without a word). Run inside Pyodide by tests/document_measure.ts, which holds the
model; this file is its Python, read as the source of a module (no import of the tree's tools).

A translation repeats where, for a period of up to PERIOD tokens, a stretch of at least REPEATED tokens (and three
periods) is each the token a period before it: a loop of the text, which a translation of prose does not have (T667's
detector is finer; this one only counts). One stops far too short where a chunk of SHORT tokens or more has a
translation of less than a quarter of them (tanakh's). A paragraph is lost where the chunk has more paragraphs (cut
at empty lines) than its translation. T667: the translation is what is left after translate_document() took back what
ran away (counted per chunk: withdrawn) and cut what still did (cut, written MARK); the counts are of that."""
import re
import time

from engine.document import Cut, Withdrawn, translate_document

PERIOD, REPEATED, SHORT = 128, 128, 200
MARK = "[CUT]"
EMPTY_LINE = re.compile(r"\n[ \t\r]*\n")


def repeats(tokens):
    """whether a stretch of tokens is a loop (the module's doc)"""
    for period in range(1, PERIOD + 1):
        run = 0
        for at in range(period, len(tokens)):
            run = run + 1 if tokens[at] == tokens[at - period] else 0
            if run >= max(REPEATED, 2 * period):
                return True
    return False


def paragraphs(text):
    return len([part for part in EMPTY_LINE.split(text.strip()) if part.strip()])


def measure(llama, text, target, chunk_tokens, history_tokens):
    """the chunks of text as translate_document() translates them into target, each with what the module's doc says"""
    tokenizer = llama.tokenizer
    count = lambda said: len(tokenizer.encode(said, literal=True))  # noqa: E731
    chunks, out = [], ""

    def told(index, number, chunk):
        chunks.append({"index": index, "of": number, "source": chunk, "began": len(out), "withdrawn": 0, "cut": 0})

    began = time.perf_counter()
    # (T667: a Withdrawn takes back what the chunk wrote that ran away, a Cut is where a piece was cut: MARK here)
    for piece in translate_document(llama, text, target, chunk_tokens=chunk_tokens, history_tokens=history_tokens, told=told):
        if isinstance(piece, Withdrawn):
            out = out[:len(out) - piece.characters]
            chunks[-1]["withdrawn"] += 1
        elif isinstance(piece, Cut):
            out += MARK
            chunks[-1]["cut"] += 1
        else:
            out += piece
    seconds = time.perf_counter() - began
    stats = dict(llama.stats)
    for at, chunk in enumerate(chunks):
        end = chunks[at + 1]["began"] if at + 1 < len(chunks) else len(out)
        translation = out[chunk["began"]:end].strip()
        source_tokens, tokens = count(chunk["source"]), tokenizer.encode(translation, literal=True)
        chunk.update({"translation": translation, "source_tokens": source_tokens, "tokens": len(tokens),
                      "repeats": repeats(tokens), "short": source_tokens >= SHORT and len(tokens) * 4 < source_tokens,
                      "lost": max(0, paragraphs(chunk["source"]) - paragraphs(translation))})
        del chunk["began"]
    return {"chunks": chunks, "seconds": seconds, "retries": stats["retries"], "cut": stats["cut"],
            "prompt_tokens": stats["prompt_tokens"], "sampled": stats["sampled"], "translation": out}
