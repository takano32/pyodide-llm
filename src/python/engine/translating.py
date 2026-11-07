# T648.4: a translator of Firefox Translations (engine/translation.py's Translator) as the page's worker has a model:
# generate(prompt, **settings) hands over a piece of text a step, stats says what the last answer took, backend how it
# runs. The worker reads it by the names it reads of a Llama (src/runtime/contracts/python.ts's Answering), and its
# generate() runs it as it runs a Llama's: the page's answer is the prompt translated.
#
#     translation(model, source, target, "en", kernels=path, disable=[...], sentences=cut, shortlist=lex)
#
# model: the bytes of the model's file (engine/marian.py reads it where it is: the weights are views of these bytes).
# source, target: the bytes of the two vocabularies, sentencepiece models (the same bytes for a model of one
# vocabulary), read as the converter reads one (convert.tokenizer: the converter's archive is placed first).
# sentences: a function of the text that gives its sentences, every character of the text in one of them (the page's
# Intl.Segmenter, as Firefox cuts a text: the worker hands it); none, the text is one sentence. The language translated
# into says what goes between two sentences of the translation where the source has no line between them (SPACES).
#
# shortlist: the bytes of the model's binary shortlist (lex.50.50.*.s2t.bin), whose rows alone the output layer then
# has, as Firefox translates (T650: fxtranslate's translations with it, faster); none, the whole vocabulary's.
#
# Every sentence is translated by itself, greedily (with the shortlist where one is given), and a sentence of more
# than PIECES pieces is cut into stretches of PIECES that are translated by themselves (fxtranslate's and Marian's
# max-length-break of 128, less the </s>). The settings of an answer (the temperature, the steps...) are a Llama's and
# are not read: a translation is the greedy search's.
import codecs
import time

from engine.marian import read_marian, read_shortlist
from engine.proxy import plain
from engine.tokenizer import Tokenizer
from engine.translation import Translator

PIECES = 127
# the decoder's bound of a sentence (Translator.steps(): min(2 S + 4, 256)), which the page reads as the context's
STEPS = 256


# what goes between two sentences of a translation into a language where the source has no line between them
SPACES = {"en": " ", "ja": ""}


def vocabulary(spm):
    """The engine's Tokenizer of a sentencepiece model's bytes, as the converter reads one (convert/tokenizer.py)."""
    from convert.tokenizer import sentencepiece_charsmap, sentencepiece_options, sentencepiece_pieces, tokenizer_bin

    spm = bytes(spm)
    pieces = list(sentencepiece_pieces(spm))
    options = sentencepiece_options(spm)
    data = tokenizer_bin(pieces, len(pieces), charsmap=sentencepiece_charsmap(spm))
    return Tokenizer(data, len(pieces), kind=options.pop("tokenizer_kind"), **options)


def whole(text):
    """The sentences of a text where nothing cuts them: the text."""
    return [text]


def translation(model, source, target, language, kernels=None, disable=(), sentences=None, shortlist=None):
    """The Translation of a model's bytes and its vocabularies' (the worker's way in: the head of this file), into
    language (its BCP 47 code: SPACES says what sets its sentences apart).
    kernels: the path of the SIMD kernels (engine/kernels.py), whose affine_q8 then does every affine, unless disable
    (?without=) names "kernels"; else NumPy does (T648.3: the same numbers). shortlist: the bytes of the model's
    shortlist, or none (T650)."""
    from engine.kernels import load_kernels

    if language not in SPACES:
        raise ValueError(f"No translation into {language!r} is known here (the page knows {', '.join(SPACES)}).")
    disable = list(plain(disable) or [])
    found = load_kernels(kernels) if kernels and "kernels" not in disable else None
    source_vocabulary = vocabulary(source)
    target_vocabulary = source_vocabulary if bytes(target) == bytes(source) else vocabulary(target)
    lex = read_shortlist(shortlist) if shortlist is not None else None
    translator = Translator(read_marian(model), source_vocabulary, target_vocabulary, kernels=found, shortlist=lex)
    backend = ("int8, SIMD kernels" if found else "int8, NumPy") + (", shortlist" if lex is not None else "")
    return Translation(translator, backend, sentences, SPACES[language])


class Translation:
    """A Translator that answers a prompt with its translation, a piece of text a step. backend: the words of how it
    runs, for the page's status line."""

    def __init__(self, translator, backend, sentences=None, space=" "):
        self.translator, self.backend = translator, backend
        self.seq_len = STEPS
        self.sentences, self.space = sentences or whole, space
        self.stats = {}
        self._run = None

    def stretches(self, text):
        """The source's ids of a sentence: each stretch of at most PIECES pieces, with its </s>."""
        pieces, eos = self.translator.source.encode(text), self.translator.eos[0]
        return [[*pieces[at:at + PIECES], eos] for at in range(0, len(pieces), PIECES)]

    def generate(self, prompt, **settings):
        """The translation of prompt, a piece of text a step of the decoder (and the space or the lines between two
        sentences). What it took is left in stats once the generator is closed, by the names of a Llama's."""
        run = self._run = object()
        translator, eos = self.translator, self.translator.eos[1]
        start = time.perf_counter()
        source_tokens = written = steps = 0
        encoding = 0.0
        first = None
        began = False  # something was written: what follows is set apart from it
        lines = 0  # the lines of the source between the last sentence written and the next
        try:
            for sentence in plain(self.sentences(prompt)):
                core = sentence.strip()
                lines += sentence[:len(sentence) - len(sentence.lstrip())].count("\n")
                for ids in self.stretches(core) if core else []:
                    # (between two sentences the source's lines, or the target's space where it has none)
                    gap = ("\n" * lines or self.space) if began else ""
                    lines = 0
                    source_tokens += len(ids)
                    encoded = time.perf_counter()
                    context = translator.context(ids)
                    encoding += time.perf_counter() - encoded
                    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
                    leading = True  # the stretch's first spaces are not written (Translator.text())
                    for logits in translator.steps(ids, context=context):
                        steps += 1
                        token = int(logits.argmax())  # (the first of equal largest)
                        if token == eos:
                            break
                        written += 1
                        text = utf8.decode(translator.piece(token))
                        if leading:
                            text = text.lstrip()
                            leading = not text
                        if text:
                            text, gap, began = gap + text, "", True
                            first = first if first is not None else time.perf_counter()
                            yield text
                    text = utf8.decode(b"", final=True)
                    if text:
                        yield text
                if core:
                    lines = sentence[len(sentence.rstrip()):].count("\n")
        finally:
            if run is self._run:
                now = time.perf_counter()
                decoding = now - start - encoding
                self.stats = {
                    "tokens": source_tokens + written,
                    "seconds": now - start,
                    "tokens_per_second": steps / decoding if decoding > 0 else 0.0,
                    "sampled": steps,
                    "prompt_tokens": source_tokens,
                    "prompt_seconds": encoding,
                    "prompt_tokens_per_second": source_tokens / encoding if encoding > 0 else 0.0,
                    "first_token_seconds": first - start if first is not None else 0.0,
                    "thought": None,
                    "thought_cut": False,
                }
