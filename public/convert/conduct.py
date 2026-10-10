# The conduct of one conversion (T374.1): which files a model needs, in which order, how much of each head, which
# tokenizer is tried next and where the chat template comes from. It asks and is answered; it fetches nothing itself.
# No I/O, no event loop and no JavaScript in here: whoever drives it (the worker's loop from T374.2 on, a test's
# dictionary, a tool's folder) gets a request from next() or send(answer) and sends the answer back.
#
#   steps = conduct(hf, dtype=..., sink=..., quantize_rows=..., readers=...)
#   request = next(steps)
#   while request[0] not in ("done", "missing"):
#       request = steps.send(answer_to(request))
#
# What is asked (where: "weights", the repository of the weights, or "vocabulary", the original's, which is asked
# only of a model that names one; a folder of the visitor's disk is one place, asked by the same names, T374.2.2):
#
#   ("text", where, name)                    the file's text, or None where the file is not there
#   ("bytes", where, name)                   the file's bytes, or None
#   ("range", where, name, begin, end)       (the bytes [begin, end), fewer where the file ends first; the size of the
#                                            whole file, or None where the answer does not know it), or None where
#                                            the file is not there
#   ("size", where, name)                    the size of the whole file (asked only after a range that did not say it)
#   ("stream", where, name, begin, end, before, total, feed)
#                                            the bytes [begin, end), in the order of the file, cut into parts as the
#                                            answerer likes: each part goes to feed(part), which returns the share
#                                            converted so far. Answered (with None) once the last part is fed.
#                                            before and total are for whoever tells of the progress: of total bytes,
#                                            before are in when this stream begins
#   ("done", conversion)                     the end: the conversion, finished (its checkpoint or what its sink took,
#                                            its options and its tokenizer)
#   ("missing", where, name)                 the other end: a file the conversion cannot do without is not there
#
# "Not there" is an answer (a 404): the conduct goes on to the next candidate, or ends with "missing" and leaves the
# words to whoever knows the place. Every other failure (the line, a refusal of the server, a cancelled load) is the
# answerer's, who stops asking: close() the generator. A file the converter cannot read is an exception of the
# converter's, as it was.
#
# The large bytes pass as they did, and not through here: the answerer hands each part to the conversion's feed()
# itself, and what comes out goes to the sink. Nothing of the conduct runs for a part (T374.2.1: a part costs what
# it did before there was a conduct, by the same call), and nothing is known here of how the parts are cut, tried
# again, kept or cancelled.
import json
from typing import Any, Callable, NamedTuple

from convert.conversion import Conversion
from convert.gguf import Incomplete, gguf_weights
from convert.sources import joined_shards

# how much of a file's beginning is asked for first: the JSON header of a safetensors file is a few dozen kilobytes
HEAD = 512 * 1024
# a GGUF's head holds its vocabulary, a few megabytes: asked for in pieces that grow until the converter reads all of it
GGUF_HEAD, GGUF_GROWS = 4 * HEAD, 4
# The tokenizers of a model that names none, in the order they are tried: the first that is there and that the
# converter can read. The one list of them (T374.2.3): a repository nobody has looked at (?hf=), a folder of the
# visitor's disk and the build's directories (convert_hf.py) all name none and get these. Only the page's look into a
# folder, which comes before there is a Pyodide to ask, keeps the names as well (TOKENIZERS of src/page/folder.ts):
# tests/worker-fetches-check.mjs holds that list to this one. A name that ends in .json is read as a tokenizer.json,
# any other as a sentencepiece model (Conversion)
TOKENIZERS = ("tokenizer.json", "tokenizer.model", "spiece.model")


class Lost(Exception):
    """A file that is needed is not there: (where, name)."""


class Another(Exception):
    """This is not that kind of source: the next row of SOURCES is tried. reason: what stands where no row takes it."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Weights(NamedTuple):
    """What a row of SOURCES found: the safetensors-like header (text) and where its tensors begin, as Conversion()
    takes them; the streams to feed, one after another: (name, begin, end); before and total: see "stream" above.
    conversion: made already, where the head held the configuration and the vocabulary as well (a GGUF alone)."""
    header: Any
    base: int
    streams: list
    before: int
    total: int
    conversion: Any = None


def vocabulary_of(hf):
    return hf.get("vocabulary") or None


def place_of(hf):
    """Where config.json, the tokenizer and the template are: the original's repository, where the model names one."""
    return "vocabulary" if vocabulary_of(hf) else "weights"


def is_gguf(hf):
    return str(hf["weights"]).endswith(".gguf")


def needed(request):
    """The answer to a request for a file there must be."""
    answer = yield request
    if answer is None:
        raise Lost(request[1], request[2])
    return answer


def beginning(where, name, end):
    """(the first end bytes of a file, the size of the whole file)."""
    data, size = yield from needed(("range", where, name, 0, end))
    if not size or size != size:  # (None, 0, or the NaN JavaScript makes of a header that is not there)
        size = yield ("size", where, name)
    return data, int(size)


def whole(data):
    return bytes(data.to_py() if hasattr(data, "to_py") else data)


def safetensors_head(name):
    """(the JSON header of a safetensors file, where its tensors begin, the size of the file). The file begins with 8
    bytes that say how long the header is."""
    data, size = yield from beginning("weights", name, HEAD)
    data = whole(data)
    length = int.from_bytes(data[:8], "little") if len(data) >= 8 else -1
    if not 2 <= length <= 100e6:
        raise ValueError("This is not a safetensors file.")
    base = 8 + length
    if base > len(data):
        # (from the file's first byte again, as the worker asked: T381 is T374.3's)
        data = whole((yield from needed(("range", "weights", name, 0, base)))[0])
    return data[8:base].decode("utf-8", "replace"), base, size


def gguf_head(name, read):
    """(read(head) of a GGUF's beginning, the size of the file): asked for in growing pieces until read() has all of
    the head (Incomplete until then; and for good where the file ends first)."""
    want = GGUF_HEAD
    while True:
        # (from the file's first byte every time, as the worker asked: T381 is T374.3's)
        data, size = yield from beginning("weights", name, want)
        try:
            return read(data), size
        except Incomplete:
            if want >= size:
                raise
        want *= GGUF_GROWS


def gguf_alone(hf, config, make):
    """T74: a GGUF holds the configuration and the vocabulary before its tensors: no config.json and no tokenizer."""
    conversion, size = yield from gguf_head(hf["weights"], lambda head: Conversion.from_gguf(head, **make))
    return Weights(None, conversion.base, [(hf["weights"], conversion.base, size)], conversion.base, size, conversion)


def gguf_with_vocabulary(hf, config, make):
    """T136: the GGUF's header as a safetensors one, once the original's config.json agrees with it."""
    (header, base), size = yield from gguf_head(hf["weights"], lambda head: gguf_weights(head, config))
    return Weights(header, base, [(hf["weights"], base, size)], base, size)


def one_file(hf, config, make):
    try:
        header, base, size = yield from safetensors_head(hf["weights"])
    except (Lost, ValueError) as error:
        raise Another(error) from None
    return Weights(header, base, [(hf["weights"], base, size)], base, size)


def shards_of(index):
    """The files model.safetensors.index.json names, in the order of their names, which is the order they are fed in
    (T105). None of them for an index that says none."""
    try:
        names = json.loads(index)["weight_map"].values()
    except (ValueError, TypeError, KeyError, AttributeError):
        return []
    # (the order of JavaScript's sort(), which the worker's was: by UTF-16 code units)
    return sorted({name for name in names if isinstance(name, str)}, key=lambda name: name.encode("utf-16-be", "surrogatepass"))


def several(files):
    """T105: the shards' headers joined into the header of one file made of their data one after another. Each shard
    is then fed from its own base, the next after it."""
    heads = []
    for name in files:
        heads.append((yield from safetensors_head(name)))
    header, lengths = joined_shards([head[0] for head in heads])
    streams = [(name, base, base + int(length)) for name, (_, base, _), length in zip(files, heads, lengths)]
    return Weights(header, 0, streams, 0, sum(end - begin for _, begin, end in streams))


def indexed(hf, config, make):
    """A model without its one file is split over several (model-00001-of-00002.safetensors, ...), or published under
    the name of a shard though there is only one (T78): the index says which."""
    index = yield ("text", "weights", f"{hf['weights']}.index.json")
    files = shards_of(index) if index else []
    if not files:
        raise Another(None)
    if len(files) == 1:
        header, base, size = yield from safetensors_head(files[0])
        return Weights(header, base, [(files[0], base, size)], base, size)
    return (yield from several(files))


class Source(NamedTuple):
    """A kind of source. takes(hf): whether the model as it is listed may be of this kind, before anything is asked.
    opened(hf, config, make): a generator that asks for the head of the weights and returns Weights, or raises
    Another where what was answered says it is not this kind. alone: nothing but the weights is asked for (config
    is None then)."""
    title: str
    takes: Callable
    opened: Callable
    alone: bool = False


# The kinds of source, in the order they are tried. To add one: a row here and its opened() above.
SOURCES = (
    Source("a GGUF alone", lambda hf: is_gguf(hf) and not vocabulary_of(hf), gguf_alone, alone=True),
    Source("a GGUF with the vocabulary of another repository", lambda hf: bool(vocabulary_of(hf)), gguf_with_vocabulary),
    Source("one safetensors file", lambda hf: not is_gguf(hf) and not vocabulary_of(hf), one_file),
    Source("the safetensors files its index names", lambda hf: not is_gguf(hf) and not vocabulary_of(hf), indexed),
)


def says_a_template(tokenizer_config):
    """Whether tokenizer_config.json has a chat template (as the worker told: whatever JavaScript calls true)."""
    def no_json(constant):  # (NaN and Infinity, which Python's reader takes and JavaScript's does not)
        raise ValueError(constant)

    try:
        template = json.loads(tokenizer_config, parse_constant=no_json).get("chat_template")
    except (ValueError, AttributeError):
        return False
    return template is not None and template is not False and template != 0 and template != ""


def candidates_of(hf):
    named = (vocabulary_of(hf) or {}).get("tokenizer") or hf.get("tokenizer")
    return [named] if isinstance(named, str) else list(named or TOKENIZERS)


def tokenized(hf, weights, config, make):
    """Conversion() with the template of the model, where it publishes one, and the first tokenizer of the candidates
    that is there and that the converter can read."""
    place = place_of(hf)
    # The format of one turn (T73). An optional file: a model without one keeps the format its item has
    tokenizer_config = (yield ("text", place, "tokenizer_config.json")) or ""
    # T127: newer repositories keep the template in chat_template.jinja instead. Asked for only where
    # tokenizer_config.json has none: most repositories have no such file, and WebKit reports each 404 as an error
    chat_template = None if says_a_template(tokenizer_config) else (yield ("text", place, "chat_template.jinja")) or None
    # Where no candidate will do, the converter's refusal of one that is there says why; a file that is not there is
    # said only where no other was there either (T144)
    refusal = missing = None
    for candidate in candidates_of(hf):
        tokenizer = yield ("bytes", place, candidate)
        if tokenizer is None:
            missing = missing or Lost(place, candidate)
            continue
        try:
            return Conversion(weights.header, weights.base, config, tokenizer, candidate, start=weights.base,
                              tokenizer_config=tokenizer_config, chat_template=chat_template, **make)
        except Exception as error:  # (whatever it was refused with: the next candidate may do)
            refusal = refusal or error
    raise refusal or missing


def opened(hf, make):
    """The conversion of a model, ready to be fed, and its Weights: by the first row of SOURCES that takes it."""
    rows = [row for row in SOURCES if row.takes(hf)]
    config = None if rows[0].alone else (yield from needed(("text", place_of(hf), hf.get("config") or "config.json")))
    reason = None
    for row in rows:
        try:
            weights = yield from row.opened(hf, config, make)
            break
        except Another as other:
            reason = reason or other.reason
    else:
        raise reason
    return weights.conversion or (yield from tokenized(hf, weights, config, make)), weights


def conduct(hf, **make):
    """The whole conversion of one model, as a generator of requests (see the top of the file).
    hf: {"weights": the name of the file, "config": config.json's name (or none), "tokenizer": a name or the names to
    try (or none: TOKENIZERS), "vocabulary": {"tokenizer": ...} where the vocabulary is another repository's (T136)}.
    Names only: where the files are (a repository, a folder) is the answerer's to know.
    make: what Conversion() takes besides the files (dtype, max_seq_len, sink, quantize_rows, readers)."""
    try:
        conversion, weights = yield from opened(hf, make)
    except Lost as lost:
        yield ("missing", *lost.args)
        return
    before = weights.before
    for name, begin, end in weights.streams:
        yield ("stream", "weights", name, begin, end, before, weights.total, conversion.feed)
        before += end - begin
    conversion.finish()
    yield ("done", conversion)
