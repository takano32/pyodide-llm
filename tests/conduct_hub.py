# conduct_hub.py (T374.1): what tests/test_conduct.py drives the conduct of a conversion (src/python/convert/conduct.py)
# with, and what it holds it to.
#
#   Hub        a dictionary for huggingface.co: {"<repository>/<file>": a text, bytes, or a File}, which answers what is
#              asked and writes each request down as tests/worker-fetches-check.mjs writes the worker's
#              ("GET owner/model@0123456 model.safetensors bytes=0-524287"). What only the answerer decides is the
#              hub's: how a stream is cut into parts (see parts()), and whether a range's answer says the file's size
#   Folder     the same for a folder of the visitor's disk (T374.2.2): {"<name>": ...}, found by its name whatever the
#              case of its letters, which writes down what is read of each file as worker-fetches-check.mjs's Files do
#              ("range model.safetensors bytes=0-524287"). The disk's is how a stream is cut: a MiB at a time
#   answered   the loop that answers a conduct from a hub or a folder: the worker's side (public/worker/conduct.js)
#   direct     a conversion with no conduct in it (T374.2.3): the files read whole, a Conversion made of them and fed
#              in one piece. What a conduct's conversion is compared with on real files, since the worker's old ladder
#              (and today(), its copy in Python, which this took the place of) is gone
#   sound      what the requests of any conduct must be, whatever the repository: what needs no second opinion to say
#   StandIn    the converter's stand-in of worker-fetches-check.mjs in Python: it reads the made-up files as far as the
#              conduct depends on it and writes down what it is handed
import json
import struct

from convert.gguf import Incomplete

MiB = 1 << 20


class File:
    """A file of size bytes that begins with head; zeros after it (as worker-fetches-check.mjs's stream())."""

    def __init__(self, head, size=None):
        self.head, self.size = bytes(head), len(head) if size is None else size

    def __getitem__(self, range):
        begin, end = range.start or 0, min(self.size if range.stop is None else range.stop, self.size)
        got = self.head[begin:end]
        return got + bytes(max(end - begin, 0) - len(got))


def safetensors(data, header=300):
    """8 bytes that say how long its header is, the header (which says how many bytes of tensors follow), the tensors"""
    text = json.dumps({"__metadata__": {"data": data}}, separators=(",", ":")).ljust(header).encode()
    return File(struct.pack("<Q", len(text)) + text, 8 + len(text) + data)


def gguf(base, data):
    """"GGUF", then 8 bytes that say where its tensors begin"""
    return File(b"GGUF" + struct.pack("<Q", base), base + data)


def made(files):
    """The files of tests/fixtures/conversion-cases.json, as the check that wrote them made them."""
    makers = {"safetensors": safetensors, "gguf": gguf}
    return {name: value if isinstance(value, str) else makers[next(iter(value))](**next(iter(value.values())))
            for name, value in files.items()}


def cut(line, unit=MiB):
    """How the worker's inOrder() cuts a stream on this line (ranges.js): the first part is 8 MiB, and so is every part
    asked for before the first one is in (one a connection, of six; two a connection where the first part is the last
    to come); the rest are 16 MiB where the first came at 4 MB/s or more and the device has more than 4 GB. Six
    connections take a MiB each in turn: the first part's 8 MiB take 48 delays."""
    rate = 8 * MiB / (48 * line["delay"] / 1000) if line.get("delay") else float("inf")
    fast = rate >= 4e6 and line.get("deviceMemory", 8) > 4
    return dict(first=8 * unit, ahead=12 if "held" in line else 6, rest=(16 if fast else 8) * unit)


class Hub:
    """hf: the model as it is listed (repo, revision, vocabulary: where each place is). unsaid: a range's answer does
    not say the size of the file. first, ahead, rest: see cut()."""

    def __init__(self, files, hf, unsaid=False, first=8 * MiB, ahead=6, rest=16 * MiB):
        self.files, self.hf, self.unsaid, self.first, self.ahead, self.rest = files, hf, unsaid, first, ahead, rest
        self.asked = []

    def at(self, where):
        place = self.hf if where == "weights" else self.hf["vocabulary"]
        return place["repo"], place["revision"]

    def found(self, method, where, name, range=None):
        repository, revision = self.at(where)
        self.asked.append(f"{method} {repository}@{revision[:7]} {name}" + (f" bytes={range[0]}-{range[1] - 1}" if range else ""))
        return self.files.get(f"{repository}/{name}")

    def refusal(self, where, name):
        """What the worker says of a file that is not there (ranges.js's refused())."""
        repository, revision = self.at(where)
        return f"{repository} has no {name} at {revision} on huggingface.co, or has no commit {revision}."

    def text(self, where, name):
        found = self.found("GET", where, name)
        return found if found is None or isinstance(found, str) else found[:].decode()

    def bytes(self, where, name):
        found = self.found("GET", where, name)
        return found.encode() if isinstance(found, str) else found if found is None else found[:]

    def range(self, where, name, begin, end):
        found = self.found("GET", where, name, (begin, end))
        if found is None:
            return None
        found = found.encode() if isinstance(found, str) else found
        return found[begin:end], None if self.unsaid else found.size if isinstance(found, File) else len(found)

    def size(self, where, name):
        found = self.found("HEAD", where, name)
        return found.size if isinstance(found, File) else len(found)

    def whole(self, where, name):
        """The bytes of a file, or None, for a test's own reading: nothing is written down as asked."""
        found = self.files.get(f"{self.at(where)[0]}/{name}")
        return found if found is None else found.encode() if isinstance(found, str) else bytes(found[:])

    def parts(self, where, name, begin, end):
        at, part = begin, 0
        while at < end:
            stop = min(at + (self.first if part < self.ahead else self.rest), end)
            yield self.found("GET", where, name, (at, stop))[at:stop]
            at, part = stop, part + 1


class Folder:
    """files: {"<name>": a text, bytes or a File}. chunk: how much of a stream the disk gives at a time."""

    def __init__(self, files, chunk=MiB):
        self.files, self.chunk, self.asked = {}, chunk, []
        for name, value in files.items():
            value = File(value.encode()) if isinstance(value, str) else File(value) if isinstance(value, bytes) else value
            self.files.setdefault(name.lower(), (name, value))  # (of two names that differ in their case alone, the first)

    def found(self, kind, name, range=None):
        """The file of that name, what is read of it written down; None, and nothing written, where there is none."""
        name, file = self.files.get(name.lower(), (None, None))
        if file is None:
            return None
        if range:
            begin = min(range[0], file.size)
            end = max(begin, min(range[1], file.size))
            range = (begin, end)
        self.asked.append(f"{kind} {name}" + (f" bytes={range[0]}-{range[1] - 1 if range[0] < range[1] else ''}" if range else ""))
        return file

    def refusal(self, where, name):
        """What the worker says of a file a folder's conduct ends for want of (conduct.js's fromFolder(), T427)."""
        return f"The folder has no {name}, which the model needs."

    def text(self, where, name):
        found = self.found("text", name)
        return found if found is None else found[:].decode()

    def bytes(self, where, name):
        found = self.found("bytes", name)
        return found if found is None else found[:]

    def range(self, where, name, begin, end):
        found = self.found("range", name, (begin, end))
        return found if found is None else (found[begin:end], found.size)

    def size(self, where, name):
        return self.files[name.lower()][1].size

    def whole(self, where, name):
        """The bytes of a file, or None, for a test's own reading: nothing is written down as read."""
        file = self.files.get(name.lower(), (None, None))[1]
        return file if file is None else bytes(file[:])

    def parts(self, where, name, begin, end):
        file = self.found("stream", name, (begin, end))
        for at in range(begin, end, self.chunk):
            yield file[at:min(at + self.chunk, end)]


def answer(hub, request):
    kind = request[0]
    if kind in ("text", "bytes", "range", "size"):
        return getattr(hub, kind)(*request[1:])
    # a stream: every part to the conduct's feed, in the order of the file; answered once the last one is fed
    for part in hub.parts(*request[1:5]):
        request[7](part)
    return None


def answered(hub, steps, told=None):
    """Answers a conduct from the hub until it ends: its last request, ("done", conversion) or ("missing", where,
    name). told: a list for the requests as they came (a stream's without its feed)."""
    request = next(steps)
    while request[0] not in ("done", "missing"):
        if told is not None:
            told.append(request[:7])
        request = steps.send(answer(hub, request))
    return request


class Absent(Exception):
    """A file direct() cannot do without is not there: (where, name)."""


def direct(whole, hf, candidates, **making):
    """The conversion of a model with no conduct in it (T374.2.3): every file read whole and at once by whole(where,
    name) (its bytes, or None), a Conversion made of them straight away, and all the bytes of the tensors fed in one
    piece. What a conduct's conversion is compared with on real files: nothing is asked for here, no head is grown and
    no part is cut, so what the two agree on (the checkpoint, the options, the tokenizer.bin) is neither's habit.
    candidates: the tokenizers to try where the model names none. Raises Absent, or what the converter refuses with."""
    from convert.conversion import Conversion
    from convert.gguf import gguf_weights
    from convert.sources import joined_shards

    def needed(where, name):
        data = whole(where, name)
        if data is None:
            raise Absent(where, name)
        return data

    def header_of(data):
        """(the JSON header of a safetensors file, where its tensors begin), or None of a file that is none."""
        length = int.from_bytes(data[:8], "little") if len(data) >= 8 else -1
        return (data[8:8 + length].decode(), 8 + length) if 2 <= length <= 100e6 else None

    weights, vocabulary = hf["weights"], hf.get("vocabulary")
    place = "vocabulary" if vocabulary else "weights"
    if weights.endswith(".gguf") and not vocabulary:
        data = needed("weights", weights)
        conversion = Conversion.from_gguf(data, **making)
        conversion.feed(data[conversion.base:])
        conversion.finish()
        return conversion
    config = needed(place, hf.get("config") or "config.json").decode()
    if vocabulary:
        data = needed("weights", weights)
        header, base = gguf_weights(data, config)
        tensors = [data[base:]]
    else:
        data = whole("weights", weights)
        heads = [header_of(data)] if data is not None else [None]
        files = [data]
        if heads[0] is None:
            # the index beside it says which files the model is in; where it says none, the one file is what is wrong
            index = whole("weights", f"{weights}.index.json")
            try:
                names = sorted(set(json.loads(index)["weight_map"].values()))
            except (TypeError, ValueError, KeyError, AttributeError):
                names = []
            if not names and data is None:
                raise Absent("weights", weights)
            if not names:
                raise ValueError("This is not a safetensors file.")
            files = [needed("weights", name) for name in names]
            heads = [header_of(file) for file in files]
        if len(files) == 1:
            (header, base), tensors = heads[0], [files[0][heads[0][1]:]]
        else:
            header, lengths = joined_shards([head[0] for head in heads])
            base, tensors = 0, [file[head[1]:head[1] + int(length)] for file, head, length in zip(files, heads, lengths)]
    tokenizer_config = (whole(place, "tokenizer_config.json") or b"").decode()
    try:
        templated = bool(json.loads(tokenizer_config).get("chat_template"))
    except (ValueError, AttributeError):
        templated = False
    chat_template = None if templated else (whole(place, "chat_template.jinja") or b"").decode() or None
    named = (vocabulary or {}).get("tokenizer") or hf.get("tokenizer") or candidates
    refusal = missing = None
    for candidate in [named] if isinstance(named, str) else named:
        tokenizer = whole(place, candidate)
        if tokenizer is None:
            missing = missing or Absent(place, candidate)
            continue
        try:
            conversion = Conversion(header, base, config, tokenizer, candidate, start=base, tokenizer_config=tokenizer_config,
                                    chat_template=chat_template, **making)
            break
        except Exception as error:
            refusal = refusal or error
    else:
        raise refusal or missing
    for data in tensors:
        conversion.feed(data)
    conversion.finish()
    return conversion


# The candidates for a tokenizer once more, as the tests know them: the second copy of convert.conduct's TOKENIZERS
# (tests/test_conduct.py holds that to this). direct() and sound() use these, so that they have nothing of the conduct's
THREE = ("tokenizer.json", "tokenizer.model", "spiece.model")


def sound(told, conversion, ended, hf, hub):
    """told: the requests of one conduct as they came (answered()'s), conversion: what it ended with (None where it
    ended otherwise), ended: how it ended in the words of the fixture's "ended". Each line below is a thing no conversion may do."""
    vocabulary = hf.get("vocabulary")
    place = "vocabulary" if vocabulary else "weights"
    named = (vocabulary or {}).get("tokenizer") or hf.get("tokenizer") or THREE
    candidates = [named] if isinstance(named, str) else list(named)
    alone = hf["weights"].endswith(".gguf") and not vocabulary
    assert {request[0] for request in told} <= {"text", "bytes", "range", "size", "stream"}
    assert {request[1] for request in told} <= ({"weights", "vocabulary"} if vocabulary else {"weights"})
    # nothing but the weights is asked of a GGUF that holds everything; of any other model, config.json comes first
    if alone:
        assert {request[2] for request in told} <= {hf["weights"]}
    else:
        assert told[0] == ("text", place, hf.get("config") or "config.json")
    # a file asked for whole is asked for once
    wholes = [request for request in told if request[0] in ("text", "bytes")]
    assert len(set(wholes)) == len(wholes)
    # the tokenizers: the model's candidates or the three, in their order, each once, none skipped, where the vocabulary is
    tried = [request for request in told if request[0] == "bytes"]
    assert tried == [("bytes", place, candidate) for candidate in candidates[:len(tried)]]
    # and it gives up on them only once every one was tried; the one it takes is the last it asked for, and is there
    if tried and conversion is None:
        assert len(tried) == len(candidates)
    if tried and conversion is not None:
        assert hub.whole(*tried[-1][1:]) is not None
    # the templates: where the tokenizer is, tokenizer_config.json before chat_template.jinja before any tokenizer
    late = [request[2] for request in told if request[1:3] in ((place, "tokenizer_config.json"), (place, "chat_template.jinja")) or request[0] == "bytes"]
    assert late[:1] in ([], ["tokenizer_config.json"]) and late.count("chat_template.jinja") <= 1
    assert "chat_template.jinja" not in late or late.index("chat_template.jinja") == 1
    # a head is asked for from the first byte of its file, and each further piece of it from where the bytes in hand
    # end (T374.3: the end of the piece before, or the file's where that came first): no byte twice, no piece that
    # begins at the file's end or past it (a server refuses that one), and nothing of the weights elsewhere
    heads = {}
    for at, request in enumerate(told):
        if request[0] == "range":
            file = hub.whole("weights", request[2])
            assert request[1] == "weights" and request[3] == heads.get(request[2], 0) < request[4]
            assert request[3] == 0 or request[3] < len(file)
            heads[request[2]] = request[4] if file is None else min(request[4], len(file))
        if request[0] == "size":
            # only of a file whose first range was just answered (without its size)
            assert told[at - 1][:4] == ("range", *request[1:], 0)
    # the streams come last, each of a file whose head was read, one after another: the progress counts across them
    streams = [request for request in told if request[0] == "stream"]
    assert told[len(told) - len(streams):] == streams and len({request[2] for request in streams}) == len(streams)
    before = streams[0][5] if streams else 0
    for _, where, name, begin, end, was, total in streams:
        assert where == "weights" and name in heads and 0 < begin <= end and was == before and total == streams[0][6]
        before += end - begin
    assert not streams or before == streams[0][6]
    if len(streams) == 1:
        # one file: everything after its head, to its last byte; the head counts as arrived
        _, _, name, begin, end, was, total = streams[0]
        assert was == begin and end == total == len(hub.whole("weights", name))
        # T374.3. A GGUF: from where the pieces of its head end, which with the stream are the file, once (or from
        # where its tensors begin, where the head was read whole before the padding after it was: less than 32 bytes
        # by default, and no tensor's). A safetensors file: from the end of its header, within what was read of it
        assert heads[name] <= begin if vocabulary or hf["weights"].endswith(".gguf") else begin <= heads[name]
    # the ends: the streams are asked for only by a conversion that was made, and all of them before it is done
    if ended == "converted":
        assert conversion is not None and streams
    else:
        assert conversion is None
    # a file said to be missing was asked for, and is not there
    lost = [request for request in told if ended == f"failed: {hub.refusal(*request[1:3])}"]
    assert all(hub.whole(*request[1:3]) is None for request in lost)
    if "has no " in ended:
        assert lost and not streams


class StandIn:
    """llama2_convert as the conduct calls it, for the made-up files: handed is what it was given, in the words of
    worker-fetches-check.mjs's stand-in (the fixture's "converter", but for the lines of what the worker keeps)."""

    def __init__(self):
        self.handed = handed = []
        said = lambda text: json.dumps(text, ensure_ascii=False)

        def base_of(first):
            # (the worker's answer is JavaScript's bytes in Pyodide: tests/worker-fetches-check.mjs)
            first = first.to_py() if hasattr(first, "to_py") else first
            base = struct.unpack_from("<Q", bytes(first[:12]), 4)[0]
            if len(first) < base:
                raise Incomplete()
            return base

        class Conversion:
            def __init__(self, header, base, config, tokenizer, name, start=0, tokenizer_config=None, chat_template=None,
                         dtype="int8", what=None, **more):
                what = what or (f"Conversion(a header of {len(header)} characters, base {base}, start {start}, the config {said(config)}, "
                                f"{name} of {len(tokenizer)} bytes, tokenizer_config {said(tokenizer_config)}, chat_template {said(chat_template)})")
                if bytes(tokenizer or b"").startswith(b"unreadable"):
                    handed.append(f"{what}: refused")
                    raise ValueError(f"This model cannot be converted: {name} is of a kind the engine does not read.")
                self.expected = None if header is None or header.startswith("gguf") else \
                    sum(json.loads(header)) if header.startswith("[") else json.loads(header)["__metadata__"]["data"]
                self.fed = self.feeds = 0
                self.base, self.given = base, dict(more, dtype=dtype)
                handed.append(f"{what}, dtype {'the worker' + chr(39) + 's choice' if callable(dtype) else dtype}")

            @classmethod
            def from_gguf(cls, first, **more):
                try:
                    base = base_of(first)
                except Incomplete:
                    handed.append(f"Conversion.from_gguf(the first {len(first)} bytes): not all of the head yet")
                    raise
                return cls(None, base, None, None, None, what=f"Conversion.from_gguf(the first {len(first)} bytes)", **more)

            def feed(self, data):
                self.fed, self.feeds = self.fed + len(data), self.feeds + 1
                return self.fed / self.expected if self.expected else 1

            def finish(self):
                short = "" if self.expected is None or self.fed == self.expected else f", NOT the {self.expected} of its tensors"
                handed.append(f"fed {self.fed} bytes in {self.feeds} pieces{short}; finish()")

        def gguf_weights(first, config):
            try:
                base = base_of(first)
            except Incomplete:
                handed.append(f"gguf_weights(the first {len(first)} bytes): not all of the head yet")
                raise
            handed.append(f"gguf_weights(the first {len(first)} bytes, the config {said(config)})")
            return "gguf header", base

        def joined_shards(headers):
            handed.append(f"joined_shards({len(headers)} headers)")
            lengths = [json.loads(header)["__metadata__"]["data"] for header in headers]
            return json.dumps(lengths, separators=(",", ":")), lengths

        self.Conversion, self.gguf_weights, self.joined_shards = Conversion, gguf_weights, joined_shards

    def into(self, monkeypatch):
        """In the conduct's place of the converter (set in the part that reads the names, not in the window)."""
        for name in ("Conversion", "gguf_weights", "joined_shards"):
            monkeypatch.setattr(f"convert.conduct.{name}", getattr(self, name))
        return self
