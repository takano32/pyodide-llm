# conduct_hub.py (T374.1): what tests/test_conduct.py drives the conduct of a conversion (public/convert/conduct.py)
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
        """What the worker's loop says of a file a folder's conduct ends for want of (conduct.js's answered())."""
        return f"The conversion needs {name}, which is not there."

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
