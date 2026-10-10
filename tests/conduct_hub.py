# conduct_hub.py (T374.1): what tests/test_conduct.py drives the conduct of a conversion (public/convert/conduct.py)
# with, and what it holds it to.
#
#   Hub        a dictionary for huggingface.co: {"<repository>/<file>": a text, bytes, or a File}, which answers what is
#              asked and writes each request down as tests/worker-fetches-check.mjs writes the worker's
#              ("GET owner/model@0123456 model.safetensors bytes=0-524287"). What only the answerer decides is the
#              hub's: how a stream is cut into parts (see parts()), and whether a range's answer says the file's size
#   answered   the loop that answers a conduct from a hub: what the worker's side becomes in T374.2
#   today      the worker's ladder as it was before T374.2.1 (public/worker/convert.js of a610edb, lines 135 to 352)
#              in Python, call for call, on the same hub: what a conduct is compared with where there is no fixture
#              (real files, made-up repositories by the hundred). The worker's own went with T374.2.1 for the models of
#              huggingface.co; this copy goes with T374.2.3
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

    def parts(self, where, name, begin, end):
        at, part = begin, 0
        while at < end:
            stop = min(at + (self.first if part < self.ahead else self.rest), end)
            yield self.found("GET", where, name, (at, stop))[at:stop]
            at, part = stop, part + 1


def answer(hub, request, streams):
    kind = request[0]
    if kind in ("text", "bytes", "range", "size"):
        return getattr(hub, kind)(*request[1:])
    if kind == "stream":
        streams[:] = [hub.parts(*request[1:5])]
    return next(streams[0], None)


def answered(hub, steps, told=None):
    """Answers a conduct from the hub until it ends: its last request, ("done", conversion) or ("missing", where,
    name). told: a list for the requests as they came."""
    streams, request = [], next(steps)
    while request[0] not in ("done", "missing"):
        if told is not None:
            told.append(request)
        request = steps.send(answer(hub, request, streams))
    return request


class NotThere(Exception):
    """The worker's error of a fetch that was answered 404."""


def truthy(value):
    """As JavaScript's Boolean() of a value JSON.parse() gave"""
    return not (value is None or value is False or value == 0 or value == "")


def parsed(text):
    """JSON.parse()"""
    def refuse(constant):
        raise ValueError(constant)
    return json.loads(text, parse_constant=refuse)


def today(hub, hf, converter, head=512 * 1024, **converting):
    """convert() of public/worker/convert.js for a model of huggingface.co, from where it begins to ask (line 135) to
    conversion.finish(): the same calls in the same order, on the hub. Returns the conversion; raises what the worker
    throws (NotThere for a refused fetch)."""
    def fetched(method, where, name, *more):
        found = getattr(hub, method)(where, name, *more)
        if found is None:
            raise NotThere(hub.refusal(where, name))
        return found

    def sized(name, result):
        data, total = result
        return data, total if total else hub.size("weights", name)

    weights, vocabulary = hf["weights"], hf.get("vocabulary")
    place = "vocabulary" if vocabulary else "weights"
    shards = None
    if weights.endswith(".gguf") and not vocabulary:
        want = 4 * head
        while True:
            first, size = sized(weights, fetched("range", "weights", weights, 0, want))
            try:
                conversion = converter.Conversion.from_gguf(first, **converting)
                break
            except Incomplete:
                if want >= size:
                    raise
            want *= 4
        base = conversion.base
    else:
        def head_of(name):
            data, total = sized(name, fetched("range", "weights", name, 0, head))
            length = int.from_bytes(data[:8], "little") if len(data) >= 8 else -1
            if not 2 <= length <= 100e6:
                raise ValueError("This is not a safetensors file.")
            start = 8 + length
            if start > len(data):
                data = fetched("range", "weights", name, 0, start)[0]
            return name, data[8:start].decode("utf-8", "replace"), start, total

        config = fetched("text", place, hf.get("config") or "config.json")
        if vocabulary:
            want = 4 * head
            while True:
                first, size = sized(weights, fetched("range", "weights", weights, 0, want))
                try:
                    header, base = converter.gguf_weights(first, config)
                    break
                except Incomplete:
                    if want >= size:
                        raise
                want *= 4
        else:
            try:
                _, header, base, size = head_of(weights)
            except Exception as error:
                try:
                    index = fetched("text", "weights", f"{weights}.index.json")
                except Exception:
                    raise error from None
                try:
                    files = sorted(set((parsed(index).get("weight_map") or {}).values()), key=lambda name: name.encode("utf-16-be"))
                except Exception:
                    files = []
                if not files:
                    raise error from None
                if len(files) == 1:
                    weights = files[0]
                    _, header, base, size = head_of(weights)
                else:
                    shards = [head_of(name) for name in files]
                    header, lengths = converter.joined_shards([shard[1] for shard in shards])
                    base, size = 0, sum(lengths)
        try:
            tokenizer_config = fetched("text", place, "tokenizer_config.json")
        except Exception:
            tokenizer_config = ""
        try:
            has_template = truthy(parsed(tokenizer_config)["chat_template"])
        except Exception:
            has_template = False
        chat_template = ""
        if not has_template:
            try:
                chat_template = fetched("text", place, "chat_template.jinja")
            except Exception:
                pass
        refusal = missing = conversion = None
        named = (vocabulary or {}).get("tokenizer") or hf.get("tokenizer")
        for candidate in named if isinstance(named, list) else [named]:
            try:
                tokenizer = fetched("bytes", place, candidate)
            except NotThere as error:
                missing = missing or error
                continue
            try:
                conversion = converter.Conversion(header, base, config, tokenizer, candidate, start=base,
                                                  tokenizer_config=tokenizer_config, chat_template=chat_template or None,
                                                  **converting)
                break
            except Exception as error:
                refusal = refusal or error
        if not conversion:
            raise refusal or missing
    if shards:
        for (name, _, start, _), length in zip(shards, lengths):
            for part in hub.parts("weights", name, start, start + length):
                conversion.feed(part)
    else:
        for part in hub.parts("weights", weights, base, size):
            conversion.feed(part)
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
