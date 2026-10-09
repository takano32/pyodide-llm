# Where the weights come from: a safetensors file, the shards of one model, arrays in memory. Each gives the same
# header and read(offset, length).
import json
import math
import struct

from convert.readers import READERS

ROTATED = "rotated"  # the key of a header's __metadata__ that says a rotated basis (T237)


def header_rotated(header):
    """T237: the rotated basis a safetensors-like header says its matrices are in (FORM's "rotated": the block and
    the signs of every width, llama2_numpy.rotated_form), None where it says none. gguf_model() writes it there from
    a GGUF's prism.hadamard.* (gguf_rotated); a file's own __metadata__ holds texts, so it is JSON's text."""
    said = (header.get("__metadata__") or {}).get(ROTATED)
    if said is None:
        return None
    try:
        rotated = json.loads(said)
        return {"block": int(rotated["block"]), "signs": {str(int(width)): str(bits) for width, bits in rotated["signs"].items()}}
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError("This file's rotated basis is not said in a way the converter reads.") from None


class Safetensors:
    """The tensors of a .safetensors file behind read(offset, length): a local file, a File of the browser, a URL."""

    def __init__(self, read):
        self.read = read
        (header_size,) = struct.unpack("<Q", bytes(read(0, 8)))
        if not 2 <= header_size <= 100_000_000:
            raise ValueError("This is not a safetensors file.")
        try:
            header = json.loads(bytes(read(8, header_size)))
            self.tensors = {name: info for name, info in header.items() if name != "__metadata__"}
        except ValueError:
            raise ValueError("This is not a safetensors file.") from None
        self.rotated = header_rotated(header)
        self.base = 8 + header_size

    def __contains__(self, name):
        return name in self.tensors

    def shape(self, name):
        return tuple(self.tensors[name]["shape"])

    def rows(self, name, start, stop):
        """Rows start..stop of a tensor (all of a vector), as float32 or float16."""
        info = self.tensors[name]
        if info["dtype"] not in READERS:
            raise ValueError(f"{name} is stored as {info['dtype']}: only float32, float16 and bfloat16 are supported.")
        itemsize, reader = READERS[info["dtype"]]
        shape = self.shape(name)
        row = math.prod(shape[1:])
        begin = self.base + info["data_offsets"][0] + int(start * row * itemsize)
        return reader(self.read(begin, int((stop - start) * row * itemsize))).reshape(stop - start, *shape[1:])


class Shards:
    """The tensors of a model split over several .safetensors files (model-00001-of-00002.safetensors, ...), as one
    source: the build's way in (convert_hf.py). Each shard is a Safetensors; a name is looked up in whichever has it."""

    def __init__(self, shards):
        self.owner = {}
        for shard in shards:
            for name in shard.tensors:
                if name in self.owner:
                    raise ValueError(f"{name} is in two shards of this model.")
                self.owner[name] = shard
        self.tensors = {name: shard.tensors[name] for name, shard in self.owner.items()}

    def __contains__(self, name):
        return name in self.owner

    def shape(self, name):
        return self.owner[name].shape(name)

    def rows(self, name, start, stop):
        return self.owner[name].rows(name, start, stop)


def joined_shards(headers):
    """The page's way in for a model split over several files (T105): the JSON headers of the shards, in the order
    their data will be fed, as the header of one file made of their tensor data one after another (base 0). Returns
    that header (text) and, for every shard, how many bytes of data it has: feed each shard from its own base (8 +
    the length of its header) for that many bytes, and Stream sees one file. Nothing of Stream changes."""
    joined, at, lengths = {}, 0, []
    for text in headers:
        try:
            header = json.loads(bytes(text.to_py() if hasattr(text, "to_py") else text).decode()
                                if not isinstance(text, str) else text)
        except ValueError:
            raise ValueError("A shard of this model is not a safetensors file.") from None
        tensors = {name: info for name, info in header.items() if name != "__metadata__"}
        length = max((info["data_offsets"][1] for info in tensors.values()), default=0)
        for name, info in tensors.items():
            if name in joined:
                raise ValueError(f"{name} is in two shards of this model.")
            begin, end = info["data_offsets"]
            joined[name] = {**info, "data_offsets": [at + begin, at + end]}
        lengths.append(length)
        at += length
    return json.dumps(joined), lengths


class Arrays:
    """The same interface for tensors that are in memory already (a PyTorch checkpoint, a test)."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __contains__(self, name):
        return name in self.tensors

    def shape(self, name):
        return tuple(self.tensors[name].shape)

    def rows(self, name, start, stop):
        return self.tensors[name][start:stop]
