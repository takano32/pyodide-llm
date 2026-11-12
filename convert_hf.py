# convert_hf.py
# Convert a Hugging Face Llama checkpoint into what llama2_numpy.py loads, with nothing but NumPy:
#   <out>.bin            llama2.c "legacy" checkpoint (7 int header, then the tensors): float32, float16 or int8
#   <out>.tokenizer.bin  llama2.c tokenizer format, from the sentencepiece model or the tokenizer.json
# It runs when the site is deployed, so no converted binary has to live in the repository. The conversion itself
# is src/python/llama2_convert.py, which the page uses too: this file adds what only the build needs, the files of a
# directory and PyTorch's pickle format.
#
#   python3 convert_hf.py <directory with config.json, pytorch_model.bin | model.safetensors | shards with an index,
#                          tokenizer.json | tokenizer.model | spiece.model> <out> [float32|float16|int8] [max seq_len]
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "python"))
from llama2_convert import (Arrays, Safetensors, Shards, bfloat16, checkpoint_form, checkpoint_header, checkpoint_size,  # noqa: E402
                            convert_weights, family_of, normalize)
# (parts, read as the parts read one another: the window does not hand them out)
from convert.conduct import TOKENIZERS  # noqa: E402
from convert.tokenizer import Later, Only  # noqa: E402
from convert.vocabulary import vocabulary_of  # noqa: E402


# ------------------------------------------------------------------------------------------------ weights
def load_torch_pickle(path):
    """Read a PyTorch zip checkpoint without PyTorch. Only tensor-rebuilding globals are allowed to unpickle."""
    archive = zipfile.ZipFile(path)
    prefix = next(name for name in archive.namelist() if name.endswith("/data.pkl"))[:-len("data.pkl")]
    readers = {"FloatStorage": lambda raw: np.frombuffer(raw, dtype=np.float32),
               "HalfStorage": lambda raw: np.frombuffer(raw, dtype=np.float16), "BFloat16Storage": bfloat16}

    def rebuild_tensor(storage, storage_offset, size, stride, *unused):
        array = storage[storage_offset:storage_offset + int(np.prod(size))].reshape(size)
        assert tuple(stride) == tuple(s // array.itemsize for s in array.strides), "non-contiguous tensor"
        return array

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("collections", "OrderedDict"):
                return dict
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild_tensor
            if module == "torch" and name in readers:
                return readers[name]
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

        def persistent_load(self, pid):
            _, reader, key, _, _ = pid
            return reader(archive.read(f"{prefix}data/{key}"))

    return Unpickler(archive.open(f"{prefix}data.pkl")).load()


def convert(directory, out_path, dtype, max_seq_len):
    config = json.loads((directory / "config.json").read_text())
    safetensors = directory / "model.safetensors"
    index = directory / "model.safetensors.index.json"
    if safetensors.exists():
        # read piece by piece, never as a whole
        data = np.memmap(safetensors, dtype=np.uint8, mode="r")
        source = Safetensors(lambda offset, length: data[offset:offset + length])
    elif index.exists():
        # split over several files (T105): the index says which; each is read like the one file above
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        maps = [np.memmap(directory / name, dtype=np.uint8, mode="r") for name in files]
        source = Shards([Safetensors(lambda offset, length, data=data: data[offset:offset + length]) for data in maps])
    else:
        source = Arrays(load_torch_pickle(directory / "pytorch_model.bin"))
    # a GPT-2 or a GPT-NeoX has more tensors than a Llama of the same header, a Qwen3 the norms of q and k and maybe
    # heads of another size (T124): the size needs what the header cannot say
    size = checkpoint_size(checkpoint_header(normalize(config), source, max_seq_len), dtype, checkpoint_form(config, source))
    out = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(size,))
    convert_weights(source, config, dtype, max_seq_len, out)
    out.flush()
    return normalize(config)["vocab_size"]  # a Qwen3.5 has it one level down (T229)


def tokenizer_of(name, data, vocab_size, last=True):
    """tokenizer.bin of one candidate's bytes, by its name: what the page's conversion makes of it (T640.7: the one
    reader, convert/vocabulary.py's VOCABULARIES, by convert.vocabulary's vocabulary_of()). The bytes do not hang on
    tokenizer_config.json, which only the options read. last: see Conversion (T257's Later)."""
    return vocabulary_of(data, name, vocab_size, last=last).tokenizer


def convert_tokenizer(directory, out_path, vocab_size):
    """The tokenizer of the directory, as the page's conduct picks it (convert.conduct's tokenized()): of the candidates
    in its order (T374.2.3: convert.conduct's TOKENIZERS, the one list), the first that is there and that the converter
    reads, and a tokenizer.json that writes out a sentencepiece model with its dummy prefix only where no candidate
    after it will do (T257). Where none will do, the refusal of one that is there says why; where a later one does, a
    line on stderr says which was left and why (T414)."""
    refusal, left, later = None, [], None
    for name in TOKENIZERS:
        if not (directory / name).exists():
            continue
        data = (directory / name).read_bytes()
        try:
            vocabulary = tokenizer_of(name, data, vocab_size, last=False)
            break
        except Later as error:
            later = later or (name, data, error)
        except Only:  # (T659: no other is read in its place)
            raise
        except Exception as error:  # (whatever it was refused with: the next candidate may do)
            refusal = refusal or error
            left.append((name, error))
    else:
        if not later:
            raise refusal or FileNotFoundError(f"{directory} has no tokenizer: none of {', '.join(TOKENIZERS)}")
        name, data, _ = later
        vocabulary = tokenizer_of(name, data, vocab_size)
    if later and later[0] != name:
        left.append(later[::2])
    for skipped, error in left:
        print(f"{directory / skipped}: not read ({type(error).__name__}: {' '.join(str(error).split())}), so the tokenizer is {name}", file=sys.stderr)
    Path(out_path).write_bytes(vocabulary)


def options_note(directory):
    """T255: the file of a SmolLM3 is a Llama's, and only the options (src/models.ts) say which layers RoPE leaves alone: a
    model listed without them runs as a Llama, writes worse text and throws nothing. The line to say it, or None."""
    config = normalize(json.loads((directory / "config.json").read_text()))
    left = family_of(config).options(config).get("unturned", [])
    return f"options: unturned {left} (to be given to the engine with the file: the file does not say it)" if left else None


if __name__ == "__main__":
    directory, out = Path(sys.argv[1]), sys.argv[2]
    dtype = np.dtype(sys.argv[3] if len(sys.argv) > 3 else "float32")
    max_seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    vocab_size = convert(directory, f"{out}.bin", dtype, max_seq_len)
    convert_tokenizer(directory, f"{out}.tokenizer.bin", vocab_size)
    print(f"{out}.bin: {Path(f'{out}.bin').stat().st_size:,} bytes ({dtype}), vocabulary {vocab_size}")
    note = options_note(directory)
    if note:
        print(note)
