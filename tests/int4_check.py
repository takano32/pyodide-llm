# int4_check.py
# T98, stages 1 and 2: how much a model loses with its weights in 4, 5 or 6 bits, measured before anything of it goes
# into the page.
# Each way of storing the weights is applied to the original and undone again (the weights come back as float32
# with the error of the format in them), converted the way the page converts a model, and run by the NumPy engine
# in float32, so that the only difference between the rows is the weights' format. The GGUF files of llama.cpp are
# read the same way (tests/gguf_check.py widens them; its reader is kept apart from the page's).
#
#   python3 tests/int4_check.py <directory: config.json, model.safetensors> <out of tests/perplexity_prepare.py
#       (for the tokenizer and the options)> <text file> [tokens = 1500] [<name>=<file.gguf> ...]
#
# Under the table, what each row writes: greedy, 32 tokens on from the text's first 40 characters, for a person to
# read (a format can keep the perplexity and still write worse; T77's fourth point).
#
# INT4_ROWS=int5,int6 measures only the rows whose label begins with one of these (the original always, and the
# GGUFs given): a row is a whole run of the model, and stage 2 needed only its own.
#
# The table: the perplexity on the text (the context starts anew every 512 tokens, as tests/perplexity.py does),
# how far it is from the original, how often the most likely token is the original's, and the bits per weight of
# the matrices (the scales included). The embedding is a matrix here too: with shared embeddings it is also the
# classifier, which T85 found to be where int8 loses most.
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
from gguf_check import BYTES, hugging_face_name, read_gguf, tensor  # noqa: E402
from llama2_convert import (Safetensors, architecture, checkpoint_header, checkpoint_size, convert_weights,  # noqa: E402
                            has_bias, normalize, quantize)
from llama2_numpy import Llama, Tokenizer  # noqa: E402

WINDOW = 512


# ------------------------------------------------------------------------------------------- the formats
def symmetric(bits, group):
    """q = round(x / s), s = the largest |x| of the group over 2^(bits-1) - 1: zero stays zero."""
    top = 2 ** (bits - 1) - 1

    def apply(rows):
        groups = rows.reshape(-1, group)
        scales = np.abs(groups).max(axis=1, keepdims=True) / top
        scales = scales.astype(np.float16).astype(np.float32)  # stored in float16, as llama.cpp stores them
        q = np.clip(np.rint(np.divide(groups, scales, out=np.zeros_like(groups), where=scales > 0)), -top - 1, top)
        return (q * scales).reshape(rows.shape)
    return apply, bits + 16 / group


def asymmetric(bits, group):
    """q = round((x - min) / s), s = (max - min) / (2^bits - 1): the whole range of the group, a minimum each."""
    top = 2 ** bits - 1

    def apply(rows):
        groups = rows.reshape(-1, group)
        low, high = groups.min(axis=1, keepdims=True), groups.max(axis=1, keepdims=True)
        scales = ((high - low) / top).astype(np.float16).astype(np.float32)
        low = low.astype(np.float16).astype(np.float32)
        q = np.clip(np.rint(np.divide(groups - low, scales, out=np.zeros_like(groups), where=scales > 0)), 0, top)
        return (q * scales + low).reshape(rows.shape)
    return apply, bits + 32 / group


def int8_ours(rows):
    """llama2_convert.quantize(): what the page does today (groups of 32, float32 scales)."""
    values, scales = quantize(rows.reshape(-1, rows.shape[-1]))
    return (values.astype(np.float32) * scales[:, None]).reshape(rows.shape)


def is_embedding(name):
    """the token embedding, and the classifier (which may be the same table): Llama's, GPT-NeoX's and GPT-2's names"""
    return any(part in name for part in ("embed_tokens", "lm_head", "embed_in", "embed_out", "wte."))


def but_embedding(matrices, embedding):
    """One format for the layers' matrices and another for the embedding (and the classifier it may be)."""
    return lambda name, rows: (embedding if is_embedding(name) else matrices)(rows)


# ------------------------------------------------------------------------------------------- the sources
class Stored:
    """The original's tensors with a format applied to every matrix, row by row (a group never crosses a row)."""

    def __init__(self, original, apply):
        self.original, self.apply = original, apply

    def __contains__(self, name):
        return name in self.original

    def shape(self, name):
        return self.original.shape(name)

    def rows(self, name, start, stop):
        values = np.asarray(self.original.rows(name, start, stop), dtype=np.float32)
        return self.apply(name, values) if values.ndim == 2 else values


class FromGGUF:
    """The tensors of a GGUF widened to float32, with Hugging Face's names and order (q and k turned back)."""

    def __init__(self, path, config):
        _, self.metadata, infos, self.data, self.base = read_gguf(path)
        self.infos = {hugging_face_name(name): info for name, info in infos.items() if name != "rope_freqs.weight"}
        self.heads = {"q_proj": config["num_attention_heads"], "k_proj": config["num_key_value_heads"]}
        self.types = sorted({info["type"] for info in self.infos.values()})

    def __contains__(self, name):
        return name in self.infos

    def shape(self, name):
        return tuple(self.infos[name]["shape"])

    def rows(self, name, start, stop):
        info = self.infos[name]
        heads = next((h for part, h in self.heads.items() if f".{part}." in name), None)
        if heads is None:
            return tensor(info, self.data, self.base, start, stop)[0]
        whole = tensor(info, self.data, self.base)[0]
        rows = whole.shape[0] // heads  # llama.cpp's adjacent pairs back to the halves of each head
        whole = whole.reshape(heads, rows // 2, 2, *whole.shape[1:]).swapaxes(1, 2).reshape(whole.shape)
        return whole[start:stop]


# ------------------------------------------------------------------------------------------- the measurement
def evaluate(source, config, tokenizer, options, tokens, reference=None):
    """perplexity, and the most likely token at every position (and how often it is the reference's)."""
    normalized = normalize(config)  # GPT-2 and GPT-NeoX spell their configs their own way (as convert_hf.py does)
    header = checkpoint_header(normalized, source, WINDOW)
    out = bytearray(checkpoint_size(header, "float32", has_bias(source), architecture(normalized)))
    convert_weights(source, config, "float32", WINDOW, out)
    llama = Llama(out, tokenizer, kernels=None, **options)  # reads the bytearray where it is: no second copy
    total, count, likely = 0.0, 0, []
    for start in range(0, len(tokens), WINDOW - 1):
        piece = [llama.bos] + tokens[start:start + WINDOW - 1]
        for pos in range(len(piece) - 1):
            logits = np.asarray(llama.forward(piece[pos], pos), dtype=np.float64)
            likely.append(int(logits.argmax()))
            logits -= logits.max()
            total -= logits[piece[pos + 1]] - math.log(np.exp(logits).sum())
            count += 1
    agreement = None if reference is None else sum(a == b for a, b in zip(likely, reference)) / len(likely)
    return math.exp(total / count), likely, agreement, llama


def written(llama, prompt, steps=32):
    """What the model writes after the prompt, greedy: the text of the sample under the table."""
    pieces = llama.generate(prompt, steps=steps, temperature=0.0, echo=False)
    return "".join(piece if isinstance(piece, str) else piece.decode("utf-8", "replace") for piece in pieces)


def main():
    directory, prepared, text = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]).read_text()
    rest = sys.argv[4:]
    count = int(rest.pop(0)) if rest and "=" not in rest[0] else 1500
    ggufs = dict(argument.split("=", 1) for argument in rest)
    config = json.loads((directory / "config.json").read_text())
    options = json.loads(Path(f"{prepared}.json").read_text())
    options["dtype"] = "float32"
    tokenizer = Path(f"{prepared}.tokenizer.bin").read_bytes()
    data = np.memmap(directory / "model.safetensors", dtype=np.uint8, mode="r")
    original = Safetensors(lambda offset, length: data[offset:offset + length])
    # the options the converter wrote: a unigram vocabulary says nothing of NFC or of a pre-tokenizer
    vocabulary = Tokenizer(tokenizer, config["vocab_size"], kind=options.get("tokenizer_kind", "bpe"),
                           nfkc=options.get("nfkc", False), nfc=options.get("nfc", False),
                           pretokenizer=options.get("pretokenizer", "gpt2"))
    tokens = vocabulary.encode(text)[:count]
    matrices = {name: math.prod(original.shape(name)) for name in original.tensors if len(original.shape(name)) == 2}

    def bits_of(of):
        """bits per weight of the matrices, the scales included: of(name) for each matrix"""
        return sum(of(name) * n for name, n in matrices.items()) / sum(matrices.values())

    q4, q4_bits = symmetric(4, 32)
    q4_16, q4_16_bits = symmetric(4, 16)
    a4, a4_bits = asymmetric(4, 32)
    a4_16, a4_16_bits = asymmetric(4, 16)
    q5, q5_bits = symmetric(5, 32)
    q6, q6_bits = symmetric(6, 32)
    int8_bits = 8 + 32 / 32
    rows = [("original (as stored, widened to float32)", None, 16),
            ("int8, groups of 32 (the page today)", lambda name, r: int8_ours(r), int8_bits),
            ("int4, groups of 32, symmetric", lambda name, r: q4(r), q4_bits),
            ("int4, groups of 16, symmetric", lambda name, r: q4_16(r), q4_16_bits),
            ("int4, groups of 32, with a minimum", lambda name, r: a4(r), a4_bits),
            ("int4, groups of 16, with a minimum", lambda name, r: a4_16(r), a4_16_bits),
            ("int4, groups of 32, symmetric; embedding int8", but_embedding(q4, int8_ours),
             bits_of(lambda name: int8_bits if is_embedding(name) else q4_bits)),
            ("int4, groups of 32, with a minimum; embedding int8", but_embedding(a4, int8_ours),
             bits_of(lambda name: int8_bits if is_embedding(name) else a4_bits)),
            # stage 2 (Fable's decision): one uniform format of 5 or 6 bits, Q5_0's layout with a symmetric scale
            ("int5, groups of 32, symmetric", lambda name, r: q5(r), q5_bits),
            ("int5, groups of 32, symmetric; embedding int8", but_embedding(q5, int8_ours),
             bits_of(lambda name: int8_bits if is_embedding(name) else q5_bits)),
            ("int6, groups of 32, symmetric", lambda name, r: q6(r), q6_bits),
            ("int6, groups of 32, symmetric; embedding int8", but_embedding(q6, int8_ours),
             bits_of(lambda name: int8_bits if is_embedding(name) else q6_bits))]
    wanted = [prefix for prefix in os.environ.get("INT4_ROWS", "").split(",") if prefix]
    if wanted:
        rows = rows[:1] + [row for row in rows[1:] if row[0].startswith(tuple(wanted))]
    print(f"{directory.parent.name}, {len(tokens)} tokens of {Path(sys.argv[3]).name}\n")
    print("| weights | bits per weight | perplexity | against the original | most likely token the original's | seconds |")
    print("|---|---:|---:|---:|---:|---:|")
    reference = base = None
    prompt, samples = text[:40], []
    for label, apply, bits in rows + [(f"GGUF {name}", path, None) for name, path in ggufs.items()]:
        started = time.perf_counter()
        if isinstance(apply, str):
            source = FromGGUF(apply, config)
            label += f" ({', '.join(str(t) for t in source.types)})"
            bits = bits_of(lambda name: BYTES[source.infos[name]["type"]] * 8)
        else:
            source = original if apply is None else Stored(original, apply)
        value, likely, agreement, llama = evaluate(source, config, tokenizer, options, tokens, reference)
        samples.append((label, written(llama, prompt)))
        del llama
        if reference is None:
            reference, base = likely, value
        print(f"| {label} | {bits:.2f} | {value:.3f} | {(value / base - 1) * 100:+.2f}% | "
              f"{'' if agreement is None else f'{agreement * 100:.1f}%'} | {time.perf_counter() - started:.0f} |", flush=True)
    print(f"\nWhat each writes after {json.dumps(prompt, ensure_ascii=False)} (greedy, 32 tokens):\n")
    for label, sample in samples:
        print(f"- {label}: {json.dumps(sample, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
