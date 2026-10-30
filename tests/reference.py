# reference.py
# T.357.4: what the tools that hold the engine to transformers itself share (tests/reference_llama.py,
# tests/reference_qwen35.py, tests/reference_lfm2.py: they run in CI, where torch is). What is the same in all of
# them is here once: how far two sets of logits are apart and what line the engine is held to (ten times what
# transformers' own two computations differ by, and no less than a least the tool says), the states of a model whose
# layers are of several kinds (States), the long text, a made-up model converted the way the page converts, a real one
# converted into a file, and the lines that say each. What a tool keeps: transformers' classes, its made-up models, what
# its layers hold and how that is read from the engine and from transformers' cache, its faults, its list of models.
#
# Only both_ways() and kept_logits() need torch (they run transformers' model); everything else runs wherever NumPy is
# (tests/test_reference.py), and tests/reference_dry.py runs the three tools from end to end with stand-ins.
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from cpu import cpu_name
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from llama2_convert import Conversion  # noqa: E402
from llama2_numpy import Llama  # noqa: E402
from conducting import Mapped, converted  # noqa: E402
from fetching import download  # noqa: E402
from made_up import safetensors_file, unigram  # noqa: E402

# a text of more than 64 tokens (transformers' chunk of a Qwen3.5's delta rule), English and Japanese
TEXT = ("Mount Fuji is the highest mountain in Japan, standing 3,776 metres above sea level on the island of Honshu. "
        "It is an active stratovolcano that last erupted from 1707 to 1708, and on clear days it can be seen from "
        "Tokyo, about 100 kilometres to the north-east. 富士山は日本で最も高い山で、古くから信仰の対象とされ、"
        "多くの絵画や文学作品に描かれてきました。夏には多くの登山者が山頂を目指します。")
LONG = 1024  # the positions of the long text (long_ids)


def option(name, default=None):
    """--name=value of the command line"""
    return next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith(f"--{name}=")), default)


def fetch(name, directory, repo, revision, optional=False):
    """A file of a repository at a revision, into the directory (tests/fetching.py: a download that stopped short is
    asked for again). optional: None where the repository has no such file."""
    return download(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", directory / name, optional=optional)


# ---------------------------------------------------------------------------------------------- logits and lines
def differences(ours, theirs):
    """(largest difference, mean difference, positions whose most likely token is the same, the largest gap between
    the two most likely tokens of theirs where the most likely token differs)"""
    ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    gap = np.abs(ours - theirs)
    same = ours.argmax(axis=1) == theirs.argmax(axis=1)
    top = np.sort(theirs, axis=1)[:, -2:]
    margins = (top[:, 1] - top[:, 0])[~same]
    return float(gap.max()), float(gap.mean()), int(same.sum()), float(margins.max()) if margins.size else 0.0


def relative(a, b):
    """how far a is from b, in b's own size: the norm of the difference over the norm of b"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def line_of(floor, least):
    """The line the engine is held to: ten times what transformers' own two computations (at once, token by token)
    differ by, which changes with the CPU, and no less than least."""
    return max(least, 10 * floor)


def within(largest, same, count, margin, line):
    """Logits that far from the reference's are within the line: no further than it, and the most likely token the same
    at every one of count positions, or wherever the reference's best two are further apart than twice the difference.
    A NaN is past the line (largest <= line is false)."""
    return largest <= line and (same == count or margin <= 2 * largest)


def engine_logits(llama, ids):
    """the engine's logits of these ids, token by token from position 0"""
    return [llama.forward(token, pos).copy() for pos, token in enumerate(ids)]


def logits_failed(say, what, ours, theirs, line, gap=False):
    """Says how far ours are from theirs and whether that is within the line; True where it is not. gap: and the
    largest gap between the reference's best two where the most likely token is another."""
    largest, mean, same, margin = differences(ours, theirs)
    ok = within(largest, same, len(ours), margin, line)
    said = f" (the largest gap between the reference's best two where it is not: {margin:.2e})" if gap else ""
    say(f"{what}: largest difference {largest:.2e}, mean {mean:.2e}, the same most likely token at {same} of {len(ours)} "
        f"positions{said}{'' if ok else f' — FAILED (the line: {line:.1e})'}")
    return not ok


def logits_said(say, label, whole, ours):
    for what, logits in (("transformers", whole), ("the engine", ours)):
        logits = np.asarray(logits)
        say(f"{label}: logits of {what}: min {logits.min():.3f}, max {logits.max():.3f}, mean {logits.mean():.4f}, "
            f"std {logits.std():.4f}")


def options_shown(options):
    """the options of a conversion for one line of the log: how many specials, not which"""
    return json.dumps({key: (value if key != "specials" else f"{len(value)} of them") for key, value in options.items()})


def ended(say, failed):
    # (T.494: an only_extra run says nothing of its runner but this)
    say(f"the CPU: {cpu_name()}")
    say("FAILED" if failed else "the engine computes what transformers computes")
    sys.exit(1 if failed else 0)


# ------------------------------------------------------------------------------- the states of layers of several kinds
class States:
    """What the engine and transformers hold after the same positions, which a fault of the engine shows in long before
    the logits do, for a model whose layers are of several kinds (a Qwen3.5's, an LFM2's).
    kinds: {the kind of a layer: the names of what such a layer holds}.
    of_engine(llama, count) and of_model(transformers' cache, count): {layer: (kind, [an array for each name])}, in
    transformers' order (the tool's: what a layer holds and how either side keeps it is the model's).
    least: the line of a state is ten times what transformers' own two computations differ by, and no less."""

    def __init__(self, kinds, of_engine, of_model, least=1e-4):
        self.kinds, self.of_engine, self.of_model, self.least = kinds, of_engine, of_model, least
        self.names = tuple(name for names in kinds.values() for name in names)

    def distances(self, ours, theirs):
        """The largest relative distance over the layers, for each of the names. A layer of another kind is infinitely
        far, and so is a state that is not a number (max() would let a NaN by)."""
        worst = dict.fromkeys(self.names, 0.0)
        for l, (kind, mine) in ours.items():
            if theirs[l][0] != kind:
                worst.update(dict.fromkeys(self.kinds[kind], float("inf")))
                continue
            for name, a, b in zip(self.kinds[kind], mine, theirs[l][1]):
                far = relative(a, b)
                worst[name] = max(worst[name], float("inf") if far != far else far)
        return worst

    def apart(self, llama, count, theirs):
        """the engine's states after count positions, from transformers'"""
        return self.distances(self.of_engine(llama, count), theirs)

    def lines(self, floor):
        return {name: line_of(floor[name], self.least) for name in self.names}

    def past(self, apart, lines):
        """the names of the states that are not within their line"""
        return [name for name in self.names if not apart[name] <= lines[name]]

    def failed(self, say, label, llama, count, at_once, stepped):
        """Says the engine's states against transformers' after count positions, each within the line of what
        transformers' own two computations (at once, token by token) differ by. True where one is not."""
        apart, floor = self.apart(llama, count, stepped), self.distances(at_once, stepped)
        lines = self.lines(floor)
        bad = self.past(apart, lines)
        say(f"{label}: the states after {count} positions against transformers': " + "; ".join(
            f"{name} {apart[name]:.1e} (transformers' own {floor[name]:.1e}, line {lines[name]:.1e}){' — FAILED' if name in bad else ''}"
            for name in self.names))
        return bool(bad)


def both_ways(model, ids, states_of):
    """transformers on ids: (logits at once, its states, logits token by token, its states)"""
    import torch

    tensor = torch.tensor([ids])
    with torch.no_grad():
        out = model(input_ids=tensor, use_cache=True)
        whole, at_once = out.logits[0].float().numpy(), states_of(out.past_key_values, len(ids))
        past, stepped = None, []
        for at in range(len(ids)):
            out = model(input_ids=tensor[:, at:at + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            stepped.append(out.logits[0, -1].float().numpy())
    return whole, at_once, np.stack(stepped), states_of(past, len(ids))


# ------------------------------------------------------------------------------------------------- the long text
# Past the positions of the main comparison: LONG positions of a text through transformers at once and through the
# engine, the logits at some of them (the keys and values of the attention layers double at 256 and 512, a state has had
# a thousand tokens to drift in) and the states at the end.
def long_ids(tokenizer, bos, at):
    """(the ids of the long text after bos, the positions of `at` it has)"""
    text = " ".join(f"Section {i}. {TEXT}" for i in range(12))
    ids = ([bos] + tokenizer.encode(text, add_special_tokens=False).ids)[:LONG]
    return ids, [position for position in at if position < len(ids)]


def kept_logits(model, ids, at, states_of):
    """transformers on ids at once: (the logits at the positions `at`, its states after them)"""
    import torch

    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=torch.tensor(at))
    return out.logits[0].float().numpy(), states_of(out.past_key_values, len(ids))


def long_failed(say, states, llama, ids, at, theirs, their_states, line):
    """The engine through ids, its logits at `at` against theirs (within line) and its states at the end against
    their_states (within the least of States: no floor is measured here). True where either is not."""
    began, kept = time.perf_counter(), {}
    for pos, token in enumerate(ids):
        logits = llama.forward(token, pos)
        if pos in at:
            kept[pos] = logits.copy()
    largest, mean, same, margin = differences(np.stack([kept[position] for position in at]), theirs)
    apart = states.apart(llama, len(ids), their_states)
    ok = within(largest, same, len(at), margin, line)
    bad = states.past(apart, dict.fromkeys(states.names, states.least))
    say(f"long: the engine ran {len(ids)} positions in {time.perf_counter() - began:.1f} s; the logits at {len(at)} of them against "
        f"transformers': largest difference {largest:.2e}, mean {mean:.2e}, the same most likely token at {same} of {len(at)}"
        f"{'' if ok else f' — FAILED (the line: {line:.1e})'}; the states after them: "
        + "; ".join(f"{name} {apart[name]:.1e}" for name in states.names)
        + (f" — FAILED (the line: {states.least:.0e}): {', '.join(bad)}" if bad else ""))
    return not ok or bool(bad)


# --------------------------------------------------------------------------------------------------- conversions
def made_up_engine(tensors, published, vocabulary=320):
    """A made-up model (transformers' own tensors and config.json) converted the way the page converts, float32, and the
    engine on it: (the engine, the options it was made with, the checkpoint). The vocabulary is nobody's: w0, w1, ..."""
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), unigram(vocabulary), "tokenizer.json",
                            dtype="float32", max_seq_len=256, start=8 + size)
    conversion.feed(file[8 + size:])
    conversion.finish()
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    checkpoint = bytes(conversion.checkpoint)
    return Llama(checkpoint, conversion.tokenizer, **options), options, checkpoint


def float32_file(answerer, hf, path):
    """A real model converted the way the page converts (T.374.4: by the conduct of a conversion, answered by `answerer`),
    float32, straight into the file at path: (the conversion, the seconds it took). The file is the caller's to unlink."""
    sink = Mapped(path)
    began = time.perf_counter()
    conversion = converted(answerer, hf, dtype="float32", sink=sink)
    sink.data.flush()
    del sink.data
    return conversion, time.perf_counter() - began
