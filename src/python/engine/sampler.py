# Drawing a token from the logits: the settings of it as one value (Sampling), and the two that draw by them, in NumPy
# and on the kernels: the repetition and presence penalties, DRY's (T668), greedy, and temperature with top-k, top-p
# and min-p.
#
# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository), and it is
# derived from two works under the MIT License, whose notice follows: tairov/llama2.py
# (https://github.com/tairov/llama2.py; its LICENSE names no copyright holder) and karpathy/llama2.c
# (https://github.com/karpathy/llama2.c), Copyright (c) 2023 Andrej.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# T668: DRY's penalty (dry_penalties()) takes its form (the four steps, the sequence breakers' limit, the Z-algorithm
# run backwards, the cap of the exponent) from llama.cpp's llama_sampler_dry_apply() (src/llama-sampler.cpp,
# https://github.com/ggml-org/llama.cpp, commit b1b770d9, 2026-11-21), under the MIT License, whose notice follows:
#
# Copyright (c) 2023-2026 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
from dataclasses import dataclass, fields

import numpy as np

REPETITION_WINDOW = 64  # the repetition penalty looks at this many of the latest tokens
# T195: what a draw says, on the kernels and in NumPy, when the largest logit is NaN or an infinity (a NaN anywhere,
# +inf anywhere, or all -inf): a broken model or an overflow. No token is drawn from logits like these
NOT_FINITE = "The model computed logits that are not finite numbers (NaN or infinity), so no token can be drawn: its weights are broken or its numbers overflowed."
# what greedy leaves no say: it takes the largest logit, which a nucleus, a top-k and a min-p all leave in
NARROWING = ("topp", "top_k", "min_p")
# T668: DRY's settings, which say nothing while it is off (Sampling.dry)
DRY = ("dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n", "dry_sequence_breakers")
DRY_REFUSAL = ("dry_multiplier is 0 or more, dry_base 1 or more, dry_allowed_length a whole number 0 or more, "
               "dry_penalty_last_n a whole number -1 (all the tokens written) or more, and dry_sequence_breakers "
               "a list of single ASCII characters.")
# llama.cpp's FLOAT_MAX_LOG (a float32), and the base past which DRY's exponent is capped by it
F32_MAX_LOG = float(np.float32(88.7228391))
CAPPED_BASE = float(np.float32(1.000001))


@dataclass(frozen=True)
class Sampling:
    """How a token is drawn from the logits: the settings of one generation, checked here and nowhere else. The names
    are the page's (generate()'s keywords, the URL's). A sampler reads them once, when a generation starts
    (drawing()), and not at every token.

    A new rule is a setting here (with the value that leaves it out as its default: the GPU's steps are judged by that,
    see beyond()) and a stage in each sampler's drawing()."""
    temperature: float = 0.0  # 0: greedy
    topp: float = 0.9  # the nucleus (0 or 1: none)
    repetition_penalty: float = 1.0  # on every token of the last REPETITION_WINDOW, the prompt's too (1: none)
    top_k: int = 0  # T274: of the top_k most probable (0: none)
    min_p: float = 0.0  # T274: without what is less than min_p times as probable as the most probable (0: none)
    # T274: taken off the logit of each of the last REPETITION_WINDOW sampled tokens, the same however often it came
    # (0: none). Not the prompt's: with those in it, the format's own stop token would lose it too (as OpenAI's)
    presence_penalty: float = 0.0
    # T668: DRY ("don't repeat yourself", llama.cpp's, by its names): a token that would carry on a run of the tokens
    # written which has come before, of dry_allowed_length tokens or more, loses
    # dry_multiplier * dry_base ** (the run - dry_allowed_length) off its logit (a multiplier of 0: none). It looks at
    # the latest dry_penalty_last_n tokens written (the sampled ones, not the prompt's: -1 all of them, 0 none), and a
    # run never reaches back past a sequence breaker, a token whose text holds one of these characters (which is not
    # penalized itself either). Single ASCII characters only, each a breaker in the tokens that hold it (llama.cpp
    # also follows a breaker of several characters over the tokens it is split into: none such is taken here)
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = 64
    dry_sequence_breakers: tuple = ("\n", ":", "\"", "*")

    def __post_init__(self):
        object.__setattr__(self, "top_k", int(self.top_k))
        # (T395: every one of them, and written so that a NaN fails each: a comparison with a NaN is false)
        if not (self.temperature >= 0.0 and math.isfinite(self.temperature) and 0.0 <= self.topp <= 1.0
                and self.repetition_penalty > 0.0 and math.isfinite(self.repetition_penalty) and self.top_k >= 0
                and 0.0 <= self.min_p <= 1.0 and self.presence_penalty >= 0.0 and math.isfinite(self.presence_penalty)):
            raise ValueError("temperature, top_k and presence_penalty are 0 or more, repetition_penalty is more than 0, "
                             "and topp and min_p are from 0 to 1.")
        # T668 (a whole number may come as a NumPy one, not as a bool or a float; the breakers as a list)
        breakers, whole = self.dry_sequence_breakers, (int, np.integer)
        if not (self.dry_multiplier >= 0.0 and math.isfinite(self.dry_multiplier) and self.dry_base >= 1.0
                and math.isfinite(self.dry_base) and isinstance(self.dry_allowed_length, whole)
                and not isinstance(self.dry_allowed_length, bool) and self.dry_allowed_length >= 0
                and isinstance(self.dry_penalty_last_n, whole) and not isinstance(self.dry_penalty_last_n, bool)
                and self.dry_penalty_last_n >= -1 and not isinstance(breakers, (str, bytes))
                and all(isinstance(breaker, str) and len(breaker) == 1 and breaker.isascii() for breaker in breakers)):
            raise ValueError(DRY_REFUSAL)
        object.__setattr__(self, "dry_allowed_length", int(self.dry_allowed_length))
        object.__setattr__(self, "dry_penalty_last_n", int(self.dry_penalty_last_n))
        object.__setattr__(self, "dry_sequence_breakers", tuple(breakers))
        # T668: whether DRY's penalty is on (a multiplier, and tokens to look at). No setting, and no property: an
        # attribute, which generate() reads with no call (T359: generate()'s frame calls the parts and no more)
        object.__setattr__(self, "dry", self.dry_multiplier != 0.0 and self.dry_penalty_last_n != 0)

    def beyond(self, known):
        """The settings a sampler that knows only those named would have to know as well, to draw as these say: the
        ones that are not at their defaults (greedy: but for NARROWING). The GPU's sampler asks (engine/external.py),
        and a setting it was never told of keeps the steps from it."""
        idle = (NARROWING if self.temperature == 0.0 else ()) + (() if self.dry else DRY)
        return [field.name for field in fields(self)
                if field.name not in known and field.name not in idle and getattr(self, field.name) != field.default]


def dry_cap(base):
    """T668: the largest exponent of DRY's penalty for a float32 base (0: none), llama.cpp's max_exponent: the
    logarithm of the largest float32 over the base's, in float32 (llama.cpp's logf() may differ from the float32 of
    the float64 logarithm in the last bit: only a quotient on a whole number would tell), where the base is more than
    1.000001. The kernel is handed it (a WebAssembly side module has no logarithm without a data segment)."""
    return int(np.float32(F32_MAX_LOG) / np.float32(math.log(base))) if base > CAPPED_BASE else 0


def dry_penalties(window, breaking, multiplier, base, allowed):
    """T668: DRY's penalty of each token it penalizes, {token: the float32 taken off its logit}, after the tokens of
    window (the latest written, the oldest first). breaking: the tokens that are sequence breakers. multiplier and
    base are float32 (as the kernel takes them), allowed a whole number.

    llama.cpp's llama_sampler_dry_apply()'s steps (the notice at the head of this file), and kernels/kernel/sample.ts's
    dry() and shaders/likecpu.ts's dryLikeCpu() take the same: (1) a run reaches back no further than the latest
    breaker; (2) for each shift k, how long a run of the tokens up to the k-th latest is also the end of the window
    (the Z-algorithm on the window backwards), at most (1)'s; (3) the token after each run of allowed tokens or more
    would carry it on, and the longest of its runs counts; (4) a token that is no breaker loses
    multiplier * base ** (that run - allowed), the exponent at most FLOAT_MAX_LOG / log(base) (in float32) where the
    base is more than 1.000001. The power is multiplied out (the base times itself, in float64) where llama.cpp calls
    pow(): the three of this page give the same float32 that way, and llama.cpp's differs from it only in the last
    bits of a float64 (in a float32 only where it falls on a border)."""
    count = len(window)
    if count <= allowed:
        return {}
    back = window[::-1]  # back[j]: the j-th latest
    # (1) the latest breaker limits a run
    limit = count
    for j in range(count):
        if back[j] in breaking:
            limit = j
            break
    if limit < allowed:
        return {}
    # (2) repeats[k]: how long a run up to the k-th latest is also the end of the window (at most limit)
    repeats = [0] * count
    left = right = 0
    for k in range(1, count):
        if k > right:
            n = 0
            while n + k < count and back[n] == back[n + k]:
                n += 1
            if n > 0:
                left, right = k, k + n - 1
        elif repeats[k - left] < right - k + 1:
            n = repeats[k - left]
        else:
            i = right + 1
            while i < count and back[i] == back[i - k]:
                i += 1
            n = i - k
            left, right = k, i - 1
        repeats[k] = min(n, limit)
    # (3) the longest run each token would carry on
    longest = {}
    for k in range(1, count):
        if repeats[k] >= allowed:
            token = back[k - 1]
            if longest.get(token, -1) < repeats[k]:
                longest[token] = repeats[k]
    # (4) the penalties
    most = dry_cap(base)
    penalties = {}
    for token, run in longest.items():
        if token in breaking:
            continue
        exponent = run - allowed
        if most > 0 and exponent > most:
            exponent = most
        power = 1.0
        for _ in range(exponent):
            power *= base
        penalties[token] = np.float32(multiplier * power)
    return penalties


def dry_window(written, last):
    """T668: the tokens written that DRY looks at: the latest last of them (-1: all)"""
    return written if last < 0 else written[max(len(written) - last, 0):]


def greedy(logits):
    """Greedy argmax sampling: take the token with the highest probability. NumPy's argmax takes a NaN for the
    largest, so the logit it picks is finite exactly when the largest one is (T195)."""
    token = int(np.argmax(logits))
    if not math.isfinite(logits[token]):
        raise ValueError(NOT_FINITE)
    return token


class NumpySampler:
    """Draws in NumPy: the reference, and what draws where there are no kernels (or without them, T52's "sampler").
    It keeps nothing from step to step."""

    def drawing(self, sampling, rng, breaking=frozenset()):
        """draw(logits, history, written) for one generation: the token of a step, by sampling (a Sampling) and the
        random numbers of rng. history is every token so far and written the sampled ones of them, both growing by
        the token drawn; the penalties are written into logits. breaking (T668): the tokens that are DRY's sequence
        breakers (the tokenizer's containing() of the setting's characters, which generate() hands where DRY is on).

        The token is one of softmax(logits / temperature): of its top_k most probable where top_k > 0, then of their
        nucleus (0 < topp < 1: their probabilities add up to one again), then without what is less than min_p times
        as probable as the most probable: the order of llama.cpp's samplers and of transformers'."""
        temperature, topp, top_k, min_p = sampling.temperature, sampling.topp, sampling.top_k, sampling.min_p
        penalty, presence = sampling.repetition_penalty, sampling.presence_penalty
        nucleus, less = 0.0 < topp < 1.0, np.float32(presence)
        # (T668: DRY's multiplier and base as float32, as the kernel takes them; each setting read once, as Sampling.dry)
        said, base = sampling.dry_multiplier, float(np.float32(sampling.dry_base))
        last, allowed, breaking = sampling.dry_penalty_last_n, sampling.dry_allowed_length, frozenset(breaking)
        dry, multiplier = said != 0.0 and last != 0, float(np.float32(said))

        def draw(logits, history, written):
            if penalty != 1.0:
                # Make the tokens of the last steps less likely: tiny models love to loop
                recent = np.unique(history[-REPETITION_WINDOW:])
                logits[recent] = np.where(logits[recent] > 0, logits[recent] / penalty, logits[recent] * penalty)
            if presence != 0.0 and written:
                logits[np.unique(written[-REPETITION_WINDOW:])] -= less
            if dry:
                # T668: after the two penalties, as llama.cpp's chain has it
                penalties = dry_penalties(dry_window(written, last), breaking, multiplier, base, allowed)
                if penalties:
                    tokens = np.fromiter(penalties, dtype=np.int64, count=len(penalties))
                    logits[tokens] -= np.fromiter(penalties.values(), dtype=np.float32, count=len(penalties))
            if temperature == 0.0:
                return greedy(logits)
            # max() keeps a NaN (T195)
            best = logits.max()
            if not math.isfinite(best):
                raise ValueError(NOT_FINITE)
            if nucleus:
                # exp() over a vocabulary of 50000 or 100000 tokens costs as much as half a forward pass, and nearly
                # all of it goes to tokens that cannot be drawn: leave out, while still cheap, whatever is less than a
                # ten millionth as probable as the best token (together far below one percent of the probability mass)
                candidates = np.flatnonzero(logits >= best + temperature * math.log(1e-7))
                probabilities = np.exp((logits[candidates] - best) / temperature).astype(np.float64)
            else:
                candidates = np.arange(logits.size)
                probabilities = np.exp((logits - best) / temperature).astype(np.float64)
            probabilities /= probabilities.sum()
            least = min(min_p, 1.0) * probabilities.max() if min_p > 0.0 else 0.0  # (the same share after a top-k)
            narrowed = 0 < top_k < probabilities.size
            if narrowed:
                # walked from the most probable then, as a nucleus is
                likely = np.argsort(-probabilities, kind="stable")[:top_k]
                candidates, probabilities = candidates[likely], probabilities[likely]
                probabilities /= probabilities.sum()
                least = min(min_p, 1.0) * probabilities[0] if min_p > 0.0 else 0.0
            if nucleus:
                # Top-p (nucleus) sampling: only the most probable tokens whose probabilities add up to topp.
                # Tokens below (1 - topp) / (n - 1) cannot be part of that set (llama2.c), so they need not be sorted.
                # That holds while one token at least stays: when all are below it (n * topp < 1, which the floor
                # above makes possible, T178), the others add up to less than (1 - topp), so the set is the most
                # probable token alone. It always stays
                cutoff = min((1.0 - topp) / max(probabilities.size - 1, 1), probabilities.max())
                likely = np.flatnonzero(probabilities >= cutoff)
                # (T561: equal ones by the lower token, as the top-k above, the kernel's and likecpu.ts's)
                likely = likely[np.argsort(-probabilities[likely], kind="stable")]
                candidates, probabilities = candidates[likely], probabilities[likely]
                cumulative = np.cumsum(probabilities)
                cumulative = cumulative[:np.searchsorted(cumulative, topp) + 1]
                if least:
                    cumulative = cumulative[:max(int(np.count_nonzero(probabilities[:cumulative.size] >= least)), 1)]
            else:
                if least:
                    stay = np.flatnonzero(probabilities >= least)
                    candidates, probabilities = candidates[stay], probabilities[stay]
                cumulative = np.cumsum(probabilities)
            # one random number on the cumulative distribution; Generator.choice() would cost a third of a millisecond
            chosen = np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right")
            return int(candidates[min(chosen, cumulative.size - 1)])

        return draw


class KernelSampler:
    """Draws on the kernels (simdkernel.so's): what NumpySampler draws, from the same random numbers.

    Sorting the candidates of the nucleus was most of the time of a step for a small model with a large
    vocabulary, and a repetition penalty, which flattens the distribution, made it worse.
    """

    def __init__(self, kernels, vocab_size):
        probabilities, order = np.empty(vocab_size, dtype=np.float32), np.empty(vocab_size, dtype=np.int32)
        probabilities_p, order_p = probabilities.ctypes.data, order.ctypes.data
        kernel_penalize, kernel_sample = kernels["penalize"], kernels["sample"]
        known = {}  # id(logits) -> (logits, address): forward() returns the same array every time

        def address(logits):
            entry = known.get(id(logits))
            if entry is None or entry[0] is not logits:
                if logits.dtype != np.float32 or not logits.flags.c_contiguous:
                    raise TypeError("the kernels need contiguous float32 logits")
                known.clear()
                entry = known[id(logits)] = (logits, logits.ctypes.data)
            return entry[1]

        def ring():
            """A ring of the latest tokens of one list, as the penalty kernel reads them: its address, and latest(tokens),
            which makes it the latest of tokens and says how many of them it holds. generate() appends one token per
            step, and then one number is written: copying 64 of them from a list costs more than the kernel takes.
            (T394: a ring for each list a penalty reads, the history and the written tokens: one ring that both
            penalties took turns at was copied whole twice a step.)"""
            recent = np.empty(REPETITION_WINDOW, dtype=np.int32)
            seen = [None, 0]  # the list that recent[] mirrors, and its length then

            def latest(tokens):
                size = len(tokens)
                if seen[0] is tokens and size == seen[1] + 1:
                    recent[(size - 1) % REPETITION_WINDOW] = tokens[-1]
                else:
                    for position in range(max(size - REPETITION_WINDOW, 0), size):
                        recent[position % REPETITION_WINDOW] = tokens[position]
                seen[0], seen[1] = tokens, size
                return min(size, REPETITION_WINDOW)
            return recent, recent.ctypes.data, latest

        (history_ring, history_p, history_latest), (written_ring, written_p, written_latest) = ring(), ring()
        self.buffers = (probabilities, order, history_ring, written_ring)  # keep them alive: the kernels only know addresses
        # T668: DRY's buffers, made by the first generation that asks for it and kept: the tokens written (as many as
        # there are; grown twice as long when full: generate() appends one a step, and then one number is written),
        # the kernel's lengths of a run (as many), its longest run of each token (the vocabulary's, all 0 between two
        # calls), and a byte for each token of the vocabulary for each set of sequence breakers asked for (1: a
        # breaker; a set for a tokenizer, so a few). window() holds them: a draw that reads it keeps them alive
        dry = {"tokens": np.empty(0, dtype=np.int32), "repeats": np.empty(0, dtype=np.int32), "seen": None, "size": 0}

        def window(tokens, last):
            """(the address of the latest last of tokens (-1: all) in the buffer that holds them all, how many, the
            address of the lengths of a run): ring()'s latest(), of all the tokens"""
            size = len(tokens)
            if size > dry["tokens"].size:
                dry["tokens"] = np.empty(max(256, 2 * size), dtype=np.int32)
                dry["repeats"] = np.empty(dry["tokens"].size, dtype=np.int32)
                dry["seen"] = None
            if dry["seen"] is tokens and size == dry["size"] + 1:
                dry["tokens"][size - 1] = tokens[-1]
            elif size:
                dry["tokens"][:size] = tokens
            dry["seen"], dry["size"] = tokens, size
            count = size if last < 0 else min(size, last)
            return dry["tokens"].ctypes.data + 4 * (size - count), count, dry["repeats"].ctypes.data

        def dry_addresses(breaking):
            """(the address of the longest runs, the address of the breakers' bytes) for a generation"""
            if "longest" not in dry:
                dry["longest"] = np.zeros(vocab_size, dtype=np.int32)
            if breaking not in dry:
                dry[breaking] = np.zeros(vocab_size, dtype=np.uint8)
                dry[breaking][[token for token in breaking if 0 <= token < vocab_size]] = 1
            return dry["longest"].ctypes.data, dry[breaking].ctypes.data

        def drawing(sampling, rng, breaking=frozenset()):
            """NumpySampler.drawing()'s, on the kernels. A closure of a closure, and no method: a step reads the
            settings, the buffers, their addresses and the kernels as local names (it is on the path of every token
            the page writes)."""
            temperature, topp, top_k, min_p = sampling.temperature, sampling.topp, sampling.top_k, sampling.min_p
            penalty, presence = sampling.repetition_penalty, sampling.presence_penalty
            multiplier, base, last, allowed = sampling.dry_multiplier, sampling.dry_base, sampling.dry_penalty_last_n, sampling.dry_allowed_length
            drying = multiplier != 0.0 and last != 0  # (Sampling.dry)
            if drying:
                kernel_dry = kernels["dry"]  # (asked for where DRY is on: a stand-in of the two others needs none)
                longest_p, breakers_p = dry_addresses(frozenset(breaking))
                most = dry_cap(float(np.float32(base)))

            def draw(logits, history, written):
                if penalty != 1.0:
                    kernel_penalize(address(logits), history_p, history_latest(history), penalty, 0.0)
                if presence != 0.0 and written:
                    kernel_penalize(address(logits), written_p, written_latest(written), 1.0, presence)
                if drying:
                    tokens_p, count, repeats_p = window(written, last)
                    kernel_dry(address(logits), tokens_p, count, breakers_p, multiplier, base, allowed, most, repeats_p, longest_p)
                if temperature == 0.0:
                    return greedy(logits)
                # the random number is drawn here, so that a seed gives the same text again
                token = kernel_sample(address(logits), logits.size, temperature, topp, rng.random(), probabilities_p, order_p,
                                      top_k, min_p)
                if token < 0:
                    raise ValueError(NOT_FINITE)
                return token

            return draw

        self.drawing = drawing
