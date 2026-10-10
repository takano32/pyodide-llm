# Drawing a token from the logits: the settings of it as one value (Sampling), and the two that draw by them, in NumPy
# and on the kernels: the repetition and presence penalties, greedy, and temperature with top-k, top-p and min-p.
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
import math
from dataclasses import dataclass, fields

import numpy as np

REPETITION_WINDOW = 64  # the repetition penalty looks at this many of the latest tokens
# T195: what a draw says, on the kernels and in NumPy, when the largest logit is NaN or an infinity (a NaN anywhere,
# +inf anywhere, or all -inf): a broken model or an overflow. No token is drawn from logits like these
NOT_FINITE = "The model computed logits that are not finite numbers (NaN or infinity), so no token can be drawn: its weights are broken or its numbers overflowed."
# what greedy leaves no say: it takes the largest logit, which a nucleus, a top-k and a min-p all leave in
NARROWING = ("topp", "top_k", "min_p")


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

    def __post_init__(self):
        object.__setattr__(self, "top_k", int(self.top_k))
        if self.top_k < 0 or not 0.0 <= self.min_p <= 1.0 or self.presence_penalty < 0.0:
            raise ValueError("top_k and presence_penalty are 0 or more, and min_p is from 0 to 1.")

    def beyond(self, known):
        """The settings a sampler that knows only those named would have to know as well, to draw as these say: the
        ones that are not at their defaults (greedy: but for NARROWING). The GPU's sampler asks (engine/external.py),
        and a setting it was never told of keeps the steps from it."""
        idle = NARROWING if self.temperature == 0.0 else ()
        return [field.name for field in fields(self)
                if field.name not in known and field.name not in idle and getattr(self, field.name) != field.default]


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

    def drawing(self, sampling, rng):
        """draw(logits, history, written) for one generation: the token of a step, by sampling (a Sampling) and the
        random numbers of rng. history is every token so far and written the sampled ones of them, both growing by
        the token drawn; the penalties are written into logits.

        The token is one of softmax(logits / temperature): of its top_k most probable where top_k > 0, then of their
        nucleus (0 < topp < 1: their probabilities add up to one again), then without what is less than min_p times
        as probable as the most probable: the order of llama.cpp's samplers and of transformers'."""
        temperature, topp, top_k, min_p = sampling.temperature, sampling.topp, sampling.top_k, sampling.min_p
        penalty, presence = sampling.repetition_penalty, sampling.presence_penalty
        nucleus, less = 0.0 < topp < 1.0, np.float32(presence)

        def draw(logits, history, written):
            if penalty != 1.0:
                # Make the tokens of the last steps less likely: tiny models love to loop
                recent = np.unique(history[-REPETITION_WINDOW:])
                logits[recent] = np.where(logits[recent] > 0, logits[recent] / penalty, logits[recent] * penalty)
            if presence != 0.0 and written:
                logits[np.unique(written[-REPETITION_WINDOW:])] -= less
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
                likely = likely[np.argsort(-probabilities[likely])]
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
        recent = np.empty(REPETITION_WINDOW, dtype=np.int32)
        probabilities_p, order_p, recent_p = probabilities.ctypes.data, order.ctypes.data, recent.ctypes.data
        self.buffers = (probabilities, order, recent)  # keep them alive: the kernels only know addresses
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

        seen = [None, 0]  # the list that recent[] mirrors, and its length then

        def latest(tokens):
            """recent[] as the latest of tokens, and how many of them it holds"""
            # recent[] is a ring of the latest tokens. generate() appends one token per step, and then one number
            # is written here: copying 64 of them from a list costs more than the kernel takes
            size = len(tokens)
            if seen[0] is tokens and size == seen[1] + 1:
                recent[(size - 1) % REPETITION_WINDOW] = tokens[-1]
            else:
                for position in range(max(size - REPETITION_WINDOW, 0), size):
                    recent[position % REPETITION_WINDOW] = tokens[position]
            seen[0], seen[1] = tokens, size
            return min(size, REPETITION_WINDOW)

        def drawing(sampling, rng):
            """NumpySampler.drawing()'s, on the kernels. A closure of a closure, and no method: a step reads the
            settings, the buffers, their addresses and the kernels as local names (it is on the path of every token
            the page writes)."""
            temperature, topp, top_k, min_p = sampling.temperature, sampling.topp, sampling.top_k, sampling.min_p
            penalty, presence = sampling.repetition_penalty, sampling.presence_penalty

            def draw(logits, history, written):
                if penalty != 1.0:
                    kernel_penalize(address(logits), recent_p, latest(history), penalty, 0.0)
                if presence != 0.0 and written:
                    kernel_penalize(address(logits), recent_p, latest(written), 1.0, presence)
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
