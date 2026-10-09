# Drawing a token from the logits: the repetition and presence penalties, greedy, and temperature with top-k, top-p
# and min-p, in NumPy and on the kernels.
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

import numpy as np

REPETITION_WINDOW = 64  # the repetition penalty looks at this many of the latest tokens
# T195: what sample() says, on the kernels and in NumPy, when the largest logit is NaN or an infinity (a NaN anywhere,
# +inf anywhere, or all -inf): a broken model or an overflow. No token is drawn from logits like these
NOT_FINITE = "The model computed logits that are not finite numbers (NaN or infinity), so no token can be drawn: its weights are broken or its numbers overflowed."


class Sampling:
    """How Llama draws a token from its logits."""

    def kernel_sampler(self, kernels):
        """penalize() and sample() on the kernels: the same as the methods below, which stay for NumPy alone.

        Sorting the candidates of the nucleus was most of the time of a step for a small model with a large
        vocabulary, and a repetition penalty, which flattens the distribution, made it worse.
        """
        probabilities, order = np.empty(self.vocab_size, dtype=np.float32), np.empty(self.vocab_size, dtype=np.int32)
        recent = np.empty(REPETITION_WINDOW, dtype=np.int32)
        probabilities_p, order_p, recent_p = probabilities.ctypes.data, order.ctypes.data, recent.ctypes.data
        self._sampler_buffers = (probabilities, order, recent)  # keep them alive: the kernels only know addresses
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

        def penalize(logits, history, penalty, presence=0.0):
            # recent[] is a ring of the latest tokens. generate() appends one token per step, and then one number
            # is written here: copying 64 of them from a list costs more than the kernel takes
            size = len(history)
            if seen[0] is history and size == seen[1] + 1:
                recent[(size - 1) % REPETITION_WINDOW] = history[-1]
            else:
                for position in range(max(size - REPETITION_WINDOW, 0), size):
                    recent[position % REPETITION_WINDOW] = history[position]
            seen[0], seen[1] = history, size
            kernel_penalize(address(logits), recent_p, min(size, REPETITION_WINDOW), penalty, presence)

        def sample(logits, temperature, topp, rng, top_k=0, min_p=0.0):
            if temperature == 0.0:
                return self.greedy(logits)
            # the random number is drawn here, so that a seed gives the same text again
            token = kernel_sample(address(logits), logits.size, temperature, topp, rng.random(), probabilities_p, order_p,
                                  top_k, min_p)
            if token < 0:
                raise ValueError(NOT_FINITE)
            return token

        return penalize, sample

    def penalize(self, logits, history, penalty, presence=0.0):
        """Make the tokens of the last steps less likely: tiny models love to loop. T274: presence is taken off the
        logit of each after that (a presence penalty, the same however often a token came; generate() gives it the
        sampled tokens alone, with a penalty of 1)."""
        recent = np.unique(history[-REPETITION_WINDOW:])
        logits[recent] = (np.where(logits[recent] > 0, logits[recent] / penalty, logits[recent] * penalty)
                          - np.float32(max(presence, 0.0)))

    @staticmethod
    def greedy(logits):
        """Greedy argmax sampling: take the token with the highest probability. NumPy's argmax takes a NaN for the
        largest, so the logit it picks is finite exactly when the largest one is (T195)."""
        token = int(np.argmax(logits))
        if not math.isfinite(logits[token]):
            raise ValueError(NOT_FINITE)
        return token

    def sample(self, logits, temperature, topp, rng, top_k=0, min_p=0.0):
        """A token of softmax(logits / temperature). T274: of its top_k most probable where top_k > 0, then of their
        nucleus (0 < topp < 1: their probabilities add up to one again), then without what is less than min_p times
        as probable as the most probable: the order of llama.cpp's samplers and of transformers'."""
        if temperature == 0.0:
            return self.greedy(logits)
        # max() keeps a NaN (T195)
        best = logits.max()
        if not math.isfinite(best):
            raise ValueError(NOT_FINITE)
        nucleus = 0.0 < topp < 1.0
        if nucleus:
            # exp() over a vocabulary of 50000 or 100000 tokens costs as much as half a forward pass, and nearly all
            # of it goes to tokens that cannot be drawn: leave out, while still cheap, whatever is less than a ten
            # millionth as probable as the best token (together far below one percent of the probability mass)
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
            # That holds while one token at least stays: when all are below it (n * topp < 1, which the floor above
            # makes possible, T178), the others add up to less than (1 - topp), so the set is the most probable token
            # alone. It always stays
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
