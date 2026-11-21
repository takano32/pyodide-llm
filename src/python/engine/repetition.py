# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository), and it is derived
# from tanakh/plamo-translate-desktop's engine::repetition
# (https://github.com/tanakh/plamo-translate-desktop at 3d69a116c45a8116159f1a405311a59039ec4b95), which is under the
# MIT License (it is under Apache-2.0 too; this is the MIT License's notice), Copyright (c) 2026 Hideyuki Tanaka.
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
"""engine/repetition.py (T667): what a run that falls into a loop is told by. A small model writing greedily can fall
into a cycle and write the same few tokens until it runs out of room (a translation of a long piece of text does:
engine/document.py). For every period up to MAX_PERIOD, the run of tokens each the same as the one a period before it
is kept, a step for each period and token: generate()'s loop does that itself where it is asked to (loops=True; in one
place of the loop, not a function: T639.12), and a stretch of at least least_span() tokens that repeats a cycle at
least MIN_REPEATS times is a loop. The functions here are called once a run: before it (least_span()) and where it
found a loop (kept()).

The form and the numbers are tanakh/plamo-translate-desktop's engine::repetition (3d69a116; the notice above)."""
import numpy as np

# the longest cycle told, in tokens (a longer one runs into the limit of the output), how many times a cycle repeats
# at least to be a loop, and the shortest stretch that is one (in a text that repeats nothing of its own)
MAX_PERIOD = 128
MIN_REPEATS = 3
MIN_LOOP_SPAN = 64


def longest_periodic_span(tokens):
    """the longest stretch of tokens that repeats a cycle of at most MAX_PERIOD tokens MIN_REPEATS times or more (0:
    none): for each period, the longest run of tokens each the same as the one a period before, with the period
    itself (a run of 2 periods is a stretch of 3)"""
    tokens = np.asarray(tokens, dtype=np.int64)
    longest = 0
    for period in range(1, min(MAX_PERIOD, len(tokens) - 1) + 1):
        same = np.concatenate(([0], (tokens[period:] == tokens[:-period]).astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(same))
        if not len(edges):
            continue
        run = int((edges[1::2] - edges[::2]).max())
        if run >= (MIN_REPEATS - 1) * period:
            longest = max(longest, run + period)
    return longest


def least_span(prompt):
    """the shortest stretch that is a loop in what a run writes after prompt: MIN_LOOP_SPAN, or twice the longest the
    prompt repeats itself where that is more (a ruler of dashes, a refrain: text the model copies)"""
    return max(MIN_LOOP_SPAN, 2 * longest_periodic_span(prompt))


def repeats_in(tokens, cycle):
    """the most times cycle comes back to back in tokens"""
    period, most = len(cycle), 0
    for start in range(len(tokens)):
        times = 0
        while tokens[start + times * period:start + (times + 1) * period] == cycle:
            times += 1
        most = max(most, times)
    return most


def kept(prompt, written, period, start):
    """how many of the tokens written are worth keeping where they end in a loop of period from start: to the end of
    the cycle's first time, or, where the prompt has the same cycle (lines the model copies), as many times as the
    prompt has it"""
    cycle = list(written[start:start + period])
    return min(start + max(repeats_in(list(prompt), cycle), 1) * period, len(written))
