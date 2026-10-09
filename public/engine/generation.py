# generate(): the prompt through the model, then a token at a time until a stop token, as pieces of text.
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
import codecs
import time

import numpy as np


class Generation:
    """How Llama writes: generate()."""

    def generate(self, prompt="", steps=256, temperature=0.0, topp=0.9, repetition_penalty=1.0, seed=None, echo=True,
                 top_k=0, min_p=0.0, presence_penalty=0.0):
        """Yield the text piece by piece, as it is generated. echo=False leaves the prompt out of it (an instruction
        wrapped in a template, which nobody wants to read back). T274: top_k (0: none), min_p (0: none) and
        presence_penalty (0: none) as sample() and penalize() tell; a step with any of them is the CPU's (the GPU's
        sampler has none of the three)."""
        top_k = int(top_k)
        if top_k < 0 or not 0.0 <= min_p <= 1.0 or presence_penalty < 0.0:
            raise ValueError("top_k and presence_penalty are 0 or more, and min_p is from 0 to 1.")
        # greedy takes the largest logit, which a top-k and a min-p leave in
        narrow = (top_k, min_p) if temperature != 0.0 else (0, 0.0)
        on_cpu = narrow != (0, 0.0) or presence_penalty != 0.0
        prompt_tokens = self.tokenizer.encode(prompt, self.specials) if prompt else []
        # Right now we cannot run for more than seq_len steps
        if steps <= 0 or steps > self.seq_len:
            steps = self.seq_len
        if len(prompt_tokens) >= steps:
            raise ValueError(f"The prompt is {len(prompt_tokens)} tokens long, but only {steps - 1} fit.")
        rng = np.random.default_rng(seed)
        # a character can be split over several tokens
        utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

        self._run += 1
        run = self._run
        token, count, sampled, forced = self.bos, 0, 0, 0
        history = [self.bos]
        written = []  # what was sampled: the presence penalty counts these, not the prompt's (T274, as OpenAI's does)
        start = sampling_start = time.perf_counter()
        first_token = None
        first = 0
        try:
            if self.forward_many is not None and len(prompt_tokens) > 1:
                # T108: the tokens of the prompt that make no logits (all but the last) go through the layers
                # prompt_block() at a time (T147: as many as the engine takes now); the text comes out as it would
                # one by one, only a block at once
                fed = [self.bos] + prompt_tokens[:-1]
                at = 0
                while at < len(fed):
                    # asked again every block: where the GPU gives up, the CPU's blocks are short again (T108)
                    block = fed[at:at + self.prompt_block()]
                    self.forward_many(block, at)
                    for pos in range(at, at + len(block)):
                        next_token = prompt_tokens[pos]
                        text = utf8.decode(self.tokenizer.decode(token, next_token, self.bos))
                        token = next_token
                        history.append(token)
                        count += 1
                        forced += 1
                        if text and echo:
                            yield text
                    at += len(block)
                first = len(fed)
                sampling_start = time.perf_counter()
            # T152: a step on the GPU (generate_many) samples there with the random numbers drawn here, several at
            # once (token_block()); a step on the CPU is the forward pass and the sampling here
            stops = sorted(self.stop_tokens)
            pos = first
            while pos < steps:
                if pos < len(prompt_tokens):
                    # Still processing the prompt: force the next token, and the logits are not needed
                    self.forward(token, pos, need_logits=False)
                    chosen = [prompt_tokens[pos]]
                    forced += 1
                    sampling_start = time.perf_counter()
                else:
                    chosen = None
                    many = 0 if on_cpu else min(self.token_block(), steps - pos)
                    if many > 0:
                        # a number for every step, drawn in the order the CPU draws them (none where greedy);
                        # those of the steps after a stop token go unused
                        randoms = [rng.random() for _ in range(many)] if temperature != 0.0 else []
                        chosen = self.generate_many(token, pos, history, many, temperature, topp, repetition_penalty,
                                                    randoms, stops)
                    if chosen is None:
                        logits = self.forward(token, pos)
                        if repetition_penalty != 1.0:
                            self.penalize(logits, history, repetition_penalty)
                        if presence_penalty != 0.0 and written:
                            # (with the prompt's tokens in it, the format's own stop token would lose it too)
                            self.penalize(logits, written, 1.0, presence_penalty)
                        chosen = [self.sample(logits, temperature, topp, rng, *narrow)]
                ended = False
                for next_token in chosen:
                    if pos >= len(prompt_tokens):
                        sampled += 1
                        if first_token is None:
                            first_token = time.perf_counter()
                        # The BOS token delimits sequences: the story is over
                        if next_token in self.stop_tokens:
                            ended = True
                            break
                    text = utf8.decode(self.tokenizer.decode(token, next_token, self.bos))
                    token = next_token
                    history.append(token)
                    if pos >= len(prompt_tokens):
                        written.append(token)
                    count += 1
                    if text and (echo or pos >= len(prompt_tokens)):
                        yield text
                    pos += 1
                if ended:
                    break
            text = utf8.decode(b"", final=True)
            if text:
                yield text
        finally:
            # an abandoned generator that is finalized late must not overwrite the stats of a newer run
            if run == self._run:
                now = time.perf_counter()
                self.stats = {
                    "tokens": count,
                    "seconds": now - start,
                    # the prompt is not counted: its tokens skip the classifier, so they are much cheaper
                    "tokens_per_second": sampled / (now - sampling_start) if now > sampling_start else 0.0,
                    # The prompt costs a forward pass per token too, and it is what the reader waits for first.
                    # sampled counts the sampling steps, one more than "tokens" shows when a stop token ended it.
                    "sampled": sampled,
                    "prompt_tokens": forced,
                    "prompt_seconds": sampling_start - start,
                    "prompt_tokens_per_second": forced / (sampling_start - start) if sampling_start > start else 0.0,
                    "first_token_seconds": first_token - start if first_token is not None else 0.0,
                }
