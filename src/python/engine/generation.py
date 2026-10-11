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

from engine.sampler import Sampling


def generate(model, prompt="", steps=256, seed=None, echo=True, **settings):
    """Yield the text piece by piece, as it is generated. echo=False leaves the prompt out of it (an instruction
    wrapped in a template, which nobody wants to read back). settings: how the tokens are drawn, Sampling's by their
    names (temperature, topp, repetition_penalty, top_k, min_p, presence_penalty), which says what each is and
    refuses what it cannot be. None of them is read here: the samplers have them.

    model: who writes, a Llama (whose generate() this is). Its parts are read off it as a run needs them and not
    handed over once, since a test or a tool may have put another in a part's place after the model was made: the
    tokenizer (with specials, bos, stop_tokens and seq_len), the forward pass (forward(); forward_many() and
    prompt_block() for a prompt's blocks, gpu_steps() and token_block() for steps on the GPU, where there are)
    and the sampler (its drawing()). What the run took is left in its stats."""
    sampling = Sampling(**settings)
    prompt_tokens = model.tokenizer.encode(prompt, model.specials) if prompt else []
    # Right now we cannot run for more than seq_len steps
    if steps <= 0 or steps > model.seq_len:
        steps = model.seq_len
    if len(prompt_tokens) >= steps:
        raise ValueError(f"The prompt is {len(prompt_tokens)} tokens long, but only {steps - 1} fit.")
    rng = np.random.default_rng(seed)
    # a character can be split over several tokens
    utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    model._run += 1
    run = model._run
    token, count, sampled, forced = model.bos, 0, 0, 0
    history = [model.bos]
    written = []  # what was sampled: the presence penalty counts these, not the prompt's (T274)
    start = sampling_start = time.perf_counter()
    first_token = None
    first = 0
    try:
        if model.forward_many is not None and len(prompt_tokens) > 1:
            # T108: the tokens of the prompt that make no logits (all but the last) go through the layers
            # prompt_block() at a time (T147: as many as the engine takes now); the text comes out as it would
            # one by one, only a block at once
            fed = [model.bos] + prompt_tokens[:-1]
            at = 0
            while at < len(fed):
                # asked again every block: where the GPU gives up, the CPU's blocks are short again (T108)
                block = fed[at:at + model.prompt_block()]
                model.forward_many(block, at)
                for pos in range(at, at + len(block)):
                    next_token = prompt_tokens[pos]
                    text = utf8.decode(model.tokenizer.decode(token, next_token, model.bos))
                    token = next_token
                    history.append(token)
                    count += 1
                    forced += 1
                    if text and echo:
                        yield text
                at += len(block)
            first = len(fed)
            sampling_start = time.perf_counter()
        # T152: a step on the GPU (gpu_steps()'s) samples there with the random numbers drawn here, several at
        # once (token_block()), where the GPU's sampler has all these settings; a step on the CPU is the forward
        # pass and the sampler's draw here. Both are made once, by the settings: no step reads one
        draw = model.sampler.drawing(sampling, rng)
        on_gpu = model.gpu_steps(sampling, rng) if model.gpu_steps is not None else None
        stops = sorted(model.stop_tokens)
        pos = first
        while pos < steps:
            if pos < len(prompt_tokens):
                # Still processing the prompt: force the next token, and the logits are not needed
                model.forward(token, pos, need_logits=False)
                chosen = [prompt_tokens[pos]]
                forced += 1
                sampling_start = time.perf_counter()
            else:
                chosen = None
                if on_gpu is not None:
                    many = min(model.token_block(), steps - pos)
                    if many > 0:
                        chosen = on_gpu(token, pos, history, many, stops)
                # (T390: no tokens are no answer either: the loop would ask for the same step for ever)
                if not chosen:
                    chosen = [draw(model.forward(token, pos), history, written)]
            ended = False
            for next_token in chosen:
                if pos >= len(prompt_tokens):
                    sampled += 1
                    if first_token is None:
                        first_token = time.perf_counter()
                    # The BOS token delimits sequences: the story is over
                    if next_token in model.stop_tokens:
                        ended = True
                        break
                text = utf8.decode(model.tokenizer.decode(token, next_token, model.bos))
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
        if run == model._run:
            now = time.perf_counter()
            model.stats = {
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
