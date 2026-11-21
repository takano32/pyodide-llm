# generate(): the prompt through the model, then a token at a time until a stop token, as pieces of text.
#
# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository), and it is
# derived from three works under the MIT License, whose notice follows: tairov/llama2.py
# (https://github.com/tairov/llama2.py; its LICENSE names no copyright holder), karpathy/llama2.c
# (https://github.com/karpathy/llama2.c), Copyright (c) 2023 Andrej, and (T667: the runs of each period that tell a
# loop) tanakh/plamo-translate-desktop's engine::repetition (https://github.com/tanakh/plamo-translate-desktop at
# 3d69a116c45a8116159f1a405311a59039ec4b95; under Apache-2.0 too), Copyright (c) 2026 Hideyuki Tanaka.
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

from engine.repetition import MAX_PERIOD, MIN_REPEATS, kept, least_span
from engine.sampler import Sampling
from engine.tokenizer import Unreplaced

# T520: the marks of a thought, in the formats of the list that think (Qwen3, Qwen3.5, DeepSeek-R1's distills, Granite
# 4.2, SmolLM3, MiniCPM5, Spark-X...): a token each in their vocabularies.
THOUGHT_OPENS, THOUGHT_CLOSES = "<think>", "</think>"
# T592.1: the marks of every kind of thought, the opening and the closing one. Harmony's (llm-jp-4's thinking forms)
# is its analysis channel, which its format opens (three tokens: no model of the list opens it itself) and its <|end|>
# closes; the answer is the final channel's message after it. T614: PLaMo 3's (PLaMo 3 610M Fin Instruct writes
# them itself after its format's header), three tokens each (<|plamo:begin_, think, :plamo|>): mark_of() finds no one
# token of them, so the engine follows no such thought and its budget closes none. The page folds the same, by the
# text (src/page/thought.ts' MARKS)
THOUGHT_MARKS = ((THOUGHT_OPENS, THOUGHT_CLOSES), ("<|channel|>analysis<|message|>", "<|end|>"),
                 ("<|plamo:begin_think:plamo|>", "<|plamo:end_think:plamo|>"))


def mark_of(model, mark):
    """T520: the token of a mark of a thought, where the model's vocabulary has it as one piece (None where it does
    not: a model that does not think, whose text may spell the mark out). Not only where it is among the specials: a
    DeepSeek-R1 distill's entry names other specials, and the model writes the token all the same."""
    try:
        tokens = model.tokenizer.encode(mark, (mark,))
    except KeyError:  # (no such piece: Tokenizer.encode() looks the special up in the vocabulary)
        return None
    return tokens[0] if len(tokens) == 1 else None


def thought_marks(model, prompt):
    """T592.1: the tokens of the marks of the thought a run follows (opens: None where the model cannot write it as one
    token; closes: None where there is no thought to follow) and whether the prompt opened it: the first kind of
    THOUGHT_MARKS that the prompt ends in the opening mark of (and the model has the closing one of), else the first
    the model may open and close itself."""
    for opening, closing in THOUGHT_MARKS:
        if isinstance(prompt, str):
            opened = prompt.rstrip().endswith(opening)
        else:  # (T666: the ids of a request, which open a thought where they end in its mark's token)
            opened = bool(prompt) and prompt[-1] == mark_of(model, opening)
        if opened and (closes := mark_of(model, closing)) is not None:
            return mark_of(model, opening), closes, True
    for opening, closing in THOUGHT_MARKS:
        if (closes := mark_of(model, closing)) is not None and (opens := mark_of(model, opening)) is not None:
            return opens, closes, False
    return None, None, False


def generate(model, prompt="", steps=256, seed=None, echo=True, thinking=0, loops=False, **settings):
    """Yield the text piece by piece, as it is generated. prompt: a text, or (T666) the ids of a request after the
    BOS (engine/pfml.py's), which go to the model as they are. echo=False leaves the prompt out of it (an instruction
    wrapped in a template, which nobody wants to read back). settings: how the tokens are drawn, Sampling's by their
    names (temperature, topp, repetition_penalty, top_k, min_p, presence_penalty, and T668's DRY: dry_multiplier and
    the rest), which says what each is and refuses what it cannot be. None of them is read here: the samplers have them.

    thinking (T520): the budget of a thought, in tokens (0: none). A model that has the marks of a thought in its
    vocabulary (mark_of()) is followed in and out of it (the format may open it: a prompt that ends in <think>, or in
    harmony's analysis channel: thought_marks()); once a thought has this many tokens and is still open, the closing
    mark is written in place of the next draws, as llama.cpp's reasoning budget does, and the model goes on to its answer. The stats say how long the thought was (thought: None
    where there was none) and whether the budget closed it (thought_cut).

    loops (T667): end the run where what it wrote falls into a loop (engine/repetition.py: a translation of a long
    text asks for it, engine/document.py; an answer does not, and nothing of it is kept for one). The stats then say
    where (loop: how many characters of the text written are worth keeping, where it ended in a loop; None where it
    did not).

    model: who writes, a Llama (whose generate() this is). Its parts are read off it as a run needs them and not
    handed over once, since a test or a tool may have put another in a part's place after the model was made: the
    tokenizer (with specials, bos, stop_tokens and seq_len), the forward pass (forward(); forward_many() and
    prompt_block() for a prompt's blocks, gpu_steps() and token_block() for steps on the GPU, where there are)
    and the sampler (its drawing()). What the run took is left in its stats."""
    sampling = Sampling(**settings)
    if isinstance(prompt, str):
        # (T258: after the BOS, which may take in the whitespace the prompt begins with)
        prompt_tokens = model.tokenizer.encode(prompt, model.specials, model.bos) if prompt else []
    else:  # T666: the ids of a request put together already (engine/pfml.py's pfml_tokens()), after the BOS
        prompt_tokens = [int(token) for token in prompt]
        if not all(0 <= token < model.vocab_size for token in prompt_tokens):  # (an id out of range reads another row)
            raise ValueError(f"The prompt has an id that is no token of the vocabulary (0 to {model.vocab_size - 1}).")
    # Right now we cannot run for more than seq_len steps
    if steps <= 0 or steps > model.seq_len:
        steps = model.seq_len
    if len(prompt_tokens) >= steps:
        raise ValueError(f"The prompt is {len(prompt_tokens)} tokens long, but only {steps - 1} fit.")
    if not (isinstance(thinking, int) and not isinstance(thinking, bool) and thinking >= 0):
        raise ValueError(f"thinking is a number of tokens, 0 or more (not {thinking!r}).")
    # T520: the thought, followed where the model has its closing mark: the token of each mark, whether a thought is
    # open (the format may have opened it), its tokens so far (None: no thought yet), and whether the budget closed it
    opens, closes, inside = thought_marks(model, prompt)
    thought = 0 if inside else None
    cut = False
    rng = np.random.default_rng(seed)
    # a character can be split over several tokens
    utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
    # T617: the replacements of the model's tokenizer turned back in what is written (none: nothing is called for it)
    unreplaced = Unreplaced(model.tokenizer.replaced) if model.tokenizer.replaced else None
    # T667: where loops are told: the shortest loop, the run of each period (runs[period - 1]: how many tokens at the
    # end of what was drawn are each the one a period before), the characters of text written after each drawn token,
    # and the loop found (its period and where it begins) and the characters of it worth keeping
    if loops:
        least, runs, ends, said = least_span(prompt_tokens), [0] * MAX_PERIOD, [], 0
    found = loop = None

    model._run += 1
    run = model._run
    token, count, sampled, forced = model.bos, 0, 0, 0
    history = [model.bos]
    written = []  # what was sampled: the presence penalty counts these, not the prompt's (T274)
    start = sampling_start = time.perf_counter()
    first_token = None
    try:
        # T108: the tokens of the prompt that make no logits (all but the last) go through the layers prompt_block()
        # at a time (T147: as many as the engine takes now), where the engine takes several; the text comes out as it
        # would one by one, only a block at once
        # (T680: a prompt of one token too: a model on the GPU alone has no CPU to run its forced token on)
        fed = [model.bos] + prompt_tokens[:-1] if model.forward_many is not None and prompt_tokens else []
        blocks = len(fed)  # (the position after the blocks, where the steps begin)
        pos = 0
        while pos < steps:
            if pos == blocks:
                if blocks:
                    sampling_start = time.perf_counter()  # (the prompt's blocks are through, their text out)
                # T152: a step on the GPU (gpu_steps()'s) samples there with the random numbers drawn here, several at
                # once (token_block()), where the GPU's sampler has all these settings; a step on the CPU is the
                # forward pass and the sampler's draw here. Both are made once, by the settings, after the prompt's
                # blocks: no step reads one. (Once: every step takes a token or more, so pos is here once.)
                # (T668: DRY's sequence breakers are tokens of the model's vocabulary: told to the sampler where it is on)
                draw = (model.sampler.drawing(sampling, rng, model.tokenizer.containing(sampling.dry_sequence_breakers))
                        if sampling.dry else model.sampler.drawing(sampling, rng))
                on_gpu = model.gpu_steps(sampling, rng) if model.gpu_steps is not None else None
                stops = sorted(model.stop_tokens)
            if pos < blocks:
                # asked again every block: where the GPU gives up, the CPU's blocks are short again (T108)
                block = fed[pos:pos + model.prompt_block()]
                model.forward_many(block, pos)
                chosen = prompt_tokens[pos:pos + len(block)]
            elif pos < len(prompt_tokens):
                # Still processing the prompt: force the next token, and the logits are not needed
                model.forward(token, pos, need_logits=False)
                chosen = [prompt_tokens[pos]]
                sampling_start = time.perf_counter()
            elif thinking and inside and thought >= thinking:
                # T520: the thought is as long as the budget: its closing mark in place of a draw. T680: the token
                # before it fed as a prompt's is, a block of one where the engine takes blocks: a model on the GPU
                # alone runs nothing on the CPU (its forward() throws, and the worker loads it again on the CPU)
                if model.forward_many is not None:
                    model.forward_many([token], pos)
                else:
                    model.forward(token, pos, need_logits=False)
                chosen, cut = [closes], True
            else:
                chosen = None
                if on_gpu is not None:
                    many = min(model.token_block(), steps - pos)
                    if thinking and inside:
                        many = min(many, thinking - thought)  # (no step past the budget: the GPU draws several at once)
                    if many > 0:
                        chosen = on_gpu(token, pos, history, many, stops)
                # (T390: no tokens are no answer either: the loop would ask for the same step for ever)
                if not chosen:
                    chosen = [draw(model.forward(token, pos), history, written)]
            # T639.12: the ledger of the run, in one place for the prompt's tokens (a block of them or one) and the
            # drawn ones: the text of each, the history, and the counts of the stats
            ended = False
            for next_token in chosen:
                drawn = pos >= len(prompt_tokens)
                if drawn:
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
                if drawn:
                    written.append(token)
                    if inside and token == closes:
                        inside = False
                    elif inside:
                        thought += 1
                    elif opens is not None and token == opens:
                        inside, thought = True, thought or 0
                else:
                    forced += 1
                count += 1
                if unreplaced is not None and (echo or drawn):
                    text = unreplaced(text)
                if text and (echo or drawn):
                    yield text
                pos += 1
                if loops and drawn:
                    # T667: each period's run a step on (a step for each period: the token is the one a period before
                    # it or the run is over), and a loop where a run and its period are least tokens and MIN_REPEATS
                    # periods (the shortest period first)
                    said += len(text)
                    ends.append(said)
                    last = len(written) - 1
                    for period in range(1, min(MAX_PERIOD, last) + 1):
                        same = runs[period - 1] + 1 if written[last - period] == token else 0
                        runs[period - 1] = same
                        if found is None and same + period >= least and same >= (MIN_REPEATS - 1) * period:
                            found = period, last + 1 - same - period
                    if found is not None:
                        ended = True
                        break
            if ended:
                break
        if found is not None:
            loop = ends[kept(prompt_tokens, written, *found) - 1]
        text = utf8.decode(b"", final=True)
        if unreplaced is not None:
            text = unreplaced(text, final=True)
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
                # T520: the tokens of the thought (to its closing mark, or all of it where it never closed; None: no
                # thought), and whether the budget closed it
                "thought": thought,
                "thought_cut": cut,
                # T667: where the run ended in a loop (only where loops were told)
                **({"loop": loop} if loops else {}),
            }
