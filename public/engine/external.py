# The forward pass outside Python: public/forward.js's, over weights in a WebAssembly memory of its own (T93). What
# the page runs; the reference in NumPy is Llama's own (engine/model.py).
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
import numpy as np

from engine.sampler import REPETITION_WINDOW

# what an ExternalForward may have for a Llama to run by, in place of the NumPy forward pass and of what a Llama
# has where there is no engine outside (no blocks of a prompt, no steps on the GPU)
STEPS = ("forward", "forward_many", "prompt_block", "generate_many", "token_block")


class ExternalForward:
    """forward() in public/forward.js (T93): the engine is made from a plan (engine/plan.py: where every tensor is, and
    the few small arrays Python computes itself), and it writes the logits into one array of Python's, which the
    sampling kernels then read.

    It keeps the JavaScript engine and that array (engine, logits) until release(), and has, of STEPS, forward()
    and whatever else the engine offers. They are closures made here, not methods: a step is one Python call and
    then JavaScript's (it is on the path of every token the page writes)."""

    def __init__(self, plan, external):
        """plan: forward_plan()'s, which is handed on and not kept (the bytes of its tables would stay in Python for as
        long as the model); external: forward.js's (Llama's argument), of which start(plan) makes the engine. Nothing
        of a model is read here: a plan and a stand-in for the engine are enough to make one."""
        engine = external.start(plan)
        self.backend = str(engine.backend)
        logits = np.zeros(plan["vocab_size"], dtype=np.float32)
        engine.bind(logits)
        self.engine, self.logits = engine, logits  # keep both alive: JS writes into the array
        run = engine.forward
        # T108: a prompt's tokens go through the layers several at a time, when forward.js offers that
        many = getattr(engine, "forwardMany", None)
        if many is not None:
            self.forward_many = lambda tokens, pos: many(list(tokens), pos)
            if getattr(engine, "promptBlock", None) is not None:
                self.prompt_block = lambda: int(engine.promptBlock)
        # T152: the steps of generate() on the GPU, where forward.js offers that
        on_gpu = getattr(engine, "generateMany", None)
        if on_gpu is not None:
            def generate_many(token, pos, history, count, temperature, topp, penalty, randoms, stops):
                ids = on_gpu(token, pos, list(history[-REPETITION_WINDOW:]), len(history), count, temperature, topp,
                            penalty, list(randoms), list(stops))
                return None if ids is None else [int(i) for i in ids]

            self.generate_many = generate_many
            self.token_block = lambda: int(engine.tokenBlock)

        def forward(token, pos, need_logits=True):
            run(token, pos, need_logits)
            return logits if need_logits else None

        self.forward = forward

    def release(self):
        """Lets go of what JavaScript holds for this engine (forward.js's forward pass and the array it fills). T205:
        what forward.js answers (a promise, settled once the GPU's worker let go of the device); None the second
        time."""
        engine, self.engine, self.logits = self.engine, None, None
        return None if engine is None else engine.release()
