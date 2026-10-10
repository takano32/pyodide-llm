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
import math

import numpy as np

from engine.checkpoint import OUTLIER_CHANNELS, Tensor, outlier_channels
from engine.sampler import REPETITION_WINDOW

# what an ExternalForward may have for a Llama to run by, in place of the NumPy forward pass and of what a Llama
# has where there is no engine outside (no blocks of a prompt, no steps on the GPU)
STEPS = ("forward", "forward_many", "prompt_block", "generate_many", "token_block")


class ExternalForward:
    """forward() in public/forward.js (T93): Python hands over where every tensor is, and the few small
    arrays it computes itself (the RoPE tables of a checkpoint that leaves them out, the outlier channels of
    T92), and gets the logits back into one array of its own, which the sampling kernels then read.

    It keeps the JavaScript engine and that array (engine, logits) until release(), and has, of STEPS, forward()
    and whatever else the engine offers. They are closures made here, not methods: a step is one Python call and
    then JavaScript's (it is on the path of every token the page writes)."""

    def __init__(self, model, external, int8, disable, kv_start):
        """model: the Llama whose tensors are outside (read here for the plan, and not kept); external: forward.js's
        (Llama's argument); int8: the matrices stay int8; disable: T52's switches; kv_start: KV_START."""
        # (the classifier of a model that has no other is its embedding, under both names)
        held = {name: getattr(model, name) for name in (*(row.name for row in model.rows), "wcls")}
        tensors = {name: tensor.plan() for name, tensor in held.items() if isinstance(tensor, Tensor)}
        derived = {name: np.ascontiguousarray(getattr(model, name), dtype=np.float32).tobytes()
                   for name in ("freq_cis_real", "freq_cis_imag") if isinstance(getattr(model, name), np.ndarray)}
        if model.rotated is not None:
            # T237: the signs of every width, with the transform's 1 / sqrt(block) in them (what the kernel multiplies by)
            scale = np.float32(1.0 / math.sqrt(model.rotated["block"]))
            derived.update({f"signs.{width}": (signs * scale).tobytes() for width, signs in model.rotated["signs"].items()})
        channels = []
        # (T237: not in a rotated basis, where the classifier reads R of its input: the rotation spreads a channel
        # over its block, and a column of the stored matrix is no channel's)
        if int8 and model.rotated is None:
            final = model.rms_final_weight
            raw = external.read(final.offset, model.dim * 4)
            weight = np.frombuffer(bytes(raw.to_py() if hasattr(raw, "to_py") else raw), dtype=np.float32)
            channels = [int(c) for c in outlier_channels(weight, min(OUTLIER_CHANNELS, model.dim))]
        plan = {"arch": model.arch, "dim": model.dim, "hidden_dim": model.hidden_dim, "n_layers": model.n_layers,
                "n_heads": model.n_heads, "n_kv_heads": model.n_kv_heads, "head_size": model.head_size,
                "vocab_size": model.vocab_size, "seq_len": model.seq_len, "rotary": model.rotary,
                "parallel_residual": bool(model.parallel_residual), "kv_start": kv_start, "rms_norm_eps": model.rms_norm_eps,
                "shared_classifier": model.wcls is model.token_embedding_table, "int8": bool(int8),
                "relaxed": "relaxed" not in disable, "tensors": tensors, "derived": derived, "outliers": channels,
                # T229: a Qwen3.5's linear-attention layers (None: none)
                "linear": model.linear,
                # T255: the layers RoPE leaves alone
                "unturned": list(model.unturned),
                # T260: an LFM2's convolution layers (None: none)
                "convolution": model.convolution,
                # T237: the block of a rotated basis (0: the model's own basis); its signs are in derived
                "rotated": model.rotated["block"] if model.rotated else 0,
                # T110: the keys and values of an int8 model may be float16 (forward.js uses that on a shared memory)
                "half_kv": bool(int8) and "kv16" not in disable}
        engine = external.start(plan)
        self.backend = str(engine.backend)
        logits = np.zeros(model.vocab_size, dtype=np.float32)
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
