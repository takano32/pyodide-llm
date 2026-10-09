# The order of a checkpoint's tensors, by architecture: what Llama.__init__ takes from the file, one after another.
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

from engine.layers import linear_widths


class TensorOrder:
    """The tensors of a checkpoint in the order of the file, for each architecture (Llama takes them in __init__)."""

    def llama_tensors(self, take, shared_weights, keep_int8, kv_dim, bias, dtype, frequencies, qk_norm=False):
        """The tensors of a Llama (and of a Qwen2, which adds the q, k and v biases at the end, and of a Qwen3, which
        adds the norms of q and k after them), in file order."""
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        # With a separate classifier the embedding table is only ever read one row at a time, so an int8 or
        # float16 table stays as it is (a quarter or half of the memory) and forward() widens the row it needs.
        self.token_embedding_table = take(self.vocab_size, dim, widen=shared_weights and not keep_int8)
        self.rms_att_weight = take(n_layers, dim, matrix=False)
        self.wq = take(n_layers, self.q_dim, dim, widen=not keep_int8)
        self.wk = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.wv = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.wo = take(n_layers, dim, self.q_dim, widen=not keep_int8)
        self.rms_ffn_weight = take(n_layers, dim, matrix=False)
        self.w1 = take(n_layers, hidden_dim, dim, widen=not keep_int8)
        self.w2 = take(n_layers, dim, hidden_dim, widen=not keep_int8)
        self.w3 = take(n_layers, hidden_dim, dim, widen=not keep_int8)
        self.rms_final_weight = take(dim, matrix=False)
        if dtype != np.int8:
            self.freq_cis_real = take(self.seq_len, self.head_size // 2, matrix=False)
            self.freq_cis_imag = take(self.seq_len, self.head_size // 2, matrix=False)
        self.wcls = self.token_embedding_table if shared_weights else take(self.vocab_size, dim, widen=not keep_int8)
        # the q, k and v biases go last, so that a checkpoint without them is the file it always was
        self.bq = take(n_layers, self.q_dim, matrix=False) if bias else None
        self.bk = take(n_layers, kv_dim, matrix=False) if bias else None
        self.bv = take(n_layers, kv_dim, matrix=False) if bias else None
        if qk_norm:
            self.q_norm = take(n_layers, self.head_size, matrix=False)
            self.k_norm = take(n_layers, self.head_size, matrix=False)
        if dtype != np.float32:
            # half precision is too coarse for the rotation angles, and int8 files leave the RoPE tables out
            angles = np.arange(self.seq_len)[:, None] * frequencies(self.head_size)
            self.freq_cis_real, self.freq_cis_imag = ((turn(angles) * self.rope_magnitude).astype(np.float32)
                                                      for turn in (np.cos, np.sin))

    def qwen35_tensors(self, take, shared_weights, keep_int8, kv_dim, dtype, frequencies):
        """The tensors of a Qwen3.5 (T229), in the order llama2_convert.layout() writes them: the stacks of the
        full-attention layers (q, its gate, k, v, o and the norms of the heads of q and k), those of the
        linear-attention layers, then the FFN of every layer as a Llama has it. The two small matrices of a linear
        layer's gates (wb, wa) are float32 in every file, like the norms: they feed a sigmoid and an exp."""
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        linear = self.linear
        mixed, _, read = linear_widths(linear)
        full = n_layers // linear["every"]
        lines = n_layers - full
        matrix = lambda *shape: take(*shape, widen=not keep_int8)
        vector = lambda *shape: take(*shape, matrix=False)
        self.token_embedding_table = take(self.vocab_size, dim, widen=shared_weights and not keep_int8)
        self.rms_att_weight = vector(n_layers, dim)
        self.wq, self.wg = matrix(full, self.q_dim, dim), matrix(full, self.q_dim, dim)
        self.wk, self.wv = matrix(full, kv_dim, dim), matrix(full, kv_dim, dim)
        self.wo = matrix(full, dim, self.q_dim)
        self.q_norm, self.k_norm = vector(full, self.head_size), vector(full, self.head_size)
        self.bq = self.bk = self.bv = None
        self.wqkv, self.wz = matrix(lines, mixed, dim), matrix(lines, read, dim)
        self.wb, self.wa = vector(lines, linear["value_heads"], dim), vector(lines, linear["value_heads"], dim)
        self.conv = vector(lines, linear["conv"], mixed)
        self.dt_bias, self.decay = vector(lines, linear["value_heads"]), vector(lines, linear["value_heads"])
        self.delta_norm = vector(lines, linear["value_dim"])
        self.wout = matrix(lines, dim, read)
        self.rms_ffn_weight = vector(n_layers, dim)
        self.w1, self.w2, self.w3 = matrix(n_layers, hidden_dim, dim), matrix(n_layers, dim, hidden_dim), matrix(n_layers, hidden_dim, dim)
        self.rms_final_weight = vector(dim)
        if dtype != np.int8:
            self.freq_cis_real = vector(self.seq_len, self.head_size // 2)
            self.freq_cis_imag = vector(self.seq_len, self.head_size // 2)
        self.wcls = self.token_embedding_table if shared_weights else matrix(self.vocab_size, dim)
        if dtype != np.float32:
            self.freq_cis_real, self.freq_cis_imag = self.partial_tables(frequencies)

    def lfm2_tensors(self, take, shared_weights, keep_int8, kv_dim, dtype, frequencies):
        """The tensors of an LFM2 (T260), in the order llama2_convert.layout() writes them: the stacks of the attention
        layers (q, k, v, o and the norms of the heads of q and k), those of the convolution layers (the matrix in,
        whose 3 dim rows are B, C and what B multiplies; the taps; the matrix out), then the FFN of every layer as a
        Llama has it."""
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        short = self.convolution["layers"].count("c")
        full = n_layers - short
        matrix = lambda *shape: take(*shape, widen=not keep_int8)
        vector = lambda *shape: take(*shape, matrix=False)
        self.token_embedding_table = take(self.vocab_size, dim, widen=shared_weights and not keep_int8)
        self.rms_att_weight = vector(n_layers, dim)
        self.wq, self.wk, self.wv = matrix(full, self.q_dim, dim), matrix(full, kv_dim, dim), matrix(full, kv_dim, dim)
        self.wo = matrix(full, dim, self.q_dim)
        self.q_norm, self.k_norm = vector(full, self.head_size), vector(full, self.head_size)
        self.bq = self.bk = self.bv = None
        self.win = matrix(short, 3 * dim, dim)
        self.conv = vector(short, self.convolution["taps"], dim)
        self.wout = matrix(short, dim, dim)
        self.rms_ffn_weight = vector(n_layers, dim)
        self.w1, self.w2, self.w3 = matrix(n_layers, hidden_dim, dim), matrix(n_layers, dim, hidden_dim), matrix(n_layers, hidden_dim, dim)
        self.rms_final_weight = vector(dim)
        if dtype != np.int8:
            self.freq_cis_real = vector(self.seq_len, self.head_size // 2)
            self.freq_cis_imag = vector(self.seq_len, self.head_size // 2)
        self.wcls = self.token_embedding_table if shared_weights else matrix(self.vocab_size, dim)
        if dtype != np.float32:
            angles = np.arange(self.seq_len)[:, None] * frequencies(self.head_size)
            self.freq_cis_real, self.freq_cis_imag = (turn(angles).astype(np.float32) for turn in (np.cos, np.sin))

    def partial_tables(self, frequencies):
        """The RoPE tables of a model that turns the first rotary values of a head only: the angles of that part, in
        tables of the shape the file has (the rest is never read)."""
        angles = np.arange(self.seq_len)[:, None] * frequencies(self.rotary)
        tables = [np.zeros((self.seq_len, self.head_size // 2), dtype=np.float32) for _ in range(2)]
        for table, values in zip(tables, (np.cos(angles), np.sin(angles))):
            table[:, :self.rotary // 2] = values
        return tables

    def gpt2_tensors(self, take, shared_weights, keep_int8, kv_dim, dtype, frequencies):
        """The tensors of a GPT-2 or a GPT-NeoX, in the order llama2_convert.layout() writes them. The two
        differ in one place: GPT-2 has a learned table of positions, GPT-NeoX the RoPE tables (left out of an
        int8 checkpoint, as everywhere)."""
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        vector = lambda n=dim: take(n_layers, n, matrix=False)
        self.token_embedding_table = take(self.vocab_size, dim, widen=shared_weights and not keep_int8)
        if self.arch == "neox":
            if dtype != np.int8:
                self.freq_cis_real = take(self.seq_len, self.head_size // 2, matrix=False)
                self.freq_cis_imag = take(self.seq_len, self.head_size // 2, matrix=False)
        else:
            self.positions = take(self.seq_len, dim, widen=True)
        self.rms_att_weight, self.ln_att_bias = vector(), vector()
        self.wq = take(n_layers, dim, dim, widen=not keep_int8)
        self.wk = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.wv = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.bq, self.bk, self.bv = vector(), vector(kv_dim), vector(kv_dim)
        self.wo = take(n_layers, dim, dim, widen=not keep_int8)
        self.bo = vector()
        self.rms_ffn_weight, self.ln_ffn_bias = vector(), vector()
        self.w1 = take(n_layers, hidden_dim, dim, widen=not keep_int8)
        self.b1 = vector(hidden_dim)
        self.w2 = take(n_layers, dim, hidden_dim, widen=not keep_int8)
        self.b2 = vector()
        self.rms_final_weight = take(dim, matrix=False)
        self.ln_final_bias = take(dim, matrix=False)
        self.wcls = self.token_embedding_table if shared_weights else take(self.vocab_size, dim, widen=not keep_int8)
        self.w3 = None
        if self.arch == "gpt2":
            # no rotation: the position is a row of a learned table, added to the embedding
            self.freq_cis_real = self.freq_cis_imag = np.zeros((self.seq_len, self.head_size // 2), dtype=np.float32)
        elif dtype != np.float32:
            self.freq_cis_real, self.freq_cis_imag = self.partial_tables(frequencies)
