# The conversion as the page and the build drive it: config.json, the tokenizer and the stream of weights together.
import json

from engine.dtypes import dtype_of
from engine.proxy import plain
from convert.config import check_config, normalize
from convert.stream import Kept, Stream
from convert.gguf import gguf_model, gguf_read, rope_freqs_values
from convert.beginning import beginning_and_stops, options_of
from convert.vocabulary import gguf_vocabulary, vocabulary_of


class Conversion:
    """A Hugging Face model converted inside the page, from the visitor's disk or from huggingface.co.

    header: the JSON at the beginning of model.safetensors (text), base: where its tensors begin, config: the text of
    config.json, tokenizer: the bytes of tokenizer.json or of a sentencepiece model. Then feed() the bytes of the
    file in order, beginning at start, and finish(). checkpoint, tokenizer and options are what Llama() takes.
    blocks (T.363.3): the tensors a GGUF holds in llama.cpp's blocks stay the bytes they are (convert/stream.py's Kept),
    and dtype is how the others are stored.
    last (T.257): no other tokenizer of the model is left to try. Without it a tokenizer.json that writes out a
    sentencepiece model with its dummy prefix is put off (convert.tokenizer's Later), for the model's own file to be
    read where there is one.
    """

    def __init__(self, header, base, config, tokenizer, tokenizer_name, dtype="int8", max_seq_len=4096, start=0,
                 tokenizer_config=None, sink=None, quantize_rows=None, chat_template=None, readers=None, blocks=False,
                 last=True):
        try:
            self.config = json.loads(config)
        except ValueError:
            raise ValueError("config.json is not JSON.") from None
        if not isinstance(self.config, dict):
            raise ValueError("config.json is not the configuration of a model.")
        # GPT-2 spells its config differently: from here on it has the names the rest of the file uses
        self.config = normalize(self.config)
        check_config(self.config)
        if not callable(dtype):
            dtype_of(dtype)
        try:
            header = json.loads(header)
        except ValueError:
            raise ValueError("This is not a safetensors file.") from None
        tokenizer = bytes(plain(tokenizer))
        vocabulary = vocabulary_of(tokenizer, tokenizer_name, self.config["vocab_size"], tokenizer_config, last)
        self.start(header, base, vocabulary, dtype, max_seq_len, start, sink, quantize_rows, chat_template, readers,
                   blocks)

    @classmethod
    def from_gguf(cls, head, dtype="int8", max_seq_len=4096, sink=None, quantize_rows=None, readers=None, blocks=False):
        """The same from a GGUF file (T.74): head is its beginning, as far as the tensors' data (Incomplete when it
        is not). Then feed() the file from self.base on. No config.json and no tokenizer: the GGUF has both."""
        metadata, tensors, base = gguf_read(head)
        # (T.505.2: llama.cpp's table of a Llama 3's RoPE scaling, which a GGUF alone has in the place of the scaling)
        header, config = gguf_model(metadata, tensors, base, table=rope_freqs_values(head, tensors, base, metadata))
        self = cls.__new__(cls)
        self.config = config
        check_config(normalize(config))  # a GPT-2's is in config.json's own spelling (T.136's third stage)
        # the review of T.203: every text begins with the BOS, and the answer stops at it and at the EOS
        # (beginning_and_stops()). A GGUF that names none would have tokens 1 and 2 there, '"' and '#' of a byte-level
        # BPE vocabulary: unsloth's Qwen3 GGUFs name no BOS, and every answer ended at its first lone '"'. The list
        # takes those with their original's config.json (gguf_weights), whose BOS is the original's
        nameless = "This GGUF names no {} token, which the engine needs: open it with its original's vocabulary and config.json."
        if not isinstance(metadata.get("tokenizer.ggml.eos_token_id"), int):
            raise ValueError(nameless.format("EOS"))
        # T.505.2: a GGUF that names no BOS is one whose texts llama.cpp begins with none (unsloth's Qwen3.5: no
        # bos_token_id, no add_bos_token), and every text of its chat template begins with the template's own token.
        # Where the template is read and begins with a token, that token is what the engine begins with
        # (convert/beginning.py's turn_head(), T.264) and the BOS nobody named is neither written nor stopped at;
        # anywhere else it is refused as it was
        named = isinstance(metadata.get("tokenizer.ggml.bos_token_id"), int)
        if not named:
            del config["bos_token_id"]
        if not callable(dtype):
            dtype_of(dtype)
        vocabulary = gguf_vocabulary(metadata, config["vocab_size"])
        self.base = base
        self.start(header, base, vocabulary, dtype, max_seq_len, base, sink, quantize_rows, readers=readers,
                   blocks=blocks, headless=None if named else nameless.format("BOS"))
        return self

    def start(self, header, base, vocabulary, dtype, max_seq_len, start, sink=None, quantize_rows=None,
              chat_template=None, readers=None, blocks=False, headless=None):
        """vocabulary: convert/vocabulary.py's Vocabulary. sink and quantize_rows: see Writer, readers: see Stream.
        checkpoint is None with a sink: the bytes went there. chat_template: the text of chat_template.jinja, where
        there is one (T.127). headless (T.505.2): what to refuse a model with whose template begins with no token of
        its own (a GGUF that names no BOS: nothing else says what its texts begin with)."""
        self.tokenizer = vocabulary.tokenizer
        # how a text begins and where an answer stops, before the weights' memory is asked for: what is refused there
        # is refused before the sink is opened (T.505.2, T.509)
        beginning = beginning_and_stops(self.config, vocabulary, chat_template, headless)
        # a context longer than max_seq_len is cut: the RoPE tables and the scratch of the attention grow with it
        self.stream = (Kept if blocks else Stream)(header, int(base), self.config, dtype, int(max_seq_len),
                                                   start=int(start), sink=sink, quantize_rows=quantize_rows,
                                                   readers=readers)
        self.options = options_of(self.stream.form, self.stream.dtype, self.config, vocabulary, beginning)
        self.checkpoint = self.stream.out

    @property
    def widened(self):
        """T.385: the stored types that have a kernel's reader, each as it was read so far ("kernel" or "NumPy")"""
        return dict(self.stream.widened)

    def feed(self, data):
        done, total = self.stream.feed(data)
        return done / total

    def finish(self):
        self.stream.finish()
