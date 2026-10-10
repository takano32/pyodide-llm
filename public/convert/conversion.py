# The conversion as the page and the build drive it: config.json, the tokenizer and the stream of weights together.
import json

from engine.layout import FORM
from engine.layers import RMS_EPS
from engine.dtypes import dtype_of
from convert.template import config_token, one_turn_template
from convert.families import family_of
from convert.config import check_config, normalize, rotary_dim
from convert.stream import Stream
from convert.gguf import gguf_model, gguf_read, gguf_tokenizer
from convert.tokenizer import (described_options, sentencepiece_charsmap, sentencepiece_options,
                               sentencepiece_pieces, sentencepiece_specials, tokenizer_bin, tokenizer_json_charsmap,
                               tokenizer_json_options, tokenizer_json_pieces)


def described(tokenizer_config):
    """A tokenizer_config.json, read: its text, its bytes or the dict itself. {} for none and for one that is no JSON
    or no object."""
    try:
        config = json.loads(tokenizer_config) if isinstance(tokenizer_config, (str, bytes)) else tokenizer_config
    except ValueError:
        config = None
    return config if isinstance(config, dict) else {}



class Conversion:
    """A Hugging Face model converted inside the page, from the visitor's disk or from huggingface.co.

    header: the JSON at the beginning of model.safetensors (text), base: where its tensors begin, config: the text of
    config.json, tokenizer: the bytes of tokenizer.json or of a sentencepiece model. Then feed() the bytes of the
    file in order, beginning at start, and finish(). checkpoint, tokenizer and options are what Llama() takes.
    """

    def __init__(self, header, base, config, tokenizer, tokenizer_name, dtype="int8", max_seq_len=4096, start=0,
                 tokenizer_config=None, sink=None, quantize_rows=None, chat_template=None, readers=None):
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
        vocab_size = self.config["vocab_size"]
        tokenizer = bytes(tokenizer.to_py() if hasattr(tokenizer, "to_py") else tokenizer)
        if tokenizer_name.lower().endswith(".json"):
            try:
                parsed = json.loads(tokenizer)
            except ValueError:
                raise ValueError("tokenizer.json is not JSON.") from None
            options = tokenizer_json_options(parsed)
            pieces = list(tokenizer_json_pieces(parsed))
            self.tokenizer = tokenizer_bin(pieces, vocab_size, spaces=options["tokenizer_kind"] != "bytebpe",
                                           charsmap=tokenizer_json_charsmap(parsed))
            added = parsed.get("added_tokens", [])
            specials = [token["content"] for token in added if token.get("special")]
            # T143: the added tokens it does not call special are read as one token wherever they are too, by the real
            # tokenizers before it splits the text: Qwen3's <think>, Pythia's runs of 2 to 24 spaces (T215's finding)
            added = [token["content"] for token in added if not token.get("special")]
        else:
            pieces = list(sentencepiece_pieces(tokenizer))
            self.tokenizer = tokenizer_bin(pieces, vocab_size, charsmap=sentencepiece_charsmap(tokenizer))
            # (T265, T308) and what only tokenizer_config.json says of a sentencepiece model
            options = {**sentencepiece_options(tokenizer), **described_options(described(tokenizer_config))}
            specials, added = sentencepiece_specials(tokenizer), []
        # T143: the BOS is the token the tokenizer names, which transformers begins a text with, where config.json says
        # another: DeepSeek-R1's Distill says 151643 there, its end of a sentence, and <｜begin▁of▁sentence｜> (151646)
        # in tokenizer_config.json (T138's review: perplexity 2.5 to 2.7 times higher with the former)
        named = config_token(described(tokenizer_config), "bos_token")
        bos = next((id for id, (text, _, _) in enumerate(pieces) if named and text == named), None)
        self.start(header, base, options, tokenizer_config, dtype, max_seq_len, start, sink, quantize_rows, specials,
                   chat_template, added, bos, readers)

    @classmethod
    def from_gguf(cls, head, dtype="int8", max_seq_len=4096, sink=None, quantize_rows=None, readers=None):
        """The same from a GGUF file (T74): head is its beginning, as far as the tensors' data (Incomplete when it
        is not). Then feed() the file from self.base on. No config.json and no tokenizer: the GGUF has both."""
        metadata, tensors, base = gguf_read(head)
        header, config = gguf_model(metadata, tensors, base)
        self = cls.__new__(cls)
        self.config = config
        check_config(normalize(config))  # a GPT-2's is in config.json's own spelling (T136's third stage)
        for token in ("bos", "eos"):
            # the review of T203: every text begins with the BOS, and the answer stops at it and at the EOS (start()).
            # A GGUF that names none would have tokens 1 and 2 there, '"' and '#' of a byte-level BPE vocabulary:
            # unsloth's Qwen3 GGUFs name no BOS, and every answer ended at its first lone '"'. The list takes those
            # with their original's config.json (gguf_weights), whose BOS is the original's
            if not isinstance(metadata.get(f"tokenizer.ggml.{token}_token_id"), int):
                raise ValueError(f"This GGUF names no {token.upper()} token, which the engine needs: open it with its "
                                 f"original's vocabulary and config.json.")
        if not callable(dtype):
            dtype_of(dtype)
        self.tokenizer, options, tokenizer_config, specials, added = gguf_tokenizer(metadata, config["vocab_size"])
        self.base = base
        self.start(header, base, options, tokenizer_config, dtype, max_seq_len, base, sink, quantize_rows, specials,
                   added=added, readers=readers)
        return self

    def start(self, header, base, options, tokenizer_config, dtype, max_seq_len, start, sink=None, quantize_rows=None,
              specials=(), chat_template=None, added=(), bos=None, readers=None):
        """sink and quantize_rows: see Writer, readers: see Stream. checkpoint is None with a sink: the bytes went there. specials: the
        tokenizer's special tokens, the ones a chat template writes between the turns. chat_template: the text of
        chat_template.jinja, where there is one (T127). added: the added tokens that are not special, one token
        wherever they are written (T143). bos: the id of the BOS the tokenizer names, where it names one (T143)."""
        own = self.config.get("bos_token_id", 1)
        bos = bos if isinstance(bos, int) else own
        eos = self.config.get("eos_token_id", 2)
        # the answer stops at the BOS, and at config.json's where that is another (T143)
        stop = [token for token in [bos, *([own] if own != bos else []), *(eos if isinstance(eos, list) else [eos])]
                if isinstance(token, int)]
        # a context longer than max_seq_len is cut: the RoPE tables and the scratch of the attention grow with it
        self.stream = Stream(header, int(base), self.config, dtype, int(max_seq_len), start=int(start), sink=sink,
                             quantize_rows=quantize_rows, readers=readers)
        self.options = {**options, "dtype": self.stream.dtype, "rope_theta": float(self.config.get("rope_theta", 10000.0)),
                        "bos": bos if isinstance(bos, int) else 1, "stop_tokens": stop}
        # the form: bias and arch always (as since T64 and T65), the rest only where it is not the default, so that
        # the options of every model before T124 stay what they were (kept.js's CONVERTER)
        self.options.update({key: value for key, value in self.stream.form.items()
                             if key in ("bias", "arch") or value != FORM[key]})
        # the format of one turn, from the model's own chat_template (T73). src/models.js wins when it has one
        template = one_turn_template(tokenizer_config, chat_template)
        if template:
            self.options["template"] = template
        # the special tokens the template writes stand for their token; spelled out they would be a dozen tokens each.
        # The added tokens that are not special, wherever they are written (T143). The longest first, so that one that
        # begins another never cuts it short
        written = {special for special in specials if special and template and special in template}
        written = sorted(written | {token for token in added if token}, key=lambda token: (-len(token), token))
        if written:
            self.options["specials"] = written
        eps = self.config.get("rms_norm_eps")
        # a GGUF says it in float32 (1e-5 is 9.99999974e-06 there): six digits are what config.json writes
        eps = float(f"{eps:.6g}") if isinstance(eps, (int, float)) and eps > 0 else RMS_EPS
        family = family_of(self.config)
        if family.rms_norm and eps != RMS_EPS:
            # T124: the epsilon of RMSNorm, where it is not the engine's 1e-5 (Qwen2.5 and Qwen3: 1e-6, which moved
            # Qwen3 0.6B's perplexity by 0.12%). Only where it differs, like qk_norm and head_dim
            self.options["rms_norm_eps"] = float(eps)
        if self.config.get("rope_scaling"):
            # the int8 file has no RoPE tables: the engine makes them, and needs the scaling for that (Llama 3)
            self.options["rope_scaling"] = dict(self.config["rope_scaling"])
        if family.partly:
            # GPT-NeoX and Qwen3.5 turn part of every head: the file does not say how much
            self.options["rotary"] = rotary_dim(self.config)
        # and what only its family says (T255: a SmolLM3's layers that RoPE leaves alone; a GPT-NeoX's branches)
        self.options.update(family.options(self.config))
        self.checkpoint = self.stream.out

    def feed(self, data):
        done, total = self.stream.feed(data)
        return done / total

    def finish(self):
        self.stream.finish()
