# The conversion as the page and the build drive it: config.json, the tokenizer and the stream of weights together.
import json

from engine.layout import FORM
from engine.layers import RMS_EPS
from engine.dtypes import dtype_of
from convert.template import config_token, model_answer_end, model_turn
from convert.families import family_of
from convert.config import check_config, normalize, rotary_dim
from convert.stream import Kept, Stream
from convert.gguf import gguf_model, gguf_read, gguf_tokenizer, rope_freqs_values
from convert.tokenizer import (Later, described_options, piece_ids, prefixed_texts, sentencepiece_charsmap,
                               sentencepiece_options, sentencepiece_pieces, sentencepiece_specials, tokenizer_bin,
                               tokenizer_json_charsmap, tokenizer_json_options, tokenizer_json_pieces, with_a_row,
                               written_out)


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
        vocab_size = self.config["vocab_size"]
        tokenizer = bytes(tokenizer.to_py() if hasattr(tokenizer, "to_py") else tokenizer)
        if tokenizer_name.lower().endswith(".json"):
            try:
                parsed = json.loads(tokenizer)
            except ValueError:
                raise ValueError("tokenizer.json is not JSON.") from None
            if not last and written_out(parsed) and prefixed_texts(parsed):
                raise Later("This tokenizer.json writes out a sentencepiece model, whose own file is read first.")
            options = tokenizer_json_options(parsed)
            pieces = with_a_row(list(tokenizer_json_pieces(parsed)), parsed, vocab_size)
            self.tokenizer = tokenizer_bin(pieces, vocab_size, spaces=options["tokenizer_kind"] != "bytebpe",
                                           charsmap=tokenizer_json_charsmap(parsed))
            added = [token for token in parsed.get("added_tokens", []) if token.get("id", 0) < vocab_size]
            specials = [token["content"] for token in added if token.get("special")]
            # T.143: the added tokens it does not call special are read as one token wherever they are too, by the real
            # tokenizers before it splits the text: Qwen3's <think>, Pythia's runs of 2 to 24 spaces (T.215's finding)
            added = [token["content"] for token in added if not token.get("special")]
        else:
            pieces = list(sentencepiece_pieces(tokenizer))
            self.tokenizer = tokenizer_bin(pieces, vocab_size, charsmap=sentencepiece_charsmap(tokenizer))
            # (T.265, T.308) and what only tokenizer_config.json says of a sentencepiece model. A model with no dummy
            # prefix has none after a special token either (T.400: its "never" over a legacy tokenizer's "wanting")
            own = sentencepiece_options(tokenizer)
            options = {**own, **described_options(described(tokenizer_config)), **({"prefixed": "never"} if own.get("prefixed") else {})}
            specials, added = sentencepiece_specials(tokenizer), []
        # T.143: the BOS is the token the tokenizer names, which transformers begins a text with, where config.json says
        # another: DeepSeek-R1's Distill says 151643 there, its end of a sentence, and <｜begin▁of▁sentence｜> (151646)
        # in tokenizer_config.json (T.138's review: perplexity 2.5 to 2.7 times higher with the former)
        named = config_token(described(tokenizer_config), "bos_token")
        bos = next((id for id, (text, _, _) in enumerate(pieces) if named and text == named), None)
        self.start(header, base, options, tokenizer_config, dtype, max_seq_len, start, sink, quantize_rows, specials,
                   chat_template, added, bos, readers, blocks)

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
        # the review of T.203: every text begins with the BOS, and the answer stops at it and at the EOS (start()).
        # A GGUF that names none would have tokens 1 and 2 there, '"' and '#' of a byte-level BPE vocabulary:
        # unsloth's Qwen3 GGUFs name no BOS, and every answer ended at its first lone '"'. The list takes those
        # with their original's config.json (gguf_weights), whose BOS is the original's
        nameless = "This GGUF names no {} token, which the engine needs: open it with its original's vocabulary and config.json."
        if not isinstance(metadata.get("tokenizer.ggml.eos_token_id"), int):
            raise ValueError(nameless.format("EOS"))
        # T.505.2: a GGUF that names no BOS is one whose texts llama.cpp begins with none (unsloth's Qwen3.5: no
        # bos_token_id, no add_bos_token), and every text of its chat template begins with the template's own token.
        # Where the template is read and begins with a token, that token is what the engine begins with (start()'s
        # head, T.264) and the BOS nobody named is neither written nor stopped at; anywhere else it is refused as it was
        named = isinstance(metadata.get("tokenizer.ggml.bos_token_id"), int)
        if not named:
            del config["bos_token_id"]
        if not callable(dtype):
            dtype_of(dtype)
        self.tokenizer, options, tokenizer_config, specials, added = gguf_tokenizer(metadata, config["vocab_size"])
        self.base = base
        self.start(header, base, options, tokenizer_config, dtype, max_seq_len, base, sink, quantize_rows, specials,
                   added=added, readers=readers, blocks=blocks, headless=None if named else nameless.format("BOS"))
        return self

    def start(self, header, base, options, tokenizer_config, dtype, max_seq_len, start, sink=None, quantize_rows=None,
              specials=(), chat_template=None, added=(), bos=None, readers=None, blocks=False, headless=None):
        """sink and quantize_rows: see Writer, readers: see Stream. checkpoint is None with a sink: the bytes went there. specials: the
        tokenizer's special tokens, the ones a chat template writes between the turns. chat_template: the text of
        chat_template.jinja, where there is one (T.127). added: the added tokens that are not special, one token
        wherever they are written (T.143). bos: the id of the BOS the tokenizer names, where it names one (T.143).
        headless (T.505.2): what to refuse a model with whose template begins with no token of its own (a GGUF that
        names no BOS: nothing else says what its texts begin with)."""
        own = self.config.get("bos_token_id", 1)
        # (the BOS the tokenizer names and config.json's, where each says one: the 1 of a config.json that has no such
        # key is a guess, T.264)
        said = [*([bos] if isinstance(bos, int) else []), *([own] if "bos_token_id" in self.config and own != bos else [])]
        bos = bos if isinstance(bos, int) else own
        eos = self.config.get("eos_token_id", 2)
        # the format of one turn, from the model's own chat_template (T.73). src/models.js wins when it has one
        ids = piece_ids(self.tokenizer, self.config["vocab_size"])
        template, head, mark = self.beginning(*model_turn(described(tokenizer_config), chat_template), [*specials, *added],
                                              options, ids)
        if headless and head is None:
            raise ValueError(headless)
        # the answer stops at the BOS, and at config.json's where that is another (T.143). Not at the token the template
        # begins with (T.264's head), though it is the engine's BOS then: it says where a message begins, not where the
        # answer ends. A harmony model (llm-jp-4) writes <|end|><|start|>assistant<|channel|>final<|message|> after its
        # analysis, and stopped there before its answer (the review of T.369); what ends an answer is what the model's own
        # files name, the EOSes.
        # A BOS that nothing names and the template does not begin with is no token of this model's to stop at: token 1
        # of a byte-level vocabulary is '"' (NeoHorse-1's config.json has no bos_token_id, and its answers ended at
        # their first quotation mark through ?hf=)
        # (T.400: also where the template does not begin with a token of its own: a guessed 1 that nothing names was a
        # stop token there, which a model with no template read stopped at too)
        stop = [token for token in [*said, *(eos if isinstance(eos, list) else [eos])] if isinstance(token, int)]
        # And at the EOS the tokenizer names, where that is a token of its own and another than config.json's (T.369): a
        # Qwen3.5's config.json says <|endoftext|> and its tokenizer <|im_end|>, which is what ends a turn of its chat
        # template; transformers' generate() stops at either. Without it an answer went on past <|im_end|> to the mark
        # of the next turn
        ending = config_token(described(tokenizer_config), "eos_token")
        ending = ids.get(ending.encode("utf-8")) if ending else None  # (no name is no piece: the padding's are empty)
        if ending is not None and ending not in stop:
            stop.append(ending)
        # T.524: and at the token the model's own template writes right after an answer, where that is one of the
        # tokens the engine reads as one: what ends a turn of the model's (a Gemma 4's <turn|>, which neither its
        # config.json nor its tokenizer names: ?hf= wrote two to five of them after each answer). Not where it is the
        # token every text begins with (T.264's head: it says where a message begins)
        end = model_answer_end(described(tokenizer_config), chat_template)
        marks = [token for token in [*specials, *added] if token and end and end.startswith(token) and token.encode("utf-8") in ids]
        closing = ids[max(marks, key=len).encode("utf-8")] if marks else None
        if closing is not None and closing != head and closing not in stop:
            stop.append(closing)
        bos = bos if head is None else head
        # a context longer than max_seq_len is cut: the RoPE tables and the scratch of the attention grow with it
        self.stream = (Kept if blocks else Stream)(header, int(base), self.config, dtype, int(max_seq_len),
                                                   start=int(start), sink=sink, quantize_rows=quantize_rows,
                                                   readers=readers)
        self.options = {**options, "dtype": self.stream.dtype, "rope_theta": float(self.config.get("rope_theta", 10000.0)),
                        "bos": bos if isinstance(bos, int) else 1, "stop_tokens": stop}
        # the form: bias and arch always (as since T.64 and T.65), the rest only where it is not the default, so that
        # the options of every model before T.124 stay what they were (kept.js's CONVERTER)
        self.options.update({key: value for key, value in self.stream.form.items()
                             if key in ("bias", "arch") or value != FORM[key]})
        if template:
            self.options["template"] = template
        # the special tokens the template writes stand for their token; spelled out they would be a dozen tokens each.
        # The added tokens that are not special, wherever they are written (T.143). The longest first, so that one that
        # begins another never cuts it short
        # (and the one the format began with, T.264: it is the BOS now, and still one token where a visitor types it)
        written = {special for special in specials if special and template and (special in template or special == mark)}
        written = sorted(written | {token for token in added if token}, key=lambda token: (-len(token), token))
        if written:
            self.options["specials"] = written
        eps = self.config.get("rms_norm_eps")
        # a GGUF says it in float32 (1e-5 is 9.99999974e-06 there): six digits are what config.json writes
        eps = float(f"{eps:.6g}") if isinstance(eps, (int, float)) and eps > 0 else RMS_EPS
        family = family_of(self.config)
        if family.rms_norm and eps != RMS_EPS:
            # T.124: the epsilon of RMSNorm, where it is not the engine's 1e-5 (Qwen2.5 and Qwen3: 1e-6, which moved
            # Qwen3 0.6B's perplexity by 0.12%). Only where it differs, like qk_norm and head_dim
            self.options["rms_norm_eps"] = float(eps)
        if self.config.get("rope_scaling"):
            # the int8 file has no RoPE tables: the engine makes them, and needs the scaling for that (Llama 3)
            self.options["rope_scaling"] = dict(self.config["rope_scaling"])
        if family.partly:
            # GPT-NeoX and Qwen3.5 turn part of every head: the file does not say how much
            self.options["rotary"] = rotary_dim(self.config)
        # and what only its family says (T.255: a SmolLM3's layers that RoPE leaves alone; a GPT-NeoX's branches)
        self.options.update(family.options(self.config))
        self.checkpoint = self.stream.out

    def beginning(self, turn, named, tokens, options, ids):
        """T.264: (the format of one turn as the engine takes it, the id of the token every text then begins with, or
        None where that stays the BOS, and that token's text) of a model's own turn (model_turn()'s), so that the page
        sends the ids of transformers' apply_chat_template for it, which puts nothing in front of what the template
        writes. generate() begins every text with the engine's BOS, so where the turn begins with the BOS the
        tokenizer names, the format is what follows it (as before T.264). Where it begins with another of the tokens
        the engine reads as one (tokens: the special ones and the added ones; a Qwen's or a Hermes 3's <|im_start|>),
        that token is the engine's BOS for this model and the format is what follows it: none of the tokenizer's own
        stands in front (a Qwen3 of 8 billion parameters was 178% worse with its <|endoftext|> there, a Hermes 3 13
        to 15%). Where it begins with text, the BOS stays in front as it was: the engine begins with a token, and the
        first one of a text is no fixed one.
        And it stays as it was where taking the token off would change how the rest is read: a sentencepiece
        tokenizer puts its dummy prefix before the first text and not before the text after a special token, so text
        that followed the token would be read as the first. Unless that is no other reading: every stretch is
        prefixed ("every"), or every one that does not begin with a space ("wanting") and what follows the token is
        the template's own text, which does not; or no stretch is ("never": T.400's sarashina2.2 and CAT-Translate).
        options: the tokenizer's, as the engine gets them. ids: piece_ids() of
        the tokenizer.bin."""
        if not turn:
            return turn, None, None
        if named and turn.startswith(named):
            return turn[len(named):], None, None
        heads = [token for token in tokens if token and turn.startswith(token) and token.encode("utf-8") in ids]
        if not heads:
            return turn, None, None
        head = max(heads, key=len)
        rest = turn[len(head):]
        prefixed = options.get("prefixed")
        read_the_same = options["tokenizer_kind"] == "bytebpe" or prefixed in ("every", "never") \
            or (prefixed == "wanting" and rest[:1] not in ("", " ") and not rest.startswith("{prompt")) \
            or any(rest.startswith(token) for token in tokens if token)
        return (rest, ids[head.encode("utf-8")], head) if read_the_same else (turn, None, None)

    def feed(self, data):
        done, total = self.stream.feed(data)
        return done / total

    def finish(self):
        self.stream.finish()
