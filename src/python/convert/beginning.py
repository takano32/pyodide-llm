# How a model's text begins, where its answer stops and which tokens its format writes as one (T.640.1): rules of
# config.json, the vocabulary (convert/vocabulary.py's Vocabulary) and the chat template alone, with no weights. And
# the options the engine gets, from those and the stream's form.
from typing import NamedTuple

from engine.layout import FORM
from engine.layers import RMS_EPS
from convert.template import config_token, model_answer_end, model_turn
from convert.families import family_of
from convert.config import rotary_dim
from convert.tokenizer import piece_ids


class Beginning(NamedTuple):
    """bos: the token every text begins with. stop_tokens: where an answer stops. template: the format of one turn as
    the engine takes it (None for none). specials: the tokens the engine reads as one wherever they are written, the
    longest first (empty for none)."""
    bos: int
    stop_tokens: list
    template: object
    specials: list


def turn_head(turn, named, tokens, options, ids):
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


def beginning_and_stops(config, vocabulary, chat_template=None, headless=None):
    """The Beginning of a model: config.json (as the conversion reads it), its Vocabulary and the text of its
    chat_template.jinja where there is one (T.127). headless (T.505.2): what to refuse a model with whose template
    begins with no token of its own (a GGUF that names no BOS: nothing else says what its texts begin with)."""
    said_config, specials, added, bos = vocabulary.described, vocabulary.specials, vocabulary.added, vocabulary.bos
    own = config.get("bos_token_id", 1)
    # (the BOS the tokenizer names and config.json's, where each says one: the 1 of a config.json that has no such
    # key is a guess, T.264)
    said = [*([bos] if isinstance(bos, int) else []), *([own] if "bos_token_id" in config and own != bos else [])]
    bos = bos if isinstance(bos, int) else own
    eos = config.get("eos_token_id", 2)
    # the format of one turn, from the model's own chat_template (T.73). src/models.ts wins when it has one
    ids = piece_ids(vocabulary.tokenizer, config["vocab_size"])
    template, head, mark = turn_head(*model_turn(said_config, chat_template), [*specials, *added], vocabulary.options, ids)
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
    ending = config_token(said_config, "eos_token")
    ending = ids.get(ending.encode("utf-8")) if ending else None  # (no name is no piece: the padding's are empty)
    if ending is not None and ending not in stop:
        stop.append(ending)
    # T.524: and at the token the model's own template writes right after an answer, where that is one of the
    # tokens the engine reads as one: what ends a turn of the model's (a Gemma 4's <turn|>, which neither its
    # config.json nor its tokenizer names: ?hf= wrote two to five of them after each answer). Not where it is the
    # token every text begins with (T.264's head: it says where a message begins)
    end = model_answer_end(said_config, chat_template)
    marks = [token for token in [*specials, *added] if token and end and end.startswith(token) and token.encode("utf-8") in ids]
    closing = ids[max(marks, key=len).encode("utf-8")] if marks else None
    if closing is not None and closing != head and closing not in stop:
        stop.append(closing)
    # T.509: an added token past the model's last row is left out (with_a_row()), which is right only where nothing
    # of the model's uses it: a BOS or a stop token of no row, or one the format writes (spelled out letter by letter
    # then, the token being no special of the engine's), is a model that would convert and then write nonsense
    tokens = {"bos_token": config_token(said_config, "bos_token"), "eos_token": config_token(said_config, "eos_token")}
    for id, text in (vocabulary.dropped or {}).items():
        uses = [("every text begins with it", ("bos_token_id" in config and id == own) or (bool(text) and text == tokens["bos_token"])),
                ("the answers stop at it", id in (eos if isinstance(eos, list) else [eos]) or (bool(text) and text == tokens["eos_token"])),
                ("the chat format writes it", bool(text) and (text in (template or "") or text in (end or "")))]
        use = next((use for use, said in uses if said), None)
        if use:
            raise ValueError(f"This tokenizer.json has {text} past the model's vocabulary of "
                             f"{config['vocab_size']}, and {use}.")
    bos = bos if head is None else head
    # the special tokens the template writes stand for their token; spelled out they would be a dozen tokens each.
    # The added tokens that are not special, wherever they are written (T.143). The longest first, so that one that
    # begins another never cuts it short
    # (and the one the format began with, T.264: it is the BOS now, and still one token where a visitor types it)
    written = {special for special in specials if special and template and (special in template or special == mark)}
    written = sorted(written | {token for token in added if token}, key=lambda token: (-len(token), token))
    return Beginning(bos if isinstance(bos, int) else 1, stop, template, written)


def options_of(form, dtype, config, vocabulary, beginning):
    """What the engine gets (Llama()'s options) of a model converted: the stream's form and the dtype it stores,
    config.json (as the conversion reads it), the Vocabulary's options and the Beginning."""
    options = {**vocabulary.options, "dtype": dtype, "rope_theta": float(config.get("rope_theta", 10000.0)),
               "bos": beginning.bos, "stop_tokens": beginning.stop_tokens}
    # the form: bias and arch always (as since T.64 and T.65), the rest only where it is not the default, so that
    # the options of every model before T.124 stay what they were (convert/version.py's CONVERTER)
    options.update({key: value for key, value in form.items() if key in ("bias", "arch") or value != FORM[key]})
    if beginning.template:
        options["template"] = beginning.template
    if beginning.specials:
        options["specials"] = beginning.specials
    eps = config.get("rms_norm_eps")
    # a GGUF says it in float32 (1e-5 is 9.99999974e-06 there): six digits are what config.json writes
    eps = float(f"{eps:.6g}") if isinstance(eps, (int, float)) and eps > 0 else RMS_EPS
    family = family_of(config)
    if family.rms_norm and eps != RMS_EPS:
        # T.124: the epsilon of RMSNorm, where it is not the engine's 1e-5 (Qwen2.5 and Qwen3: 1e-6, which moved
        # Qwen3 0.6B's perplexity by 0.12%). Only where it differs, like qk_norm and head_dim
        options["rms_norm_eps"] = float(eps)
    if config.get("rope_scaling"):
        # the int8 file has no RoPE tables: the engine makes them, and needs the scaling for that (Llama 3)
        options["rope_scaling"] = dict(config["rope_scaling"])
    if family.partly:
        # GPT-NeoX and Qwen3.5 turn part of every head: the file does not say how much
        options["rotary"] = rotary_dim(config)
    # and what only its family says (T.255: a SmolLM3's layers that RoPE leaves alone; a GPT-NeoX's branches)
    options.update(family.options(config))
    return options
