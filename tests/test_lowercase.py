"""Lower case before the tokenizer reads a text (T265, T369): rinna's japanese-gpt2."""
from made_up_tokenizers import byte_level, converted, sentencepiece


def test_do_lower_case_is_read_for_a_sentencepiece_model():
    """T265: rinna's japanese-gpt2 (tokenizer_config.json's do_lower_case, a vocabulary with no capital Latin letter):
    the text is lowered before the model reads it, a special token is not."""
    data, ids = sentencepiece()
    options, engine = converted(data, "spiece.model", do_lower_case=True)
    assert options["lowercase"] is True
    assert engine.encode("Hello WORLD") == engine.encode("hello world") == [ids["▁hello"], ids["▁world"]]
    assert engine.encode("Hello</s>", ("</s>",)) == [ids["▁hello"], ids["</s>"]]
    plain, as_it_was = converted(data, "spiece.model")
    assert "lowercase" not in plain and ids["▁hello"] not in as_it_was.encode("Hello")
    for said in (False, "true", 1, None):  # (only true says so)
        assert "lowercase" not in converted(data, "spiece.model", do_lower_case=said)[0]


def test_do_lower_case_is_not_read_for_a_tokenizer_json():
    """A tokenizer.json lowers its text with a normalizer of its own or not at all: tokenizer_config.json's key is what
    transformers' slow tokenizers read."""
    tokenizer, _ = byte_level(["<|endoftext|>"])
    assert "lowercase" not in converted(tokenizer, do_lower_case=True)[0]


def test_the_model_reads_as_its_options_say():
    """The options go to Llama() as the worker spreads them, and the model's own tokenizer reads by them: lower case
    (T265) and the dummy prefix after a special token (T308)."""
    import json
    import llama2_convert
    from llama2_numpy import Llama
    from made_up_tokenizers import model
    data, ids = sentencepiece()
    settings, published, header, base = model()
    conversion = llama2_convert.Conversion(header, base, json.dumps(published), data, "spiece.model", dtype="float32",
                                           max_seq_len=settings["seq_len"],
                                           tokenizer_config=json.dumps({"do_lower_case": True, "tokenizer_class": "LlamaTokenizer"}))
    options = conversion.options
    assert options["lowercase"] is True and options["prefixed"] == "wanting"
    llama = Llama(conversion.checkpoint, conversion.tokenizer, **options)
    assert llama.tokenizer.encode("HELLO</s>World", ("</s>",)) == [ids["▁hello"], ids["</s>"], ids["▁world"]]
    plain = Llama(conversion.checkpoint, conversion.tokenizer, **{key: value for key, value in options.items() if key not in ("lowercase", "prefixed")})
    assert plain.tokenizer.encode("hello</s>world", ("</s>",)) == [ids["▁hello"], ids["</s>"], ids["world"]]
    assert ids["▁hello"] not in plain.tokenizer.encode("HELLO")
