# convert/version.py (T.628.4): the version of what the converter makes.
# T.116: the version of what llama2_convert.py writes, the checkpoint's bytes and the options the engine gets.
# Raise it when either changes: a conversion kept by an older converter is then converted again, and deleted (the
# options of T.106, BOS and specials, stayed wrong in what was kept before). 1: the manifests without it. 3: T.127,
# the templates the converter reads now (selectattr, namespace(), chat_template.jinja, trim_blocks) in the options;
# 4: and the control pieces of a sentencepiece model as their specials; 5: the nmt and collapse of a sentencepiece
# model's normalizer (the review of T.126: rinna's newlines); 6: a template's strftime_now() as {date:format}, filled
# when the prompt is sent, and or / and, comments and escapes as Jinja reads them (the review of T.127); 7: the
# rms_norm_eps of a model whose is not 1e-5 (T.124: the Qwen2.5 of the list, TinySwallow, DeepSeek-R1 and llm-jp-4, at 1e-6);
# 8: {prompt:trim} for a template that trims what was typed (T.138: Llama 3.1 and 3.2, Swallow 8B); 9: the batch the
# owner held for one version (2026-09-28): the added tokens that are not special as specials, the BOS the tokenizer
# names, a GGUF's padding as empty pieces, a byte-level tokenizer.bin's own U+2581 (T.143), and a sentencepiece
# model's charsmap after the pieces in place of the nfkc and nmt options (T.216). 10: the round of T.369 (2026-10-10): the
# token a text begins with where the model's chat template was read (T.264: the template's own first token, so that the
# page sends the ids of apply_chat_template), lower case for a sentencepiece model that asks for it (T.265), the dummy
# prefix after a special token (T.308), the templates jinja2 reads and the reader did not (T.397, and T.269's), and
# qk_norm said by a Llama's file alone. 11: a ternary group's scale a float16, PQ2_0's own (T.533: 34 bytes a group of
# 128, not 36; a ternary model's kept conversion of 10 is fetched and converted again).
# (the page reads it before any Python runs, src/runtime/kept.ts's CONVERTER: python_facts.py writes it there)
CONVERTER = 11
