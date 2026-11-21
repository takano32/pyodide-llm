"""engine/pfml.py (T665): the request of PLaMo 3 Translate, PFML3, put together from its parts: the language to
translate into, the language of the text (or none: the model tells), the form of a Japanese translation, a context,
a glossary, the pairs translated before and the text. The order is the official chat_template.jinja's (the GGUF
repositories' and the original ones'): dataset, context, the pairs, glossary, input, output. It takes the form of
tanakh/plamo-translate-desktop's crate pfml (3d69a116c45a8116159f1a405311a59039ec4b95: none of its lines; the table
of languages is the model card's, as is theirs).

The marks (<|plamo:tag|> ...) are their tokens, and every text between them is read literally (Tokenizer.encode's
literal): a <|plamo:tag|> that a person typed is its characters, where the template refuses such a text and a text
handed to the tokenizer whole would make it the mark. A text without one is the ids the template's text is read as:
PLaMo's tokenizer cuts a text around every <|plamo:…|> and searches each stretch alone (tests/test_pfml.py, and by
hand against the template and PLaMo's own tokenizer: tests/pfml_check.py).

What comes back follows the BOS (<|plamo:bos|>, 1), which the engine puts in front of every prompt itself, as it
does a format's. The values are Python's: a page's arrays would be unwrapped where they come in (engine/proxy.py)."""

# the 53 languages of PLaMo 3 Translate's model card, in its order: the code the request says, its English name and
# its own
LANGUAGES = {
    "ja": ("Japanese", "日本語"), "en": ("English", "English"), "zh-hans": ("Chinese (Simplified)", "简体中文"),
    "zh-hant": ("Chinese (Traditional)", "繁體中文"), "ko": ("Korean", "한국어"), "ar": ("Arabic", "العربية"),
    "az": ("Azerbaijani", "Azərbaycan dili"), "be": ("Belarusian", "Беларуская"), "bg": ("Bulgarian", "Български"),
    "bn": ("Bengali", "বাংলা"), "ca": ("Catalan", "Català"), "ceb": ("Cebuano", "Cebuano"), "de": ("German", "Deutsch"),
    "es": ("Spanish", "Español"), "et": ("Estonian", "Eesti"), "eu": ("Basque", "Euskara"), "fa": ("Persian", "فارسی"),
    "fr": ("French", "Français"), "gu": ("Gujarati", "ગુજરાતી"), "he": ("Hebrew", "עברית"), "hi": ("Hindi", "हिन्दी"),
    "hr": ("Croatian", "Hrvatski"), "hu": ("Hungarian", "Magyar"), "hy": ("Armenian", "Հայերեն"),
    "id": ("Indonesian", "Bahasa Indonesia"), "it": ("Italian", "Italiano"), "jv": ("Javanese", "Basa Jawa"),
    "ka": ("Georgian", "ქართული"), "kk": ("Kazakh", "Қазақ тілі"), "km": ("Khmer", "ខ្មែរ"),
    "lt": ("Lithuanian", "Lietuvių"), "lv": ("Latvian", "Latviešu"), "ml": ("Malayalam", "മലയാളം"),
    "mr": ("Marathi", "मराठी"), "ms": ("Malay", "Bahasa Melayu"), "my": ("Burmese", "မြန်မာ"), "ne": ("Nepali", "नेपाली"),
    "nl": ("Dutch", "Nederlands"), "pa": ("Punjabi", "ਪੰਜਾਬੀ"), "pl": ("Polish", "Polski"), "ru": ("Russian", "Русский"),
    "si": ("Sinhala", "සිංහල"), "sk": ("Slovak", "Slovenčina"), "sl": ("Slovenian", "Slovenščina"),
    "sr": ("Serbian", "Српски"), "sw": ("Swahili", "Kiswahili"), "ta": ("Tamil", "தமிழ்"), "te": ("Telugu", "తెలుగు"),
    "th": ("Thai", "ไทย"), "tl": ("Tagalog", "Tagalog"), "ur": ("Urdu", "اردو"), "uz": ("Uzbek", "Oʻzbekcha"),
    "vi": ("Vietnamese", "Tiếng Việt")}
# easy Japanese, which the card does not list (tanakh's calls it experimental): PFML3 writes it ja-easy (the template
# says so since 2026-11-07, the model repository's 2c6192f7; its first release wrote ja(easy)), and the template
# takes ja(easy) as an alias of it, as this does
EASY_JAPANESE = "ja-easy"
EASY_ALIAS = "ja(easy)"
# the forms of a translation into Japanese (only "ja": not easy Japanese), plain (だ・である) and polite (です・ます)
FORMS = ("plain", "polite")
# the language whose translation may have a form
FORMED = "ja"


def mark(name):
    """the piece of one of PFML3's marks (tag, msg, key, val, index, item)"""
    return f"<|plamo:{name}|>"


def written(value, said, empty=False):
    """refuses what is not a text, or (unless empty) is only whitespace"""
    if not isinstance(value, str):
        raise ValueError(f"{said[0].upper()}{said[1:]} is a text, not {value!r}.")
    if not empty and not value.strip():
        raise ValueError(f"{said[0].upper()}{said[1:]} is empty.")


def pfml_pieces(text, target, source=None, form=None, context=None, glossary=(), history=()):
    """The request as its pieces in order: (True, a mark's piece) or (False, a text). text: what to translate;
    target and source: codes of LANGUAGES (or EASY_JAPANESE, or its alias EASY_ALIAS; source None: not said); form:
    one of FORMS, into Japanese only (None: the model's own choice); context: a text about the whole (None or empty:
    none); glossary: (a term, its translation) pairs; history: (a text, its translation) pairs translated before, the earliest first.
    What the template refuses is refused here too, in its order (the pairs' texts and the text, the languages, the
    context, the form, the glossary), but for a mark written in a text (read literally), and a language the card does
    not list."""
    history, glossary = history or (), glossary or ()  # None: not said, as the template's `is not none`
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in (*history, *glossary)):
        raise ValueError("A pair of the history or of the glossary is two texts.")
    said_texts = [(each, f"translated pair {i + 1}") for i, pair in enumerate(history) for each in pair] + [(text, "the text")]
    glossary_texts = [(each, f"glossary pair {i + 1}") for i, pair in enumerate(glossary) for each in pair]
    for value, said in said_texts:
        written(value, said)
    if target is None:
        raise ValueError("A translation needs the language to translate into.")
    target, source = (EASY_JAPANESE if code == EASY_ALIAS else code for code in (target, source))
    for code, said in ((target, "target"), (source, "source")):
        if code is not None and code != EASY_JAPANESE and code not in LANGUAGES:
            raise ValueError(f"The {said} language {code!r} is not one PLaMo 3 Translate knows "
                             f"({', '.join(LANGUAGES)} or {EASY_JAPANESE}).")
    if context is not None:
        written(context, "the context", empty=True)
    if form is not None and (form not in FORMS or target != FORMED):
        raise ValueError(f"A form is one of {', '.join(FORMS)}, and only a translation into {FORMED} has one.")
    for value, said in glossary_texts:
        written(value, said)
    tag, msg, key, val = mark("tag"), mark("msg"), mark("key"), mark("val")

    def element(name, body=None, **attributes):
        said = [(True, tag), (False, name)]
        for attribute, value in attributes.items():
            said += [(True, key), (False, attribute), (True, val), (False, value)]
        return said + [(True, msg)] + ([(False, body)] if body else [])

    source_said = {"lang": source} if source is not None else {}
    pieces = element("dataset", "translation")
    if context:
        pieces += element("context", context)
    for said, translation in history:
        pieces += element("input", said, **source_said) + element("output", translation, lang=target)
    if glossary:
        pieces += [(True, tag), (False, "glossary")]
        for term, translation in glossary:
            pieces += [(True, mark("index")), (False, term), (True, mark("item")), (False, translation)]
    pieces += element("input", text, **source_said)
    return pieces + element("output", lang=target, **({"form": form} if form is not None else {}))


def pfml_tokens(tokenizer, text, target, **parts):
    """The request's ids (after the BOS) in the tokenizer of a PLaMo 3 (a "plamo" one): each mark its token, each text
    read literally. parts: pfml_pieces()'s."""
    tokens = []
    for marked, piece in pfml_pieces(text, target, **parts):
        if marked:
            token = tokenizer.index.get(piece.encode("utf-8"))
            if token is None:
                raise ValueError(f"This vocabulary has no {piece}: not PLaMo's.")
            tokens.append(token)
        else:
            tokens += tokenizer.encode(piece, literal=True)
    return tokens
