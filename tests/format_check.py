# format_check.py
# The format of every Hugging Face model of the list against the real one (T124, T127, T138): the IDs the page
# sends ([bos] + llama2_numpy.Tokenizer on the converter's tokenizer.bin, with the options the worker merges and
# filled()'s rules) against transformers' apply_chat_template on the files of the pinned revision. Only config.json,
# the tokenizer files and the heads of the weights (by Range) are fetched; no weights. The conversion is the page's:
# the conduct of a conversion (src/python/convert/conduct.py) answered until it asks for the weights (T374.4,
# tests/conducting.py's Heads).
#
#   python3 tests/format_check.py [--hf] [--prompt <text> ...] <directory for the downloads> [model id ...]
#
# --prompt: one more prompt after PROMPTS (T221: what a visitor types, such as Qwen3's <tool_call>), as often as given.
# --hf: as ?hf= opens the repository, with what the converter makes alone (T143): the list's options and format are
# left out. A difference known for the list's format is then no error when it is gone. Each line then says for how many
# prompts the ids are the real ones exactly, with no BOS of the page's in front (T264).
#
# Needs the reference tools, which the page never uses: a venv with tests/requirements-reference.txt (docs/notes/dev-setup.md).
# The IDs must be the real ones exactly; one more BOS in front of them (the page always starts with one, T131) is let
# by only for LENIENT's entries, where what it does was measured, and UNMEASURED's, said on their line (T312). The
# other differences known are in KNOWN, each
# taking out only its own difference, and anything else makes the exit status 1 (T145). What a card always passes besides the prompt
# (Swallow-MS's and llm-jp's system message) is in SYSTEM. Models with a sentencepiece tokenizer.model are compared
# through transformers' slow tokenizer, which is not the real one for every model: for those of SENTENCEPIECE the
# reference is the real template's text through the real sentencepiece instead.
import inspect
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
import conducting  # noqa: E402
import llama2_numpy  # noqa: E402
from convert.conduct import candidates_of  # noqa: E402

PROMPTS = ["これからの流行りを3つ挙げてください。", "What will be popular next? Name three things.",
           "  leading and trailing spaces  ", "trailing only \n", "改行\nを含む\n\n文", "A", "ＡＢＣ１２３ｶﾀｶﾅ①",
           "Hello  world\ttab", "<think> and <|im_end|> typed"]
# the system message a model card always passes with the prompt. llm-jp's instruct models: the page's format is the one
# their template writes with it (T144; without it the real one leaves out the first sentence of LLM_JP_INSTRUCT)
LLM_JP_SYSTEM = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
SYSTEM = {"hf-swallow-ms-7b-instruct": "あなたは誠実で優秀な日本人のアシスタントです。",
          # T250: the system message of ELYZA's card
          "hf-llama-3-elyza-jp-8b": "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。",
          **{id: LLM_JP_SYSTEM for id in ("hf-llm-jp-3-150m-instruct3", "hf-llm-jp-3-440m-instruct3",
                                          "hf-llm-jp-3-980m-instruct3", "hf-llm-jp-3.1-1.8b-instruct4",
                                          "hf-llm-jp-3-1.8b-instruct3", "hf-llm-jp-3-3.7b-instruct3",
                                          "hf-llm-jp-3-7.2b-instruct3", "hf-llm-jp-3.1-13b-instruct4",
                                          # T607.7: AXCXEPT's EZO of llm-jp-3 3.7B, whose cards pass it as well
                                          "hf-llm-jp-3-3.7b-instruct-ezo-common", "hf-llm-jp-3-3.7b-instruct-ezo-humanities")},
          # T596.1: the two system messages of the card, which name the direction
          "hf-lfm2-350m-enjp-mt-to-japanese": "Translate to Japanese.", "hf-lfm2-350m-enjp-mt-to-english": "Translate to English."}
# where transformers builds a slow tokenizer out of the tokenizer.model that is not the real one (2026-09-26,
# transformers 5.16.1: sarashina's split "挙げてください" and "next", CAT-Translate's spell everything a character at a time
# although they have a tokenizer.json): the real sentencepiece reads the real template's text instead (T144)
SENTENCEPIECE = {"hf-sarashina2.2-0.5b-instruct", "hf-sarashina2.2-1b-instruct", "hf-sarashina2.2-3b-instruct",
                 "hf-cat-translate-0.8b", "hf-cat-translate-1.4b", "hf-cat-translate-3.3b",
                 # T607.2: Vecteus v1's (a tokenizer.model alone, legacy LlamaTokenizer): transformers' puts no dummy
                 # prefix ("USER" where the real sentencepiece writes "▁US", "ER")
                 "hf-vecteus-v1"}
# T252: where transformers (5.16.1) builds a tokenizer of sentencepiece's kind (a Metaspace pre-tokenizer: "Hello world" is
# "H", "elloworld", Japanese is dropped) for a tokenizer_class LlamaTokenizerFast whose tokenizer.json is byte-level
# (Llama 3's vocabulary under DeepSeek's names): the real template's text through the tokenizers library on
# tokenizer.json itself, which is what a LlamaTokenizerFast was when the model came out
# T610.2: also where the tokenizer_class is GPT2Tokenizer and transformers (5.16.1) builds it out of the vocabulary and the
# merges with GPT-2's own split, not tokenizer.json's (Phi-4's, as tiktoken's cl100k): "\n\n" before a letter is then two
# pieces where tokenizer.json, and the page, has one
TOKENIZERS = {"hf-deepseek-r1-llama-8b", "hf-shisa-v2.1-unphi4-14b"}
# T145: the differences known, each with what it takes out of the comparison (the rest still counts, so a new break
# in these models shows: two of them were 0/9 whatever else went wrong). One that no prompt needs any more is an error
# too, to be taken out of here.
KNOWN = {
    # the real tokenizer.json puts no dummy space before the first piece ("A"), the page does ("▁A"): T131, refused
    "hf-rakutenai-2.0-mini-instruct": {"why": "no dummy space before the first piece (T131)", "first_piece": True},
    # T610.2: the real tokenizer's special tokens strip the spaces beside them (lstrip, rstrip); the list's format trims
    # the prompt, which leaves only the spaces beside a special token typed in it, kept by the page
    "hf-shisa-v2.1-unphi4-14b": {"why": "spaces beside a special token typed in the prompt (T610.2)", "unstripped": True},
    # the page asks for the answer's channel at the end of the format (T132): the real text with it is the page's
    "hf-llm-jp-4-8b-instruct": {"why": "the page ends with <|channel|>final<|message|> (T132)",
                                "text": lambda text, prompt: text + "<|channel|>final<|message|>"},
    "hf-llm-jp-4-8b-thinking": {"why": "the page ends with <|channel|>analysis<|message|>, the thought (T592.2)",
                                "text": lambda text, prompt: text + "<|channel|>analysis<|message|>"},
    "hf-llm-jp-4.1-8b-thinking": {"why": "the page ends with <|channel|>analysis<|message|>, the thought (T304)",
                                  "text": lambda text, prompt: text + "<|channel|>analysis<|message|>"},
    # the list's own ChatML has no system turn; the real template writes its default one
    "hf-smollm2-135m-instruct": {"why": "the list's ChatML, without the default system turn",
                                 "text": lambda text, prompt: re.sub(r"^<\|im_start\|>system\n.*?<\|im_end\|>\n", "", text, flags=re.S)},
    # tokenizer.model's NFKC where the real tokenizer.json has none (the page follows the former), and the template
    # strips the whole turn, which trims only the end of what was typed where the page trims both ends (T138). T192:
    # the real text with those two done to it must be the page's, so that another break in those prompts still shows
    # (the two prompts were taken out whole before)
    "hf-swallow-ms-7b-instruct": {"why": "NFKC of tokenizer.model, and the turn stripped as a whole (T138)",
                                  "text": lambda text, prompt: unicodedata.normalize("NFKC", trimmed(text, prompt))},
    # T249: the same of EuroLLM. Its vocabulary has no full-width letters: the real tokenizer.json spells them as
    # bytes, the page (tokenizer.model's NFKC) writes the half-width ones. The page is the real sentencepiece's (the
    # review: tokenizer.model's normalizer_spec is nfkc with a charsmap of 237,562 bytes, and its ids are the page's for
    # 71,648 sentences "a" + one character + "b" and 3,000 random ones of mixed scripts, 0 apart), the conversion to
    # tokenizer.json keeps only a Prepend and a Replace of " " by "▁": which of the two the model was trained on is the
    # sentencepiece model's, the one its authors trained
    "hf-eurollm-1.7b-instruct": {"why": "NFKC of tokenizer.model (T249)",
                                 "text": lambda text, prompt: unicodedata.normalize("NFKC", text)},
}
# T312: the page's IDs must be the real ones exactly. One more BOS in front of the real IDs (the page always starts
# with a token, T131) passes only for the entries below, each where what that BOS does was measured and found small
# (tests/start_check.py or .ts on plain text, tests/chat_nll.py on answers; the records in docs/tasks/). Before T312
# it passed everywhere but in a list of families found to lose by it after they were in the list (T236's Qwen3.5,
# Granite 4.2, MiniCPM5, Ternary Bonsai 2, Hermes 3, the 8B Qwen3s: up to +144% on plain text). A new entry that sends
# one more token in front now fails until it is measured and put here. A key is an entry's id, or, ending with "-", a
# family measured on some of its sizes (only Qwen2.5's: every hf-qwen2.5- entry the list gets is let by, measured or not)
LENIENT = {
    # T131: Qwen2.5 0.5B Instruct 23.30 with <|endoftext|>, 23.43 without (Japanese), 19.25 and 19.29 (English).
    # T252: Qwen2.5 Coder 1.5B, 3B, 7B, the answers +0.5%, +0.4%, +0.2% (KL 0.0025, 0.0009, 0.0012)
    "hf-qwen2.5-": "Qwen2.5 and Qwen2.5 Coder: within 3% on plain text, +0.5% at most on answers (T131, T252)",
    # T593: ELYZA-Shortcut 1.0 Qwen 7B (a Qwen2.5 7B) +0.88% on plain text
    "hf-elyza-shortcut-1.0-qwen-7b": "+0.88% on plain text (T593)",
    # T602: TinySwallow 1.5B (the base) -1.43% on plain text; the instruct is the same model's distillation
    "hf-tinyswallow-1.5b-instruct": "the base -1.43% on plain text (T602)",
    # T250's review: the Qwen3s of 4B and below +5.4% at most on plain text (<|im_start|> in front is 5.5 to 10.2%
    # worse), 0.6B +4.0% and 1.7B +1.5% on their own answers. Not the 8B ones (+144%: they begin at <|im_start|>)
    **{f"hf-qwen3-{size}{form}": "the Qwen3s of 4B and below: +5.4% at most on plain text (T250's review)"
       for size in ("0.6b", "1.7b", "4b") for form in ("", "-thinking")},
    # (T250's review: the 2507s were measured on their own, Instruct -1.2%, Thinking +0.3% on plain text)
    "hf-qwen3-4b-instruct-2507": "-1.2% on plain text (T250's review)",
    "hf-qwen3-4b-thinking-2507": "+0.3% on plain text (T250's review)",
    # T250's review (and T236: the 1.7B's answers KL 0.008): Ternary Bonsai 1.7B -1.5%, 4B -1.5% on plain text
    "hf-ternary-bonsai-1.7b": "-1.5% on plain text (T250's review, T236)",
    "hf-ternary-bonsai-4b": "-1.5% on plain text (T250's review)",
    # T286.2, T286.3: the 1-bit Bonsai 1.7B +0.29%, 4B +0.50% on plain text
    "hf-bonsai-1.7b": "+0.29% on plain text (T286.2)", "hf-bonsai-4b": "+0.50% on plain text (T286.3)",
    # T250's review: Qwen3 Swallow 8B RL +3.3% on plain text; T606: the SFT +0.98%, the CPT +1.13%, the medical +0.33%
    **{f"hf-qwen3-swallow-8b{kind}{form}": "+3.3% at most on plain text (T250's review, T606)"
       for kind in ("", "-sft", "-cpt") for form in ("", "-thinking")},
    **{f"hf-medical-qwen3-swallow-8b{form}": "+0.33% on plain text (T606)" for form in ("", "-thinking")},
    # T250's review: CAT-Thinking 8B +3.5% on plain text, +0.4% on its own answers (KL 0.0045)
    "hf-cat-thinking-8b": "+0.4% on its answers (T250's review)",
    # T517: LINE's instruction-sft, 1.7B </s> better than none on plain text and -5.76% on the answers, 3.6B <s> chosen
    **{f"hf-japanese-large-lm-{size}-instruction-sft": "the BOS measured and chosen per size (T517)"
       for size in ("1.7b", "3.6b")},
    # T304, T604: llm-jp-4.1 8B thinking -7.08%, Lumi v2 -1.90% on plain text (T399: llm-jp-4 8B thinking was
    # -0.74% here, +9.88% on handwritten answers, and begins at the real <|start|> since)
    "hf-llm-jp-4.1-8b-thinking": "-7.08% on plain text (T304)",
    "hf-lumi-v2-llm-jp-4-8b": "-1.90% on plain text (T604)",
    # T599, T601: calm2 +1.11%, youri +1.16% on plain text
    "hf-calm2-7b-chat": "+1.11% on plain text (T599)", "hf-youri-7b-chat": "+1.16% on plain text (T601)",
    # T600: RakutenAI 7B instruct +0.62%, StableLM instruct gamma +0.01% on plain text
    "hf-rakutenai-7b-instruct": "+0.62% on plain text (T600)",
    "hf-japanese-stablelm-instruct-gamma-7b": "+0.01% on plain text (T600)",
    # T602: EvoLLM-JP-A -0.21%, v1 +0.74%; T605: ELYZA's Llama 2 instruct +0.55%, fast -0.15% on plain text
    "hf-evollm-jp-a-v1-7b": "-0.21% on plain text (T602)", "hf-evollm-jp-v1-7b": "+0.74% on plain text (T602)",
    # (their card writes the BOS itself, left out of ELYZA_LLAMA2_CARD below)
    "hf-elyza-japanese-llama-2-7b-instruct": "the card writes it; +0.55% on plain text (T605)",
    "hf-elyza-japanese-llama-2-7b-fast-instruct": "the card writes it; -0.15% on plain text (T605)",
    # T607.1 to T607.4: ArrowPro +1.13%, Vecteus +0.24%, Oumuamua +1.10%, Gleipnir -0.51% on plain text
    "hf-arrowpro-7b-kujira": "+1.13% on plain text (T607.1)", "hf-vecteus-v1": "+0.24% on plain text (T607.2)",
    "hf-oumuamua-7b-instruct-v2": "+1.10% on plain text (T607.3)", "hf-umievo-gleipnir-7b": "-0.51% on plain text (T607.4)",
    # T399: on handwritten answers (tests/chat_nll.py), the real IDs against the entry's with one more BOS in front:
    # EuroLLM -7.76% and zephyr -11.80% (the BOS better), RakutenAI chat +1.02% and TinyLlama +0.51% (within the SE)
    "hf-eurollm-1.7b-instruct": "-7.76% on handwritten answers (T399)",
    "hf-zephyr-7b-beta": "-11.80% on handwritten answers (T399)",
    "hf-rakutenai-7b-chat": "+1.02% on handwritten answers; its card adds the BOS (T399, T125)",
    "hf-tinyllama-1.1b-chat": "+0.51% on handwritten answers (T399)",
}
# T312: the entries that send one more token in front of the real IDs with nothing measured of it. Let by, so that the
# check is still of the rest of their IDs, and said on their line. Take one out of here when it is measured: to
# LENIENT, or its entry made exact. Empty since T399 (llm-jp-4 8B instruct, the last, was measured and made exact:
# the owner's choice, 2026-11-04); an entry added that is not measured goes here
UNMEASURED = {}
# T233: what an entry's format is the real template's with, besides enable_thinking (or that too, T597.1): Ternary Bonsai 2 27B's entry that
# thinks is its reasoning_effort "medium" (no system turn: T236's format of a Qwen3.5), not its default, "xhigh"
# (T364.3: and Underdog Saluki 27B's, whose template is the same one)
TEMPLATE_SAYS = {"hf-ternary-bonsai-2-27b-thinking": {"reasoning_effort": "medium"},
                 "hf-underdog-saluki-27b-thinking": {"reasoning_effort": "medium"},
                 # T597.1: the one form of an entry whose card says to answer at once (its name says no form)
                 "hf-spark-x2.5-4b-japanese": {"enable_thinking": False},
                 # T649.3: PLaMo 3 Translate's request in the direction of each entry, the source's language said
                 **{f"hf-plamo-3-translate-{size}-{source}-{target}": {"source_lang": source, "target_lang": target}
                    for size in ("0.8b", "2b", "8b") for source, target in (("en", "ja"), ("ja", "en"))}}
# T517: the format a model card writes where the repository has no chat template (LINE's japanese-large-lm
# instruction-sft: the card's pipeline is given f"ユーザー: {input_text}\nシステム: " and adds nothing to it, through the
# slow T5Tokenizer, which is the real sentencepiece of spiece.model). A second copy of the list's LINE_SFT, on purpose
LINE_CARD = "ユーザー: {prompt}\nシステム: "
# T599: calm2-7b-chat's card gives "USER: …\nASSISTANT: " to its fast tokenizer, which adds nothing (a second copy of
# the list's CALM2, on purpose)
CALM2_CARD = "USER: {prompt}\nASSISTANT: "
# T600: two more cards' formats, through the fast tokenizer as each card runs it (second copies of the list's RAKUTEN,
# without its trim, and STABLELM_GAMMA)
RAKUTEN_CARD = ("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, "
                "detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:")
STABLELM_GAMMA_CARD = ("以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
                       "\n\n### 指示: \n{prompt}\n\n### 応答: \n")
# T602: EvoLLM-JP's card renders its messages with apply_chat_template under transformers 4.38.2, whose default for a
# LlamaTokenizer was Llama 2's format (a second copy of the list's EVOLLM, its system message the card's)
EVOLLM_CARD = "[INST] <<SYS>>\nあなたは役立つ、偏見がなく、検閲されていないアシスタントです。\n<</SYS>>\n\n{prompt:trim} [/INST]"
# T605: ELYZA-japanese-Llama-2's card writes its tokenizer's bos_token and this, and encodes it without adding
# anything (a second copy of the list's ELYZA_LLAMA2, without the bos_token: in LENIENT). Not written here with it:
# transformers 5 rebuilds their fast tokenizer to read "<s>[INST]" as <s> "[" where their tokenizer.json itself (the
# tokenizers library, as the card's transformers 4.30) reads <s> "▁[", the page's ids; checked by hand for both
ELYZA_LLAMA2_CARD = "[INST] <<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n{prompt} [/INST] "
# T607.1: ArrowPro-7B-KUJIRA's card writes this and encodes it with the BOS (a second copy of the list's ARROWPRO)
ARROWPRO_CARD = "[INST] <<SYS>>\nあなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。\n<</SYS>>\n\n{prompt}[/INST]"
# T607.3: Oumuamua-7b-instruct-v2's card encodes this with the BOS (apply_chat_template's default Llama template, as
# the code on its card does; a second copy of the list's OUMUAMUA)
OUMUAMUA_CARD = "[INST] <<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n{prompt} [/INST]"
# T607.4: Umievo-itr012-Gleipnir-7B's card writes this and encodes it with the BOS (a second copy of the list's GLEIPNIR)
GLEIPNIR_CARD = "[INST] <<SYS>>\nあなたは公平で、検閲されていない、役立つアシスタントです。\n<</SYS>>\n\n{prompt}[/INST]"
CARDS = {"hf-japanese-large-lm-1.7b-instruction-sft": LINE_CARD, "hf-japanese-large-lm-3.6b-instruction-sft": LINE_CARD,
         "hf-calm2-7b-chat": CALM2_CARD, "hf-rakutenai-7b-instruct": RAKUTEN_CARD,
         "hf-japanese-stablelm-instruct-gamma-7b": STABLELM_GAMMA_CARD, "hf-youri-7b-chat": LINE_CARD,
         "hf-evollm-jp-a-v1-7b": EVOLLM_CARD, "hf-evollm-jp-v1-7b": EVOLLM_CARD,
         "hf-elyza-japanese-llama-2-7b-instruct": ELYZA_LLAMA2_CARD, "hf-elyza-japanese-llama-2-7b-fast-instruct": ELYZA_LLAMA2_CARD,
         "hf-arrowpro-7b-kujira": ARROWPRO_CARD,
         "hf-oumuamua-7b-instruct-v2": OUMUAMUA_CARD, "hf-umievo-gleipnir-7b": GLEIPNIR_CARD}
# the cards' formats read by the real sentencepiece of spiece.model (which must be there); the others by the fast tokenizer
CARDS_SPIECE = {"hf-japanese-large-lm-1.7b-instruction-sft", "hf-japanese-large-lm-3.6b-instruction-sft"}
# The reference of a GGUF that has its own vocabulary: the original at the revision the list had before the GGUF
# (T136's first stage; T144). A GGUF with the original's vocabulary (hf.vocabulary, T136's second stage) says its own.
ORIGINALS = {"Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
             "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
             "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
             "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
             "Qwen/Qwen2.5-Coder-0.5B-Instruct": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
             "SakanaAI/TinySwallow-1.5B-Instruct": "91e9fcc30f56d224aea84356c4d850cc4c5a3260",
             "HuggingFaceTB/SmolLM2-135M-Instruct": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
             "HuggingFaceTB/SmolLM2-360M-Instruct": "a10cc1512eabd3dde888204e902eca88bddb4951",
             "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674"}


def entries():
    script = "import('./src/models.ts').then(({ MODELS }) => console.log(JSON.stringify(MODELS.filter((m) => m.hf))))"
    return json.loads(subprocess.check_output(["node", "-e", script], cwd=HERE.parent))


def gguf(entry):
    return entry["hf"]["weights"].endswith(".gguf")


def converted(entry, directory, dtype="int8"):
    """(the folder of the files transformers' reference reads, the conversion of the entry as the page makes it: its
    options, its tokenizer and its header, none of its weights).
    The conversion: by its conduct, which is answered with the small files and the heads of the weights (kept under
    <directory>/<owner>__<name>/<revision>, a file the repository does not have marked <name>.missing) until it asks
    for the weights themselves. What it writes goes nowhere.
    The reference's files: of the repository the vocabulary comes from; of the entry's own where it names none; and
    for a GGUF that has its own vocabulary, of the original at ORIGINALS' revision. tokenizer.json and
    special_tokens_map.json too: transformers reads them where they are. dtype: what the conversion makes (the ids are
    the same whatever it is; tests/write_options.py asks for the one the page keeps the model in)."""
    hf = entry["hf"]
    # (T610.1: an entry whose blocks are kept as the page converts it, so its options say the rows' kinds)
    blocks = {"blocks": True} if entry.get("conversion", {}).get("blocks") else {}
    made = conducting.converted(conducting.Heads.of(hf, directory, between="__", remember=True), hf, weights=False,
                                dtype=dtype, sink=conducting.Nothing(), **blocks)
    original = hf.get("vocabulary") or ({"repo": entry["original"], "revision": ORIGINALS[entry["original"]]} if gguf(entry) else hf)
    reference = conducting.Fetched.of(original, directory, between="__", remember=True)
    for name in dict.fromkeys(["config.json", "tokenizer_config.json", "chat_template.jinja", "tokenizer.json",
                               "special_tokens_map.json", *candidates_of(hf)]):
        reference.path("weights", name)
    # T649.3: a tokenizer of the repository's own code (auto_map: PLaMo's tokenization_plamo.py) and its file
    module = own_code(reference.folder("weights"))
    if module:
        reference.path("weights", f"{module}.py")
    return reference.folder("weights"), made


def own_code(folder):
    """The module of a tokenizer that the repository writes itself (tokenizer_config.json's auto_map), where it has no
    tokenizer.json for transformers to read without it, or None."""
    config = folder / "tokenizer_config.json"
    if not config.exists() or (folder / "tokenizer.json").exists():
        return None
    named = (json.loads(config.read_text()).get("auto_map") or {}).get("AutoTokenizer")
    return named[0].split(".")[0] if named and named[0] else None


def reference_of(folder):
    """transformers' tokenizer of the folder. One of the repository's own code (T649.3: PLaMo's Plamo3Tokenizer) is
    its class loaded from the file and read by its own from_pretrained, as AutoTokenizer(trust_remote_code=True)
    would, without the copy of the code transformers keeps under $HOME. PLaMo's imports numba, which the venv has not:
    PYTHONPATH names a folder of .tmp that has it (tests/plamo_tokenizer_check.py says how)."""
    from transformers import AutoTokenizer
    module = own_code(folder)
    if not module:
        return AutoTokenizer.from_pretrained(folder)
    import importlib.util
    named = json.loads((folder / "tokenizer_config.json").read_text())["auto_map"]["AutoTokenizer"][0]
    spec = importlib.util.spec_from_file_location(module, folder / f"{module}.py")
    code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code)
    return getattr(code, named.split(".")[1]).from_pretrained(folder)


def sentencepiece_ids(model_file, text):
    """text as the real sentencepiece reads it where a tokenizer.json cuts it at the special tokens: the control
    pieces (<s>, </s>, <|user|> ...) stand for themselves, and every stretch between them is encoded (a user-defined
    piece sentencepiece keeps whole by itself)"""
    import sentencepiece
    model = sentencepiece.SentencePieceProcessor(model_file=str(model_file))
    specials = sorted({model.id_to_piece(i) for i in range(model.get_piece_size())
                       if model.is_control(i)} - {""}, key=len, reverse=True)
    ids, at = [], 0
    for found in re.finditer("|".join(map(re.escape, specials)), text):
        ids += model.encode(text[at:found.start()]) if found.start() > at else []
        ids.append(model.piece_to_id(found.group()))
        at = found.end()
    return ids + (model.encode(text[at:]) if at < len(text) else [])


def tokenizers_ids(tokenizer_file, text, strip=True):
    """text as the tokenizers library reads it with tokenizer.json alone (nothing added: the template wrote the BOS);
    strip=False: its added tokens strip no spaces beside them, as the page's tokenizer (T610.2)"""
    import tokenizers
    if strip:
        return tokenizers.Tokenizer.from_file(str(tokenizer_file)).encode(text, add_special_tokens=False).ids
    read = json.loads(Path(tokenizer_file).read_text())
    for token in read["added_tokens"]:
        token["lstrip"] = token["rstrip"] = False
    return tokenizers.Tokenizer.from_str(json.dumps(read)).encode(text, add_special_tokens=False).ids


def filled(template, prompt):
    """src/models.ts's filled(), for today"""
    template = re.sub(r"\{date(?::([^}]*))?\}", lambda found: time.strftime(found.group(1) or "%Y-%m-%d"), template)
    return re.sub(r"\{prompt(:trim)?\}", lambda found: prompt.strip() if found.group(1) else prompt, template, count=1)


def main():
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    arguments = sys.argv[1:]
    alone = "--hf" in arguments  # T143: the converter's options and format only, as ?hf= has them
    arguments = [argument for argument in arguments if argument != "--hf"]
    prompts = list(PROMPTS)
    while "--prompt" in arguments:
        at = arguments.index("--prompt")
        prompts.append(arguments[at + 1])
        del arguments[at:at + 2]
    directory, only = Path(arguments[0]), arguments[1:]
    accepted = set(inspect.signature(llama2_numpy.Tokenizer.__init__).parameters) - {"self", "data", "vocab_size", "kind"}
    failed, stale = [], []
    for entry in entries():
        if only and entry["id"] not in only:
            continue
        folder, made = converted(entry, directory)
        options = {**made.options, **({} if alone else entry.get("options", {}))}
        template = (None if alone else entry.get("template")) or made.options.get("template")
        # (T312: a base model's entry whose format is what was typed, where its tokenizer has its instruct's template.
        # Whether the entry is a base model is not looked at: any entry whose format is "{prompt}" alone reads so)
        if not template or template == "{prompt}":
            print(f"none {entry['id']}: no format (what was typed is continued as it is)", flush=True)
            continue
        tokenizer = llama2_numpy.Tokenizer(made.tokenizer, abs(made.stream.header[5]), kind=options["tokenizer_kind"],
                                           **{key: value for key, value in options.items() if key in accepted})
        reference = reference_of(folder)
        # T236: said either way. A Qwen3's template thinks unless told not to, a Qwen3.5 0.8B's only when told to
        # (--hf, T253's review: what ?hf= gets is the template with nothing said of thinking, so the real one is asked with
        # nothing said either: a "(no thinking)" entry was held to enable_thinking=False in this mode, a form of the list's
        # own that the converter alone cannot know, and MiniCPM5's two sizes were 0/9 for that and nothing else)
        thinking = {} if alone else {"enable_thinking": False} if "(no thinking)" in entry["name"] else \
            {"enable_thinking": True} if "(thinking)" in entry["name"] else {}
        thinking = {**thinking, **({} if alone else TEMPLATE_SAYS.get(entry["id"], {}))}
        known = KNOWN.get(entry["id"], {})
        same, explained, diffs, exact, fronts = 0, 0, [], 0, set()
        for prompt in prompts:
            page = [options["bos"]] + tokenizer.encode(filled(template, prompt), tuple(options.get("specials", ())))
            messages = ([{"role": "system", "content": SYSTEM[entry["id"]]}] if entry["id"] in SYSTEM else []) \
                + [{"role": "user", "content": prompt}]
            # (T517: no template to render where the card writes the format: the card's text, through the real sentencepiece)
            text = filled(CARDS[entry["id"]], prompt) if entry["id"] in CARDS else \
                reference.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **thinking)
            if entry["id"] in SENTENCEPIECE or entry["id"] in CARDS_SPIECE:
                model = "spiece.model" if entry["id"] in CARDS_SPIECE else "tokenizer.model"
                encoded = lambda text: sentencepiece_ids(folder / model, text)
                real = encoded(text)
            elif entry["id"] in CARDS:
                encoded = lambda text: list(reference(text, add_special_tokens=False)["input_ids"])
                real = encoded(text)
            elif entry["id"] in TOKENIZERS:
                encoded = lambda text: tokenizers_ids(folder / "tokenizer.json", text)
                real = encoded(text)
            else:
                # a text of KNOWN's as apply_chat_template(tokenize=True) encodes the text it renders
                encoded = lambda text: list(reference(text, add_special_tokens=False)["input_ids"])
                real = reference.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **thinking)
                real = list(real["input_ids"] if hasattr(real, "keys") else real)
            def matches(real, page=page, prompt=prompt):
                # (T312: the prompts that match only with one more BOS in front, said on the entry's line)
                if page != real and ids_match(entry["id"], page, real, alone):
                    fronts.add(prompt)
                return ids_match(entry["id"], page, real, alone)
            exact += page == real
            if matches(real):
                same += 1
            elif "text" in known and matches(encoded(known["text"](text, prompt))):
                explained += 1
            elif known.get("unstripped") and matches(tokenizers_ids(folder / "tokenizer.json", text, strip=False)):
                explained += 1
            elif known.get("first_piece") and first_piece(reference.convert_ids_to_tokens(page[1:]),
                                                          reference.convert_ids_to_tokens(real)):
                explained += 1
            else:
                diffs.append(f"{prompt!r}\n    page {reference.convert_ids_to_tokens(page)[:60]}"
                             f"\n    real {reference.convert_ids_to_tokens(real)[:60]}")
        where = "the list" if entry.get("template") and not alone else "the converter"
        verdict = "DIFF" if diffs else "known" if explained else "ok  "
        note = f", {explained} as known: {known['why']}" if explained else ""
        # T264 (T369): what ?hf= sends for a template the converter read is the real ids, none in front, where the
        # template begins with a token (said, not required: one that begins with text keeps the BOS in front)
        note += f", {exact} of them the real ids exactly" if alone else ""
        # T312: how many have one more BOS in front, and why that is let by (UNMEASURED's: not measured yet)
        why = None if alone else lenient(entry["id"])
        before = len(fronts)
        note += (f", {before} with one more BOS in front"
                 f"{' NOT MEASURED' if entry['id'] in UNMEASURED else ''}: {why}") if before and why else ""
        print(f"{verdict} {entry['id']}: {same}/{len(prompts)}{note} (format of {where})", flush=True)
        for diff in diffs[:3]:
            print("  ", diff)
        if diffs:
            failed.append(entry["id"])
        if known and not explained and not alone:
            print(f"STALE {entry['id']}: the difference known ({known['why']}) is gone: take it out of KNOWN")
            stale.append(entry["id"])
    if failed:
        print(f"{len(failed)} differ beyond what KNOWN says: {' '.join(failed)}")
    sys.exit(1 if failed or stale else 0)


def trimmed(text, prompt):
    """The real text with what was typed trimmed at both ends, as the page's {prompt:trim} writes it (T192): the real
    template strips the whole turn, which leaves the start of what was typed as it was."""
    for typed in (prompt, prompt.rstrip(), prompt.lstrip()):
        if typed and typed in text:
            return text.replace(typed, prompt.strip(), 1)
    return text


def same_ids(page, real):
    """Whether the page's IDs are the real ones: the same, or with one more token in front, the BOS the page always
    starts with (T131). Not when that token is the one the real IDs begin with: then the format writes the BOS it was
    given again, as Llama 3's <|begin_of_text|> was written twice before T106, and a BOS set to the format's own first
    token (T236's <|im_start|>) with the format left beginning with it (T250's review: the lenient rule passed it)."""
    return page == real or (page[1:] == real and page[:1] != real[:1])


def ids_match(model_id, page, real, alone=False):
    """Whether the page's IDs are the real ones for this entry: exactly (T312), or as same_ids() says for an entry of
    LENIENT or UNMEASURED. alone (--hf, T269): what ?hf= sends has the converter's BOS in front, never the entry's, so
    there same_ids() is all that can be asked (a Granite 4.2 opened so sends <s> and then the real IDs)."""
    return same_ids(page, real) if alone or lenient(model_id) else page == real


def lenient(model_id):
    """Why one more BOS in front is let by for this entry (T312): LENIENT's or UNMEASURED's word, or None."""
    return next((why for key, why in {**LENIENT, **UNMEASURED}.items()
                 if model_id == key or key.endswith("-") and model_id.startswith(key)), None)


def first_piece(page, real):
    """The same pieces but the first, which the page writes with the dummy space the real one leaves out."""
    return len(page) == len(real) > 0 and page[1:] == real[1:] and page[0] == "▁" + real[0]


if __name__ == "__main__":
    main()
