# T.537 note の言語の語とモデルのカード（2026-10-15）

持ち主「English / 中文に、他にも同じような例があればカードに従う」（Spark-X2.5 の note を問われて）。一覧の 144 項目の note の言語の語（「日本語」「English」「中文」「translates 日本語 ⇄ English」「… and 33 more languages」）を、カードと 1 つずつ並べた。

## 見たもの

- カード: 項目の `original`（無ければ語彙のリポジトリ、`hf.repo`、`source`）の HF の API の `cardData.language`。食い違いの候補と、メタデータに言語の無い項目は、README の本文（「Language(s)」「languages」「Japanese」「Multilingual」の行）も。GGUF だけのリポジトリのカード（Spark-X2.5・Saluki）も読んだ。
- 線: **note がカードの言わない言語を言う**、または **カードが言う日本語・英語を note が落とす** ものを直す。カードが 8 つ・29 個・100 を超える言語を言い、note がそのうちの 2 つ（日本語 / English）か 1 つ（English）だけを言うものは、食い違いでないとしてそのまま（並びも同じ）。
- 本文もメタデータも言語を言わないカードは、元のモデル（`base_model`）のカードで見た（NeoHorse-1・Agents-A1 は Qwen3.5、Ternary Bonsai は Qwen3）。

## 直した項目（14）

| id | note の言語の語（前 → 後） | カード | 並び（前 → 後、144 のうち） |
|---|---|---|---|
| `hf-spark-x2.5-1.7b-thinking`・`hf-spark-x2.5-1.7b` | 日本語 / English → **English / 中文** | GGUF のカード（XHToken/Spark-X2.5-1.7B-GGUF）の language が en・zh。原本のカードはメタデータに言語が無く、本文は中国の試験（Gaokao）を並べる | 54・55 → 122・123（英語の側） |
| `hf-spark-x2.5-4b-thinking`・`hf-spark-x2.5-4b` | 同上 | 同上（-4B-GGUF） | 75・76 → 133・134（英語の側） |
| `hf-underdog-saluki-27b-thinking`・`hf-underdog-saluki-27b` | 日本語 / English → **English** | ConwayResearch/Underdog-Saluki-27B-1.0 の language が en だけ。本文は言語を言わない。元の Qwen/Qwen3.8-27B のカードも言語を並べない | 86・87 → 136・137（英語の側、最後） |
| `hf-llm-jp-3-150m-instruct3` | 日本語 → **日本語 / English** | language が en・ja、本文の学習データの表は Japanese・English・Chinese・Korean・code（同じ系統の base の llm-jp-3 150M・440M はもう「日本語 / English」） | 動かない |
| `hf-llm-jp-3-440m-instruct3`・`hf-llm-jp-3-980m-instruct3`・`hf-llm-jp-3-1.8b-instruct3`・`hf-llm-jp-3.1-1.8b-instruct4`・`hf-llm-jp-3-3.7b-instruct3` | 同上 | 同上 | 動かない（1.8B の 2 つと 3.7B は、前の Spark が抜けた分だけ番号が 2〜6 上がる） |
| `hf-sarashina2.2-0.5b`・`hf-sarashina2.2-1b` | 日本語 → **日本語 / English** | language が ja・en、本文「trained the model on 10 trillion tokens, including Japanese, English, and code」（instruct の 3 つは「Language(s): Japanese」で、そのまま） | 動かない |

並びの決まり（T.128、`src/models.js`）: 同じグループの中は note に「日本語」がある項目が先。Spark と Saluki の 6 つは「日本語」が無くなって英語の側に移り、その間の項目（Qwen3.5 2B から NeoHorse-1 9B まで）は 2〜6 つ前に詰まる。全部の動きは下の「動いた項目」。

## 直さなかった項目（130）

| 項目 | note | カード | そのままの訳 |
|---|---|---|---|
| tiny-lm（と f16）・llm-jp-3 150M（と f16）・llm-jp-3 440M・llm-jp-4 8B・Takenoko の Qwen3.5 0.8B SFT 2 つ・LFM2.5 1.2B JP・Shisa の 5 つ・RakutenAI 2 つ・Swallow の 4 つ・ELYZA・CAT-Thinking | 日本語 / English | ja・en | 合う |
| rinna の 5 つ・LINE の 4 つ・stockmark・TinySwallow・sarashina2.2 instruct の 3 つ | 日本語 | ja | 合う |
| CAT-Translate の 3 つ | translates 日本語 ⇄ English | ja・en | 合う |
| LFM2・LFM2.5 の 4 つ（230M・350M・700M・1.2B Instruct） | 日本語 / English | 8〜10 の言語（en・ja を含む） | 線の内（2 つを言う） |
| Granite 4.2 の 4 つ | 日本語 / English | 12 の言語（en・ja を含む） | 線の内 |
| EuroLLM 1.7B | 日本語 / English and 33 more languages | 35 の言語 | 合う |
| Qwen2.5 Instruct の 4 つ | 日本語 / English | メタデータは en だけ、本文「Multilingual support for over 29 languages, including Chinese, English, … Japanese …」 | 本文が日本語を言う |
| Qwen3 の 10 つ・Qwen3.5 の 8 つ | 日本語 / English | メタデータ無し、本文 100 を超える言語（Qwen3）・201 の言語（Qwen3.5） | 本文が言う多言語の内 |
| NeoHorse-1 の 4 つ・Agents-A1 の 2 つ | 日本語 / English | メタデータも本文の言語の行も無い。元の Qwen3.5 は 201 の言語。Agents-A1 の本文は「respond … in the user's language」 | 元のモデルのカードに従う |
| Ternary Bonsai 1.7B・4B・8B | 日本語 / English | メタデータ無し、元の Qwen3 は 100 を超える言語 | 同上 |
| Ternary Bonsai 2 27B の 2 つ | 日本語 / English | Prism のカードも元の Qwen3.8-27B のカードも言語を言わない | 食い違う言葉が無い（迷い 2） |
| Gemma 3 の 2 つ・Gemma 4 の 2 つ | 日本語 / English | 140 を超える言語 | 線の内 |
| MiniCPM5 の 4 つ | English / 中文 | en・zh | 合う |
| Karpathy・Ellis Hoag の TinyStories 5 つ | English | メタデータ無し（TinyStories は英語の物語） | 合う |
| Pythia の 8 つ・GPT-2 の 4 つ・DistilGPT2・SmolLM2 の 5 つ・TinyLlama・Hermes 3・zephyr・Qwen2.5 Coder の 4 つ | English | en | 合う |
| Llama 3.2 1B・3B Instruct・Llama 3.1 8B Instruct | English | 8 の言語（en・de・fr・it・pt・hi・es・th、日本語なし） | English はカードの内（迷い 3） |
| SmolLM3 3B の 2 つ | English | 6 の言語（en・fr・es・it・pt・zh）と ar・ru | 同上 |
| DeepSeek-R1 Distill の 3 つ・Mistral 7B Instruct の 2 つ | English | メタデータ無し、本文も言語を言わない | 食い違う言葉が無い |

## 迷いが残るもの

1. **Saluki を「English」にした。** カードのメタデータは en だけで、本文は言語を言わない。T.364 で日本語の問いに読める日本語を書いた（docs/notes/t364-saluki-2026-10-13.md）が、持ち主の言葉は「カードに従う」。note の文は 2026-10-13 に持ち主が決めた文（`tests/models-check.mjs` が丸ごと持つ）で、その中の言語の語だけを替えた。戻すのは `hf-families.js` の `languages: "English"` を外す 1 か所と試験の 1 行。
2. **Ternary Bonsai 2 27B は「日本語 / English」のまま。** 同じ Qwen3.8-27B から作った Saluki を English にしたのに、Prism のカードは言語を何も言わない（メタデータも無い）ので、食い違いが無いとして残した。Saluki と同じにするなら English。
3. **カードが 3 つ以上の言語を言うものを 1 つ・2 つで言う note**（Llama 3.x の English、SmolLM3 の English、LFM2・Granite・Gemma・Qwen の日本語 / English）は、間違いではないとして残した。EuroLLM の形（「… and 33 more languages」）にそろえるかは持ち主の決め。
4. **Spark-X2.5 の問いの例（prompt と placeholder）は日本語のまま。** English / 中文 の MiniCPM5 は英語の例を持つが、Spark-X2.5 は日本語の問いに日本語で答えた（T.343: 18/18）ので、note の語だけを替えた。

## 動いた項目（一覧の番号、前 → 後）

Spark-X2.5 1.7B の 2 つ 54・55 → 122・123、4B の 2 つ 75・76 → 133・134、Saluki の 2 つ 86・87 → 136・137。その間の項目は前に詰まる: llm-jp-3 1.8B instruct3 56 → 54 から NeoHorse-1 9B 105 → 99 まで（4 つの Spark の分は 2〜4、Saluki の分も足して最大 6）、英語の側の Pythia 14M 106 → 100 から Hermes 3 136 → 132 まで（6 つ）、Pythia 6.9B 137 → 135。グループの頭の 53 項目（stories と日本語の軽いもの）は動かない。
