# T.511 スマホのユーザ向けに一覧を見直す（小さい ELYZA を含む）（2026-10-15）

調べと表だけ。製品のコードは変えていない。Hugging Face の API（一覧と検索 約 45 回）、カードの README、config.json を読んだ。重みは取っていない。数字は API・カード・config・一覧（`src/models.js` を Node で読んだもの）と台帳（TASKS.md・AGENTS.md・docs/performance.md）から取った。書いていない数は未計測。

## まとめ

- **ELYZA に小さいモデルは無い。** `elyza` の公開は 20 件で、いちばん小さいものも 7B（Llama 2 の 7B、Qwen2.5 7B の Shortcut、Dream 7B の拡散モデル）。ほかの作り手の写しと GGUF も 7B 以上で、名前に ELYZA を含む 1.8B は ELYZA の課題で学習した llm-jp-3 の派生（作り手は ELYZA でない）。スマホ向けの ELYZA は足せない。
- **スマホの大きさで新しく足せる日本語のモデルは 4〜5 つ**。今の形（LFM2・Qwen3.5・Gemma 3）だけで足りる。いちばん勧めるのは Shisa の LFM2 350M（日本語の会話の評価で 1B 未満の最上位と作り手が言う、int8 約 0.4 GB）と、個人の日本語版 Qwen3.5 0.8B（v2.5）。
- **今の一覧は 130 項目のうち 97 が「desktop only」**。日本語でスマホの大きさ（「desktop only」なし）は 17、そのうち問いに答えるものは 9。「desktop only」の線は手書きで、およそ int8 0.8 GB（LFM2 700M の 836 MB から）。ページ自身のメモリの決まり（`deviceMemory` 4 の端末で警告が出るのは約 1.85 GB から）とも、持ち主の Android の実測（Llama 3.2 3B が 6.0 tok/s で動いた）とも合っていない。
- **既定の tiny-lm は、取得の小ささでは今もいちばん**（33 MB、ほかは 171 MB 以上）。日本語の質はいちばん低い（1 文字あたり 2.507 nats、llm-jp-3 150M は 1.950）。問いに答える小さいモデル（LFM2.5 230M・Gemma 3 270M）と同じ物差しの数は無い（未計測）。替えるかは持ち主に聞く。

## 足す候補の勧め（順に）

1. **T.511.1 Shisa V2.1c LFM2 350M**（`shisa-ai/shisa-v2.1c-lfm2-350m`、GGUF は mradermacher の Q8_0 379 MB → int8 約 0.40 GB）: スマホの大きさで、問いに日本語で答える力がいまの 350M 級より上（カードの Shaberi 平均 4.51、元の LFM2-350M 3.76、Qwen3-0.6B 4.14、Gemma 3 270M 2.53）。LFM2 の形（一覧の LFM2 700M と同じ）なので項目だけ。**カードにライセンスの欄が無い**（元は LFM Open License v1.0）: 持ち主に聞くもの (A)。仕事は小（項目と GGUF の突き合わせ・書式・書かせて読む）。
2. **T.511.2 Qwen3.5 0.8B Japanese SFT v2.5**（`Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2.5`、Q8_0 812 MB → int8 約 0.85 GB、apache-2.0）: 一覧の Qwen3.5 0.8B と同じ大きさで、日本語の会話に寄せた版（カード: 中国語の混入 1% → 0%、「AI なので」の断り 17% → 5%、JCommonsenseQA の生成 42.6% → 71.0%）。トークナイザ・書式・`tokenizer_config.json` は元の Qwen3.5-0.8B とバイト単位で同じ（API の blobId が一致）ので、NeoHorse-1 と同じ形で項目だけ。**個人の非公式の派生**。仕事は小（考える形とすぐ答える形の 2 項目）。
3. **T.511.3 Shisa V2.1 LFM2 1.2B**（`shisa-ai/shisa-v2.1-lfm2-1.2b`、Q8_0 1.25 GB → int8 約 1.3 GB、LFM Open License v1.0）: 1B 級で日本語を最もよく話すと作り手が言うもの（Shaberi V2.1 5.35。同じ表に元の LFM2-1.2B は無い）。8 と言う Android なら入る。仕事は小。
4. **T.511.4 LFM2 350M ENJP MT**（`LiquidAI/LFM2-350M-ENJP-MT`、Q8_0 379 MB → int8 約 0.40 GB、LFM Open License v1.0）: スマホで使える翻訳（一覧の翻訳は CAT-Translate 0.8B の 0.9 GB から上で、どれも「desktop only」）。**system の文が必須で向きが 2 つ**（「Translate to Japanese.」「Translate to English.」）なので項目は 2 つ。仕事は小〜中（書式の手書き）。
5. （低い）**LFM2.5 1.2B Thinking**（`LiquidAI/LFM2.5-1.2B-Thinking`、Q8_0 1.25 GB → int8 約 1.3 GB）: 考える形の 1.2B。カードの言語に日本語。スマホでは 16 tok/s ほどの見込みで、考える分だけ答えが遅い。

ほかに一覧そのものの直し（足さない）を 2 つ（下の 3 節）: **T.511.5**「desktop only」の線を規則にする（文面なので持ち主）、**T.511.6** 選択欄でスマホに入らない項目の見え方（同上）。

## 持ち主に聞くもの（選択肢、勧めを先頭に）

- **(A) ライセンスの欄の無い Shisa V2.1c LFM2 350M を入れるか**
  1. **入れる。ライセンスは元の「LFM Open License v1.0」と書く**（元のモデルの条項が派生にも及ぶ。カードの本文は元を名指している: `base_model: LiquidAI/LFM2-350M-ENJP-MT`）。
  2. 作り手に聞いて、カードに書かれるまで待つ（待つ間は入れない）。
  3. 入れない（V2.1 LFM2 1.2B だけにする: こちらはカードが lfm1.0 と書く）。
- **(B) 個人の非公式の派生（Qwen3.5 0.8B Japanese SFT v2.5）を一覧に入れるか**
  1. **入れる**（apache-2.0、元と同じトークナイザ。方針 8 はライセンスで外さないと言うだけで、作り手の種類の決まりは無い）。
  2. 入れない（作り手が会社か研究機関のものだけにする）。
- **(C) スマホの既定のモデル**（方針 4。今は tiny-lm、イントロのボタンは llm-jp-3 150M）
  1. **今のまま（tiny-lm）**: 取得 33 MB で準備がいちばん速い。訪問者は選び直せる。
  2. 既定は tiny-lm のまま、**イントロのボタンに問いに答える小さいモデルを 1 つ足す**（例: LFM2.5 230M 247 MB、または T.511.1 の Shisa 350M 約 380 MB）。文面は持ち主。
  3. 既定を llm-jp-3 150M（171 MB）にする: 日本語の質は上がる（1 文字あたり 2.507 → 1.950 nats）が、毎回の取得が 5 倍。
  4. 既定を問いに答えるモデル（LFM2.5 230M など）にする: 取得 247 MB・HF から取って変換（GitHub Pages のサイトの外）。
- **(D)「desktop only」の線**（文面）
  1. **規則にする: 「`deviceMemory` が 4 の端末でページが警告を出すもの」（int8 + 0.3 GB が 2 GiB を超える、約 1.85 GB から）に付ける**。今「desktop only」で 1.85 GB 以下の 31 項目（Qwen3.5 0.8B・Gemma 3 1B・LFM2.5 1.2B・TinySwallow 1.5B など。サイトの組の 3 つを含む）から外れる。
  2. 今の線（約 0.8 GB）のまま、規則として書き残すだけ。
  3. 「desktop only」を「8 GB phones and up」のような端末の大きさの言い方に替える。
  4. note から外し、メモリの警告（`memoryWarning()`）だけに任せる。

## 1. ELYZA（`elyza` の公開の全部と、ほかの作り手の写し）

`https://huggingface.co/api/models?author=elyza&limit=100`（2026-10-15）は 20 件。どれもゲートなし。

| リポジトリ | 作成 | 大きさ | 形 | ライセンス | ここで | スマホ |
|---|---|---|---|---|---|---|
| Llama-3-ELYZA-JP-8B（と AWQ・GGUF） | 2024-06-25 | 8B | Llama 3 | llama3 | **一覧にある**（int8 9.0 GB） | × |
| ELYZA-Shortcut-1.0-Qwen-7B | 2025-04-30 | 7.6B | Qwen2.5 | apache-2.0 | 開く見込み（Qwen2。Q8_0 は mmnga の 8.10 GB） | × |
| ELYZA-Shortcut-1.0-Qwen-32B・Thinking-1.0-Qwen-32B | 2025-04-30 | 32B | Qwen2.5 | apache-2.0 | 16 GiB を越える | × |
| ELYZA-Thinking-1.0-llm-jp-4-33b | 2026-10-02 | 33B | Llama | apache-2.0 | 16 GiB を越える | × |
| ELYZA-Thinking-1.0-llm-jp-4-32b-a3b | 2026-10-02 | 32B（MoE） | qwen3_moe | apache-2.0 | 形が無い（MoE） | × |
| ELYZA-Diffusion-Base・Instruct-1.0-Dream-7B | 2025-12-10 | 7B | 拡散（Dream） | apache-2.0 | 形が無い | × |
| ELYZA-japanese-Llama-2-7b（-fast・-instruct・-fast-instruct） | 2023-08-28 | 7B | Llama 2 | llama2 | 開く見込み | × |
| ELYZA-japanese-Llama-2-13b（4 つ） | 2023-12-25 | 13B | Llama 2 | llama2 | 16 GiB の近く | × |
| ELYZA-japanese-CodeLlama-7b（2 つ） | 2023-11-07 | 7B | Llama 2 | llama2 | 開く見込み | × |

- **ほかの作り手**（`search=elyza` の上位 200、ダウンロード順）: 写しと GGUF と派生はどれも 7B・8B・13B・32B と MoE。名前に ELYZA を含む小さいものは `kumapo/llm-jp-3-1.8b-…-elyza-…`（ELYZA の課題のデータで学習した llm-jp-3 1.8B の個人の版、ダウンロード 25〜39）だけで、ELYZA のモデルではない。
- **結論**: スマホの大きさの ELYZA は無い。ELYZA の名前で足せるのは 7B 以上（デスクトップだけ）。足すなら ELYZA-Shortcut-1.0-Qwen-7B（apache-2.0、Qwen2 の形で今のまま開く見込み、Q8_0 の GGUF あり）が今の一覧の 8B との違い（Qwen2.5 の元・ライセンス）がある。スマホには関係しないので勧めの順には入れない。

## 2. スマホの大きさの日本語の候補（新しいものと漏れていたもの）

探したもの: `filter=ja&pipeline_tag=text-generation` のダウンロード上位 1000、作り手 15（sbintuitions・llm-jp・SakanaAI・rinna・cyberagent・pfnet・stockmark・kotoba-tech・tokyotech-llm・Rakuten・LiquidAI・google・tohoku-nlp・lightblue・Aratako・shisa-ai）の全部、名前の検索 18 語（Gemma 3・Qwen3.5・Qwen3・LFM2・SmolLM3・TinySwallow・sarashina2.2 の日本語の派生など）。

### 足す候補（今の形で開く見込み）

| モデル（リビジョン、作成） | 形 | 取り込み元（Q8_0） | int8 の見込み | ライセンス・ゲート | 日本語の手掛かり（カード） | 要るもの |
|---|---|---|---|---|---|---|
| shisa-ai/shisa-v2.1c-lfm2-350m（`a902123e`、2025-10-12） | LFM2（16 層、`ccaccaccacacacac`、`conv_bias` 偽）。元は LFM2-350M-ENJP-MT | mradermacher/…-GGUF `38445ae7` の Q8_0 379,216,512 B | 約 0.40 GB | **カードに欄なし**（元は lfm1.0）・ゲートなし | Shaberi 平均 4.51（ELYZA 100 4.30・JA-MT 4.75・Rakuda 5.03・Tengu 3.95）。同じ表で LFM2-350M 3.76、Qwen3-0.6B 4.14、Gemma 3 270M 2.53、Llama 3.2 3B Instruct 4.49 | 項目（`lfm2()`、LFM2 700M の `lfm2Old` の形）・`gguf_check.py tensors`・`format_check.py`・`start_check.py`・`write.sh` |
| Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2.5（`b069f75c`、2026-10-11） | Qwen3.5（`qwen3_5`、24 層、語彙 248,320） | 作り手の …-GGUF `101c43b4` の Q8_0 811,842,912 B | 約 0.85 GB | apache-2.0・ゲートなし | v2.5 と元（同じ設定、llama.cpp）: JCommonsenseQA の生成 42.6 → 71.0%、一般常識 30 問の日本語 17 → 22、中国語の混入 1 → 0%、断り 17 → 5%、MGSM 日本語（考える）4.4 → 27.4% | 項目 2 つ（`qwen35` の options・書式・止まりは Qwen3.5 0.8B と同じ: 3 つのファイルの blobId が元と一致）・`gguf_check.py`・書かせて読む（カードのすぐ答える形の組 presence 2.0 は、一覧の Qwen3.5 0.8B で文が崩れた組: AGENTS.md） |
| shisa-ai/shisa-v2.1-lfm2-1.2b（`292863c4`、2025-11-13） | LFM2（1.2B） | mradermacher `4a433b91` の Q8_0 1,246,254,048 B（XpressAI の Q8_0 もある） | 約 1.3 GB | lfm1.0・ゲートなし | Shaberi V2.1 5.35・JA AVG 43.4（Shisa の表。元の LFM2-1.2B の数は無い）。CLTL（違う言語の語の混ざり）が元の 4.32% から 0.32% | 項目・同上 |
| LiquidAI/LFM2-350M-ENJP-MT（`80367784`、2025-09-03） | LFM2（350M） | 作り手の …-GGUF `889c88c3` の Q8_0 379,215,648 B | 約 0.40 GB | lfm1.0・ゲートなし | 英日・日英の翻訳だけ。system の文が必須（「Translate to Japanese.」「Translate to English.」）、温度 0.5 を勧める | 項目 2 つ（向きごと）、書式の手書き |
| LiquidAI/LFM2.5-1.2B-Thinking（`f3134789`、2026-01-20） | LFM2（1.2B） | 作り手の …-GGUF `9584426e` の Q8_0 1,246,254,080 B | 約 1.3 GB | lfm1.0・ゲートなし | カードの言語に ja。日本語の数は無い | 項目・考える形の書式 |

### 読んだが勧めないもの（訳つき）

| モデル | 訳 |
|---|---|
| Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2（2026-10-02） | v2.5 が同じ作り手の後の版（会話の質は同じで考える形が良い、とカード）。v2 は `tokenizer.json` が元と違う（19,989,325 B、元は 12,807,982 B） |
| alphaedge-ai/gemma-3-1b-it-jpn-16384・-32768（2026-03） | Gemma 3 1B の語彙を 16,384 に刈ったもの（0.72B、int8 約 0.75 GB の見込み）。分類器が小さくなるので 1 トークンの読みが約 3 割減る見込み（式だけ）。ただし `base_model_relation: quantized`、質の数がカードに無い、ダウンロード 13。読んだだけ |
| tatsuyaaaaaaa/gemma-3-1b-it-japanese-unsloth(2) | 元が bnb の 4 ビット、カードが空に近い、ダウンロード 40 前後、Gemma の派生に apache-2.0 と書く |
| uneeverse/unee-0.8b（Qwen3.5 0.8B） | 判定（確率を返す）とアプリ向けの多言語の版。日本語向けでない |
| yuiseki/qwen3-0.6b-jp-gov-v0.1・llm-jp-3-440m-jp-gov | 官公庁の文の用途の個人の版 |
| pfnet/plamo-2-1b・plamo-3-610m-fin-instruct | 形が無い（plamo2・plamo3） |
| tiiuae/Falcon-H1-1.5B、RWKV7 の 0.1〜1.5B | 形が無い（状態空間の層・RWKV） |
| tohoku-nlp/nanochat-jp-v3・v4-sft-hf（1.12B） | 形が無い（`nanochat_jp`） |
| SakanaAI/SparseLM 0.5〜2B | 形が無い（`llama_sparse_relu`）。日本語向けでない |
| llm-jp-4 の系統 | いちばん小さいものが 8B（llm-jp-4.1 8B thinking の作り手の GGUF は BF16 と Q4_K_M だけで、Q8_0 は今も無い: `llm-jp/llm-jp-4.1-8b-thinking-gguf` `3afab819`） |
| rinna・kotoba-tech・stockmark・tokyotech-llm（Swallow）・sbintuitions・Rakuten・SakanaAI（TinySwallow） | 2025-06 より後に 2.3B 以下の文の生成モデルを出していない（sbintuitions は TTS・埋め込み・ModernBERT、tokyotech-llm は 8B 以上） |
| CMSManhattan/JiRackUltra_1b | `filter=ja` で 1 位（250 万ダウンロード）だが、DeepSeek-R1 の書式の Qwen2 の 1.8B で、日本語の手掛かりがカードに見当たらない。読んだだけ |

- **Gemma 3 の日本語の派生**: 270M と 1B の日本語版で、作り手が会社か研究機関のもの、または数の付いたものは見つからなかった（上の 2 つと、ブラウザ向けの MLC の版だけ）。
- **SmolLM3 の日本語の派生**: 検索で 0 件。
- **LFM2.5 の日本語**: 230M・350M の日本語版は無い。1.2B JP は一覧が新しい版（JP-202606）を使っている。2.6B の個人の JP 版（CloudGoat、cc-by-nc-sa-4.0）は 2.7B でスマホに入らない。

## 3. 今の一覧をスマホの目で（`src/models.js` の 130 項目）

### 数（`modelBytes()`・`memoryWarning()`・`weightsFor()` を Node で）

| | 数 |
|---|---:|
| 全部 / 日本語が使える / 英語だけ | 130 / 85 / 45 |
| 「desktop only」あり / なし | 97 / 33 |
| 「Chrome and Firefox」（64 ビットのメモリが要る: Safari は開かない） | 41 |
| 日本語で「desktop only」なし / あり | 17 / 68 |
| そのうち問いに答える（note に answers・thinks・translates） | 9 |
| `deviceMemory` 4 の端末で警告が出る / 6 ビットになる | 66 / 61 |
| `deviceMemory` 8（持ち主の Android）で警告が出る | 22 |
| int8 で 0.5 / 0.8 / 1.0 / 1.5 / 1.85 GB 以下（うち日本語） | 28（12）/ 35（17）/ 43（24）/ 54（30）/ 64（37） |

スマホの大きさで問いに答える日本語の 9 つ: llm-jp-3 150M instruct3、LFM2.5 230M、Gemma 3 270M、LFM2.5 350M、Ternary Bonsai 1.7B、llm-jp-3 440M instruct3、Qwen2.5 0.5B Instruct、Qwen3 0.6B（考える形とすぐ答える形）。

### 見つけたこと

1. **「desktop only」は手書きで、線は約 0.8 GB**。付いていない最大は Qwen3 0.6B の 671 MB、付いている最小は（Hugging Face の組で）LFM2 700M の 836 MB。台帳の記録も割れている（T.249 の (h) は「int8 が 1.4 GiB 以上」と書き、T.251 は「0.8 GB 以上」と書く。今の一覧は後者）。
   - ページ自身の決まりとずれる: `memoryWarning()` が `deviceMemory` 4 の端末に警告を出すのは int8 + 0.3 GB が 2 GiB を超えるとき（約 1.85 GB から）。1.85 GB 以下の 31 項目（Hugging Face の組の 28 とサイトの組の 3）は、4 の端末でも警告が出ないのに「desktop only」と書いてある。
   - 持ち主の Android（8 と言う）は「desktop only」の Llama 3.2 3B Instruct（3.6 GB）を 6.0 tok/s で動かした（TASKS.md、2026-09-28）。8 と言う端末には「desktop only」の 3B 級まで実際は開く。
   - **iPhone の上限は未計測**（Safari は `deviceMemory` を言わず、警告も出ない。Page memory の節は iPhone で未確認: T.173）。線を 1.85 GB に上げると、iPhone で落ちる項目が「desktop only」なしになる恐れがある（覆す条件: 持ち主の iPhone で 1 GB 級を開いてタブが落ちたら、Safari のための線を別に置く）。
   - サイトの組の `stories42M`（int8 47 MB）と `stories42M-f32`（167 MB）にも「desktop only」が付いている。大きさでは当たらない。台帳に訳は見つからなかった。
2. **選択欄の並び**: グループは「Models of this site」（6）→「Unquantized originals」（4）→「From Hugging Face …」（120）。Hugging Face の組の中は日本語の 81（軽い → 重い）の後に英語だけの 39。**スマホに入る英語だけの小さいモデル（Pythia 14M・SmolLM2・GPT-2 など 12）は、デスクトップだけの日本語 67 項目の後ろ**（82 番目から）。日本語の人には困らないが、スマホの英語の人は 67 個のデスクトップだけの項目を越えないと見つからない。T.128 の並び（日本語の軽い → 重い、英語の軽い → 重い）は持ち主の決めなので変えていない。案（持ち主）: 「desktop only」の項目を各組の後ろに回す、または `deviceMemory` が小さい端末では optgroup を分ける（「This device may not have the memory」）。
3. **選択欄の長さ**: 130 項目、そのうち 97 がスマホで「desktop only」。英語だけで int8 3 GB 以上の大きいモデルが 14（Pythia 2.8B〜Llama 3.1 8B）。考える形とすぐ答える形の対で 2 項目になるものが 19 対。
4. **イントロのボタン**（`src/pages/index.astro`）は llm-jp-3 150M（素の文の続きを書くモデル）を指す。チャットの形のページで、スマホの人が最初に問いを打つと、続きを書くモデルは答えにならない。問いに答える小さいモデル（llm-jp-3 150M instruct3、LFM2.5 230M）はボタンに無い。

### 既定のモデル（tiny-lm）と、イントロのボタン（llm-jp-3 150M）を、小さいものと数で比べる

| モデル | 取得 | 準備完了（CI の Chromium、速い回線） | tok/s（CI の 4 論理コア、本番） | 日本の家の回線（7〜9 MB/s）の取得の見込み | 日本語の質（1 文字あたりの nats、低いほど良い） | 問いに答える |
|---|---:|---:|---:|---:|---:|---|
| tiny-lm 29M（既定） | 33 MB | 未計測 | 約 280（Chromium）、455（本番、2 本、T.249） | 4〜5 秒 | 2.507 | いいえ（続きを書く） |
| llm-jp-3 150M（ボタン） | 171 MB | 未計測（instruct3 は 13 秒） | 75〜79（256 トークン）、106（instruct3） | 19〜24 秒 | 1.950 | いいえ |
| japanese-gpt2 xsmall | 156 MB → 42 MB | 未計測 | 237（本番、2 本） | 17〜22 秒 | 2.077 | いいえ（答えの 19% に `<unk>`） |
| LFM2.5 230M | 247 MB | 11.6 秒（`?hf=` の GGUF、T.505.3） | 93.5 | 27〜35 秒 | 未計測 | はい |
| Gemma 3 270M | 292 MB | 15.0 秒 | 66.5（T.510.1） | 32〜42 秒 | 未計測（書かせた 12 のうち日本語 10、崩れ 0/24: T.261.3） | はい |
| LFM2.5 350M | 379 MB | 未計測 | 50〜57 | 42〜54 秒 | 未計測 | はい |
| Qwen3.5 0.8B | 812 MB | 未計測 | 28.2〜28.7 | 90〜116 秒 | 未計測 | はい |

- 日本語の質の数は T.249 の物差し（日本語 Wikipedia の 3 記事、float32 の transformers。tiny-lm と llm-jp-3 150M はページのエンジンの int8）で、問いに答えるモデルには同じ物差しの数が無い。問いへの答えの質を比べる数（同じ 12 の問いを書かせて読む: `tests/answers.mjs`）も、この 7 つを並べたものは無い。**既定を替える前に要る測り**（案、T.511.7）: 上の 7 つに同じ 12 の問いを greedy と既定の標本抽出で書かせて並べ、日本語の割合（かなで数える）・崩れ・繰り返しを数える（CI、1 本あたり数分）。
- 方針 4 の訳（公開のページでスマホに無確認で大きく落とさせない、準備がいちばん速い）は今も tiny-lm に当てはまる。問いに答える最小は LFM2.5 230M の 247 MB で、tiny-lm の 7.5 倍。

## 4. 持ち主の Android（Xiaomi 13T Pro、Chrome、8 論理コア、`deviceMemory` 8）での見込み

**実測**（台帳）: llm-jp-3 150M（int8 171 MB）は 1 本 110・2 本 123・4 本 38.6・8 本 32.3 tok/s（ページは 2 本を選ぶ。`/benchmark/` の CPU だけの書くこと 107 tok/s）。Llama 3.2 3B Instruct（int8 3.6 GB）は GPU で 6.0 tok/s。1 トークンの読みは 1 本 12.9〜18.9 GB/s、4 本 27.6〜28.7 GB/s（`/benchmark/` の CPU の節）。

**式**: バッチ 1 の生成は重みを読む帯域で決まる。tok/s ≈ B ÷（int8 の大きさ）。B は 2 つの実測から約 21 GB/s（123 × 171 MB = 21.0、6.0 × 3.6 GB = 21.6）。幅は 0.65〜1.3 倍（下は 1 本の 13.7 GB/s、上は CI で LFM2・Qwen3.5 が式より 1.3 倍ほど速く出た分: CI の 4 コアで LFM2.5 230M は式の 70 に対して 93.5、Qwen3.5 0.8B は 21 に対して 28）。

| モデル | int8 | 見込み tok/s（中心、幅） |
|---|---:|---:|
| llm-jp-3 150M | 171 MB | **実測 123**（2 本） |
| LFM2.5 230M | 259 MB | 81（53〜105） |
| Gemma 3 270M | 302 MB | 70（45〜90） |
| LFM2.5 350M・Shisa V2.1c LFM2 350M・LFM2 350M ENJP MT | 399 MB | 53（34〜68） |
| Qwen3 0.6B | 671 MB | 31（20〜41） |
| Qwen3.5 0.8B・Japanese SFT v2.5 | 850 MB | 25（16〜32） |
| Gemma 3 1B | 1.1 GB | 19（12〜25） |
| LFM2.5 1.2B・Shisa V2.1 LFM2 1.2B・LFM2.5 1.2B Thinking | 1.3 GB | 16（11〜21） |
| Llama 3.2 3B Instruct | 3.6 GB | **実測 6.0**（GPU） |

- 小さいモデル（tiny-lm など 100 MB 未満）は層ごとの待ち合わせと forward の外の費用が効くので、この式は当てない（未計測）。
- GPU に置かないモデル（LFM2・Qwen3.5・Gemma 3）は CPU の数。Gemma 3 は 4 本で崩れる型（llm-jp-3 150M の 4 本 38.6）が当たるかは未計測（ページが本数を測って選ぶので、崩れる本数は選ばれない見込み）。
- **覆す条件**: 持ち主の端末で上の 1 つを開いて、tok/s が幅の外なら式の B を直す（端末でまとめて見るもの の一覧に積む案: LFM2.5 230M と Qwen3.5 0.8B の 2 つ）。

## 確かめた事実と、読んだだけのもの

- **確かめた（API・カード・config の文字で）**: 上の表のリビジョン（40 桁は `.tmp/t511/info/` に、文書には頭 8 字）、作成日、ゲート、カードの `license` の欄、ファイルの大きさ（GGUF の Q8_0、safetensors）、config の `model_type`・層の並び・`conv_bias`・語彙・BOS と EOS、Takenoko v2.5 の `chat_template.jinja`・`tokenizer.json`・`tokenizer_config.json` が元の Qwen3.5-0.8B（`2fc06364`）と同じ blobId、ELYZA の 20 件。一覧の数（Node で `src/models.js` を読んだ）。
- **読んだだけ（数は作り手のカードのまま、ここで測っていない）**: Shaberi・JA AVG・JCommonsenseQA などの日本語の数、CLTL の割合、Gemma 3 1B の刈った版の「同じように動く」。
- **未確認**: どの候補も変換器に通していない（`gguf_check.py tensors`・書式・頭のトークン・書かせて読む は T.511.1〜.4 で）。LFM Open License v1.0 の条文を派生が引き継ぐかの法的な読み（カードの `base_model` から読んだ）。持ち主の端末の tok/s（式の見込みだけ）。iPhone の Safari が開ける大きさ。
- **見ていないもの**: `filter=ja` の付いていない GGUF だけのリポジトリの網羅（名前の検索 18 語で拾った分だけ）、2026-10-15 より後の公開、ゲート付きのリポジトリ（Gemma 3 の google の原本は API の読みだけ）、英語だけの小さいモデルの新しいもの（この回は日本語に絞った）。
