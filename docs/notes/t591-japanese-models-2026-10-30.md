# T.591 有名で動かせる日本語モデルの調べ（2026-10-30）

持ち主「有名で動かせる日本語のモデルを徹底的に限界まで調査してタスクとして書き起こしてほしい」。調べ手 3 つ（学術・国・大手研究所 / 日本の企業 / コミュニティ・世界のモデルの日本語版・HF の ja の順位・順位表）が HF の API・config.json・tokenizer の設定・カード・GGUF の見出し（Range だけ）で調べた。重みは取っていない。書き起こしたタスクは TASKS.md の T.592〜T.619。

**読み方**: 「動く見込み」は model_type とトークナイザの形からの見込みで、どれも未計測（`gguf_check.py tensors`・`format_check.py`・書かせて読むを通すまで「動く」と言わない: AGENTS.md の「モデルを一覧に足す」）。メモリの数（int8 + 文脈 4096 の KV）は調べ手の見積もりで、`footprint()` では出していない。sha は 2026-10-30 の main。生のデータ（`.tmp/jp-survey/`）は手元にしか無い。

## まとめ

- **いまのエンジンで開く見込み**: llm-jp-4 8B thinking（Q8_0 あり）・ELYZA-Shortcut 1.0 Qwen 7B・Tanuki 8B・llm-jp-3 7.2B instruct3・LFM2 350M ENJP-MT・LFM2.5 2.6B・Spark-X2.5 4B Japanese・sarashina の残り・calm2 7B chat・Japanese StableLM gamma 7B・RakutenAI 7B の instruct と base・rinna の youko と youri・CAT-Translate 7B・Sakana の小さいもの・Lightblue と ABEJA の Qwen・2026 のコミュニティの 8B・ELYZA の Llama 2。
- **メモリの線の近く**（14B 級、見積もり 16.4〜16.5 GB / 上限 17.18 GB）: Qwen3-14B（Swallow の順位表で ≤14B の 2 位）・shisa-v2.1-unphi4-14b・CyberAgent の DeepSeek-R1-Distill-Qwen-14B-Japanese・ABEJA-Qwen3-14B-Agentic。13B の MHA（llm-jp-3.1 13B・stockmark-13b など）は int8 では越える見込みで、塊（IQ4_XS など）なら入る。
- **仕事が要るもの**（開くもの）: Tekken の前分割（Mistral-Nemo-Japanese 12B）、gemma2（gemma-2-2b-jpn-it・Gemma-2-Llama-Swallow・gemma-2-baku）、Gemma 3 のまるごとの config（Gemma 3 4B・12B）、plamo3（PLaMo 3）、Mamba2 の層（PLaMo 2・Nemotron-Nano-9B-v2-Japanese）、stablelm（JSLM 3B）、Qwen v1（nekomata）、`<NL>` の置き換え（rinna の gpt-neox 3.6B・bilingual 4B）、取り込み元の問題（weblab-10b・Fugaku-LLM 13B・open-calm）、lfm3。
- **やらない**: MoE（llm-jp-4 32B-A3B・ELYZA-Thinking 32B-A3B・Qwen3-Swallow 30B-A3B・GPT-OSS-Swallow・RakutenAI 2.0 8x7B と 3.0・Tanuki 8x8B・karakuri 8x7B・sarashina2-8x70b）、大きすぎる密なモデル（32B 以上: bakeneko 32B・karakuri 70B・stockmark-100b・PLaMo 100B・calm3 22B は Q4_K_M の塊なら入るかもしれないが未計測）、音声・画像・拡散・分類器。重みを公開していない: NTT tsuzumi・富士通 Takane・NEC cotomi・リコー・NICT（日本語の汎用の LLM）。LY Corp は新しいものが無い（4 つとも一覧にある）。消えたもの（401）: rinna/japanese-gpt-neox-3.6b・-instruction-sft・qwen2.5-bakeneko-32b-instruct-v2。

## 順位表（≤14B の公開モデル）

- Swallow LLM leaderboard（後訓練、日本語の平均）: Qwen3.5-9B 0.534（一覧）・Qwen3-14B 0.524（無い）・Qwen3-Swallow-8B-RL 0.510（一覧）・NVIDIA-Nemotron-Nano-9B-v2-Japanese 0.503（無い: Mamba2）・Qwen3-8B 0.491（一覧）・Gemma 4 E4B・Qwen3.5-4B・llm-jp-4-8b-thinking（無い）・Gemma 4 E2B。下に Gemma 3 12B 0.405・4B 0.292（無い）。
- Nejumi 4 は第三者のまとめしか読めなかった（10B 以下の 1 位を Nemotron-Nano-9B-v2-Japanese とも Qwen3.5 とも書く）。
- HF の ja の順位（ダウンロード・いいね・流行の上位 300）は世界の多言語モデルばかりで、日本語だけのコミュニティのモデルは入っていない。

---

## 付録: 調べ手の表（academic）

### 日本語 LLM の調べ: 学術・国・大手研究所の組（2026-10-30）

調べ方: HF の API（`/api/models?author=…`、`/api/models/<repo>?blobs=true`、`resolve/<sha>/config.json`・`tokenizer_config.json`）、GGUF は Range で見出しだけ（`acad_ggufhead.py`）。数字の生データは `.tmp/jp-survey/acad/*.json`。重みは取っていない。
DL/likes は API の値（直近 30 日のダウンロード）。safetensors の GB は API の blob の合計。

#### 0. 結論（足す価値の順）

| 順 | 候補 | 今すぐ動くか | 取るもの | 一言 |
|---|---|---|---|---|
| 1 | **llm-jp-4.1 8B thinking**（NII） | 動く見込み（llama、harmony の書式は llm-jp-4 8B instruct の `HARMONY` の形） | **Q8_0 の GGUF は今も無い**。作り手の GGUF は BF16 17.19 GB と Q4_K_M 5.50 GB だけ、masahiroid も Q4_K_M 5.3 GB だけ。safetensors 17.18 GB か、Q4_K_M を塊で（`rebuilt: true`） | llm-jp の最新の 8B。T.250 で待ちにしたもの。持ち主の判断: 17 GB を取るか Q4_K_M の塊か |
| 2 | **llm-jp-4 8B thinking**（NII） | 動く見込み（同上） | **Q8_0 あり**: mmnga-o `llm-jp-4-8b-thinking-gguf` 9.133 GB（alfredplpl にも同じ大きさ） | 4.1 の 1 つ前。likes 52 で llm-jp で最多。考える形の llm-jp を今すぐ足せる |
| 3 | **Qwen3-Swallow 8B SFT v0.2**（Science Tokyo） | 動く（qwen3、一覧の RL 版と同じ形） | mmnga-o Q8_0 8.71 GB | DL 5,133 で RL 版（一覧にある）より多い。ただし RL 版と中身が近く、足す意味は小さい |
| 4 | **ELYZA-Shortcut-1.0-Qwen-7B**（ELYZA） | 動く（qwen2） | mmnga Q8_0 8.099 GB | T.511 で「足すならこれ」と言われた ELYZA。Qwen2.5 7B の元・Apache-2.0、考えを省いた答え |
| 5 | **Tanuki-8B-dpo-v1.0**（松尾研 GENIAC） | 動く見込み（llama、Unigram の tokenizer.json に NFKC と `(?<!\n)^` の Replace: llm-jp-3 と同じ形） | team-hatakeyama-phase2 Q8_0 7.984 GB | 国の GENIAC の看板。書式に全角空白入りの Jinja（`{{　bos_token }}`）: 変換器の読み手と jinja2 で読めるか要確認 |
| 6 | **llm-jp-3 7.2B instruct3**（NII） | 動く（llama、llm-jp-3 の一覧と同じ） | mmnga Q8_0 7.751 GB | 一覧の llm-jp-3 の 3.7B と 13B の間を埋める |
| 7 | **TinySwallow-1.5B（素の形）**・**TAID-LLM-1.5B**（Sakana） | 動く（qwen2） | mmnga Q8_0 1.647 GB / bartowski Q8_0 1.646 GB | 小さい。素の形は書式なし、TAID は 2024 の蒸留の研究モデル（日本語特化ではない） |
| 8 | **Llama-3-Karamaru-v1**（Sakana） | 動く（llama、ELYZA-JP 8B の後訓練） | mradermacher Q8_0 8.541 GB | 江戸時代の文体で話す話題のモデル（Llama3 ライセンス） |
| 9 | **EvoLLM-JP-A-v1-7B / v1-7B**（Sakana） | 動く（mistral） | mmnga Q8_0 7.696 GB（GGUF v3） | 2024 年に話題の進化的マージ。v1-7B は Microsoft Research License（非商用）、A は Apache-2.0 |
| 10 | **ELYZA-japanese-Llama-2-7b-fast-instruct / 7b-instruct**（ELYZA） | 動く（llama 2） | mmnga Q8_0 7.275 / 7.161 GB（GGUF v3） | 2023 年の有名どころ（likes 83・75）。今は古い |
| 11 | **weblab-10b-instruction-sft**（松尾研） | 形は gpt_neox で読める。**取り込み元が無い**: GGUF は v1（2023-08、読まない・NeoX の古い並び）、本線は `.bin` だけ | safetensors は PR の枝 `refs/pr/4`（commit `3f4b5eeb74637041e7e50f9ac1907600c7a82c5a`、float16 21.42 GB、3 シャード） | 有名（likes 74）だが 21 GB を取る。cc-by-nc-4.0 |
| 12 | **Fugaku-LLM-13B-instruct**（富岳） | 形は gpt2（head 144、語彙 49,152 の sentencepiece「llama」型）。原本はゲート（auto）で語彙が取れない | mmnga Q8_0 14.28 GB | GGUF だけの道（`from_gguf`）で語彙が読めるか次第。13B でも文脈 2048 で 16 GiB に入る見込み（int8 ≈14.1 + KV f16 1.7 GB）|
| — | 13B 級の llama（llm-jp-3.1 13B instruct4・llm-jp-3 13B instruct3・Swallow-13b-instruct-v0.1・ELYZA-13b-fast-instruct） | 形は読めるが **int8 + 文脈 4096 の KV（MHA、f16 3.36 GB）で 16 GiB（17.18 GB）を越える見込み**（int8 ≈14.0〜14.7 GB）。塊（IQ4_XS・Q4_K_M）なら入る | Q8_0 14.0〜14.6 GB、IQ4_XS 7.43 GB | llm-jp-3.1 13B instruct4 は llm-jp-3 系の最良（likes 19）。塊の項目にするなら候補 |
| × | MoE・別の形・大きすぎ | 下の 5 節 | | |

#### 1. llm-jp（NII）

`llm-jp` は 274 リポジトリ（大半は optimal-sparsity・BTX・Dense などの研究用 MoE と実験の重み）。文を書くモデルの系統:

| 名前 | repo / sha | 形 | パラメータ | 文脈 | ライセンス | 種類 | 書式 | safetensors | 最良の GGUF | 動くか | DL / likes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llm-jp-4.1 8B thinking | llm-jp/llm-jp-4.1-8b-thinking `7c19bc90e49b717381e4f474ee95964c05217a28`（2026-09-15） | llama `LlamaForCausalLM`、32 層・4096・GQA 8、語彙 196,608 | 8.59B | 65,536 | apache-2.0 | thinking（harmony） | あり | 17.18 GB | **Q8_0 なし**: llm-jp/llm-jp-4.1-8b-thinking-gguf `7e48aef0…` は BF16 17.19 GB・Q4_K_M 5.50 GB、masahiroid/…-gguf `91f2e36f…` は Q4_K_M 5.3 GB | 動く見込み。取得 17 GB か Q4_K_M の塊（Q6_K の層は int8 に広がる、T.513） | 9,242 / 16 |
| llm-jp-4 8B thinking | llm-jp/llm-jp-4-8b-thinking `fc7c15c262710016c19bcda4372a96ff68875846`（2026-03-16） | 同上 | 8.59B | 65,536 | apache-2.0 | thinking（harmony） | あり | 17.18 GB | **mmnga-o/llm-jp-4-8b-thinking-gguf `d10eb8c8470f89b2082e793a5ffa086a402a23fc` Q8_0 9.133 GB**（alfredplpl/… `f8193a42…` も Q8_0 9.133 GB、作り手の GGUF は BF16・Q4_K_M だけ） | 動く見込み（一覧の llm-jp-4 8B instruct と同じ道） | 4,805 / 52 |
| llm-jp-4 8B base | llm-jp/llm-jp-4-8b-base `eb7a6efc0e28d180ad1efb26a399f340042f51eb` | 同上 | 8.59B | 65,536 | apache-2.0 | base | あり（tokenizer_config） | 17.18 GB | 見つからず（safetensors のみ） | 動く見込み、17 GB の取得 | 16,428 / 8 |
| llm-jp-4 8B instruct | 一覧にある | | | | | | | | | | 8,089 / 15 |
| llm-jp-4.1 33B thinking・4 33B thinking・4 33B base | llm-jp/llm-jp-4.1-33b-thinking `de581ee4353568c480d47129948d1be99f02f4ff` ほか | llama、64 層・5120・GQA 8 | 33.2B | 65,536 | apache-2.0 | thinking | あり | 66.44 GB | modulon-labs/llm-jp-4.1-33b-thinking-GGUF（IQ3_M 15.09 GB から）、mmnga-o/llm-jp-4-33b-thinking-gguf（IQ3_XXS 13.09 GB・IQ3_S 14.72 GB） | int8 は入らない。IQ3_XXS の塊（13.1 GB + KV f16 1.07 GB）なら 16 GiB に入りうる（Saluki の型の内。ただし出力・埋め込みの型を見出しで確かめる要あり）。重い仕事 | 2,058 / 14・1,065 / 45（GGUF の 4-33b は 269,838 DL で llm-jp 最多） |
| llm-jp-4(.1) 32B-A3B thinking | llm-jp/llm-jp-4.1-32b-a3b-thinking `cda260706786758045e5e96bf4d738bbc01155b5` | **qwen3_moe**（128 専門家） | 32.1B | 65,536 | apache-2.0 | thinking | あり | 64.28 GB | 作り手の GGUF あり | **× MoE** | 2,382 / 25 |
| llm-jp-4 VL 9B | llm-jp/llm-jp-4-vl-9b | 画像と文 | | | | | | | | × 画像 | 1,379 / 19 |
| llm-jp-3.1 13B instruct4 | llm-jp/llm-jp-3.1-13b-instruct4 `da86a55d7a754d1a45bc2d860f8548841a84f42f`（2025-05-23） | llama、40 層・5120・MHA 40、語彙 99,584 | 13.71B | 4,096 | apache-2.0 | instruct | あり | 27.42 GB | mmnga/llm-jp-3.1-13b-instruct4-gguf `7809b3c67450cc2f7dadf0f4fdc8bc56debc96bb` Q8_0 14.568 GB、IQ4_XS 7.434 GB、Q4_K_M 8.346 GB | 形は動く。int8 ≈14.7 GB + KV f16 3.36 GB で 16 GiB を越える見込み → 塊（IQ4_XS など）で | 1,157 / 19 |
| llm-jp-3.1 1.8B（base） | llm-jp/llm-jp-3.1-1.8b `4ff5880c4de1cff97be961bc704cba61ceeb4d5d` | llama 24 層・2048 | 1.87B | 4,096 | apache-2.0 | base | なし | 3.74 GB | 作り手・mmnga に無し（ago99 の派生の Q4_K_M だけ） | 動く（safetensors 3.7 GB → int8 約 2 GB） | 1,153 / 14 |
| llm-jp-3 7.2B instruct3 | llm-jp/llm-jp-3-7.2b-instruct3 `cdd4c7f3296fdc7785423a864bd9a86ce4c15915`（2025-01-31） | llama 32 層・4096・MHA | 7.29B | 4,096 | apache-2.0 | instruct | あり | 14.58 GB | mmnga/llm-jp-3-7.2b-instruct3-gguf `368515ca8b89be62ee58cbdfbc225c762564c0c2` Q8_0 7.751 GB | 動く（一覧の llm-jp-3 instruct3 と同じ道、KV は MHA で f16 2.1 GB） | 876 / 4 |
| llm-jp-3 13B instruct3 | llm-jp/llm-jp-3-13b-instruct3 `76aa77ba36e2c2a17d94635cea5ade03777c9869` | llama 40 層・5120 | 13.71B | 4,096 | apache-2.0 | instruct | あり | 27.42 GB | mmnga/llm-jp-3-13b-instruct3-gguf `f9214f6f…` Q8_0 14.568 GB | 3.1 13B と同じ（塊なら）。3.1 が上位版なので飛ばす | 657 / 8 |
| llm-jp-3 base（980M・1.8B・3.7B・7.2B・13B） | llm-jp/llm-jp-3-980m `37f78937…`・-1.8b `60acda32…`・-3.7b `de730074…`・-7.2b `8be0b455…`・-13b `cd3823f4…` | llama | 0.99〜13.7B | 4,096 | apache-2.0 | base | なし | 1.98・3.74・7.57・14.58・27.42 GB | GGUF はほぼ無し（safetensors から） | 動く（13B は上と同じ） | 3,014・4,030・1,829・1,641・582 |
| llm-jp-3 8x1.8B / 8x13B・3.1 8x13B | llm-jp/llm-jp-3-8x1.8b-instruct3 `4c6a5348…` ほか | **mixtral** | 9.27B〜 | 4,096 | apache-2.0 | | | 18.53 GB〜 | | **× MoE** | 193 / 4 |
| llm-jp-3 172B | | llama | 172B | | | | | | | × 大きすぎ | |
| llm-jp v1/v2（1.3B・13B） | llm-jp/llm-jp-1.3b-v1.0 `343b15a7…`（gpt2）、llm-jp-13b-v2.0 `1fa7d940…`（llama） | | | 2,048 / 4,096 | apache-2.0 | base と instruct | | `.bin` だけ（2.63・27.36 GB） | | 取り込み元が `.bin`（PR の safetensors を見ていない）。古いので飛ばす | 800 / 15・442 / 15 |
| llm-jp-moshi・modernbert・layoutlm・SAE | | 音声・埋め込みなど | | | | | | | | × 文の生成でない | |

日本語の質: llm-jp-4 のカードは MT-Bench（JA/EN）を gpt-5.4 の審査で出し、llm-jp-3 の数字より低く出ると注記（比べられない）。4.1 の評価は技術ブログ（日本語）。Nejumi の数字は今回見ていない。

#### 2. Swallow（Science Tokyo・tokyotech-llm）

138 リポジトリ（うち約 80 は code/math の ablation のチェックポイント）。一覧にあるのは Swallow-MS 7B instruct・Llama-3.1-Swallow 8B Instruct v0.5・Qwen3-Swallow 8B RL v0.2（考える形とすぐ答える形）。

| 名前 | repo / sha | 形 | パラメータ | 文脈 | ライセンス | 種類 | 書式 | safetensors | 最良の GGUF | 動くか | DL / likes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Swallow 8B SFT v0.2 | tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2 `496cd5558fef4af1d426e96327d7a74681063280`（2026-01-01） | qwen3 36 層・4096・GQA 8 | 8.19B | 40,960 | apache-2.0 | thinking（SFT のみ） | あり | 16.38 GB | mmnga-o/Qwen3-Swallow-8B-SFT-v0.2-gguf `70d77c73962e9e7571ee3c4f25324f270fcdbe53` Q8_0 8.71 GB | 動く | 5,133 / 5 |
| Qwen3-Swallow 8B CPT v0.2 | tokyotech-llm/Qwen3-Swallow-8B-CPT-v0.2 `95c7dae19b632152dcf058df2f7b1a69fc79e36f` | 同上 | 8.19B | 40,960 | apache-2.0 | base（続けての事前学習） | あり | 16.38 GB | tatsuyaaaaaaa/Qwen3-Swallow-8B-CPT-v0.2-gguf `b7783aa6…` Q8_0 8.71 GB | 動く | 1,914 / 2 |
| Medical-Qwen3-Swallow-8B | tokyotech-llm/Medical-Qwen3-Swallow-8B `f9a3d4dc38270d39196a8c5b5e14f56088fd0876`（2026-06-10） | qwen3 | 8.19B | 40,960 | apache-2.0 | thinking（医療） | あり | 16.38 GB | 見つからず（30B-A3B の Q4_K_M だけ） | 動く（safetensors 16.4 GB） | 889 / 4 |
| Llama-3.1-Swallow 8B v0.5（base） | tokyotech-llm/Llama-3.1-Swallow-8B-v0.5 `9392cd19c931ab816c56ecce32abe26ba001f320` | llama | 8.03B | 131,072 | llama3.3 + gemma | base | なし | 16.06 GB | mradermacher/Llama-3.1-Swallow-8B-v0.5-GGUF `a15134df31edf4d1bb5b1c4e3a6dc25c9eb78990` Q8_0 8.541 GB | 動く | 1,221 / 9 |
| Llama-3.1-Swallow 8B Instruct v0.3・Llama-3-Swallow 8B Instruct v0.1 | `0858f50e…`・`2ed77ce6…` | llama | 8.03B | 8,192 | llama3.1+gemma・llama3 | instruct | あり | 16.06 GB | mmnga の GGUF あり | 動くが v0.5 が一覧にあるので **飛ばす**（古い版） | 4,915・8,977 |
| Gemma-2-Llama-Swallow 2B / 9B it v0.1 | tokyotech-llm/Gemma-2-Llama-Swallow-2b-it-v0.1 `85b7e9acc9d0e7e4f73fe4733cde21222bb4995d`、-9b-it-v0.1 `9d643c2f9808d44d3151531d527c0eb480fb6417`（2025-04） | **gemma2**（26 層・2304 / 42 層・3584、語彙 256,000） | 2.61B / 9.24B | 8,192 | gemma + llama3.3 | instruct | あり | 5.23 / 18.48 GB | mmnga/Gemma-2-Llama-Swallow-2b-it-v0.1-gguf `d4f8b46a…` Q8_0 2.784 GB、-9b-it `edc96cbb…` Q8_0 9.827 GB | **× gemma2 が無い**。仕事: Gemma 3 の表に近い（窓 4096 を 1 層おき、`(1+w)` の norm、後ろの norm、√dim）が、**attention の logits の softcap（50）**と query_pre_attn_scalar、最後の logits の softcap（30、Gemma 4 にある）が要る。attention の softcap はカーネル（CPU と GPU のシェーダ）に触る: 中くらいの仕事。q・k の norm は無い | 555 / 4・1,121 / 4（2B は小さく日本語が強いので、gemma2 を足すなら第一の的） |
| Gemma-2-Llama-Swallow 27B | | gemma2 | 27B | | | | | | | × 形と大きさ | 211 |
| GPT-OSS-Swallow 20B / 120B | tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1 `39286942…` | **gpt_oss**（MoE 32 専門家） | 20.9B | 131,072 | apache-2.0 | thinking | あり | 41.83 GB | mmnga-o など | **× MoE・新しい形** | 536 / 20 |
| Qwen3-Swallow 30B-A3B | | **qwen3_moe** | 30.5B | | apache-2.0 | | | | | **× MoE** | 346 |
| Qwen3-Swallow 32B（RL・SFT・CPT）・Medical 32B | tokyotech-llm/Qwen3-Swallow-32B-RL-v0.2 `23d2fa7d…` | qwen3 64 層・5120 | 32.8B | 40,960 | apache-2.0 | | | 65.52 GB | ultimatechris/…-GGUF | int8 は入らない（塊の 3 ビットなら 13〜15 GB、重い） | 964 / 3 |
| Llama-3.x-Swallow 70B 系・Swallow-70b | | llama | 70B | | | | | | | × 大きすぎ | |
| Swallow-MX 8x7B | `52eaf168…` | mixtral | 46.7B | | | | | | | × MoE | |
| Swallow 7b / 13b（Llama 2、instruct-hf・instruct-v0.1・plus・NVE） | tokyotech-llm/Swallow-7b-instruct-v0.1 `a42f08c049ffee7a3750dafedec3344d0f82efe3`、-13b-instruct-v0.1 `25c64b333abc4078737bf38daa653a8e6d74a805` | llama（語彙 43,176） | 6.83B / 13.13B | 4,096 | llama2 | instruct | あり | 13.66 / 26.26 GB | mmnga/tokyotech-llm-Swallow-7b-instruct-v0.1-gguf `6962a8a9…` Q8_0 7.259 GB、-13b `6137f27f…` Q8_0 13.953 GB | 7B は動く。13B は int8 + KV で 16 GiB を越える見込み（塊なら） | 174・140（-hf の旧版 1,875 / 44） |

日本語の質: Swallow の公式の頁（swallow-llm.github.io）が Swallow の評価表（日本語の課題・MT-Bench）を出す。今回は数字を写していない。

#### 3. Sakana AI

| 名前 | repo / sha | 形 | パラメータ | 文脈 | ライセンス | 種類 | 書式 | safetensors | 最良の GGUF | 動くか | DL / likes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TinySwallow-1.5B（素の形） | SakanaAI/TinySwallow-1.5B `535d1322615c9f8171c3d60b10ef6955cc896fe9`（2024-12-26） | qwen2 28 層・1536 | 1.54B | 32,768 | apache-2.0 | base | あり（Qwen 由来） | 3.09 GB | mmnga/TinySwallow-1.5B-gguf `717ac6b609f797f36cac88055beff8d8520f8d70` Q8_0 1.647 GB | 動く | 714 / 35（Instruct は一覧にある） |
| TAID-LLM-1.5B | SakanaAI/TAID-LLM-1.5B `2e649507bc2da0c99fc6817731f3311e65c36964` | qwen2 | 1.54B | 32,768 | apache-2.0 | instruct（Qwen2-1.5B-Instruct から蒸留、日本語特化ではない） | あり | 3.09 GB | bartowski/TAID-LLM-1.5B-GGUF `df6449a5…` Q8_0 1.646 GB | 動く | 110 / 6 |
| Llama-3-Karamaru-v1 | SakanaAI/Llama-3-Karamaru-v1 `17411929aa0ef76daa51e7c0d0b986d65c877a58`（2025-03-19） | llama | 8.03B | 8,192 | llama3 | instruct（江戸時代の文体、元は Llama-3-ELYZA-JP-8B） | あり | 16.06 GB | mradermacher/Llama-3-Karamaru-v1-GGUF `84ad8f40…` Q8_0 8.541 GB | 動く | 85 / 19 |
| EvoLLM-JP-A-v1-7B | SakanaAI/EvoLLM-JP-A-v1-7B `a8b1356458c9883147c3aeffc1a2e51c99bfe70b`（2024-03） | mistral（語彙 32,000） | 7.24B | 32,768（窓なし？ config の sliding_window を要確認） | apache-2.0 | instruct（マージ） | tokenizer_config に無し（カードは apply_chat_template: 当時の Mistral の既定） | 14.48 GB | mmnga/SakanaAI-EvoLLM-JP-A-v1-7B-gguf `3c89db3c…` q8_0 7.696 GB（GGUF v3） | 動く見込み（書式は手書き） | 111 / 13 |
| EvoLLM-JP-v1-7B | SakanaAI/EvoLLM-JP-v1-7B `a7acd2604d3e9f0fca2987d2448ff5dc19778c21` | mistral | 7.24B | 32,768 | Microsoft Research License（非商用、WizardMath を含むため） | instruct | 同上 | 14.48 GB | mmnga/SakanaAI-EvoLLM-JP-v1-7B-gguf `e7df4f20…` q8_0 7.696 GB | 動く見込み | 158 / 36 |
| EvoLLM-JP-v1-10B | SakanaAI/EvoLLM-JP-v1-10B `78cad5aa…` | **evomistral**（独自コード、44 層の層の並べ替えマージ `num_hops` 65） | 9.86B | | other | | | 19.72 GB | RichardErkhov（DL 0） | **× 独自の形**（層を 65 回たどる。表で書けるかは未確認） | 186 / 40 |
| RLT-7B・RLT-32B・RePo・SparseLM・fwpkm・DroPE・ctm | | 英語・研究 | | | | | | | | 日本語モデルでない | |
| kame・Metom | SakanaAI/kame `6a199108…` | moshi の音声 | | | | | | | | × 音声 | 274 / 43 |
| EvoVLM・Evo-Ukiyoe など | | 画像 | | | | | | | | × | |

#### 4. ELYZA

| 名前 | repo / sha | 形 | パラメータ | 文脈 | ライセンス | 種類 | 書式 | safetensors | 最良の GGUF | 動くか | DL / likes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ELYZA-Shortcut-1.0-Qwen-7B | elyza/ELYZA-Shortcut-1.0-Qwen-7B `d33448648d9b1b7d69cbadb3f1a8d6fb4c8a01ee`（2025-04-30） | qwen2 28 層・3584・GQA 4 | 7.62B | 32,768 | apache-2.0 | instruct（考えを省いた答え） | あり | 15.23 GB | mmnga/ELYZA-Shortcut-1.0-Qwen-7B-gguf `fc1fb43597cab7c9edbe71e41c109fadb83d3815` Q8_0 8.099 GB | 動く | 533 / 3 |
| ELYZA-japanese-Llama-2-7b-fast-instruct | elyza/ELYZA-japanese-Llama-2-7b-fast-instruct `89de33d1ad568855853196802aeaecd799c6586f`（2023-08） | llama（語彙 45,043） | 6.85B | 4,096 | llama2 | instruct | なし（カードの `[INST] <<SYS>>` を手書き） | 本線は `.bin` 13.69 GB、safetensors は PR `refs/pr/4`（`844ba021…`、13.69 GB） | mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf `0f0072a8…` q8_0 7.275 GB（GGUF v3） | 動く見込み（古い GGUF なので `gguf_check.py tensors` を先に） | 720 / 83 |
| ELYZA-japanese-Llama-2-7b-instruct | elyza/ELYZA-japanese-Llama-2-7b-instruct `48fa08b3098a23d3671e09565499a4cfbaff1923` | llama（語彙 32,000） | 6.74B | 4,096 | llama2 | instruct | なし | `.bin` 13.48 GB、PR `refs/pr/2`（`9b96057b…`） | mmnga/…-7b-instruct-gguf `2d708f9c…` q8_0 7.161 GB | 動く見込み | 2,292 / 75 |
| ELYZA-japanese-Llama-2-13b-fast-instruct | elyza/ELYZA-japanese-Llama-2-13b-fast-instruct `2a0b52cd…` | llama | 13B | 4,096 | llama2 | instruct | なし | `.bin` 26.29 GB | mmnga q8_0 13.97 GB | int8 + KV で 16 GiB を越える見込み | 813 / 24 |
| ELYZA-japanese-CodeLlama-7b-instruct | elyza/…-CodeLlama-7b-instruct `a41530e8…` | llama | 6.74B | 16,384 | llama2 | instruct（コード） | なし | 13.48 GB | 未確認 | 動く見込み | 173 / 17 |
| ELYZA-Thinking-1.0-llm-jp-4-33b | elyza/ELYZA-Thinking-1.0-llm-jp-4-33b `6ca556b2ddc4642580752b3a1ae2d7e7681106fc`（2026-10-02） | llama（llm-jp-4 33B の後訓練） | 33.2B | 65,536 | apache-2.0 | thinking | あり | 66.44 GB | mmnga-o/…-gguf `5d192b36…` は Q8_0 35.3 GB だけ | × int8 では大きすぎ。3 ビットの GGUF が出れば llm-jp-4 33B と同じ話 | 1,022 / 12 |
| ELYZA-Thinking-1.0-llm-jp-4-32b-a3b | elyza/…-32b-a3b `5260ecc2…` | qwen3_moe | 32.1B | | apache-2.0 | | | | | **× MoE** | 970 / 9 |
| ELYZA-Thinking / Shortcut 1.0 Qwen-32B | | qwen2 | 32B | | apache-2.0 | | | | | × 大きすぎ | 90・84 |
| ELYZA-Diffusion 1.0 Dream-7B | elyza/ELYZA-Diffusion-Instruct-1.0-Dream-7B `c7c6fe12…` | Dream（拡散） | 7.6B | | apache-2.0 | | | | | **× 拡散** | 247 / 14 |
| Llama-3-ELYZA-JP-8B | 一覧にある | | | | | | | | | | 4,725 / 150 |

小さい ELYZA は今も無い（T.511 の結論のまま）。

#### 5. 松尾研（weblab）・GENIAC（Tanuki）・富岳・NICT

| 名前 | repo / sha | 形 | パラメータ | 文脈 | ライセンス | 種類 | 書式 | safetensors | 最良の GGUF | 動くか | DL / likes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tanuki-8B-dpo-v1.0 | weblab-GENIAC/Tanuki-8B-dpo-v1.0 `e2f8d17243f709043b18c24d88c3d7853749859c`（2024-08-12） | llama 32 層・4096・GQA 8、語彙 65,024（Unigram の tokenizer.json、NFKC + `(?<!\n)^` → ▁、byte_fallback） | 7.51B | 8,192 | apache-2.0 | instruct（DPO） | あり（Alpaca 形「### 指示: / ### 応答:」、既定の system 文あり、**Jinja の `{{` の中に全角空白**） | 15.02 GB | team-hatakeyama-phase2/Tanuki-8B-dpo-v1.0-GGUF `d3e6436bf6256612bf9c7f8647826b326f7e07e8` Q8_0 7.984 GB（4k 版 `bb1fb334…` も同じ大きさ） | 動く見込み | 352 / 44 |
| Tanuki-8x8B-dpo-v1.0 | weblab-GENIAC/Tanuki-8x8B-dpo-v1.0 `4a18628b…` | 独自の `Tanuki`（MoE 8 専門家） | 47B | | apache-2.0 | | | 93.95 GB | | **× MoE** | 117 / 41 |
| weblab-10b-instruction-sft・weblab-10b | matsuo-lab/weblab-10b-instruction-sft `112a5ad9f556078ab14a5cd93511b9db4a0d4413`、matsuo-lab/weblab-10b `d6fc4329…`（2023-08） | gpt_neox 36 層・4864・head 38（128）、rotary_pct 0.25、並列残差、語彙 50,277 | 約 10.7B | 2,048 | cc-by-nc-4.0 | instruct / base | なし（カードの「以下は、タスクを説明する指示です…### 指示: / ### 応答:」を手書き） | 本線は `.bin` 21.42 GB だけ。**PR `refs/pr/4`（`3f4b5eeb74637041e7e50f9ac1907600c7a82c5a`）に safetensors 3 シャード 21.42 GB** | mmnga/matsuolab-weblab-10b-instruction-sft-gguf `364fb870…` q8_0 11.389 GB は **GGUF v1**（読まない。2023 年の NeoX の古い並びでもある） | 形は読める。PR の safetensors を revision に取れば動く見込み（21 GB の取得 → int8 約 11 GB + KV 2.9 GB で 64 ビット）。PR の commit を一覧の revision にしてよいかは持ち主 | 162 / 74 |
| AscleLM-1-10B | weblab-LLM-M/AscleLM-1-10B `4a9cd27c…`（2026-04、医療） | 未確認（config を読めず） | 10.4B | | other | | あり | 41.53 GB（float32） | 無し | 未確認、無名（DL 4） | 4 / 3 |
| weblab-LLM-M の Ramen・MedLLM | | Qwen3-30B-A3B・Next 80B・235B・gpt-oss・GLM の後訓練 | | | | | | | | × MoE・大きすぎ | |
| Fugaku-LLM-13B-instruct | Fugaku-LLM/Fugaku-LLM-13B-instruct `847b9aac4818d73d1e1037216b366656cda7ac4c`（2024-05-07、**ゲート auto**） | gpt2（GGUF の見出し: 40 層・5184・head 36（144）、FFN 20736、eps 1e-5、tokenizer.ggml.model「llama」語彙 49,152、BOS 1・EOS 7） | 13.17B | 2,048 | Fugaku-LLM 利用規約（other、商用可の独自規約） | instruct | 未確認（ゲートで tokenizer_config を読めず） | 26.34 GB | mmnga/Fugaku-LLM-13B-instruct-gguf `de037693c2c69b90ee770507fbf18da93cfe67e8` Q8_0 14.28 GB（作り手の GGUF repo は空） | 形は gpt2 で読める。語彙は原本がゲートなので GGUF だけの道（`from_gguf`: sentencepiece の BPE の点数の形の「llama」だけ読む。Unigram の点数なら断る）か、ゲートの無い写し（solidrust/Fugaku-LLM-13B-AWQ が tokenizer を持つかは未確認）。メモリは int8 ≈14.1 GB + KV f16 1.7 GB で入る見込み | 9 / 29（base は 12 / 134。GGUF は 2,958 DL） |
| NICT | `NICT` は 0 件、`nict-astrec-att` は BART・ByT5・XLM-R の 4 件だけ | | | | | | | | | 公開の日本語 LLM は無し | |

#### 6. 足さないもの・重なり

- **古い版**: Llama-3-Swallow 8B Instruct v0.1、Llama-3.1-Swallow 8B Instruct v0.1〜v0.3（v0.5 が一覧にある）、Swallow 7b-instruct-hf（v0.1 が後継）、llm-jp-3 の instruct・instruct2（instruct3 が後継）、llm-jp-3 13B instruct3（3.1 13B instruct4 が後継）、llm-jp v1/v2。
- **重なり**: Qwen3-Swallow 8B SFT/CPT と Medical は一覧の RL v0.2 と同じ形・同じ元（足すなら Medical か CPT の素の形が違いのある項目）。TinySwallow 素の形は Instruct と同じ形。EvoLLM-JP の 2 つは同じ元のマージで、A が Apache-2.0。
- **MoE（全部 ×）**: llm-jp-4(.1) 32B-A3B、ELYZA-Thinking llm-jp-4-32b-a3b、llm-jp-3 8x1.8B・8x13B・3.1 8x13B、Qwen3-Swallow 30B-A3B、GPT-OSS-Swallow 20B/120B、Swallow-MX 8x7B、Tanuki-8x8B、llm-jp の optimal-sparsity 群。
- **新しい形が要る（×）**: gemma2（Gemma-2-Llama-Swallow 2B・9B・27B: attention の softcap）、evomistral（EvoLLM-JP-v1-10B）、Dream（ELYZA-Diffusion）、moshi（kame・llm-jp-moshi）、gpt_oss。
- **大きすぎ（×、塊でも重い）**: 32〜33B の dense（llm-jp-4(.1) 33B・ELYZA-Thinking 33B・Qwen3-Swallow 32B・ELYZA 32B）、70B 以上、llm-jp-3 172B。33B は IQ3_XXS（13.09 GB）の塊なら 16 GiB に入りうるが、Saluki（27B）級の仕事で、まず 8B が先。

#### 7. 調べで分かった落とし穴

- llm-jp-4.1 8B thinking の Q8_0 は 2026-10-30 も無い（作り手 BF16・Q4_K_M、masahiroid Q4_K_M、ほかは MLX・ONNX・bnb・FP8）。T.250 の待ちはまだ解けていない。**llm-jp-4（4.1 でない）8B thinking には mmnga-o の Q8_0 がある**。
- weblab-10b の mmnga の GGUF は GGUF v1。ELYZA Llama 2 と EvoLLM の mmnga の GGUF は v3（2023〜2024 の古いもの: `gguf_check.py tensors` を先に）。
- weblab と ELYZA Llama 2 の本線は `.bin` だけで、safetensors は変換ボットの PR の枝にしかない（revision に PR の commit を使う）。
- Tanuki の chat_template は `{{　bos_token }}`（U+3000 が Jinja の `{{` の中）: jinja2 の字句解析と自前の読み手が通すかを見る。
- Fugaku-LLM の原本はゲート（auto）: ページは語彙を原本から取れない。
- 13B 級の MHA の llama（llm-jp-3/3.1 13B、Swallow 13B、ELYZA 13B）は int8 と文脈 4096 の KV で 16 GiB の線に当たる（見積もり、未計測。`footprint()` で確かめること）。

## 付録: 調べ手の表（companies）

### 日本の企業の日本語モデル（2026-10-30 の調べ）

出典: HF API（`/api/models/<repo>?blobs=true`・`author=`・`search=`）、各 repo の `config.json`・`tokenizer.json`・`tokenizer_config.json`・README（ヘッダとカードだけ。重みは取っていない）。
作業ファイル: `.tmp/jp-survey/co/`（`orgs.txt`・`search.txt`・`info{1,2,3}.jsonl` が repo ごとの生の事実、`gguf.txt`・`gguf2.txt` が GGUF の一覧）。
dl = API の downloads（直近 30 日）、♥ = likes。sha は調べた時点の main。大きさは GB = 10^9 バイト。
「メモリ」は int8 の重み + 文脈 4096 の KV のおおよそ（MHA で共有メモリなら KV は float16、GQA なら float32: AGENTS.md の規則）。上限 16 GiB = 17.18 GB。

#### A. 今のエンジンで開くはずのもの（勧めの順）

| # | モデル | 作り手 | 原本 repo @ sha | model_type / arch | 規模 | 文脈 | ライセンス | 形 | 書式 | safetensors | 良い GGUF（Q8_0） | 足す仕事・注意 | 有名さ | 日本語 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mistral-Nemo-Japanese-Instruct-2408 | CyberAgent | cyberagent/Mistral-Nemo-Japanese-Instruct-2408 @ 8591f78522a5d651209fc9f354c848508db7a3eb | mistral / MistralForCausalLM（40 層、q32・kv8、head 128） | 12.25B | 1,024,000（ページは 4096） | apache-2.0 | instruct | あり | 24.5 | QuantFactory/…-GGUF @ f85f4a17…（13.02 GB）・mmnga/cyberagent-…-gguf @ 0bd1756e…（13.02 GB） | **前分割が Tekken**（`tokenizer.json` の Split の正規表現、`ignore_merges: true`）: `PRETOKENIZERS` に無い → 型を 1 つ足す（MiniCPM5・Spark と同じ大きさの仕事）。メモリ ≈ 12.3 + KV 1.3 ≈ 13.8 GB で 64 ビットに入る | dl 7,671・♥49（このグループで最多の dl） | Nemo 12B の日本語の続き学習+指示。この大きさの日本語の会話で評判が良い |
| 2 | sarashina2.2-3b（base） | SB Intuitions | sbintuitions/sarashina2.2-3b @ 9a06e768a2156c21bccb96cf4c99ae4b98755577 | llama | 3.36B | 8192 | mit | base | なし | 6.71 | base の Q8_0 は見つからない（instruct は一覧済み） | 一覧の 0.5B・1B base と instruct の梯子の残り 1 段。`tokenizer.model` で一覧の sarashina2.2 と同じ道。safetensors のまま（約 3.8 GB int8） | dl 1,323・♥22 | sarashina2.2 系は小さい割に日本語が強い |
| 3 | calm2-7b-chat | CyberAgent | cyberagent/calm2-7b-chat @ 3e110c7624ff7a940badebfd5f9a91569f98cc56 | llama（MHA 32） | 7.0B | 32,768 | apache-2.0 | chat | **なし**（カードの文 `USER: …\nASSISTANT: `） | 14.02 | TheBloke/calm2-7B-chat-GGUF @ 381297d4…（7.45 GB）・mmnga/cyberagent-calm2-7b-chat-gguf @ e3920e21… | BPE（NFC + Digits + ByteLevel = `gpt2-digits`、変換器が読む）。書式は手書き。base `calm2-7b` は bin だけ（GGUF で） | ♥77・dl 592 | 2023 年の日本語 7B の代表格 |
| 4 | japanese-stablelm-instruct-gamma-7b（と base-gamma-7b） | Stability AI Japan | stabilityai/japanese-stablelm-instruct-gamma-7b @ bb4590d461327a6d63f52a5cdbaa64cfceabbb4a（base @ ffc474da0e4f5dbb9db54e96a94a22c081fc7fbf） | mistral（Mistral 7B v0.1 の続き学習） | 7.24B | 4096 | apache-2.0 | instruct / base | なし（カードの Alpaca 風の日本語の文） | 14.48 | TheBloke/japanese-stablelm-instruct-gamma-7B-GGUF @ 5722f638…（7.7 GB、dl 5,751）・base は RichardErkhov/…base-gamma-7b-gguf @ 67bbafb0…（7.7 GB） | sentencepiece の BPE を書き出した形（Prepend + Replace、byte_fallback）: T.257 の道。書式は手書き | ♥52 / 25 | JSLM の中で一番強い 7B（gamma） |
| 5 | RakutenAI-7B-instruct・RakutenAI-7B（base） | Rakuten | Rakuten/RakutenAI-7B-instruct @ eb48a29043cda2dba50c088556f0ee411badcf2e、Rakuten/RakutenAI-7B @ eb61524c1ab1a7c656457bd68be4a423371cbacb | mistral（語彙 48,000 に拡張） | 7.37B | 32,768 | apache-2.0 | instruct / base | なし（カードの `USER: … ASSISTANT:`） | 14.75 | RichardErkhov/Rakuten_-_RakutenAI-7B-instruct-gguf @ 079d7aef…（7.84 GB）・mradermacher/RakutenAI-7B-GGUF @ 1923f28f…（7.84 GB） | chat は一覧済み（`hf-rakutenai-7b-chat`）。同じ道で 1 行ずつ | ♥51・52 | 一覧の chat と同じ系統 |
| 6 | llama-3-youko-8b-instruct・llama-3-youko-8b | rinna | rinna/llama-3-youko-8b-instruct @ 681390a11903461d2066dbb68ab8bc7f66f3eb67、base @ 6af890bc6294b8f311e25a1462fcd38554d95b8e | llama（Llama 3 8B） | 8.03B | 8192 | llama3 | instruct（書式あり）/ base | instruct はあり | 16.06 | mradermacher/llama-3-youko-8b-instruct-GGUF @ 3574c586…（8.54 GB）・mmnga/rinna-llama-3-youko-8b-gguf @ 387dc680… | 今のまま開く見込み（Llama 3 の道は一覧に多数）。"Built with Meta Llama 3" の表示の件（T.249 の持ち主の判断 (b)） | ♥62（base）・15 | rinna の Llama 3 続き学習 |
| 7 | youri-7b-chat・-instruction・youri-7b | rinna | rinna/youri-7b-chat @ 12c3e0e288efbdb525aa98abcab0c92d8445fcbc、-instruction @ b82221c9eaeee6fe41dd08cb27a1887bbe061c5c、base @ 507265585a74cc5730ce7961c22252057f5325f9 | llama（Llama 2 7B、語彙 32,000 のまま） | 6.74B | 4096 | llama2 | chat / instruct / base | なし（`ユーザー:`/`システム:` の手書き） | 13.48 | mradermacher/youri-7b-chat-GGUF @ 689eb667…・-instruction-GGUF @ 3d7edca5…・youri-7b-GGUF @ 951b85a0…（各 7.16 GB） | 開く見込み。語彙が日本語に広げていないので 1 字あたりのトークンが多い（遅い） | ♥14〜21 | 2023-10 の rinna の主力 |
| 8 | japanese-gpt-neox-3.6b-instruction-ppo・-sft-v2 | rinna | rinna/japanese-gpt-neox-3.6b-instruction-ppo @ 018e1f11cfe7ceeaad26c5c6f7dc570281516a51、-sft-v2 @ eef8e9298286ecf785fe4ce92218a7d73b4a3e9c | gpt_neox | 3.76B | 2048 | mit | instruct | なし | 7.37 | mmnga/rinna-japanese-gpt-neox-3.6b-instruction-ppo-gguf @ 7944c651…（3.84 GB、2023 の GGUF: LINE・stockmark と同じく `attn_qkv` が HF の並びの恐れ → safetensors を勧める） | `spiece.model`（一覧の rinna neox small と同じ道）。**書式が改行を `<NL>` に置き換える**（カード: 「tokenizer は "\n" を知らない」）: 入力と出力の改行の置き換えが要る（ページに小さな仕組み、項目の options）。`ユーザー: …<NL>システム: ` | ♥74（ppo）・26 | 2023 年の日本語チャットの代表。古い |
| 9 | bilingual-gpt-neox-4b-instruction-ppo（と -sft・base・-8k） | rinna | rinna/bilingual-gpt-neox-4b-instruction-ppo @ 512ee77fef7a6081bb195cbd214237686b2ab194、base @ 803fb7671ac30766ffc6d21139d809b549ee26a3 | gpt_neox | 3.95B | 2048（8k は config も 2048） | mit | instruct / base | なし | 7.74 | mmnga/rinna-bilingual-gpt-neox-4b-instruction-ppo-gguf @ de4847b9…（4.04 GB、2023 の GGUF） | 8 と同じ（`<NL>` あり、`spiece.model`、語彙 65,536） | ♥14〜29 | 日英 |
| 10 | stockmark-13b・stockmark-13b-instruct | Stockmark | stockmark/stockmark-13b @ 77a620e2c3c7efd970c02130465938432b53df8b、-instruct @ cd0e71911752ed5dcb9c429a1695574215aa876e | llama（MHA 40） | 13.2B | 2048 | mit | base / instruct | なし | 26.4 | TheBloke/stockmark-13B-GGUF @ 19a8a6b7…（14.03 GB）・mmnga/stockmark-13b-gguf @ 0b3df1ad… | `gpt2-digits` + NFKC で読める。メモリ ≈ 13.2 + KV（MHA、文脈 2048、f16）1.7 ≈ 15 GB: 64 ビットの上限の近く（要計測） | ♥40 | ビジネス文書の続き学習（スクラッチ） |
| 11 | sarashina2-7b | SB Intuitions | sbintuitions/sarashina2-7b @ 15b989b8a4eb6f68b990787fc1016def7dabf0bb | llama（MHA） | 7.3B | 4096 | mit | base | なし | **bin だけ**（14.63） | QuantFactory/sarashina2-7b-GGUF @ 83d29cb7…（7.78 GB） | GGUF の道で（段 ② で語彙は原本の `tokenizer.model`）。MHA で KV が大きい | dl 1,990・♥28 | スクラッチの日本語 7B |
| 12 | sarashina2.1-1b | SB Intuitions | sbintuitions/sarashina2.1-1b @ 30348c324bc3d1b7b33d9687ca20478b2e5b3dd9 | llama（GQA 16/8） | 1.41B | 8192 | **Sarashina Model NonCommercial License**（LICENSE の 1 行目で確認） | base | なし | 2.82 | mradermacher/sarashina2.1-1b-GGUF @ 9f977c5c…（1.5 GB） | 開く見込み。小さく電話にも。2.2 の 1B と並ぶので優先は低め | ♥21 | 2.2 の前の版 |
| 13 | CAT-Translate-7b | CyberAgent | cyberagent/CAT-Translate-7b @ e36edfc6858a9a6ba41f8d9fa34427f3d8461eca | mistral | 7.49B | 32,768 | mit | 翻訳 | あり | 14.97 | mradermacher/CAT-Translate-7b-GGUF @ cd7d970b…（7.96 GB）・mmnga-o/CAT-Translate-7b-gguf @ 21aa7986… | 一覧の 0.8b・1.4b・3.3b の梯子の最上段。トークナイザは T.257 で既に読めると記録（AGENTS.md「CAT-Translate 7B」） | dl 692・♥16 | TranslateGemma-12B より良いとカード |
| 14 | CAT-Paws-8B（2026-06） | CyberAgent | cyberagent/CAT-Paws-8B @ 37034349462270dfe4553e0e75f74d60c2e8c88b | qwen3 | 8.19B | 40,960 | apache-2.0 | 考える（エージェント向け） | あり | 16.38 | mradermacher/CAT-Paws-8B-GGUF @ fc62659e…（8.71 GB） | 開く見込み。カード自身が単発の問いには一覧済みの CAT-Thinking-8B を勧める → 優先は低い | ♥3 | 日本語で考える |
| 15 | ABEJA-Qwen2.5-7b-Japanese-v0.1 | ABEJA | abeja/ABEJA-Qwen2.5-7b-Japanese-v0.1 @ b0027a80acb4052482d4b55499100f568e937644 | qwen2 | 7.62B | 32,768 | apache-2.0 | instruct | あり | 15.23 | mmnga/ABEJA-Qwen2.5-7b-Japanese-v0.1-gguf @ e9edf6f2…（8.1 GB） | 開く見込み（T.81 でも「通った」） | ♥10 | GENIAC の成果 |
| 16 | DeepSeek-R1-Distill-Qwen-14B-Japanese | CyberAgent | cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese @ cc1ebcc1e37521c3f4f69145ce922ca8f01e4cd0 | qwen2（48 層、q40・kv8） | 14.77B | 131,072 | mit | 考える | あり | 29.54 | mmnga/cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-gguf @ a8f469dc…（15.7 GB） | **上限すれすれ**: int8 14.8 + KV f32 1.6 ≈ 16.4 GB（KV を f16 にすれば ≈ 15.6）。Q6_K/Q4_K_M の塊のまま（Saluki の道）なら余裕 | ♥94 | 32B 版は ♥254 で有名（32B は大きすぎ） |
| 17 | ABEJA-Qwen3-14B-Agentic-256k-v0.1（2026-02） | ABEJA | abeja/ABEJA-Qwen3-14B-Agentic-256k-v0.1 @ c18c202ab2deefede0e50eabb8f3ede832eb9cdb | qwen3（yarn ×8） | 14.77B | 262,144 | apache-2.0 | 考える（エージェント） | あり | 29.54 | mmnga-o/ABEJA-Qwen3-14B-Agentic-256k-v0.1-gguf @ 4bda31b7…（15.7 GB） | 16 と同じくすれすれ。yarn はページの文脈 4096 では Bonsai と同じく素の RoPE が良いかも（T.235 の件） | ♥8 | エージェント用途 |
| 18 | Qwen3-1.7B-pfn-qfin・Qwen2.5-1.5B-pfn-qfin | PFN | pfnet/Qwen3-1.7B-pfn-qfin @ 3e454cbe406cc778feb6f2e9bc4b4397e5cc98bf、pfnet/Qwen2.5-1.5B-pfn-qfin @ ee0e12a5f3bae9011b3ffad6840bcf1a0a7f4e6c | qwen3 / qwen2 | 1.72B / 1.54B | 32k / 128k | plamo-community-license | 金融の続き学習（base 由来、書式はあり） | あり | 3.44 / 3.09 | GGUF なし → safetensors | 開く見込み。小さく珍しい（金融）。dl 747・616 | ♥0 | 金融ドメイン |

### A の続き: 日本の企業だが別グループかもしれないもの（重なりは取りまとめ役が除く）

| モデル | 原本 @ sha | 形 | 規模・文脈 | ライセンス | 形態 | Q8_0 GGUF | 注意 | dl・♥ |
|---|---|---|---|---|---|---|---|---|
| ELYZA-japanese-Llama-2-7b-fast-instruct（と -instruct、13b-fast-instruct） | elyza/ELYZA-japanese-Llama-2-7b-fast-instruct @ 89de33d1ad568855853196802aeaecd799c6586f（-instruct @ 48fa08b3098a23d3671e09565499a4cfbaff1923） | llama | 6.85B・4096 | llama2 | instruct、書式なし（Llama 2 の `[INST]`） | mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf @ 0f0072a8…（7.27 GB） | 原本は bin だけ → GGUF の道。13B は 13.0B で MHA の KV が大きく上限の近く | dl 720・♥83（instruct ♥75） |
| ELYZA-Shortcut-1.0-Qwen-7B | elyza/ELYZA-Shortcut-1.0-Qwen-7B @ d33448648d9b1b7d69cbadb3f1a8d6fb4c8a01ee | qwen2 | 7.62B・32k | apache-2.0 | 考えずに答える（Thinking の蒸留） | mmnga/ELYZA-Shortcut-1.0-Qwen-7B-gguf @ fc1fb435…（8.1 GB） | T.511 でも「開く見込み」 | dl 533・♥3 |
| EvoLLM-JP-v1-7B・EvoLLM-JP-A-v1-7B | SakanaAI/EvoLLM-JP-v1-7B @ a7acd2604d3e9f0fca2987d2448ff5dc19778c21・-A @ a8b1356458c9883147c3aeffc1a2e51c99bfe70b | mistral | 7.24B・32k | v1: other（非商用）、A: apache-2.0 | 指示（マージ） | mmnga/SakanaAI-EvoLLM-JP-v1-7B-gguf @ e7df4f20…・-A @ 3c89db3c…（7.7 GB） | 開く見込み。10B は `evomistral`（層の繰り返し）で読めない | ♥36・13 |
| TinySwallow-1.5B（base）・TAID-LLM-1.5B | SakanaAI/TinySwallow-1.5B @ 535d1322615c9f8171c3d60b10ef6955cc896fe9・TAID @ 2e649507bc2da0c99fc6817731f3311e65c36964 | qwen2 | 1.54B・32k | apache-2.0 | base / instruct | なし（instruct の GGUF は一覧済み） | 安い追加 | ♥35・6 |
| Llama-3-Karamaru-v1（江戸の言葉） | SakanaAI/Llama-3-Karamaru-v1 @ 17411929aa0ef76daa51e7c0d0b986d65c877a58 | llama | 8.03B・8192 | llama3 | chat | mradermacher/Llama-3-Karamaru-v1-GGUF @ 84ad8f40…（8.54 GB） | 話題性はある | ♥19 |
| Tanuki-8B-dpo-v1.0（GENIAC 松尾研） | weblab-GENIAC/Tanuki-8B-dpo-v1.0 @ e2f8d17243f709043b18c24d88c3d7853749859c | llama | 7.51B・8192 | apache-2.0 | chat（書式あり） | team-hatakeyama-phase2/Tanuki-8B-dpo-v1.0-GGUF @ d3e6436b…（7.98 GB） | Unigram + llm-jp と同じ「(?<!\n)^」の normalizer（読める型）。スクラッチ | ♥44 |
| suzume-llama-3-8B-japanese・DeepSeek-R1-Distill-Qwen-7B-Japanese | lightblue/suzume-llama-3-8B-japanese @ 3306e6bac49e1bd80bfe54001994a828ceeb67fa・lightblue/DeepSeek-R1-Distill-Qwen-7B-Japanese @ f5f497038d5da4221631058ff1e9b9563fe539d2 | llama / qwen2 | 8.0B / 7.6B | llama-3 / apache-2.0 | chat / 考える | mmnga/lightblue-suzume-llama-3-8B-japanese-gguf @ f16a01d0…（8.54）・mmnga/lightblue-DeepSeek-R1-Distill-Qwen-7B-Japanese-gguf @ d5a63bd4…（8.1） | 開く見込み（Lightblue は東京の会社） | ♥25・33（multilingual 版 ♥114） |
| ArrowPro-7B-KUJIRA・ArrowCanaria-Llama-8B-SFT-v0.1（2026-03） | DataPilot/ArrowPro-7B-KUJIRA @ 755e478ac9404cdd551126e20ea055dfa42ddd1d・DataPilot/ArrowCanaria-Llama-8B-SFT-v0.1 @ aaab5f55825af2720d574ef40123da29d920b4dd | mistral / llama | 7.2B / 8.0B | apache-2.0 / llama3.3+gemma | instruct | mmnga/DataPilot-ArrowPro-7B-KUJIRA-gguf @ 257e00a4…（7.7）・mmnga-o/ArrowCanaria-Llama-8B-SFT-v0.1-gguf @ d0818631…（8.54） | 開く見込み | ♥64・23 |
| EZO-Llama-3.2-3B-Instruct-dpoE・Llama-3.1-8B-EZO-1.1-it・Qwen3-EZO-8B-beta | AXCXEPT/EZO-Llama-3.2-3B-Instruct-dpoE @ 0b374d6e4b55192d833ec5b08ca842aaac4147eb・Llama-3.1-8B-EZO-1.1-it @ 0739a830486ce958412070cd1d2ce8e53ef2209f・Qwen3-EZO-8B-beta @ c7458182662d2fa15da5e41a17427319e6f265f5 | llama / llama / qwen3 | 3.2B / 8.0B / 8.2B | llama3.2 / llama3.1 / apache-2.0 | instruct | MaziyarPanahi/EZO-Llama-3.2-3B-Instruct-dpoE-GGUF @ 01d8df8d…（3.42）; Qwen3-EZO-8B-beta の Q8_0 は見つからず | 開く見込み | ♥10・18・9 |

#### B. 新しい形が要るもの（形ごとにまとめた）

| 形（model_type） | 開くモデル（作り手・原本 @ sha・規模・ライセンス・Q8_0 GGUF） | 仕事の大きさ |
|---|---|---|
| **plamo3**（PLaMo 3。2025-10〜） | pfnet/plamo-3-610m-fin-instruct @ a000b59209f1688322c0804009dc9f3fd67e62c1（0.89B、文脈 262,144、PLaMo community、instruct・考える `<\|plamo:begin_think:plamo\|>`、safetensors 1.78、mradermacher/plamo-3-610m-fin-instruct-GGUF @ 71155204…（0.95 GB）、dl 727）; pfnet/plamo-3-nict-2b-base @ 8abb6747cc5763e66b6c3c1a7026f8a0cfc7288d（2.6B、**ゲート auto**、mmnga-o/plamo-3-nict-2b-base-gguf @ 0b028e35…（2.77 GB、dl 2,582））; pfnet/plamo-3-nict-8b-base @ 70068110b56247cb92594e707488009b98897295（8.1B、ゲート、mmnga-o/plamo-3-nict-8b-base-gguf @ e4b163b3…（8.6 GB））; 31B（2 つ）は大きすぎ | **中**。層は Gemma 3 にとても近い（`modeling_plamo.py` を読んだ）: 前と後ろの norm（pre/post mixer・pre/post MLP）、q・k の head ごとの RMSNorm、窓 2048 の局所の層と 8 層ごとの全体の層（`sliding_window_pattern` 8、局所の RoPE base 1e4、全体は 1e6 + yarn ×256）、GQA（610M は kv 1）。違いは SwiGLU（`gate_up_proj` が 1 つ）・norm の重みに足す定数（1、後ろの norm は 1/5 と 1/5^1.5: 変換器で畳める）・埋め込みの倍率なし。**重い所はトークナイザ**: `tokenizer.jsonl`（PLaMo 独自、Aho-Corasick の unigram + バイト）。llama.cpp には PLaMo2 の語彙の型があり GGUF に載る。PLaMo 2 と共通のトークナイザなので、作れば B の次の行も楽になる |
| **plamo2**（PLaMo 2、Mamba2 との混合） | pfnet/plamo-2-translate @ 7546b474fc8eba5e8354d4634770130428e9dd68（9.53B、PLaMo community、翻訳、♥128・dl 1,358、mmnga/plamo-2-translate-gguf @ adf2a564…（10.13 GB）、mitmul/plamo-2-translate-GGUF）; pfnet/plamo-2-1b @ 92c75fd6eea9018bcb9c33ee8921589febe071fa（1.29B、apache-2.0、base、♥40、mmnga/plamo-2-1b-gguf）; pfnet/plamo-2.1-2b-cpt @ 07d01cbc359dc2748ac3fd56eb55efd9ef7f3a00（2.45B、base）; plamo-2-8b（ゲート） | **大**。Mamba2 の SSM の層（状態を持つ層の種類を 1 つ足す: 手順の「層の種類を足す」）+ PLaMo のトークナイザ。plamo-2-translate は日本の翻訳で一番有名なので、価値は高い |
| **gemma2** | rinna/gemma-2-baku-2b-it @ b6e5db993e122fe53557f5364a21ef30a9e67258（2.61B、gemma、書式あり、♥30、QuantFactory/gemma-2-baku-2b-it-GGUF @ e9a1b7c6…（2.78 GB））と base @ 62e400a695edae0d84a4783bc5005d37d5fe9e3f; AXCXEPT/EZO-Common-9B-gemma-2-it（9.24B、♥33）、EZO-gemma-2-2b-jpn-it; （別グループ: google/gemma-2-2b-jpn-it） | **小〜中**。Gemma 3 から q・k の norm を外し、attention の logit softcap（50）と最後の softcap（30。Gemma 4 にある）、局所と全体を 1 層おき（窓 4096）、√head でなく `query_pre_attn_scalar`。GPU は softcap の分 CPU のまま（`REASONS.pair`） |
| **stablelm / stablelm_epoch** | stabilityai/japanese-stablelm-3b-4e1t-instruct @ 4bcd24cf3397c4563062e7d2897c2fa5ea16acc2（2.80B、apache-2.0、♥31、mmnga/japanese-stablelm-3b-4e1t-instruct-gguf @ b82bb95f…（2.97 GB））と base @ e7d2ebdd200a28be321058be9a67514780073736; stabilityai/japanese-stablelm-2-instruct-1_6b・base-1_6b（1.64B、**ゲート auto**、Stability の JSLM2 の許諾、config は取れず） | **中**。LayerNorm（bias 付き）+ head の 25% だけ回す RoPE（NeoX の `rotary_pct` と同じ）+ SwiGLU。GPT-NeoX と Llama の部品の組み合わせ（T.81 の読み）。JSLM2 は q・k・v に bias |
| **qwen**（Qwen 1、`QWenLMHeadModel`） | rinna/nekomata-7b-instruction @ 97eb98107ee62d9c4fc0a0e8df937cb2f06e2922（7.72B、tongyi-qianwen、mradermacher/nekomata-7b-instruction-GGUF @ af373b28…（8.21 GB））・nekomata-7b @ 51aac119c45eaf386923e945e7bd4b6d1d816d2f・nekomata-14b(-instruction)（14.17B、上限の外: int8 + KV で 17 GB 超、塊なら可）; pfnet/nekomata-7b-pfn-qfin | **小〜中**。層は Qwen2 とほぼ同じ（c_attn が 1 つで bias 付き、`logn` 等は文脈 4096 以下なら効かない）。トークナイザが `qwen.tiktoken`（tokenizer.json が無い）: tiktoken のファイルを読む道が要る。GGUF の道なら語彙も GGUF に入る（GGUF だけの変換、T.505）が、BPE の前分割の名前 `qwen` の突き合わせが要る |
| **plamo**（PLaMo 1） | pfnet/plamo-13b @ 7f9b5577fa24b59f00afeafcd8078a9ab19ca9f7（13.1B、apache-2.0、♥86）・plamo-13b-instruct @ 1d47ec900f6800f49ee38537ee3b1c79768973a2（文脈 8192） | **中**、しかも 13B の MHA で上限の近く。低い |
| **gpt_neox_japanese** | abeja/gpt-neox-japanese-2.7b @ c4958c0a96d523dd8841fc38da0741056ce470ec（2.7B、mit、bin だけ、dl 5,176・♥59） | **中〜大**（独自の層の bias の置き方、独自のトークナイザ `SubwordJapaneseTokenizer` と絵文字の表）。古い base で価値は低い |
| JapaneseStableLMAlpha（独自） | stabilityai/japanese-stablelm-base-alpha-7b（♥120）・instruct-alpha-7b-v2 | NeoX 風の独自コード + NovelAI の sentencepiece。beta・gamma があるので不要 |
| evomistral | SakanaAI/EvoLLM-JP-v1-10B | 層の繰り返しとスケール。低い |
| mamba | kotoba-tech/kotomamba-2.8B-v1.0（dl 21） | 不要 |

#### C. 大きすぎる・MoE・その他（足さない）

- **MoE**: Rakuten/RakutenAI-3.0（`deepseek_v3`、671B、256 専門家、2026-03、apache-2.0、♥77。RakutenAI の最新版だがこの頁では無理）、RakutenAI-2.0-8x7B(-instruct)（mixtral 46.8B）、karakuri-lm-8x7b-chat/instruct、sbintuitions/sarashina2-8x70b（465B）、elyza/ELYZA-Thinking-1.0-llm-jp-4-32b-a3b（qwen3_moe）、weblab-GENIAC/Tanuki-8x8B、abeja の Mixtral 4 つ、DataPilot/ArrowNeo-AME-4x3B。
- **大きすぎる（int8 で 16 GiB を越える）**: cyberagent/calm3-22b-chat（22.5B、apache-2.0、♥85。**塊なら候補**: grapevine-AI/CALM3-22B-Chat-GGUF @ 327049fb… に Q4_K_M（約 13〜14 GB の見込み、Q6_K の層は int8 に広がる）。Saluki の道で載るかは要計測。Q8_0 は無い）; rinna の qwen2.5-bakeneko-32b(-instruct)、deepseek-r1-distill-qwen2.5-bakeneko-32b、qwq-bakeneko-32b（qwen2 32.8B、apache-2.0。qwq-bakeneko ♥21。Q4_K_M の塊で約 20 GB: 入らない、Q2_K/IQ3 なら可能性）; cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese（♥254、同上）; cyberagent/Llama-3.1-70B-Japanese-Instruct-2407; llama-3-youko-70b; karakuri-lm-70b、karakuri-lm-32b-thinking-2501-exp; stockmark-100b、Stockmark-2-100B-Instruct（2025-07、llama 96B）; sarashina2-70b、sarashina1-65b; plamo-100b; plamo-3-nict-31b; ELYZA-Thinking-1.0-llm-jp-4-33b（2026-10、Q8_0 35.3 GB）・-Qwen-32B; ABEJA-Qwen2.5-32b、ABEJA-QwQ32b; Preferred-MedLLM-Qwen-72B など。
- **上限の境**: sbintuitions/sarashina2-13b @ 07b16d21dc7cf0115283312fef8c348c946d165b（13.7B、mit、MHA 40、文脈 4096: int8 13.7 + KV f16 3.4 ≈ 17.1 GB で入らない見込み。int6 か Q4_K の塊なら可。RichardErkhov/sbintuitions_-_sarashina2-13b-gguf @ 6b441194… の Q8_0 14.6 GB）; nekomata-14b; ELYZA-japanese-Llama-2-13b; plamo-13b。
- **言語モデルでない・用途違い**: karakuri-lm-7b-apm-v0.1/v0.2（応答の属性を 0〜4 で採点するモデル: カード）、SakanaAI/kame（音声対話 Moshi）、cyberagent/markupdm（SVG の生成）、karakuri-vl・sarashina2.2-vision/ocr/tts・plamo-2.1-vl（視覚・音声）、sbintuitions/diafill-*（対話の穴埋め、bin 13.4 GB の float32）、cyberagent/ca-reward-3b-ja（報酬モデル）。
- **古い・重複で勧めない**: cyberagent/open-calm-*（small 〜 7b、gpt_neox、CC-BY-SA-4.0、**bin だけ**、open-calm-7b ♥205 で有名。mmnga/cyberagent-open-calm-{1b,3b,7b}-gguf の 2023 の GGUF（1b 1.52・3b 2.96・7b 7.31 GB）は qkv の並びの恐れ。トークナイザは NFKC + ByteLevel の BPE で読める。base だけで質は低い: 有名さのために 1 つ足すなら 7b を GGUF の確かめの後に）; sarashina1-7b/13b（gpt_neox、bin だけ）; japanese-stablelm-*-beta-7b（Llama 2。gamma のほうが良い。instruct-beta は TheBloke の Q8_0 7.16 GB で開く見込み）・ja_vocab-beta; youri の GPTQ・nekomata の各量子化; abeja/gpt2-large-japanese（gpt2、bin だけ、GGUF なし、`T5Tokenizer` の spiece）; lightblue/karasu-1.1B（bin だけ、TinyLlama）; sbintuitions/tiny-lm-chat（4 層 256 幅、bin だけ 0.06 GB。サイトの tiny-lm の chat 版で、safetensors に直せば最小のチャットとして面白い: 持ち主の判断）。
- **消えた**: rinna/japanese-gpt-neox-3.6b（base）と -instruction-sft（v1）、rinna/qwen2.5-bakeneko-32b-instruct-v2 は 401（非公開か削除。org の一覧にも無い）。mmnga/rinna-japanese-gpt-neox-3.6b-gguf（@ 90e4e71d…）に base の写しが残る。
- **重みを出していない企業**: NTT（tsuzumi / tsuzumi 2）、富士通（Takane）、NEC（cotomi）、リコー、Kotoba（kotomamba だけ）、LY Corp（line-corporation は 2023 の japanese-large-lm 4 つだけで、全部一覧済み。新しい版なし）。HF の `author=` で 0 件か言語モデルなし（2026-10-30）。

#### D. 新しい版（一覧にあるものの）

- RakutenAI: 一覧は 7B-chat と 2.0-mini-instruct。新しいのは RakutenAI-3.0（MoE、開けない）と RakutenAI-2.0-mini の base（Rakuten/RakutenAI-2.0-mini @ 3740b0473010eb05d59143858408358bead777ba、mistral 1.53B、文脈 131k、apache-2.0、書式あり、safetensors 3.07、GGUF なし）。
- sarashina2.2: 新しい言語モデルは無い（2025-03 の v0.1 のまま。以後は vision・ocr・tts・embedding）。
- CAT: 2026-02 の CAT-Translate-7b と 2026-06 の CAT-Paws-8B が一覧に無い。CAT-Translate-3.3b-beta（@ b596062e…）は 3.3b の前の版で不要。
- LINE / LY: 新しい版なし。
- rinna: 2025-03 の bakeneko 32B の後に新しい公開なし。

## 付録: 調べ手の表（community）

### Japanese community/startup fine-tunes, global JP releases, HF ja top lists (survey 2026-10-30)

Sources: HF API (`/api/models/<repo>?blobs=true`, `resolve/main/config.json`, `tokenizer_config.json`), the ja lists
(`language=ja`, sort downloads / likes / trendingScore, text-generation and image-text-to-text, top 300 each), author lists,
the Swallow LLM leaderboard (post-trained table), and a third-party write-up of Nejumi 4. Raw dumps are in
`.tmp/jp-survey/community/` (facts1-4.txt, gguf1-2.txt, authors1-2.txt, lang_*.txt).
Sizes are bytes/1e9. "int8 ≈" is the Q8_0 size (the int8 the page loads is about the same). The 16 GiB cap is 17.18 GB.
Downloads are HF's last-30-day count.

#### Leaderboards (≤ 14B open models)

- **Swallow LLM leaderboard (post-trained, Japanese average / JA MT-Bench)**: Qwen3.5-9B 0.534/0.707 (listed) · **Qwen3-14B 0.524/0.712 (not listed)** · Qwen3-Swallow-8B-RL-v0.2 0.510 (listed) · **NVIDIA-Nemotron-Nano-9B-v2-Japanese 0.503/0.689 (not listed, unsupported arch)** · Qwen3-8B 0.491 (listed) · Gemma 4 E4B 0.484/0.758 (listed) · Qwen3.5-4B 0.456 · llm-jp-4-8b-thinking 0.448 · Gemma 4 E2B 0.431/0.726 · **Gemma 3 12B IT 0.405/0.659 (not listed)** · Llama 3.1 Swallow 8B v0.5 0.341 · **Gemma 3 4B IT 0.292/0.572 (not listed)** · Sarashina2.2 3B 0.260 · llm-jp-3.1-13b-instruct4 0.258 (other group) …
- **Nejumi 4** (third-party report only, primary table not retrievable): Nemotron-Nano-9B-v2-Japanese #1 in the sub-10B class; Qwen3.5 the top sub-10B score 0.711. Both conflict in the source; unverified.
- **Shaberi v2.1 / JA MT-Bench (Shisa's own card)**: shisa-v2.1-unphi4-14b 7.71 / 7.87 · shisa-v2.1-qwen3-8b 7.35 / 7.78 (listed) · shisa-v2-llama3.1-8b 6.43 · shisa-v2.1-llama3.2-3b 6.23 (listed) · shisa-v2.1-lfm2-1.2b 5.35 (listed). On the GPT-4-Turbo judge, shisa-v2-qwen2.5-7b 8.16 and shisa-v2-llama3.1-8b 7.97.

#### A. Run today (a model_type and tokenizer the converter reads): ranked by how much they are worth adding

| # | Model | Maker | Original repo @ sha | model_type / arch | Params | Ctx | License | Kind | Template | safetensors | Best GGUF (Q8_0) | Runs? | Fame | JA note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LFM2-350M-ENJP-MT | Liquid AI | LiquidAI/LFM2-350M-ENJP-MT @ 80367784d525777ad7565b24534ba5810eeac59f | lfm2 / Lfm2ForCausalLM | 354M | 128000 | LFM Open License v1.0 | instruct (JA⇄EN translation) | yes | 0.71 GB | LiquidAI/LFM2-350M-ENJP-MT-GGUF @ 889c88c3d3681b4f342ae210a1d2a1369aba0e10, LFM2-350M-ENJP-MT-Q8_0.gguf 0.38 GB (Liquid AI) | yes. The entry needs a hand-written system prompt ("Translate to Japanese." / "Translate to English.") like the CAT-Translate entries. Small. | 720 dl, 98 likes; GGUF 2.6k dl | The card claims translation on par with models 10× its size. Phone-sized. |
| 2 | shisa-v2.1-unphi4-14b | Shisa.AI | shisa-ai/shisa-v2.1-unphi4-14b @ fb2ae1891b6296fc2ecb4c2399912ef2caed4719 | llama / LlamaForCausalLM (phi-4 rewritten as Llama) | 14.66B | 16384 | MIT | instruct | yes | 29.32 GB | mradermacher/shisa-v2.1-unphi4-14b-GGUF @ 3f3044899c094d10d71e4a2c51c85c77cd330633, shisa-v2.1-unphi4-14b.Q8_0.gguf 15.58 GB (mradermacher) | Probably yes. Llama with 40 layers, 40 heads, 10 KV heads, head 128, untied vocab 100352, rope θ 250000. tokenizer.json is the cl100k regex (the same as the converter's `llama3` type) with `ignore_merges: false`. **Memory is tight**: int8 ≈ 15.6 GB plus KV (float16 at 4096: 0.84 GB) ≈ 16.5 GB against the 17.18 GB cap. Run `footprint()` first. The GGUF has no `rope_freqs`, and its pre-tokenizer name is unchecked: use the original's vocabulary. | 88 dl, 6 likes; GGUF 2.4k dl | Shisa's best ≤14B: JA AVG 72.6, Shaberi 7.71, JA MT-Bench 7.87 (beats the listed qwen3-8b). |
| 3 | Spark-X2.5-4B-Japanese (unofficial) | Takenoko12345678 (individual; also made the listed Qwen3.5-0.8B-Japanese-SFT) | Takenoko12345678/Spark-X2.5-4B-Japanese @ bfb70f0adf24aa4fb155f2a8481e1fa5c63766c1 | spark2_5 / Spark2_5ForCausalLM | 4.11B | 1M | Apache-2.0 | thinking + instruct (same as Spark-X2.5) | yes | 8.22 GB | Takenoko12345678/Spark-X2.5-4B-Japanese-GGUF @ 79fd825680c88ba3b4f823202f0ef94aa90ba006, Spark-X2.5-4B-Japanese-Q8_0.gguf 4.38 GB (maker) | yes (same arch and tokenizer as the listed Spark-X2.5 4B). | 497 dl; GGUF 438 | The card says Chinese leakage fell from 18% to 0% (SFT plus distillation from Qwen3.8-27B). Fixes the listed Spark's weak point. |
| 4 | DeepSeek-R1-Distill-Qwen-7B-Japanese | Lightblue | lightblue/DeepSeek-R1-Distill-Qwen-7B-Japanese @ f5f497038d5da4221631058ff1e9b9563fe539d2 | qwen2 | 7.62B | 131072 | Apache-2.0 | thinking (in Japanese) | yes | 15.23 GB | mmnga/lightblue-DeepSeek-R1-Distill-Qwen-7B-Japanese-gguf @ d5a63bd4cb76aec396fad7602a72352cc02e6006, …-Q8_0.gguf 8.10 GB (mmnga) | yes (same as the listed R1 distill 7B: GGUF pre `deepseek-r1-qwen` is known) | 131 dl, 33 likes | Thinks in Japanese, unlike the listed English-thinking R1 distills. |
| 5 | DeepSeek-R1-Distill-Qwen-1.5B-Multilingual | Lightblue | lightblue/DeepSeek-R1-Distill-Qwen-1.5B-Multilingual @ 9c9c7b64e4d7e2154db8ee4cae0e015d7fac5e82 | qwen2 | 1.78B | 131072 | Apache-2.0 | thinking in the user's language | yes | 3.55 GB | mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-Multilingual-GGUF @ 06eaa38aae801e83d2887cb2d06d4f84e8716f0e, …Q8_0.gguf 1.89 GB | yes | 51 dl, 24 likes | Small thinker that reasons in Japanese. The 7B-Multilingual (31108fec…) is a sibling of #4: skip one of them. |
| 6 | ArrowCanaria-Llama-8B-RL-v0.1 (and its SFT) | DataPilot | DataPilot/ArrowCanaria-Llama-8B-RL-v0.1 @ 2b487b422b5a30dbe5f4127cd6b60a15d25ad226 (SFT: aaab5f55825af2720d574ef40123da29d920b4dd) | llama | 8.03B | 131072 (card: 4096 trained) | Llama 3.1 Community License + Gemma Terms | instruct (chat, RP, empathy) | yes | 16.06 GB | mradermacher/ArrowCanaria-Llama-8B-RL-v0.1-GGUF @ ec04d0610d5ef9a7d6c2e01e814023db6b3e7b9d, …Q8_0.gguf 8.54 GB. SFT: mmnga-o/ArrowCanaria-Llama-8B-SFT-v0.1-gguf @ d08186310e1d7c605042c6963deeb135465d7ab6, 8.54 GB | yes (base is the listed Llama-3.1-Swallow-8B v0.5) | SFT 134 dl / 23 likes, RL 133 / 7 (2026-03) | Natural conversational Japanese ("AItuber" use); GRPO on top of Swallow. |
| 7 | Lumi-v2-llm-jp-4-8b | SousiOmine (individual) | SousiOmine/Lumi-v2-llm-jp-4-8b @ 0af15b25d14de36972c39501e883da7f07efbe61 | llama | 8.59B | 65536 | Apache-2.0 | instruct + thinking SFT (ChatML) | yes | 17.18 GB | SousiOmine/Lumi-v2-llm-jp-4-8b-GGUF @ 820b4d43d7f89097d77073437cc53f11579793b1, Lumi-v2-llm-jp-4-8b-Q8_0.gguf 9.13 GB (maker) | yes (base llm-jp-4-8b-base is the listed llm-jp-4 8B's) | 212 dl, 4 likes (2026-08) | A ChatML SFT of llm-jp-4 base on 1.7B tokens (llm-jp-4 thinking data and Nemotron JA). |
| 8 | ABEJA-Qwen2.5-7b-Japanese-v0.1 | ABEJA | abeja/ABEJA-Qwen2.5-7b-Japanese-v0.1 @ b0027a80acb4052482d4b55499100f568e937644 | qwen2 | 7.62B | 32768 | Apache-2.0 | instruct | yes | 15.23 GB | mmnga/ABEJA-Qwen2.5-7b-Japanese-v0.1-gguf @ e9edf6f2d3a4890dac56d6dde158e388f7d2690c, …-Q8_0.gguf 8.10 GB | yes (same as listed Qwen2.5 7B) | 254 dl, 10 likes | Distilled from ABEJA-Qwen2.5-32B-Japanese plus a ChatVector. A well-known startup. |
| 9 | ELYZA-Shortcut-1.0-Qwen-7B | ELYZA | elyza/ELYZA-Shortcut-1.0-Qwen-7B @ d33448648d9b1b7d69cbadb3f1a8d6fb4c8a01ee | qwen2 | 7.62B | 32768 | Apache-2.0 | instruct (answers without the thinking of ELYZA-Thinking) | yes | 15.23 GB | mmnga/ELYZA-Shortcut-1.0-Qwen-7B-gguf @ fc1fb43597cab7c9edbe71e41c109fadb83d3815, …-Q8_0.gguf 8.10 GB | yes | 533 dl, 3 likes | ELYZA's 2025 7B (the listed ELYZA is the 2024 Llama-3 8B). |
| 10 | shisa-v2-qwen2.5-7b | Shisa.AI | shisa-ai/shisa-v2-qwen2.5-7b @ ae8f95eaa26b3422df52ee48d339e22c61b7fcce | qwen2 | 7.62B | 32768 | Apache-2.0 | instruct | yes | 15.23 GB | mmnga/shisa-v2-qwen2.5-7b-gguf @ 5264e5c741637a3a34457aa4c90a6f888cee20b8, …-Q8_0.gguf 8.10 GB | yes | 110 dl, 12 likes | JA MT-Bench (GPT-4-Turbo) 8.16. Superseded by the listed v2.1-qwen3-8b, so low priority. |
| 11 | Qwen3-EZO-8B-beta | AXCXEPT (formerly HODACHI) | AXCXEPT/Qwen3-EZO-8B-beta @ c7458182662d2fa15da5e41a17427319e6f265f5 | qwen3 | 8.19B | 40960 | Apache-2.0 | thinking (Qwen3) | yes | 16.38 GB | mradermacher/Qwen3-EZO-8B-beta-GGUF @ 1867b17511f6bb9659fc7b6f585cae4a62bdcac7, …Q8_0.gguf 8.71 GB | yes | 28 dl, 9 likes | EZO is a known JA brand. A beta; the many YOYO merges of it are third-party. |
| 12 | EZO-Llama-3.2-3B-Instruct-dpoE | AXCXEPT | AXCXEPT/EZO-Llama-3.2-3B-Instruct-dpoE @ 0b374d6e4b55192d833ec5b08ca842aaac4147eb | llama | 3.21B | 131072 | Llama 3.2 | instruct | yes | 6.43 GB | mradermacher/EZO-Llama-3.2-3B-Instruct-dpoE-GGUF @ 65ba0474ccc3c07a4ca4c383e5f3c0f174a81f48, 3.42 GB | yes | 66 dl, 10 likes | 2024. Overlaps the listed shisa-v2.1-llama3.2-3b (newer, better). |
| 13 | llm-jp-3-3.7b-instruct-EZO-Common / -Humanities | AXCXEPT | AXCXEPT/llm-jp-3-3.7b-instruct-EZO-Common @ 8cd47420b77e96061f4b1fa1f006eec7b0ff4609 (Humanities @ ade3da043ec830f0d4d54cfc0e9a91335dfe3f55) | llama | 3.78B | 4096 | Apache-2.0 | instruct | yes | 7.57 GB | ineair/llm-jp-3-3.7b-instruct-EZO-Common-gguf @ 31fe00046f1a04f12980d3e1f0d030c85b9a1cd7, 4.02 GB (third party) | yes | 60 / 54 dl | 2024 fine-tune of the listed llm-jp-3 3.7B. Low priority. |
| 14 | Qwen3-8B-RP-v0.1 / Qwen3-8B-NSFW-JP / Qwen3-8B-ERP-v0.1 | Aratako (individual) | Aratako/Qwen3-8B-RP-v0.1 @ aacfb07d1a7293054ad47ac3de98644ab237b7bd; NSFW-JP @ 610294652e72d3a0a2873e3f481c96dbc1890015 | qwen3 | 8.19B | 40960 | MIT | RP / NSFW instruct | yes | 16.38 GB | Aratako/Qwen3-8B-RP-v0.1-GGUF @ ff4bdc6cb369912f59bf369d3f6d47677dd83abe, Qwen3-8B-RP-v0.1-Q8_0.gguf 8.71 GB (maker); mradermacher/Qwen3-8B-NSFW-JP-GGUF @ 1e1cbb43fd0e07b100f7c19503d1cd3e08eb1805, 8.71 GB | yes | RP GGUF 1.4k dl; NSFW 141 dl, 25 likes | Aratako is the best-known JA RP tuner. **NSFW content: the owner's call.** |
| 15 | sarashina2.2-3b-RP-v0.2 / ArrowNeo-Neko-3B-Japanese | Aratako / DataPilot | Aratako/sarashina2.2-3b-RP-v0.2 @ 72d25fc67f72075f1f7daf6991efc46f65536697; DataPilot/ArrowNeo-Neko-3B-Japanese-v0.1 @ 36875e4aefd463bde66c6e462ae25202b2571e20 | llama | 3.36B | 8192 | MIT | RP / instruct | yes | 6.71 GB | Aratako/sarashina2.2-3b-RP-v0.2-GGUF (Q8_0 not checked) | yes (base is the listed sarashina2.2 3B) | 600 dl (GGUF) / 46 | Niche. |
| 16 | NeoHorse-1-9B-Math-SFT | Takenoko12345678 | Takenoko12345678/NeoHorse-1-9B-Math-SFT @ 2713299ae66312c55ce0af560e970d348a90d684 | qwen3_5_text / Qwen3_5ForCausalLM | 8.95B | 262144 | Apache-2.0 | thinking (math) | yes | 17.91 GB | Takenoko12345678/NeoHorse-1-9B-Math-SFT-GGUF @ 83b9929220104aa4871e97f59916e7d2d1bdea41, 9.53 GB; mradermacher …-GGUF @ a640e9a4a581fcf383ffc067cdbf906caacb5f66 | yes (same as the listed NeoHorse-1 9B) | 386 dl | Math niche. |
| 17 | Qwen3.5-Hasutsubomi-9B (pre-alpha) | Local-Novel-LLM-project | Local-Novel-LLM-project/Qwen3.5-Hasutsubomi-9B @ 3e32f7e3650af6f80d5a36b61a46a805ab6480f7 | qwen3_5 (VLM; text_config qwen3_5_text) | 8.95B | 262144 | Apache-2.0 | instruct (creative writing) | yes | 17.91 GB | mradermacher/Qwen3.5-Hasutsubomi-9B-GGUF @ 65285b7417c5dfcb55a053636f51c91f6ef8b066, 9.53 GB | yes (text part, like the listed Qwen3.5 9B) | 133 dl, 15 likes | The card calls itself pre-alpha and unstable. |
| 18 | Mistral-7B JP merges, 2024: ArrowPro-7B-KUJIRA, Vecteus-v1, Oumuamua-7b-instruct-v2, Umievo-itr012-Gleipnir-7B | DataPilot / Local-Novel-LLM-project / nitky / umiyuki | DataPilot/ArrowPro-7B-KUJIRA @ 755e478ac9404cdd551126e20ea055dfa42ddd1d; Local-Novel-LLM-project/Vecteus-v1 @ d3375ec5f7e149916e44cb0d47c8bda30b986f50; nitky/Oumuamua-7b-instruct-v2 @ 315cee7d10fd84a130ab0973942ff76bb1582c49; umiyuki/Umievo-itr012-Gleipnir-7B @ 5187cc9e441f3d876eda345f52f53eda949c5aee | mistral | 7.24–7.33B | 4096–32768 | Apache-2.0 | instruct | **no** for KUJIRA, Oumuamua and Gleipnir (hand-written template needed); Vecteus yes | 14.5 GB | mmnga/Vecteus-v1-gguf @ 59df491bb68ce3bc6879c3b292942e4c57274d9b, 7.70 GB; mradermacher/Oumuamua-7b-instruct-v2-GGUF @ b5ba02aa55bd31d2365354b492500229535bb592, 7.79 GB | yes (Mistral/Swallow-MS style like the listed Swallow-MS 7B) | KUJIRA 64 likes, Vecteus 36, Oumuamua 24, Gleipnir 22 | The 2024 merge era (Shaberi top of that year). Obsolete next to Qwen3/Swallow 8B. Add at most one (Vecteus has a template). |
| 19 | suzume-llama-3-8B-multilingual / -japanese | Lightblue | lightblue/suzume-llama-3-8B-multilingual @ 0cb15aa9ec685eef494f9a15f65aefcfe3c04c66 (japanese @ 3306e6bac49e1bd80bfe54001994a828ceeb67fa) | llama (Llama 3) | 8.03B | 8192 | Llama 3 | instruct | yes | 16.06 GB | mmnga/lightblue-suzume-llama-3-8B-multilingual-gguf @ d8dce6241fb56355d154df6994beb9ecdaeded78, 8.54 GB | yes | 8.9k dl, 114 likes | 2024; superseded by the listed Llama-3-ELYZA-JP and Swallow 8B. |
| 20 | LFM2-350M-PII-Extract-JP | Liquid AI | LiquidAI/LFM2-350M-PII-Extract-JP @ c9d135815653b380c46eaada7cbb8de120c99d73 | lfm2 | 354M | 128000 | LFM Open License v1.0 | task (JSON PII extraction) | yes | 0.71 GB | LiquidAI/LFM2-350M-PII-Extract-JP-GGUF @ ecf710c7594f12aaacf9b62a431021cbbc48481c, 0.38 GB | yes | 1.7k dl, 62 likes | Not a chat model: it outputs JSON. Skip for a chat page unless the owner wants task demos. |

#### B. Need work (not runnable today): ranked by worth ÷ cost

| # | Model | Maker | Original repo @ sha | model_type | Params | Ctx | License | Kind | Template | safetensors | Best GGUF (Q8_0) | What is missing, and size of the work | Fame | JA note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | Mistral-Nemo-Japanese-Instruct-2408 | CyberAgent | cyberagent/Mistral-Nemo-Japanese-Instruct-2408 @ 8591f78522a5d651209fc9f354c848508db7a3eb | mistral | 12.25B | 1024000 | Apache-2.0 | instruct | yes | 24.50 GB | mradermacher/Mistral-Nemo-Japanese-Instruct-2408-GGUF @ 705b672d49d12be13038decd333109c0adf43ccc, 13.02 GB | **A new pre-tokenizer type (Tekken)**: a Split regex with case classes (`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}…]+|…|\p{N}|…`, one digit per piece) plus ByteLevel and `ignore_merges: true`. That is a `CharClasses`/`STAGES` entry like `spark2_5`, plus the GGUF name `tekken`. head_dim 128 ≠ 5120/32 is already read for Mistral. 40 layers and 8 KV heads: int8 ≈ 13 GB plus KV, so it fits (64-bit). **Small–medium** (tokenizer only). Same work unlocks shisa-v2-mistral-nemo-12b and Aratako's NemoAurora/Mistral-Nemo RP. | 7.7k dl, 49 likes | Was a top-scoring JA 12B in 2024–25. |
| B1' | shisa-v2-mistral-nemo-12b | Shisa.AI | shisa-ai/shisa-v2-mistral-nemo-12b @ 63f3d399b0013b868fa1bcd006bf45490cc1579c | mistral | 12.25B | 131072 | Apache-2.0 | instruct | yes | 24.50 GB | mradermacher/shisa-v2-mistral-nemo-12b-GGUF @ d99e19bd10bc80adba7d0ff698aa598d1dcf4fde, 13.02 GB | same as B1 | 366 dl, 7 likes | Shisa v2 (2025-04). |
| B2 | gemma-2-2b-jpn-it | Google | google/gemma-2-2b-jpn-it @ 6b046bbc091084a1ec89fe03e58871fde10868eb (gated; config read from a mirror) | gemma2 / Gemma2ForCausalLM | 2.61B | 8192 | Gemma | instruct | yes (in the gated repo; mirrors have it) | 5.23 GB | bartowski/gemma-2-2b-jpn-it-GGUF @ bd1518c8610c36695a89c3d7df74ae8bf61e72c2, gemma-2-2b-jpn-it-Q8_0.gguf 2.78 GB | **A new family `gemma2`**: alternating 4096-window and global layers (the engine has windows), pre and post norms (Gemma 3 has them), the final logit softcap 30 (Gemma 4 has it), no q/k norm, query_pre_attn_scalar 256 = head. **Missing: the attention-score softcap (50·tanh(s/50))** in the CPU attention kernel, and on the GPU the flash tiles (or `REASONS` to keep it on the CPU). **Medium** (one kernel argument, one Layout field, a family, tests). Unlocks EZO-gemma-2-2b-jpn-it (AXCXEPT @ d65ae078…), webbigdata/gemma-2-2b-jpn-it-translate (@ fc5e9eab…, GGUF 1.4k dl) and EZO-Common-9B-gemma-2-it (9.24B). | 6.0k dl, 218 likes | Google's only dedicated JP release. Good for 2B in 2024; now below Gemma 3/4 small models. |
| B3 | gemma-3-4b-it / gemma-3-12b-it (text part) | Google | google/gemma-3-4b-it @ 093f9f388b31de276ce2de164bdc2081324b9767; google/gemma-3-12b-it @ 96b6f1eccf38110c56df3a15bffe176da04bfd80 | gemma3 (VLM; text_config gemma3_text) | 4.30B / 12.19B | 131072 | Gemma | instruct | yes | 8.60 / 24.37 GB | unsloth/gemma-3-4b-it-GGUF @ 5a3566e716d80f709ed7b79817eaf7733d2a1fce, 4.13 GB; unsloth/gemma-3-12b-it-GGUF @ d15e4c7dc21dc55d56bf8549db57a71ad8a2a35d, 12.51 GB | (1) A `gemma3` whole-model family like `GEMMA4_WHOLE`/`QWEN35_WHOLE` (flatten text_config, `language_model.` prefix). Today model_type `gemma3` falls to the Llama family and is refused. (2) **Linear RoPE scaling factor 8 on the full-attention layers only**: check whether `gemma3_check()` refuses it and add it to the global table. **Small–medium.** Unlocks DataPilot/ArrowMint-Gemma3-4B-YUKI-v0.1 (@ c04488df…, 8 likes; GGUF mradermacher 4.13 GB), Aratako/gemma-3-4b-it-RP-v0.1, AXCXEPT/EZO2.5-gemma-3-12b-it-Preview (@ 6c9f13be…) and google/translategemma-4b-it (@ 10042cb0…, 893 likes). | 4B 956k dl / 1.6k likes; 12B 344k / 896 | Swallow: Gemma 3 12B 0.405 / MTB 0.659; 4B 0.292 / 0.572. Weaker than the listed Gemma 4 E2B/E4B. |
| B4 | Qwen3-14B | Alibaba (Qwen) | Qwen/Qwen3-14B @ 40c069824f4251a91eefaf281ebe4c544efd3e18 | qwen3 | 14.77B | 40960 | Apache-2.0 | thinking + instruct | yes | 29.54 GB | Qwen/Qwen3-14B-GGUF @ 530227a7d994db8eca5ab5ced2fb692b614357fd, Qwen3-14B-Q8_0.gguf 15.70 GB (Qwen) | Architecture runs today. **Only memory**: int8 ≈ 15.7 GB plus KV (40 layers, 8 KV heads, float16 at 4096: 0.67 GB) ≈ 16.4 GB of 17.18 GB. Verify `footprint()`; it may need a context below 4096. **Small if it fits.** Same for abeja/ABEJA-Qwen3-14B-Agentic-256k-v0.1 (@ c18c202ab2deefede0e50eabb8f3ede832eb9cdb; mmnga-o …-Q8_0.gguf 15.7 GB) and lightblue/DeepSeek-R1-Distill-Qwen-14B-Multilingual (qwen2, 14.77B). | 12.4M dl, 526 likes | **Swallow #2 of ≤14B (0.524, MTB 0.712)**: the strongest JA model near the cap. Global, so possibly another group's. |
| B5 | NVIDIA-Nemotron-Nano-9B-v2-Japanese | NVIDIA | nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese @ 3979dd16634988c34cc3bd911583c51e6a731d10 | nemotron_h / NemotronHForCausalLM (Mamba2 + attention hybrid) | 8.89B | 131072 | NVIDIA Open Model License | thinking + instruct | yes | 17.78 GB | (not searched; would need llama.cpp nemotron_h) | **A new state layer kind (Mamba2 SSM, `nemotron_h` layer pattern)**: Python layer + `STATEFUL_LAYERS`, `LAYER_KINDS` entry, new kernels; GPU optional. **Large.** | 3.3k dl, 146 likes | Nejumi 4: #1 sub-10B (third-party). Swallow 0.503 / MTB 0.689. |

#### C. Skip (duplicates, obsolete, out of scope)

- **Shisa**: shisa-v2.1-qwen3-8b, -llama3.2-3b, -lfm2-1.2b and v2.1c-lfm2-350m are listed. shisa-v2.1-llama3.3-70b, v2 mistral-small-24b, qwen2.5-32b, llama3.3-70b and 405b are too big. shisa-v2-llama3.1-8b (b6dd0ddc…) and shisa-v2-unphi4-14b are superseded by v2.1. shisa-v1-7b-v2.1 is an old base re-tuned (29 dl). shisa-v2.1c-lfm2-350m-sft3 is an intermediate checkpoint. shisa-de-1/2/2.1 are a Gemma 4 26B-A4B MoE (classifier-like, 25.8B): MoE.
- **Lightblue**: karasu-1.1B (.bin only, 2048 ctx, 2024), Karasu-DPO-7B and kurage-* (RAG-specific Qwen2 7B), qwarasu/ao-karasu/Karasu-Mixtral (big or MoE), Jamba (hybrid).
- **AXCXEPT**: phi-4-deepseek-R1K-RL-EZO and Borea-Phi-3.5 (phi3: unsupported), glm-4-9b (glm), EZO-Qwen2.5-32B/72B (too big), Llama-3.1-8B-EZO-1.1-it (2024).
- **Aratako**: calm3-22b-RP, Mistral-Small 24B, Amaterasu-123B (too big), Qwen3-30B-A3B NSFW/ERP (MoE), Ministral-3 TextOnly (mistral3 new arch; possibly mistral-loadable, unchecked), ELYZA MoE merges.
- **DataPilot**: sarashina2.2 MoE, ArrowNeo-AME MoE, Arrival/SKYCAVE 32B, ArrowIdeative-13b (llm-jp-3-13b, no template, 13.7B experimental).
- **Takenoko**: Qwen3.5-0.8B-Japanese-SFT v1/v2 (superseded by the listed v2.5), Spark-X2.5-4B-Japanese-Pitcher (themed variant).
- **TokenRhythm**: NeoHorse-1 4B/9B are listed. NeoHorse-Jev-4B is a classifier.
- **kotoba-tech**: kotomamba-2.8B (mamba, 2024). **SpiralAI / HODACHI**: no current models on HF (HODACHI is now AXCXEPT).
- **kirp/jpt-0.8b/4b/9b** (qwen3_5, CC BY-NC): a "decision" model that returns option probabilities in one forward pass, not a chat model.
- **LiquidAI/LFM2.5-1.2B-JP** (58ef35ca…, January 2026): superseded by the listed LFM2.5-1.2B-JP-202606.
- **EQUES/JPharmatron-7B** (gated, pharma domain, Qwen2.5-7B, float32 30 GB): niche.
- **ELYZA**: Thinking-1.0-Qwen-32B, Thinking/llm-jp-4-33b and 32b-a3b (too big or MoE), Diffusion (diffusion), Llama-2 era (obsolete).
- **karakuri-ai**: 8x7b (MoE), 70b and 32b (too big), 7b-apm (2024, attribute-prompt niche).
- **stockmark**: 13b (2023, 13B llama, no template), 100B. **ABEJA**: QwQ-32B and Qwen2.5-32B (too big), Mixtral (MoE), gpt-neox-japanese-2.7b (gpt_neox_japanese: unsupported).
- nagohachi/Llama-3.2-tiny-lm-japanese-500m-sft-v2 uses a custom `scratch_llama` model_type: hobby.

#### D. HF ja top lists: text models not listed, ≤ ~14B (mostly global multilingual; may belong to another group)

The `language=ja` lists are dominated by global multilingual models; no Japanese-only community model reaches the top 100 by downloads or likes. Among non-listed ≤14B entries:

| Model | sha | model_type | Runs? | Note |
|---|---|---|---|---|
| Qwen/Qwen3-14B | 40c06982… | qwen3 | memory-borderline (B4) | Swallow #2. |
| google/gemma-3-4b-it, gemma-3-12b-it | 093f9f38… / 96b6f1ec… | gemma3 VLM | needs B3 | |
| LiquidAI/LFM2.5-2.6B | b212e75975bbda9676a94d3cb7b2ca25b65fed48 | lfm2 | **yes** | 93k dl, 859 likes; GGUF LiquidAI/LFM2.5-2.6B-GGUF @ e7caca5d…, Q8_0 2.87 GB. LFM's JA is strong (LFM2.5 JP line). Strong candidate if no other group takes it. |
| XHToken/Spark-X2.6-4B, Spark-X2.6-1.7B | d1b5f260… / f18c032c… | spark2_5 | **likely yes** (same model_type as the listed X2.5) | Newer version of the listed Spark-X2.5 (2026-10-27). 4B: 912 likes, GGUF XHToken/Spark-X2.6-4B-GGUF @ 0e8aa39a…, Q8_0 4.38 GB. 1.7B: GGUF @ cdc3d889…, 1.82 GB. Check the tokenizer is unchanged. |
| LiquidAI/LFM3-1.7B | ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7 | lfm3 (new) | no | Conv layers plus 256-window and full layers, partial rotary 0.5 on the full ones, sandwich norm. Medium work on top of LFM2 and Gemma 3. 504 likes, GGUF @ bad5729b…, Q8_0 1.85 GB. |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 | 49843eed… | gemma4 | likely yes (a Gemma 4 E4B derivative) | OpenMDW license. |
| ornith-ai/Ornith-1.5-9B | 489cb979… | qwen3_5 | likely yes | 463k dl, MIT; GGUF Q8_0 9.79 GB. |
| Nanbeige/Nanbeige4.2-3B | b82e54bd… | nanbeige (new) | no | Unknown arch. |
| MemTensor/MemT-1.0-Nano, KRAFTON/Raon-1-mini-preview | — | MoE / custom | no | |

#### Duplicates and versions worth flagging against the listed set

- The listed Spark-X2.5 1.7B/4B have a **newer Spark-X2.6** (2026-10-27).
- The listed LFM2.5-1.2B-JP-202606 is the newest JP LFM. LFM2.5-2.6B is the next size up.
- The listed shisa-v2.1 set is missing only **unphi4-14b** (the best one) and the 70B.
- The listed Takenoko Qwen3.5-0.8B v2.5 is the newest. The same maker's Spark-X2.5-4B-Japanese is new.

