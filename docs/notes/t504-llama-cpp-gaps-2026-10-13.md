# T504: llama.cpp ができて、ここがまだできないこと（2026-10-13）

調べだけの子タスク。コードは変えていない。表は持ち主に見せて、次に取り込む物を選んでもらうためのもの。

- llama.cpp は `ggml-org/llama.cpp` のコミット `4413a192410fad1f054e9216114087a3e01527a0`（T361〜T364 と同じ）。要るファイルだけを取って読んだ（`ggml/include/ggml.h`・`ggml/src/ggml.c`・`include/llama.h`・`src/llama-quant.cpp`・`src/llama-vocab.{h,cpp}`・`src/llama-arch.{h,cpp}`・`src/llama-sampler.cpp`・`common/common.h`・`common/sampling.cpp`・`src/models/` の 35 個）。行は写していない（名前と形だけ）。
- ここの木は本線 `abaa7e7b`（T364.3 が入った所）。
- Hugging Face の数は 2026-10-13 の API（`/api/models?filter=gguf&sort=downloads`、上位 500 リポジトリを 1 回）と、GGUF の頭の Range 読み 13 ファイル。

## まとめ

1. **いちばん効くのは「GGUF をそのまま開く」こと**。ダウンロード上位 500 の GGUF のリポジトリのうち 202（ダウンロードの 39.0%）は、ここがもう動かせる形（`llama`・`qwen2`・`qwen3`・`qwen35`・`granite`・`smollm3`・`lfm2`・`gpt2`・`gptneox`）なのに、ページから開けるのは一覧に手で書いた項目だけ。足りないのは形ではなく、入口と型と、GGUF だけで開くときの 3 つの断り。
2. **Q4_K_M を開くのに足りない型は Q6_K（必ず）と Q5_K（Qwen3.5 の系統）の 2 つ**。頭を読んだ 8 つの Q4_K_M のどれも、重みの 15.7〜36.5% が Q6_K だった。Q4_K はもう読める。
3. **どれも Python で書ける**（読み手・語彙・書式・入口の段取り）。層を回す所（TypeScript の forward）に手が入るのは、塊のまま掛けるカーネルを足すときと、新しい形の層を足すときだけ。Python に寄せてモデルが狭まる項目は、この表には「層を回す所を Python に戻す」案のほかに無い。
4. 形で足りないもののうち、使われているのは Gemma 4（上位 500 に 45）と MoE の系統（`qwen35moe` 32・`qwen4exp` 15）。Gemma は台帳に段取りがある（T257・T261・T328〜T334）。MoE は台帳に無く、大きい。
5. 標本抽出は 6 つの設定が在り、llama.cpp の 20 個のうちモデルのカードがよく言うものは足りている。**違いは温度を掛ける位置**（ここは先、llama.cpp の既定は最後）。

## 持ち主に選んでもらう案（次に取り込む物）

1. **（勧め）GGUF を URL から開く: 残りの K 量子化の読み手と、`?hf=` で `.gguf` を名指す入口**。Q4_K_M・Q5_K_M・Q6_K の GGUF が URL とシートから開く。小さいモデルは今の道で int8 に直して速いカーネルで動かし、int8 で入らない大きいものだけ塊のまま（カーネルは次の段）。Python と、ページの入口の小さな直し。文面（シートと断りの文）が要る。
2. **GGUF だけで開くときの断りを外す**（BOS を言わない GGUF・`rope_freqs` の表・sentencepiece の語彙・前分割の名前）。1 と一緒でないと効き目が半分: いま Llama 3.1 と Qwen3.5 の実物の GGUF は、Q8_0 でも GGUF だけでは断られる（下の 2 節、測った）。Python だけ。
3. **形を 1 つ足す: Gemma 3 → Gemma 4**（T257・T261・T328〜）。足りない形のうちいちばん使われている。層の部品が増えるので TypeScript の forward にも手が入る。規模 大。
4. **会話の履歴と頭の使い回し**。いまは 1 ターンだけ。訪問者に見える違いはいちばん大きいが、llama.cpp との差というよりページの仕様（文面と画面が要る）。
5. 標本抽出を llama.cpp の既定の順にそろえる（温度の位置、frequency の罰）。小。答えの文が変わる（今の種が別の文を書く）。

## 1. 量子化の型

llama.cpp: `ggml.h` の `enum ggml_type` に生きている番号が 35、そのうち重みの型は 29（整数 4 つ・F64・掛け算の途中だけの `Q8_K` を除く）。塊の大きさは `ggml.c` の `type_traits`（`QK_K` = 256 個の型と、32 個の型）。ファイルの「混ぜ方」は `llama.h` の `enum llama_ftype` に 36。

ここ: `convert/readers.py` の `SOURCES` が読む型は 16（ggml の 14 と、Prism ML の fork の 2 つ）。うち `engine/dtypes.py` の `BLOCKS`・`jobs.ts` の `PACKED`・`kernels/blocks/` が塊のまま掛けるのは 10。**ggml の重みの型 29 のうち読めないのは 15**。

| 型（GGUF の番号） | llama.cpp | ここ | 取り込むなら | 狭まるか | 手間 | 効き目 | 勧め |
|---|---|---|---|---|---|---|---|
| F32・F16・BF16（0・1・30） | 読む | 読む（`SOURCES`） | — | — | — | — | 済み |
| Q8_0（8） | 読む・掛ける | 読む。int8 に損なく戻る | — | — | — | 上位 500 のうち 371 が持つ | 済み |
| Q4_K・Q2_K（12・10） | 同上 | 読む・塊のまま掛ける（`SUPERBLOCKS`・`BLOCKS`、T362・T363） | — | — | — | — | 済み |
| IQ1_S・IQ1_M・IQ2_XXS・IQ2_XS・IQ2_S・IQ3_XXS・IQ3_S・IQ4_XS | 同上 | 同上（8 つとも） | — | — | — | — | 済み |
| **Q6_K（14）** | 256 個の塊 | 読めない（`gguf_model()` が「not the K-quants」で断る） | 読み手は Python だけ（`superblocks.py` に 1 つ、表の 1 行、`BLOCKS` の 1 行、試験 4 か所: T362 の形）。塊のまま掛けるなら + カーネル（`blocks/types.ts` のクラスと型で分ける 4 か所、`jobs.ts` の 2 行、`Stored`: T363.4・.5 の形） | 狭まらない | 読み手 小 / カーネル 中 | **Q4_K_M・Q5_K_M・Q6_K・IQ4_XS のどの混ぜ方にも入る**（下の表）。上位 500 のうち Q4_K_M 354・Q6_K 327 | **やる（読み手を先に）** |
| **Q5_K（13）** | 256 個の塊 | 読めない | 同上 | 狭まらない | 同上 | Q5_K_M の本体。Q4_K_M でも q・k・v が 1 つの行列の形（Qwen3.5、GPT-2 の系統）で 18% | **やる** |
| Q3_K（11） | 256 個の塊 | 読めない | 同上 | 狭まらない | 同上 | Q3_K_S・M・L（上位 500 に 158・219・114）と Q2_K の混ぜ方 | やる（上の 2 つと同じ回） |
| Q4_0・Q5_0（2・6） | 32 個の塊 | 読めない | Python だけ（`readers.py` に読み手と `SOURCES` の 1 行: Q8_0 の形。int8 に直す） | 狭まらない | 小 | 古い GGUF（上位 500 に 147・52）と、**幅が 256 で割れない行列の代わりの型**（`llama-quant.cpp` の `tensor_type_fallback()`: Q4_K → Q5_0、Q2_K・Q3_K → Q4_0） | やる（代わりの型なので、K 量子化を開くなら要る） |
| Q4_1・Q5_1（3・7） | 32 個の塊 | 読めない | 同上 | 狭まらない | 小 | Q5_K の代わりの型が Q5_1。単独では上位 500 に 84・27 | 後で |
| IQ4_NL（20） | 32 個の塊 | 読めない | 同上（表の 16 個の値を持つ: 表を写すなら許諾文） | 狭まらない | 小 | i 量子化の代わりの型（幅が 256 で割れない行列）。単独で 74 | 後で |
| TQ1_0・TQ2_0（34・35） | 256 個の塊の 3 値 | 読めない（読むのは fork の PQ2_0・PTQ1_0: 別の並び） | Python（3 値の dtype に損なく入るかは未確認）。カーネルは今の 3 値のもの | 狭まらない | 小〜中 | BitNet（T283）。上位 500 に 6・2 | 後で（T283 と一緒に） |
| Q1_0・Q2_0（41・42） | 新しい型 | 読めない | 未確認（fork の PQ2_0 と同じ並びかは見ていない） | — | 未確認 | 1-bit Bonsai（T286）の見込み。未確認 | 後で（T286 の調べ） |
| MXFP4・NVFP4（39・40） | MoE の FFN の型 | 読めない | — | — | — | `gpt-oss`（MoE: 形が無い） | やらない |
| Q8_1・Q8_K（9・15） | ファイルの混ぜ方に出ない（`Q8_K` は掛け算のときの活性値の型） | — | — | — | — | — | 要らない |

**混ぜ方が使う型**（`llama-quant.cpp` の `llama_tensor_get_type_impl()` を読み、実物の頭で確かめた。MoE と Falcon と 70B の枝は除く）:

| 混ぜ方 | 使う型 | 実物（重みの割合） | ここで足りないもの |
|---|---|---|---|
| Q8_0 | Q8_0 | — | 無し |
| Q4_K_M | Q4_K が本体。分類器（と埋め込みが同じ表のとき埋め込み）が Q6_K、v と FFN の出の一部の層が Q6_K、q・k・v が 1 つの行列なら Q5_K | Qwen3 4B: Q4_K 78.0%・Q6_K 22.0%。Llama 3.1 8B: 80.9・19.1。Qwen2.5 32B: 83.3・16.7。LFM2.5 230M: 63.5・36.5。Mistral 7B v0.2: 84.3・15.7。TinyLlama: 83.1・16.9。HyperCLOVAX 1.5B: 76.4・23.6。**Qwen3.5 4B: Q4_K 57.5・Q6_K 24.4・Q5_K 18.0・Q8_0 0.1** | **Q6_K**、Qwen3.5 は **Q5_K** も |
| Q4_K_S | Q4_K、最初の層の v と FFN の出が Q5_K、分類器が Q6_K | 見ていない | Q5_K・Q6_K |
| Q5_K_M・Q5_K_S | Q5_K、分類器と一部が Q6_K | Qwen3 4B の Q5_K_M: Q5_K 78.0%・Q6_K 22.0% | Q5_K・Q6_K |
| Q6_K | Q6_K（代わりは Q8_0） | 見ていない | Q6_K |
| Q3_K_S・M・L、Q2_K | Q3_K・Q2_K に Q4_K・Q5_K、分類器が Q6_K | 見ていない | Q3_K・Q5_K・Q6_K |
| IQ4_XS | IQ4_XS、一部が Q5_K、分類器が Q6_K | Qwen3.5 4B: IQ4_XS 78.3%・Q6_K 15.1%・Q5_K 6.5% | Q5_K・Q6_K |
| IQ1〜IQ3 の系統 | 本体は読める型。分類器が Q5_K（IQ3_XS・S・M は Q6_K）、埋め込みに Q2_K・IQ3_S・Q4_K | Saluki（T364）は読める 10 種類だけで出来ている | 作り手しだいで Q5_K・Q6_K |

**答え**: 「Q4_K_M の GGUF を開く」には、読み手が Q6_K と Q5_K の 2 つ要る。幅が 256 で割れない行列を持つモデル（型を見た 10 ファイルには無かった）のために、代わりの型 Q5_0（と分類器の Q8_0: 済み）もあると断りが減る。**読み手だけ足せば、今の「広げて int8 に直す」変換で開く**（`conversion.blocks` の無い読み込みは塊を広げる: `weights.ts` の `runsPacked()`）。そのときメモリは int8 の大きさ（4B で約 4.2 GB、取得は 2.5 GB）。塊のまま持つ（メモリが取得とほぼ同じ）には Q5_K・Q6_K のカーネルが要る。

未計測: Q4_K・Q6_K を float32 に広げて int8 に丸め直したときの質（Q8_0 から作った int8 との perplexity の差）、NumPy の読み手で 4B 級を広げる秒（3 値の読み手は NumPy で 216 秒がカーネルで 57 秒だった: T273）。

## 2. GGUF を開く入口

| 項目 | llama.cpp | ここ | 取り込むなら | 狭まるか | 手間 | 効き目 | 勧め |
|---|---|---|---|---|---|---|---|
| GGUF 1 つで開く | 語彙・書式・形を全部メタデータから（`llama-vocab.cpp` の `load()`、`llama-arch.cpp` の鍵） | 段取りに道が在る（`conduct.py` の `SOURCES` の「a GGUF alone」→ `Conversion.from_gguf()`・`gguf_tokenizer()`）。一覧の段 ① の項目が使う | — | — | — | — | 済み（一覧の項目だけ） |
| **`?hf=` とシートから `.gguf`** | — | 開けない: `src/page/address.ts` の `hfEntry()` が重みを `model.safetensors` と決め打ち | ページの入口（`address.ts` の 1 か所と、ファイルの名前を受ける URL の鍵かシートの欄）。段取り・worker は触らない（`is_gguf(hf)` が名前で分ける） | 狭まらない | 小（文面が要る: シートと note） | 上位 500 のうち形が合う 202 リポジトリが URL から開く候補になる（下の断りと型を外した分だけ） | **やる** |
| フォルダから `.gguf` | — | 開けない: `src/page/folder.ts` の門が `.safetensors` 1 つだけ（T410、持ち主の判断待ち） | ページの門だけ（段取りとフォルダの答え手はもう読める: T410 の記録） | 狭まらない | 小（門の文が変わる） | 手元の GGUF が開く | やる（上と同じ回に） |
| BOS を言わない GGUF | 無ければ既定で進む | **断る**（`from_gguf()`:「This GGUF names no BOS token…」）。測った: `unsloth/Qwen3.5-4B-GGUF` の Q8_0 はこれで断られる（`tokenizer.ggml.bos_token_id` が無い。EOS は 248046） | Python だけ: 書式を読めたら頭のトークンを BOS にする決まり（T264・T369）を GGUF の道にも。読めなければ今までどおり断る | 狭まらない | 小（T203 のレビューの穴を戻さないこと） | Qwen3.5 の系統（上位 500 に 91 リポジトリ）が GGUF だけで開く | **やる** |
| `rope_freqs` の表（Llama 3 の RoPE の縮め） | テンソルを読む | GGUF だけでは**断る**（`gguf_model()`:「…a rope_freqs table, which the engine does not read」）。原本の config があれば突き合わせて通す（`rope_freqs_agree()`）。測った: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` の Q8_0 は断られる | Python だけ: 表から縮め方（llama3 の 4 つの数）を戻すか、表そのものを RoPE の表に使う | 狭まらない | 小〜中（表から式に戻せない表をどうするか） | Llama 3.1・3.2 とその派生が GGUF だけで開く | やる |
| 形の突き合わせ | 無い（GGUF を信じる） | 原本つきのときだけ `gguf_agrees()`。GGUF だけの道は家族の `gguf_config()` を信じる | — | — | — | — | 今のまま（URL から開く物は信じる。一覧に入れる物は `gguf_check.py`） |
| 書式 `tokenizer.chat_template` | Jinja を自前で（`common/chat*`: 読んでいない） | 読む（`gguf_tokenizer()` が `chat_template` を渡し、`convert/template.py` が jinja2 で 1 ターンを描く） | — | — | — | — | 済み |
| 語彙が sentencepiece（`tokenizer.ggml.model` = `llama`） | SPM | **断る**（`gguf_tokenizer()`: gpt2 だけ）。測った: Mistral 7B v0.2 と TinyLlama の GGUF は `llama` | 3 節 | | | | |

測った 3 つ（Q8_0 の頭を `Conversion.from_gguf()` に渡した）: Qwen3 4B は通る、Llama 3.1 8B は `rope_freqs` で断る、Qwen3.5 4B は BOS で断る。**型を足しても、この 2 つの断りを外さないと、いちばん使われている 2 系統が URL から開かない。**

## 3. トークナイザ

llama.cpp: 語彙の種類は `llama.h` の `enum llama_vocab_type` に 9（`NONE` と `TEST` を除いて 7）。前分割は `llama-vocab.h` の `enum llama_vocab_pre_type` に 62、GGUF の名前（`tokenizer.ggml.pre`）は `llama-vocab.cpp` に 93、正規表現の枝は 37（中身が別のものは 32）。

ここ: 種類は 3（`engine/tokenizer.py` の `kind`: `bpe`（llama2.c の形）・`unigram`・`bytebpe`）、前分割の型は 6（`PATTERNS` と `STAGED`: `gpt2`・`gpt2-digits`・`qwen`・`llama3`・`qwen35`・`minicpm5`）。GGUF の名前から引く表 `gguf.py` の `GGUF_PRETOKENIZERS` は 9 行（llama.cpp の名前は 8）。

| 項目 | llama.cpp | ここ | 取り込むなら | 狭まるか | 手間 | 効き目 | 勧め |
|---|---|---|---|---|---|---|---|
| BPE（`gpt2`） | `LLAMA_VOCAB_TYPE_BPE` | 読む（`gguf_tokenizer()`） | — | — | — | — | 済み |
| **SPM（`llama`）を GGUF から** | `LLAMA_VOCAB_TYPE_SPM`（語片と点数） | GGUF からは読めない。エンジンの `bpe` は同じ「点数の高い対をつなぐ」形で、sentencepiece の model からは読める | Python だけ: 語片・点数・種類を `tokenizer_bin()` に（正規化の charsmap は GGUF に無い: 空白の前置きなどの鍵を読む） | 狭まらない | 小〜中（本物の sentencepiece と ID を突き合わせる） | Llama 2・Mistral v0.x・TinyLlama の GGUF（後の 2 つは実物で見た） | やる（2 の入口と同じ回か次） |
| UGM（`t5`） | `LLAMA_VOCAB_TYPE_UGM`、`precompiled_charsmap` も GGUF に在る | GGUF からは読めない（エンジンの `unigram` と `Charsmap` は在る） | Python だけ | 狭まらない | 小 | T5 の系統だけ（形が無い） | 後で |
| WPM（`bert`）・RWKV・PLaMo-2/3 | 在る | 無い | Python（新しい種類） | 狭まらない | 中 | 埋め込みのモデル・RWKV・PLaMo（どれも形が無い） | やらない（形が先） |
| `gemma4`（生の UTF-8 の BPE） | BPE の枝 `byte_encode = false` | 無い（T257: tokenizer.json の sentencepiece 流の BPE） | Python だけ | 狭まらない | 中 | Gemma 4 の前提 | 後で（案 3 と一緒） |
| 前分割: ここの型と同じ枝の名前 | 52 個の名前が、ここの 6 つの型と同じ枝に落ちる（下） | 表に在るのは 8 個 | Python だけ: 表の 1 行ずつ。名前ごとに `ignore_merges`・NFC を確かめる | 狭まらない | 小（行は 1 つ、確かめは本物の tokenizers と） | 断りが減る。測った例: HyperCLOVAX 1.5B は名前が `dbrx`（Llama 3 と同じ枝）で、いまは断られる | やる（使われている名前から） |
| 前分割: 別の枝（26 枝） | Tekken（Mistral）・GPT-4o と Llama 4（Phi-4）・DeepSeek の 3 つ・Spark・既定（`default`）など | 無い。台帳: Tekken と Phi-4 は T256、Spark は T340 | Python だけ（`CharClasses` を細かくする: T256） | 狭まらない | 中 | Mistral の新しいモデル・Phi-4・DeepSeek V3 | 後で（形と一緒に） |

**同じ枝に落ちる名前**（llama.cpp の中で正規表現の並びが同じ、と確かめた。ここの型が llama.cpp の枝と同じ切り方かは、表にもう在る 8 個についてだけ確かめてある: T253・T254・T260 の記録。ほかは未確認）:

- `llama3` の枝（14 個）: `llama3`・`llama-v3`・**`llama-bpe`**・`falcon3`・`falcon-h1`・`pixtral`・`midm-2.0`・**`lfm2`**・`jina-v5-nano`・`dbrx`・`smaug-bpe`・`chatglm-bpe`・`glm4`・`glm5`
- `gpt2` の枝（19 個）: **`gpt-2`**・`phi-2`・`jina-es`・`jina-de`・`gigachat`・`jina-v2-es`・`jina-v2-de`・`a.x-4.0`・`mellum`・`modern-bert`・`jina-v1-en`・`jina-v2-code`・`roberta-bpe`・`exaone4`・`mpt`・`olmo`・`jais`・`trillion`・**`granite-docling`**
- `gpt2-digits` の枝（8 個）: `starcoder`・`refact`・`command-r`・**`smollm`**・`codeshell`・`exaone`・`minerva-7b`・`mellum2`
- `qwen` の枝（9 個）: **`qwen2`**・`deepseek-r1-qwen`・`kormo`・`f2llmv2`・`megrez`・`stablelm2`・`hunyuan`・`solar-open`・`grok-2`
- `qwen35`・`minicpm5`: 1 個ずつ（どちらも在る）

太字が表に在るもの。

## 4. 標本抽出

llama.cpp: `include/llama.h` に `llama_sampler_init_*` が 20、`common/common.h` の `enum common_sampler_type` に 11。既定の並び（`common_params_sampling` の `samplers`）は「罰 → DRY → top-n-sigma → top-k → typical → top-p → min-p → XTC → 温度」で、既定の値は top-k 40・top-p 0.95・min-p 0.05・温度 0.8・罰は無し（窓 64）。

ここ: `engine/sampler.py` の `Sampling` に 6 つの欄（`temperature`・`topp`・`repetition_penalty`・`top_k`・`min_p`・`presence_penalty`）。順は「繰り返しの罰 → presence の罰 → **温度** → top-k → top-p → min-p」。1 つ足す手間は、欄 1 つ・NumPy の段・カーネル（`sample.ts`）と ctypes の署名・JavaScript の模型の 3 つを同じ形に、で、GPU の歩からは自動で外れる（AGENTS.md の「標本抽出の規則を足す」）。

| 項目 | llama.cpp | ここ | 取り込むなら | 狭まるか | 手間 | 効き目 | 勧め |
|---|---|---|---|---|---|---|---|
| top-k・top-p・min-p・温度・greedy | 在る | 在る | — | — | — | Qwen・Qwen3.5・Granite・Bonsai・Saluki のカードの設定は全部これで書けている（`src/models/sampling.js`） | 済み |
| 繰り返しの罰・presence の罰 | `llama_sampler_init_penalties` | 在る（presence は書いた語だけ: T274） | — | — | — | — | 済み |
| **温度の位置** | 既定は最後（絞ってから温度） | 先（温度を掛けた確率で絞る: llama2.c の形） | Python + カーネル（`sample` の中の順）+ 模型 | 狭まらない | 小〜中 | 同じ設定でも絞られる語が違う（温度 1 なら同じ）。今の種は別の文を書く | 持ち主の判断（どちらが正しいとは言えない。transformers は温度が先のはずだが、この回では読んでいない） |
| frequency の罰 | `penalties` の中 | 無い | Python + カーネル + 模型（`penalize` に引数 1 つ） | 狭まらない | 小 | OpenAI の形の設定を言うカード | 後で |
| typical・top-n-sigma・XTC・動的な温度 | 在る | 無い | 同上（どれも語を絞る段か温度の式） | 狭まらない | 小ずつ | 読んだカードでは勧められていない（Gemma・Llama・Mistral のカードは見ていない: 未確認） | 後で |
| DRY | 在る（直近の列の繰り返しを探す） | 無い | Python + カーネル（履歴を見る段） | 狭まらない | 中 | 小さいモデルの繰り返しに効く見込み（未計測） | 後で |
| mirostat v1・v2、adaptive-p | 在る（状態を持つ） | 無い | Python + カーネル（生成の間の状態） | 狭まらない | 中 | 使われ方は未確認 | やらない |
| logit bias | 在る | 無い（止まりの語は別の仕組み） | Python だけで書ける（logits に足す 1 行。カーネルの前） | 狭まらない | 小 | 「`<unk>` を引かせない」（rinna の件、持ち主の決め待ち）に使える | 後で |
| 文法（GBNF）・lazy な文法 | 在る（`llama_sampler_init_grammar*`） | 無い | Python（文法の読み手と、歩ごとに語彙を絞る段: 語彙 × 歩の仕事なので遅い） | 狭まらない | 大 | JSON を書かせる・道具の呼び出し。ページに使う所が無い | やらない |
| infill・考える長さの上限（reasoning budget） | 在る | 無い | Python | 狭まらない | 中 | 考える形が文脈を使い切る件に効く見込み | 後で |

## 5. モデルの形

llama.cpp: `llama-arch.h` の `enum llm_arch` に 157（`UNKNOWN` を除く）。ここ: `engine/layout.py` の `LAYOUTS` に 5（`llama`・`gpt2`・`neox`・`qwen35`・`lfm2`）、家族は `convert/families/` の 11、GGUF の形の名前は 9。

上位 500 の GGUF のリポジトリに出る形の名前は 80（絵・音・埋め込みも入る）。文を書く形のうち主なもの（かっこは上位 500 のリポジトリの数。「違い」は `src/models/<形>.cpp` を名前で拾い読みしたもので、式は確かめていない）:

| 形 | ここ | 今の 5 つとの違い | 取り込むなら | 手間 | 勧め |
|---|---|---|---|---|---|
| `qwen35`（91）・`llama`（42）・`qwen3`（36）・`qwen2`（21）・`lfm2`（6）・`granite`（5）・`smollm3`・`gpt2`・`gptneox` | 動く | — | — | — | 済み（合わせて 202、ダウンロードの 39.0%） |
| `gemma4`（45）・`gemma3`（9）・`gemma2`（2）・`gemma3n` | 台帳: T257・T261・T328〜T334 | 窓つきの attention、norm が層に 4 つ、GELU の FFN、埋め込みの倍率、logits の頭打ち（tanh）。Gemma 4 は層ごとの埋め込みと KV の共有も | Python + TypeScript の forward（窓・層の部品）+ GPU は別 | 大 | 案 3 |
| `qwen35moe`（32）・`qwen3moe`（6）・`qwen4exp`（15）・`gpt-oss`（5）・`deepseek2`（4）・`deepseek4`（6） | 無い（台帳にも無い） | FFN が専門家の混合（MoE: 語ごとに選んだ数個の FFN だけを掛ける）。DeepSeek は attention も別の形（MLA） | Python + TypeScript の forward + カーネルの仕事の組み方 | 大（ほとんどが 16 GiB を越える） | やらない（入る大きさのものが出たら） |
| `qwen3vl`（10）・`qwen2vl`（2） | 無い | 文の側は Qwen3・Qwen2 に近い（位置の回し方が多次元）。絵は別のファイル | 未確認 | 未確認 | 後で（文だけで開くかの調べ） |
| `phi3`（5） | 台帳: T258（前分割は T256） | q・k・v が 1 つの行列、窓、Phi-4 は longrope | Python が主（行列を割るのは変換器）。窓は forward | 中 | 後で |
| `mistral3`（3）・`mistral4` | Mistral は Llama として動く。新しいトークナイザは T256 | Llama に近い（attention の温度の倍率が任意で付く）。前分割が Tekken | Python が主 | 中（トークナイザ） | 後で |
| `spark2_5`（4） | 台帳: T339〜T344 | 窓、層の種類ごとの RoPE、head ごとの gate、GELU | Python + forward | 中 | 後で |
| `bitnet` | 台帳: T283 | 層の中にもう 1 つずつ norm（sub-norm）、型は TQ | Python + forward の層の部品 | 中 | 後で |
| `falcon`・`stablelm`・`starcoder2`・`phi2` | 無い | LayerNorm の系統。q・k・v が 1 つ、norm の bias、FFN の形が少しずつ違う | Python が主（今の事実の表の組み合わせで足りるかは未確認） | 小〜中ずつ | やらない（上位 500 に無い） |
| `rwkv6`・`rwkv7`・`mamba`・`mamba2`・`falcon-h1`・`granite-hybrid`・`jamba` | 無い | attention でない層（状態を持つ）。Mamba は状態空間の層 | Python + forward に新しい種類の層（T375.3 の手順）+ カーネル | 大ずつ | やらない |
| `t5`・`t5encoder`（2） | 無い | エンコーダとデコーダ、位置の偏り、cross attention | 生成の流れから別 | 大 | やらない |
| ほか約 100（BERT の系統 9、絵と音、MoE の各社、`laguna`・`muse-glimmer` など） | 無い | 読んでいない | — | — | 付録 |

## 6. 文脈と状態

| 項目 | llama.cpp | ここ | 取り込むなら | 狭まるか | 手間 | 効き目 | 勧め |
|---|---|---|---|---|---|---|---|
| 会話の履歴（複数のターン） | 書式に messages の列を渡す（`llama_chat_apply_template`、common の Jinja） | 無い: ページは user 1 つの 1 ターンを送り、変換器は 1 ターンを `{prompt}` の形に戻す（`template.py` の `one_turn()`） | Python（書式を messages の列で描く）+ ページ（吹き出しの持ち方と文面）。台帳に番号は見つからなかった | 狭まらない | 中〜大（画面と文面） | 訪問者にいちばん見える違い | 案 4（仕様から） |
| 頭の使い回し（前のターンの KV を残す） | `cache_prompt`・`n_cache_reuse`（common）、`llama_memory_seq_*` | 無い: `generate()` は毎回位置 0 から。KV は位置ごとに書き直せる（状態を持つ層は位置 0 からだけ） | Python（どこまで同じかを比べて続きから）。forward は今のままの見込み（未確認: GPU の `gpuEnd` と状態を持つ層） | 狭まらない | 中 | 履歴があるとき、2 ターン目からの待ちが短い | 履歴と一緒に |
| 文脈を越えたときのずらし | `ctx_shift`・`n_keep`、`llama_memory_seq_add`・`seq_rm` | 無い（文脈いっぱいで止まる） | Python + forward（KV を動かす。RoPE は位置を持つので、ずらし方の検証が要る） | 狭まらない | 中 | 長い答え。ページでは起きにくい（文脈 4096） | やらない |
| KV の保存と読み込み | `llama_state_save_file`・`llama_state_seq_*` | 無い（モデルの変換は OPFS に保存する。KV は保存しない） | TypeScript（メモリの持ち主が forward）+ `kept.ts` | — | 中 | 訪問をまたぐ会話 | やらない |
| 埋め込みの出力 | `llama_get_embeddings*`・`llama_pooling_type` | 無い。台帳: T318（最後の隠れ状態を返す道、MLX の Clef のため） | forward から層の出を返す口 + Python | 狭まらない | 小〜中 | Clef（T316〜）の前提 | 後で（T318） |
| LoRA・制御ベクトル | `llama_adapter_lora_init`・`llama_set_adapters_lora`・`llama_set_adapter_cvec` | 無い | LoRA は変換のときに重みに足せば Python だけ（量子化の前の重みが要る） | 狭まらない | 中 | ページに使う所が無い | やらない |
| 複数の列の同時の生成 | `n_seq_max`・`n_parallel`、`llama_batch` | 無い（1 列。プロンプトは 16〜64 トークンをまとめて通す） | forward | — | 大 | サーバの機能 | やらない |
| モデルを量子化して書く | `llama_model_quantize` | 無い（変換は int8・6 ビット・3 値に直して保存するだけ） | — | — | — | — | やらない |

## 7. 速さの形（参考）

llama.cpp の CPU の道は、OS のスレッド、K 量子化と i 量子化の内積で活性値を 8 ビットの塊（`Q8_K`、`ggml.c` の `type_traits` に在る）に丸める形、KV の型は選べる（`llama.h` の `type_k`。既定の値は読んでいない）。ここは、取りまとめ役とソフトウェアスレッドが共有メモリの上で行の塊を取り合い、int8 と 3 値は relaxed SIMD の内積、塊の型は自前の `matmul_sb*`、KV の型は `footprint()` が決める。測った差は T364 の Saluki 27B だけ: 同じ CI のランナー（4 論理コア）で、ページの forward は素の llama.cpp の 0.47 倍（EPYC 9V74、0.52 tok/s）と 0.56 倍（Neoverse-N2、0.75 tok/s）。int8 のモデルでの比べ（wllama と同じ GGUF で: T95）は未着手で、数字は無い。

## 確かめたこと・読んだだけのこと・見ていないこと

- **確かめた（ソースの名前と数を機械で数えた）**: 型 35、混ぜ方 36、語彙の種類 9、前分割の型 62・名前 93・枝 37、サンプラーの関数 20・common の種類 11 と既定の並びと値、形 157。ここの `SOURCES`・`BLOCKS`・`LAYOUTS`・`FAMILIES`・`GGUF_PRETOKENIZERS`・`Sampling`・`hfEntry()`・`folder.ts` の門・`from_gguf()` の断り。
- **確かめた（実物）**: 上位 500 の数、13 ファイルの頭（型の割合、語彙の種類、前分割の名前、BOS と EOS、書式の有無）、`from_gguf()` が 3 つの実物の頭に言うこと。
- **読んだだけ**: `llama-quant.cpp` の混ぜ方の枝（Q4_K_M・Q5_K_M・IQ4_XS は実物と合った。ほかの混ぜ方は実物を見ていない）、形の違いの 1 行（`src/models/*.cpp` の名前の拾い読み）、llama.cpp の速さの形。
- **見ていない**: llama.cpp の書式の読み手（`common/chat*`）と文法、`unicode.cpp`、サーバ（`tools/server`）、マルチモーダル（`mtmd`）、Q1_0・Q2_0・TQ の並び、Gemma・Llama・Mistral のカードの標本抽出の設定、足す型を int8 に直したときの質と変換の秒、上位 500 より下のリポジトリ、名前に型の札が無い 32 リポジトリの中身。
- 上位 500 の数え方: ファイル名の札（`Q4_K_M` など）で数えた。`mmproj` のファイルは除いた。形の名前は API の `gguf.architecture`。

## 選ばなかった案（参考に 1 行）

Python が層を回し、掛け算だけ ctypes でカーネルを呼ぶ forward をもう 1 回作って、今の TypeScript の forward と速さを並べる: Pyodide の中のメモリは 32 ビットでスレッドと分け合えないので 1 コア・4 GB までのモデルに限られ、T93 の記録では Python の輪を外すだけで 1.15〜1.42 倍だった。

## 付録 A: ggml の型（`enum ggml_type`、生きている 35）

読める（14）: F32・F16・BF16・Q8_0・Q2_K・Q4_K・IQ2_XXS・IQ2_XS・IQ3_XXS・IQ1_S・IQ3_S・IQ2_S・IQ4_XS・IQ1_M。
読めない重みの型（15）: Q4_0・Q4_1・Q5_0・Q5_1・Q8_1・Q3_K・Q5_K・Q6_K・IQ4_NL・TQ1_0・TQ2_0・MXFP4・NVFP4・Q1_0・Q2_0。
重みの型でない（6）: Q8_K・I8・I16・I32・I64・F64。
ここだけが読む（ggml に無い、Prism ML の fork）: PQ2_0（142）・PTQ1_0（143）。

## 付録 B: 上位 500 の GGUF のリポジトリの数（2026-10-13）

- ダウンロードは 12,335,986 から 75,703。形の名前は 80。pipeline は text-generation 190・image-text-to-text 118・札なし 92・ほか 100。
- 形（多い順）: qwen35 91・gemma4 45・llama 42・qwen3 36・qwen35moe 32・qwen2 21・qwen4exp 15・名前なし 15・wan 12・clip 11・qwen3vl 10・qwen_image 9・gemma3 9・flux 8・audiocpp 6・qwen3moe 6・bert 6・deepseek4 6・lfm2 6・gpt-oss 5・granite 5・phi3 5・parakeet 4・laguna 4・asr 4・muse-glimmer 4・spark2_5 4・deepseek2 4・mistral3 3・ほか。
- 型の札を持つリポジトリ（全部 500 / ここの形の 202）: Q8_0 371 / 153、Q4_K_M 354 / 162、Q6_K 327 / 150、Q5_K_M 282 / 140、Q2_K 227 / 103、Q3_K_M 219 / 104、Q4_K_S 188 / 88、Q5_K_S 181 / 87、BF16 168 / 57、Q3_K_S 158 / 76、IQ4_XS 151 / 82、Q4_0 147 / 61、F16 122 / 45、Q3_K_L 114 / 70、IQ2_M 99 / 50、IQ3_XXS 95 / 38、Q4_1 84 / 35、IQ4_NL 74 / 37、IQ2_XXS 67 / 29、IQ1_M 64 / 29、IQ3_XS 58 / 40、IQ3_M 58 / 32、Q5_0 52 / 17、IQ1_S 52 / 28、IQ2_XS 52 / 28、IQ3_S 36 / 12、IQ2_S 31 / 17、Q5_1 27 / 4、F32 27 / 3、TQ1_0 6 / 0、TQ2_0 2 / 0。札が見つからない 32 / 13。
- Q4_K_M が在って Q8_0 が無い: 51 / 23。札が在って Q8_0 も F16・BF16・F32 も無い（いまは読める型のファイルが 1 つも無い）: 80 / 29。
- **ここの形の 202 のうち多くは 27B**（`qwen35` の 91 の大半）: Q8_0 は 16 GiB に入らず、塊のまま持てる小さい混ぜ方だけが候補になる（どの混ぜ方が入るかは未計測）。

## 付録 C: 頭を読んだ 13 ファイル

| ファイル | 形 | 語彙 | 前分割 | BOS / EOS | 型（重みの割合） |
|---|---|---|---|---|---|
| unsloth/Qwen3.5-4B-GGUF の Q4_K_M | qwen35 | gpt2 | qwen35 | 無し / 248046 | Q4_K 57.5・Q6_K 24.4・Q5_K 18.0・Q8_0 0.1 |
| 同 IQ4_XS | qwen35 | gpt2 | qwen35 | 無し / 248046 | IQ4_XS 78.3・Q6_K 15.1・Q5_K 6.5・Q8_0 0.1 |
| 同 Q8_0 | qwen35 | — | — | 無し | `from_gguf()` は BOS で断る |
| Qwen/Qwen3-4B-GGUF の Q4_K_M | qwen3 | gpt2 | qwen2 | 151643 / 151645 | Q4_K 78.0・Q6_K 22.0 |
| 同 Q5_K_M | qwen3 | gpt2 | qwen2 | 同 | Q5_K 78.0・Q6_K 22.0 |
| 同 Q8_0 | qwen3 | — | — | — | `from_gguf()` は通る |
| bartowski/Meta-Llama-3.1-8B-Instruct-GGUF の Q4_K_M | llama | gpt2 | llama-bpe | 128000 / 128009 | Q4_K 80.9・Q6_K 19.1 |
| 同 Q8_0 | llama | — | — | — | `rope_freqs.weight` が在り、`from_gguf()` は断る |
| LiquidAI/LFM2.5-230M-GGUF の Q4_K_M | lfm2 | gpt2 | lfm2 | 1 / 7 | Q4_K 63.5・Q6_K 36.5 |
| bartowski/Qwen2.5-32B-Instruct-GGUF の Q4_K_M | qwen2 | gpt2 | qwen2 | 151643 / 151645 | Q4_K 83.3・Q6_K 16.7 |
| rippertnt/HyperCLOVAX-SEED-Text-Instruct-1.5B の Q4_K_M | llama | gpt2 | dbrx | 100257 / 100275 | Q4_K 76.4・Q6_K 23.6 |
| TheBloke/Mistral-7B-Instruct-v0.2-GGUF の Q4_K_M | llama | llama | 無し | 1 / 2 | Q4_K 84.3・Q6_K 15.7 |
| TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF の Q4_K_M | llama | llama | 無し | 1 / 2 | Q4_K 83.1・Q6_K 16.9 |

13 のどれも書式（`tokenizer.chat_template`）を持つ。リビジョンは `main`（固定していない）。

## 付録 D: llama.cpp の形の名前（`enum llm_arch`、157）

CLIP LLAMA LLAMA4 DECI FALCON BAICHUAN GROK GPT2 GPTJ GPTNEOX MPT STARCODER REFACT BERT MODERN_BERT NOMIC_BERT NOMIC_BERT_MOE NEO_BERT JINA_BERT_V2 JINA_BERT_V3 EUROBERT BLOOM STABLELM QWEN QWEN2 QWEN2MOE QWEN2VL QWEN3 QWEN3MOE QWEN3NEXT QWEN3VL QWEN3VLMOE QWEN35 QWEN35MOE CLEF VLLM_DECISION2 QWEN4EXP PHI2 PHI3 PHIMOE PLAMO PLAMO2 PLAMO3 CODESHELL ORION INTERNLM2 MINICPM MINICPM3 GEMMA GEMMA2 GEMMA3 GEMMA3N GEMMA4 GEMMA4_ASSISTANT GEMMA_EMBEDDING GEMMA_EMBEDDING2 STARCODER2 MAMBA MAMBA2 MAPLE JAMBA FALCON_H1 XVERSE COMMAND_R COHERE2 COHERE2MOE DBRX OLMO OLMO2 OLMOE MUSE_GLIMMER OPENELM ARCTIC DEEPSEEK DEEPSEEK2 DEEPSEEK2OCR DEEPSEEK32 DEEPSEEK4 CHATGLM GLM4 GLM4_MOE GLM_DSA BITNET T5 T5ENCODER JAIS JAIS2 NEMOTRON NEMOTRON_H NEMOTRON_H_MOE EXAONE EXAONE4 EXAONE_MOE RWKV6 RWKV6QWEN2 RWKV7 ARWKV7 GRANITE GRANITE_MOE GRANITE_HYBRID GRANITE_SWITCH GRANITE_SWA CHAMELEON WAVTOKENIZER_DEC PLM BAILINGMOE BAILINGMOE2 BAILINGMOE3 DOTS1 DOTS3NOTE ARCEE AFMOE LAGUNA ERNIE4_5 ERNIE4_5_MOE HUNYUAN_MOE HUNYUAN_DENSE HUNYUAN_VL HY_V3 HY_V4 SMOLLM3 OPENAI_MOE LFM2 LFM2MOE DREAM SMALLTHINKER LLADA LLADA_MOE SEED_OSS GROVEMOE APERTUS MINIMAX_M2 COGVLM RND1 PANGU_EMBED MISTRAL3 MISTRAL4 PADDLEOCR MIMO2 STEP35 SPARK2_5 LLAMA_EMBED MAINCODER KIMI_LINEAR KIMI_K3 GLM5_NEXT TALKIE MELLUM EAGLE3 MINIMAX_M3 DFLASH NANBEIGE QWEN3TTS POCKETTTS MINIMAX_01 HRM_TEXT K2_HORIZON

ここの GGUF の形の名前（9）: LLAMA・QWEN2・QWEN3・QWEN35・GRANITE・SMOLLM3・LFM2・GPT2・GPTNEOX。Mistral（v0.x）は GGUF では `llama` と書かれる。

## 付録 E: 前分割の型（`enum llama_vocab_pre_type`、62）

DEFAULT LLAMA3 DEEPSEEK_LLM DEEPSEEK_CODER FALCON MPT STARCODER GPT2 REFACT COMMAND_R STABLELM2 QWEN2 OLMO DBRX SMAUG PORO CHATGLM3 CHATGLM4 VIKING JAIS TEKKEN SMOLLM CODESHELL BLOOM GPT3_FINNISH EXAONE CHAMELEON MINERVA DEEPSEEK3_LLM GPT4O SUPERBPE TRILLION BAILINGMOE LLAMA4 PIXTRAL SEED_CODER HUNYUAN KIMI_K2 HUNYUAN_DENSE GROK_2 GRANITE_DOCLING MINIMAX_M2 AFMOE SOLAR_OPEN YOUTU EXAONE_MOE QWEN35 TINY_AYA JOYAI_LLM JAIS2 GEMMA4 SARVAM_MOE MINICPM5 WHITESPACE GRANITE_EMB_MULTI MELLUM2 LAGUNA HY_V4 SPARK2_5 UFAKZEKA MMBERT K2_HORIZON

正規表現の並びが同じ型の組（2 つ以上のもの）: {LLAMA3・DBRX・SMAUG・CHATGLM4}、{GPT2・MPT・OLMO・JAIS・TRILLION・GRANITE_DOCLING}、{STARCODER・REFACT・COMMAND_R・SMOLLM・CODESHELL・EXAONE・MINERVA・MELLUM2}、{STABLELM2・QWEN2・HUNYUAN・SOLAR_OPEN・GROK_2}、{DEEPSEEK3_LLM・HUNYUAN_DENSE・JOYAI_LLM・HY_V4}、{PORO・BLOOM・GPT3_FINNISH}、{GPT4O・MINIMAX_M2}、{GEMMA4・SARVAM_MOE・MMBERT}（生の UTF-8）。枝を持たない型（既定の枝に落ちる）: CHATGLM3・LLAMA4・PIXTRAL（名前の `llama4` は GPT4O の型に、`pixtral` は LLAMA3 の型に引かれる）。
