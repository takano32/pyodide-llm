# T648: Firefox Translations の推論の手順と、pyodide-llm に載せる道筋

2026-11-05。調べるだけで、実装はしていない。モデルの二進形式と形（層の数・幅・int8 の詰め方・ショートリストの形）は対の記録 [t648-firefox-translations-format-2026-11-05.md](t648-firefox-translations-format-2026-11-05.md) にあり、ここでは繰り返さない。ここは推論の手順・参考の実装・載せる道筋。

捨ての道具は `.tmp/t648-engines/scripts/` に置いた（コミットしない）: `marian_ref.py`（NumPy で 3 通りに訳す: 下の 2）・`fx_ref.py`・`fx_lex.py`（fxtranslate をショートリストなし・ありで）・`compare.py`。

## 結論

- **手本は fxtranslate**（Rust、MPL-2.0）。Marian の C++ を写さず、1 文の推論を 2,000 行ほどで書いていて、式の置き場がはっきりしている。`results/bergamot.jsonl` はこれの **貪欲（greedy）・ショートリストなし・int8（lean-embed）・`translate_long`** の訳。
- **探索は貪欲**。fxtranslate は貪欲しか持たず、Firefox 自身も `beam-size: 1`。だから「同じ原文で同じ訳」は fxtranslate そのものを参照にできる（手元で 100 文とも `results/bergamot.jsonl` と同じ文が出た）。
- **ただし Firefox はショートリストを使い、fxtranslate の既定（と bergamot.jsonl）は使わない**。ショートリストを付けると 100 文のうち 38 文の訳が変わった（en-ja 33・ja-en 5）。どちらを参照にするかは持ち主の決め（下の「決めること」）。
- **int8 の掛け算は intgemm の `int8shiftAlphaAll`**: 活性値は行列ごとの**固定の係数**で ±127 に丸め、重みは行列ごとに 1 つの係数、整数の和を 1 回だけ float に戻す。pyodide-llm の int8（32 個ごとの**動的な**係数）とは活性値の丸めが違う。NumPy で比べると、intgemm を真似た形は fxtranslate と ja-en 47/50・en-ja 42/50、pyodide-llm の形は 35/50・29/50、float は 36/50・27/50 が同じ訳になった（下の 2）。
- **道筋の勧め**: ページの `createForward()` を曲げず、**翻訳のエンジンを別に作る**。モデルが小さい（44〜60 MB、幅 384〜512）ので、層の輪を Python に置いても 1 コア・32 ビットのメモリで足りる（2026-10-13 の決めの「狭まる」に当たらない）。行列積だけをカーネルに（intgemm と同じ数を出す新しいカーネル 1 つ）。

## 出典

| 実装 | 置き場 | コミット | ライセンス | 読んだ所 |
|---|---|---|---|---|
| fxtranslate（Rust） | https://github.com/gregtatum/translations の `inference-rs/` | `f2f77cd3`（2026-10-04） | MPL-2.0（`crates/fxtranslate/Cargo.toml`） | `crates/fxtranslate/src/engine.rs`・`ops.rs`・`weights.rs`・`shortlist.rs`・`segment.rs`・`spm.rs`、`architecture.md`、`gemm-backends.md`、`notes/12-wasm.md`・`notes/23-float-model-support.md` |
| mozilla/translations の推論 | https://github.com/mozilla/translations の `inference/` | `8eada038`（2026-10-29） | MPL-2.0（README の 5 行目: モデルのファイルも MPL 2.0） | `inference/src/translator/text_processor.cpp`・`html.h`、`inference/marian-fork/src/models/transformer.h`・`src/rnn/cells.h`・`src/tensors/cpu/intgemm_interface.h`・`src/common/config_parser.cpp`（marian は MIT） |
| bergamot-translator | https://github.com/mozilla/bergamot-translator | `5ae1b1eb`（浅く取った HEAD） | MPL-2.0 | 置き場の確かめだけ（中身は上の `inference/` に移っている） |
| slimt（C++） | https://github.com/jerinphilip/slimt | `9f0b1a20` | **GPL-2.0**（行を写さない） | `slimt/Model.cc` 151〜168 行（貪欲だけ）を見ただけ |
| Firefox の推論のワーカー | https://github.com/mozilla-firefox/firefox の `toolkit/components/translations/content/translations-engine.worker.js` | `935e9f7e`（main） | MPL-2.0 | 35〜117 行（揃え・前処理）、556〜573 行（Marian の設定）、701〜715 行（頼みの options） |

行の範囲は以下、断りの無いものは fxtranslate の `crates/fxtranslate/src/` の中。

## 1. 1 文の推論の手順

記号: D = 幅（384、ja-en 3.0 は 512）、H = head 8、dk = D / H、F = FFN の幅（1536 か 2048）、V = 語彙。行列 W は論理の形 [K, N] で、`y = x W + b`。

1. **前処理**（Firefox のワーカー `cleanText()` 94〜117 行）: 前後の空白を外して後で戻す、ソフトハイフン U+00AD を消す、ja・ko・zh では「。！？」の直後の “ の前に空白を入れる（Intl.Segmenter の切りを直すため）。
2. **文に切る**: Firefox は WASM の中から `Intl.Segmenter(lang, {granularity: "sentence"})`（`inference/src/translator/text_processor.cpp` 109〜210 行）。fxtranslate は ICU4X の文の切り（`segment.rs`、`engine.rs` 361〜387 行）。
3. **語片に**: sentencepiece の unigram（`spm.rs` 1〜16 行: charsmap で正規化、空白を ▁、Viterbi、バイトの語片）。**BOS は無く、最後に EOS（`</s>`、ID 0）を足す**（`engine.rs` 332〜335 行）。127 語片を越える文は 127 ずつに切って別々に訳し、つなぐ（`translate_within_window` 389〜406 行、`segment.rs` 20〜31 行: Marian の `max-length-break` 128 − 1）。
4. **埋め込み**（`embed_into` 1827〜1878 行、Marian の `transformer.h` 92〜96 行）:
   `x_t = √D · E[id_t] + PE(t)`、`PE(t)[c] = sin(t · f_c + o_c)`、`f_c = 1e-4 ^ ((c mod D/2) / (D/2 − 1))`、`o_c = ⌊c / (D/2)⌋ · π/2`（前半が sin、後半が cos。交互でない）。E は int8 の行を係数で割って float に。**デコーダの位置 0 は埋め込みを 0 にし PE(0) だけ**（Marian は目標の埋め込みを右にずらして頭を 0 で埋める。種の EOS を埋めると訳の頭が小文字になった: `notes/23-float-model-support.md`）。埋め込みの後に norm は無い。
5. **エンコーダの層 ×6**（`encoder_layer` 889〜899 行、`multihead` 1531〜1599 行、`ffn` 1778〜1797 行、`postnorm` 1800〜1808 行）。**post-norm**（norm は足した後）、自己注意は双方向（マスク無し）:
   ```
   q = x Wq + bq; k = x Wk + bk; v = x Wv + bv          # 各 [S, D]、head ごとに dk ずつ
   a_h = softmax(q_h k_hᵀ / √dk) v_h                     # 行の最大を引いてから exp
   x = LN(x + concat(a) Wo + bo)                         # LN は self_Wo_ln
   x = LN(x + relu(x W1 + b1) W2 + b2)                   # LN は ffn_ffn_ln
   LN(z) = γ ⊙ (z − mean) / sqrt(var + 1e-6) + β          # 分散は 2 回目の和、eps 1e-6（engine.rs 23 行、ops.rs 27〜75 行）
   ```
   エンコーダの出 C（[S, D]）は 1 文で 1 回。
6. **原文への注意の K・V を前もって**（`cross_attn_kv` 1117〜1131 行）: デコーダの層ごとに `K_l = C Wk + bk`、`V_l = C Wv + bv`。全部のステップで同じ（メモ化で、数は毎ステップ作るのとビット単位で同じ）。
7. **デコーダの 1 ステップ ×層 4（ja-en 3.0 は 2）**（`decode_step` 947〜990 行。Marian の `cells.h` 982〜1066 行の SSRU）。**自己注意は無く、SSRU**。状態は層ごとに D 個の float 1 本（文の頭で 0）:
   ```
   u = x                                   # 埋め込み（または前の層の出）
   x̃ = u W                                 # rnn_W、bias 無し
   f = u Wf + bf                           # rnn_Wf、rnn_bf
   c = σ(f) ⊙ c_prev + (1 − σ(f)) ⊙ x̃      # 状態を c に置き換える（ops.rs 103〜125 行の highway）
   x = LN(u + relu(c))                     # rnn_ffn_ln
   x = LN(x + Attn(x Wq + bq, K_l, V_l) Wo + bo)   # context_Wo_ln、マスクは原文の長さだけ
   x = LN(x + relu(x W1 + b1) W2 + b2)     # ffn_ffn_ln
   ```
   ゲートは今の入力だけを読み、前の状態を読まない。
8. **出力の層**（`project_argmax` 832〜877 行、`weights.rs` 598〜654 行）: `logits = h Eᵀ + b_out`（`decoder_ff_logit_out_b`）。E は目標の埋め込みと同じ表（共有の語彙なら `Wemb` 1 つ、en-ja のように語彙が分かれていれば `decoder_Wemb`）。**softmax は取らない**（最大を取るだけ）。
9. **次のトークン**: 全語彙（かショートリストの候補）の最大。**同点は小さい ID**（`argmax` 1958〜1968 行: 厳密に大きいときだけ替える）。ビームも長さの罰も無い。
10. **止まり**: EOS が出たら止め、EOS は出さない。長さの上限は `min(⌈2·S⌉ + 4, 256)`（S は EOS を含む原文の語片の数、`greedy` 454〜499 行。Firefox の設定は `max-length-factor: 2.0`）。
11. **文に戻す**: 語片を並べ ▁ を空白に（`spm.rs`）。文の間は原文の空白でつなぐ（`segment.rs` の `reassemble()`）。

### 寸法と 1 文の重さ（式から数えた、ja-en 3.1 の幅 384）

1 ステップのデコーダ: 層 4 × (D² × 4 + D·F × 2) ≈ 7.1M 積和、出力の層 D × V（32k なら 12.3M）。エンコーダは 1 語片あたり 層 6 × (D² × 4 + D·F × 2) ≈ 10.6M。30 語片の文を 40 語片に訳すと、エンコーダ 0.32G + デコーダ 0.28G + 出力 0.49G ≈ 1.1G 積和。ショートリストがあれば出力は候補の数ぶん（数千）で、1 ステップの半分以上が消える。

## 2. intgemm の int8 の掛け算

### 何をしているか（`ops.rs` 456〜578 行、`weights.rs` 301〜357・794〜864 行。Marian は `intgemm_interface.h`）

- 重み: int8 の [N, K]（論理の [K, N] の転置）と **行列に 1 つの係数 qB**（データの後ろの float32）。
- 活性値: **行列ごとの固定の係数 qA**（`<行列の名前>_QuantMultA`、float32 の 1 値。出力の層は `none_QuantMultA` か、語彙が分かれていれば int8 の 1 値 `decoder_Wemb_QuantMultA` で、値は raw / 係数）。
  `q = clamp(round_ties_even(x · qA), −127, 127) + 127`（u8 の 0〜254: 「ずらし」）。**対称。動的な係数ではない**。範囲外は切る。
- 和: `acc = Σ_k q_k · W[k, n]` を **i32 で厳密に**（x86 の AVX2 の intgemm は i16 で飽和しうる: `gemm-backends.md`。ARM の i8mm、x86 の VNNI、スカラは厳密）。
- 戻し: `un = 1 / (qA · qB)`、`y = un · acc + (b − 127 · un · Σ_k W[k, n])`（ずらしの補正を bias に畳む）。
- **float32 のまま掛けるもの**: 注意の q·k と重みつきの和、softmax、LN、ReLU、σ、highway、埋め込みの表引き（int8 を係数で割る）。int8 で掛けるのは全部の affine（自己注意と原文の注意の Wq・Wk・Wv・Wo、FFN の W1・W2、SSRU の W・Wf）と出力の層（fxtranslate の lean-embed: Python の版はこれ）。

### pyodide-llm の int8 に載るか

pyodide-llm の `matmul_q8`（`kernels/kernel/matmul.ts` 78〜98 行）は、重みも活性値も **32 個ごとに係数**を持ち、活性値の係数は `quantize_x`（`kernels/kernel/quantize.ts` 124〜154 行）が群の最大 / 127 で毎回決め、群ごとの整数の和に「重みの係数 × 活性値の係数」を掛けて float で足す。

- **重みは詰め直しても損が無い**: int8 をそのまま使い、32 個ごとの係数を全部 1 / qB にすればよい（値は同じ整数）。
- **活性値は違う丸めになる**: 固定の qA と群ごとの動的な係数は別の整数を作る。足し方も「1 回だけ float に」と「群ごとに float」で違う。だから同じ訳になる保証は無い。
- **測った**（`marian_ref.py`、100 文、各文を 1 単位で `translate()`、ショートリストなし）:

  | 形 | ja-en（50 文） | en-ja（50 文） |
  |---|---|---|
  | intgemm を真似る（固定の qA・i32・ずらしの補正） | 47 / 50（違う 3 文のうち 97 番は fxtranslate の復号の癖: 下の 5） | 42 / 50 |
  | pyodide-llm の int8（重みは同じ整数、活性値は 32 個ごとの動的な係数） | 35 / 50 | 29 / 50 |
  | float32（重みを広げ、活性値は丸めない） | 36 / 50 | 27 / 50 |

  数は fxtranslate の `translate()` と同じ文の数。読み方: (1) pyodide-llm の int8 の形は float と同じくらい fxtranslate から離れる（3 割ほどの文で訳が変わる）。訳の質が落ちるという意味ではなく、intgemm の丸めに乗った貪欲の道から外れるということ。(2) intgemm を真似ると大半は戻るが全部ではない。残りは NumPy の float の足し順（BLAS）・sin・exp の違いで同点に近い語が入れ替わったものと読んだ（fxtranslate の語片の ID と突き合わせていないので未確認: 下）。違う文の訳はどれも言い換え（「プッシュされている」と「プッシュされた」など）。
- **だから「fxtranslate と同じ訳」を試験にするなら、新しいカーネルが要る**: 固定の係数で ±127 に丸める量子化 1 つと、行全部を i32 で足して 1 回だけ戻す積 1 つ（`matmul_q8` の `dot32` はそのまま使える。i16 で 2 つの積を足しても 2·127·128 = 32512 で溢れない）。relaxed の道（`matmul_q8r` の 7 ビットの活性値）は 0〜254 を持てないので使えない。
- **fxtranslate の数そのものも端末で揺れうる**: x86 の AVX2 の intgemm は i16 で飽和しうる（Firefox の WASM は SSE の `maddubs` を写した道: `notes/12-wasm.md` 242〜247 行）。fxtranslate と Marian（どちらも int8）の貪欲の訳は en-ru・en-fr・en-es の各 20 文で 18〜19 文が同じ（`notes/23-float-model-support.md` 64〜76 行）。残りは同点に近い語の入れ替わり。

## 3. ショートリスト

- 形と読み方（`shortlist.rs` 1〜125 行、Marian の `src/data/shortlist.cpp`）: 頭の `firstNum` 個の目標の ID（頻度の高い語）、共有の語彙なら原文の各語片そのもの、各原文の語片の訳の候補、を合わせて 8 の倍数まで詰める。**1 文に 1 回**作り、出力の層の掛け算と最大を候補の列だけにする（Marian は `intgemmSelectColumnsB`）。
- 日英のモデルにある（Remote Settings の `fileType: "lex"`、`lex.50.50.enja.s2t.bin` と `lex.50.50.jaen.s2t.bin`）。Firefox のワーカーは `lex` を必ず渡す（`translations-engine.worker.js` 526〜583 行）。
- **訳を変える**: fxtranslate に付けて 100 文を `translate()` で訳すと 38 文が変わった（en-ja 33、ja-en 5）。良くなったものも悪くなったものもある（例: en-ja の 1 番は「systemctl restartworkwork」→「systemctl restart ワーカー」、ja-en の 86 番は「when he/she agrees to」→「when he/shering in」）。en-ja は語彙が分かれているので原文の語片は候補に入らない（fxtranslate の `shared = false`、Marian も同じかは未確認）。
- 速さには効く（出力の層が 1 ステップの半分以上: 上の 1）。

## 4. 長い文と HTML

- fxtranslate の `translate_long`（`engine.rs` 361〜406 行）: ICU4X で文に切り、文ごとに訳して原文の空白でつなぐ。1 文が 127 語片を越えたら語片で切る（継ぎ目は受け入れる）。文の間の文脈は使わない。`bergamot.jsonl` はこれで、`translate()` で 1 単位に訳したものとは 16 文が違う（文が 2 つ以上ある原文）。
- Firefox: `Intl.Segmenter` で文に切り（上の 1 の 2）、127 語片で包む（`text_processor.cpp` の `wrap()`）。文は `mini-batch-words: 1024` までまとめて 1 回に訳す（文ごとに独立）。
- **HTML**: Firefox は `html: isHTML` を bergamot に渡し、bergamot の `HTML` クラス（`inference/src/translator/html.h`）が札を外して訳し、**原文の注意から作った硬い対応（`alignment: "soft"` を頼む）**で札を訳の語片に戻す。fxtranslate には HTML が無い（`translate_aligned` 656 行の対応はある）。pyodide-llm のチャットは平文なので、最初は HTML を持たなくてよい。

## 5. 探索の既定と、貪欲の参照

- fxtranslate: **貪欲だけ**（`architecture.md`「Greedy only (argmax; no beam search in this engine)」、`engine.rs` 454〜499 行）。設定で幅を選ぶ口は無い。
- Firefox: `beam-size: "1"`、`normalize: "1.0"`、`word-penalty: "0"`、`max-length-factor: "2.0"`、`gemm-precision: int8shiftAlphaAll`（`translations-engine.worker.js` 556〜573 行）。幅 1 なので長さの罰は効かない。slimt も貪欲だけ。
- **`results/bergamot.jsonl` の設定**（`scripts/run_bergamot.py` と fxtranslate-py の `Cargo.toml` から）: fxtranslate 0.5.0（PyPI）、貪欲、**ショートリストなし**、`translate_long`（ICU4X）、lean-embed（出力の層も int8）。ja-en は **Android の 3.1** の模型（対の記録）。
- **手元で作り直せた**: 同じ wheel（`bergamot-vs-shisa/.venv`）で 100 文を `translate_long` すると 100 文とも `bergamot.jsonl` と同じ文。この機械（Neoverse-N1: i8mm が無い）での fxtranslate の積は `backend() == "scalar"`（i32 で厳密）。速さは 1 文 0.86 秒（en-ja、load 10 前後で測ったので読まない）。
- **fxtranslate の復号の癖**: バイトの語片を文字に戻さない。ja-en の 97 番は「<0xE2><0x98><0x95>」と書き、sentencepiece なら「☕」。同じ訳を試験にするなら、この行の扱いを決める。

## 6. pyodide-llm に載せる道筋

### いまのエンジンと合わない所

| モデルの事実 | いまのエンジン |
|---|---|
| エンコーダ（双方向、6 層）とデコーダ（4 層）の 2 つの束、層の数が違う | 1 つの束（`createForward()`、`LAYOUTS`） |
| **post-norm**: `x = LN(x + sub(x))` | pre-norm: 層は「流れの norm を読み、流れに足す」（`forward/kinds.ts` の `Step`） |
| LayerNorm（bias あり、eps 1e-6） | LayerNorm は GPT-2 の形で eps 1e-5 固定（`kernels/kernel/activations.ts` 22 行、AGENTS.md） |
| 正弦波の位置（加える）、デコーダの位置 0 は埋め込み 0 | RoPE か学習した位置の表（GPT-2） |
| bias つきの全部の affine、ReLU の FFN | GPT-2 の FFN は bias つき GELU、ReLU は無い |
| SSRU（層ごとに D 個の状態） | 状態を持つ層の種類（`STATEFUL_LAYERS`・`LAYER_KINDS`: 畳み込み・linear）の形に合う |
| 原文への注意（K・V は 1 文で 1 回、ステップでは q だけ） | 無い |
| int8 は行列に 1 つの係数、活性値は固定の係数 | 32 個ごとの動的な係数 |
| チェックポイントは Marian の `.bin`（見出しに名前と形） | llama2.c の legacy 形式（見出し 7 つの int） |
| 書式なし、原文 → 訳 | チャットの書式・`generate()` |

### 案

**A（勧め）: 翻訳のエンジンを Python に別に作り、行列積だけをカーネルに。**
層の輪・LN・softmax・SSRU は NumPy（Pyodide の中、1 コア）、affine は ctypes のカーネル。ページの `createForward()`・`LAYER_KINDS`・計画の鍵には触らない（名前を比べない決まりはそのまま）。訳: (1) モデルが 44〜60 MB で 1 コア・32 ビットのメモリで足りる。層の輪を Python にすると狭まるという 2026-10-13 の決めは大きいモデルの話で、この家族は狭まらない。(2) post-norm・2 つの束・原文への注意は、今の流れの形（pre-norm の 1 つの束）を曲げないと載らない。曲げると Llama 系の道に「一緒にしか正しくない」枝が増える。(3) 持ち主の「なるべく Python」。(4) 1 ステップの呼び出しは 層 4 × 約 10 回のカーネルと小さな NumPy で、Python から JS の呼び出しは 1 回 1.3〜1.7 µs（AGENTS.md）。速さは未計測。
**B: TypeScript の forward に翻訳の別の engine（`createTranslator()`）を作る。** スレッドと GPU を後で使えるが、規模は A の数倍。A が遅いと分かってから。
**C: `createForward()` に post-norm と 2 つの束を足す。** 勧めない（上の (2)）。

### 子タスクの候補（A の場合、取りかかる順）

| 番号の案 | 中身 | 規模の見込み |
|---|---|---|
| T648.1 | 取り込み: Marian の `.bin` の読み手（Python、変換器の側）。**変換しない**: 行列は int8 のまま、名前と置き場の表を作る。sentencepiece の語彙は今の読み手で（対の記録: 13 文で同じ ID）。ショートリストの読み手 | 小（読み手 150 行ほど、試験） |
| T648.2 | NumPy の参照の翻訳のエンジン（`engine/` の下に翻訳の部品: エンコーダ・SSRU・原文への注意・貪欲・止まり・長さの上限）。intgemm と同じ丸めを NumPy で。fxtranslate の訳と 100 文で比べる試験の道具（手で回す: wheel は PyPI） | 中（300〜400 行） |
| T648.3 | カーネル: 固定の係数の量子化と、i32 で行を足して 1 回戻す int8 の積（ずらしの補正つき）。`kernels-in-browser.ts` と smoke に足す。x86-64 と arm64 で測る | 小〜中 |
| T648.4 | ページ: 翻訳の項目（チャットの書式なし、生成の設定は貪欲だけ、向き）、worker の読み込みの道（`.bin`・語彙・lex の 3〜4 ファイル）、文に切る（`Intl.Segmenter` をページの側で: 試験は Node） | 中（文面は持ち主） |
| T648.5 | 配り方: `make models` で Remote Settings から取り（sha256 を見て zstd を広げる）`public/models/` に。CDN の添付は CORS を返さない（下）ので、ページから直に取れない | 小（サイトが約 100 MB 増える: 目安 1 GB） |
| 後 | ショートリスト・HTML・B の TypeScript の道・GPU | 後で |

### 決めること（持ち主に）

1. **参照の訳**: (a) fxtranslate の既定（ショートリストなし: `bergamot.jsonl` と同じ）か、(b) Firefox と同じショートリストあり。勧めは (a) を最初の試験に、(b) は後で（ショートリストは速さに効くので入れるなら試験も替える）。
2. **ja-en の版**: Android の 3.1（幅 384、今の比べの材料）か、デスクトップの 3.0（幅 512、59.5 MB）。
3. **「同じ訳」の線**: intgemm の丸めを真似ても float の足し順・sin などで同点に近い語は入れ替わりうる（上の表の intgemm の形の行）。100 文の全部を求めるか、数で線を引くか。
4. 配るモデルのライセンス: mozilla/translations の README は「The model files are distributed under the MPL 2.0 license」。一覧の項目にはこれを写す（方針 8）。

## 確かめられなかったこと

- Remote Settings の添付の CDN（`firefox-settings-attachments.cdn.mozilla.net`）は、Origin つきの GET（Range）にも `Access-Control-Allow-Origin` を返さなかった。モデルの置き場の Google Cloud Storage（README の Model Registry）が CORS を返すかは未確認。
- Marian（`translator-cli`）の訳との比べ: ビルドしていない。fxtranslate の README と記録の数（18〜19 / 20）だけ。fxtranslate をソースから建てて語片の ID を出す道も試していない（外のコードを建てる操作は止められた）。だから intgemm を真似た NumPy の形と fxtranslate の違いが、語片の ID の違いか同点の入れ替わりかは分けていない。
- Marian でショートリストの候補に語彙の分かれたモデルの原文の語片が入るか（fxtranslate は入れない）。
- Pyodide の中での速さ（案 A）。fxtranslate の WASM の版（`fxtranslate-wasm`）の速さも測っていない。
- AVX2 の飽和が日英の 100 文で訳を変えるか。
