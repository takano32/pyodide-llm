# T648: Firefox Translations の日英モデルの形（ファイルから読んだもの）

2026-11-05。調べるだけで、実装はしていない。読む道具は `.tmp/t648-format/` に置いた（コミットしない）: `read_marian.py`（見出し）・`yml.py`（設定）・`ints.py`（int8 の範囲）・`translate.py`（NumPy で訳す確かめ）・`spm.py`・`tok_engine.py`（語彙）・`lex.py`（ショートリスト）・`fetch_extra.py`。

## 結論

- 3 本とも **Marian の transformer（エンコーダは自己注意、デコーダは SSRU + 原文への注意）**。post-norm、ReLU の FFN、正弦波の位置、head 8。
- 重みの行列は **int8、行列ごとに 1 つの係数**（行ごとでもグループごとでもない）。ファイルの int8 は **タイルに並べ替えていない**: 転置して丸めただけ。タイルの並べ替えは読み込みの時に Marian がする。
- 活性値の係数（alphas）も **行列ごとに 1 つの固定値**（`<行列の名前>_QuantMultA`）。
- 読み方の確かめ: 重みを float32 に戻して NumPy で greedy に訳すと、3 本とも正しい訳になった（下の「確かめ」）。
- 語彙は 3 本とも sentencepiece の unigram（nmt_nfkc、バイトの語片あり）。**pyodide-llm のエンジンの読み手で、試した 13 文は sentencepiece と同じ ID**。
- ショートリスト（`lex`）は Remote Settings に**ある**。手元の取得の道具（`scripts/fetch_models.py`）が取っていないだけ。
- **手元の ja-en は Android 向けの版**（3.1、`base-memory`）。デスクトップの Firefox が使うのは別の版（3.0、`base`: 幅 512、59.5 MB）。

## ファイルと版

Remote Settings（`translations-models-v2`）の項目。sha256 は手元のファイルと一致を見た。

| 向き | fileType | 名前 | 版 | architecture | 展開後の大きさ | 条件 |
|---|---|---|---|---|---|---|
| en-ja | model | model.enja.intgemm.alphas.bin | 3.0 | base-memory | 43,849,787 | なし |
| en-ja | srcvocab | srcvocab.enja.spm | 3.0 | base-memory | 796,275 | なし |
| en-ja | trgvocab | trgvocab.enja.spm | 3.0 | base-memory | 827,144 | なし |
| en-ja | lex | lex.50.50.enja.s2t.bin | 3.0 | base-memory | 4,128,360 | なし |
| ja-en | model | model.jaen.intgemm.alphas.bin | **3.1** | base-memory | 43,977,787 | `env.appinfo.OS == 'Android'` |
| ja-en | vocab | vocab.jaen.spm | 3.1 | base-memory | 1,443,222 | Android |
| ja-en | lex | lex.50.50.jaen.s2t.bin | 3.1 | base-memory | 9,348,172 | Android |
| ja-en | model | model.jaen.intgemm.alphas.bin | 3.0 | base | 59,504,955 | `env.appinfo.OS != 'Android'` |
| ja-en | vocab | vocab.jaen.spm | 3.0 | base | 1,443,222（3.1 と同じハッシュ） | Android 以外 |
| ja-en | lex | lex.50.50.jaen.s2t.bin | 3.0 | base | 9,346,816 | Android 以外 |

手元の取得の道具は版の大きいほうを採るので、ja-en は Android の 3.1 になっている。デスクトップの 3.0 とショートリスト 3 つは `.tmp/t648-format/fetched/` に取って読んだ。

## 1. Marian の二進形式（.bin）

形は marian-dev の `src/common/binary.cpp` の `loadItems()`・`saveItems()` から（すべて little-endian）。

| 位置 | 中身 |
|---|---|
| 0 | uint64 版（`BINARY_FILE_VERSION`）。3 本とも 1 |
| 8 | uint64 項目の数 n |
| 16 | 見出し n 個。各 32 バイト: uint64 名前の長さ（終わりの NUL を含む）・uint64 型・uint64 形の次元の数・uint64 データのバイト数 |
| … | 名前を n 個続けて（NUL 終わり） |
| … | 形を n 個続けて（int32 × 次元の数） |
| … | uint64 offset、続けて offset バイトの 0（データの始まりを 256 の倍数に） |
| … | データを n 個続けて。各項目のバイト数は 256 の倍数に 0 で詰めてある（3 本とも、どの項目も 256 にそろって始まり、詰めは全部 0 だった） |

型の番号（`src/common/types.h` の `Type`）: float32 = 0x0404、int8 = 0x0101、**intgemm8 = 0x4101**。3 本にあるのはこの 3 つだけ。

データの始まり: en-ja と ja-en 3.1 は 16,384、ja-en 3.0 は 12,544。終わりはファイルの終わりにちょうど合う。

### 型の中身

| 型 | 中身 |
|---|---|
| float32 | 値をそのまま、行優先 |
| intgemm8（行列） | int8 を要素の数だけ、すぐ後ろに float32 の **quantMult**（4 バイト、語にそろっていない）、そのあと 0 の詰め |
| int8 | `special:model.yml` だけ: YAML の文字（NUL なし、1339 バイト） |

## 2. 見出しの一覧

### 数と内訳

| | en-ja 3.0 | ja-en 3.1（手元） | ja-en 3.0（デスクトップ） |
|---|---|---|---|
| 項目の数 | 253 | 253 | 195 |
| エンコーダの層 | 6 × 22 項目 | 6 × 22 | 6 × 22 |
| デコーダの層 | 4 × 29 項目 | 4 × 29 | 2 × 29 |
| 層の外 | 5 | 5 | 5 |
| 重みの数（float と int8 の合計） | 43.54 M | 43.57 M | 59.10 M |
| うち埋め込み | 24.58 M（2 本） | 24.58 M（1 本） | 32.77 M（1 本） |
| エンコーダ / デコーダの行列 | 10.62 M / 8.26 M | 10.62 M / 8.26 M | 18.87 M / 7.34 M |
| 活性値の係数（`_QuantMultA`） | 69 | 70 | 54 |

（6 × 22 + 4 × 29 + 5 = 253、6 × 22 + 2 × 29 + 5 = 195 で、一覧は下の型で全部。）

### 層の外の項目

| 名前 | 型 | 形 en-ja | ja-en 3.1 | ja-en 3.0 |
|---|---|---|---|---|
| `encoder_Wemb` | intgemm8 | [32000, 384] | （無い） | （無い） |
| `decoder_Wemb`（en-ja）/ `Wemb`（ja-en） | intgemm8 | [32000, 384] | [64000, 384] | [64000, 512] |
| `decoder_Wemb_QuantMultA`（en-ja）/ `Wemb_QuantMultA`（ja-en） | **intgemm8** | [1, 1] | [1, 1] | [1, 1] |
| `decoder_ff_logit_out_b` | float32 | [1, 32000] | [1, 64000] | [1, 64000] |
| `none_QuantMultA` | float32 | （無い） | [1, 1] | [1, 1] |
| `special:model.yml` | int8 | [1339] | [1339] | [1339] |

### エンコーダの層 `encoder_l{1..6}_…`（D = 384、ja-en 3.0 は 512。F = 1536、ja-en 3.0 は 2048）

| 名前の後ろ | 型 | 形 |
|---|---|---|
| `self_Wq`・`self_Wk`・`self_Wv`・`self_Wo` | intgemm8 | [D, D] |
| `self_Wq_QuantMultA`・`self_Wk_…`・`self_Wv_…`・`self_Wo_QuantMultA` | float32 | [1, 1] |
| `self_bq`・`self_bk`・`self_bv`・`self_bo` | float32 | [1, D] |
| `self_Wo_ln_scale`・`self_Wo_ln_bias` | float32 | [1, D] |
| `ffn_W1` | intgemm8 | [D, F] |
| `ffn_W2` | intgemm8 | [F, D] |
| `ffn_W1_QuantMultA`・`ffn_W2_QuantMultA` | float32 | [1, 1] |
| `ffn_b1` / `ffn_b2` | float32 | [1, F] / [1, D] |
| `ffn_ffn_ln_scale`・`ffn_ffn_ln_bias` | float32 | [1, D] |

### デコーダの層 `decoder_l{1..4}_…`（ja-en 3.0 は 1..2）

| 名前の後ろ | 型 | 形 |
|---|---|---|
| `rnn_W`・`rnn_Wf` | intgemm8 | [D, D] |
| `rnn_W_QuantMultA`・`rnn_Wf_QuantMultA` | float32 | [1, 1]（2 つは同じ値: 同じ入力） |
| `rnn_bf` | float32 | [1, D] |
| `rnn_ffn_ln_scale`・`rnn_ffn_ln_bias` | float32 | [1, D] |
| `context_Wq`・`context_Wk`・`context_Wv`・`context_Wo` と `_QuantMultA` 4 つ・`context_bq`…`bo`・`context_Wo_ln_scale`・`_bias` | エンコーダの `self_` と同じ | 同じ |
| `ffn_…` 8 つ | エンコーダと同じ | 同じ |

`context_Wk_QuantMultA` と `context_Wv_QuantMultA` は全部の層で同じ値（エンコーダの出が入力）。手元の行ごとの値は `.tmp/t648-format/out/model.*.txt`（名前・型・形・バイト数・quantMult・alpha）。

## 3. 埋め込まれた設定（`special:model.yml`）

en-ja の全文:

```yaml
bert-train-type-embeddings: true
bert-type-vocab-size: 2
dec-cell: ssru
dec-cell-base-depth: 2
dec-cell-high-depth: 1
dec-depth: 4
dim-emb: 384
dim-rnn: 1024
dim-vocabs:
  - 32000
  - 32000
enc-cell: gru
enc-cell-depth: 1
enc-depth: 6
enc-type: bidirectional
factors-combine: sum
factors-dim-emb: 0
input-types: []
layer-normalization: false
lemma-dependency: ""
lemma-dim-emb: 0
output-omit-bias: false
right-left: false
skip: false
tied-embeddings: true
tied-embeddings-all: false
tied-embeddings-src: false
transformer-aan-activation: swish
transformer-aan-depth: 2
transformer-aan-nogate: false
transformer-decoder-autoreg: rnn
transformer-decoder-dim-ffn: 0
transformer-decoder-ffn-depth: 0
transformer-dim-aan: 2048
transformer-dim-ffn: 1536
transformer-ffn-activation: relu
transformer-ffn-depth: 2
transformer-guided-alignment-layer: last
transformer-heads: 8
transformer-no-affine: false
transformer-no-bias: false
transformer-no-projection: false
transformer-pool: false
transformer-postprocess: dan
transformer-postprocess-emb: d
transformer-postprocess-top: ""
transformer-preprocess: ""
transformer-rnn-projection: false
transformer-tied-layers: []
transformer-train-position-embeddings: false
type: transformer
ulr: false
ulr-dim-emb: 0
ulr-trainable-transformation: false
version: v1.12.14 2d067af 2024-02-16 11:44:13 -0500 
```

ja-en 3.1 との違いは 2 か所だけ: `dim-vocabs` が 64000 と 64000、`tied-embeddings: false`・`tied-embeddings-all: true`。ja-en 3.0（デスクトップ）は 3.1 から `dec-depth: 2`・`dim-emb: 512`・`transformer-dim-ffn: 2048` の 3 か所が違う。

`enc-cell`・`enc-type`・`dim-rnn`・`dec-cell-*-depth`・`aan`・`bert`・`factors` の行は Marian の既定の値が書き出されただけで、`type: transformer` では使われない（重みにも対応する項目が無い）。

### 2 の答え

| | en-ja 3.0 | ja-en 3.1 | ja-en 3.0 |
|---|---|---|---|
| エンコーダの層 | 6 | 6 | 6 |
| デコーダの層 | 4 | 4 | 2 |
| emb（幅） | 384 | 384 | 512 |
| ffn | 1536 | 1536 | 2048 |
| head | 8（head の幅 48） | 8（48） | 8（64） |
| デコーダの自己回帰 | SSRU（`transformer-decoder-autoreg: rnn`・`dec-cell: ssru`） | 同じ | 同じ |
| FFN の活性化 | ReLU、2 段 | 同じ | 同じ |
| norm の位置 | post（`dan` = 足してから LayerNorm）。埋め込みの後に norm は無い（`postprocess-emb: d` は dropout だけ）。最後の norm も無い（`postprocess-top: ""`） | 同じ | 同じ |
| LayerNorm | scale と bias あり、eps 1e-6（`src/layers/generic.h` の `layerNorm()`） | 同じ | 同じ |
| 語彙 | 原文 32000・訳文 32000（別の語彙） | 64000（1 本） | 64000 |
| 埋め込みの共有 | デコーダの埋め込みと出力の層を共有（`tied-embeddings`）。エンコーダは別 | エンコーダ・デコーダ・出力の全部で 1 本（`tied-embeddings-all`） | 同じ |
| 位置 | 正弦波（学習しない） | 同じ | 同じ |
| 出力の層 | logits = h · Wembᵀ + `decoder_ff_logit_out_b`（bias あり） | 同じ | 同じ |
| 注意の bias | q・k・v・o の全部にあり | 同じ | 同じ |

### 式（marian-dev の `src/models/transformer.h`・`src/rnn/cells.h`・`src/graph/node_initializers.cpp`・`src/tensors/cpu/tensor_operators.cpp` から）

- 埋め込み: x = √D · E[id] + P(pos)。P(p) の前の半分 i < D/2 が sin(p · exp(−i · ln(10000) / (D/2 − 1)))、後ろの半分が同じ角の cos（交互ではない）。
- エンコーダの層: x = LN(MHA(x, x) + x; `self_Wo_ln`)、x = LN(FFN(x) + x; `ffn_ffn_ln`)。
- MHA: Q = q Wq + bq、K = kv Wk + bk、V = kv Wv + bv、head ごとに softmax(Q Kᵀ / √(D/8)) V、つないで Wo + bo。
- FFN: ReLU(x W1 + b1) W2 + b2。
- デコーダのステップ t の入力: t = 0 は **0 のベクトル**（BOS の語は無い: `src/models/decoder.h` の `embeddingsFromPrediction()`）、t > 0 は前に出した語の埋め込み。位置は t。
- デコーダの層 l: SSRU、原文への注意、FFN の順。
  - SSRU: f = y Wf + bf、g = sigmoid(f)、c_t = g ⊙ c_{t−1} + (1 − g) ⊙ (y W)（W に bias は無い）、h = ReLU(c_t)。c は層ごとに 1 本の D の状態で、最初は 0。y = LN(h + y; `rnn_ffn_ln`)。
  - 原文への注意: y = LN(MHA(y, エンコーダの出) + y; `context_Wo_ln`)。K と V は原文 1 文につき 1 回作って使い回す。
  - y = LN(FFN(y) + y; `ffn_ffn_ln`)。
- デコーダは自己注意を持たないので、KV のキャッシュは原文の側だけ。答えのステップの状態は層ごとの c（D 個の float）だけ。

## 4. int8 の詰め方と係数

marian-dev の変換（`src/tensors/cpu/expression_graph_packable.h`）と読み込み（`src/common/binary.cpp`・`src/tensors/cpu/integer_common.h`）から。

### どれが int8 か

- intgemm8: 名前が `_W` の 2〜3 字で終わる行列（`_Wq`・`_Wk`・`_Wv`・`_Wo`・`_W1`・`_W2`・`_W`・`_Wf`）と、名前に `Wemb` を含むもの。
- float32: bias・LayerNorm・出力の bias・活性値の係数。

### 重み

- 係数は**行列ごとに 1 つ**: quantMult = 127 / max|W|（行列の全部の要素の最大）。
- int8 = W · quantMult を丸めたもの（3 本の全部の行列で範囲は −127〜127、−128 は無い。丸めの向きは intgemm の `PrepareA` で、その中は読んでいない）。
- 戻す式: **W ≈ int8 / quantMult**。
- 並び:
  - 埋め込み（`Wemb` を含む名前）: 形 [V, D] のまま行優先。並べ替えなし。
  - ほかの行列: 見出しの形は元の [in, out] だが、**データは転置した Wᵀ の行優先**（out 行 × in 列）。元に戻すのは `data.reshape(out, in).T`。（`integer_common.h` のコメント「we only transposed, but didn't update the shape」。）
- **ファイルにタイルの並びは無い**（「architecture agnostic」の形）。intgemm の「prepared B」の並びは、読み込みの時に `PrepareBQuantizedTransposed`（WASM では `int8PrepareBFromQuantizedTransposed`、ARM では写すだけ）が作る。だからタイルを元に戻す式は要らない。intgemm 本体（kpu/intgemm）は読んでいない。
- 読み込みのとき、`Wemb` を含む intgemm8 は float32 に戻される（`unquantizeWemb()`: 値 = int8 × (1 / quantMult)）。埋め込みは Marian の中では float32 で、メモリは 4 倍（en-ja で 2 × 49 MB、ja-en で 98 MB）。

### 活性値の係数（alphas）

- `<行列の名前>_QuantMultA` は float32 の 1 個で、**その行列に入る活性値の quantMult**（= 127 / その入力の最大の大きさ、学習データで前もって測った値）。行列ごとに 1 つ、トークンにも行にもよらない固定値。値は 4〜42 ほど（en-ja）。
- Firefox は名前が `alphas.bin` で終わるモデルに `gemm-precision: int8shiftAlphaAll` を渡す（Firefox の `translations-engine.worker.js`）: 係数をファイルから取り（`fetchAlphaFromModelNodeOp`: 名前 = 行列の名前 + `_QuantMultA`）、活性値を A_int8 = round(A · qa) に丸め（±127 で切る）、積は y = (A_int8 · W_int8) / (qa · qb) + bias。
- 「shift」は intgemm の実装の工夫: A に 127 を足して uint8 にし、bias から 127 · Σ_k W_int8[k, j] / (qa · qb) を引いて打ち消す（`PrepareBiasForBNodeOp` の `unquant_mult = −(127/qa)(127/qb)/127`）。数としては上の式と同じ。
- **出力の層の係数は intgemm8 で書いてある**（`decoder_Wemb_QuantMultA`・`Wemb_QuantMultA`: [1, 1]、1 バイト 0x7F と float の quantMult）。名前が `Wemb` を含むので、読み込みのとき float32 に戻され、値は 127 / quantMult になる: en-ja 127 / 19.1986 = 6.615、ja-en 3.1 127 / 18.6449 = 6.8115、ja-en 3.0 127 / 23.8281 = 5.3298。ja-en の `none_QuantMultA` は同じ値だった（6.8115・5.32984）。`none` は名前の付かない節（ショートリストで選んだ列）の名前と読んだが、どの道で引かれるかは確かめていない。
- 確かめていない: 学習の時に alphas をどう測ったか（最大か、百分位か）。

### pyodide-llm の int8 との違い（事実だけ）

pyodide-llm の int8 は行の中のグループごとに尺度を持ち、活性値も実行時にグループごとに丸める。このモデルは重みも活性値も**行列ごとに 1 つ**で、活性値の係数は固定。

## 5. ショートリスト

- Remote Settings には `lex.50.50.<pair>.s2t.bin` がある（上の表）。Firefox はそれを `TranslationModel` に渡す（`translations-engine.worker.js` の `lex ?? null`）。
- 形（marian-dev の `src/data/shortlist.h`・`shortlist.cpp`、`BinaryShortlistGenerator`）:

| 位置 | 中身 |
|---|---|
| 0 | uint64 magic `0xF11A48D5013417F5` |
| 8 | uint64 checksum（`util::hashMem`、firstNum から終わりまで: 確かめていない） |
| 16 | uint64 firstNum |
| 24 | uint64 bestNum |
| 32 | uint64 wordToOffset の長さ（原文の語彙 + 1） |
| 40 | uint64 shortLists の長さ |
| 48 | uint64 × wordToOffset |
| … | uint32 × shortLists（訳文の語の ID） |

| ファイル | firstNum | bestNum | 原文の語 | shortLists | 1 語あたり（平均 / 最大） |
|---|---|---|---|---|---|
| en-ja 3.0 | 50 | 50 | 32000 | 968,076 | 30.3 / 50 |
| ja-en 3.1 | 50 | 50 | 64000 | 2,209,029 | 34.5 / 50 |
| ja-en 3.0 | 50 | 50 | 64000 | 2,208,690 | 34.5 / 50 |

大きさは 3 つとも見出しから出す値とちょうど合った。

- 使い方（`generate()`）: 訳文の ID 0〜firstNum−1（よく出る 50 語）、原文の各語の shortLists[wordToOffset[s] .. wordToOffset[s+1]]、語彙を分け合う形（`shared_`）なら原文の語そのものも選ぶ。数を 8 の倍数にそろえ（firstNum から上の空きで埋める）、昇順に並べる。出力の層はその列だけを掛ける。
- ショートリストなしでも全語彙で出力できる（下の確かめは全語彙で訳した）。Firefox の `shared_` の値は確かめていない。

## 6. 語彙（sentencepiece）

`sentencepiece_model_pb2` で読んだ。3 本とも同じ形:

| | srcvocab.enja | trgvocab.enja | vocab.jaen |
|---|---|---|---|
| 語片 | 32000 | 32000 | 64000 |
| 型 | UNIGRAM | UNIGRAM | UNIGRAM |
| 学習の文 | data.src.txt（英） | data.trg.txt（日） | 両方 |

- 特殊: ID 0 `</s>`（CONTROL、eos）、ID 1 `<unk>`、ID 2〜6 USER_DEFINED `__source__`・`__target__`・`__done__`・`__start__`・`__end__`、ID 7〜262 バイトの語片 `<0x00>`〜`<0xFF>`（byte_fallback が真）。bos と pad は無い（−1）。
- 正規化: `nmt_nfkc`（precompiled_charsmap 237,561 バイト）、add_dummy_prefix 真、remove_extra_whitespaces 真、escape_whitespaces 真。split_digits は偽。
- en-ja の原文の語彙は日本語をほぼ持たない（「こんにちは」がバイトの語片になる）。訳文の語彙は英語の語を細かく割る。
- **pyodide-llm の読み手**（`convert/tokenizer.py` の `sentencepiece_pieces()`・`sentencepiece_charsmap()`・`sentencepiece_options()` で tokenizer.bin を作り、`engine/tokenizer.py` の `Tokenizer` で読む）: 3 本とも unigram・`collapse: True` と読み、英・日・全角・半角カナ・絵文字・タブ・余分な空白・`__source__` の文字を含む 13 文で sentencepiece 0.2.2 と同じ ID だった。全符号位置では比べていない。decode も比べていない。

## 7. 前処理と後処理

bergamot-translator の `src/translator/text_processor.cpp`・`parser.cpp` と Firefox の `translations-engine.worker.js` から。

- Firefox の設定: `beam-size 1`（greedy）、`normalize 1.0`、`word-penalty 0`、`max-length-break 128`、`mini-batch-words 1024`、`workspace 128`、`max-length-factor 2.0`、`skip-cost true`、`cpu-threads 0`、`gemm-precision int8shiftAlphaAll`、`alignment soft`。`ssplit-mode` は書かず、bergamot の既定は `paragraph`。
- 文の切り方: bergamot が ssplit（`ug::ssplit::SentenceSplitter`）で文に切る。Firefox は `ssplit-prefix-file` を渡していない（正規表現だけの切り方になる、とコードのコメント）。前後の空白は Firefox が外して戻す（`WHITESPACE_REGEX`）。
- 原文: sentencepiece で ID にし（`addEOS=false`）、bergamot が後ろに `</s>`（0）を足す。128 トークンを越える文は 127 トークンずつに切り、それぞれに `</s>`。BOS は無い。大文字小文字はそのまま（正規化は nmt_nfkc だけ）。
- 訳: デコーダの最初は 0 のベクトル（上の式）、`</s>`（0）で止まる。`max-length-factor 2.0` は訳の長さの上限を原文の長さの 2 倍にする設定（Marian の意味として読んだ: 長さの数え方は確かめていない）。訳は訳文の語彙で decode する（`</s>` は出さない）。

## 確かめ（読み方が正しいか）

`translate.py`: 重みを上の式で float32 に戻し、活性値は丸めず、全語彙・greedy で訳した。

| モデル | 原文 | 訳 |
|---|---|---|
| en-ja 3.0 | Hello, world. | こんにちは、世界。 |
| en-ja 3.0 | The weather is nice today, so let's go for a walk in the park. | 今日は天気が良さそうですね。公園を散歩しましょう。 |
| en-ja 3.0 | Firefox can translate web pages on your device. | Firefoxはデバイス上のウェブページを翻訳できます。 |
| ja-en 3.1 | こんにちは、世界。 | Hello, world. |
| ja-en 3.1 | 今日は天気がいいので、公園を散歩しましょう。 | The weather is fine today, so let's take a walk in the park. |
| ja-en 3.1 | 吾輩は猫である。名前はまだ無い。 | I'm a cat. I don't have a name yet. |
| ja-en 3.0 | 今日は天気がいいので、公園を散歩しましょう。 | The weather is nice today, so let's take a walk in the park. |
| ja-en 3.0 | 吾輩は猫である。名前はまだ無い。 | I am a cat, and I don't have a name. |

行列の転置・埋め込みの向き・SSRU・位置・norm のどれかを誤れば訳にならないので、読み方は合っていると見る。この道具は原文の K・V も使い回さない素の形なので、かかった秒は速さの目安にならない（未計測）。int8 の活性値の丸めを入れた形とは比べていない。

## 確かめられなかったこと

- intgemm の丸めの向き（`PrepareA` の中）と、学習の時の alphas の測り方。
- `none_QuantMultA` がどの道で読まれるか。en-ja に無い訳。
- ショートリストの checksum の式と、Firefox での `shared_`。
- bergamot の ssplit の切り方の細部（正規表現）と、Firefox がページの文をどの単位で渡すか。
- `max-length-factor` の長さの数え方。
- 語彙の全符号位置での一致と decode。
- ネイティブの fxtranslate の速さ（1 文 0.2 秒）と CAT-Translate の 4 秒は依頼の文の値で、ここでは測っていない。

## 出典

- marian-dev（browsermt の fork、MIT）: https://github.com/browsermt/marian-dev コミット `2781d735d4a10dca876d61be587afdab2726293c`。`src/common/binary.cpp`・`src/common/types.h`・`src/tensors/cpu/integer_common.h`・`src/tensors/cpu/intgemm_interface.h`・`src/tensors/cpu/expression_graph_packable.h`・`src/models/transformer.h`・`src/models/decoder.h`・`src/rnn/cells.h`・`src/layers/generic.h`・`src/layers/generic.cpp`・`src/graph/node_initializers.cpp`・`src/tensors/cpu/tensor_operators.cpp`・`src/data/shortlist.h`・`src/data/shortlist.cpp`。形を読んだだけで、行は写していない。
- bergamot-translator: https://github.com/browsermt/bergamot-translator コミット `9271618ebbdc5d21ac4dc4df9e72beb7ce644774`。`src/translator/text_processor.cpp`・`src/translator/parser.cpp`。
- Firefox: https://github.com/mozilla-firefox/firefox コミット `515c39efb40b54bf877184846dbd07df66c806c0` の `toolkit/components/translations/content/translations-engine.worker.js`。
- Remote Settings: https://firefox.settings.services.mozilla.com/v1/buckets/main/collections/translations-models-v2/records（2026-11-05 に読んだ。項目 404 件）。
- intgemm（https://github.com/kpu/intgemm）は読んでいない（ファイルにタイルの並びが無いので要らなかった）。
