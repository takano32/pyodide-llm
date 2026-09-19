# TODO

このプロジェクトの作業台帳。背景・方針・落とし穴・検証手順は [AGENTS.md](AGENTS.md) にあるので、先にそちらを読むこと。

- 着手したら状態を `進行中` にする。終えたら「完了したタスク」の末尾に `- [x]` で移し、結果の要点（実測値・理由）とコミットを 1〜2 行で書く。詳しい知見は AGENTS.md に足す。**作業と同じコミットで更新する。**
- タスク番号は通し番号で、再利用しない。
- 未着手のタスクは優先度順。各タスクは単独で着手できる。

## これからのタスク

### T24 停止ボタン — 状態: 未着手
- 目的: 生成を途中で止められるようにする（llm-jp は 256 トークンに 30 秒かかる）。
- 触るファイル: `public/worker.js`（`generate()` と `onmessage`）、`src/pages/index.astro`（送信ボタン、`onsubmit`）。
- 手順:
  1. `worker.js` の `generate()` を `async` にする。ジェネレータを `for...of` で回す代わりに `pieces.next()` を呼び、8 トークンごとにイベントループへ制御を返す。`setTimeout` は 4ms の下限で遅くなるので使わない。`MessageChannel` で返す: `await new Promise((r) => { const c = new MessageChannel(); c.port1.onmessage = r; c.port2.postMessage(0); })`。
  2. `onmessage` で `{type: "stop"}` を受けたら停止フラグを立てる。`generate()` はフラグを見てループを抜け、`finally` で `pieces.destroy()`、その後いつもどおり `done` を送る（`llama.stats` は途中終了でも入る）。
  3. ページ側は生成中だけ送信ボタンを停止ボタン（■）にして、押したら `worker.postMessage({type: "stop"})`。生成中も Enter で二重送信されないこと。
- 完了条件: 生成中に止められ、直後に次の生成ができる。tiny-lm の tok/s が変更前の 95% 以上。`node tests/e2e.mjs` が通る。

### T27 生成設定の UI — 状態: 未着手
- 目的: temperature・最大トークン数・シードを画面から変える。シードを固定すれば int8 と原本を公平に比べられる。
- 触るファイル: `src/pages/index.astro`。エンジンは `generate(prompt, steps, temperature, topp, repetition_penalty, seed)` に対応済みで、Worker は受け取ったオプションをそのまま渡す。
- 手順: 入力欄の上に `<details>` で設定を置く。初期値は `model.generation`。モデルを切り替えたら初期値に戻す。シードは空欄なら毎回ランダム。
- 完了条件: 同じモデル・同じシードで 2 回生成すると同じ文になる。設定を開かなければ今と同じ動作。

### T28 計測パネル — 状態: 未着手
- 目的: 実験としての数値を見せる。
- 内容: 起動時間の内訳（Pyodide のロード / モデルのダウンロード / `Llama()` の構築）、最初のトークンまでの時間、プロンプト処理と生成それぞれの tok/s。`worker.js` で `performance.now()` を取り、`ready` と `done` のメッセージに載せる。エンジンの `stats` にプロンプト処理の時間を足す。
- 完了条件: 各値が回答の下か折りたたみに出る。tok/s が計測のせいで落ちない。

### T29 エンジンの単体テスト — 状態: 未着手
- 目的: 手元でやった検証を資産にする。`tests/` に pytest（ネイティブの Python + NumPy で動く。Pyodide は不要）。
- 内容: (1) BPE と unigram のエンコード → デコード往復（日本語、絵文字、語彙外文字、空白・タブ・改行）。(2) `quantize.py` → `dtype="int8"` で読んだ重みと元の重みの誤差がグループの最大値の 1/127 以内。(3) 小さな合成チェックポイント（dim 32、2 層、GQA あり / なし）で `forward` の logits を、素朴なループ実装と相対誤差 1e-4 以内で比較。(4) `generate` は同じシードで再現し、BOS で止まり、長すぎるプロンプトは `ValueError`。
- モデルのダウンロードが要るテストは `make models` 済みのときだけ走るように分ける。

### T31 README に計測結果を載せる — 状態: 未着手
- gist の要点（実装別 tok/s の表、int8 の品質、ブラウザ別の速度）を README に入れる。数値は AGENTS.md と `TODO.md` の完了タスクにあるものだけを使い、新しく推測しない。

## 完了したタスク

古い順。括弧内は対応するコミット。

- [x] **T1 遅さの原因を実測する。** 純 Python 版は 0.26 tok/s。原因は行列積の内側ループをインタプリタが回していること。同じ機械で NumPy 54、素の JS 38、WASM SIMD 170〜190 tok/s。スレッド化は効かない（ネイティブ OpenMP でも同じ）。
- [x] **T2 類似の試みとブラウザ基盤を調査し、レポートを公開する。** llama2.c のブラウザ移植、wllama、WebLLM、Transformers.js、WebGL、モバイルの制約など。gist: https://gist.github.com/takano32/196c6f93979ad44f98cee5712fdd3901
- [x] **T3 WASM Python の土台を決める。** Pyodide を継続。MicroPython の WASM 版は起動 20〜50ms だが計算は 1/3.5（ulab の行列積は 1/11）。Pyodide 内で ctypes 読み込みの SIMD カーネルを使うと 181〜348 tok/s まで出ることを実証（未導入 → T30）。
- [x] **T4 GitHub Pages で成立するか実ブラウザで確かめる。** 特別な HTTP ヘッダなしの静的配信で、wheel のインストールから推論まで Chromium と Firefox で動作。COOP/COEP は不要。
- [x] **T5 http-server の `--cors` の typo を直す。**（755222a）
- [x] **T6 生成物を `.gitignore` に入れる。** バイナリはコミットしない方針。（5b51e0b）
- [x] **T7 GitHub Actions を最新版にする。** checkout v7、configure-pages v6、upload-pages-artifact v5、deploy-pages v5。（210963c）
- [x] **T8 `tokenizer.bin` を llama2.c 本家のものにする。** 旧ファイルは語彙が 204 個重複し、句読点や大文字が未学習の ID になっていた。（54fda6c）
- [x] **T9 Pyodide を常に最新版で読み込む。** 実行時に jsDelivr の API で解決。`?pyodide=<version>` で強制可。固定バージョンやフォールバックは置かない。（4f248fe）
- [x] **T10 NumPy 版のエンジンを書く。** `llama2_numpy.py`。llama2.c の C 実装と 256 トークンまでバイト単位一致、約 50 tok/s。BOS で停止、プロンプト中は分類器を省く、ダミープレフィックス（無いとパープレキシティ +8.5%）。（bdac468）
- [x] **T11 Hugging Face 形式の変換ツールを書く。** `convert_hf.py`、PyTorch 不要。bfloat16、Q/K の並べ替え（あり 4.45 / なし 5.04 の NLL で検証）。（513442c、f1ad824）
- [x] **T12 全モデルを `make models` で取得・変換する。** ローカル・Docker・デプロイで共通。（ff255b9）
- [x] **T13 Web Worker 化、逐次表示、モデル選択。** 日本語モデル tiny-lm を追加（unigram トークナイザは sentencepiece と 16 例一致）。greedy だと反復するのでサンプリング + 繰り返しペナルティ。`<meta charset>` 欠落による文字化けも修正。（a6ea457、6b654fb）
- [x] **T14 ダウンロードの進捗を表示する。** GitHub Pages が `.bin` を gzip で送るため Content-Length が使えず、モデル一覧に展開後の `bytes` を持たせた。（7ab8448、6f5ebb3）
- [x] **T15 入力欄を最下部に固定し、出力だけがスクロールするようにする。**（59f3053、6d38f09）
- [x] **T16 大きいモデルを int8 で配布し、原本も選べるようにする。** `quantize.py`（グループ 32）。品質は原本と区別できず（stories15M +0.04%、tiny-lm 91.3 → 91.1、llm-jp 22.76 → 22.69）、int4 は +16.8% で不可。（ef4c7d2、1b24d49、c36992e）
- [x] **T17 質の高い日本語モデル llm-jp-3-150m を追加する。** パープレキシティ 22.8（tiny-lm は 91）、約 8.5 tok/s、メモリ約 500MB。文脈長を 512 に切り、埋め込み表は int8 のまま保持。（355e93f、0b63548）
- [x] **T18 モデルを 8 MiB の部品に分けて 4 並列で取得し、Pyodide のロードと同時に進める。** 本番 CDN で約 1.8 倍速。ブラウザ側の Range 分割は gzip のため不可。（0b63548）
- [x] **T19 デプロイ直後に新旧のコードが混ざらないようにする。** Worker の URL にコミットのハッシュを付与。（b620c40）
- [x] **T20 既定モデルを日本語でいちばん軽い tiny-lm（int8）にする。** デモは速さ優先。既定プロンプトは「昔々、」。（4f8fd6a、c2c6e57）
- [x] **T21 ページを Astro でチャット風に作り直す。** Node 24 LTS、ライト / ダーク自動、GitHub へのリボンとコーナーをランダム表示。（9d09e80〜2f20fec）
- [x] **T22 プロジェクトを pyodide-llama-py に改名する。** Pages の旧 URL は転送されない。（adbd2c1）
- [x] **T23 引き継ぎ書と台帳を作る。** `AGENTS.md`、`TODO.md`。（268a141）
- [x] **T26 CI のスモークテスト。** `tests/smoke.mjs`: Node 上の最新 Pyodide で、stories260K（float32・GQA）の greedy 出力が参照どおりであることと、tiny-lm（変換 + int8 + unigram）が生成できシードで再現することを確認。約 8 秒、メモリ約 450MB。エンジンを壊すと失敗することを確認済み。デプロイでは `pyodide@latest` を入れ直してから走らせる（ページが実行時に最新版を使うため）。
- [x] **T25 モデルのブラウザ内キャッシュ。** 部品を Cache API（`models-v1`）に保存し、キーに展開後のバイト数を含める。2 回目の読み込みはモデル部品のネットワーク要求が 0 件、準備完了が 11.3 秒 → 7.1 秒（tiny-lm、ローカル）。サイズが変わった古い部品は読み込み後に削除。`navigator.storage.persist()` はページ側から要求（Worker からは呼べない。ヘッドレス Chromium では許可されず false のまま）。
- [x] **T30 SIMD カーネルの導入。** `kernels/*.ts` を `make kernels` でビルドし、`llama2_numpy.py` が ctypes で読み込む（失敗時・GQA・32 の倍数でない int8 は NumPy にフォールバック、`?kernel=off` で NumPy を強制）。float32 は NumPy と同じ出力で 53 → 200 tok/s、int8 は重みを int8 のまま計算して stories15M 351、tiny-lm 422 tok/s（Node 上の Pyodide、greedy）。llm-jp-3-150m は 9.3 → 81 tok/s、WASM ヒープ 897MB → 283MB。Chromium では tiny-lm 43 → 149、llm-jp 8.5 → 47、stories15M 50 → 296 tok/s。Firefox でも動作、Safari は未確認。スモークテストが両経路を確認する。
- [x] **T32 サンプリングの高速化。** `exp` の前に、最有力の 1000 万分の 1 未満のトークンを落とし、llama2.c と同じ厳密な足切り（(1 − top-p)/(n − 1) 未満は nucleus に入らない）で並べ替えの対象を絞り、抽選は累積和と乱数 1 個にした。`sample()` は 2.2 → 0.84 ms（tiny-lm、語彙 51200）。Chromium の tiny-lm: サンプリングあり 263 tok/s（greedy は 318）、繰り返しペナルティ付きは 171 tok/s で目標の 250 には未達 → T34。nucleus の厳密さ・頻度・シードの再現性は検証済み。
- [x] **T33 カーネルの GQA 対応。** `attention` が n_kv_heads と 4 の倍数でないヘッド長（stories3_5M は 26）に対応し、KV キャッシュは `[seq][kv_dim]`。これで全モデルがカーネルで動く。NumPy 版と同じ greedy 出力をスモークテストで確認。Chromium で stories3_5M 141 → 402 tok/s、stories260K 268 → 951 tok/s。
- [x] **T34 サンプリングをカーネルに移す。** 繰り返しペナルティ（`penalize`）と、softmax・足切り・top-p の選択・抽選（`sample`）を `kernels/kernel.ts` に移した。`exp` は SIMD で 4 個ずつ、並べ替えは自前のクイックソートで **nucleus が埋まるところまでしか進めない**（候補は平均 2000・最大 8000 個あるが、必要なのはたいてい数十個）。乱数は Python の `rng.random()` を 1 個渡すので、シードの再現性はそのまま。直近 64 トークンは NumPy 配列をリングとして持ち、1 ステップに 1 個だけ書く。NumPy 版の `Llama.penalize()` / `Llama.sample()` はフォールバックとして残した。ペナルティ + 抽選は 0.92 → 0.29 ms（tiny-lm、Node 上の Pyodide）。Chromium の tiny-lm（既定の生成設定）: 171 → 252〜274 tok/s（3 回計測）、llm-jp-3-150m は 67 tok/s。検証: 40 種の分布で nucleus が厳密、20 万回の抽選の頻度が 5σ 以内、同じ乱数で NumPy 版と 300/300 で同じトークン、ペナルティは NumPy 版と一致、同じシードで同じ文章。要点は `tests/smoke.mjs` に入れた。はまった点: カーネルに渡すだけの作業用配列は Python から参照が消えて解放され、たまに `memory access out of bounds` で落ちた → `self._sampler_buffers` で保持。

## やらないと決めたこと

- JS + WASM 版のエンジン（別プロジェクト）。
- WebGL / WebGPU（このモデル規模では CPU SIMD に勝てない。モデルを 0.5B 以上にするなら再検討）。
- マルチスレッド（効果がなく、GitHub Pages では COOP/COEP が要る）。
- Pyodide のバージョン固定やフォールバックの連鎖。
- 「MobileLLM」への改名（Meta の既存モデル名と衝突）。
