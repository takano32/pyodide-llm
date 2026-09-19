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

### T25 モデルのブラウザ内キャッシュ — 状態: 未着手
- 目的: 再訪時に 33〜171MB を取り直さない（GitHub Pages の HTTP キャッシュは 10 分で切れる）。
- 触るファイル: `public/worker.js` の `download()`。
- 手順:
  1. 部品を取る前に `caches.open("models-v1")` を見て、あればそこから読む。なければ `fetch` し、`response.clone()` を `cache.put()` する（ストリームは一度しか読めないので clone が要る）。
  2. キャッシュのキーは部品の URL に `?bytes=<model.bytes>` を付けたものにする。モデルを作り直してサイズが変われば自然に取り直される。
  3. 読み込み成功後に `navigator.storage.persist?.()` を呼ぶ（失敗しても無視）。
  4. Cache API が使えない環境（`caches` が未定義）では今までどおり動くこと。
- 完了条件: 2 回目の読み込みで `models/` へのネットワーク要求が 0 件（Playwright の `page.on("request")` で数える）。`src/models.js` の `bytes` を変えると取り直す。

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

### T30 SIMD カーネルの導入（本丸） — 状態: 未着手（担当: Fable）
- 目的: 「WASM Python の限界」を本番に入れる。実証値は 181（float32）/ 282（int8）/ 348（relaxed SIMD int8）tok/s で NumPy の 4〜7 倍。int8 のまま計算するので llm-jp のメモリも減る。
- 出発点: `experiments/simd-kernel/`（動作確認済みのカーネル、ビルドスクリプト、Python からの呼び方、守るべき制約）。**先にその README を読むこと。**
- 手順の骨子:
  1. カーネルのビルドを `make models` の流れに入れる（`assemblyscript` を devDependencies に。生成物 `public/simdkernel.so` などはコミットしない）。
  2. `llama2_numpy.py` にカーネル用の forward を足す。`ctypes.CDLL` の読み込みに失敗したら NumPy の forward のまま動く（Safari や将来の Emscripten 変更への備え。方針 2 と両立させる）。
  3. `attention` を GQA と `[kv_heads][seq][head_size]` のキャッシュ配置に対応させるか、カーネル使用時だけキャッシュ配置を変える。GQA のモデル（stories260K、3.5M）と行長が 32 の倍数でないモデルは NumPy のままでもよい。
  4. int8 のモデルは重みを float32 に戻さず、`quantize.py` の形式のまま `matmul_q8` に渡す。relaxed SIMD 版は `try/except` で選ぶ。
  5. `worker.js` がカーネルのファイルを Pyodide の FS に書く（`?v=` を付けること。AGENTS.md の落とし穴を参照）。
- 完了条件: float32 のモデルで NumPy 版と同じ greedy 出力。int8 は破綻しない出力。Chromium で tiny-lm と stories15M が NumPy 版の 3 倍以上。カーネルの読み込みをわざと失敗させても NumPy で動く。T26 のスモークテストと `tests/e2e.mjs` が通る。iOS Safari は確認手段がなければ「未確認」と AGENTS.md に書く。

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

## やらないと決めたこと

- JS + WASM 版のエンジン（別プロジェクト）。
- WebGL / WebGPU（このモデル規模では CPU SIMD に勝てない。モデルを 0.5B 以上にするなら再検討）。
- マルチスレッド（効果がなく、GitHub Pages では COOP/COEP が要る）。
- Pyodide のバージョン固定やフォールバックの連鎖。
- 「MobileLLM」への改名（Meta の既存モデル名と衝突）。
