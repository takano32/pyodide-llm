# TODO

このプロジェクトの作業台帳。背景・方針・落とし穴・検証手順は [AGENTS.md](AGENTS.md) にあるので、先にそちらを読むこと。

- 着手したら状態を `進行中` にする。終えたら「完了したタスク」の末尾に `- [x]` で移し、結果の要点（実測値・理由）とコミットを 1〜2 行で書く。詳しい知見は AGENTS.md に足す。**作業と同じコミットで更新する。**
- タスク番号は通し番号で、再利用しない。
- 未着手のタスクは優先度順。各タスクは単独で着手できる。

## これからのタスク

### T35 生成設定の変更 — 状態: 未着手
- 目的: temperature・最大トークン数・シードを画面から変える。T27 から分けた「変える」側。
- 決めたこと: ハンバーガーメニューにはしない（ナビゲーションの記号で、右上は GitHub のリボンが使っている）。入力欄の上に `<details>` を置く案もやめた（ページがスクロールしない設計なので、スマホでキーボードが出ていると会話の領域を潰す）。
- 手順:
  1. 入力欄の左端にスライダー（または歯車）のアイコンのボタンを 1 つ置き、PC ではポップオーバー、スマホ幅では下から出るシートで設定を開く。Popover API か `<dialog>` を使い、ライブラリは足さない。
  2. 初期値は `model.generation`。モデルを切り替えたら初期値に戻す。シードは空欄なら毎回ランダム（T27 の仕組みのまま）。既定値から変えている間はアイコンに点を付ける。
  3. 回答の下に出ているシード（T27）をタップすると、そのシードを次の生成に固定する。「回答のシードをタップ → モデルを切り替え → 送信」で int8 と原本を比べられること。
- 完了条件: 同じモデル・同じシードで 2 回生成すると同じ文になる。設定を開かなければ今と同じ動作。常に見える要素の増加はアイコン 1 つだけ。

### T39 URL で指定した llama2.c 形式のモデルを使う — 状態: 未着手
- 目的: サイトに置いていないモデル（例: Hugging Face の `karpathy/tinyllamas` にある `stories110M.bin`）を URL で指定して動かす。「このリンクを開けばこのモデルが動く」と共有できる。
- 決めたこと: **UI は足さない。クエリパラメータだけ**（`?checkpoint=<URL>&tokenizer=<URL>`、任意で `&config=<設定の JSON の URL>`。いまの `?model=`・`?kernel=off`・`?pyodide=` と同じ流儀）。T38（ローカル、ネットワークに何も出さない）とは論点が違うので分けた。承認が必要な（gated な）モデルには対応しない（トークンを入力させる UI は作らない）。
- 手順:
  1. T38 の部品（`dtype` の自動判定、設定の JSON、読み込み元の差し替え、T37 の中止）の上に、読み込み元として URL を足す。コンボボックスには T38 と同じく一時的な項目（ファイル名 + `(URL)`）を出す。
  2. サイズは `HEAD`（または `Range: bytes=0-0` の `Content-Range`）で先に調べる。Hugging Face は CORS を許可し、Range に正しく応える（GitHub Pages と違って gzip の断片にならない）ので、8 MiB の Range を 8 並列で取れる。Range に応えないサーバーは 1 本のストリームで取る。実際に HF の URL で CORS と Range を確かめてから設計を固めること（未確認）。
  3. Cache API に入れるかは、サイズと `navigator.storage.estimate()` を見て決める（入れるならキーに URL とサイズ）。1GB を超えるものは T38 と同じ警告。
  4. 失敗（CORS で拒否、404、形式が違う）は分かりやすいエラーにして、コンボボックスから配布モデルへ戻れること。
- 完了条件: `stories110M.bin` か、それが重すぎるなら `stories42M.bin` を HF の URL から読んで、greedy の出力が llama2.c と同じに始まる。URL が壊れているときにエラーが出て復帰できる。読み込み中にモデルを選び直すと中止される。

### T40 Hugging Face 形式をブラウザの中で変換する — 状態: 未着手（難しいので Fable 向き）
- 目的: safetensors + `tokenizer.json` + `config.json` を、ローカルのファイル（T38 の入口）または HF のリポジトリ（T39 の入口）から読み、ブラウザの中で変換して動かす。`convert_hf.py` は NumPy だけで書いてあるので Pyodide で動くはず。「変換も WASM Python でやる」という実験。
- いちばんの壁はメモリ: 素朴にやると safetensors の原本、float32 への展開、量子化した結果が同時にヒープに載る（llm-jp-3-150m で 300MB + 600MB + 170MB。WebAssembly のメモリは 32 ビットで縮まない）。テンソルを 1 つずつ読み（safetensors はヘッダにオフセットがあるので `File.slice()` や Range で部分的に読める）、その場で int8（グループ 32）にして最終的なバッファへ書き、原本は手放す、というストリーム処理に書き直す必要がある。`convert_hf.py` と `quantize.py` のコードは共有し、ビルド時の変換結果とバイト単位で一致させる。
- ほかに要るもの: `config.json` の検証（Llama 系だけ受け付ける。`rope_theta`、GQA、語彙の大きさ、`tie_word_embeddings`）、トークナイザの種類の判定（unigram / BPE、NFKC）、対応外のモデルへの分かりやすいエラー、進捗の表示。
- 完了条件: `sbintuitions/tiny-lm` をブラウザの中で変換したものが、`make models` の `tiny-lm.bin` とバイト単位で一致する。変換中のヒープの最大値を測って記録する。llm-jp-3-150m はこの開発機のメモリでは試せない可能性が高いので、試せなければ「未確認」と書く。

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
- [x] **T34 サンプリングをカーネルに移す。** 繰り返しペナルティ（`penalize`）と、softmax・足切り・top-p の選択・抽選（`sample`）を `kernels/kernel.ts` に移した。`exp` は SIMD で 4 個ずつ、並べ替えは自前のクイックソートで **nucleus が埋まるところまでしか進めない**（候補は平均 2000・最大 8000 個あるが、必要なのはたいてい数十個）。乱数は Python の `rng.random()` を 1 個渡すので、シードの再現性はそのまま。直近 64 トークンは NumPy 配列をリングとして持ち、1 ステップに 1 個だけ書く。NumPy 版の `Llama.penalize()` / `Llama.sample()` はフォールバックとして残した。ペナルティ + 抽選は 0.92 → 0.29 ms（tiny-lm、Node 上の Pyodide）。Chromium の tiny-lm（既定の生成設定）: 171 → 252〜274 tok/s（3 回計測）、llm-jp-3-150m は 61〜66 tok/s（256 トークンを 4 回。最初に書いた 67 は 61 トークンで終わった 1 回だけの値だったので測り直した）。検証: 40 種の分布で nucleus が厳密、20 万回の抽選の頻度が 5σ 以内、同じ乱数で NumPy 版と 300/300 で同じトークン、ペナルティは NumPy 版と一致、同じシードで同じ文章。要点は `tests/smoke.mjs` に入れた。はまった点: カーネルに渡すだけの作業用配列は Python から参照が消えて解放され、たまに `memory access out of bounds` で落ちた → `self._sampler_buffers` で保持。
- [x] **T24 停止ボタン。**（担当: Opus、レビューと仕上げ: Fable）送信ボタンが生成中だけ停止ボタン（■）になる。要素は増やしていない。Worker の `generate()` を `async` にして `pieces.next()` で回し、イベントループに返して `{type: "stop"}` を受け取る。生成の前後に届いた stop は次の生成を止めない。生成中の `load` は先に止めて待つ。返す間隔は最初「8 トークンごと」だったが、stories260K（900 tok/s）だけ約 5% 遅くなったので **50ms ごと**に変えた（903〜941 tok/s で変更前と同じ、tiny-lm 256〜268）。実ブラウザで確認: 押して 0.13〜0.19 秒で止まり、stats は途中終了でも正しく、直後の生成は最後まで走り、生成中の Enter は二重送信も停止もしない。未確認: Firefox。はまった点（Enter が送信ボタンの click になる）は AGENTS.md。
- [x] **T29 エンジンの単体テスト。**（担当: Opus）`tests/test_*.py` と `conftest.py`。78 件が 7.5 秒、モデルのファイルが無ければ 54 件 + スキップ 24 件。合成チェックポイント 5 構成（MHA、GQA、MQA、分類器が別、head_size 4）で `forward` を Python のループの参照実装と相対誤差 1e-4 以内で比較、`quantize.py` の誤差がグループ最大値の 1/127 以内、BPE / unigram の往復（日本語、絵文字、語彙外、空白類）、`sample` の nucleus が厳密、`generate` のシード再現・停止・`ValueError`。エンジンのバグは見つからず。デプロイのワークフローでも走らせる。カーネルの経路は対象外（`smoke.mjs`）。
- [x] **T31 README に計測結果を載せる。**（担当: Opus）`## Measurements` に実装別 tok/s、Chromium でのモデル別（NumPy / カーネル）、メモリ、int8 の品質、分割ダウンロードの表。文書にある数値だけを使い、Safari は未計測と明記。その過程で文書間の食い違いを直した: llm-jp-3-150m は測り直して 61〜66 tok/s、tiny-lm は llm-jp の「3 倍」ではなく 4 倍、分割ダウンロードは「約 2 倍」ではなく 1.8 倍。
- [x] **T27 生成設定の表示 / T28 計測の表示。**（担当: Opus、レビューと仕上げ: Fable）回答の下の行が `tiny-lm 29M · 90 tokens · 255.0 tok/s · temp 0.7 · seed 1234567890`（greedy のモデルは `… · greedy`）になり、タップすると内訳（最初のトークンまでの時間、プロンプトの tok/s、生成の tok/s と全体の秒数）が開く。ステータス行も準備完了後は開けて、Pyodide のロード / ダウンロード（同時進行なので足さずに並べて表示）/ `Llama()` の構築を出す。開けることが伝わるように行末に ▸ を付けた。シードはサンプリングするモデルのときだけページが `crypto.getRandomValues` で引いて渡す。エンジンの `stats` に `sampled`・`prompt_tokens`・`prompt_seconds`・`prompt_tokens_per_second`・`first_token_seconds` を追加（既存のキーは同じ意味のまま）。実ブラウザで確認: シードを固定した 2 回は同じ文、別のシードなら別の文、390px でページはスクロールしない、途中で止めた回答の行も正しい。速度は変わらず（tiny-lm 250〜274、stories260K 918〜965 tok/s）。実測の例（tiny-lm、キャッシュなし）: Pyodide 8.07 秒・ダウンロード 8.09 秒（同時）・`Llama()` 0.25 秒・最初のトークン 0.03 秒。未確認: Firefox、ライトテーマの見た目。
- [x] **T36 Enter は改行、Ctrl / Cmd + Enter で送信。**（Fable）入力欄を `<textarea>` にして、行数に合わせて 5 行まで伸び、それ以上は欄の中でスクロールする（ページはスクロールしない）。日本語の変換確定の Enter で送信されることがなくなった。スマホは送信ボタンで送る。`tests/e2e.mjs` は Ctrl + Enter で送信する。実ブラウザで確認: Enter と Shift + Enter は改行、Ctrl + Enter と Meta + Enter は送信、生成中の Ctrl + Enter は何もしない、停止ボタンはそのまま動く。
- [x] **T37 ダウンロード中のモデル切り替え。**（Fable）モデルの選択はロード中も有効（無効なのは生成中だけ）。選び直すと Worker が実行中のロードを `AbortController` で中止し、新しいモデルを取りにいく。Pyodide のロードはやり直さない。ロードには番号を振り、Worker の報告（status / progress / ready / error）に付けて、ページは古い番号の報告を捨てる。届き終わっていた部品は Cache API に残るので、元のモデルに戻すと続きから（実測: 21 部品中 6 個が済んでいて、再取得は 15 個）。回線を遅くした実ブラウザで確認: Pyodide のロード中の切り替え、バッファ確保後の切り替え、4 連続の切り替え（最後の 1 つだけが準備完了になる）、切り替え後の生成の出力、エラーなし。メモリ: 中止したロードの後始末を待たずに次のバッファを確保するとヒープが 587MB まで膨らんだ → 後始末を待ってから確保するようにし、モデルを手放すときに `gc.collect()` も呼ぶようにした（同じ操作の後で 266MB。llm-jp を普通に読んだときの約 283MB と同じ水準）。ただし一度 llm-jp のバッファ（171MB）を確保すると、中止してもヒープは 210MB から縮まない（WebAssembly のメモリは縮まない）。
- [x] **T38 ローカルのモデルファイルを使う。**（Fable）モデル選択の右のフォルダのボタン（隠した `<input type="file" multiple>` の `<label>`）か、ページへのドラッグ & ドロップで、チェックポイント + `tokenizer.bin`（+ 任意で設定の `.json`。`src/models.js` の 1 項目と同じ形）を選ぶ。`.json` 以外の大きいほうがチェックポイント。ファイルは `File` のまま Worker に渡し、`file.stream()` でダウンロードと同じ Python バッファへ読む（アップロードなし、Cache API にも入れない）。`dtype` はエンジンの `checkpoint_dtype()` がヘッダとファイルの大きさから判定し、`check_tokenizer()` が語彙の大きさの合わない `tokenizer.bin` を断る（エンジンは大きい語彙の先頭だけを黙って読んでしまうため）。どちらも大きなファイルを読む前に走り、トレースバックなしの文で出る。読んだモデルはコンボボックスに `名前 — local · 33 MB` の一時的な項目として出て、配布モデルとの行き来はダイアログなしでできる。1GB を超えるファイルは `confirm()` で確認する。実ブラウザで確認（Chromium。`node tests/e2e.mjs local` は Firefox でも）: float32（stories260K、期待どおりの出力）、int8 + 設定の JSON（tiny-lm、unigram・NFKC・サンプリング）、ドロップした int8（stories15M）、読み込み中にモデルの取得リクエストが 0 件、トークナイザ 2 つ / 語彙の合わない組 / 1 ファイルだけ、のそれぞれでエラーが出て配布モデルへ戻れる、390px でスクロールしない。手元の全モデル（float32 / float16 / int8 の 12 ファイル）で判定が正しく、トークナイザ 5 つは「チェックポイントではない」と断られる。未確認: 1GB 超のファイル（この機械では試せない）、読み込み中の切り替えによる中止（経路は T37 と同じだが、ローカルの読み込みは一瞬で終わるので試せていない）、スマホ実機のファイル選択。

## やらないと決めたこと

- JS + WASM 版のエンジン（別プロジェクト）。
- WebGL / WebGPU（このモデル規模では CPU SIMD に勝てない。モデルを 0.5B 以上にするなら再検討）。
- マルチスレッド（効果がなく、GitHub Pages では COOP/COEP が要る）。
- Pyodide のバージョン固定やフォールバックの連鎖。
- 「MobileLLM」への改名（Meta の既存モデル名と衝突）。
