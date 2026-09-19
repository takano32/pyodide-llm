# AGENTS

このリポジトリで作業する人と AI エージェントへの引き継ぎ書。会話の文脈なしで読めるように書いてある。
これからやる作業は [TODO.md](TODO.md) にある。

## 作業の決まり

- 着手前にこのファイルの「方針」「落とし穴」「検証手順」を読む。
- 作業で新しく分かった決定・実測値・落とし穴は、**同じコミットでこのファイルに足す**。タスクの状態は TODO.md で更新する（完了したタスクも消さずに残す）。
- git の履歴で分かること（どのファイルをどう変えたか）は書かない。書くのは、決定とその理由、実測値、落とし穴。
- 推測で数値を書かない。測っていないものは「未計測」と書く。
- コミットと push は持ち主に頼まれたときだけ行う。

## プロジェクトの目的と方針

ブラウザ上の WebAssembly 版 Python（Pyodide）だけで言語モデルを動かす**実験**。実用のエンジンではなく、「WASM Python でどこまでできるか」を測って公開することに価値を置く。JavaScript + WASM 版は別プロジェクトで作る（ここではやらない）。

決定済みの方針（変えるときは持ち主に確認する）:

1. **バイナリをリポジトリに入れない。** モデルのダウンロード・変換・量子化・分割はすべてビルド時（`make models`）に行う。
2. **Pyodide は常に最新版**を実行時に解決して読み込む（`public/worker.js`）。手で上げるバージョン定数は置かない。CDN が落ちていたら動かなくてよい（信頼性のための仕組みは足さない）。
3. **Node は 24 LTS**（`.nvmrc`）。ページは Astro、チャット風 UI。
4. **既定モデルは日本語でいちばん軽い tiny-lm（int8）。** デモは速さ優先。質の高い llm-jp-3-150m は選択式。
5. 大きいモデルは **int8 で配布**し、量子化前の原本も選べるようにする。
6. GitHub Pages（静的ホスティング、HTTP ヘッダ変更不可）で動くこと。スレッド（SharedArrayBuffer）には頼らない。
7. コミットは関心ごとに分け、英語の命令形の件名。master に push するとデプロイされる。

## 現在の構成

| ファイル | 役割 |
|---|---|
| `public/llama2_numpy.py` | NumPy 版の推論エンジン。llama2.c の legacy 形式（7 個の int ヘッダ + テンソル）を読む。float32 / float16 / int8。トークナイザは BPE（llama2.c 方式）と unigram（Viterbi）。`generate()` はテキスト片を返すジェネレータ |
| `public/worker.js` | Web Worker。最新 Pyodide の解決、モデル部品の並列ダウンロード（8 MiB × 4 並列、Pyodide のロードと同時進行）、Python バッファへの直接書き込み、トークンの逐次送信 |
| `src/pages/index.astro` | チャット風のページ。Worker の報告を描画するだけ |
| `src/models.js` | モデル一覧（ファイル名、バイト数、エンジンのオプション、生成設定、既定プロンプト） |
| `convert_hf.py` | Hugging Face の Llama チェックポイント → legacy 形式 + tokenizer.bin。PyTorch 不要（NumPy のみ）。bfloat16、safetensors、`tokenizer.json`（unigram）対応 |
| `quantize.py` | float32 → int8（グループ 32、グループごとに float32 のスケール） |
| `Makefile` | `make models` が全モデルを取得・変換・量子化し、`public/models/` に 8 MiB の部品として置く。`make run` は開発サーバー |
| `tests/smoke.mjs` | デプロイ前のスモークテスト。Node 上の最新 Pyodide でエンジンとモデルを確かめる（`make models` の後に `node tests/smoke.mjs`、約 8 秒） |
| `tests/e2e.mjs` | 実ブラウザでの通しテスト（Playwright） |
| `experiments/simd-kernel/` | SIMD カーネルの試作一式（未導入。TODO の T30 の出発点） |
| `.github/workflows/deploy.yml` | `make models` → `npm run build` → GitHub Pages |

公開先: https://takano32.github.io/pyodide-llama-py/ （リポジトリの旧名は pyodide-llama2-py。Pages の旧 URL は転送されない）

## これまでに分かったこと（決定・実測・理由）

計測環境: ARM big.LITTLE（Cortex-A78×4 + A55×4、スマホ級）、メモリ約 6.6GB・スワップなし、Node 24（V8）、stories15M、greedy。

- **ボトルネックの特定。** 元の純 Python 版は 0.26 tok/s。原因は行列積の内側ループをインタプリタが回していること。NumPy 化で約 50 tok/s（約 200 倍）。WASM SIMD カーネルなら 170〜190、int8 で 288〜334 tok/s。スレッド化はこの規模では効かない（ネイティブ OpenMP でも同じ）。GPU も 15M 級では固定費負けする。詳細は gist: https://gist.github.com/takano32/196c6f93979ad44f98cee5712fdd3901
- **NumPy エンジンの正しさ。** llama2.c の C 実装と 5 プロンプト × 256 トークンでバイト単位一致。GQA は相対誤差 1e-6 以内。unigram トークナイザは本物の sentencepiece と 16 例すべて一致。
- **tokenizer.bin は llama2.c 本家のもの**を使う。llama2.py 付属の古いファイルは語彙が 204 個重複しており、句読点や大文字が学習されていない ID になっていた。
- **プロンプトの先頭に空白を付ける**（sentencepiece のダミープレフィックス）。付けないとパープレキシティが 8.5% 悪化する。
- **int8 の品質は原本と区別できない。** stories15M で +0.04%、tiny-lm 91.3 → 91.1、llm-jp-3-150m 22.76 → 22.69、最尤トークン一致率 約 98%。int4 は +16.8% で不可。greedy の出力は途中から原本と分岐するが破綻はしない。
- **モデル。** tiny-lm（29M、MIT、日英 Wikipedia、質は低い：パープレキシティ 91）、llm-jp-3-150m（Apache-2.0、質は段違い：22.8、ただし約 8 tok/s・メモリ約 500MB）、TinyStories 260K / 3.5M / 15M / 42M。小さいモデルは greedy だと反復するので、日本語モデルは temperature 0.7 / top-p 0.9 / 繰り返しペナルティ付き。
- **ブラウザでの速度（Chromium）。** tiny-lm 約 40、stories15M 約 45〜50、3.5M 約 107、260K 約 300、llm-jp-3-150m 約 8.5 tok/s。
- **分割並列ダウンロードは約 1.8 倍速い**（本番 CDN で 167MB が 20.4 秒 → 11.2 秒）。
- **Pyodide + ctypes の SIMD カーネル（未導入・実証済み）。** カーネルを Emscripten のサイドモジュールとして `ctypes.CDLL` で読み込み、NumPy のメモリを直接計算すると、Python が制御したまま 181（float32）/ 282（int8）/ 348（relaxed SIMD int8）tok/s。emcc なしでも、AssemblyScript の出力に `dylink.0` セクションを付ければ読み込める（Pyodide 0.29.4 と 314.0.7 で確認、Chromium と Firefox で動作）。

## 落とし穴（実際に踏んだもの）

- **GitHub Pages は全ファイルを 10 分キャッシュさせる。** デプロイ直後に新しいページが古い `worker.js` を動かして 404 になった。対策として Worker の URL にコミットのハッシュを付けている（`astro.config.mjs` の `__BUILD__`）。Worker が読む新しいファイルを足すときは同じ `?v=` を付けること。
- **GitHub Pages は `.bin` も gzip で送り、Range 要求には gzip ストリームの断片を返す。** だから Content-Length は展開後のサイズではなく、ブラウザ側での範囲分割はできない。進捗とバッファ確保には `src/models.js` の `bytes`（展開後の正確なサイズ）を使う。**モデルのサイズが変わったら `bytes` も直す**（合わないと Worker がエラーで止まる）。
- **`loadPyodide()` は wasm の取得に失敗してもエラーにならず固まる。** 例外を前提にしたフォールバックは機能しない。
- **ページに `<meta charset>` がないと日本語が化けてモデルに渡る。**
- **Pyodide の NumPy は BLAS も SIMD も無効**でビルドされている（スカラー WASM 相当）。SciPy の OpenBLAS に差し替えても 1.15 倍で、wheel が 16MB 増えるだけ。
- **AssemblyScript のサイドモジュールには静的データを置けない**（再配置されない）。標準の数学関数は使わず、テーブルなしの実装を書くこと。relaxed SIMD を使うカーネルは別ファイルにして `try/except` で読む（Safari は未対応。インストーラが `*.so` を全部先読みするので拡張子も変える）。
- **開発機はメモリが少ない。** 空き 600MB でヘッドレスブラウザを動かしてマシンごと落ちたことがある。ブラウザのテスト前に `free -m` で空きが 1GB 以上あることを確認する。llm-jp-3-150m の float16 原本（ブラウザで約 800MB）はこの機械では試さない。
- **止められたシェルコマンドが途中まで実行されていることがある。** 止められたら `git status` で状態を確かめる。
- サイトは約 840MB。GitHub Pages の目安は 1GB なので、モデルを足すなら何かを削る。

## 検証手順

1. エンジンを変えたら、まず `node tests/smoke.mjs`（デプロイでも走る）。加えてネイティブの Python で回帰確認する: stories15M（float32）で `Once upon a time` の greedy 出力が `Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine.` で始まること。stories260K なら `...She loved to play outside in the park.`。余裕があれば llama2.c の `run.c` を `gcc -O2` でビルドして全文一致を見る。
2. ページや Worker を変えたら、**実ブラウザで通しで確認する**: `npm run build && node tests/e2e.mjs [モデル ID] [chromium|firefox]`。準備（`playwright-core` とブラウザの入れ方）はスクリプト冒頭のコメントにある。「Run が有効になる → Enter → 回答の下に tok/s が出る」まで待ち、スマホ幅でページ自体がスクロールしないことと、決定的なモデルでは出力の冒頭も確かめる。
3. push 後は `gh run watch` でデプロイを待ち、本番に対して同じ確認をする: `node tests/e2e.mjs stories260K chromium https://takano32.github.io/pyodide-llama-py/`。
