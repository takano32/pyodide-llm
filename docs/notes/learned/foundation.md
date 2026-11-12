# 土台

AGENTS.md の「これまでに分かったこと」節から移した（T658、2026-11-12）。要点は AGENTS.md に。

計測環境: 2026-09-26 までの「この開発機」は ARM big.LITTLE（Cortex-A78×4 + A55×4）、メモリ約 6.6GB・スワップなし。2026-09-26 からは a1-free（OCI の Ampere A1、Neoverse-N1 × 2、メモリ 11.9GB・スワップ 4GB、Ubuntu 26.04.1）。どちらも Node 24。**変更の前後を 2 つの機械の数字で比べない**。モデルごとの数字の表・run の ID は `docs/tasks/T番号.md` と `docs/notes/` にある。

### 土台

- **a1-free の手元の時間。** pytest 5〜10 秒、smoke 12 秒、forward-check 52 秒、`make kernels` 24 秒、`make models` 2 分 11 秒（翻訳のモデルの取得 `fetch_translations.py` は別に 19.5 秒: zstd で 117 MB）、build 9 秒。1 コアの 1 トークン: llm-jp-3 150M 12.0 ms、tiny-lm 1.62 ms。huggingface.co から 5.4 MB/s。
- **なぜ Pyodide か。** WASM の上の Python で NumPy と動的リンク（ctypes）の両方があるのは Pyodide だけだった。stories15M の tok/s: 素の Python 0.26 → NumPy 52〜54 → 数値の処理を全部カーネルに（f32 181・int8 282・relaxed 348）で、ネイティブの llama2.c `-Ofast`（214）と並んだ。
- **gist**: https://gist.github.com/takano32/196c6f93979ad44f98cee5712fdd3901 （`00-` が要約と目次、`10-` が概要。大きな成果が出たら `10-` を更新する）。
- **スレッドで伸びるのは約 2 倍、頭打ちはメモリ帯域。** バッチ 1 の int8 は 1 バイトあたり 1 積和。**大小のコアに均等に割ると 2 つの大コアより遅い**。単スレッドのネイティブ（NEON）と WASM SIMD カーネルはほぼ同じ。
- **エンジンの正しさ。** llama2.c の C 実装と 5 プロンプト × 256 トークンでバイト単位一致。**tokenizer.bin は llama2.c 本家のもの**（llama2.py 付属のものは語彙が 204 個重複）。**プロンプトの先頭に空白を付ける**（sentencepiece のダミープレフィックス。付けないと perplexity が 8.5% 悪化）。
- **速さ（Chromium、カーネルあり）**: tiny-lm 約 270〜295、stories15M 約 300（int8）、llm-jp-3-150m 75〜79 tok/s（256 トークン）。カーネルなしは約 40・50・8.5。位置とともに遅くなる（llm-jp-3-150m で位置 8 で約 85、2000 で 50、4070 で 35 tok/s）: 文書の tok/s は 256 トークンでの値。小さいモデルは greedy だと反復するので、日本語モデルは temperature 0.7 / top-p 0.9 / 繰り返しペナルティ。
- **int8 の品質は原本と区別できない**（stories15M +0.04%）。int4 は +16.8% で不可。活性値も量子化する経路（llm-jp-3-150m）: 8 ビットの活性値で原本と ±0.1%、7 ビット（relaxed）で +0.39%、速度は 3 割増し。
- **GPT-2 の 7 ビットの崩れは分類器の入力だけ（T92）。** 最終 LayerNorm の重みが数チャネルを 12〜17 倍に増幅し、同じグループの残りが丸めで消える。norm の重みが大きい 8 チャネルを量子化の前に抜き、その列だけ float32 で別に掛ける（`outlier_channels()` / `outlier_columns()`、カーネル `add_columns`）。**門**: 最終 norm の重みの最大/中央値が 4 以上のときだけ（GPT-2 13.9、ほかは 1.1〜3.8。門なしは効かずに約 3% 遅い）。GPT-2 の perplexity は 41.431 → 36.127。覆す条件: 線の近くで損が 3% を越えたとき。
- **int8 より小さい重み（T98）。** 4 ビットはどの形でも +10〜48%、**6 ビットは最悪 +3.37%** で 6 ビット（`int6`: 値は「下 2 ビットが 0 の int8」、詰め方 `pack6` の外は int8 のまま動く）。大きさは int8 の 7/9、**速さは int8 の約 0.45 倍**。だから**ページは int8 が入らないときだけ 6 ビット**（`weightsFor()`・`automaticBits()`、`?bits=6` / `?bits=8`）。**64 ビットのメモリのあるブラウザ（Chrome・Firefox）では int8 のまま 64 ビットに置く**（`automaticDtype()`）。6 ビットになるのは 64 ビットのメモリの無いブラウザ（Safari。Playwright の WebKit 27.2 は持つ: 本物の Safari は未確認）と `deviceMemory` の小さい端末だけ。
- **`navigator.deviceMemory`。** Chromium 系だけが言う。**頭打ちの 8 は今は Android だけ**（Chrome 152 以上の PC は 16 や 32 も言う）。近い 2 のべきに丸める（6 GB のスマホは 4）。コードは `>= 8` を「8GB 以上」と読む。言わないブラウザは 4 GB と見る。
- **1 トークンの内訳（`tests/profile.ts`）。** llm-jp-3 150M は行列積 92%（分類器 46%）・attention 3%（位置 2000 で 48%、4000 で 65%）。
- **最適化の寄与（T52）。** `?without=kernels,int8,relaxed,sampler` で 1 つずつ外せる（知らない名前は `ValueError`）。tiny-lm 48.6 → 187 → 248 → 315 → 416 tok/s。ステータス行は何が動いたかを必ず言う。
- **OS 5 種 × ブラウザ 5 種で動く（`browsers.yml`、表は `docs/notes/t644.9-kernels-measured-2026-09-19.md`）。** relaxed SIMD の無いブラウザは `matmul_q8`（3 値・1 ビットは `matmul_t2`・`matmul_b1`）で動く。Playwright の WebKit は 26.4 で無く 27.2 で有る（T247・T290）。本物の Safari 27 が持つかは未確認（持ち主の端末で: TASKS.md）で、`?without=relaxed` がその代わり。Windows ARM の Playwright の Chromium と WebKit は x86-64 のエミュレーションで 3 分の 1。ランナーの速さは OS をまたいで比べない。
- **ダウンロード。** 分割並列は約 1.8 倍速い。モデルの取得は Pyodide の読み込みより先に始まる（T68）。**準備完了の内訳の秒数は、それぞれが実際に動いていた時間だけ**。NumPy の wheel は版を解決した直後に `pyodide-lock.json` から URL を組んで先読みする。
- **SIMD カーネル。** Emscripten のサイドモジュールとして `ctypes.CDLL` で読み込む（AssemblyScript の出力に `dylink.0` を付ける）。カーネルにアドレスを渡す配列は Python 側で必ず保持する。Python の `kernels=` はいまサンプリングのため（と変換器の詰め手・読み手）。**Python のサイドモジュールは `simdkernel.so` の 1 つで、relaxed のものは作らない**（T644.2: Python は relaxed のカーネルを呼ばない。relaxed は forward.ts が自分でコンパイルする `simdkernel_relaxed_*.wasm` で、`?without=relaxed` もそちらが読む）。
