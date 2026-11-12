# 設計の決まりの細部

AGENTS.md の「設計の決まり（いまの形）」節から移した（T658、2026-11-12）。要点は AGENTS.md に。 AGENTS.md の同じ項には 1〜2 行の要点がある。

## 木と束（「設計の決まり」の項から）

- **ページが実行時に取るライブラリ（T556）**: 答えの Markdown の marked と DOMPurify は束に入れず、`src/page/markdown.ts` が書式のあるモデルの準備完了の時に 1 訪問 1 回、Pyodide と同じく最新の版を解決して jsDelivr の `+esm` を `import(/* @vite-ignore */ 番地)` で読む。**番地は `import()` の前に変数に作る**（引数の中の `await` は束で構文の誤りになった）。取れない・描けないときは警告 1 行で平文。描くのは段落・リスト・太字・斜体・見出し・コード・表で、HTML・リンク・画像などは文字のまま。描く範囲を変えるなら `markdown.ts` の描き手と `ELEMENTS` と、`tests/markdown-check.ts`・`tests/markdown-in-browser.ts` の 2 つ目の写し。
- **Service Worker の登録・`?coi=off` の外し方・1 回の読み込み直しは、2 つのページとも `src/page/isolation.ts` の `isolating(how)` の 1 か所**（T643.8。登録を外す輪 `unregistered()` も、`receiver.ts` の Pyodide が来なかった枝が読む）。`/benchmark/`（`src/benchmark/isolation.ts`）はそれを import して違いだけを言う: `how.fallback`（`coi-fallback` の印のあるタブで、モデルのページは外して読み込み直し、`/benchmark/` は外さずに隔離なしで進む）と `how.offline`（モデルのページは `?offline=` を覚える `remembered.ts` の `offlineKept()`、`/benchmark/` は覚えた値を読むだけ）。**`isolation.ts` は何も import しない**（`/benchmark/` の束にモデルのページの部品と一覧を持ち込まない）。2 つのページの台本が読む部品は束ね手が 2 つの共通の塊（今は `updated.<ハッシュ>.js`）に入れるので、`dist-check` の 9 は「ページの台本だけが import する塊」を DOM の名前を読む側に数える（worker が読む塊は今までどおり `AMBIENT` まで）。
- **開発サーバ**（`npm run dev`）は `vite.server.headers` で自分で COOP・COEP を送る。Python の zip 2 つは `astro.config.mjs` のプラグイン `python-archives` が、ビルドと開発サーバの始めに作り、開発サーバの間は `src/python/` の `.py` が変わるたびに作り直してページを読み込み直させる（T433: npm を通さない `astro dev`・`astro build` でも zip がある）。端末でない所の `astro dev` はデーモンで残る（`npx astro dev stop`）。

## Python のエンジンと変換器（「設計の決まり」の項から）

- **考えの上限（T520）は `generate()` の引数 `thinking`**（0 は上限なし）: 語彙に考えの印が 1 つの語片としてあるモデル（`mark_of()`）の考えを追い、上限の長さになったら次のステップで閉じる印を書く。**印の種類は表 `THOUGHT_MARKS`**（`engine/generation.py`、ページの写しは `src/page/thought.ts` の `MARKS`）: `<think>`…`</think>` と、harmony の analysis のチャンネル `<|channel|>analysis<|message|>`…`<|end|>`（T592.1: 開く印は 3 トークンなので書式が開くときだけ追い、答えは閉じた後の final のチャンネル。ページは `<|start|>assistant<|channel|>final<|message|>` を外して描く）と、PLaMo 3 の `<|plamo:begin_think:plamo|>`…`<|plamo:end_think:plamo|>`（T614: どちらも 3 トークンで、モデルが自分で開くので、エンジンは追わず上限も効かない。ページは文で畳む）。どの種類かは `thought_marks()` が、プロンプトが開く印で終わる種類、無ければモデルが自分で開ける種類から選ぶ。GPU には上限の残りより多いステップを頼まない。stats に `thought` と `thought_cut`。上限は項目の `generation.thinking`（窓口が note が「thinks」で始まる項目に `THOUGHT_BUDGET` = 3,584 を置く。表は docs/notes/t520-thinking-budget-2026-10-16.md）と URL の `&thinking=`。

## ページのエンジン（forward）（「設計の決まり」の項から）

- **層ごとの埋め込み（Gemma 4 の `per_layer`）は `forward/perlayer.ts`**。**表の行は「行の読み手」から**（`createForward({ rows })`）。**表はメモリに置かず保存から読む**（持ち主、2026-10-15）: 初めての訪問は変換のはじめに開く保存のファイル（`keeper()`: OPFS のある訪問の全部の変換で開き、使わなければ落とす）に届くまま書いてそのハンドルを読み手にし、2 回目は表の部品を読まずに飛ばして `openRows()` で開く。**ハンドルは 1 つのファイルに 1 つ**なので、持ち主は `weights.ts` の `rowsNow`、閉じるのは `letGo()`。表をメモリに持つのは: OPFS が無い・保存の dtype が違う・`?without=kernels`・別のタブがファイルを持つ（警告 1 行）・表を書き終える前に書けなくなった。
- **行ごと・層ごとに型の違うモデル（llama.cpp の塊のまま持つモデル）**: 計画のテンソルは自分の型を言い、層ごとに型の違う束は `kind: "layers"` と `layers` を持つ。計画の `own_kinds` で、エンジンは束を層ごとの行列として引き、**同じ入力を読む行列を「活性値の取り方」（`PACKED` の `takes`）ごとの段に分ける**。塊のカーネルは別のモジュールで、**塊の行のあるモデルのエンジンだけ**が同じメモリの上に載せる（スレッドも）。表 34,992 バイトを 1 回作り（`sb_tables`）、relaxed の内積が符号つき × 符号つきを厳密に掛けるかを 1 回聞いて（`signed_dots`）、真なら `matmul_sbw`。linear の層の出の行列は、計画の `tiled` が真なら値の head を並べた枠を読む。GPU には置かない（**例外は GPU が取る詰め方だけの束**: Q4_K の行列と、層ごとに int8 か Q4_K の束。下の WebGPU の「llama.cpp の塊を GPU に」）。**見積もり**: `footprint()` は形の `kinds` を読み、ファイルの dtype のままの重みの数は Python が数えて worker が options の `weights` で渡す。**worker**: 項目の `conversion.blocks` が変換に届くのは int8 のカーネルで掛ける読み込みだけ（`weights.ts` の `runsPacked()`）。そのとき塊のカーネルを全部のビルドぶん先に取る（`runtime.ts` の `blockKernels()`）。保存の名前は末尾に `:blocks`。ステータス行は「llama.cpp's blocks」（持ち主の文）。**試験の線**: 塊のカーネルは活性値を 8 ビットに丸めるので、深いモデルは同じ丸めの NumPy とも最大の logit の 1e-2 までしか合わない（1 層の型の取り違えと同じ大きさ）。だから `tests/blocks-forward-check.ts` は呼び出しを 1 つずつ計画とカーネルの模型に照らす。

## シェーダと端末の鍵（「設計の決まり」の項から）

- **一様バッファ（WGSL の `struct Params`・`Norm`）のバイトは、その構造体を持つ部品の作り手で**（T642.6）: flash のタイルの `Params` は `shaders/attention.ts` の `flashParams()`（vec の形は `flashVecParams()`）、ステップの融合の `Params` は `shaders/fused.ts` の `fusedParams()`、`Norm` は `shaders/steps.ts` の `normParams()`（GPU の Worker は `gpu/device.ts` の `normShape()` 越し）。GPU の Worker で `new ArrayBuffer` に構造体の欄を手で並べない（並びの誤りは tsc も名前も言わない。網の `gpusteps` が書き込みのバイトで落とす）。
