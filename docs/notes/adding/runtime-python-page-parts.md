# 実行時のファイル・Python の部品・ページの部品を足す

AGENTS.md の「物を足す手順」節から移した（T658、2026-11-12）。要点は AGENTS.md に。

どの手順も「触る所」と「忘れると誰が教えるか」まで書く。**教えないと書いてある所は、黙って別の動きになる**。試験の側の 2 つ目の写し（`FACTS`・`ORDER`・`KINDS` など）はわざと 2 回書いてあるもので、足したら一緒に直す。

### 実行時のファイル・Python の部品・ページの部品を足す

- **実行時の部品**: `src/runtime/<窓口>/` に置いて窓口か隣から静的に import（届かないファイルは検査が落とす）。新しいグローバルは `AMBIENT`。状態を持つなら持ち主の形で、`tests/*-parts-check.ts` に部品の試験。
- **Python の部品**: パッケージのフォルダに置くだけ。外に出す名前は窓口の import に。`make python`。
- **ページの部品**: `src/page/` に置き、`index.astro` が順に名指しで読む。**ページが読む URL の名前は `parameters.ts` の表 `NAMES` に 1 行**（T643.9: 名前は訪問者のブックマークと README が持つので変えない。読むのは `address.ts` の `parameters` だけで、その `get()` は表の名前と `Sampled` の鍵のほかを tsc が断る。書くのは `addressWith()`・`addressOf()` だけで、モデルを名指す名前（役 `"model"`）は `addressOf()` が全部消してから書く。ほかの部品が `searchParams`・`URLSearchParams`・`location.search` を書くと `page-parts-check` が落とす（worker に `search` として丸ごと渡すのは除く）。表の 2 つ目の写しも `page-parts-check`。worker が読む名前（`runtime.ts`・`request.ts`・`pyodide.ts`）と `/benchmark/` の名前は表の外）。持ち主の関数は誰かが呼んでいること。確かめは実ブラウザ。**モデルのページ（`src/page/`）の保存（`localStorage`・`sessionStorage`）を読み書きするのは `remembered.ts` の `recall(名前, 読み方, 無いときの値, 置き場)`・`keep(名前, 文か null, 置き場)` だけ**（T643.3: catch はこの 2 つで、`silent-check` の `SILENT` の行も 2 つ。覚える物ごとの関数（`rememberGpu()`・`rememberThreads()` など）も `remembered.ts` に置き、呼ぶ側が無いときの値を選ぶ。鍵の名前は訪問者の端末が持っているので変えない。前の値が読めなければ新しい値で上書きする。`/benchmark/`（`src/benchmark/`）はまだ直に読み書きする。`isolation.ts` の Service Worker の登録の try の中の `sessionStorage` は登録の catch が持つ）。**この訪問者のブラウザが保存した変換があるかは `list.ts` の `keptEntries`（項目の id の集合: `showKept()` が `choosing.ts` の純な `keptOf(保存, 項目)` で置く）で読む**（T643.5: 選択肢の「kept in this browser」の文は描くだけで、聞かずに始めるかの判断に使わない。`page-parts-check` が見る）。
- **1 ファイルを `.ts` にする**（`src/` の `.js` は T480 で無くなった。次に要るのは `public/coi.js` のような束ねない物だけで、これは `// @ts-check` のまま）: `git mv` と名指す行を 1 コミット、型を足す（文を変えない）、`npm run types`・build・`dist-same`・`same-syntax`・軽い組・網。
