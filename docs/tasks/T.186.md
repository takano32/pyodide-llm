### T.186 [描画][計測実行] ベンチの小さな直し — 状態: **完了（本線に入れた 3dd134d、レビュー済み T.192）**。画面は持ち主が見た
- スマホ幅で機能の表の「storage the browser grants」の値が右で切れる（T.177 の画面で見つけた）。
- T.175 の `sameAsNormsApart` と量子化の比べが報告に ok / WRONG しか出ない: 細かい数字を 1 行で。
- 見た目なので `preview.yml` の画面を持ち主に見せる。
- 作ったもの（2026-09-27）: (1) 列が 2 つの表（名前と値。いまは「This browser」だけ）はセルを折り返す（`table.pairs`、`overflow-wrap: break-word`）。ほかの表は今までどおり折り返さず、表の中で横にスクロールする。(2) 層の判定の後ろに 1 行（`src/bench.js` の `layerCheckNumbers()`）: 4 つの量子化したベクトルのスケールの差の最大と ±1 の数の合計（「quantized: scales 2.5e-7, 6 of 8416 off by 1」、外れたものがあればその 1 つの名前と理由）、DP4A の融合の形は norm を分けた形との差（「norms apart: 2 ulp, 1 off by 1, stream 3.0e-7」、ビット単位で同じなら「bit for bit」だけ）。判定の `quantized` は文ではなく 4 つの `{point, wrong, scale, apart, of}` にし、CI のログ（`bench-check.mjs`）は今までどおり 4 つを別々に出す。報告の長さ（T.185）: 層の形 1 つにつき約 40〜90 字増える（未計測）。

- **持ち主が画面を見て「入れてよい」（2026-09-27）。本線に入れた。**
- **レビュー（Opus xhigh、2026-09-27、T.192）: OK、must-fix 0。** bench.yml（run 36326833940）の Linux と macOS の SwiftShader で、ページの報告に層の 1 行が出た（「a layer, DP4A, fused (T175) ok (worst 1.8e-7; quantized: scales 6.7e-7, 0 of 8416 off by 1; norms apart: bit for bit)」）。bench-check の 4 つずつの行も出て、ログに NaN と undefined は 0。`quantized` を文から配列に変えたのを読む所はほかに無い（bench-check と `layerCheckNumbers()` だけ）。`quantizedOff()` の返す鍵（wrong・scale・apart・of）は関数の読む鍵と同じ。2 列の表は機能・重みのほかの費用・GPU の上限の 3 つで、どれも折り返してよい。preview.yml（run 36316368969）で 360 px の機能の表は 328 px の箱に 328 px、ページは 360 px。
