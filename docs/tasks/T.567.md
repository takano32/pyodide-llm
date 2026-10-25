### T.567 [バグ][WebGPU] `gpuBytes()` が linear の層の畳み込みのタップ分を数えない — 状態: 完了（2026-10-25、Opus medium。前の状態: 未着手、2026-10-16、T.271 のレビューの気づき。規模 小）
- `conv × mixed × 4` × linear の層（9B で約 3 MiB）。

- **結果**: `layersOf()` の `held` に、畳み込みが読む行（場所ごとにタップ × mixed）を足した（前はタップのベクトルだけ）。GPU が linear の層ごとに持つもの（状態・行・タップ・wb・wa・dt_bias・decay・norm）が `gpu/linear.ts` と計画のベクトルに 1 つずつ合う。増える見積もりは 0.8B で約 1.7 MiB、4B・9B で約 3 MiB、27B で約 7〜8 MiB（式から）。試験は `sizes-check`: タップが 1 つ多いと linear の層 3 つ（4 層のうち、4 層に 1 つが attention）× 2 × mixed × 4 バイト増える（前の式は半分で落ちる）。網の `choices`・`calls` は同じ（置き場の判定は変わらない）。軽い組 451 秒。CI を省いた。訳: 見積もりの数だけで、シェーダも GPU の道も変わらない。
- レビュー（Sonnet high、2026-10-25）: must-fix なし。数え方は GPU の確保と計画のベクトルに合い、CPU の側（`staged()`・混ぜる側の `layersOnGpu()`）は前から正しい。置き場の判定が変わるモデルは無い（線にいちばん近い 0.8B の `deviceMemory` 4 で余裕約 90 MiB）。気づき（入れない）: 層をまたぐ作業のバッファ（linear の層の z など、9B で約 6 MiB）は `gpuBytes()` がどの種類でも前から数えない作り。
