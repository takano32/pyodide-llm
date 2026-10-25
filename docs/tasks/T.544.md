### T.544 [試験][CPU] `kernels-in-browser` の 3 値に、4 で割り切れないグループの通りを足す — 状態: 完了（2026-10-25、Opus medium。前の状態: 未着手、2026-10-16、T.533 のレビューの気づき）
- いまは groups=16 の 1 通りだけで、残りのグループの道（`half()`）を Chromium・Firefox・WebKit で踏んでいない。groups=7 のような通りを足す。
- **結果**: `tests/kernels-in-browser.mjs` に 1 行 7 グループの行列（自分の重み・スケール・活性値・答え）で `matmul_t2`・`matmul_t2r`・`matmul_t2r_tile`（5 トークン: 4 つの組と余り 1）をビット単位で。行のスケールは 14 バイトなので、語の中の始まりが行ごとにずれる。CI の WebKit 27.2・Chromium 156・Firefox 157 で ok（run 40158238771）。壊し方: 余りの道で群 0 のスケールを読む形を relaxed と素の kernel のそれぞれに入れて、7 グループの通りだけが落ちる（16 グループの通りは通る）。軽い組 465 秒。
- レビュー（Sonnet low、2026-10-25）: must-fix なし。素の kernel も余りの道に入る、置き場はほかの検査と重ならず 8 MiB の内、答えの足し順は kernel と同じ。勧め（素の kernel の壊し方も）を回して落ちた。
