### T.543 [整理][Gemma4] `perlayer.ts` が層ごとの埋め込みの表のスケールを float32 と読む前提 — 状態: 完了（2026-10-25、Opus medium。前の状態: 未着手、2026-10-16、T.533 のレビューの気づき）
- `forward/perlayer.ts` は `F[scales/4]` で float32 のスケールを読む。この表が 3 値になる道は無い（`ternary_x` が断る）ので今は当たらない。足すなら `packing.scale !== 4` を断る一行。
- **結果**: `perLayer()` の断りに `packing.scale !== 4` を足した（3 値の float16 のスケールを float32 と読まない）。試験は `layer-parts-check`: 3 値の表を断る（main の perlayer.ts では落ちる）。軽い組を通した。CI を省いた。訳: CPU のエンジンを組み立てる所だけで、一覧のどのモデルもこの表を 3 値で持たない。
- レビュー（Sonnet low、2026-10-25）: must-fix なし。int8 は通り、int6 と塊の型は前から断られる。この表を読むほかの所（GPU、NumPy）に同じ前提は無い。断りを確保の前に動かす案は入れない（投げれば forward ごと捨てられる）。
