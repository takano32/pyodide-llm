### T.453 [バグ][遠隔試験] `tests/write.mjs` が dtype に依らず「int8 N GB」と書く — 状態: 完了（2026-10-11、T.235 から。規模 小）
- 3 値のファイルが「int8 0.48 GB」と出る。チェックポイントの dtype を言う。
- **2026-10-30**: 行の dtype は `llama2_numpy.checkpoint_dtype(見出し, 大きさ, options)`（エンジンが読む dtype そのもの: options の `dtype` の欄は読まない）。手元で `tests/make_ternary.py` の作り物 2 つ（ternary と int8）に回して「ternary 0.00 GB」「int8 0.00 GB」。CI を省いた。訳: 書く行 1 つの道具の直しで、手元で両方の dtype を回せた。検査は足していない（人が読む行）ので壊し方も無い。
- レビュー（Sonnet low、2026-10-30）: OK。引数は読み込みと同じ見出しと大きさ、options の余分な鍵は `form_of()` が捨てる。見出しとして成り立たないときは前から `footprint()` で落ちるので新しく落ちる道は無い。塊のファイルは "int8" と答える（docstring、走らせてはいない）。
