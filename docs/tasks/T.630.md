### T.630 [修正][遠隔試験] 全部の組の Python を 3.14 に留める（Unicode 17 と tokenizers） — 状態: 完了（2026-10-31、取りまとめ役が見つけた。規模 小）
- 毎晩の全部の組（10-28 から）と T.629.1 の全部の組で `test_bytebpe.py` の全符号位置の 7 つが落ちていた: `setup-python` の `3.x` が CPython 3.15.1 になり、その `unicodedata` は Unicode 17。エンジンの前分割は Python の表で字の種類を決めるので Unicode 17 で増えた字（U+088F など 4,657）を字と読み、本物の tokenizers 0.23.3 は Unicode 16 のまま読まない。手元（3.14.4・Unicode 16）では tokenizers 0.23.1 でも 0.23.3 でも通る。**取りまとめ役の決め**: `tests.yml` だけ 3.14 に留める（ページの Pyodide は自分の Python を持つので訪問者には関わらない）。覆す条件: tokenizers が Unicode 17 を読むようになる。
- 確かめ: CI の全部の組（`full=true`、run 41186578813）が通った（`test_bytebpe` の 7 つも）。
- レビュー（Sonnet low、reviewer、2026-10-31、T.630）: must-fix なし。全符号位置の試験は全部の組だけ（`tests.yml`）が回すので、ほかのワークフローは `3.x` のままでよい、と確かめた。入れた勧め: Python の Unicode が 16 より新しいときの失敗の文に訳と直し方を書いた、戻す確かめ方を下に。
- **戻す確かめ方**: tokenizers の新しい版が出たら、`tests.yml` を `3.x` にしたブランチで `full=true` を 1 回回す（毎晩は 3.14 で通るので、ほかに知る手は無い）。3.14 の支えが切れる前に。
