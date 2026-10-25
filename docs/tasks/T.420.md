### T.420 [整理][遠隔試験] `tests/test_conduct.py` が 1041 行 — 状態: 完了（2026-10-25 に分けた。前の状態: 未着手、2026-10-11、T.374.3 から。目安 800 行。規模 小）
- GGUF の頭の試験を自分のファイルに分ける。代役の変換器は「頭が揃った」を「バイトが `base` に届いた」と同じに見る（詰め物の誤りを隠した形: AGENTS.md に教えを書いた）。

- **結果（2026-10-25）**: GGUF の頭の試験（`GGUF_HF` から `test_a_vocabulary_makes_the_weights_a_gguf_whatever_their_name` まで、251 行）を `tests/test_conduct_gguf.py` に、中を 1 字も変えずに動かした（使わなくなった import だけ両方から外した）。`tests/test_conduct.py` は 1042 → 791 行。共有の `HF`・`REVISION`・`repository()`・`requests_of()` は `test_conduct` から取る（`test_spelled_bpe.py` と同じ形）。段取りの試験 4 ファイルで前と同じ 181 が通る。網の `python` の記録は試験の名前（モジュールの道）が替わるので、この回の前後は「違う」と出る（製品は触っていない）。代役の変換器の「頭が揃った」の見方の教えは AGENTS.md の試験の作りにある（前の回で書いた）。CI を省いた。訳: 試験を動かしただけで、pytest で回した。
- レビュー（Sonnet low、2026-10-25、T.420）: must-fix なし。動かした行は 1 字も変わらず（`comm` で突き合わせ）、`test_conduct` から取るのは `test_*` でない名前だけで 2 回集まらない。勧めを入れた: AGENTS.md の「取り込み元の種類」に新しいファイル、頭のコメントの行数。
