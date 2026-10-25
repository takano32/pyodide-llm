### T.431 [整理][遠隔試験] `tests/test_converter_version.py --write` が `llama2_convert` を自分で道に置かない — 状態: 完了（2026-10-25。前の状態: 未着手、2026-10-11、T.367.1 から。前からある。規模 小）
- **結果（2026-10-25）**: `tests/test_converter_version.py` は import の前に `tests/tree.py` の `python_folder()` を `sys.path` に置く（pytest では `conftest.py` が置くが、`--write` で直に走らせると「No module named 'llama2_convert'」で止まっていた: 直す前に再現した）。`--write` が 0 で終わり fixture は同じ、pytest も通る。CI を省いた。訳: 手で走らせる試験の道具 1 か所で、手元で両方の走り方を回した。
- レビュー（Sonnet low、2026-10-25、T.431）: must-fix なし。conftest と同じ道が重なるだけ、根は `Path(__file__)` からなので網の相手の木では相手を指す、文を読むのは `kept.ts` の行だけ。
