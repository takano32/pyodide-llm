### T.570 [運用][文書] TODO.md を TASKS.md に改名する — 状態: 完了（2026-10-24、持ち主「TODO.md を TASKS.md にリネーム」。記録だけなのでデプロイは走らない。規模 小）
- `git mv` と、木の中の「TODO.md」の指しを全部「TASKS.md」に（AGENTS.md・`.claude/agents/`・`docs/`・コードと道具のコメント・`tests/ledger.py` の読む先と文）。`tests/ledger.py` の変数は `LEDGER`（`TASKS` は `docs/tasks/` のフォルダの名前で使っていた）。
- **変えなかったもの**: `tests/fixtures/long-27b.txt`（27B の比べの入力の文: 1 字変えるとトークンが変わる）、済んだ T.526 の題名「TODO.md を分ける」（当時の名前）、持ち主の言葉の中の「TODO」。
- CI を省いた。訳: 触ったのは文書・コメントと `tests/ledger.py` の 1 つの道で、`python tests/ledger.py index` と `--check` を手元で回した。
- レビュー（Sonnet low、2026-10-24）: OK。`TODO.md` の残りは T.570 自身・T.526 の題名と目次・fixture の 3 行だけ、`ledger.py --check` は ok。
