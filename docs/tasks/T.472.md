### T.472 [整理][その他] `.ts` にした後の古い名前 — 状態: 完了（2026-10-11、T.368.1 から。規模 小）
- `jobs.ts` のエラーの文「…: see jobs.js」（束に入る文字列: 替えると束が変わる）。`.claude/agents/perf-auditor.md` が `public/forward.js`・`public/helper.js`・`public/jobs.js` のまま。
- **2026-10-30**: `jobs.ts` の制御領域の検査のエラーの文を「see jobs.ts」に（束の文が 1 つ変わる。端末の鍵に入らず、どの検査も読まない文）。`perf-auditor.md` は T.437（`faceb109`）が直していた。コメントの中の `forward.js`・`threads.js` のような名前は、束と記録が使うモジュールの名前として残す（このタスクの外）。CI を省いた。訳: 起きない検査の文 1 つ。
- レビュー（Sonnet low、2026-10-30）: OK。その文を読む試験・文書は無く、束の文を固定する記録も無い。T.437 の直しも確かめた。取りまとめ役がビルドして `dist-check` も通した。
