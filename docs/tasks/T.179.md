### T.179 [バグ][遠隔試験] bench-check が Windows の WebKit の close() で止まる — 状態: **済（2026-09-27、持ち主「CI まだ不具合のこってるじゃねえか」）**（規模 小）
- 根拠: bench.yml（36296244022、T.176 の後）の Windows のジョブが、WebKit の節の報告（05:15）の後、次の Chrome を始めないまま 55 分止まり、取り消した。T.141 で `e2e.mjs`・`bench-browser.mjs`・`stock-firefox.mjs` の `browser.close()` に 15 秒の見切りを入れたが、T.134 で足した `tests/bench-check.mjs` に入れていなかった。
- 直し: `bench-check.mjs` の close にも 15 秒の見切り（見切ったら 1 行言って次のブラウザへ。終わりは前から `process.exit`）。
- 確かめ: bench.yml を走らせ直し、Windows のジョブが終わること。
