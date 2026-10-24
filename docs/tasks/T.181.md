### T.181 [バグ][遠隔試験] Windows の CI で bench-check のブラウザが 1 つも起動せず、それでも成功になる — 状態: **直した（2026-09-27）**（規模 小）
- 根拠: T.180 のプロファイルの場所を `new URL(import.meta.url).pathname` から作ったので、Windows では「/D:/a/…」が `D:\D:\a\…` になり、5 つのブラウザが全部「could not start: … ENOENT」（bench.yml 36301878917）。起動しないブラウザは飛ばす作り（ランナーに Chrome や Edge が無いこともある）なので、何も測らずにジョブは success だった。
- 直し: パスは `fileURLToPath` で作る。ブラウザが 1 つも起動しなければ exit 1。
- ついでに分かったこと: T.180 の後の macOS は、前から落ちていた WebKit の保存の節（UnknownError）も通った（ディスクの上のプロファイルで）。
