### T.494 [整理][遠隔試験] T.357.4 の後の残り — 状態: 完了（2026-10-30、Opus medium、ブランチ `t.494-cpu`。前の状態: 未着手、2026-10-12、T.357.4 とそのレビューから。規模 小〜中）
- **この回にしたこと**: 参照の道具の終わり（`tests/reference.py` の `ended()`）が「the CPU: <名前>」を 1 行言う（Linux の `/proc/cpuinfo` の model name、arm64 は「unknown」: Node の `os.cpus()[0].model` と同じ読み）。`reference_dry.py` が道具ごとに 1 行あることを見る（壊し方: 2 行と求めると「1 lines with 'the CPU: ', not 2」で落ちる）。手元: `reference_dry.py`（約 2 分）。**CI を省いた。訳: 試験の道具だけの変更で、乾いた回が `ended()` を使う 6 つの道具を通る。**
- **レビュー（Sonnet low、2026-10-30）: 直し 1 つ、直した。** `ended()` を通らない道具が 4 つあった（`reference_yarn.py`・`reference_27b.py`・`reference_saluki.py`・`reference_saluki_numpy.py`）。読みを `tests/cpu.py` の `cpu_name()` に置き、`yarn`・`27b`・`saluki_numpy` の終わりも CPU を言う（3 つとも import で確かめ、乾いた回は通る。走らせたのは CI だけの道具なので名前の行は CI の次の run で見る）。`reference_saluki.py` は取得と突き合わせの段の道具で、言わない（速さを比べるのは `saluki_numpy` と素の llama.cpp の段）。arm64 の「unknown」の読みと 2 つの決めは妥当。
- **取りまとめ役の決め**: `made_up_engine()` の「options から書式を外す」行は残す（8 つの道具が同じ形で書き、実物の変換は書式を持つ。作り物では何もしないだけ）。3 つの `real()` を寄せるのは比べ方の違い（トークナイザの突き合わせ・greedy の文の比べ）をそろえる設計の回で、今は行数も減らない: 置いたまま。代役の通らない道（`--gguf`・`--layers`・`ByLayer`・`--model=2B|4B`・`large()`・`chat_nll.py`）は、次にその道を使う回に 1 本回す（下の行のまま）。
- 3 つの参照の道具の `real()` は今も 110〜170 行ずつで、流れは別々。寄せるなら「トークナイザの突き合わせ」「greedy の文の比べ」を段にする所からで、3 つの比べ方の違いを先にそろえる決めが要る。行数は寄せてもほとんど減っていない（1881 → 1856）。
- 代役の通らない道（`reference_llama.py` の `--gguf`・`--layers`・`ByLayer`、`reference_qwen35.py` の `--model=2B|4B` と `large()`）は、骨組みへの呼び替えの後、CI でも乾いた回でも 1 回も走らせていない（読んだだけ）。次にその道を使う回に、まず 1 本回す。`tests/chat_nll.py` の import も見ていない。arm64 の CI も。
- `only_extra=true` の run が CPU の名前を出さない（道具が自分で 1 行出せば、run どうしの比べが「同じ CPU か」を言える）。
- `made_up_engine()` の「options から書式を外す」は走らない行（作り物の変換は書式を作らない）。消すか、訳を書いて残すか。
- 弱い誤りを 1 つ足すとき、試験の側で触る所が 1 つ増えた（乾いた回の `WEAK`）。
- 実装の回に、記録のコミットに試験の 6 行の直しが混ざった。裏で回した壊し方の道具が期限の後も生きていて、次の 1 本と入り混じった（やり直した）。次の頼みには書いてある。
