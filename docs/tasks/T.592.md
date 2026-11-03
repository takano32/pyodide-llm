### T.592 [追加][モデル] llm-jp-4 8B thinking — 状態: 完了（2026-11-03、T.591 から。規模 小）
- **持ち主の決定（2026-11-03、選択肢で「考えを畳めるようにして足す（勧め）」）**: harmony の考え（`<|channel|>analysis` … `<|end|>`）をエンジンの考えの追い方（T.520 の上限と stats）とページの畳み方（`src/page/thought.ts`）が読めるようにしてから、T.592 と T.304（Q4_K_M の塊）を入れる。
- llm-jp/llm-jp-4-8b-thinking @ fc7c15c262710016c19bcda4372a96ff68875846（llama、8.59B、文脈 65536、apache-2.0、harmony の書式、likes 52 で llm-jp でいちばん多い）。Q8_0: mmnga-o/llm-jp-4-8b-thinking-gguf @ d10eb8c8470f89b2082e793a5ffa086a402a23fc（9.133 GB）。
- 一覧の llm-jp-4 8B instruct と同じ形（`HARMONY` の書式、止まり [1, 2, 13]）。考えは見せる（T.304 の持ち主の決め）。Swallow の順位表で一覧に無いものの上位。
- **T.304（4.1 8B thinking）は今も Q8_0 が無い**（作り手は BF16 17.19 GB と Q4_K_M 5.50 GB だけ、masahiroid も Q4_K_M）。T.304 の決めのまま待つ。4.1 を Q4_K_M の塊（`rebuilt`）で入れる道は T.304 の持ち主の判断に足す。


#### T.592.1 harmony の考えを、エンジンが追いページが畳む — 状態: 完了（2026-11-03、Opus medium）
- **子タスクの分け方（取りまとめ役の決め）**: T.592.1 で考えの印の表を作り、T.592.2 で llm-jp-4 8B thinking（Q8_0）を足す。T.304（4.1 の Q4_K_M の塊）はその後に自分の番号で。
- **したこと**: 考えの印を表に: エンジンの `THOUGHT_MARKS`（`engine/generation.py`: `<think>`…`</think>` と harmony の `<|channel|>analysis<|message|>`…`<|end|>`）と、どの種類を追うかを決める `thought_marks()`（プロンプトが開く印で終わる種類、無ければモデルが 1 トークンで開ける種類）。上限で閉じるのは種類の閉じる印（harmony は `<|end|>`）。ページの写し `src/page/thought.ts` の `MARKS`（閉じた後に答えの前に書くもの `answers`: harmony は `<|start|>assistant<|channel|>final<|message|>`、書いている途中の頭は答えに出さない）と、書式が考えを開くかの `opensThought()`（`composer.ts` の正規表現の代わり）。書式が開いた考えは、閉じる印が先に出た種類で読む。`tests/answers.ts` の「考えを終えた」も表の閉じる印で、smoke の作り物の印は表を差し替える形に。
- **取りまとめ役の決め（訳）**: harmony の開く印は 3 トークンなので、モデルが自分で開く考えは追わない（考える形の書式は analysis のチャンネルを開いて終える: すぐ答える形の HARMONY が final を開くのと同じ形。本物の書式は `<|start|>assistant` で終わるので `format_check.py` の `KNOWN` に足すのは T.592.2）。上限で書くのは `<|end|>` だけ（final のチャンネルの頭はモデルが書く）。
- **確かめ**: pytest の全部（3345 通る）、`test_thinking.py` に harmony の 2 つ（上限で `<|end|>`、自分で閉じた考え、final のチャンネルを開く書式では追わない）、smoke、`page-parts-check`（harmony の考えの 5 つ）、`page-modules-check`、`npm run types`、`models-check`。実ブラウザ（手元の Chrome、DevTools MCP）: llm-jp-3 150M が答え、Qwen3 0.6B (thinking) の考えが前のとおり畳まれる（「Thought for 137 tokens」）、コンソールに警告もエラーも無い。手元の Playwright の Chromium は版が合わず（入れると `$HOME` に書く）回していない。
- **レビュー（Sonnet low、2026-11-03）**: 直すべきものは無し。あるとよい点 6 つのうち、`answer.ts` のコメントと `generate()` の説明の行の長さを直した。取りまとめ役の決め（訳）: 2 つの表（`THOUGHT_MARKS` と `MARKS`）を突き合わせる検査は T.592.2 で足すか決める（実物の harmony の文で、特殊トークンが文字のまま復号されるかも T.592.2 で見る）。harmony の考えの中に `</think>` の文字があると先に閉じたと読む端は、まれなのでそのまま。モデルが自分で開く道が harmony に当たらないのは表の形なのでそのまま。
- **本番**（デプロイ run 41581197844、`models.yml` run 41582708680、Chromium）: Qwen3 0.6B (thinking) は準備完了 17.5 秒・27.2 tok/s、llm-jp-3 150M は 5.1 秒・137.9 tok/s。
- CI を省いた。訳: 触ったのは Python の生成の輪の印の選びとページの純な部品で、pytest・smoke・ページの部品の検査・実ブラウザが全部踏む。harmony の考えを実物で書かせるのは T.592.2 の CI。

#### T.592.2 llm-jp-4 8B thinking（mmnga-o の Q8_0） — 状態: 完了（2026-11-03、Opus medium）
- **したこと**: 1 項目を `hf-japanese.ts` の llm-jp-4 8B instruct の後に（hf-llm-jp-4-8b-thinking、mmnga-o @ d10eb8c8470f89b2082e793a5ffa086a402a23fc の Q8_0 9,132,708,672 バイト、語彙と config は原本 @ fc7c15c262710016c19bcda4372a96ff68875846、Apache-2.0 は 2 つのカードから）。書式 `HARMONY_THINKING` は instruct の `HARMONY` に、本物の書式が既定で書く「Reasoning: medium」の行を足し、終わりを analysis のチャンネルを開く形に（T.592.1 でエンジンが追いページが畳む）。options `harmonyThinking` は `harmony` の止まりから `<|start|>`（10）と `<|end|>`（11）を外したもの（考えの後に final のチャンネルを書くため: [1, 2, 13]）。`models-check` の「考える項目」の数え方に、書式が analysis のチャンネルで終わる項目を足した。
- **取りまとめ役の決め（訳）**: 標本抽出はカードが言わないので、考える項目の `thinking`（0.6・0.95・繰り返しの罰 1.0）。instruct の `sampled(1.1)` の繰り返しの罰は長い考えを曲げうる。書かせて読んで決め直す。
- **GGUF と原本（`gguf.yml` の candidates、run 41581446145）**: 通る。どのテンソルも原本を Q8_0 に丸めた最も近い値と同じ（線 1e-3 を越えるもの 0、いちばん大きい相対は 0.0111）、q と k は llama.cpp の回した並び。埋め込みの 1 行（195203「𕔆」）は Q8_0 の float16 の尺度の丸めだけ。語彙は 196,608 で、同じ id で違う語片が 13（ページは語彙を原本から取るので数えない）。
- **確かめ**: `format_check.py` は 9/9 が `KNOWN` のとおり（本物の書式は `<|start|>assistant` で終わり、ページは analysis のチャンネルを開いて終える: instruct の final と同じ形）、`--hf` は 9/9（9 つとも本物の ID と同じ）。`models-check`・`npm run types` は通る。
- **書かせ（`tests.yml`、run 41581446017、ページのエンジン・int8 9.66 GB・64 ビット、5 問 × greedy と種 1、2500 トークンまで）**: 10/10 が止まりで止まり、考えは 10/10 が自分で `<|end|>` で閉じた（25〜145 トークン、中央 83: 上限 3,584 に遠い）。繰り返し 0、`<unk>` 0。富士山（3,776 m）・東京・408（途中の式も）はどれも正しい。日本語の問いの答えは 3/3 が日本語（道具の「かなが 5 分の 1」は 2/3: 17 × 24 は考えが英語で式が多く 8〜9%、答えの本文は日本語）。考えはどの問いも英語。CI の 4 本で約 1.2〜1.4 tok/s。標本抽出は `thinking` のままにする（崩れなし）。
- **頭のトークン（`start_check`、1500 トークン）**: なし 11.862、`<|startoftext|>`（1）11.774（−0.74%）。instruct と同じ `bos: 1` のまま。
- **実物で分かった直し**: モデルは final のチャンネルの頭を語の前に空白を置いて書く（`<|end|><|start|> assistant<|channel|> final<|message|> 答え`: 復号した文で）。ページの `thoughtOf()` は頭を印と語に分けて空白を飛ばしながら外す形にした（`thought.ts` の `past()`、`page-parts-check` に 3 例）。エンジンは考えの終わりを閉じる印の ID で見るので触らない。
- **取りまとめ役の決め（訳）**: T.592.1 のレビューの「`THOUGHT_MARKS` と `MARKS` を突き合わせる検査」は足さない（印の種類は 2 つで、`answers.ts` は Python の表を読み、`page-parts-check` がページの表の両方の種類を踏む。種類を足す回に写しを並べて直す）。
- CI を回した。訳: 9 GB の GGUF の取得と書かせは手元で回せない（7B 以上は手元で動かさない）。ページの直しは手元の部品の検査で。
- **レビュー（Sonnet low、2026-11-03）**: 直すべきものは無し。取りまとめ役の決め（訳）: `past()` の例を足す点は、途中で切れた語と、頭が合って本文が続く例がもう 3 例にあるのでそのまま。`harmonyThinking` に「final は `<|return|>` で終わる」を書く点は、上のコメントがもう言う。`tests/fixed_outputs.py` の固定値は足さない（新しいアーキテクチャでも取り込みの道でもない: instruct と同じ Llama と GGUF の道で、手順 7 に当たらない）。
- **本番**（デプロイ run 41593406671、`models.yml` run 41594973900、Chromium）: 準備完了 93.7 秒、256 トークンを 3.9 tok/s（4 本）。
- **締め（T.592）**: 子タスク 2 つ（T.592.1・T.592.2）が本線に入り本番で準備完了した。llm-jp-4.1 8B thinking は自分の番号 T.304 で。
