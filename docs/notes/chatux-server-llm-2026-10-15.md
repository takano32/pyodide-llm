# T.516 chatux-server-llm から取り込める設計を見る（2026-10-15）

調べと表だけ。製品のコードは変えていない。相手のリポジトリ https://github.com/takano32/chatux-server-llm（sotokisehiro/chatux-server-llm の fork）を浅く clone して全部のテキストのファイルを読み、GitHub の API でブランチと履歴を、Hugging Face の API で LINE のモデルを見た。相手のコードは走らせていない（走らせなくても読み切れる大きさ: Python 3 ファイル・計 約 2.8 KB の本体）。

## まとめ

- **chatux-server-llm は、手元の PC で動かす日本語のチャットのサーバ**（2023-09〜2024-04、MIT）。FastAPI が 1 つの GET `/chat_api?text=…` に答え、エンジンは llama-cpp-python（ELYZA Llama 2 7B の Q4_K_M の GGUF）か CTranslate2（LINE japanese-large-lm 3.6B instruction-sft の int8）を起動の引数で選ぶ。画面は外の部品 chatux（riversun、MIT）を CDN から読むだけ。
- **pyodide-llm がまだ持っていない設計はほとんど無い**。会話の履歴も持たない（1 ターンずつ、答えは全部できてから 1 回で返す）、ストリーミングも止めるボタンも設定も無い。エンジンの抽象（`Engine` の `generate_text()`）は pyodide-llm の `Llama` と `generate()` より狭い。
- 取り込めるのは小さいものが 2 つ: **LINE の instruction-sft の書式（「ユーザー: …\nシステム: 」）で一覧に 1 項目**と、**文字列の止まり（stop の語）**の考え。ほかはサーバ・インストーラ・Docker で、方針 6（GitHub Pages で動くこと）の外か、pyodide-llm がもう持っている。
- chatux の「スマホでは画面いっぱいに重ねる」部品は、pyodide-llm のページがすでにスマホの幅で 1 画面のチャットなので、取り込む物ではない。

## 取り込む候補の勧め（順に）

1. **T.516.1（案）LINE japanese-large-lm 1.7B instruction-sft を一覧に**: chatux-server-llm の CTranslate2 の道が使う LINE の instruction-sft（3.6B）の、小さいほう。一覧の japanese-large-lm 1.7B（素のモデル）と同じ形（`gpt2`、sentencepiece）で、問いに答える版が入る。訪問者には「LINE の 1.7B が質問に答える」項目が 1 つ増える。ただし int8 約 1.9 GB で、T.511.4 の規則では「desktop only」（スマホの候補ではない）。仕事は小（項目・書式・`format_check`・書かせて読む）。
2. **T.516.2（案）文字列の止まり**: chatux-server-llm は llama-cpp-python に `stop=["Instruction:", "Input:", "Response:"]` を渡し、決まった語が出たら答えを切る。pyodide-llm の止まりはトークンの ID だけ（`stop_tokens`）。書式の印が普通の文字のモデル（llm-jp-3 instruct の「### 指示:」、LINE の「ユーザー:」）で、モデルが次のターンを自分で書き始めたときに切れる。訪問者には「答えの後ろに偽の次の問いが続く」ことが減る見込み（いまそれが起きている項目の数は未計測）。仕事は中（Python の `generate()` と、ページの表示の切り方。T.504 の表に行が無い）。
3. （低い）**T.516.3（案）実物のモデルの答えに語が入るかの試験**: chatux-server-llm の pytest は本物のモデルに「自己紹介してください。」と聞き、答えに「ELYZA」や「アシスタント」などの語があるかを見る。pyodide-llm には固定の 16 トークン（`fixed_outputs.py`）と 12 の問い（`answers.mjs`）があり、より強い。足すなら「問いに対して名前を言う」ような小さな意味の検査として `answers.mjs` に（仕事は小。勧めは低い）。

## 持ち主に聞くもの

- **(A) T.516.1 を足すか**（選択肢、勧めを先頭に）
  1. **足す（1.7B instruction-sft だけ）**: Q8_0 の GGUF がある（mmnga、1.89 GB）。素の 1.7B と同じ道で開く見込み。
  2. 足さない: スマホの大きさでなく、日本語で答える 1〜2B 級は一覧にもう多い（T.511 の Shisa 1.2B など）。
  3. 3.6B instruction-sft も足す: chatux-server-llm が使っていたもの。Q8_0 の GGUF は mmnga（`5f562eb5`）の 3,950,123,264 B で、int8 約 4.0 GB の見込み（未計測）: desktop only。
- **(B) T.516.2 を台帳に積むか**: 訪問者に見える動き（答えが途中で切れる）が変わる。先に「次のターンを書き始める項目」を数える（未計測）か、積まないか。
- **会話の履歴は聞かない**: chatux-server-llm も持っていない。T.504 で持ち主が選ばなかったまま。

## 1. chatux-server-llm が何か（確かめた事実）

読んだもの: `main` の先頭 `ab103206`（2024-04-07、Merge #22 `input_text`）。ブランチは `main`・`develop`（`main` より 28 進み、変わるのは `.github/workflows/workflow.yml`・`docker-compose.yml`・`docker/Dockerfile` だけ: CI と Docker の版上げ）・`fix-ruff-option`（同じ 3 つ）。上流 sotokisehiro/chatux-server-llm の最後は 2024-09-15（Merge #24 `fix-ruff-option`）。持ち主のコミットは型（`Engine` を ABC に、mypy）・CI（3 OS × Python 3.9〜3.12）・Docker の Ubuntu の版・引数の名前。

| 所 | 中身 |
|---|---|
| `main.py` | argparse（`--listen`・`--port`・`--maxspeed=ON` で `os.cpu_count()` 本・`--aiengine` 0 か 1・`--modelname`: **読まれていない**、エンジンは名前を自分で持つ）。FastAPI の `GET /chat_api?text=` が `engine.generate_text(text)` を呼び、`html.escape(res).replace("\n", "<br>")` を chatux の JSON（`{"output":[{"type":"text","value":…}]}`）で返す。`/` は `html/` を配る。uvicorn で `127.0.0.1:8001` |
| `engine/engine.py` | `Engine(ABC)`: `__init__(cpu_thread)` と抽象の `generate_text(text_input) -> str` だけ |
| `engine/llama_cpp.py` | `LlamaCppEngine`: llama-cpp-python の `Llama(model_path="models/ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf", n_threads, seed=1234)`。書式は Llama 2 の `[INST] <<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n{text} [/INST] ` を手で組む。`temperature=0.1, repeat_penalty=1.1, top_k=10000, stop=["Instruction:", "Input:", "Response:"]`（最大の長さは llama-cpp-python の既定） |
| `engine/ctranslate2.py` | `CTranslate2Engine`: `ctranslate2.Generator("line-sft")`（`install_linux.sh` が `ct2-transformers-converter --quantization int8` で LINE japanese-large-lm 3.6B instruction-sft から作る）と transformers の `AutoTokenizer(use_fast=False)`。書式は `"ユーザー :" + text + "システム :"`（カードの `ユーザー: {text}\nシステム: ` と空白と改行が違う）。`max_length=100, sampling_topk=20, sampling_temperature=0.5, repetition_penalty=1.1` |
| `html/index.html` | chatux（`https://riversun.github.io/chatux/chatux.min.js`）を `renderMode: "auto"`、`api.endpoint: "/chat_api"`（GET、JSON）、送信の文「送信」、placeholder「質問してください」、窓の題「CSLLM」、ボットの顔の画像で始めるだけ |
| `tests/test_llma_cpp.py`・`test_ctranslate2.py` | 本物のモデルを載せ「自己紹介してください。」の答えに語があるか（llama.cpp は「ELYZA」、CTranslate2 は「アシスタント」「AI」など 11 語のどれか） |
| `.github/workflows/workflow.yml` | push ごと、Ubuntu・Windows・macOS × Python 3.9〜3.12（macOS は 3.11 だけ）。flake8・black・mypy・ruff、モデルを `actions/cache` に（鍵は固定の名前）、pytest |
| `install_linux.sh`・`install_windows.ps1`・`docker/`・`start*.sh` | 依存の入れ方・モデルの取得（`wget`）・Docker（起動のたびに `git pull` する）。`start_linux.sh` は無い `ct2-main.py` を呼ぶ（古い名残り） |
| `README.md` | CPU だけで動く、最低の要件は Core i3 第 8 世代・メモリ 8 GB・ディスク 10 GB。ToDo に「各種パラメータの微調整」「設定ファイル」「LoRA」 |

**会話の扱い**: 1 ターンだけ（履歴は送らず、サーバも持たない）。system の文はエンジンに固定（ELYZA だけ）。標本抽出の設定はエンジンに固定で、画面から変えられない。**答えは全部できてから 1 回で返る**（ストリーミングなし、止めるボタンなし）。モデルの選び方は起動の引数で 2 つ、取得は手で `wget`、保存は手元のファイル。

**chatux（画面の部品）**: riversun/chatux（MIT、最後の push 2023-03-03）。README によれば PC では邪魔にならない小窓、スマホとタブレットではページに重ねて出す（`renderMode: 'auto' | 'pc' | 'mobile'`）。持ち主の README の画面（`img/img01.png`）は右下の丸いボタンで開く縦長の窓。

## 2. 取り込めそうなもの（項目ごと）

| 項目 | chatux-server-llm のどこ | pyodide-llm の今 | 取り込むと | 仕事 | 方針 |
|---|---|---|---|---|---|
| LINE japanese-large-lm instruction-sft | `engine/ctranslate2.py`（3.6B を CTranslate2 で） | 一覧は素の 1.7B だけ（`src/models/hf-japanese.js` の `hf-japanese-large-lm-1.7b`、BOS は `</s>`: AGENTS.md の T.249） | 問いに答える LINE の項目。1.7B instruction-sft は `gpt2`・apache-2.0・ゲートなし（API、`6817da05`）、GGUF は `mmnga/line-corp-japanese-large-lm-1.7b-instruction-sft-gguf`（`40b58a82`）の Q8_0 1,888,727,232 B。書式はカードの `ユーザー: {prompt}\nシステム: `、カードの標本抽出は温度 0.7・top-p 0.9・繰り返し 1.1。**スマホではない**（約 1.9 GB、T.511.4 の線 約 1.85 GB の上） | 小 | 触らない |
| 文字列の止まり | `engine/llama_cpp.py` の `stop=[…]` | 止まりは ID（`stop_tokens`、`engine/generation.py`）。書式の印が普通の文字の項目（`formats.js` の `LLM_JP_INSTRUCT`「### 指示:」など）は文字では切れない | 次のターンを書き始めた所で答えを切る。どの項目で起きているかは未計測 | 中（Python と表示。カーネルは触らない） | 触らない |
| 意味の小さな検査 | `tests/test_*.py`（答えに語があるか） | `tests/fixed_outputs.py`（greedy 16 トークンの固定）、`tests/answers.mjs`（12 の問い、かなの割合と繰り返し） | 名前を言えるかのような弱い検査が 1 つ増える。今の道具のほうが強い | 小 | 触らない |

## 3. 取り込まない方がよいもの（訳）

| もの | 訳 |
|---|---|
| FastAPI のサーバ・`/chat_api`・uvicorn・Docker・インストーラ | サーバが要る: 方針 6（GitHub Pages で動く）と目的（ブラウザの Pyodide だけ）の外 |
| llama-cpp-python・CTranslate2 をエンジンに | ネイティブのライブラリ。pyodide-llm は llama.cpp を wasm にして呼ぶ形を取らない（持ち主の指示、T.504）。CTranslate2 も Pyodide に無い |
| chatux の部品（CDN の `chatux.min.js`） | pyodide-llm のページ（`src/pages/index.astro`・`src/page/`）はもう吹き出し・ストリーミング・止めるボタン・設定のシート・モデルの選択を持ち、スマホの幅で 1 画面。chatux は答えを 1 回で受ける形（ストリーミングと止めるに合わない見込み: 部品の中は読んでいない）、外の script は jsDelivr の外の所（riversun.github.io）から読むことになる |
| `Engine` の抽象（`generate_text()` 1 つ） | pyodide-llm の `Llama`（部品の合成）と `generate()`（ジェネレータで 1 トークンずつ、`Sampling` の値）のほうが広い。実装が 2 つある所にだけパターンを置く決まり（AGENTS.md の物差し）にも、新しく足す所は無い |
| ELYZA Llama 2 7B（fast-instruct） | 7B でデスクトップだけ。T.511 の表で「開く見込み」と書いたもの。一覧には Llama-3-ELYZA-JP 8B が同じ system の文（「あなたは誠実で優秀な日本人のアシスタントです。」）で入っている（`formats.js` の `ELYZA`）。Llama 2 の `[INST] <<SYS>>` の形は `SWALLOW_MS` で持っている |
| 固定の標本抽出（温度 0.1・top-k 10000 など） | pyodide-llm は項目ごとにカードの値（`src/models/sampling.js`）と設定のシートを持つ。0.1 の根拠は相手のリポジトリに書かれていない |
| `html.escape` して `<br>` | pyodide-llm は `textContent` で書く（`draw.ts` の `message()`）: もう安全 |
| 3 OS × Python の版の CI・lint の組・モデルの `actions/cache` | pyodide-llm は `browsers.yml`（OS 5 種 × ブラウザ）・`tests/types.mjs`・`make models` のキャッシュをもう持つ。Python は Pyodide の 1 つの版で動く |
| モデルを起動の引数で選ぶ・手で `wget` | pyodide-llm は一覧・`?model=`・`?hf=`・フォルダから開く・OPFS の保存を持つ |

## 4. 確かめたことと、読んだだけのこと

- **確かめた（読んだ・API で見た）**: 相手の `main` の全部のテキストのファイル（`README.md`・`main.py`・`engine/*.py`・`html/index.html`・`tests/*.py`・`.github/workflows/workflow.yml`・`install_*`・`docker/*`・`start*.sh`・`requirements.txt`・`mypy.ini`）、ブランチ 3 つの差（GitHub の compare）、上流の最後のコミット、`img/img01.png`。LINE の 4 リポジトリと mmnga の GGUF（HF の API: リビジョン・ライセンス・ファイルの大きさ）、1.7B instruction-sft のカード（書式と標本抽出）。chatux の README の `renderMode` の行とライセンス。pyodide-llm の側: `src/page/composer.ts`・`answer.ts`・`draw.ts`・`index.astro` の要素、`formats.js` の `ELYZA`・`SWALLOW_MS`・`LLM_JP_INSTRUCT`、`hf-japanese.js` の LINE と ELYZA の項目、T.504 と T.511 の表。
- **読んだだけ・未確認**: 相手のコードを走らせていない。LINE 1.7B instruction-sft が今の道（`gguf_check.py tensors`・`format_check.py`）を通るか、書かせた文、3.6B の int8 の大きさ（GGUF は 3.95 GB を API で見ただけ）。chatux の部品の中身（ストリーミングを受けられるか）。pyodide-llm の項目で「次のターンを書き始める」ものがあるか（T.516.2 の前に数える）。
- **見ていない**: 上流の fork の網（ほかの fork）、相手の Issue と PR の議論、YouTube の動画、`img/sample01.mp4`。
