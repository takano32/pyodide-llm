# T535 一覧のモデルの名前の頭の「作り手」の書き方（勧め、2026-10-15、Fable max）

検討だけ。コードは変えていない。読んだもの: docs/notes/t523-maker-names-2026-10-15.md、docs/tasks/T523.md、今の一覧（144 項目。T523 の後に T343 の Spark-X2.5 が 4 つ、Gemma 4 が 2 つ、Shisa の 4 ビットが 1 つ増えている）、Hugging Face の組織と利用者のページ（`/api/organizations/<組織>/overview`・`/api/users/<人>/overview` の `fullname`）、作り手のカード（README）、GitHub の組織のページ、作り手のサイト。

## まとめ

- **144 項目のうち変えるのは 18、変えないのは 126。** 持ち主が問うた 2 つは、**Qwen の「Alibaba」はそのまま**、**Swallow の 3 つ（名前 4 つ）は「TokyoTech-LLM」→「Science Tokyo」**。
- ほかに直すもの 4 組: **EuroLLM は作り手を外す**（「EuroLLM」が計画の名で、llm-jp と同じ扱い）、**Agents-A1 は「Intern Science」→「Shanghai AI Lab」**（カードが「上海人工知能実験室の InternAgent チーム」と言う）、**Spark-X2.5 は「XHToken」を外す**（XHToken は handle で、組織の名は SparkLLM。「Spark」が名前にあるので Mistral と同じ扱い）、**個人 2 人は「Andrej Karpathy」「Atemoya」**（HF の表示名。handle は書かない）。
- 決め方は下の 5 つの決まり。今の 126 項目はこの決まりでそのまま通る（SB Intuitions・LINE・OpenBMB・TinyLlama は迷ったが変えない: 「迷いが残るもの」）。
- 長さ: 閉じた選択欄の約 32 字を越える名前は 40 → 37、最長は 61 → 52 字。**文面だけでは閉じた選択欄の切れは直らない**（作り手を別の欄に持って閉じた選択欄では出さない形は、別のタスクで持ち主の決め）。

## 決まり（この勧めの読み。T523 の決め方を 5 つに直した）

1. **作り手 = モデルを訓練した組織か人**（派生なら派生を作った組織。写し・量子化の作り手は入れない）。T523 のまま。
2. **書き方 = 作り手が自分のカードか組織のページで自分を呼ぶ名を、いちばん通りのよい短い形に。** チームや銘柄の上に会社・研究所があって、作り手自身がその名で自分を言うなら、会社・研究所のほうを書く: Qwen Team → 「Alibaba Cloud's」→ Alibaba、InternAgent Team → 上海 AI Lab、Swallow team → Science Tokyo、Meta Llama → Meta、IBM Granite → IBM、HuggingFaceTB・H4 → Hugging Face、Google DeepMind → Google。自分でその名を言わないものは、その会社の名のまま（SB Intuitions を SoftBank と書かない、OpenBMB を Tsinghua と書かない）。
3. **名前がもう作り手を言うなら重ねない**: llm-jp・Shisa・Underdog・RakutenAI・ELYZA・TinyLlama・DeepSeek・Mistral（T523）に、**EuroLLM と Spark** を足す。
4. **改名・統合した組織は、作り手自身の今のカードが使う名**: Swallow は Science Tokyo（2025 年以降のカードが Institute of Science Tokyo）。LINE はカードも HF の組織も「LINE」のままなので LINE（LY Corporation と書かない）。
5. **個人は、その人が HF で表示している名**: 本名なら姓名（Andrej Karpathy・Ellis Hoag）、筆名なら筆名（Atemoya）。handle（karpathy・ellishg・Takenoko12345678）は書かない。

## 変える項目（18）

| id | 今の名前 | 勧める名前 | 訳と根拠 |
|---|---|---|---|
| `hf-swallow-ms-7b-instruct` | TokyoTech-LLM Swallow-MS 7B instruct | **Science Tokyo** Swallow-MS 7B instruct | 決まり 2・4。作り手は東京科学大学（岡崎研・横田研）と産総研の Swallow チーム。TokyoTech-LLM は HF の handle で、大学は 2024-10 に東京工業大学から Institute of Science Tokyo になった。公式の英語の短い名は「Science Tokyo」（https://www.isct.ac.jp/en）。Swallow の自分のサイトも「Institute of Science Tokyo, together with … (AIST)」（https://swallow-llm.github.io/index.en.html）。このモデルのカード（2024-04）だけは「developed by TokyoTech-LLM」だが、同じチームの今のカードは Science Tokyo（下の 2 つ）。産総研は入れない（Google DeepMind を Google と書くのと同じ、下の「迷い」） |
| `hf-llama-3.1-swallow-8b-instruct` | TokyoTech-LLM Llama 3.1 Swallow 8B Instruct | **Science Tokyo** Llama 3.1 Swallow 8B Instruct | 同上。カードの Authors: 「Okazaki Laboratory, Institute of Science Tokyo」「YOKOTA Laboratory, Institute of Science Tokyo」「Artificial Intelligence Research Center, AIST」（https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5） |
| `hf-qwen3-swallow-8b-thinking` | TokyoTech-LLM Qwen3 Swallow 8B RL (thinking) | **Science Tokyo** Qwen3 Swallow 8B RL (thinking) | 同上。カードの Authors は「Swallow LLM」、計算機は「TSUBAME4.0 supercomputer at Institute of Science Tokyo」と「ABCI 3.0 provided by AIST」（https://huggingface.co/tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2） |
| `hf-qwen3-swallow-8b` | TokyoTech-LLM Qwen3 Swallow 8B RL (no thinking) | **Science Tokyo** Qwen3 Swallow 8B RL (no thinking) | 同上 |
| `hf-eurollm-1.7b-instruct` | UTTER EuroLLM 1.7B Instruct | EuroLLM 1.7B Instruct | 決まり 3。カードの「Developed by」は Unbabel・Instituto Superior Técnico・Instituto de Telecomunicações・University of Edinburgh・Aveni・University of Paris-Saclay・University of Amsterdam・Naver Labs・Sorbonne Université の 9 つで「Funded by: European Union」、本文は「The EuroLLM project」（https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct）。作り手は EuroLLM という計画そのもので、llm-jp（LLM-jp 計画の llm-jp-3）と同じ形。UTTER は資金の出た EU の計画の名（HF の組織の fullname「UTTER - Unified Transcription and Translation for Extended Reality」）で、訪問者には何も言わない |
| `hf-agents-a1-4b-thinking` | Intern Science Agents-A1 4B (thinking) | **Shanghai AI Lab** Agents-A1 4B (thinking) | 決まり 2。カードの system 文が「developed by InternAgent Team, Shanghai Artificial Intelligence Laboratory. 你是Intern-A1，一个由上海人工智能实验室的InternAgent团队开发的…」（https://huggingface.co/InternScience/Agents-A1-4B）、GitHub の組織の説明も「Shanghai Artificial Intelligence Laboratory」（https://github.com/InternScience）。Intern Science は HF の組織の表示名（上海 AI Lab の Intern の銘柄）で、Qwen を Alibaba と書くのと同じ向きで研究所の名に。短い形は論文が使う「Shanghai AI Lab」（公式は Shanghai Artificial Intelligence Laboratory、https://www.shlab.org.cn/） |
| `hf-agents-a1-4b` | Intern Science Agents-A1 4B (no thinking) | **Shanghai AI Lab** Agents-A1 4B (no thinking) | 同上 |
| `hf-spark-x2.5-1.7b-thinking` | XHToken Spark-X2.5 1.7B (thinking) | Spark-X2.5 1.7B (thinking) | 決まり 2・3。XHToken は handle で、HF の組織の fullname も GitHub の組織の名も「SparkLLM」、カードの引用は「SparkLLM Team」（https://huggingface.co/XHToken、https://github.com/XHToken）。組織の名 Spark(LLM) が名前の Spark-X2.5 にあるので、Mistral AI の Mistral 7B・DeepSeek の DeepSeek-R1 と同じく重ねない。T343 の「XHToken」は T523 の決め方（組織の表示名）からも外れていた。会社のサイト xhtoken.ai は今日は届かず未確認 |
| `hf-spark-x2.5-1.7b` | XHToken Spark-X2.5 1.7B (no thinking) | Spark-X2.5 1.7B (no thinking) | 同上 |
| `hf-spark-x2.5-4b-thinking` | XHToken Spark-X2.5 4B (thinking) | Spark-X2.5 4B (thinking) | 同上 |
| `hf-spark-x2.5-4b` | XHToken Spark-X2.5 4B (no thinking) | Spark-X2.5 4B (no thinking) | 同上 |
| `hf-qwen3.5-0.8b-japanese-sft-v2.5-thinking` | Takenoko12345678 Qwen3.5 0.8B Japanese SFT v2.5 (thinking) | **Atemoya** Qwen3.5 0.8B Japanese SFT v2.5 (thinking) | 決まり 5。HF の表示名は「Atemoya」（https://huggingface.co/Takenoko12345678）、カードが指す GitHub は atemoyacode-sudo/Pitcher。Takenoko12345678 は数字つきの handle で、その人が名乗る名ではない。9 字短くなり、一覧でいちばん長い名前が 61 → 52 字 |
| `hf-qwen3.5-0.8b-japanese-sft-v2.5` | Takenoko12345678 Qwen3.5 0.8B Japanese SFT v2.5 (no thinking) | **Atemoya** Qwen3.5 0.8B Japanese SFT v2.5 (no thinking) | 同上 |
| `stories260K` | Karpathy TinyStories 260K | **Andrej Karpathy** TinyStories 260K | 決まり 5。姓だけでは組織の名と区別がつかない（Ellis Hoag は姓名で書いてある）。README もすでに「TinyLlamas by Andrej Karpathy」（README.md:193、https://huggingface.co/karpathy/tinyllamas）。HF の表示名「Andrej K」は縮めた形なので本名の姓名に。7 字長くなるが 32 字に収まる |
| `stories15M` | Karpathy TinyStories 15M | **Andrej Karpathy** TinyStories 15M | 同上（31 字） |
| `stories42M` | Karpathy TinyStories 42M | **Andrej Karpathy** TinyStories 42M | 同上（31 字） |
| `stories15M-f32` | Karpathy TinyStories 15M (original) | **Andrej Karpathy** TinyStories 15M (original) | 同上（42 字。閉じた選択欄では前から「(original)」が切れていた） |
| `stories42M-f32` | Karpathy TinyStories 42M (original) | **Andrej Karpathy** TinyStories 42M (original) | 同上 |

置き場（実装する人へ）: `src/models/built.js` の 17・23・26・38・41 行（Karpathy）、`src/models/hf-japanese.js` の 112（EuroLLM）・190（Swallow-MS）・198（Llama 3.1 Swallow）行、`src/models/hf-families.js` の 24（Qwen3 Swallow）・70（Atemoya）・124（Agents-A1）・254・259（Spark）行。名前を文字で持つ試験（`tests/models-check.mjs` の Alibaba・Google Gemma 4・Hugging Face・Meta の行、`tests/ladder-check.mjs` の EleutherAI）はどれもこの 18 項目に当たらない（grep で見た）。`tests/format_check.py` が読む「(thinking)」はそのまま。記録（docs/notes/t523 の表、docs/tasks/T343.md の「XHToken Spark-X2.5」）は直さず、T535 の行を足す。

## 変えない項目（126）

| 作り手の書き方 | 数 | そのままでよい訳 |
|---|---|---|
| Alibaba（Qwen2.5・Qwen2.5 Coder・Qwen3・Qwen3.5） | 26 | **持ち主の問いへの答え: そのまま。** Qwen は自分を Alibaba Cloud のものと言う（GitHub の組織 QwenLM の説明「Alibaba Cloud's general-purpose AI models」https://github.com/QwenLM、Qwen2.5 の既定の system 文「You are Qwen, created by Alibaba Cloud.」、Qwen3.5 のライセンスの「Copyright 2026 Alibaba Cloud」）。「Alibaba」は Google DeepMind を Google と書くのと同じ短い形で、日本語の報道も「アリババの Qwen」。「Alibaba Cloud」は正確だが 26 項目で 6 字ずつ長くなり、「Qwen」だけでは持ち主が知らなかった事実（アリババのもの）を伝えない |
| llm-jp | 10 | 決まり 3。LLM-jp は NII の LLM 研究開発センターに置かれた計画の名で、名前の頭にある |
| Hugging Face（SmolLM2・SmolLM3・DistilGPT2・zephyr） | 9 | HuggingFaceTB（fullname「Hugging Face Smol Models Research」）・HuggingFaceH4・distilbert はどれも Hugging Face の組織。DistilGPT2 のカード「Developed by: Hugging Face」 |
| EleutherAI（Pythia） | 8 | 組織の名のまま |
| SB Intuitions（tiny-lm・sarashina2.2） | 7 | 決まり 2 の後半。カードは「trained by SB Intuitions」で、自分を SoftBank と言わない（下の「迷い」） |
| rinna | 5 | 組織の名「rinna Co., Ltd.」の短い形。小文字は組織の綴り |
| Liquid AI（LFM2・LFM2.5） | 5 | 組織の名のまま |
| Prism ML（Ternary Bonsai） | 5 | 組織の名のまま |
| Shisa（V2.1c LFM2 350M とその 4 ビット・V2.1 LFM2 1.2B・Llama 3.2 Shisa V2.1 3B・V2.1 Qwen3 8B） | 5 | 決まり 3。組織の名 Shisa.AI が名前にある（3B の名前はカードが決めた順） |
| Google（Gemma 3・Gemma 4） | 4 | 作り手は Google DeepMind、短い形は Google |
| CyberAgent（CAT-Translate・CAT-Thinking） | 4 | 組織の名のまま |
| LINE（japanese-large-lm） | 4 | 決まり 4。カード「trained by LINE Corporation」、HF の組織の fullname も「LINE」（検証済みの組織）。会社は 2023-10 に LY Corporation に統合されたが、作り手自身のカードも組織も LINE のまま。持ち主も「LINE も入れるべき」 |
| IBM（Granite 4.2） | 4 | 組織の名「IBM Granite」の会社の部分 |
| TokenRhythm（NeoHorse-1） | 4 | カード「fine-tuned by TokenRhythm」、会社のサイト tokenrhythm.ai |
| OpenBMB（MiniCPM5） | 4 | カードも組織も OpenBMB で、会社（ModelBest）や大学（清華）の名をカードが言わない（下の「迷い」） |
| OpenAI（GPT-2） | 4 | 置き場の openai-community は写し、作り手は OpenAI |
| Meta（Llama 3.1・3.2） | 3 | 組織の名「Meta Llama」の会社の部分 |
| DeepSeek（R1 Distill） | 3 | 決まり 3 |
| RakutenAI | 2 | 決まり 3（Rakuten Group） |
| Underdog（Saluki） | 2 | 決まり 3。カードの最後が「Built by Underdog.」（HF の組織 ConwayResearch の fullname は Conway だが、作り手が名乗るのは Underdog） |
| Mistral | 2 | 決まり 3（Mistral AI） |
| Ellis Hoag（TinyStories 3.5M） | 1 | 決まり 5。HF の表示名が姓名 |
| Stockmark | 1 | 組織の名「Stockmark Inc.」の短い形 |
| Sakana AI（TinySwallow） | 1 | カードの Authors は Sakana AI と Swallow Team の 2 つ。公開した側の Sakana AI を書く（産総研を Swallow に入れないのと同じ） |
| Llama-3-ELYZA-JP | 1 | 決まり 3 |
| TinyLlama | 1 | 決まり 3。組織の名も TinyLlama。大学（SUTD の StatNLP）はカードに無く、計画の名のほうが通っている（下の「迷い」） |
| Nous Research（Hermes 3） | 1 | カード「by Nous Research」 |

一貫性の確かめ: 同じ組織は全部の項目で同じ綴り（Alibaba 26・Hugging Face 9・Science Tokyo 4 など）。T523 で入れた名のうち、組織の表示名でなく handle だったのは TokyoTech-LLM・XHToken・Takenoko12345678 の 3 つで、今回全部直る。

## 長さ（スマホの閉じた選択欄）

- 閉じた `<select>` は `${name} — ${note}` を 1 行で出し、360 px では名前の頭から約 32 字しか見えない（T523 の実測）。
- 今回の 18 の前後: 32 字を越える名前は 40 → 37、最長は 61 → 52 字（Atemoya）。短くなるのは Spark 4 つ（−8）、Atemoya 2 つ（−9）、EuroLLM（−6）。長くなるのは Andrej Karpathy 5 つ（+7、260K は 32 字でぎりぎり、15M・42M は 31 字）と Shanghai AI Lab 2 つ（+1）。Science Tokyo は TokyoTech-LLM と同じ 13 字。
- **文面だけでは直らない**: 切れる名前 37 のほとんどは、作り手を入れる前から長い（Qwen3.5 0.8B Japanese SFT v2.5 (thinking) で 41 字）。閉じた選択欄で大きさと「(thinking)」を見せたいなら、作り手を項目の別の欄（`maker`）に持って、閉じた選択欄では名前の後ろ（または出さない）、開いた一覧とステータス行では頭に出す形になる。これはコードの変更で、持ち主の決め（採番していない）。

## 迷いが残るもの

1. **Alibaba か Alibaba Cloud か。** Qwen が自分で言うのは「Alibaba Cloud」（決まり 2 を厳密に当てればこちら）。「Alibaba」にしたのは、短い形を取る決まり（Google・Meta・IBM と同じ）と 26 項目の長さ。持ち主が正確さを取るなら「Alibaba Cloud」に替えるだけ（文字列 26 か所、`tests/models-check.mjs` の 67 行目の文も）。
2. **Science Tokyo か Tokyo Tech か。** 「Tokyo Tech」（東工大）のほうが日本の訪問者には今も通りがよいかもしれないが、大学の今の名は Institute of Science Tokyo で、作り手の今のカードもそれを使う（決まり 4）。Swallow-MS（2024-04、東工大の時代）にも今の名を当てる: 同じチームで、一覧の中で綴りを 1 つにするため。
3. **産総研（AIST）を入れない。** Swallow は東京科学大学と産総研の共同で、カードは両方を書く。「Science Tokyo & AIST」は 20 字で長すぎ、先頭に立つ大学だけにした（Gemma を Google DeepMind でなく Google、TinySwallow を Sakana AI だけ、と同じ）。訪問者に産総研の名も見せたいなら note に書く案（文面は持ち主）。
4. **SB Intuitions か SoftBank か。** Qwen を Alibaba と書くのと同じ向きなら SoftBank だが、SB Intuitions は別の会社（ソフトバンクの子会社）で、カードも組織も SB Intuitions としか言わない。決まり 2 の「自分でその名を言うなら」で線を引いた。LINE を LY Corporation にしないのも同じ線。
5. **Atemoya か Takenoko12345678 か。** 表示名は本人が後で変えられる（handle は repo に残る）。それでも、名乗っている名を書くのが個人への礼で、Karpathy・Ellis Hoag と同じ決まりになる。一覧の名前は文字列だけなので、変わっても壊れるものは無い。
6. **Andrej Karpathy か Karpathy か。** 「Karpathy」で通じる人には +7 字が無駄に見える。姓名にしたのは、知らない訪問者に「人」と分かること、Ellis Hoag と README との一貫。持ち主が短さを取るなら「Karpathy」のまま（この 5 項目だけ戻す）。
7. **Shanghai AI Lab か Intern Science か。** 研究所の名のほうが「誰が作ったか」を言うが、Intern（InternLM）の銘柄を知る人には Intern Science のほうが通る。Qwen を Alibaba にした持ち主の向き（会社・研究所が分かるほうがよい）に合わせた。
8. **OpenBMB。** MiniCPM は清華大学 NLP 研と ModelBest（面壁智能）のもので、OpenBMB はその公開の名。カードが会社も大学も言わないので OpenBMB のまま。Tsinghua と書く案もある。
9. **TinyLlama。** 作り手は SUTD（シンガポール工科デザイン大学）の StatNLP だが、カードは大学を言わず、TinyLlama の名のほうが通っている。Swallow に大学の名を入れて TinyLlama に入れないのは、カードが大学を名指すかどうかで分けた。
10. **EuroLLM に Unbabel を入れる案。** 9 つの機関の先頭は Unbabel（会社）だが、カードは 9 つを並べて 1 つを立てない。計画の名で足りると読んだ。
