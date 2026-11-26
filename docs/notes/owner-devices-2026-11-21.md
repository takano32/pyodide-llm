# 持ち主の端末でまとめて見る手順（2026-11-21、T311）

TASKS.md の「持ち主の端末でまとめて見るもの」の一覧（39 項目）と、済んだタスクの記録に残っている端末の確かめ（T313・T296・T270）を、端末ごとに 1 回で済む順に並べ直したもの。一覧の項目はそのまま残してあり、どの手順で答えが出るかは最後の対応表にある。

URL の旗と `/benchmark/` の節やボタンの名前は、2026-11-21 の本線（d086f927）のコードで確かめた（`src/page/parameters.ts` の `NAMES`、worker が読む `?gpuTest=`・`?without=`・`?bits=`・`?threads=`、`src/bench/sections.ts` の `TITLES`、`src/pages/benchmark.astro` のボタン）。モデルの ID は `src/models.ts` の一覧から写した。

## はじめに: 共通の決まり

- **サイトの頭**: `https://takano32.github.io/pyodide-llm/`（下では「頭」と書く）。モデルは `頭?model=<ID>` で開く。計測のページは `頭benchmark/`。
- **順の考え方**: (1) サイトのデータを消してから見るもの → (2) `/benchmark/` → (3) 小さいモデル → (4) 中くらい → (5) GB 級のモデル → (6) タブが落ちうるもの（`/benchmark/` の Page memory）の順。**GB 級の取得は Wi-Fi で、充電しながら**。PC は保存の空き（ディスク）を先に見る: 27B は 1 つで約 7〜8 GB を保存する。
- **各節の見出しの所要の時間は推測**（未計測。取得の回線と端末の速さで大きく変わる）。
- **★の付いた手順**は、止まっているタスクの決めに要るもの。時間が足りなければ★だけ見て、残りは次の機会でよい。
- **「CPU だけ」で比べたいとき**は `&threads=1` を足す（1 本だと GPU を使わない: 理由「one thread was asked for」）。速さは 1 本のぶん遅いので、比べるのは文の質だけ。GPU を切る旗は無い。効いたかは [状] が「prompts on the CPU (one thread was asked for)」と言うかで分かる（コンソールにも `gpu:` で始まる同じ訳の行が出るはず）。GPU だけに置かれるモデルにはこの旗は効かない（`&threads=1` を付けても [状] が「prompts and answers on WebGPU」のままで、「N threads」は付かない）。
- `?gpuTest=on`・`?gpuTest=only` は試験の旗（ソフトウェアの GPU も GPU と見なし、速さを比べずに GPU を使う）。本物の GPU のある端末でだけ使う。ステータス行に「(gpuTest)」と出るのは `on` のときだけで、`only` では何も付かない。

### 写すもの（記号）

| 記号 | 写すもの | どうやって |
|---|---|---|
| [状] | 準備完了のステータス行 | 画面の上のステータス行の文をそのまま（長押しで選んで写すか、スクリーンショット） |
| [答] | 答えの下の行（tok/s と本数）と、答えが読めるか | 行をそのまま。答えは「読める・崩れた・止まらない」のどれかと、気になれば頭の数行 |
| [コ] | コンソールの全部 | PC は F12 → Console → 右クリック → 「Save as…」でファイルに。Android は PC の Chrome で `chrome://inspect` を開き USB でつないだスマホのタブの「inspect」から同じく保存。iPhone は Mac の Safari の「開発」メニューから。保存したファイルをそのまま渡してくれればよい（`gpu:`・`threads:`・`memory:` で始まる行を読む） |
| [報] | `/benchmark/` の報告 | 終わったら下の「Report them as an issue」を押す（長いとクリップボードに全文が入るので、開いた Issue に貼る）。Issue にしたくなければ「Copy the results」で写す |
| [目] | 目で見たこと | タブが落ちた・読み直された・警告の文が出た・止まらなかった、など |

[コ] がつなげない日は、[状] と [答] だけでも多くの項目に答えが出る。[コ] が要る手順にはそう書いた。

### 使う問い（コピーして送る）

- **短い問い J**: `日本でいちばん高い山はどこですか？`
- **流行りの問い**: `これからの流行りを3つ挙げてください。`
- **英語の問い E**: `What will be popular next? Name three things.`
- **長い問い L**（64 トークンを越え、200 トークンほど）:
  `次の文章を読んで、大事な点を三つにまとめてください。日本の鉄道は、明治時代の初めに新橋と横浜の間で開業したのが始まりとされています。その後、全国に路線が広がり、人や物を運ぶ大きな力になりました。戦後には新幹線が登場し、東京と大阪の間の移動時間が大きく縮まりました。いまでは、都市の通勤電車から地方のローカル線まで、さまざまな列車が走っています。一方で、人口が減っている地域では利用者が少なくなり、路線を残すかどうかが話し合われています。観光列車を走らせたり、駅を地域の集まる場所として使ったりする工夫も増えています。これからの鉄道には、環境にやさしい乗り物としての役割も期待されています。`
- **長い答えの問い**: `日本の歴史を、古い時代から順にできるだけ詳しく説明してください。`

### 前の一覧から変わっていること（読み違えないために）

一覧の項目には書いた時の見込みの文が残っている。いまの本線では次のものが替わっているので、**ステータス行は見込みの文と比べず、そのまま写してくれればよい**。

- 3 値のモデル（Ternary Bonsai）はもう GPU に置ける（T232）。T230・T246 の「prompts on the CPU (ternary weights are not on the GPU yet)」は古い。
- Qwen3.5 と LFM2 の層も GPU に置ける（状態を持つ層を GPU に）。T229 の「linear-attention layers are not on the GPU yet」、T260 の「convolution layers are not on the GPU yet」は古い（Qwen3.5 で特別な形のものだけ前の文が残る）。
- Gemma 4 の答えのステップは T334.2 から GPU に置けるようになった。T334.1 の「a Gemma 4's answers are not on the GPU yet」は古い。置き場は T682（2026-11-26）から: メモリを 8 と言う端末で **E4B は GPU だけ**（CPU にモデルを持たない）。**E2B は保存（OPFS）に層ごとの埋め込みの表を置ける訪問では CPU と GPU の両方**（プロンプトのブロックも答えのステップも、測って速いほう）、**置けない訪問では GPU だけ**（表は CPU のメモリに）。下の A-8 と B-9。
- 端末の鍵は T367 の後も T379.2・T261.4.1・T334・T344 で何度か動いた。T367.3.2・T367.5 の「前に覚えた形を選び直さない」は、いまはもう見られない（鍵が動いたので最初の 1 回は必ず選び直す）。残りの見方（警告が出ない・tok/s が前と同じ桁・`/benchmark/` の GPU の節が通る）は A-2・A-5・B-1 で見る。

---

## A. 持ち主の Android（Xiaomi 13T Pro、Chrome）: 全 11 手順、約 3〜4 時間（取得込みの見込み）

最適にする的の端末なので、いちばん先に。前の準備: Wi-Fi・充電・できれば PC と USB でつないで [コ] を取れるように。Chrome のバージョン（設定 → Chrome について）を 1 行メモ。

**A-1 ★ サイトのデータを消して、初めての訪問（T557・T129 (a)）**
- Chrome の設定 → サイトの設定 → すべてのサイト → `takano32.github.io` → データを消す。
- 開く: `頭`（モデルを名指さない: 既定の Shisa V2.1c LFM2 350M、379 MB の取得が始まる）。
- 取得の途中で、(i) 別のアプリに 1 分切り替えて戻る。取得の MB を切り替える前と戻った後にメモ。「got nothing from the network for 30 seconds」が出たかを見る。(ii) 続けて画面を消して 1 分待ってから点ける。MB の前と後、ページが読み直されたかを見る。
- (i) は、ページを開いて取得が始まったらすぐ（Pyodide の読み込みの間に）やる。379 MB は Wi-Fi なら 1 分たたずに終わりうるので、(i) か (ii) の間に取得が終わってしまったら、その分は A-8 の 1（Gemma 4 E2B、約 5 GB）の取得の途中でやる。
- 写す: [目]（MB の前と後 × 2 回、出た文、読み直されたか）。

**A-2 ★ 既定のモデルのまま（T545・T240・T367.5・T379.2）**
- 準備完了の [状]。
- 長い問い L を送る: 最初のトークンが出るまでの秒（目で数えてよい）と [答]。
- 続けて長い答えの問いを送り、最後まで書かせる。
- 写す: [状]、[答] × 2、[コ]（`threads:` の行に「(timed while the GPU got ready: searched again once it is)」が付いた行のあと、付かない行が出るか。`gpu:` の行。警告の行が無いか）。

**A-3 ★ 4-bit の Shisa（T547.2.3・T547.2.4 → T578・T579.2 の決め）**
- 開く: `頭?model=hf-shisa-v2.1c-lfm2-350m-q4`
- A-2 と同じく長い問い L と短い問い J。
- 写す: [状]、[答] × 2、長い問いの最初のトークンまでの秒。A-2 の int8 と比べて決める（答えの tok/s が int8 以上なら T579.2 は入れない、長い問いを読む速さが int8 の 0.9 倍以上なら T578 は入れない）。

**A-4 llm-jp-3 150M（T223・T239・T243）**
- 開く: `頭?model=llm-jp-3-150m`
- 短い問いを 2 つ、そのあと長い問い L を 1 つ。
- 写す: [状]、[答] × 3、[コ]（`threads:` の行の本数と比べの行。「not finite numbers」が出ないこと）。

**A-5 ★ `/benchmark/` の Run all（T624.2・T241・T219 (2)・T224・T296・T313・T270・T555・T225 (a)・T379.2・T367.3.2・T156 (c)・T263）**
- 開く: `頭benchmark/` → 「Run all」を押す（モデルは既定の llm-jp-3 150M のまま。ほかの箱は触らない）。終わるまで画面を前に出したまま（画面は自動では消えない）。所要は未計測（10〜30 分の見込み）。
- 写す: [報]。読むのは主に、GPU の節の「An int8 matrix × vector」の「llama.cpp mul_mat_vec (engine)」「ORT DP4A small M (engine)」の行（T624.2）、`a layer, DP4A, …` が ok か（T241・T225 (a)）、「Quantizing the vector」の µs（T241）、「the sampling alone」の平らな logits の 1 ワークグループの ms が 6.71 以下か（T219 (2)）、「A token's attention alone, by the positions it reads」の 128・1024・4096 位置（T224・T296）、「Ternary weights」の表と「The codes made signed by a table against the engine's form」の 2 つの形の比（T313・T270）、CPU の節の「Kernels of ternary weights」（T270）、モデルの節の表の下の「The last round ran the first one's switches again at N%」（T555）。
- T263: `/benchmark/` の Run all の報告に GPU の節の「NaN and infinities」の表が入る（新しい手順は要らない: GPU の節の 2 段目で、数秒もかからない見込み）。読むのは「failed」の行の有無（エンジンの印が頼る場合で、この端末が NaN か無限大を運ばなかったもの）と、要約の「GPU, NaN and infinities」の 1 行（運ばなかった場合があるときだけ出る）。

**A-6 小さいモデル（〜0.8 GB）（T509・T261.4.1・T260・T236・T229・T245 (c)・T226・T251 (b)・T253 (b)）**
1. Gemma 3 270M `頭?model=hf-gemma-3-270m`: 短い問いを 1 つ送り最初のトークンまでの秒、続けて 2 つ目の問いの秒（T509）。答えの下の行が「… temp 0.7」のとき（種が出るのは温度のあるときだけ: 出なければ `頭?model=hf-gemma-3-270m&temperature=0.7`）、種のボタンを押して内訳が開かないか・要約を押して開くか・種が要約の下の行に回って読めるか（T372）。長い問い L を 2 回（T261.4.1: CPU が勝ってよい）。[状]・[答]・[コ]（`gpu: a block of 64 tokens: … ms on the GPU, … ms on the CPU`）。
2. Gemma 3 1B `頭?model=hf-gemma-3-1b`: 長い問い L を 2 回。[状]・[答]・[コ]（T261.4.1）。
3. LFM2.5 350M `頭?model=hf-lfm2.5-350m`: 短い問い J と流行りの問い。[状]・[答]・[コ]（`threads:` の本数: T260）。
4. Qwen3.5 0.8B `頭?model=hf-qwen3.5-0.8b`: 日本語で 2 つ（J と流行りの問い）。[状]・[答]・[コ]（`threads:`）。同じ 2 つを `頭?model=hf-qwen3.5-0.8b&without=relaxed`（8 ビット）でも答えさせ、崩れや繰り返しが片方だけに出るか読み比べ（T236・T229・T245 (c)）。
5. Qwen3.5 0.8B の考える形 `頭?model=hf-qwen3.5-0.8b-thinking`: 1 つ聞いて、1 分たっても `</think>` が出なければ停止を押す。止まるか・画面が固まらないか・その間の tok/s（T236）。
6. Qwen3 0.6B `頭?model=hf-qwen3-0.6b`（または Qwen2.5 0.5B `頭?model=hf-qwen2.5-0.5b-instruct`）: 1 つ。[状]（答えが GPU か CPU か: T226）。
7. GPT-2 medium `頭?model=hf-gpt2-medium`（英語の文が続くか）、Pythia 14M `頭?model=hf-pythia-14m`（開くか）、SmolLM2 135M `頭?model=hf-smollm2-135m` と 360M `頭?model=hf-smollm2-360m`（T251 (b)）。[状] と [目] だけ。
8. MiniCPM5 1B `頭?model=hf-minicpm5-1b`: 英語の問い E。[状]・[答]（T253 (b)）。

**A-7 中くらい（約 0.5〜2 GB）（T235・T230・T231・T232・T313・T344・T249 (b)・T247）**
1. ★ Ternary Bonsai 1.7B `頭?model=hf-ternary-bonsai-1.7b`: 長い問い L を 2 つ、短い問い J を 2 つ。[状]・[答]・[コ]（行列の形の名前と ms、「a token by … ternary (a step of a run of 4: … N ms」の N、「a whole block: 16 tokens A ms, 64 tokens B ms」、`wrong:` が無いこと、「no packed int8 dot here」が出ないこと）。N が 49 を切れば T313 の覆す条件に当たる（T232 (f)、T313）。同じ J を `頭?model=hf-ternary-bonsai-1.7b&bits=8` でも答えさせ tok/s と読みやすさを比べる。そのあと `&bits=8` 無しにもう 1 度開き、準備完了が数秒のまま（取り直しにならない）かを見る（T230・T231 (b)(c)(f)）。
2. Spark-X2.5 1.7B `頭?model=hf-spark-x2.5-1.7b`: 長い問い L を 2 回。[状]・[答]・[コ]（`gpu: a block of 64 tokens …` と `gpu: a step of a generation …`: T344）。
3. llm-jp-3 1.8B instruct3 `頭?model=hf-llm-jp-3-1.8b-instruct3`: 1 つ。[状]・[答]・[目]（タブが落ちないか: T249 (b)）。
4. Qwen3.5 2B `頭?model=hf-qwen3.5-2b`: 1 つ。開けるか・メモリの警告が出るか・[状]（T247）。
5. GPT-2 large `頭?model=hf-gpt2-large` と XL `頭?model=hf-gpt2-xl`: 開くか（「desktop only」の線: T251 (b)）。[状]・[目]。

**A-8 GB 級（Wi-Fi で）（T246・T571・T337・T130・T226・T210・T249 (b)・T253 (b)・T255・T258・T334.1・T334.2・T334.3・T344）**
1. ★ Gemma 4 E2B `頭?model=hf-gemma-4-e2b`: 長い問い L を 1 つ。[状]（両方に置いたときは「… 4 threads, prompts and answers on WebGPU」のようにスレッドの本数が付く。GPU だけのときは「prompts and answers on WebGPU」で本数が付かない）・[答]・[コ]（`gpu:` の「a whole block: 16 tokens … ms, 64 tokens … ms」、`gpu: a token by … (a step of a run of 1: …)` の ms、`gpu: both copies would not fit` か `kept:` で始まる行があればそれも）（T334.1・T334.2・T682）。
2. ★ 同じ E2B をもう一度開く（2 回目の訪問: 保存から読む）: 長い問い L を 1 つ。[状]・[答]・[コ]（1. と同じ）。1. と [状] が違えば（片方だけスレッドの本数が付く）それも書く（持ち主の決め（2026-11-26）: E2B は両方に置いて測る。保存が使えればどちらの訪問も本数が付き、保存が使えなければ GPU だけで本数が付かない）。
3. Ternary Bonsai 4B `頭?model=hf-ternary-bonsai-4b` と 8B `頭?model=hf-ternary-bonsai-8b`: 1 つずつ日本語で。[状]・[答]・[目]（8B でタブが落ちないか）（T246）。
4. Qwen3.5 4B `頭?model=hf-qwen3.5-4b` と Agents-A1 4B `頭?model=hf-agents-a1-4b`: 同じ短い問い J を送り、最初のトークンまでの秒の差（T337）。Qwen3.5 4B が GPU だけ（[状] が「prompts and answers on WebGPU」で「N threads」が付かない）で準備完了まで行くか（T571）。余裕があれば Qwen3.5 9B `頭?model=hf-qwen3.5-9b` も（T571）。
5. Spark-X2.5 4B `頭?model=hf-spark-x2.5-4b`: 1 つ。[状]・[答]（GPU だけに替わって tok/s が前より落ちていないか: T344）。
6. Qwen2.5 3B `頭?model=hf-qwen2.5-3b-instruct`: 長い答えの問いを送り、答えの下の位置が 2048 を越えるまで書かせる（途中で止まったら「続けて」と送る）。メモリ不足にならないか、[コ] に `memory: the browser gave a shared memory of …` が出たか（T130・T226）。
7. Llama 3.2 3B `頭?model=hf-llama-3.2-3b-instruct`: 1 つ。[状]（GPU だけか）・[答]（T210）。8B `頭?model=hf-llama-3.1-8b-instruct` が入るかも（T210: 開けるか・タブが落ちないか。入らなければ CPU に読み直すまでの秒）。
8. Llama 3.2 Shisa V2.1 3B `頭?model=hf-shisa-v2.1-llama3.2-3b`（GPU だけの見込み: T249 (b)）、Granite 4.2 3B `頭?model=hf-granite-4.2-3b`（開くか: T253 (b)）、SmolLM3 3B `頭?model=hf-smollm3-3b`（英語の問いを 2 つ。8 と言う端末では GPU だけになる見込みで、そのときは `&threads=1` を付けても CPU にならず比べられない: [状] が「prompts and answers on WebGPU」で「N threads」が付かなければ、GPU のまま読める文かだけ見る。GPU と CPU の両方に置く端末では同じ問いを `&threads=1` でも答えさせ、GPU の側だけ文が崩れないか: T255）。どれも [状]・[答]。
9. （任意、最後）Phi-3 mini `頭?model=hf-phi-3-mini-4k-instruct`・Phi-4 mini `頭?model=hf-phi-4-mini-instruct`（取得 7.6・7.7 GB。答えが `<|end|>` で止まるか: T258）、Ternary Bonsai 2 27B `頭?model=hf-ternary-bonsai-2-27b`（開けるか・メモリの警告が出るかだけ: T233・T237。警告が出ずに 5.95 GB の取得が始まったら止めてよい）。

**A-9 ★ `/benchmark/` をもう 2 回（T555）と Page memory（T173）**
- 端末が温まった状態で `頭benchmark/` の Model の節の「Run」を 2 回押し、表の下の N% を 2 回分写す（T555）。
- 最後に Page memory の節の「Run」を押す（タブが落ちることがある。画面は前に出したまま）。写す: [報] か画面の文（T173）。

**A-10 機内モードで 2 回目の訪問（T129 (b)）**
- 機内モードにしてから `頭` を開く。開いて答えるか。[目]。

**A-11（任意）遅い回線（T129 (c)）**
- 速度制限中の回線（約 128 kbps）があるときだけ: `頭` を開き、準備完了まで行くか、フォールバック（1 コア）に入らないか。[状]・[目]。

---

## B. PC（Windows の Chrome、NVIDIA）: 全 10 手順、約 4〜6 時間（GB 級の取得が多い）

前の準備: 機械のメモリ（GB）とグラフィックカードのメモリ（GB）、Chrome のバージョンを 1 行メモ。F12 でコンソールを開いておく。保存の空きを 60 GB ほど。大きいモデルは見終わったら About の一覧から保存を消してよい。

**B-1 ★ `/benchmark/` の Run all を 2 回目まで（T271.5 の中の T241・T225・T227・T226 (4)・T624.2・T296・T313・T270・T379.2・T367.3.2）**
- 開く: `頭benchmark/` → 「Run all」。終わったら [報]。
- 続けて GPU の節の「Run」をもう 1 回押し、2 回目の値も [報]（T225 (b): DP4A の層が前の 5 倍遅かった件。「The whole layer by the GPU's own clock」、llama.cpp の層の行、「Unsteady: …」、「what a token costs besides the weights」を読む）。
- 読むもの: `a layer, …` と `tokens on the GPU` が ok か、各行の `stages:` の「K and V N to the nearest float16, M toward zero, 0 away from it, 0 farther」の M（T225 (a)）、報告の頭の Warnings の並び（T227）、機能の節の `deviceMemory`（T226 (4)）、ほかは A-5 と同じ。

**B-2 llm-jp-3 150M（T223・T239・T225 (c)・T243）**
- 開く: `頭?model=llm-jp-3-150m`。短い問いを 2 つ、長い問い L を 1 つ。
- 写す: [状]・[答]・[コ]（`threads:` の行が 8 のままか 2 になるか、比べの行の数、`gpu: a token by …` の行に `wrong:` が無いか、「not finite numbers」が無いか）。

**B-3 ★ 既定の Shisa と 4-bit の Shisa（T545・T547.2.3・T547.2.4）**
- `頭` と `頭?model=hf-shisa-v2.1c-lfm2-350m-q4` で A-2・A-3 と同じ（長い問い L と短い問い J）。[状]・[答]・最初のトークンまでの秒。

**B-4 小さいモデル（T236・T229・T260・T261.4.1・T344・T235・T232・T313）**
1. Qwen3.5 0.8B `頭?model=hf-qwen3.5-0.8b`: 日本語で 2 つ。[状]・[答]。考える形 `頭?model=hf-qwen3.5-0.8b-thinking` で 1 つ（T236・T229）。
2. LFM2.5 350M `頭?model=hf-lfm2.5-350m`: 2 つ。LFM2.5 1.2B JP `頭?model=hf-lfm2.5-1.2b-jp`: 1 つ（T260）。
3. Gemma 3 1B `頭?model=hf-gemma-3-1b`: 長い問い L を 2 回。[コ]（T261.4.1）。
4. Spark-X2.5 1.7B `頭?model=hf-spark-x2.5-1.7b`: 長い問い L を 2 回。[コ]（T344）。
5. ★ Ternary Bonsai 1.7B `頭?model=hf-ternary-bonsai-1.7b`: 長い問い L を 2 つ・短い問いを 2 つ。[状]・[答]・[コ]。`&bits=8` でも同じ J（A-7 の 1. と同じ: T235・T230・T231・T232・T313）。
6. MiniCPM5 1B `頭?model=hf-minicpm5-1b`: 英語の問い E（T253 (b) は Android だが PC でも可。任意）。

**B-5 3〜4B 級（T226・T247・T253 (a)・T251 (a)・T255・T246・T232 (c)）**
1. Qwen3 4B `頭?model=hf-qwen3-4b`: 1 つ。[状]（「prompts and answers on WebGPU」か）・[答]（T226）。
2. Qwen3.5 2B `頭?model=hf-qwen3.5-2b` と 4B `頭?model=hf-qwen3.5-4b`: 日本語で 1 つずつ。準備完了の秒・[状]・[答]。考える形 `頭?model=hf-qwen3.5-4b-thinking` で 1 つ: 考えが終わって答えまで行くか（T247）。
3. Granite 4.2 3B `頭?model=hf-granite-4.2-3b`: 短い問い J を 3 回（富士山と答える割合）。[状]・[答]。そのあと考える形 `頭?model=hf-granite-4.2-3b-thinking` を開いてから、また `頭?model=hf-granite-4.2-3b` に替え、2 つ目の準備完了が数秒か（T253 (a)）。
4. Hermes 3 Llama 3.2 3B `頭?model=hf-hermes-3-llama-3.2-3b`: 英語の問い E を 2 回（「1. Virtual Reality …」と整うか）。Qwen2.5 Coder 3B `頭?model=hf-qwen2.5-coder-3b-instruct`: `Write a Python function that reverses a string.`（T251 (a)）。
5. SmolLM3 3B `頭?model=hf-smollm3-3b`: 英語の短い問いを 2 つ。同じ問いを `&threads=1` でも（GPU の側だけ崩れないか）。考える形 `頭?model=hf-smollm3-3b-thinking` で最初のトークンまでの秒（T255）。
6. Ternary Bonsai 4B `頭?model=hf-ternary-bonsai-4b` と 8B `頭?model=hf-ternary-bonsai-8b`: 1 つずつ。[状]・[答]・[コ]・[目]（8B でタブが落ちないか）（T246・T232 (c)）。

**B-6 7〜9B 級（Wi-Fi で、1 つ 5〜10 GB）（T249 (a)・T251 (a)・T253 (a)・T247）**
- どれも 1 つ聞いて、準備完了の秒・[状]（GPU だけか、カードに入らず CPU で読み直したか。読み直したならそこまでの秒）・[答]・[コ]（「loaded again on the CPU」の有無）。
1. ★ Shisa V2.1 Qwen3 8B `頭?model=hf-shisa-v2.1-qwen3-8b` と Qwen3 8B `頭?model=hf-qwen3-8b`: 流行りの問い（答えの頭: Shisa は「2024年以降の流行りとして注目される3つのトレンドは…」、Qwen3 8B は「もちろんです！以下は、2024年以降に…」で始まり、途中で途切れず「坊主と」を繰り返さないか）（T249 (a)）。Ternary Bonsai 8B も同じ問いで（B-5 の 6. で済ませてよい）。
2. CAT-Thinking 8B `頭?model=hf-cat-thinking-8b`、Llama-3-ELYZA-JP 8B `頭?model=hf-llama-3-elyza-jp-8b`、CAT-Translate 3.3B `頭?model=hf-cat-translate-3.3b`、llm-jp-3 3.7B `頭?model=hf-llm-jp-3-3.7b-instruct3`、Llama 3.2 Shisa V2.1 3B `頭?model=hf-shisa-v2.1-llama3.2-3b`（T249 (a)）。
3. DeepSeek-R1 Distill Qwen 7B `頭?model=hf-deepseek-r1-qwen-7b`: `What is 17 times 24? Think first.`（`</think>` の後に 408 と答えるまでの秒）。Llama 3.1 8B Instruct `頭?model=hf-llama-3.1-8b-instruct`、DeepSeek-R1 Distill Llama 8B `頭?model=hf-deepseek-r1-llama-8b`（T251 (a)）。
4. Granite 4.2 8B `頭?model=hf-granite-4.2-8b`（T253 (a)）、Qwen3.5 9B `頭?model=hf-qwen3.5-9b`（メモリに余裕があれば: T247）。

**B-7 Phi（取得 7.6・7.7 GB）（T258）**
- Phi-3 mini `頭?model=hf-phi-3-mini-4k-instruct` と Phi-4 mini `頭?model=hf-phi-4-mini-instruct`（日本語で）: 1 つずつ。準備完了の秒・[状]（GPU の準備が通ったか、CPU に戻ったならその訳）・[答]（`<|end|>` で止まるか。`<|user|>` などの印が続けば止まりの誤り）。

**B-8 27B の 3 つ（1 つ 6〜8 GB、メモリ 16 GB の PC で）（T547.3・T271.5・T233・T233 のレビュー (b)(c)・T237・T238）**
1. ★ Underdog Saluki 27B `頭?model=hf-underdog-saluki-27b`: 1 つ。GPU だけに置かれるか、GPU の準備が終わるまでの秒、[状]（「prompts and answers on WebGPU」か）、[答]、[コ]（ステップの ms、「loaded again on the CPU」か「the CPU as /benchmark/ measured it was faster here than this model on the GPU alone」が出て CPU に戻ったか）、[目]（タブが落ちないか）（T547.3）。
2. Ternary Bonsai 2 27B `頭?model=hf-ternary-bonsai-2-27b`: 流行りの問い。準備完了の秒（取得 5.95 GB）・[状]（GPU だけか: 「prompts and answers on WebGPU」で「N threads」が付かない）・[答]（tok/s と本数）・[コ]（T271.5・T233・T237・T238）。考える形 `頭?model=hf-ternary-bonsai-2-27b-thinking` で 1 つ: 考えが終わるまでのトークン数と時間（T233）。最後にページを読み直し、2 回目の準備完了の秒（保存から開くか: T233 のレビュー (c)）。
3. PC のメモリが 16 GB 未満なら: 27B を開いてどこで止まるか（準備完了の前にタブが落ちるか、「ran out of memory」が出るか: T233 のレビュー (b)）。16 GB 以上なら手順なし。

**B-9 Gemma（T334.1・T334.2・T334.3）**
1. ★ Gemma 4 E2B `頭?model=hf-gemma-4-e2b`: A-8 の 1. と 2. と同じ。E4B `頭?model=hf-gemma-4-e4b` も長い問い L で（メモリのある PC で。GPU だけ: [状] は「prompts and answers on WebGPU」で本数が付かない）。[状]・[答]・[コ]（`gpu: a token by … (a step of a run of 1: …)` の ms と、`gpu: both copies would not fit: … on the GPU alone` の行）（T334.3・T682）。**E4B が GPU に載ってステータス行が出るか**（スマホでも開けるなら: 保存が書けない訪問では表を 32 ビットの共有メモリに約 3.2 GB 持つので、予約が通らなければ「This browser gave no shared memory for a model on the GPU alone.」で落ちる見込み。未計測。T682 のレビュー）。
2. Gemma 3 4B `頭?model=hf-gemma-3-4b`、Gemma 3 12B `頭?model=hf-gemma-3-12b`、PLaMo 3 Translate 8B `頭?model=hf-plamo-3-translate-8b-ja-en`: 1 つずつ。[状]・[答]・[コ]（「loaded again on the CPU」の有無: GPU に載らずに読み直したか）（T334.3）。

**B-10 大きすぎる断り（T129 (d)）**
- 開く: `頭?hf=Qwen/Qwen2.5-32B-Instruct`。数秒で「This model is too large for a web page: it needs about 42 GB …」が出るか。そのあと devtools の Application → Storage（OPFS）に空のフォルダが残っていないか。[目]。

---

## C. PC の Firefox（と Edge）: 全 3 手順、約 30 分〜（27B を開くなら +30 分）

**C-1 27B が開けるか（T233・T237）**
- Firefox で `頭?model=hf-ternary-bonsai-2-27b`: 開けて答えるかだけ（4 GiB を越える共有の 64 ビットのメモリは実ページで未確認）。B-8 の保存は Firefox とは分け合わないので取得し直しになる。[状]・[答]・[目]。

**C-2 デプロイをまたいだページ（T475）**（時期による）
- Firefox でモデルのページ（例 `頭?model=llm-jp-3-150m`）を開いたままにして、**GPU の Worker のファイルが替わるデプロイ**（シェーダか GPU の部品に触るデプロイ）が本線に入るのを待ち、別のモデルを選ぶ。ステータス行の終わりに「the site was updated: reload for full speed」が出るか。中身の変わらないデプロイではファイルの名前が変わらず出ないので、そういうデプロイがある日に見る（来なければ後回し）。Firefox が WebGPU を持つ版であること。

**C-3（任意）Edge で `/benchmark/` とモデルのページ（T367.3.2）**
- `頭benchmark/` の GPU の節の「Run」と、`頭?model=llm-jp-3-150m` で 1 つ。[報]・[状]・[答]。

---

## D. iPhone（Safari）: 全 7 手順、約 1.5〜2 時間

Mac があれば Safari の「開発」メニューで [コ] が取れる（無くてもよい）。

**D-1 サイトのデータを消して、初めての訪問（T557）**
- 設定 → Safari → 詳細 → Web サイトデータ → `takano32.github.io` を削除。
- `頭` を開き、取得が始まったらすぐ別のアプリに 1 分切り替えて戻り（T129 (a)）、続けて画面を消して 1 分待ってから点ける。MB の前と後 × 2 回、読み直されたか、「got nothing from the network for 30 seconds」が出たか。取得が先に終わってしまったら、A-1 と同じく大きいモデルの取得の途中でやる。[目]。

**D-2 ★ GPU の答え（T548）と既定のモデル**
- 既定のモデル（D-1 のまま）で 1 つ聞く: [状]（「prompts and answers on WebGPU」か）、[目]（タブが落ちないか）。答えの下の行に「seed N」のボタンが出たら（T372）、押して内訳が開かないか・要約を押して開くか・種が要約の下の行に回って読めるか。
- `頭?model=llm-jp-3-150m` でも同じ。落ちたら Safari だけ戻す（T548）。

**D-3 ★ `/benchmark/` の Run all（T548・T205・T214・T219・T242・T367.3.2）**
- `頭benchmark/` → 「Run all」。モデルの節が最後まで走るか（落ちた所）、GPU の節の「sampling」と「sampling in chunks」の検査が ok か。[報]・[目]。

**D-4 小さいモデル（T249 (c)・T251 (c)・T253 (c)・T260・T229・T247・T230 (d)・T232 (e)）**
1. rinna japanese-gpt2 xsmall `頭?model=hf-japanese-gpt2-xsmall` と medium `頭?model=hf-japanese-gpt2-medium`: 日本語を書くか（T249 (c)）。
2. GPT-2 medium `頭?model=hf-gpt2-medium`・DistilGPT2 `頭?model=hf-distilgpt2`・SmolLM2 135M `頭?model=hf-smollm2-135m-instruct`: 開いて書くか（T251 (c)）。
3. MiniCPM5 1B `頭?model=hf-minicpm5-1b`: 開けるか（T253 (c)）。
4. LFM2.5 350M `頭?model=hf-lfm2.5-350m`: 開くか・読めるか（T260）。
5. Qwen3.5 0.8B `頭?model=hf-qwen3.5-0.8b` と 2B `頭?model=hf-qwen3.5-2b`: 開けるか・メモリの警告（T229・T247）。
6. Ternary Bonsai 1.7B `頭?model=hf-ternary-bonsai-1.7b`: 開けるか、[状]（relaxed SIMD の有無と、GPU か CPU か）（T230 (a)(d)・T232 (e)）。
- どれも [状]・[答]・[目]。

**D-5 GB 級（Wi-Fi で）（T290・T275・T249 (c)・T251 (c)・T253 (c)・T506 (a)・T233）**
1. ★ Ternary Bonsai 4B `頭?model=hf-ternary-bonsai-4b` と 8B `頭?model=hf-ternary-bonsai-8b`: 日本語で 1 つずつ。準備完了まで行くか、タブが落ちないか、[答] の tok/s、[状] に「relaxed SIMD」があるか（T290・T275。無ければ Bonsai 1.7B の tok/s を D-4 の 6. と比べる）。
2. EuroLLM 1.7B `頭?model=hf-eurollm-1.7b-instruct`、Llama 3.2 Shisa V2.1 3B `頭?model=hf-shisa-v2.1-llama3.2-3b`、CAT-Translate 3.3B `頭?model=hf-cat-translate-3.3b`（開くか・タブが落ちるか: T249 (c)）、GPT-2 XL `頭?model=hf-gpt2-xl`（落ちるか: T251 (c)）、Granite 4.2 3B `頭?model=hf-granite-4.2-3b`（警告が出るか: T253 (c)）。
3. Qwen3.5 4B `頭?model=hf-qwen3.5-4b`: [状] に「64-bit memory」か「int6」か（T506 (a)）。
4. Ternary Bonsai 2 27B `頭?model=hf-ternary-bonsai-2-27b`: 断りが数秒で出るか（T233。取得が始まったら止める）。

**D-6（任意）機内モード（T129 (b)）**: A-10 と同じ。

**D-7 ★ Page memory（T173）**: 最後に `頭benchmark/` の Page memory の「Run」（タブが落ちることがある）。[報] か画面の文。

---

## E. Mac（Safari）: 全 5 手順、約 1〜2 時間（Saluki を開くなら +30 分）

前の準備: Mac のメモリ（GB）と Safari のバージョンをメモ。「開発」メニューを出しておく（[コ] 用）。

**E-1 ★ Ternary Bonsai 4B・8B（T290・T275）**: D-5 の 1. と同じ。CI の WebKit 27.2（M1）は 4B 11.1・8B 3.8 tok/s。

**E-2 Qwen3.5（T245・T247 のレビュー (a)、T506 (a)）**
- `頭?model=hf-qwen3.5-4b` と `頭?model=hf-qwen3.5-2b`: 開くか・答えるか・メモリ不足の断り。[状] に「64-bit memory」か「int6」か。

**E-3 Saluki（T506 (b)）**: E-2 で「64-bit memory」と出たときだけ `頭?model=hf-underdog-saluki-27b`（取得 7.9 GB）: 準備完了まで行くか、「このウェブページは大量のメモリを使用していたため再読み込みされました」の類が出て読み直されるか、読み直した後に同じモデルをまた読み始めるか。出なければ「…Safari has none yet…」の断りが数秒で出ることだけ見る。[状]・[目]。

**E-4 `/benchmark/` とモデルのページ（T367.3.2）**: `頭benchmark/` の「Run all」（[報]）と `頭?model=llm-jp-3-150m` で 1 つ（[状]・[答]）。

**E-5 デプロイをまたいだページ（T475）**（時期による）: C-2 と同じ（WebGPU のある Safari で）。

---

## F. Chromebook（ARM、Chrome）: 全 4 手順、約 40 分〜1 時間

**F-1 Chrome のバージョン（T233 のレビュー (a)）**: `chrome://version` の版を写す。142 以下なら、4 GiB を越えるモデル（例 `頭?model=hf-qwen3-4b`: Qwen3 4B、int8 4.5 GB。2026-11-26、持ち主の決め（T311）で Ternary Bonsai 8B から替えた）に 1 つ聞いて、壊れた文を書かないか。143 以上なら 1 行写すだけ。

**F-2 ★ `/benchmark/` の Run all（T226）**: `頭benchmark/` → 「Run all」。機能の節の `deviceMemory` が 4 か 8 か。[報]。

**F-3 `deviceMemory` が 4 なら（T226・T230 (g)・T232 (d)・T245 (b)）**
1. Qwen3 0.6B `頭?model=hf-qwen3-0.6b` か Qwen2.5 1.5B `頭?model=hf-qwen2.5-1.5b-instruct`: [状]（GPU だけか）と [答] の tok/s が、F-2 の CPU の節から見積もる速さより遅くないか（T226）。
2. Ternary Bonsai 1.7B `頭?model=hf-ternary-bonsai-1.7b` と 4B `頭?model=hf-ternary-bonsai-4b`: 開くか、[状]、ページを読み直して 2 回目が取り直しにならないか（T232 (d)・T230 (g)）。
3. Qwen3.5 2B `頭?model=hf-qwen3.5-2b`: [状] に int6 が出るか、日本語が読めるか、タブが落ちないか（T245 (b)）。

**F-4 8 と言うなら**: F-3 は手順なし（`deviceMemory` が 4 の端末を持っていなければ、T226 の 4 の端末・T230 (g)・T245 (b) は見られない）。

---

## 対応表: 一覧の項目 → 手順

一覧（TASKS.md の「持ち主の端末でまとめて見るもの」）の順。「手順なし」は理由つき。

| 一覧の項目 | 手順 |
|---|---|
| T547.2.3・T547.2.4（4-bit の Shisa、T578・T579.2 の決め） | A-2・A-3、B-3 |
| T547.3（Saluki 27B を GPU だけに） | B-8 の 1 |
| T271.5（Bonsai 2 27B を GPU だけに） | B-8 の 2 |
| 　その中の T241（DP4A の層・量子化の µs） | A-5、B-1 |
| 　その中の T225（`a layer` と丸めの M） | B-1 |
| 　その中の T227（Warnings の並び） | B-1（A-5・D-3 の報告でも） |
| 　その中の T226（Qwen3 4B の GPU、`deviceMemory`） | B-5 の 1、B-1 |
| 　その中の T235（Bonsai 1.7B） | B-4 の 5、A-7 の 1 |
| 　その中の T223・T239（本数の検索の行） | B-2、A-4 |
| 　その中の T246（Bonsai 4B・8B） | B-5 の 6、A-8 の 3 |
| 　その中の T243（有限でない K・V） | A-4、B-2 |
| Android の束: T571（Qwen3.5 4B・9B を GPU だけに） | A-8 の 4 |
| 　T337（Agents-A1 の system の費用） | A-8 の 4 |
| 　T241・T219 (2)・T224・T173・T156 (c) | A-5、A-9（T173）。T156 (c) は A-5 の後に開く GPU だけのモデル（A-8 の 4・7）の [状] |
| 　T223・T240・T130・T210・T226・T129・T235・T246・T243 | A-4・A-2・A-8 の 6・A-8 の 7・A-6 の 6・A-1/A-10・A-7 の 1・A-8 の 3・A-4 |
| Chromebook: T226（4 GB と言う端末） | F-2・F-3 |
| T367.3.2（束ねた後、どの端末でも） | A-2・A-5、B-1、D-3、E-4、C-3。「選び直さない」は手順なし（鍵がその後動いた） |
| T367.5（GPU の Worker のシェーダの塊） | A-2（警告なしで GPU の判定まで行くか）。「選び直さない」は手順なし（同上）。読み込みが遅くなっていないかは目で |
| T475（デプロイをまたいだページ、Firefox と Safari） | C-2、E-5（時期による） |
| T624.2（行列 × ベクトルのエンジンの文） | A-5、B-1 |
| T509（Gemma 3 270M の special の型） | A-6 の 1 |
| T372（答えの下の種のボタン） | A-6 の 1、D-2 |
| T555（ラウンドのずれの N%） | A-5、A-9 |
| T290・T275（本物の Safari の 3 値） | D-5 の 1、E-1 |
| T548・T242（iPhone の GPU の答え、共有メモリ） | D-2・D-3 |
| T129 のレビュー (a)〜(d) | A-1（a）、A-10・D-6（b）、A-11（c、任意: 遅い回線が無ければ手順なし）、B-10（d） |
| T225 のレビュー (a)(b)(c) | B-1（a・b）、A-5（a）、B-2（c） |
| T236（Qwen3.5 0.8B と考える形） | A-6 の 4・5、B-4 の 1 |
| T247（Qwen3.5 2B・4B・9B） | B-5 の 2、B-6 の 4、A-7 の 4、D-4 の 5 |
| T229 のレビュー（Qwen3.5 の状態と 7 ビット） | A-6 の 4、B-4 の 1、D-4 の 5 |
| T245・T247 のレビュー (a)(b)(c) | E-2（a）、F-3 の 3（b: 4 と言う端末が無ければ手順なし）、A-6 の 4（c） |
| T506（Safari の 64 ビットのメモリ） | E-2・E-3、D-5 の 3 |
| T230・T231（Bonsai 1.7B の 3 値） | A-7 の 1、B-4 の 5、D-4 の 6、F-3 の 2。(a) は T232 (a) に替わった。(e)（前の int8 の保存）は手順なし: その後 `CONVERTER` が上がり、古い保存はもう消えている |
| T249・T250 のレビュー (a)(b)(c) | B-6 の 1・2、A-7 の 3・A-8 の 8、D-4 の 1・D-5 の 2 |
| T233（27B） | B-8 の 2、C-1、A-8 の 9、D-5 の 4 |
| T233 のレビュー (a)(b)(c) | F-1（a）、B-8 の 3（b: 16 GB 以上の PC なら手順なし）、B-8 の 2（c） |
| T237・T238（27B の速さ） | B-8 の 2、C-1、A-8 の 9 |
| T232（3 値を GPU に）(a)〜(h) | A-7 の 1、B-4 の 5・B-5 の 6（c）、F-3 の 2（d）、D-4 の 6（e） |
| T253・T254 のレビュー (a)(b)(c) | B-5 の 3・B-6 の 4（a）、A-6 の 8・A-8 の 8（b）、D-4 の 3・D-5 の 2（c） |
| T260 のレビュー（LFM2.5） | A-6 の 3、B-4 の 2、D-4 の 4 |
| T251・T252 のレビュー (a)(b)(c) | B-5 の 4・B-6 の 3（a）、A-6 の 7・A-7 の 5（b）、D-4 の 2・D-5 の 2（c） |
| T255 のレビュー（SmolLM3） | B-5 の 5、A-8 の 8 |
| T261.4.1（Gemma 3 の GPU） | A-6 の 1・2、B-4 の 3 |
| T258 のレビュー（Phi-3・Phi-4） | B-7、A-8 の 9 |
| T545（既定のモデルの GPU） | A-2、B-3 |
| T379.2（シェーダの文の掃除） | A-2・A-5、B-1 |
| T334.1（Gemma 4 のプロンプトを GPU に） | A-8 の 1、B-9 の 1 |
| T334.2（Gemma 4 の答え） | A-8 の 1、B-9 の 1 |
| T334.3・T682（Gemma 4 の置き場） | A-8 の 1・2、B-9 の 1 |
| T344（Spark-X2.5） | A-7 の 2・A-8 の 5、B-4 の 4 |
| T557 の前に（画面を消した間の取得） | A-1、D-1 |

一覧の外から拾ったもの（済んだタスクの記録に「端末の数字で決める」と残っているもの）:

| 項目 | 手順 |
|---|---|
| T313（3 値の表引きの形をエンジンの候補に足すか。1 ステップの N が 49 ms を切るか） | A-5・B-1（「The codes made signed by a table against the engine's form」）、A-7 の 1（コンソールの N） |
| T296（1 トークンの attention の形の覚えの偏り） | A-5（「A token's attention alone」の 128・1024・4096 位置）、A-2 の [コ]（「its attention by」） |
| T270（`/benchmark/` の 3 値の行の数） | A-5・B-1（GPU の「Ternary weights」、CPU の「Kernels of ternary weights」） |
| T263（本物の GPU が NaN と無限大を運ぶか） | A-5・B-1・D-3（GPU の「NaN and infinities」） |

## 持ち主の判断で手順が変わるもの

- **T334.3・T682（Gemma 4 E2B・E4B の置き場）**: 持ち主の決め（2026-11-26、正しい前提で聞き直した）で決まった: E2B は両方に置いて測る（保存が使えれば 2 重でスレッドの本数が付く、使えなければ GPU だけ）、E4B はいつも GPU だけ（本数が付かない）。上の手順はこの形。
- **T236 の (3)（考える形の項目）・T229 の 7 ビット**: A-6 の 4・5 の結果が材料。決めた後で見直す項目は無い。
- **T129 の判断（遅い回線のフォールバック）**: A-11 が材料（任意）。
