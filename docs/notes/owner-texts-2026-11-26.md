# 持ち主に決めてもらう文面（2026-11-26）

持ち主の決め（2026-11-26）: 全部 (a)

訪問者に見える文（画面・断りの文・README など）のうち、持ち主の答えを待っているものの一覧です。
全部で 26 項目。うち 4 つ（23〜26）はもう本番に入っていて、確かめてもらうだけです。
各項目の「案」は勧めが (a) です。「全部勧めのとおり」か、番号ごとに直すかを教えてください。

## A. GPU を使わなかった訳の文（ステータス行とコンソールに出る）

### 1. T547: Q4_K のモデルで GPU の機能が足りないとき
- いつ出るか: Q4_K の塊を持つモデル（Saluki・4-bit の Shisa など）を、WebGPU に packed int8 dot の無いブラウザ（いまの Safari など）で開いたとき。まれ。
- いまの文: `Q4_K weights need the packed int8 dot product of WGSL, which this browser lacks`（3 値のときの文に型の名前を入れたもの）
- 案: (a) このまま。(b) 技術の言葉を減らす: `this browser's WebGPU lacks what Q4_K weights need (packed int8 dot products)`
- 勧めの訳: 3 値の同じ文がもう出ていて、型の名前だけが違う形がそろう。

### 2. T547: 分類器が浮動小数でないのに「float weights」と言う
- いつ出るか: 答えのステップが CPU に残り、ステータス行の括弧に出る。T547.3.6 から Q4_K などの塊の分類器も GPU に乗るので、いま出るのは「塊の分類器で外れ値の列のあるモデル」と「分類器と埋め込みの詰め方が違うモデル」だけ。一覧のモデルで出るかは確かめていない。
- いまの文: `a classifier of float weights is not on the GPU's tokens`（`REASONS.floatClassifier`。分類器が本当に浮動小数のときも同じ文）
- 案: (a) 本当に浮動小数のときはいまの文のまま、それ以外は新しい文 `a classifier packed as this one is not on the GPU's tokens`。(b) 1 つの文にまとめる: `this classifier is not on the GPU's tokens`。(c) このまま。
- 勧めの訳: 訳の文が事実と合い、浮動小数のモデルの文は変わらない。

### 3. T573: GPU だけに置けない訳（状態の行列が大きすぎる）
- いつ出るか: GPU だけに置くモデルで、1 つの層の状態が GPU の 1 つの置き場に入らないとき。一覧の実在のモデルには出ない。いまは画面にもコンソールにも出ない（9 番を参照）。
- いまの文: `a layer's state is past a buffer of this GPU`（対の文は `a layer's matrices are past a buffer of this GPU`）
- 案: (a) このまま。(b) `a layer's state is larger than this GPU can bind at once`
- 勧めの訳: 対の文と形がそろい、実在のモデルには出ない。

### 4. T542: 3 値（と 1 ビット）の重みの断り
- いつ出るか: 3 値か 1 ビットの GGUF で、行が 128 個の組の偶数個に割れないモデルを `?hf=` で開いたとき。一覧のモデルには出ない。とてもまれ。
- いまの文: `Ternary weights need rows of whole groups of 128, an even number of them in each tensor, and this model has other rows.`（1 ビットは頭が `Binary weights`。レビュー: 少し硬い）
- 案: (a) `Ternary weights come in groups of 128 values, an even number of groups in each tensor, and this model's rows do not divide that way.`（1 ビットも同じ形）。(b) このまま。
- 勧めの訳: 意味は同じで、読む順が自然。

### 5. T502.3・T502.4: 形の `kinds` に読めない型を書いた設定
- いつ出るか: 手元のフォルダの設定 `.json` に、自分で `kinds` を書いたときだけ（変換器はこの形を作らない）。とてもまれ。
- いまの文: T502.3（int6・float16 を書いた）は何も出ず、黙って広げた写しを取る（メモリの見積もりに入らない）。T502.4（知らない型の名前）は `dtype must be float32, float16, int8, int6, ternary or binary, not Q9_X.`（塊の型も書けるのに、一覧に塊の型が無い）
- 案: (a) どちらも断り、同じ文に寄せる: `The form gives {行} the kind {型}: a row of a kind of its own is one of llama.cpp's blocks, int8 or float32.`（台帳の案）。(b) T502.3 は断らずに見積もりで数え、T502.4 だけ (a) の文。(c) このまま。
- 勧めの訳: 届く道が 1 つ減り、文が事実と合う。T502.4 の残り 2 つ（zip の無い所の塊の行、`tiled` と回した基底）は届かない道なので文を作らない。

### 6. T489: フォールバックのアダプタの文が 2 つある
- いつ出るか: CPU が GPU の代わりをする「アダプタ」しか無い端末（ふつうの訪問者にはほぼ無い。CI の試験の環境）。訪問者に見えるのは 1 つ目だけ。
- いまの文: 見える側 `a fallback adapter: the CPU in the GPU's place`、見えない側（GPU だけに置くかの判定）`a fallback adapter`
- 案: (a) 見えない側も見える側の文にそろえる。(b) このまま 2 つ。
- 勧めの訳: 同じ訳に文が 1 つになる。9 番で見えない側を出すなら、なおさら。

### 7. T489: 4 の倍数でない head の文が 2 つある
- いつ出るか: head の大きさが 4 の倍数でないモデル（一覧には無い見込み）。見えるのは 1 つ目だけ。
- いまの文: 見える側 `heads of a size that is no multiple of 4 are not on the GPU`、見えない側 `heads of a size that is no multiple of 4`
- 案: (a) 見えない側も見える側の文にそろえる。(b) このまま 2 つ。
- 勧めの訳: 6 番と同じ。

### 8. T489: `/benchmark/` で WebGPU があっても「WebGPU が無い」と言う
- いつ出るか: `/benchmark/` のモデルの節のラウンドの読み込みで。ページに出るかは確かめていない（台帳の記録）。
- いまの文: `no WebGPU in a worker here`
- 案: (a) ベンチのラウンドが GPU を使わない作りなら、その訳を言う: `the GPU is not used in this round`（出る所を確かめてから）。(b) このまま。
- 勧めの訳: 誤った事実を言わない。ただし出る所は実装のときに確かめる。

### 9. T489: GPU だけに置けなかった訳が、どこにも出ない
- いつ出るか: 2 重に持てない大きいモデル（Pythia 6.9B・Qwen3.5 4B・9B など）が GPU だけに置けず、CPU だけで動くとき。大きいモデルを選ぶ人には出うる。
- いまの文: いまは何も出ない（訳は判定の中だけで使われる）。
- 案: (a) コンソールに 1 行: `gpu: {訳}: the model stays on the CPU alone`。(b) (a) に加えて、ステータス行の括弧に「prompts on the CPU ({訳})」。(c) このまま。
- 勧めの訳: 正常な訪問ではコンソールの行は増えず（その場合だけ出る）、調べる手段ができる。

### 10. T491: 固まった GPU の Worker（動きと文）
- いつ出るか: GPU だけに置くモデルで、GPU の Worker がエラーも出さずに止まったとき。まれ。
- いまの文: 60 秒で読み込みが失敗する。投げる文は `the GPU took no weights for 60 s`（画面での前後の文は確かめていない）。
- 案: (a) CPU で読み直し、訳は今ある文を使う: `the GPU said nothing for 60 s`（`REASONS.quiet`、秒は実際の値）。(b) CPU で読み直し、新しい文 `the GPU's worker stopped taking the weights`。(c) このまま失敗にする。
- 勧めの訳: 訪問者は答えを得られ、文は今あるもので足りる。
- **決まり済み（2026-11-26）: (a)。** 始まらない GPU の Worker は、分かった時点で打ち切って CPU 用に読み込む、も決まり済み。

## B. GGUF とフォルダと `?hf=`

### 11. T508: README の「GGUF だけのリポジトリは開かない」が古い
- いつ出るか: README を読む人。
- いまの文: `It does not open a repository that has only GGUF files: the GGUFs of the list are read together with the vocabulary and configuration of their original repository.`
- 案: (a) `It also opens a repository that has only GGUF files: it takes the first file it can read in the order Q8_0, the ternary files, F16, BF16, F32, Q6_K, Q5_K_M, Q4_K_M and on (or the one &file= names), and reads the vocabulary, the configuration and the chat template from the GGUF itself. The GGUFs of the list are read together with the vocabulary and configuration of their original repository.` (b) 最初の文を消すだけ。
- 勧めの訳: いまの動き（`?hf=` とシートが GGUF も開く）を言う。同じ段落の読める形の一覧（Llama・Qwen2・Qwen3・GPT-2・GPT-NeoX）も古い（12・13 番と一緒に直す）。

### 12. T508: `docs/architecture.md` の GGUF の説明が古い
- いつ出るか: 文書を読む人。
- いまの文: `Converts a Hugging Face model (or a GGUF: Q8_0, or the ternary PQ2_0 and PTQ1_0) as its file arrives, …`
- 案: (a) `(or a GGUF: Q8_0, F16, BF16, the ternary PQ2_0 and PTQ1_0, the 1-bit Q1_0, and llama.cpp's K and i quantizations)` (b) 型を並べずに `(or a GGUF)`
- 勧めの訳: 今の読み手と合う。

### 13. T508: 選択欄の「Other repository…」の読める形が古い
- いつ出るか: モデルの選択欄を開く全員。
- いまの文: `Other repository… — any Llama, Mistral, Qwen2, Qwen3, GPT-2 or GPT-NeoX`
- 案: (a) `Other repository… — a Hugging Face repository, safetensors or GGUF`（形の一覧は README に任せる）。(b) 形を足して並べる: `… — Llama, Mistral, Qwen2, Qwen3, Qwen3.5, Gemma, LFM2, GPT-2, GPT-NeoX and more`
- 勧めの訳: 形を足すたびに直さずに済む。 ページの本文と README の「開けるモデルの種類」の文は、変換器の表から自動で作ると決まり済み（T309）。

### 14. T508: 「kept in this browser」の一覧が同じ名前を 2 回並べる
- いつ出るか: 同じリポジトリの 2 つの GGUF を `?hf=` で開いて保存した人。まれ。
- いまの文: 1 行ずつ `{リポジトリ} ({N} MB) delete`
- 案: (a) GGUF のときだけファイルを付ける: `{リポジトリ} · {ファイル名} ({N} MB) delete`。(b) このまま。
- 勧めの訳: どちらを消すかが分かる。safetensors の行は変わらない。

### 15. T508: シートの Open の後、最大 10 秒なにも出ない
- いつ出るか: シートにリポジトリを書いて Open を押した人。ファイルの一覧を待つ間（ふつう 1〜3 秒、最大 10 秒）。
- いまの文: いまは何も出ない。
- 案: (a) ステータス行に `Looking at the files of {owner/repo}…`。(b) Open のボタンを押せなくして文は出さない。
- 勧めの訳: 押したのに何も起きないように見えるのを避ける。

### 16. T410・T508: 手元のフォルダから `.gguf` と分割した safetensors を開く門
- いつ出るか: フォルダのボタンでファイルを選んだ人が、`.gguf` や分割した safetensors を選んだとき。
- いまの文: `A Hugging Face model needs three files together: one .safetensors file (a model in several shards is not supported), config.json, and tokenizer.json, tokenizer.model or spiece.model.`（無いファイルの文は段取りの `The conversion needs {名前}, which is not there.`）
- 案: (a) 門の文は変えず、`.gguf` を 1 つだけ選んだら開く（ボタンの説明に「…, or one .gguf file」を足す: 17 番）。(b) 門を開いて文も直す: `Choose one .gguf file, or the files of a Hugging Face model: its .safetensors (one file, or the shards with their index), config.json, and tokenizer.json, tokenizer.model or spiece.model.` (c) やらない。
- 勧めの訳: 台帳の勧めは無い。読み手はもう両方とも読めるので、(a) が文の変わりが一番小さい。分割した safetensors も開くなら (b)。

### 17. T416: フォルダのボタンの説明（`title`）
- いつ出るか: ボタンにマウスを載せた人。
- いまの文: `Open a model from this device: a llama2.c checkpoint and its tokenizer.bin, or the files of a Hugging Face model (.safetensors, config.json, and tokenizer.json, tokenizer.model or spiece.model), and optionally settings as .json. Nothing is uploaded.`
- 案: (a) 文は同じで、トークナイザの 3 つの名前を門の一覧から作る（候補を足すと自分で変わる）。16 番で (a) なら `, or one .gguf file` を足す。(b) このまま手書き。
- 勧めの訳: いまの文は一字も変わらず、写しが 1 つ減る。

### 18. T510: GGUF が原本と合わないときの文に Python の値がそのまま出る
- いつ出るか: Gemma 3 の GGUF を、合わない原本の設定と組んで `?hf=` で開いたとき。とてもまれ。
- いまの文: `This GGUF does not belong with the original's config.json: its {何} is {ここの値} here and {あちらの値} there.` の値の所に、`{'layers': [0, …], 'window': 512, 'rope_theta': 10000.0}` や `None`、`('linear', 8.0)` がそのまま出る。
- 案: (a) 値を文にする: 局所の層は `layers 0 to 4, 6 to 10, … with a window of 512 and a RoPE base of 10000`、無いときは `none`、RoPE の縮めは `linear by 8`。(b) 値を出さず `… its {何} differs from the original's.` だけ。
- 勧めの訳: 何が違うかが読める。ほかの項目（層の数など）は今も数で出ているので形がそろう。

### 19. T402: `docs/quantization.md` の `?hf=Qwen/Qwen3.5-0.8B` の 1 文
- いつ出るか: 文書を読む人。
- いまの文: `What ?hf=Qwen/Qwen3.5-0.8B still loses is plain text, where there is no chat format and the converter's BOS comes first.`
- 案: (a) `Opened with ?hf=, a Qwen3.5 gets the same beginning since the converter reads its template (2026-10).`（台帳の案）。(b) 文を消す。
- 勧めの訳: T369 から変換器が書式を読むので、いまの文は合わない。

### 20. T283: BitNet の 16 ビット版を開いたときの断り
- いつ出るか: `microsoft/bitnet-b1.58-2B-4T-bf16` を `?hf=` で開いた人。まれ。
- いまの文: `This model cannot be converted: its matrices are not the ternary ones of a file of BitNet's packed weights.`
- 案: (a) `This is the 16-bit copy of BitNet; this page reads the ternary one (microsoft/bitnet-b1.58-2B-4T).`（台帳の案。ほかの BitNet の形の断りはいまの文のまま）。(b) このまま。
- 勧めの訳: 何を開けばよいかが分かる。

### 21. T659: 書式の印が無い `tokenizer.model` を読まない訳
- いつ出るか: `?hf=` で、tokenizer.json に書式の印があり、それが読めないモデル（Phi-3 の形の一部）。まれ。
- いまの文: tokenizer.json の読み手の断りの文そのまま（文は場合ごとに違う）。
- 案: (a) 最後に 1 句: `… (its tokenizer.model, which lacks the marks of the format, is not read in its place)`。(b) このまま。
- 勧めの訳: 「tokenizer.model があるのになぜ読まないか」が分かる。台帳の別の決め: 次に `CONVERTER` を上げる回に T659 を乗せる。

### 22. T270: `/benchmark/` の CPU の節の説明に 3 値の 1 文を足すか
- いつ出るか: `/benchmark/` を開いた人。
- いまの文: `The model page's own forward pass on a made-up int8 model as wide as Llama 3.2 1B (211 MB of weights a token), with 1, 2, 4 and on software threads, doubling up to the logical cores. Beside it, the CPU's ceilings: how fast it reads alone, and loops of int8 dots and of float multiply-adds.`
- 案: (a) 最後に `Last, the kernels of ternary weights beside int8's, on one thread.` を足す。(b) このまま（表の頭に説明があるので）。
- 勧めの訳: 節の説明が節の中身と合う。

## C. もう本番に入っているもの（確かめるだけ）

### 23. T270: `/benchmark/` の 3 値の表の文（入っている）
- 表の頭の対照の 1 文: `The engine's form is timed once more at the end of each such table: the two rows' difference is the noise (the order the forms are timed in, the device's drift) that a ratio between the forms is to be read against.`
- 直した句: `the sums made as each workgroup loads the vector`
- 要約の括弧: `(the engine's form against itself 1.02×)`（数は端末ごと）

### 24. T313.2: `/benchmark/` の ORT の表引きの形（入っている）
- 表の頭: `and ONNX Runtime's form for 2-bit weights, which makes the codes signed by a table of 256 words in the workgroup's memory (four reads of the table a word of 16 codes, the table filled as each workgroup starts).`
- 表の下と要約: `**The codes made signed by a table against the engine's form** (above 1×: faster than the engine's here): …`、要約 `GPU, ternary weights: the codes made signed by a table against the engine's form …`

### 25. T656: 12B の 3 つ（入っている）
- 名前: `CyberAgent Mistral NeMo Japanese Instruct 2408`・`Shisa V2 Mistral NeMo 12B`・`Google Gemma 3 12B`
- note: `answers instructions · 日本語 / English · fetches 13.0 GB (GGUF) → int8 13.8 GB`（Gemma は `12.5 GB (GGUF) → int8 13.2 GB`。「desktop only」などは窓口が付ける）
- 標本抽出: NeMo Japanese は温度 0.5（カードのコード）、Shisa は一覧の日本語の組（0.7・top-p 0.9・罰 1.1。罰はカードに無い）、Gemma は 1.0・top-k 64・top-p 0.95。

### 26. T283: BitNet の項目（入っている）
- 名前: `Microsoft BitNet b1.58 2B4T`
- note: `answers instructions · English · ternary weights · fetches 1.2 GB, kept mostly as it is: 0.9 GB`

## 確かめられなかったもの
- 2 番: いまの一覧のモデルで `floatClassifier` が出るものがあるか（コードの条件から、塊の分類器に外れ値の列があるか、分類器と埋め込みの詰め方が違うモデルだけと読んだ）。
- 8 番: `no WebGPU in a worker here` が `/benchmark/` のページに本当に出るか（台帳も「未確認」）。
- 10 番: 60 秒の失敗が画面でどんな前後の文と一緒に出るか（コードで見つけたのは投げる文 `the GPU took no weights for 60 s` だけ）。
- 21 番: 断りの文そのものは場合ごとに違うので、1 つの「いまの文」として写せない。
- 入れなかったもの: T506 の Safari の文（台帳の勧めは「本物の Safari を見るまで製品を変えない」）、T669・T670・T322 の画面（まだ作っていない）、T234 の README と gist（決まり済みと聞いた分）。
