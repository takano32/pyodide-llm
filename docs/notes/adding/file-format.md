# ファイルの形式を変える・行を足す

AGENTS.md の「物を足す手順」節から移した（T658、2026-11-12）。要点は AGENTS.md に。

どの手順も「触る所」と「忘れると誰が教えるか」まで書く。**教えないと書いてある所は、黙って別の動きになる**。試験の側の 2 つ目の写し（`FACTS`・`ORDER`・`KINDS` など）はわざと 2 回書いてあるもので、足したら一緒に直す。

### ファイルの形式を変える・行を足す

legacy 形式はメタデータを持たない: dtype は見出しから計算した大きさとファイルの大きさの一致で判定する（`checkpoint_dtype()`）。

1. 行を足す・動かす・形を変えるのは `layout.py` のそのレイアウトの関数だけ（`attention()`・`gated_ffn()`・`head_norms()`・`after()`）。**層によって形の違う行は、形ごとに別の名前の束にする**（`per` の `NARROW`・`WIDE`・`NARROW_KEPT`…と `Dims.layers(per)`、名前は `wq_wide` のように）。**1 層だけの束は変換器が束と読まない**。行を引くだけの表は役 `ROWS`。
2. 新しい行に、家族の `*_sources()` で取り込み元を。モデルによって無い行は `Llama.__init__` の `None` を置く行に名前を足す。
3. ファイルから分からない設定は `FORM` に鍵（既定は None か今の動き）、`Llama` は `form_of()` で読むので触らない（T639.1）、変換器の `checkpoint_form()`。後ろの大きさが変わるなら種類の `state()`・`frame()`。**設定は変換器から `footprint()` まで 1 つの辞書で渡る**（`worker-sink-check`）。`arch`・`dtype` を受け取る関数を足したら呼び出し側を全部探す。
4. 保存の型は行ごとに `kind_of(row, dtype)` が選ぶ（行列はファイルの型、ベクトルは量子化のファイルで float32、形の `kinds` に名前のある行はその型）。**`kinds` の値は型の名前か、束の層ごとの名前の並び**（`placed()` は層ごとの置き場 `Place.layers` を出す）。型は `DTYPES` の 6 つか `BLOCKS` の塊の型で、**塊の型の行を持つファイルにだけ、最後に符号表の行 `codebooks`**（`convert/grids.py` の `codebooks()` と `engine/dtypes.py` の `CODEBOOKS`）。**ファイルの dtype は 6 つのまま**（T634 の binary を足した。行列を全部名指すファイルは `int8` と答える）。**GGUF の塊をそのまま持つ変換は `Conversion(blocks=True)`**（`convert/stream.py` の `Kept`）: テンソルが全部塊の型で「行の移動だけ」で届く行をバイトのまま写す（束の 1 層でも当たらなければ丸ごと広げる）。行を動かすテンソルは行の番号に `untiled()`・`transformed()` を当てて入れ替える（`order()`）。**GPT-2 の Conv1D の行列も行の移動**（T636: GGUF は (out, in) で持ち、読み手の `transposed` と計画の `transpose` か c_attn の `part` が打ち消すので、行はファイルの行かその続いた範囲。`order()` は行の番号を (in, out) の 1 行にして計画に通す。ほかの段と組んだ転置は通さない）。NeoX の `query_key_value` も行の移動で残る（`tests/test_kept.py`）。**列の並べ替え（Qwen3.5 の `ssm_out`）はバイトでは戻せない**: ファイルの並びのまま置き、形の鍵 `tiled` を立てる。**頼まれない変換は前と同じ**。
5. 新しい行が bias なら `tests/made_up.py` の `BIASES` に。`test_layout.py` の `ORDER` を直す（**行を動かしたらファイルの形式の変更: `CONVERTER`**。入れ替えはエンジンと参照の比べでは見えず、`ORDER` と本物の llama2.c のファイルだけが落とす）。
6. 詰めた dtype（int6・3 値）は、行が塊に割れないモデルを `check_suited()` が断る。大きさと置き場は `math.prod` で数える（Pyodide の NumPy の整数は 32 ビット）。
