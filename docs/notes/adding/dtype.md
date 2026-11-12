# dtype（詰め方）を足す

AGENTS.md の「物を足す手順」節から移した（T658、2026-11-12）。要点は AGENTS.md に。

どの手順も「触る所」と「忘れると誰が教えるか」まで書く。**教えないと書いてある所は、黙って別の動きになる**。試験の側の 2 つ目の写し（`FACTS`・`ORDER`・`KINDS` など）はわざと 2 回書いてあるもので、足したら一緒に直す。

### dtype（詰め方）を足す

（ファイルの dtype の話。**行だけが持てる塊の型**は「GGUF の型」の項。）

1. `engine/packing.py` に詰め方と戻し方、`engine/dtypes.py` の `DTYPES` に 1 行。大きさ・`Writer`・NumPy の読み・`checkpoint_dtype()` の候補・断りの文が付く（**同じ見出しでほかの dtype と大きさが重ならないことを `tests/test_ternary.py` の形で**）。カーネルの詰め手があれば行に export の名前。
2. カーネル（次の項）、`jobs.ts` の `ADDRESSES`、仕事の種類を `WORK` に（relaxed の有無）、**`PACKED` に 1 行**（1 行のバイト数・`scale`・重みを読む式 `value` か 1 行を広げる `row`・仕事の種類・補正のカーネル・活性値の用意とその取り方 `takes`・`xs`・`group`・GPU の `gpu`）。`forward/engine.ts`・`memory.ts`・`gpuside.ts`・worker は触らない。
3. 型: `contracts/plan.ts` の `Stored`、`contracts/form.ts` の `Dtype`。`kept.ts` の `replaced()` の名前の一覧と `src/models/memory.ts` の比は `QUANTIZED` と `stored_bytes()` から書き出される（T628.3: 手で書かない。比は `memory.ts` が `int6`・`ternary` を名指すので、新しい dtype の比を使うならそこに 1 行）。保存の名前（`keptName()`）と、GPU だけに置けるなら `mayKeep` も。GPU だけの判定を覚える端末の鍵（plain・ternary・blocks）は `forward/alone.ts` の `keyKindOf()` が `PACKED` の `gpu.as` から選ぶので、worker は触らない（T641.9。2 つ目の写しは `kernel-registry-check`）。
4. 試験: `tests/kernel-registry-check.ts` の `FACTS`・`ASKED`・`GPU`、`tests/test_dtypes.py`。
5. GPU にも置くなら「ステップの行列の詰め方」と、`gpu/packings.ts` の `PACKINGS` に行の形の 1 行（GPU での行のバイト数・GPU の尺度の数・チェックポイントでのバイト数・尺度のバイト数・値と尺度の上げ方 `up`・`scaleUp`（T642.8: `gpu/weights.ts` の表 `UPLOADS` の行の名前。写す・GPU で広げる・CPU で散らす・float16 を float32 に。共有メモリから上げる `into` と、GPU だけのモデルの届くままの `routed` の 2 つを持ち、新しい上げ方はそこに 1 行。`kernel-registry-check` が詰め方と突き合わせる）・検査の答えの値。`kernel-registry-check` が `PACKED` の行と突き合わせる）と `contracts/gpu/protocol.ts` の `GpuPacking` に名前（`kernel-registry-check` が `PACKED` の `gpu` と突き合わせる）。
