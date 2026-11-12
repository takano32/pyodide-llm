# カーネル・仕事の種類を足す

AGENTS.md の「物を足す手順」節から移した（T658、2026-11-12）。要点は AGENTS.md に。

どの手順も「触る所」と「忘れると誰が教えるか」まで書く。**教えないと書いてある所は、黙って別の動きになる**。試験の側の 2 つ目の写し（`FACTS`・`ORDER`・`KINDS` など）はわざと 2 回書いてあるもので、足したら一緒に直す。

### カーネル・仕事の種類を足す

1. `kernels/` にカーネル。**形の決まり**: 静的データを置けない（数学関数はテーブルなしで）。relaxed SIMD を使うものは別ファイル。内側のループに定数のベクトルを書かず引数から作る。シャッフルを続けた形は疑う（JavaScriptCore が誤って畳んだ）。x86 に 8 ビットのシフトと `f32x4.max` の命令は無い。int8 の積の和を i16 で足すならあふれる組を数える。Makefile の規則は `kernels/` とその下のフォルダの `.ts` を全部 `$(wildcard)` で拾う（`KERNEL_SOURCES`: ファイルを足しても Makefile は触らない。`kernels/` の中に作る物の源でない `.ts` を置かない）。部品が使う共通の関数は末尾の `export { … };` で出す。
2. `jobs.ts` の `ADDRESSES` に名前と番地の引数（64 ビットのメモリで番地は BigInt、仕事の表は float64）。段の仕事なら **`WORK` の最後に 1 行**（`from`・`one(k)`（1 トークンの仕事は 9 つ目の引数に out stride の欄を受けられる: T612）・`tile(k, スレッドの番号)`・`rowBytes`・タイルが一度に取る行の数 `tileRows`（プロンプトの塊はこの数ずつに切る: 1 なら書かない）・`warm`）。組む所が `KIND.<名前>` と書く。`runner()`・`helper.ts`・`threads.ts` は触らない。
3. 試験: `kernel-registry-check.ts` の `KINDS`、JavaScript の同じ順の計算と比べる検査（残りが 0〜3 のどれも通る幅で）、**`tests/kernels-in-browser.ts` に足す**、smoke。任意の機能の準備は別の `try` に。**Python が ctypes で呼ぶカーネルは `engine/kernels.py` の `load_kernels()` の署名の表と restype の 1 行**: 引数の型（`usize` は `c_void_p`、`i32`・`f32`・`f64` は `c_int32`・`c_float`・`c_double`）・数・戻り値が `export function` の宣言と違えば、軽い組の `tests/kernel_signatures.py` が名指して落とす（Python が束ねたのに `kernel.ts` が出さないカーネルも。出すだけで Python が束ねないもの（forward.ts のもの）は一覧に言うだけ）。宣言は 1 行に書き、型は 4 つのどれかで（読めない形は道具が断る）。今ある違い（`to_f16` の戻り値を Python が読まない）は道具の `KNOWN` に訳つきで、無くなれば落ちる。
4. 速さは x86-64 と arm64 で、行列をメモリから読む形で測り、的の arm で決める。int8 は加算順で文が変わる（回帰試験を「前と同じ文」にしない）。
5. **塊の型（`BLOCKS`）のカーネルは別のモジュール**（`kernels/blocks.ts`・`blocks_relaxed.ts`）。型を足すなら `blocks/types.ts` に型のクラスと GGUF の番号、塊のバイト数（`bytesOf`・`sizeOf`）、16 個の重みの読み（`magnitudes`・`minuses`）、`scaleOf`・`blockScale`、`bySixteen`・`hasSigns`。`blocks/row.ts` の `floatScale`。**型の番号で分ける並びが 4 か所**（`rowsOf`・`tileOf`・`matmul_sbw_tile`・`sb_row`: どれも輪の外で型を 1 回選び、型ごとの輪 `rowsOfKind<K, R>`・`tileOfKind<K, R>`・`spread<K>`・`rowOf<K>` に渡す（T644.3: 1 つの塊ごとに 10 段の比べをしない）。並びに 1 行。忘れるとその型は何も書かない。`blocks-check` が落とす）。試験は `tests/blocks-model.ts` の `TYPES`。**型と relaxed の有無は型引数で渡す**（`blockOf<XXS2, R>`）。ページの側は `PACKED` に 1 行（`blocks(BLOCK_TYPES.<名前>)`。`BLOCK_TYPES` は `SUPERBLOCKS` から書き出される: T628.3）、`Stored`。塊のカーネルは暖機しない。
