### T.228 [調査][Bonsai] Bonsai 2 と Qwen3.8 の形を調べる — 状態: 完了（2026-10-15、T.527: 調べは本線に入り、決めることは T.229〜T.235 で決まった）。前の見出し: **反映済み**（2026-10-01、調べだけ。本線に入れた）
- 結果は [docs/notes/t228-bonsai-2-2026-10-01.md](docs/notes/t228-bonsai-2-2026-10-01.md)（出どころの URL と版つき）。台帳の名前と大きさは実物と合っていた。
- **形**: linear attention は Gated DeltaNet。64 層のうち 4 層ごとの 4 つ目（16 層）が full attention、48 層が linear。linear の層は KV を持たず、状態は 48 head × 128 × 128 の float32（48 層で 144 MiB、文脈に依らない）と畳み込みの前 3 トークン。full の層は q が 2 倍幅（gate つき）、head 256、RoPE は先頭 64 次元。KV は 1 位置 64 KiB（float16）。
- **3 値**: PQ2_0 は 34 バイト（float16 のスケール + 2 ビット × 128）、PTQ1_0 は 28 バイト（3 進で 5 個 / バイトを 24 バイト、4 個 / バイトを 2 バイト、float16 のスケール）。読み方は F16 のファイルと頭の数十行でビット単位に合った。埋め込みと分類器も 3 値。fork（PrismML-Eng/llama.cpp、`88c4bc60`）は MIT。
- **台帳に無かったもの**: 重みは回した基底（Hadamard、1024 個の塊、符号つき）で入っていて、行列の前に毎回活性値を変換する。F16 のファイルも同じ。前分割は新しい型（`qwen35`: 結合文字を字に入れる）、chat_template はマクロを呼ぶので手で書く。原本の Qwen3.8 は 3 値でないので、重みを比べる相手は F16 の GGUF だけ。
- **入るか**: 重み 5.53 GiB（PTQ1_0）、文脈 4096 で合計 5.92 GiB。16 GiB の 64 ビットのメモリに入る（文脈は約 16 万位置まで）。Safari の 4 GiB には入らない。持ち主のスマホで持てるかは未計測。
- **速さ（見積もり、未計測）**: 1 トークンに 25.6 G の積和。int8 に広げて掛けるなら持ち主の Android の 4 本で約 1.1〜1.7 tok/s、CI の 4 本で約 1.0 tok/s。考える形（既定）の勧めの出力は 16384 トークンで、1 tok/s なら 4.5 時間。GPU は帯域の上限で 5.8〜7.1 tok/s まで。
- **小さい実物**: Ternary-Bonsai の 1.7B・4B・8B は Qwen3 の形で回した基底でない（PQ2_0 463 MB・1.07 GB・2.18 GB、yarn の RoPE）。Qwen3.5 の 0.8B・2B・4B・9B は同じ hybrid attention（3 値でない）。
- **持ち主に決めてもらうこと**: 27B を最後まで進めるか、小さい 2 つを一覧に足すか、27B の詰め方（PTQ1_0 か PQ2_0）、既定をすぐ答える形にするか、文脈をいくつで切るか、top_k・min_p・presence_penalty を足すか。
- 頼みの文（残す）: Qwen3.8 の hybrid attention の中身（linear attention の種類: Gated DeltaNet などか、層の並び、状態の大きさ、RoPE の有無）、tokenizer と chat_template（思考の形）、config の項目。PTQ1_0・PQ2_0 のブロックの並び（Prism の llama.cpp の fork の `ggml` の型の定義と、CPU の参照の dequant）。ライセンス（apache-2.0、NOTICE）と KNOWN_ISSUES.md。prism-ml のほかの小さいモデル（前の版の Bonsai など）が一覧に入れやすいか。結果として、T.229〜T.234 の規模と、ブラウザの上限（64 ビットのメモリ 16 GB、iPhone）で動く見込みを数字で出す。元ネタの許諾（fork は MIT のはず、確かめる）。
