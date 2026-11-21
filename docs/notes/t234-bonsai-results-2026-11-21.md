# T234: Bonsai 2 の結果のまとめ（2026-11-21、Opus medium）

Ternary Bonsai（1.7B・4B・8B・2 27B）と 1-bit の Bonsai（1.7B・4B・8B・27B）の速さ・メモリ・品質を、記録のあるものだけで集めた。推測の数は書かない。測っていないものは「未計測」。

## 訪問者向けの英語の文書に置いた所

- 3 値の 1.7B・4B・8B・27B は前から `docs/quantization.md` の「Ternary weights: 2 bits」と「Ternary Bonsai 2 27B」、`docs/performance.md`（27B のページの数）、`kernels/README.md`（カーネルの形と速さ）、`docs/webgpu.md`（3 値の GPU の道）にあった。T234 では足していない。
- **足したもの**: `docs/quantization.md` に節「1-bit Bonsai」（表 1 つと段落 1 つ）。出どころは `docs/tasks/T286.md`（T286.2〜.5）と `docs/tasks/T634.md`（T634.1〜.3）。
- **直したもの**: `docs/quantization.md` の 3 値の 1.7B の「It runs on the CPU only for now (no GPU path)」（T232 で 2026-10-02 から GPU にも乗るので古い）、`docs/webgpu.md` の 3 値の GPU の段落に 1-bit の Bonsai（T634.3）。
- `README.md` は変えていない（下の案を持ち主に見せる）。

## 数と出どころ（1-bit の Bonsai）

| | 1.7B | 4B | 8B | 27B | 出どころ |
|---|---:|---:|---:|---:|---|
| 取得（Q1_0） | 248,302,272 B | 572,270,624 B | 1,158,654,496 B | 3,803,452,480 B | T286.2〜.5 |
| 持つ大きさ（1 ビット） | 242,357,180 B | 566 MB（一覧の見積もり） | 1.2 GB（一覧の見積もり） | 3,883,513,884 B | T634.2（run 41825322023・41831632627）、4B・8B は note の語 |
| 3 値で持っていた頃 | 457,345,212 B | 1,069,018,852 B | 2,176,234,140 B | 7,242,233,884 B | T286.2〜.5 |
| GGUF と原本の最大の相対差 | 3.5e-5 | 4.8e-5 | 1.7e-8 | 1.4e-7 | gguf.yml run 41681725146・41690146899・41701199253・41717761636 |
| perplexity 1500 トークン（英 / 日、素の RoPE、3 値の道） | 27.507 / 73.399 | 20.095 / 49.702 | 17.891 / 38.251 | 同じ形では未計測（T633 の窓 512・3 記事: `<\|im_start\|>` 34.904） | run 41682850540・41690147060・41701200621、T633 run 41727340001 |
| 本番 Chromium（CI、4 本） | 準備 20.3 s・16.9 tok/s・2 回目 3.9 s | 39.4 s・7.6 tok/s（3 値で持っていた頃） | 60.3 s・4.5 tok/s・2 回目 5.8 s | 204.9 s・1.3 tok/s（3 値で持っていた頃） | models.yml run 41871037505・41703218428・41725072480 |

- 1 ビットと 3 値で logits はビット単位で同じ（1.7B の 64 位置、27B の 16 位置、relaxed の有無の両方: T634.2）。
- カーネル: 1 本は 3 値と同じ、4 本で x86-64（EPYC 7763）1.02 倍・arm64（Neoverse-N2）1.06 倍（T634.1、run 41807275629・41807279476）。
- 27B は 1 ビットでも 32 ビットのメモリに入らない（`footprint()` で 4.181 GiB: T634.2）。
- 書かせ: 1.7B は日本語で崩れる（英語の項目）、4B・8B・27B は日本語の問いに日本語で答えた（T286.2〜.5）。
- GPU: 3 値の符号に広げて 3 値のシェーダで。確かめは CI のソフトウェアのアダプタだけ（T634.3、run 41818418882）。実機の GPU は未計測。
- 端末（持ち主の Android・iPhone）: どれも未計測。

## 3 値（参考、前から文書にある数）

- 1.7B: 457 MB、perplexity 英 22.066・日 45.761（3 値、8 ビットの活性値）、1 本 10.9 tok/s（arm64）。
- 4B・8B: 本番 Chromium の x86-64 4 本で 6.4・4.2 tok/s（2026-10-02）。
- 2 27B: 7.24 GB（2026-10-15 から。performance.md の 7.66 GB はその前の float32 の尺度の大きさ）、本番 Chromium で x86-64 1.2 tok/s・arm64 2.2 tok/s、Node 4 本で 1.56・2.42 tok/s、fork の CPU は 0.70・1.13 tok/s。

## 持ち主に見せる文の案

### README（変えていない）

`README.md` の「From Hugging Face, converted in this browser」の項の 2 か所:

- 今: 「for the Ternary Bonsai models their ternary files, PQ2_0 or PTQ1_0, read with the original repository's vocabulary and configuration」
  案: 「for the Ternary Bonsai models their ternary files, PQ2_0 or PTQ1_0, and for the 1-bit Bonsai models Q1_0, read with the original repository's vocabulary and configuration」
- 今: 「(Ternary Bonsai stays in its 2 bits)」
  案: 「(Ternary Bonsai stays in its 2 bits, the 1-bit Bonsai in 1 bit)」
- 同じ項の「one ternary 27B」は、いまは 3 値の 27B と 1-bit の 27B の 2 つ。案: 「a ternary and a 1-bit 27B」。

### gist（https://gist.github.com/takano32/196c6f93979ad44f98cee5712fdd3901 、更新していない）

足す節の案（英語、gist の言葉に合わせて直してよい）:

> **Ternary and 1-bit Bonsai (Prism ML).** The page runs Ternary Bonsai 1.7B, 4B, 8B and 2 27B and the 1-bit Bonsai
> 1.7B, 4B, 8B and 27B without widening their weights: 2 bits (ternary) or 1 bit a weight, with a float16 scale for
> every 128, multiplied by WebAssembly SIMD kernels written for them. The 27B models take 7.24 GB (ternary) and 3.88 GB
> (1-bit), on a 64-bit memory (Chrome and Firefox). On the deployed page in Chromium on CI's runners with 4 threads,
> Ternary Bonsai 2 27B writes 1.2 tok/s (x86-64) and 2.2 tok/s (arm64), and in Node it writes about twice what Prism
> ML's fork of llama.cpp writes on the same CPUs. The 1-bit Bonsai 1.7B writes 16.9 tok/s and the 8B 4.5 tok/s.
> Against the fork's float32 activations, the 27B's most likely token is the same at 154 of 155 positions. Not
> measured on a phone yet. Details: docs/quantization.md.
