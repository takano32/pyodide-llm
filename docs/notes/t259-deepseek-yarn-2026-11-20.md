# T259: DeepSeek-R1-0528-Qwen3-8B の yarn の `attn_factor` をどう読むか（2026-11-20）

## 結論

- **読む。意味は「yarn の倍率に掛ける」**（vLLM 0.8.5〜0.9・SGLang 0.4.6.post1 と、llama.cpp の GGUF の鍵 `rope.scaling.attn_factor` と同じ）。このモデルの値 0.8782488562869419 はちょうど 1 / (0.1 ln 4 + 1) で、cos と sin の倍率を 1.1386 から **1.0** に戻す（yarn の角はそのまま、温度だけを消す）。
- ページの道（int8、カーネル、7 ビットの活性値）の perplexity で、**transformers と llama.cpp の読み（1.1386）が 4 つの測りのすべてでいちばん悪い**（+2.1〜+4.6%）。読む形（1.0）と素の RoPE は ±1.4% の内で向きがそろわない（引き分け）。
- 変換器は `normalize()` の中の `yarn_attention()` で `attn_factor` を transformers の `attention_factor`（倍率そのもの）に直す。エンジンは今までどおり `attention_factor` を読む。`attention_factor` か `mscale` と一緒に書かれた `attn_factor` は、実装ごとに読みが分かれるので今までどおり断る。
- **モデルは一覧に無い**（一覧にあるのは R1 の Distill の Qwen 1.5B・7B と Llama 8B だけ）。足すかは持ち主の判断。いまは `?hf=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` で開ける（前は「its yarn RoPE scaling sets attn_factor, which the engine does not read」で断った）。

## 各実装の読み（式）

倍率 m は cos と sin に掛かるので、attention の点数は m² 倍になる。f は factor（4）。

| 実装 | m | このモデル |
|---|---|---|
| transformers 5.16.1（`_compute_yarn_parameters`） | `attention_factor` があればそれ、`mscale` と `mscale_all_dim` があれば (0.1·mscale·ln f + 1) / (0.1·mscale_all_dim·ln f + 1)、無ければ 0.1 ln f + 1。**`attn_factor` は読まない** | 1.1386 |
| vLLM 0.8.5・0.9.0（`YaRNScalingRotaryEmbedding`） | (0.1 ln f + 1) × `attn_factor` | **1.0** |
| vLLM の main（2026-11-20 に読んだ） | transformers と同じ名前だけ（`attn_factor` は渡さない） | 1.1386 |
| SGLang 0.4.6.post1 と main | (0.1 ln f + 1 か mscale の比) × `attn_factor` | **1.0** |
| llama.cpp（master 7716ec8） | `rope_yarn()` の mscale = cparams.yarn_attn_factor × (1 + 0.1 ln f)。cparams.yarn_attn_factor = get_mscale(f, 1) / (1 + 0.1 ln f) × `rope_attn_factor` = `rope_attn_factor`（`yarn_log_mul` の無いとき、llama-context.cpp）。`rope_attn_factor` は GGUF の `rope.scaling.attn_factor`（無ければ 1） | 下の行 |
| llama.cpp の変換器（`conversion/base.py`） | config の `attention_factor` か `attn_factor` を `rope.scaling.yarn_attn_factor` に書く。この鍵を読むのは `src/models/grok.cpp` だけ（GitHub のコード検索）で、qwen3 は読まない | 1.1386 |
| 公開の GGUF（unsloth・lmstudio-community・bartowski・cecloud33 の Q8_0） | どれも `attn_factor` の鍵を持たない（factor 4・元の文脈 32768 だけ） | 1.1386 |
| mmnga の GGUF | yarn の鍵が無い（素の RoPE） | 素 |
| このエンジン（変えた後） | `attention_factor` = `rope_magnitude(残り)` × `attn_factor` | **1.0** |

- カード（`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` @`6e8885a6`）: 「Qwen3-8B と同じ形で動かせる。ただし設定のファイルは全部このリポジトリのものを使う」。Qwen3-8B のカード（@`b968826d`）は配備に `sglang>=0.4.6.post1` か `vllm>=0.8.5` を挙げる。どちらの版も `attn_factor` を掛ける（上の表）。つまり作り手の勧める道では m = 1.0。
- DeepSeek がこの値を何で決めたか（学習のときの倍率）は書かれていない（未確認）。値が 1 / (0.1 ln 4 + 1) と一致するので、「yarn の角で、温度なし」を意図したと読める。

## perplexity（ページの道）

道具はブランチ `t259-probe` の `tests/t259_probe.sh`（本線には入れない）: bartowski の Q8_0（@`75b8e3d4`）を原本の config.json・語彙と合わせて int8 に変換し（config.json の `attn_factor` は外して変換、9,215,463,452 バイト、BOS 151643）、3 つの読みは options の `rope_scaling` だけを替える: yarn（`attn_factor` なし: m 1.1386）、attn（`attention_factor: 1.0`）、plain（`{}`）。`tests/perplexity.ts --file --rows=0 --wide`（`--window=` を足した。既定は 512 のまま）で日本語と英語の Wikipedia（T85 の記事）。

| 文と長さ | yarn（1.1386） | attn（1.0） | 素の RoPE | yarn ÷ attn − 1 | attn ÷ 素 − 1 |
|---|---:|---:|---:|---:|---:|
| 日本語 1500 トークン・窓 512 | 16.529 | 15.802 | 16.027 | +4.60% | −1.40% |
| 英語 1500 トークン・窓 512 | 13.563 | 13.239 | 13.194 | +2.45% | +0.34% |
| 日本語 4096 トークン・窓 4096 | 10.543 | 10.324 | 10.331 | +2.12% | −0.07% |
| 英語 4096 トークン・窓 4096 | 8.523 | 8.217 | 8.113 | +3.72% | +1.28% |

- run: 1500 トークンは 44788389143（yarn、Xeon Platinum 8573C）・44788392635（attn、EPYC 9V45）・44788397193（plain、Xeon 8573C）。4096 トークンは 44788401545（yarn、EPYC 9V45）・44788406235（attn、EPYC 7763）・44788409943（plain、EPYC 7763）。同じ CPU で 3 つを並べた対照 44797804867（EPYC 7763、日本語 1500・512）が 16.529・15.802・16.027 で、上の行と全部の桁が同じ（CPU で数は変わらない）。道具の短い走り 44786910694（32 トークン）。
- 読み: 倍率 1.1386 は 4 つの全部で損（2〜5%）。T235 の Ternary Bonsai と同じく、ページの文脈（4096 以下）では yarn の温度が損になる。角の違い（attn と素）は ±1.4% で、向きが文によって逆（T235 の Bonsai では素が 2〜7% 良かったが、このモデルでは差が出ない）。
- 測っていないもの: transformers の参照（8B の float32 は 16 GB のランナーに入らない）、書かせた文、8192 より長い文脈、カードの課題の点。

## 変えたもの

- `src/python/convert/config.py` の `yarn_attention()`（`normalize()` が呼ぶ）と、GGUF だけの変換（`Conversion.from_gguf()`）の config に同じものを当てる。GGUF の `rope.scaling.attn_factor`（llama.cpp でも同じ意味）も読むようになった。`rope.scaling.yarn_attn_factor`・`yarn_log_multiplier`・`yarn_ext_factor` は今までどおり断る。
- 今あるモデルで `attn_factor` を書くものは無いので、一覧のどの変換の出力も変わらない（`tests/conversions_compare.py` を origin/main と: 216 項目すべて同じ。CI の `reference_qwen35.sh --only=real` も通った: run 44806949047）。だから `CONVERTER` は上げない（前に断っていたので、保存された変換も無い）。
- 公開の GGUF（`attn_factor` を持たない）を原本の config.json（持つ）と合わせると、`gguf_agrees()` が「yarn RoPE scaling が違う」で断る。GGUF の重みで一覧に足すなら、このずれの扱いを決める（下）。

## 一覧に足すなら

- 大きさ: 原本の bf16 は約 16.4 GB、Q8_0 の GGUF は約 8.7 GB、int8 は 9.2 GB（64 ビットのメモリ: Chrome と Firefox）。CPU だけで 8B（T247 の Qwen3.5 9B と同じ級）。
- 取り込み元: (a) 原本の safetensors（取得 16.4 GB、`gguf_agrees()` は通らなくてよい）か、(b) bartowski などの Q8_0（取得 8.7 GB）。(b) は `gguf_check.py tensors` を先に通し、`gguf_agrees()` が「GGUF は `attn_factor` を書かない（llama.cpp の変換器が別の鍵に書くか、書かない）」ことを許す形が要る（例: 原本の yarn が `attn_factor` だけ多いときは合うと読む）。
- RoPE: 勧めは config のまま（m 1.0）。素の RoPE と引き分けなので、Bonsai のように `rope_scaling: {}` にする理由は無い。
- ほか: 書式（DeepSeek の書式、考える形）と BOS（151643）・止まりを `format_check.py` と `start_check` で、名前と note と標本抽出（カードは温度 0.6・top-p 0.95）を持ち主に。
