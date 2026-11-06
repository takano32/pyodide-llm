// The models built with the site (`make models`) and their unquantized originals.
// Every file of these two groups but two originals' (T653, below) is fetched when the site is built (see the Makefile): llm-jp-3 and tiny-lm are
// converted from their Hugging Face checkpoints by convert_hf.py, and the larger models are quantized to int8. bytes
// is the checkpoint size: it sizes the download buffer and the progress bar.
import { JAPANESE, STORY, llmJp, unigram } from "./formats.ts";
import { hf } from "./builders.ts";
import { greedy, sampled } from "./sampling.ts";
import type { Entry } from "./builders.ts";

export const BUILT: Entry[] = [
  { id: "tiny-lm", name: "SB Intuitions tiny-lm 29M", note: "日本語 / English · int8 · 33 MB", japanese: true,
    source: "sbintuitions/tiny-lm", checkpoint: "tiny-lm.bin", bytes: 32891932, tokenizer: "tiny-lm.tokenizer.bin",
    options: { dtype: "int8", ...unigram, stop_tokens: [1, 2] },
    generation: sampled(1.3), prompt: "これからの流行りは", placeholder: JAPANESE },
  { id: "llm-jp-3-150m", name: "llm-jp-3 150M", note: "日本語 / English · int8 · 171 MB", japanese: true,
    source: "llm-jp/llm-jp-3-150m", checkpoint: "llm-jp-3-150m.bin", bytes: 171395100, tokenizer: "llm-jp-3-150m.tokenizer.bin",
    options: { dtype: "int8", ...unigram, stop_tokens: [1, 2, 7] },
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { id: "stories260K", name: "Andrej Karpathy TinyStories 260K", note: "English · float32 · 1 MB · tiny", japanese: false,
    source: "karpathy/tinyllamas", checkpoint: "stories260K.bin", bytes: 1056540, tokenizer: "tok512.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories3_5M", name: "Ellis Hoag TinyStories 3.5M", note: "English · float32 · 15 MB · fast", japanese: false,
    source: "ellishg/tinyllamas", checkpoint: "stories3_5M-v4k.bin", bytes: 14887004, tokenizer: "tok4096.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories15M", name: "Andrej Karpathy TinyStories 15M", note: "English · int8 · 17 MB", japanese: false,
    source: "karpathy/tinyllamas", checkpoint: "stories15M.bin", bytes: 17101468, tokenizer: "tokenizer.bin", options: { dtype: "int8" },
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories42M", name: "Andrej Karpathy TinyStories 42M", note: "English · int8 · 47 MB", japanese: false,
    source: "karpathy/tinyllamas", checkpoint: "stories42M.bin", bytes: 46925852, tokenizer: "tokenizer.bin", options: { dtype: "int8" },
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  // the unquantized originals, to compare with int8
  { group: "original", id: "tiny-lm-f16", name: "SB Intuitions tiny-lm 29M (original)", note: "日本語 / English · float16 · 59 MB", japanese: true,
    source: "sbintuitions/tiny-lm", checkpoint: "tiny-lm.f16", bytes: 58724892, tokenizer: "tiny-lm.tokenizer.bin",
    options: { dtype: "float16", ...unigram, stop_tokens: [1, 2] },
    generation: sampled(1.3), prompt: "これからの流行りは", placeholder: JAPANESE },
  // T653: llm-jp-3 150M's original and stories42M's are not files of this site any more (it is smaller by 450 MiB): the
  // first is converted to float16 in the page from its repository, as the site's was by convert_hf.py (the same
  // tokenizer.bin; its options the site's, which the converter's are but stop_tokens: 7 is the converter's no stop),
  // and the second is fetched at its address as it is, a llama2.c checkpoint, read with this site's tokenizer.bin.
  // loaded: float16 is widened when loaded, three times the file as the site's was (src/models/memory.ts)
  { group: "original", id: "llm-jp-3-150m-f16", name: "llm-jp-3 150M (original)", note: "日本語 / English · fetches 305 MB → float16 305 MB", japanese: true,
    hf: hf("llm-jp/llm-jp-3-150m", "b112feef602fff752e4dac4c30af6a2c2fa41c7a"), download: 304649360, conversion: { dtype: "float16" }, loaded: 305161244 * 3,
    options: llmJp, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "original", id: "stories15M-f32", name: "Andrej Karpathy TinyStories 15M (original)", note: "English · float32 · 61 MB", japanese: false,
    source: "karpathy/tinyllamas", checkpoint: "stories15M.f32", bytes: 60816028, tokenizer: "tokenizer.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { group: "original", id: "stories42M-f32", name: "Andrej Karpathy TinyStories 42M (original)", note: "English · float32 · fetches 167 MB", japanese: false,
    source: "karpathy/tinyllamas", url: { checkpoint: "https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories42M.bin", tokenizer: "tokenizer.bin" },
    bytes: 167020572, options: {}, generation: greedy, prompt: "Once upon a time", placeholder: STORY },
];
