# t236_bos.py (T236's review, a probe for CI, not for main): what the first token of the page's text costs Qwen3.5 0.8B,
# by a metric and not by six readings. The page began every text with <|endoftext|> (the converter's BOS) until T236,
# and now begins the entry's formats with their own first token <|im_start|> (248045: the very ids the real template
# makes, none more). Two parts:
#
#   chat: on 24 prompts, the form that answers at once and the thinking form. The real model (transformers, float32,
#     the original's weights) writes greedily from the real template's ids (prefix A) and from <|endoftext|> + the
#     real ids (prefix B: the page's design before T236). Then the same continuations are scored under both prefixes:
#     the mean negative log likelihood of each continuation (the real model's own under B is how far the old design
#     moved the model), the Kullback-Leibler distance between the two prefixes' next-token distributions along the
#     real model's answer, how often their most likely tokens are the same, and how many tokens the two greedy
#     answers share. And the page's own engine (NumPy over the entry's GGUF converted to float32, with the entry's
#     options and the template of the entry: the page's ids) the same way against transformers' answer: how
#     long it writes what the real model writes, under the new design and its NLL of the real answer under both.
#   plain: the perplexity of 4 texts (2 English, 2 Japanese) of 512 tokens with no first token, <|endoftext|>, and
#     <|im_start|> in front, over the same targets, by position (what ?hf=Qwen/Qwen3.5-0.8B has: the converter's
#     BOS and no format).
#
#   python tests/t236_bos.py --original <dir> --gguf <out> --entries <json> [--tokens 40] [--texts a.txt ...] | --tiny
import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
started = time.time()

PROMPTS = [
    "これからの流行りを3つ挙げてください。", "日本でいちばん高い山は何ですか？", "17 × 24 はいくつですか？",
    "春の季語を使って俳句を一つ作ってください。", "「ありがとう」を英語で言うと？", "健康のために毎日できることを3つ教えてください。",
    "東京から大阪まで新幹線で何時間かかりますか？", "犬と猫の違いを簡単に説明してください。",
    "次の文を英語に訳してください。「今日は天気がいいので、散歩に行きます。」", "光合成とは何ですか？一文で答えてください。",
    "自己紹介を短くしてください。", "夏目漱石の代表作を2つ挙げてください。",
    "What is the capital of Japan?", "Write a haiku about autumn.", "What is 17 times 24?",
    "Give me three tips for sleeping better.", "Explain photosynthesis in one sentence.",
    "What is the difference between a virus and a bacterium?", "Translate \"Good morning\" into French.",
    "Name three programming languages and one use for each.", "Why is the sky blue?",
    "Summarize the plot of Cinderella in two sentences.", "How many days are in a leap year?",
    "Suggest a name for a pet goldfish.",
]


def say(*parts):
    print(f"T236BOS [{time.time() - started:6.0f} s]", *parts, flush=True)


def log_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Reference:
    """transformers' model, float32, for the logits of the last positions of a text, and a greedy continuation"""

    def __init__(self, model, endoftext, im_end):
        self.model, self.endoftext, self.im_end = model, endoftext, im_end

    def tail(self, prefix, cont):
        """log probabilities (len(cont), vocabulary) of the tokens of cont given the prefix and what precedes each"""
        import torch
        ids = torch.tensor([prefix + cont])
        with torch.no_grad():
            out = self.model(input_ids=ids, use_cache=False, logits_to_keep=len(cont) + 1)
        return log_softmax(out.logits[0, :-1].float().numpy())

    def write(self, ids, n):
        import torch
        inputs = torch.tensor([ids])
        with torch.no_grad():
            out = self.model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=n, do_sample=False,
                                      eos_token_id=[self.endoftext, self.im_end], pad_token_id=self.endoftext)
        return out[0, len(ids):].tolist()


def nll(lp, cont):
    return float(-lp[np.arange(len(cont)), cont].mean())


def chat_part(ref, count, n, forms, ids_of, engine_part):
    rows = []
    for form in forms:
        for index in range(count):
            began = time.time()
            ids_a = ids_of(form, index)
            ids_b = [ref.endoftext] + ids_a
            c_a, c_b = ref.write(ids_a, n), ref.write(ids_b, n)
            row = {"form": form, "prompt": index, "n_a": len(c_a), "n_b": len(c_b), "same_first": common(c_a, c_b)}
            lp = {(p, c): ref.tail(prefix, cont) for p, prefix in (("A", ids_a), ("B", ids_b))
                  for c, cont in (("cA", c_a), ("cB", c_b))}
            row["nll_cA_A"], row["nll_cA_B"] = nll(lp["A", "cA"], c_a), nll(lp["B", "cA"], c_a)
            row["nll_cB_B"], row["nll_cB_A"] = nll(lp["B", "cB"], c_b), nll(lp["A", "cB"], c_b)
            pa, pb = np.exp(lp["A", "cA"]), lp["B", "cA"]
            row["kl_A_B"] = float((pa * (lp["A", "cA"] - pb)).sum(axis=1).mean())
            row["top1_A_B"] = float((lp["A", "cA"].argmax(1) == lp["B", "cA"].argmax(1)).mean())
            row["first_A"], row["first_B"] = c_a[:12], c_b[:12]
            row["ids_a"] = len(ids_a)
            if engine_part is not None:
                row.update(engine_part(form, index, ids_a, c_a, lp["A", "cA"]))
            rows.append(row)
            say("CHAT", json.dumps(row, ensure_ascii=False), f"({time.time() - began:.0f} s)")
            del lp
            gc.collect()
    return rows


def summary(rows, label):
    def mean(key):
        values = [row[key] for row in rows if key in row]
        return sum(values) / len(values) if values else float("nan")
    keys = ["nll_cA_A", "nll_cA_B", "nll_cB_B", "nll_cB_A", "kl_A_B", "top1_A_B"]
    parts = [f"{key} {mean(key):.4f}" for key in keys]
    same = [row["same_first"] for row in rows]
    parts.append(f"A and B write the same first {sum(same) / len(same):.1f} tokens on average, "
                 f"all {rows[0]['n_a']}+ on {sum(1 for row in rows if row['same_first'] >= min(row['n_a'], row['n_b']))} of {len(rows)}")
    extra = [k for k in ("engine_same_as_A", "engine_tf_top1_A", "engine_tf_top1_B", "engine_nll_A", "engine_nll_B",
                         "engine_ids_ok") if any(k in row for row in rows)]
    parts += [f"{key} {mean(key):.4f}" for key in extra]
    say(f"SUMMARY {label} ({len(rows)} prompts): " + "; ".join(parts))
    # how many prompts the old design (B) scores worse than the new design on the real model's own answer, per prompt
    worse = sum(1 for row in rows if row["nll_cA_B"] > row["nll_cA_A"])
    say(f"SUMMARY {label}: the real answer is scored worse under B (<|endoftext|> first) than under A on {worse} of {len(rows)} prompts; "
        f"B's own greedy answer is scored worse under A than under B on "
        f"{sum(1 for row in rows if row['nll_cB_A'] > row['nll_cB_B'])} of {len(rows)}")


def plain_part(model, tokenizer, files, endoftext, im_start, length):
    import torch
    out = {}
    for path in files:
        text = Path(path).read_text()
        ids = tokenizer.encode(text, add_special_tokens=False).ids[:length]
        targets = ids[1:]
        result = {}
        for name, prefix in (("none", []), ("endoftext", [endoftext]), ("im_start", [im_start])):
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([prefix + ids[:-1]]), use_cache=False).logits[0].float().numpy()
            logits = logits[len(prefix):]  # (the position of each prefix token predicts the next one: only the text's own)
            losses = np.empty(len(targets))
            for start in range(0, len(targets), 128):
                lp = log_softmax(logits[start:start + 128])
                losses[start:start + 128] = -lp[np.arange(lp.shape[0]), targets[start:start + 128]]
            result[name] = losses
            del logits, lp
            gc.collect()
        buckets = [(0, 15), (15, 63), (63, 255), (255, len(targets))]
        for name, losses in result.items():
            parts = [f"[{a + 1}:{b + 1}] {math.exp(losses[a:b].mean()):.3f}" for a, b in buckets if b > a]
            say(f"PLAIN {Path(path).name} ({len(ids)} tokens, sha256 prefix of the text {hashlib.sha256(text.encode()).hexdigest()[:10]}) "
                f"prefix {name}: perplexity {math.exp(losses.mean()):.3f}; by position " + ", ".join(parts))
        out[Path(path).name] = {name: float(losses.mean()) for name, losses in result.items()}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original")
    parser.add_argument("--gguf")
    parser.add_argument("--entries")
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--plain-tokens", type=int, default=512)
    parser.add_argument("--texts", nargs="*", default=[])
    parser.add_argument("--only", default="")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--prompts", type=int, default=len(PROMPTS))
    args = parser.parse_args()
    import torch
    import transformers
    say(f"transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}, {torch.get_num_threads()} threads")
    if args.tiny:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        config = Qwen3_5TextConfig(vocab_size=320, hidden_size=32, intermediate_size=64, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=16, max_position_embeddings=256, eos_token_id=7,
                                   num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
                                   linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                                   linear_value_head_dim=8,
                                   rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                                    "mrope_section": [11, 11, 10], "mrope_interleaved": True})
        torch.manual_seed(1)
        model = Qwen3_5ForCausalLM(config).to(torch.float32).eval()
        ref = Reference(model, 7, 8)
        rng = np.random.default_rng(1)
        tiny = [[int(t) for t in rng.integers(10, 300, 12)] for _ in range(3)]
        chat_part(ref, 3, 8, ["at once"], lambda form, index: tiny[index], None)
        say("the tiny model's chat part ran")
        return
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    import tokenizers
    directory = Path(args.original)
    entries = {entry["id"]: entry for entry in json.loads(Path(args.entries).read_text())}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(directory), dtype=torch.float32).eval()
    chat = AutoTokenizer.from_pretrained(directory)
    ENDOFTEXT, IM_START, IM_END = 248044, 248045, 248046
    ref = Reference(model, ENDOFTEXT, IM_END)
    prompts = PROMPTS[:args.prompts]

    def ids_of(form, index):
        real = chat.apply_chat_template([{"role": "user", "content": prompts[index]}], add_generation_prompt=True, tokenize=True,
                                        enable_thinking=(form == "thinking"))
        return [int(t) for t in (real["input_ids"] if hasattr(real, "keys") else real)]

    # ---------------------------------------------------------------- the page's engine over the GGUF's weights
    from llama2_numpy import Llama
    engine = None
    if args.gguf:
        conversion = json.loads(Path(f"{args.gguf}.json").read_text())
        entry = entries["hf-qwen3.5-0.8b"]
        engine = Llama(np.memmap(f"{args.gguf}.bin", dtype=np.uint8, mode="r"), Path(f"{args.gguf}.tokenizer.bin").read_bytes(),
                       kernels=None, **{**conversion, **entry["options"]})
        say(f"the engine: options {json.dumps({k: (v if k != 'specials' else len(v)) for k, v in {**conversion, **entry['options']}.items()})}")
    stops = {ENDOFTEXT, IM_START, IM_END}
    templates = {"at once": entries["hf-qwen3.5-0.8b"]["template"], "thinking": entries["hf-qwen3.5-0.8b-thinking"]["template"]}

    def page_ids(form, index, design):
        """what the page sends: [bos] + the encoded format filled with the prompt ({prompt:trim})"""
        text = templates[form].replace("{prompt:trim}", prompts[index].strip())
        if design == "old":
            return [ENDOFTEXT] + engine.tokenizer.encode("<|im_start|>" + text, engine.specials)
        return [IM_START] + engine.tokenizer.encode(text, engine.specials)

    def engine_scores(ids, cont):
        """the engine's log probabilities of cont after ids, and its most likely tokens, token by token"""
        for pos, token in enumerate(ids[:-1]):
            engine.forward(token, pos, need_logits=False)
        lps, tops, token = [], [], ids[-1]
        for step, target in enumerate(cont):
            logits = np.asarray(engine.forward(token, len(ids) - 1 + step), dtype=np.float64)
            lp = log_softmax(logits)
            lps.append(float(lp[target]))
            tops.append(int(lp.argmax()))
            token = target
        return lps, tops

    def engine_writes(ids, n):
        for pos, token in enumerate(ids[:-1]):
            engine.forward(token, pos, need_logits=False)
        out, token = [], ids[-1]
        for step in range(n):
            logits = engine.forward(token, len(ids) - 1 + step)
            token = int(np.argmax(logits))
            out.append(token)
            if token in stops:
                break
        return out

    def engine_part(form, index, ids_a, c_a, lp_a):
        new, old = page_ids(form, index, "new"), page_ids(form, index, "old")
        # the page's ids are the real template's under the new design and <|endoftext|> + them under the old
        row = {"engine_ids_ok": float(new == ids_a and old == [ENDOFTEXT] + ids_a)}
        mine = engine_writes(new, args.tokens)
        row["engine_same_as_A"] = float(common(mine, c_a))
        row["engine_wrote_A"] = mine[:12]
        lps, tops = engine_scores(new, c_a)
        row["engine_nll_A"], row["engine_tf_top1_A"] = float(-np.mean(lps)), float(np.mean(np.array(tops) == np.array(c_a)))
        lps, tops = engine_scores(old, c_a)
        row["engine_nll_B"], row["engine_tf_top1_B"] = float(-np.mean(lps)), float(np.mean(np.array(tops) == np.array(c_a)))
        return row

    forms = [form for form in ("at once", "thinking") if form in args.only or not args.only]
    rows = chat_part(ref, len(prompts), args.tokens, forms, ids_of, engine_part if engine is not None else None)
    for form in forms:
        summary([row for row in rows if row["form"] == form], form)
    summary(rows, "both forms")
    del engine
    gc.collect()
    if args.texts:
        plain_part(model, tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json")), args.texts, ENDOFTEXT, IM_START,
                   args.plain_tokens)
    say("done")


if __name__ == "__main__":
    main()
