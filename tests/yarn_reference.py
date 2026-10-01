# yarn_reference.py (T235, a probe for CI, not for main): what transformers itself makes of Ternary-Bonsai 1.7B under
# the three readings of its RoPE, to hold the engine's yarn to: config.json as it is (yarn, factor 4), the same with
# attention_factor 1 (yarn's angles, cos and sin not scaled), and without rope_scaling (plain RoPE). The perplexity
# of the same texts with the same ids the page sends ([151643] + the tokenizer's), and 16 greedy tokens of the list's
# prompt in the model's template.
#
#   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers==4.57.6
#   python3 tests/yarn_reference.py <directory for the download> <text file>:<tokens> ...
import json
import math
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO, REVISION, BOS = "prism-ml/Ternary-Bonsai-1.7B-unpacked", "3aca840085293d026ce6f6b80fafdae937fd2eeb", 151643
PROMPT = "<|im_start|>user\nこれからの流行りを3つ挙げてください。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

directory = Path(snapshot_download(REPO, revision=REVISION, local_dir=Path(sys.argv[1]) / "original"))
tokenizer = AutoTokenizer.from_pretrained(directory)
texts = []
for argument in sys.argv[2:]:
    path, count = argument.rsplit(":", 1)
    ids = [BOS] + tokenizer(Path(path).read_text(), add_special_tokens=False)["input_ids"][:int(count)]
    texts.append((path, ids))
    print(f"{path}: {len(ids) - 1} tokens after the BOS, the first {ids[1:9]}, their sum {sum(ids)}", flush=True)
prompt = [BOS] + tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
print(f"the prompt: {len(prompt) - 1} tokens after the BOS", flush=True)

config = json.loads((directory / "config.json").read_text())
yarn = config["rope_scaling"]
for name, scaling in (("config.json as it is (yarn)", yarn), ("yarn with attention_factor 1", {**yarn, "attention_factor": 1.0}),
                      ("no rope_scaling (plain RoPE)", None)):
    variant = Path(sys.argv[1]) / name.split(" (")[0].replace(" ", "-").replace(".", "-")
    variant.mkdir(parents=True, exist_ok=True)
    for file in directory.iterdir():
        if file.is_file() and file.name != "config.json" and not (variant / file.name).exists():
            (variant / file.name).symlink_to(file.resolve())
    changed = {key: value for key, value in config.items() if key != "rope_scaling"}
    if scaling:
        changed["rope_scaling"] = scaling
    (variant / "config.json").write_text(json.dumps(changed))
    began = time.time()
    model = AutoModelForCausalLM.from_pretrained(variant, torch_dtype=torch.float32)
    model.eval()
    rotary = model.model.rotary_emb
    print(f"{name}: rope_type {rotary.rope_type}, attention_scaling {rotary.attention_scaling:.6f}, inv_freq[16:36] over the "
          f"plain ones {[round(float(v), 4) for v in (rotary.inv_freq * 1e6 ** (torch.arange(0, 128, 2) / 128))[16:36]]}", flush=True)
    with torch.no_grad():
        for path, ids in texts:
            logits = model(torch.tensor([ids])).logits[0, :-1].double()
            nll = torch.nn.functional.cross_entropy(logits, torch.tensor(ids[1:]), reduction="mean")
            print(f"{name}: {path}: perplexity {math.exp(float(nll)):.3f} over {len(ids) - 1} tokens", flush=True)
        # the four most likely tokens after the prompt and each of the first four tokens every reading writes
        forced = prompt + [87752, 15322, 5373, 50230]
        logits = model(torch.tensor([forced])).logits[0].double()
        for pos in range(len(forced) - 5, len(forced)):
            top = torch.topk(logits[pos], 4)
            print(f"{name}: logits at {pos}: {[[int(i), round(float(v), 3)] for v, i in zip(top.values, top.indices)]}", flush=True)
        print(f"{name}: the prompt's ids {prompt}", flush=True)
        written = model.generate(torch.tensor([prompt]), max_new_tokens=16, do_sample=False)[0, len(prompt):]
        print(f"{name}: greedy {json.dumps(tokenizer.decode(written), ensure_ascii=False)} {written.tolist()}", flush=True)
    print(f"{name}: {time.time() - began:.0f} s", flush=True)
    del model
