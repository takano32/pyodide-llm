# probe_t249.py (T249, a probe: not for main): three base models wrote nonsense on the page's engine (rinna's
# japanese-gpt2 xsmall, stockmark's gpt-neox-japanese 1.4B, LINE's japanese-large-lm 1.7B). Is it the model with what
# the page puts in front of the text, or the engine? For each, greedy and for the same prompt:
#   - transformers and torch on the original's safetensors (float32), with the IDs of the page's tokenizer and four
#     ways to begin: the page's BOS in front, nothing in front, the EOS in front, the EOS after the text (what
#     T5Tokenizer does when called);
#   - the engine's NumPy forward pass on the float32 conversion of the list's source (a GGUF's values where it is one),
#     with the page's BOS.
#
#   python3 tests/probe_t249.py <directory> <model id> ...
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import format_check  # noqa: E402
from format_check import llama2_numpy  # noqa: E402

COUNT = 24
accepted = set(inspect.signature(llama2_numpy.Tokenizer.__init__).parameters) - {"self", "data", "vocab_size", "kind"}
directory, only = Path(sys.argv[1]), sys.argv[2:]
for entry in format_check.entries():
    if entry["id"] not in only:
        continue
    id = entry["id"]
    made = format_check.conversion(entry, *format_check.fetch(entry, directory / "small"))
    options = {**made.options, **entry.get("options", {})}
    options.pop("template", None)
    tokenizer = llama2_numpy.Tokenizer(made.tokenizer, abs(made.stream.header[5]), kind=options["tokenizer_kind"],
                                       **{key: value for key, value in options.items() if key in accepted})
    ids = tokenizer.encode(entry["prompt"])
    bos, eos = options["bos"], [stop for stop in options["stop_tokens"] if stop != options["bos"]] or [options["bos"]]
    original = entry["hf"].get("vocabulary") or entry["hf"]
    print(f"probe {id}: prompt {entry['prompt']!r} is {ids}, bos {bos}, stops {options['stop_tokens']}", flush=True)

    def decoded(tokens, before):
        text, previous = b"", before
        for token in tokens:
            text += tokenizer.decode(previous, token, bos)
            previous = token
        return text.decode("utf-8", errors="replace")

    model = AutoModelForCausalLM.from_pretrained(original["repo"], revision=original["revision"], torch_dtype=torch.float32)
    model.eval()
    for name, begun in (("the page's BOS in front", [bos] + ids), ("nothing in front", ids), ("the EOS in front", [eos[0]] + ids),
                        ("the EOS after the text", ids + [eos[0]])):
        with torch.no_grad():
            written = model.generate(torch.tensor([begun]), attention_mask=torch.ones((1, len(begun)), dtype=torch.long),
                                     max_new_tokens=COUNT, do_sample=False, pad_token_id=eos[0])[0].tolist()[len(begun):]
        print(f"probe {id}: transformers, {name}: {decoded(written, begun[-1])!r} {written[:8]}", flush=True)
    del model

    source = subprocess.check_output([sys.executable, str(HERE / "hf_fetch.py"), id, str(directory / "downloads")], text=True).strip().split("\n")[-1]
    for dtype in ("float32", "int8"):
        out = directory / f"{id}.{dtype}"
        subprocess.run([sys.executable, str(HERE / "perplexity_prepare.py"), source, str(out), dtype], check=True, stdout=subprocess.DEVNULL)
        llama = llama2_numpy.Llama(np.memmap(f"{out}.bin", dtype=np.uint8, mode="r"), made.tokenizer, kernels=None, **{**options, "dtype": dtype})
        written = "".join(llama.generate(entry["prompt"], steps=len(ids) + COUNT, temperature=0.0, echo=False))
        print(f"probe {id}: the engine's NumPy on {dtype}, the page's BOS in front: {written!r}", flush=True)
        del llama
        Path(f"{out}.bin").unlink()
    subprocess.run(["rm", "-rf", str(directory / "downloads")])
