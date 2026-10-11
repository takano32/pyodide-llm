# fixed_outputs.py
# Real models against the text they wrote when a person read it and found it sound (T86). Twice last week every
# test passed while the page wrote broken text (T72: v permuted as well, rotary never passed on), because the
# synthetic models of the unit tests cannot tell a sensible sentence from a wrong one. These can.
#
# One small model per architecture and source, converted the way the page converts it (the conduct of a conversion,
# answered from the files fetched: tests/conducting.py) to float32, then 16 greedy tokens from the model's own prompt and template. float32
# because int8 changes its text with the order of additions (AGENTS.md); float32 with NumPy is the same, to the
# character, as float32 with the kernels. Where each model comes from, its prompt and template are read from
# src/models.js (with Node), so nothing here can drift from the list.
#
#   python3 tests/fixed_outputs.py <directory for the downloads> [--write] [--only=<id>,<id>]
#
# --write records what the models write now, into tests/fixtures/fixed-outputs.json: do that only after reading it.
# --only runs those models alone (a broken converter tried on the one model it should break, T204).
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from llama2_numpy import Llama  # noqa: E402
from conducting import Fetched, Mapped, converted as conducted  # noqa: E402

FIXTURES = HERE / "fixtures" / "fixed-outputs.json"
NEW_TOKENS = 16
# one per architecture and way in: GPT-NeoX, GPT-2, a Llama from a GGUF (T74), a Llama with a Unigram tokenizer.json,
# and a Qwen3 (T124: the norms of q and k, heads of 128 in a dim of 1024), from its GGUF and from its safetensors
# (T212). Pythia and GPT-2 come from GGUFs since
# T136's third stage, so a GPT-2 from a safetensors file too (T204): rinna's, whose names begin with "transformer.",
# whose Conv1D matrices the plan transposes as they come, and whose sentencepiece model normalizes as nmt_nfkc.
# The list takes Qwen3 from a GGUF since T203, so its safetensors, the way ?hf= opens a Qwen3 (the norms of q and k and
# the size of the heads read from config.json and the tensors' names), gets a fixed output of its own too (T212): the
# same entry with the weights of the original repository its vocabulary comes from, named as the page's hfEntry()
# names them (no tokenizer: the conduct's candidates, as for any repository nobody looked at). T235: Ternary-Bonsai 1.7B, the one model of PQ2_0 blocks and of yarn's RoPE (its angles and the longer
# cos and sin), and the one whose template comes from chat_template.jinja. Its float32 checkpoint is 6.9 GB.
# T229 and T236: a Qwen3.5 (hybrid attention: Gated DeltaNet layers between full-attention ones), the 0.8B, the same
# two ways: from the list's GGUF (llama.cpp's names for the linear-attention layers, the norms that come with their 1,
# -exp(A_log), the two small matrices of the gates as Q8_0 rounds them) and from the safetensors of the original (the
# revision tests/reference_qwen35.py holds the engine to transformers on, whose weights are one file under a shard's
# name). The format is the list's: the model's own calls a macro, which the converter does not read.
# T247: Qwen3.5 2B, the smallest of the sizes added then (heads that fill dim, where the 0.8B's do not), the same two
# ways. The text of the safetensors is what transformers' generate() writes for the same ids on the float32 original
# (tests/reference_qwen35.py --model=2B); the GGUF's, Q8_0's rounding of those weights, leaves it at the fourth token.
# Each float32 checkpoint is 7.5 GB.
# T253: Granite 4.2 3B, the smallest of the models whose q the converter scales (a Granite's attention multiplies its
# scores by config.json's attention_multiplier), from the list's GGUF, where the multiplier is metadata and q is turned
# as a Llama's. Its float32 checkpoint is 14.6 GB. T254: MiniCPM5 1B, the model of the two-stage pre-tokenizer and the
# first Llama from a GGUF whose heads are not dim / heads (128 in 1536).
# T260: an LFM2 (convolution layers among attention layers), LFM2.5 350M, the same two ways: from the list's GGUF
# (llama.cpp's names, the convolution without its axis of one, the layers said by their key-value heads) and from the
# safetensors of the original (the revision tests/reference_lfm2.py holds the engine to transformers on, which writes
# the same text for the same ids there). Each float32 checkpoint is 1.4 GB.
# {the id here: (the list's entry, the file of the original's weights)}
SAFETENSORS = {"hf-qwen3-0.6b-safetensors": ("hf-qwen3-0.6b", "model.safetensors"),
               "hf-qwen3.5-0.8b-safetensors": ("hf-qwen3.5-0.8b", "model.safetensors-00001-of-00001.safetensors"),
               "hf-qwen3.5-2b-safetensors": ("hf-qwen3.5-2b", "model.safetensors-00001-of-00001.safetensors"),
               "hf-lfm2.5-350m-safetensors": ("hf-lfm2.5-350m", "model.safetensors")}
MODELS = ["hf-pythia-70m", "hf-gpt2", "hf-japanese-gpt2-small", "hf-smollm2-135m-instruct", "hf-llm-jp-3-150m-instruct3",
          "hf-qwen3-0.6b", "hf-qwen3-0.6b-safetensors", "hf-ternary-bonsai-1.7b", "hf-qwen3.5-0.8b",
          "hf-qwen3.5-0.8b-safetensors", "hf-qwen3.5-2b", "hf-qwen3.5-2b-safetensors", "hf-granite-4.2-3b",
          "hf-minicpm5-1b", "hf-lfm2.5-350m", "hf-lfm2.5-350m-safetensors"]


def entries():
    wanted = [SAFETENSORS.get(id, (id,))[0] for id in MODELS]
    script = ("import('./src/models.js').then(({ MODELS }) => console.log(JSON.stringify("
              f"MODELS.filter((m) => {json.dumps(wanted)}.includes(m.id)))))")
    found = {entry["id"]: entry for entry in json.loads(subprocess.check_output(["node", "-e", script], cwd=HERE.parent))}
    for id, (of, weights) in SAFETENSORS.items():
        original = found[of]["hf"]["vocabulary"]
        found[id] = {**found[of], "id": id, "hf": {"repo": original["repo"], "revision": original["revision"],
                     "weights": weights, "config": "config.json"}}
    return [found[id] for id in MODELS]


def converted(entry, directory):
    """(the conversion of the entry to float32, the file of its checkpoint), by the conduct of a conversion (T374.4):
    the files it asks for are fetched as it asks for them (tests/fetching.py through conducting.Fetched: three tries,
    a download that stopped short asked for again, a 404 "not there"), each repository's into its own folder. The
    checkpoint goes into a file beside the weights, a memory map (Qwen3 0.6B's is 2.4 GB, T124)."""
    hf = entry["hf"]
    answerer = Fetched.of(hf, directory)
    sink = Mapped(answerer.folder("weights") / "float32.bin")
    conversion = conducted(answerer, hf, dtype="float32", sink=sink)
    sink.data.flush()
    return conversion, sink.path


def written(entry, conversion, checkpoint):
    """What the page would write with this model at temperature 0: the prompt in the model's template, the
    options of the conversion with those of src/models.js on top (as the worker merges them)."""
    options = {**conversion.options, **entry.get("options", {})}
    template = entry.get("template") or options.pop("template", None)
    options.pop("template", None)
    # filled() of src/models.js: {prompt:trim} trims (T138); the prompts here have no spaces to trim
    prompt = template.replace("{prompt:trim}", "{prompt}").replace("{prompt}", entry["prompt"]) if template else entry["prompt"]
    llama = Llama(np.memmap(checkpoint, dtype=np.uint8, mode="r"), conversion.tokenizer, kernels=None, **options)
    steps = len(llama.tokenizer.encode(prompt, llama.specials)) + NEW_TOKENS
    return "".join(llama.generate(prompt, steps=steps, temperature=0.0, echo=False))


def main():
    directory, write = Path(sys.argv[1]), "--write" in sys.argv
    only = [arg.split("=", 1)[1].split(",") for arg in sys.argv if arg.startswith("--only=")]
    if write and only:
        sys.exit("--write records every model: not with --only")
    expected = json.loads(FIXTURES.read_text()) if FIXTURES.exists() else {}
    got, failures = {}, []
    for entry in entries():
        if only and entry["id"] not in only[0]:
            continue
        conversion, checkpoint = converted(entry, directory)
        text = written(entry, conversion, checkpoint)
        checkpoint.unlink()  # 2.4 GB for each Qwen3 0.6B, 3.0 GB for Qwen3.5 0.8B: one float32 checkpoint at a time on the runner's disk
        got[entry["id"]] = {"prompt": entry["prompt"], "text": text}
        same = expected.get(entry["id"], {}).get("text") == text
        print(f"{'ok  ' if same else 'DIFF'} {entry['id']}: {json.dumps(text, ensure_ascii=False)}", flush=True)
        if not same:
            failures.append(entry["id"])
    if write:
        FIXTURES.parent.mkdir(exist_ok=True)
        FIXTURES.write_text(json.dumps(got, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {FIXTURES}")
    elif failures:
        print(f"FAILED: {', '.join(failures)} wrote something else than {FIXTURES.name} says")
        sys.exit(1)


if __name__ == "__main__":
    main()
