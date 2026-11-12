# reference_phi3.py
# T258: the engine's Phi-3 (Microsoft's Phi-3 mini and Phi-4 mini, transformers' model_type "phi3": a Llama whose q, k
# and v are one matrix and whose gate and up are another, whose RoPE may turn part of each head) against transformers
# itself (Phi3ForCausalLM of 5.16.1). PyTorch is not on the development machine, so this runs in CI:
#
#   python tests/reference_phi3.py <directory for the download> [--only=made-up|real|fetch] [--model=phi-3|phi-4]
#                                  [--positions=96] [--layers=8] [--lean=on] [--faults=off]
#   (--layers=32 --lean=on: every layer, transformers held in bfloat16 as published and each layer made float32 for its own
#   pass, the faults left off: T258's review)
#
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_phi3.sh" --ref <branch> --grep "phi3:"
#
# Every line of the log begins with "phi3:".
#   made-up: tiny random models (tests/made_up.py's phi3_model()), written to a folder and loaded by transformers.
#     transformers' logits, over the whole text at once and token by token with its cache, against the naive reference
#     of the unit tests (test_phi3.naive_phi3_logits) and against the engine, converted the way the page converts.
#   real: the model at a fixed revision, in float32, its first --layers layers only (a config.json that says so beside
#     the published files): transformers holds a whole 3.8B in float32 in 15 GB, more than a runner of 16 GB has beside
#     the engine. The same token ids through transformers and through the engine (NumPy) of a float32 file and of an int8
#     one (the page's), each in a process of its own (T283: one process that held both lost the runner), the files mapped.
#   faults: the float32 engine with one thing wrong at a time (faults()), each against the same numbers of transformers:
#     every fault has to be past the line the right engine is held to.
# The lines it holds the engine to are at the end of each part; anything past them is exit 1.
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from llama2_numpy import Llama  # noqa: E402
from conducting import Directory, Mapped, converted  # noqa: E402
from made_up import phi3_model, safetensors_file  # noqa: E402
from reference import (TEXT, differences, ended, engine_logits, fetch, float32_file, line_of, logits_failed,  # noqa: E402
                       logits_said, made_up_engine, option, options_shown)
from test_phi3 import LONGROPE, naive_phi3_logits  # noqa: E402

# by --model: the repository, its revision, its two shards, and its BOS (the converter's: the format's first token,
# which a plain text has not, so the text begins with the tokenizer's own <s> or <|endoftext|>)
MODELS = {
    "phi-3": ("microsoft/Phi-3-mini-4k-instruct", "f39ac1d28e925b323eae81227eaba4464caced4e", 1),
    # (T258.2: three quarters of a head turned, longrope's attention factor of 1.190 in cos and sin)
    "phi-4": ("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083", 199999),
}
SHARDS = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]
LOGITS_LINE = 1e-3  # the least of the logits' line (made-up models: 2e-3)
# the real part's phases each in a process of its own (real()); tests/reference_dry.py runs them in its own, on its
# stand-ins, which a process of their own would not have
SEPARATE = True
# what main() read of the command line (--model, --layers, --positions, --only=fetch), and these where nothing did:
# the dry run calls real() of its own, whatever another tool's arguments left in sys.argv
RUN = {"model": "phi-3", "layers": 8, "positions": 96, "fetch": False, "lean": False, "faults": True}

say = lambda *parts: print("phi3:", *parts, flush=True)


def transformers_model(folder, lean=False):
    """lean (T258 review: every layer of a 3.8B, which in float32 is 15 GB, more than a runner of 16 GB holds beside the
    engine's file): the model is held as published, in bfloat16 (7.6 GB), and each decoder layer is turned to float32
    for its own forward pass only and back after it (a bfloat16 weight is exact in float32 and back), the embedding,
    the final norm and the classifier staying float32: the arithmetic is float32's all the same."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16 if lean else torch.float32)
    model.eval()
    if lean:
        for part in (model.model.embed_tokens, model.model.norm, model.lm_head):
            part.float()
        for layer in model.model.layers:
            layer.register_forward_pre_hook(lambda module, args: module.float() and None)
            layer.register_forward_hook(lambda module, args, output: module.bfloat16() and None)
    return model


def at_once_and_stepped(model, ids):
    """transformers' logits of ids at once, and token by token with its cache"""
    import torch
    tensor = torch.tensor([ids])
    with torch.no_grad():
        whole = model(input_ids=tensor).logits[0].float().numpy()
        past, stepped = None, []
        for at in range(len(ids)):
            out = model(input_ids=tensor[:, at:at + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            stepped.append(out.logits[0, -1].float().numpy())
    return whole, np.stack(stepped)


# ------------------------------------------------------------------------------------------------------ made-up
MADE_UP = {"a Phi-3 mini's": dict(), "three quarters of a head turned, tied": dict(rotary=0.75, shared=True),
           "one key head, three layers": dict(n_kv_heads=1, layers=3), "a quarter turned, another base": dict(rotary=0.25, theta=500.0),
           # (T258.2: longrope, as tests/test_phi3.py's; transformers reads its short factors within the original context)
           "longrope, three quarters turned": LONGROPE,
           "longrope, its factor said": {**LONGROPE, "scaling": {**LONGROPE["scaling"], "factor": 2.0}}}


def made_up(directory):
    failed = False
    for name, settings in MADE_UP.items():
        failed |= made_up_one(name, settings, directory)
    return failed


def made_up_one(name, settings, directory=HERE.parent / ".tmp" / "reference-phi3"):
    """One made-up model, written to a folder and loaded by transformers as a published one is: the unit tests'
    reference and the engine are held to ten times what transformers' own two ways differ by."""
    rng = np.random.default_rng(0)
    # (within a longrope model's original context of 16, past which transformers turns by its long factors)
    ids = [int(token) for token in rng.integers(0, 320, 16 if "scaling" in settings else 40)]
    tensors, config = phi3_model(**{"seq_len": 64, **settings})
    folder = directory / "made-up" / name.replace(" ", "-").replace(",", "").replace("'", "")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.safetensors").write_bytes(safetensors_file(tensors))
    (folder / "config.json").write_text(json.dumps({**config, "torch_dtype": "float32"}))
    model = transformers_model(folder)
    whole, stepped = at_once_and_stepped(model, ids)
    floor = differences(whole, stepped)[0]
    label = f"made-up, {name}"
    say(f"{label}: transformers at once and token by token: {floor:.2e} apart")
    line = line_of(floor, 2 * LOGITS_LINE)
    naive = np.stack(naive_phi3_logits(tensors, config, ids))
    failed = logits_failed(say, f"{label}: the unit tests' reference against transformers", naive, stepped, line)
    llama, options, _ = made_up_engine(tensors, config)
    failed |= logits_failed(say, f"{label}: the engine against transformers", engine_logits(llama, ids), stepped, line)
    del model
    gc.collect()
    return failed


# --------------------------------------------------------------------------------------------------------- real
def faults(llama):
    """The real model's engine with one thing wrong at a time: (what, how to make it, how to put it back). What is
    wrong is put in a copy in the attribute's place (a mapped file's arrays are read-only), and the original put back."""
    kept = {}

    def replaced(*names, made):
        def make():
            for name in names:
                kept[name] = getattr(llama, name)
            for name, value in made(*[kept[name] for name in names]).items():
                setattr(llama, name, value)
        return make, lambda: [setattr(llama, name, kept.pop(name)) for name in names]

    def scaled(name, layer, by):
        def made(stack):
            stack = np.array(stack)
            stack[layer] *= np.float32(by)
            return {name: stack}
        return replaced(name, made=made)

    tables = ("freq_cis_real", "freq_cis_imag")
    # the angles of every position the first position's (RoPE turns nothing), and the angles one position behind
    unturned = lambda real, imag: {name: np.repeat(table[:1], len(table), axis=0) for name, table in zip(tables, (real, imag))}
    behind = lambda real, imag: {name: np.concatenate([table[:1], table[:-1]]) for name, table in zip(tables, (real, imag))}
    return [("the gate and the up swapped", *replaced("w1", "w3", made=lambda w1, w3: {"w1": w3, "w3": w1})),
            ("RoPE turning nothing", *replaced(*tables, made=unturned)),
            ("RoPE a position behind", *replaced(*tables, made=behind)),
            ("the first layer's v a quarter larger", *scaled("wv", 0, 1.25)),
            ("the last layer's k a quarter larger", *scaled("wk", len(llama.wk) - 1, 1.25)),
            # (T258.2: a longrope model's tables without its attention factor, where it has one)
            *([("RoPE without longrope's attention factor",
                *replaced(*tables, made=lambda real, imag: {name: table / np.float32(llama.rope_magnitude)
                                                            for name, table in zip(tables, (real, imag))}))]
              if getattr(llama, "rope_magnitude", 1.0) != 1.0 else [])]


def real_folder(directory, name):
    """The published files, and beside them a folder whose config.json says the first --layers layers (the shards
    linked): what both transformers and the converter read."""
    repo, revision, _ = MODELS[name]
    folder, cut = directory / name, directory / f"{name}-cut"
    folder.mkdir(parents=True, exist_ok=True)
    cut.mkdir(parents=True, exist_ok=True)
    for file in FILES + SHARDS:
        fetch(file, folder, repo, revision)
    config = json.loads((folder / "config.json").read_text())
    (cut / "config.json").write_text(json.dumps({**config, "num_hidden_layers": RUN["layers"]}))
    for file in FILES[1:] + SHARDS:
        if not (cut / file).exists():
            os.symlink((folder / file).resolve(), cut / file)
    return cut


def real(directory):
    name = RUN["model"]
    real_folder(directory, name)
    if RUN["fetch"]:
        return False
    if not SEPARATE:
        return real_transformers(directory, name) or any([real_engine(directory, name, dtype) for dtype in ("float32", "int8")])
    failed = False
    for phase in ("transformers", "float32", "int8"):
        said = subprocess.run([sys.executable, __file__, str(directory), f"--phase={phase}", f"--model={name}",
                               f"--positions={RUN['positions']}", f"--layers={RUN['layers']}",
                               f"--lean={'on' if RUN['lean'] else 'off'}", f"--faults={'on' if RUN['faults'] else 'off'}"])
        say(f"real, the phase {phase}: exit {said.returncode}")
        failed |= said.returncode != 0
        if phase == "transformers" and said.returncode != 0:
            break
    return failed


def real_ids(folder, bos):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    return ([bos] + tokenizer.encode(TEXT, add_special_tokens=False).ids)[:RUN["positions"]]


def real_transformers(directory, name):
    """transformers' logits at once and token by token, into a file"""
    folder = directory / f"{name}-cut"
    ids = real_ids(folder, MODELS[name][2])
    began = time.perf_counter()
    model = transformers_model(folder, RUN["lean"])
    say(f"real, {name}: transformers loaded {model.config.num_hidden_layers} layers in {time.perf_counter() - began:.0f} s")
    whole, stepped = at_once_and_stepped(model, ids)
    say(f"real, {name}: transformers at once and token by token: {differences(whole, stepped)[0]:.2e} apart")
    np.savez(directory / f"{name}-transformers.npz", whole=whole, stepped=stepped)
    return False


def real_engine(directory, name, dtype):
    """the engine (NumPy) on a file of this dtype against transformers' numbers (and, of float32, the faults)"""
    import mmap
    repo, revision, bos = MODELS[name]
    folder = directory / f"{name}-cut"
    ids = real_ids(folder, bos)
    theirs = dict(np.load(directory / f"{name}-transformers.npz"))
    line = line_of(differences(theirs["whole"], theirs["stepped"])[0], LOGITS_LINE)
    path, hf = directory / f"{name}-{dtype}.bin", {"repo": repo, "revision": revision, "weights": "model.safetensors",
                                                   "config": "config.json", "tokenizer": "tokenizer.json"}
    began = time.perf_counter()
    if dtype == "float32":
        conversion, seconds = float32_file(Directory(folder), hf, path)
    else:
        sink = Mapped(path)
        conversion = converted(Directory(folder), hf, dtype=dtype, sink=sink)
        sink.data.flush()
        del sink.data
        seconds = time.perf_counter() - began
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    say(f"real, {name}, {dtype}: converted in {seconds:.0f} s, {path.stat().st_size / 1e9:.3f} GB; options {options_shown(options)}")
    with open(path, "rb") as file:
        checkpoint = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    llama = Llama(checkpoint, conversion.tokenizer, **options)
    began = time.perf_counter()
    ours = engine_logits(llama, ids)
    say(f"real, {name}, {dtype}: the engine ran {len(ids)} positions in {time.perf_counter() - began:.0f} s")
    if dtype != "float32":
        logits_failed(say, f"real, {name}, {dtype}: the engine (the page's rounding) against transformers (said, no line)", ours,
                      theirs["stepped"], float("inf"))
        path.unlink()
        return False
    logits_said(say, f"real, {name}", theirs["stepped"], ours)
    failed = logits_failed(say, f"real, {name}, float32: the engine against transformers", ours, theirs["stepped"], line, gap=True)
    weakest = None
    for what, make, undo in faults(llama) if RUN["faults"] else []:
        make()
        try:
            largest = differences(engine_logits(llama, ids), theirs["stepped"])[0]
        finally:
            undo()
        say(f"faults: {what}: largest difference {largest:.2e} ({largest / line:.1f} times the line)"
            f"{'' if largest > line else ' — FAILED: the line lets it through'}")
        failed |= not largest > line
        weakest = min(weakest or (largest, what), (largest, what))
    if weakest:
        say(f"faults: the weakest, {weakest[1]}, is {weakest[0] / line:.1f} times the line")
    path.unlink()
    return failed


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    RUN.update(model=option("model", "phi-3"), layers=int(option("layers", "8")), positions=int(option("positions", "96")),
               fetch=option("only") == "fetch", lean=option("lean") == "on", faults=option("faults") != "off")
    phase, name = option("phase"), RUN["model"]
    if phase:
        failed = real_transformers(directory, name) if phase == "transformers" else real_engine(directory, name, phase)
        sys.exit(1 if failed else 0)
    only, failed = option("only"), False
    if only in (None, "made-up"):
        failed |= made_up(directory)
    if only in (None, "real", "fetch"):
        failed |= real(directory)
    if only == "fetch":
        return
    ended(say, failed)


if __name__ == "__main__":
    main()
