# reference_bitnet.py
# T283: the engine's BitNet b1.58 (Microsoft's: a Llama whose matrices are ternary, one scale a matrix, packed four
# values a byte; a norm in front of wo and one in front of w2; the square of a ReLU in the FFN's gate) against
# transformers itself (BitNetForCausalLM, AutoBitLinear of a file of quantization_mode "offline"). PyTorch is not on the
# development machine, so this runs in CI:
#
#   python tests/reference_bitnet.py <directory for the download> [--only=made-up|real|fetch] [--positions=96]
#   (--only=fetch: the real model's files into the directory and no more)
#
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_bitnet.sh" --ref <branch> --grep "bitnet:"
#
# The model rounds the input of every matrix to 8 bits by the largest of the token's values (transformers' ActQuant),
# which the engine does not: its NumPy forward pass multiplies floats, and the page's kernels round by groups of 32, as
# for every int8 and ternary model. So transformers runs twice: as the model is ("rounded"), and with ActQuant left out
# ("floats"), which the engine is held to. Every line of the log begins with "bitnet:".
#   made-up: tiny random models (tests/made_up.py's bitnet_model(): packed as the published file is), written to a folder
#     and loaded by transformers as the published one is (its deserializer unpacks them). transformers' logits, over the
#     whole text at once and token by token with its cache, against the naive reference of the unit tests
#     (test_bitnet.naive_bitnet_logits, rounded and not) and against the engine, converted the way the page converts.
#   real: microsoft/bitnet-b1.58-2B-4T at a fixed revision, float32. The same token ids through transformers (floats, and
#     rounded) and through the engine (NumPy) of a float32 file and of an int8 one (the page's: its embedding int8, and
#     what the matrices read rounded by groups of 32, as the page's kernels do): the largest difference of the logits,
#     how often the most likely token is the same, and 16 greedy tokens of each for the chat prompt.
#   faults: the real model's engine with one thing wrong at a time (FAULTS), each against the same numbers of
#     transformers: every fault has to be past the line the right engine is held to, or the line would let it through.
# The lines it holds the engine to are at the end of each part; anything past them is exit 1.
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from llama2_numpy import Llama, silu  # noqa: E402
from conducting import Directory, Mapped, converted  # noqa: E402
from made_up import bitnet_model, safetensors_file  # noqa: E402
from reference import (TEXT, differences, ended, engine_logits, fetch, float32_file, line_of, logits_failed,  # noqa: E402
                       logits_said, made_up_engine, option, options_shown)
from test_bitnet import naive_bitnet_logits  # noqa: E402

REPO, REVISION = "microsoft/bitnet-b1.58-2B-4T", "04c3b9ad9361b824064a1f25ea60a8be9599b127"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors"]
BOS = 128000  # <|begin_of_text|>: what its tokenizer puts first, and the page
# the chat format of one turn, as its chat_template writes it (the converter reads that template)
CHAT = "User: {prompt}<|eot_id|>Assistant: "
PROMPT = "What is the capital of Japan? Answer in one sentence."
SPECIALS = ["<|begin_of_text|>", "<|eot_id|>"]
NEW_TOKENS = 16
LOGITS_LINE = 1e-3  # the least of the logits' line (made-up models: 2e-3)
# the real part's phases each in a process of its own (real()); tests/reference_dry.py runs them in its own, on its
# stand-ins, which a process of their own would not have
SEPARATE = True

say = lambda *parts: print("bitnet:", *parts, flush=True)


def rounding(on):
    """transformers' ActQuant as the model has it (on), or left out: every matrix then reads its input as it is."""
    from transformers.integrations import bitnet
    if not hasattr(rounding, "kept"):
        rounding.kept = bitnet.ActQuant.apply
    bitnet.ActQuant.apply = rounding.kept if on else (lambda activation: activation)


def transformers_model(folder):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)
    model.eval()
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


def greedy(model, ids, count=NEW_TOKENS):
    import torch
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), attention_mask=torch.tensor([[1] * len(ids)], dtype=torch.long),
                             max_new_tokens=count, do_sample=False)
    return out[0, len(ids):].tolist()


def engine_greedy(llama, ids, count=NEW_TOKENS):
    for pos, token in enumerate(ids):
        logits = llama.forward(token, pos)
    made = []
    for pos in range(len(ids), len(ids) + count):
        made.append(int(np.argmax(logits)))
        logits = llama.forward(made[-1], pos)
    return made


# ------------------------------------------------------------------------------------------------------ made-up
MADE_UP = {"small": dict(), "a classifier of its own": dict(shared=False), "one key head, three layers": dict(n_kv_heads=1, layers=3),
           "heads that fill the stream": dict(head_dim=32), "another epsilon": dict(eps=1e-6)}


def made_up(directory):
    failed = False
    for name, settings in MADE_UP.items():
        failed |= made_up_one(name, settings, directory)
    return failed


def made_up_one(name, settings, directory=HERE.parent / ".tmp" / "reference-bitnet"):
    """One made-up model, written to a folder and loaded by transformers as the published one is. In floats, the unit
    tests' reference and the engine are held to ten times what transformers' own two ways differ by. Rounded (the
    model's ActQuant), a value that lies on the edge of two of the 256 steps goes to either as the sums' order has it, and
    the logits behind it move by as much as a step (transformers' own two ways were 2e-2 apart in one made-up model, the
    reference 0.17 from them in another, on CI's first run): the reference is held there by its mean difference, a
    tenth of what the rounding moves at most (transformers rounded against transformers in floats), and the same most
    likely token at every position."""
    rng = np.random.default_rng(0)
    ids = [int(token) for token in rng.integers(0, 320, 40)]
    tensors, _, config = bitnet_model(seq_len=64, **settings)
    folder = directory / "made-up" / name.replace(" ", "-").replace(",", "")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.safetensors").write_bytes(safetensors_file(tensors))
    (folder / "config.json").write_text(json.dumps({**config, "torch_dtype": "float32"}))
    model, failed, theirs = transformers_model(folder), False, {}
    for on in (False, True):
        rounding(on)
        whole, stepped = at_once_and_stepped(model, ids)
        theirs[on] = stepped
        floor = differences(whole, stepped)[0]
        label = f"made-up, {name}, {'rounded' if on else 'floats'}"
        naive = np.stack(naive_bitnet_logits(tensors, config, ids, quantized=on))
        say(f"{label}: transformers at once and token by token: {floor:.2e} apart")
        if not on:
            line = line_of(floor, 2 * LOGITS_LINE)
            failed |= logits_failed(say, f"{label}: the unit tests' reference against transformers", naive, stepped, line)
            llama, options, _ = made_up_engine(tensors, config)
            failed |= logits_failed(say, f"{label}: the engine against transformers", engine_logits(llama, ids), stepped, line)
        else:
            moves = differences(stepped, theirs[False])[1]
            largest, mean, same, _ = differences(naive, stepped)
            ok = mean <= moves / 10 and same == len(ids)
            say(f"{label}: the unit tests' reference against transformers: largest difference {largest:.2e}, mean {mean:.2e} "
                f"(the rounding moves transformers' by {moves:.2e} on the mean), the same most likely token at {same} of {len(ids)}"
                f"{'' if ok else ' — FAILED (the line: a tenth of what the rounding moves, and every most likely token the same)'}")
            failed |= not ok
    rounding(True)
    del model
    gc.collect()
    return failed


# --------------------------------------------------------------------------------------------------------- real
def faults(llama):
    """The real model's engine with one thing wrong at a time: (what, how to make it, how to put it back)."""
    kept = {}

    def keep(name, value):
        kept[name] = getattr(llama, name)
        setattr(llama, name, value)

    def back(*names):
        for name in names:
            setattr(llama, name, kept.pop(name))

    def scaled(name, layer, by):
        def make():
            getattr(llama, name)[layer] *= np.float32(by)
        def undo():
            getattr(llama, name)[layer] /= np.float32(by)
        return make, undo

    relu = lambda x: np.maximum(x, np.float32(0.0))
    q_up, q_back = scaled("wq", 0, 1.25)
    middle = len(llama.wo) // 2
    o_up, o_back = scaled("wo", middle, 1.25)
    return [
        ("no norm in front of wo", lambda: keep("rms_att_sub_weight", None), lambda: back("rms_att_sub_weight")),
        ("no norm in front of w2", lambda: keep("rms_ffn_sub_weight", None), lambda: back("rms_ffn_sub_weight")),
        ("a ReLU in the gate, not its square", lambda: keep("activation", relu), lambda: back("activation")),
        ("a SiLU in the gate", lambda: keep("activation", silu), lambda: back("activation")),
        ("the first layer's q a quarter larger (its scale)", q_up, q_back),
        (f"layer {middle}'s wo a quarter larger (its scale)", o_up, o_back),
    ]


def real(directory):
    """The fetch here; transformers and each of the engine's two files in a process of its own (--phase=): the first
    run of this part on CI's runner of 16 GB lost the runner (run 43262511808), transformers' 2B unpacked in float32
    (about 10 GB) and the engine's widened matrices (about 9 GB) being more than it has where one process holds both,
    what torch keeps after its model is let go of among them. The phases hand on what they found in a file."""
    folder = directory / "real"
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        fetch(name, folder, REPO, REVISION)
    if option("only") == "fetch":
        return False
    failed = False
    if not SEPARATE:
        return real_transformers(directory) or any([real_engine(directory, dtype) for dtype in ("float32", "int8")])
    for phase in ("transformers", "float32", "int8"):
        said = subprocess.run([sys.executable, __file__, str(directory), f"--phase={phase}",
                               f"--positions={option('positions', '96')}"])
        say(f"real, the phase {phase}: exit {said.returncode}")
        failed |= said.returncode != 0
        if phase == "transformers" and said.returncode != 0:
            break
    return failed


def real_ids(folder):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    positions = int(option("positions", "96"))
    ids = ([BOS] + tokenizer.encode(TEXT, add_special_tokens=False).ids)[:positions]
    chat = [BOS] + tokenizer.encode(CHAT.format(prompt=PROMPT), add_special_tokens=False).ids
    return tokenizer, ids, chat


def real_transformers(directory):
    """transformers in floats and rounded: its logits at once and token by token, and its greedy tokens, into a file"""
    folder = directory / "real"
    tokenizer, ids, chat = real_ids(folder)
    began = time.perf_counter()
    model = transformers_model(folder)
    say(f"real: transformers loaded the model in {time.perf_counter() - began:.0f} s")
    kept = {}
    for on, name in ((False, "floats"), (True, "rounded")):
        rounding(on)
        whole, stepped = at_once_and_stepped(model, ids)
        made = greedy(model, chat)
        kept.update({f"{name}_whole": whole, f"{name}_stepped": stepped, f"{name}_greedy": np.array(made)})
        say(f"real, {name}: transformers at once and token by token: {differences(whole, stepped)[0]:.2e} apart; "
            f"greedy: {tokenizer.decode(made)!r}")
    largest, mean, same, _ = differences(kept["rounded_stepped"], kept["floats_stepped"])
    say(f"real: transformers rounded against transformers in floats (what the model's own rounding moves): largest {largest:.2e}, "
        f"mean {mean:.2e}, the same most likely token at {same} of {len(ids)}")
    np.savez(directory / "transformers.npz", **kept)
    return False


def real_engine(directory, dtype):
    """the engine (NumPy) on a file of this dtype against transformers' numbers (and, of float32, the faults)"""
    import mmap
    folder = directory / "real"
    tokenizer, ids, chat = real_ids(folder)
    theirs = dict(np.load(directory / "transformers.npz"))
    floats, rounded = theirs["floats_stepped"], theirs["rounded_stepped"]
    line = line_of(differences(theirs["floats_whole"], floats)[0], LOGITS_LINE)
    path, hf = directory / f"bitnet-{dtype}.bin", {"repo": REPO, "revision": REVISION, "weights": "model.safetensors",
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
    say(f"real, {dtype}: converted in {seconds:.0f} s, {path.stat().st_size / 1e9:.3f} GB; options {options_shown(options)}")
    with open(path, "rb") as file:
        checkpoint = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    llama = Llama(checkpoint, conversion.tokenizer, **options)
    began = time.perf_counter()
    ours = engine_logits(llama, ids)
    say(f"real, {dtype}: the engine ran {len(ids)} positions in {time.perf_counter() - began:.0f} s")
    failed = False
    if dtype == "float32":
        logits_said(say, "real", floats, ours)
        failed |= logits_failed(say, "real, float32: the engine against transformers in floats", ours, floats, line, gap=True)
    else:
        logits_failed(say, f"real, {dtype}: the engine (the page's rounding) against transformers in floats (said, no line)", ours,
                      floats, float("inf"))
    logits_failed(say, f"real, {dtype}: the engine against transformers rounded (said, no line)", ours, rounded, float("inf"))
    made = engine_greedy(llama, chat)
    say(f"real, {dtype}: the engine's greedy: {tokenizer.decode(made)!r}; the same tokens as transformers' in floats: "
        f"{sum(a == b for a, b in zip(made, theirs['floats_greedy']))} of {NEW_TOKENS}, rounded: "
        f"{sum(a == b for a, b in zip(made, theirs['rounded_greedy']))}")
    if dtype == "float32":
        weakest = None
        for what, make, undo in faults(llama):
            make()
            try:
                largest = differences(engine_logits(llama, ids), floats)[0]
            finally:
                undo()
            say(f"faults: {what}: largest difference {largest:.2e} ({largest / line:.1f} times the line)"
                f"{'' if largest > line else ' — FAILED: the line lets it through'}")
            failed |= not largest > line
            weakest = min(weakest or (largest, what), (largest, what))
        say(f"faults: the weakest, {weakest[1]}, is {weakest[0] / line:.1f} times the line")
    path.unlink()
    return failed


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    phase = option("phase")
    if phase:
        failed = real_transformers(directory) if phase == "transformers" else real_engine(directory, phase)
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
