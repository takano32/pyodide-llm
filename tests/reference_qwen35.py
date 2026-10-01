# reference_qwen35.py
# T229: the engine's Qwen3.5 (hybrid attention: Gated DeltaNet layers between full-attention ones) against
# transformers itself. PyTorch is not on the development machine, so this runs in CI:
#
#   pip install numpy tokenizers torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
#   pip install safetensors "transformers @ git+https://github.com/huggingface/transformers@7fb5bcd1d4b8a5c225a2c33429b2e9e023dd61ae"
#   python tests/reference_qwen35.py <directory for the download> [--only=made-up|real|fetch] [--positions=96]
#   (--only=fetch: the real model's files into the directory and no more, for tests/page_qwen35.sh)
#
#   node tests/ci.mjs run tests.yml extra="bash tests/reference_qwen35.sh" --ref <branch> --grep "qwen35"
#
# Two parts, every line of the log beginning with "qwen35:".
#   made-up: tiny random models of transformers' own class (Qwen3_5ForCausalLM), which hold what the real 0.8B does
#     not: three value heads to a key head (the 27B's), key and value heads of two sizes, a classifier of its own.
#     transformers' logits, over the whole text at once (its chunked delta rule) and token by token with its cache
#     (its recurrent rule), against the naive reference of the unit tests (conftest.naive_qwen35_logits) and against
#     the engine, converted the way the page converts (llama2_convert.Conversion).
#   real: Qwen/Qwen3.5-0.8B at a fixed revision, float32. The same token ids through transformers and through the
#     engine (NumPy): the largest difference of the logits, how often the most likely token is the same, and 16 greedy
#     tokens of each for the chat prompt.
# The lines it holds the engine to are at the end of each part; anything past them is exit 1.
import gc
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
import llama2_convert  # noqa: E402
from llama2_convert import Conversion  # noqa: E402
from llama2_numpy import Llama  # noqa: E402

REPO, REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json", WEIGHTS]
BOS = 248044  # <|endoftext|>: what the page begins every text with (llama2_convert.normalize)
# a text of more than 64 tokens (transformers' chunk of the delta rule), English and Japanese
TEXT = ("Mount Fuji is the highest mountain in Japan, standing 3,776 metres above sea level on the island of Honshu. "
        "It is an active stratovolcano that last erupted from 1707 to 1708, and on clear days it can be seen from "
        "Tokyo, about 100 kilometres to the north-east. 富士山は日本で最も高い山で、古くから信仰の対象とされ、"
        "多くの絵画や文学作品に描かれてきました。夏には多くの登山者が山頂を目指します。")
# the chat format of one turn without thinking, as chat_template.jinja writes it (the template calls a macro, which
# the converter's reader does not read: T236 writes it into the list by hand)
CHAT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPT = "What is the capital of Japan? Answer in one sentence."
SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
NEW_TOKENS = 16
CHUNK = 8 << 20

say = lambda *parts: print("qwen35:", *parts, flush=True)


def fetch(name, directory):
    target = directory / name
    if not target.exists():
        directory.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", timeout=60) as response, \
                        open(partial, "wb") as out:
                    while block := response.read(CHUNK):
                        out.write(block)
                break
            except OSError:
                if attempt == 2:
                    raise
        partial.rename(target)
    return target


class File:
    """The converter's sink: the float32 checkpoint into a memory-mapped file, never whole into memory (3 GB)."""

    def __init__(self, path):
        self.path = path

    def open(self, size, header, dtype, form):
        self.data = np.memmap(self.path, dtype=np.uint8, mode="w+", shape=(size,))

    def write(self, offset, raw):
        self.data[offset:offset + raw.size] = raw


def differences(ours, theirs):
    """(largest difference, mean difference, positions whose most likely token is the same, the smallest gap between
    the two most likely tokens of theirs where the most likely token differs)"""
    ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    gap = np.abs(ours - theirs)
    same = ours.argmax(axis=1) == theirs.argmax(axis=1)
    top = np.sort(theirs, axis=1)[:, -2:]
    margins = (top[:, 1] - top[:, 0])[~same]
    return float(gap.max()), float(gap.mean()), int(same.sum()), float(margins.max()) if margins.size else 0.0


# --------------------------------------------------------------------------------------------- made-up models
MADE_UP = {
    "three value heads a key head": dict(linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=8,
                                         linear_value_head_dim=12, num_hidden_layers=8),
    "one value head a key head, tied": dict(linear_num_key_heads=4, linear_num_value_heads=4, linear_key_head_dim=8,
                                            linear_value_head_dim=8, num_hidden_layers=6, tie_word_embeddings=True,
                                            full_attention_interval=3),
    "every second layer, whole heads turn": dict(linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=16,
                                                  linear_value_head_dim=4, num_hidden_layers=4, full_attention_interval=2,
                                                  partial_rotary_factor=1.0, linear_conv_kernel_dim=3),
}


def made_up(name, settings, positions=80):
    import torch
    from conftest import naive_qwen35_logits
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    settings = dict(settings)
    every, layers = settings.pop("full_attention_interval", 4), settings["num_hidden_layers"]
    rotary = settings.pop("partial_rotary_factor", 0.25)
    config = Qwen3_5TextConfig(
        vocab_size=320, hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=256, eos_token_id=7,
        layer_types=["linear_attention" if (layer + 1) % every else "full_attention" for layer in range(layers)],
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": rotary,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        **settings)
    torch.manual_seed(229)
    model = Qwen3_5ForCausalLM(config).to(torch.float32).eval()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key.endswith("linear_attn.norm.weight"):
                parameter.copy_(1.0 + 0.3 * torch.randn_like(parameter))
            elif not (key == "lm_head.weight" and config.tie_word_embeddings):
                parameter.copy_(0.3 * torch.randn_like(parameter))
    if config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight, "the classifier is not tied"
    rng = np.random.default_rng(229)
    tokens = [int(token) for token in rng.integers(0, 320, positions)]
    ids = torch.tensor([tokens])
    with torch.no_grad():
        whole = model(input_ids=ids).logits[0].numpy()
        past, stepped = None, []
        for at in range(positions):
            out = model(input_ids=ids[:, at:at + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            stepped.append(out.logits[0, -1].numpy())
    tensors = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    if config.tie_word_embeddings:
        tensors.pop("lm_head.weight", None)
    published = {"model_type": "qwen3_5", "text_config": json.loads(config.to_json_string(use_diff=False))}
    text = published["text_config"]
    if "rope_parameters" not in text:  # an older transformers writes them at the top
        text["rope_parameters"] = {"rope_theta": text.get("rope_theta", 1e7), "partial_rotary_factor": rotary}
    text["rope_parameters"].setdefault("partial_rotary_factor", rotary)
    naive = naive_qwen35_logits(tensors, published, tokens)

    from test_convert import safetensors_file
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(320)]}}).encode()
    conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary, "tokenizer.json",
                            dtype="float32", max_seq_len=256, start=8 + size)
    conversion.feed(file[8 + size:])
    conversion.finish()
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    llama = Llama(bytes(conversion.checkpoint), conversion.tokenizer, **options)
    ours = [llama.forward(token, pos).copy() for pos, token in enumerate(tokens)]

    failed = False
    # the line, as for the real model: ten times what transformers' own two computations differ by (it runs the
    # delta rule in float32 whatever the model's type, at once in chunks of 64 and token by token: what float32
    # leaves between two right computations of these random weights, 2.3e-3 for the first model in CI's first run),
    # and no less than 2e-3
    floor = differences(stepped, whole)[0]
    line = max(2e-3, 10 * floor)
    say(f"made-up ({name}), transformers token by token against transformers at once: largest difference {floor:.2e}")
    pairs = [("the naive reference against transformers at once", naive, whole),
             ("the engine against transformers at once", ours, whole),
             ("the engine against transformers token by token", ours, stepped)]
    for what, a, b in pairs:
        largest, mean, same, margin = differences(a, b)
        ok = largest <= line and (same == positions or margin <= 2 * largest)
        failed |= not ok
        say(f"made-up ({name}), {what}: largest difference {largest:.2e}, mean {mean:.2e}, the same most likely "
            f"token at {same} of {positions} positions{'' if ok else f' — FAILED (the line: {line:.1e})'}")
    say(f"made-up ({name}): options {json.dumps({key: options[key] for key in ('arch', 'linear', 'head_dim', 'rotary', 'rope_theta', 'rms_norm_eps') if key in options})}")
    return failed


# ------------------------------------------------------------------------------------------------- the real model
def real(directory, positions):
    import tokenizers
    import torch
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration

    say(f"real: {REPO}@{REVISION}, transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}")
    for name in FILES:
        fetch(name, directory)
    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))

    # the conversion, the way the page does it: the file in its own order, float32
    weights = directory / WEIGHTS
    data = np.memmap(weights, dtype=np.uint8, mode="r")
    (length,) = np.frombuffer(bytes(data[:8]), dtype="<u8")
    first = 8 + int(length)
    sink = File(directory / "float32.bin")
    began = time.perf_counter()
    conversion = Conversion(bytes(data[8:first]).decode(), first, (directory / "config.json").read_text(),
                            (directory / "tokenizer.json").read_bytes(), "tokenizer.json", dtype="float32",
                            tokenizer_config=(directory / "tokenizer_config.json").read_text(), sink=sink, start=first)
    for start in range(first, len(data), CHUNK):
        conversion.feed(bytes(data[start:start + CHUNK]))
    conversion.finish()
    sink.data.flush()
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    shown = {key: (value if key != "specials" else f"{len(value)} of them") for key, value in options.items()}
    say(f"real: converted in {time.perf_counter() - began:.1f} s, {sink.path.stat().st_size} bytes, options {json.dumps(shown)}")
    if "template" in conversion.options:
        say(f"real: the converter read a template: {json.dumps(conversion.options['template'])}")
    llama = Llama(np.memmap(sink.path, dtype=np.uint8, mode="r"), conversion.tokenizer, kernels=None,
                  **{**options, "specials": SPECIALS, "stop_tokens": [248044, 248046]})

    # the same ids for both: the page's BOS, then the text as the real tokenizer splits it (and the engine's own
    # tokenizer has to split it the same)
    ids = [BOS] + tokenizer.encode(TEXT, add_special_tokens=False).ids
    mine = [BOS] + llama.tokenizer.encode(TEXT, llama.specials)
    chat = CHAT.format(prompt=PROMPT)
    chat_ids = tokenizer.encode(chat, add_special_tokens=False).ids
    chat_mine = llama.tokenizer.encode(chat, llama.specials)
    failed = mine != ids or chat_mine != chat_ids
    say(f"real: {len(ids)} tokens of text and {len(chat_ids)} of the chat prompt, the engine's tokenizer gives "
        f"{'the same ids' if not failed else 'OTHER IDS — FAILED'}")
    ids = ids[:positions]

    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(directory), dtype=torch.float32).eval()
    tensor = torch.tensor([ids])
    with torch.no_grad():
        began = time.perf_counter()
        whole = model(input_ids=tensor).logits[0].float().numpy()
        say(f"real: transformers, {len(ids)} positions at once in {time.perf_counter() - began:.1f} s")
        past, stepped = None, []
        for at in range(len(ids)):
            out = model(input_ids=tensor[:, at:at + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            stepped.append(out.logits[0, -1].float().numpy())
        generated = model.generate(torch.tensor([[BOS] + chat_ids]), max_new_tokens=NEW_TOKENS, do_sample=False)[0].tolist()
    theirs = generated[1 + len(chat_ids):]
    del model, past, out
    gc.collect()

    began = time.perf_counter()
    ours = [llama.forward(token, pos).copy() for pos, token in enumerate(ids)]
    say(f"real: the engine (NumPy, float32), {len(ids)} positions in {time.perf_counter() - began:.1f} s")
    for what, logits in (("transformers", whole), ("the engine", ours)):
        logits = np.asarray(logits)
        say(f"real: logits of {what}: min {logits.min():.3f}, max {logits.max():.3f}, mean {logits.mean():.4f}, "
            f"std {logits.std():.4f}")
    floor = differences(stepped, whole)[0]
    say(f"real: transformers token by token against transformers at once: largest difference {floor:.2e} "
        f"(its two forms of the delta rule: what float32 leaves between two right computations)")
    # the line: ten times what transformers' own two computations differ by, and no less than 2e-2 (logits of the
    # order of 10 through 24 layers in float32); the most likely token the same wherever the reference's best two
    # are further apart than twice the difference
    line = max(2e-2, 10 * floor)
    for what, reference in (("transformers at once", whole), ("transformers token by token", stepped)):
        largest, mean, same, margin = differences(ours, reference)
        ok = largest <= line and (same == len(ids) or margin <= 2 * largest)
        failed |= not ok
        say(f"real: the engine against {what}: largest difference {largest:.2e}, mean {mean:.2e}, the same most "
            f"likely token at {same} of {len(ids)} positions (the largest gap between the reference's best two where "
            f"it is not: {margin:.2e}){'' if ok else f' — FAILED (the line: {line:.1e})'}")

    # 16 greedy tokens for the chat prompt, by transformers' generate() and by the engine's
    text = "".join(llama.generate(chat, steps=len(chat_ids) + NEW_TOKENS, temperature=0.0, echo=False))
    encoded = llama.tokenizer.encode(text, llama.specials) if text else []
    wrote = tokenizer.decode(theirs, skip_special_tokens=False)
    say(f"real: transformers wrote {json.dumps(wrote, ensure_ascii=False)} {theirs}")
    say(f"real: the engine wrote {json.dumps(text, ensure_ascii=False)} {encoded}")
    stop = next((at for at, token in enumerate(theirs) if token in (248044, 248046)), len(theirs))
    same = tokenizer.decode(theirs[:stop], skip_special_tokens=False) == text
    failed |= not same
    say(f"real: the two wrote {'the same' if same else 'OTHER TEXTS — FAILED'}")
    sink.path.unlink()
    return failed


def main():
    directory = Path(sys.argv[1])
    only = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--only=")), None)
    positions = int(next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--positions=")), 96))
    failed = False
    if only == "fetch":
        for name in FILES:
            fetch(name, directory)
        return
    if only in (None, "made-up"):
        for name, settings in MADE_UP.items():
            failed |= made_up(name, settings)
    if only in (None, "real"):
        failed |= real(directory, positions)
    say("FAILED" if failed else "the engine computes what transformers computes")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
