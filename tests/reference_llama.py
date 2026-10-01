# reference_llama.py
# T253 and T254: a model of the list that is a Llama to the engine, against transformers itself, on the original's
# safetensors. PyTorch is not on the development machine, so this runs in CI:
#
#   node tests/ci.mjs run tests.yml extra="bash tests/reference_llama.sh hf-granite-4.2-3b" --ref <branch> --grep "reference:"
#   python tests/reference_llama.py <directory for the downloads> [--only=made-up] [<id of the list> ...] [--positions=96]
#
# Two parts, every line of the log beginning with "reference:".
#   made-up: tiny random Granites of transformers' own class (GraniteForCausalLM), whose attention multiplies its
#     scores by attention_multiplier (T253: the converter puts it into q, llama2_convert.query_scale). transformers'
#     logits, over the whole text at once and token by token with its cache, against the engine converted the way the
#     page converts. And the same tensors converted as a Llama's, which must be another model: the comparison sees the
#     multiplier. A Granite 4.1's other multipliers are refused.
#   an id of the list (src/models.js): its original repository at the list's revision. The text as the real
#     tokenizer splits it against the engine's tokenizer, the chat prompt as the list's format writes it against
#     apply_chat_template, the logits of the same ids through transformers and through the engine (NumPy, float32,
#     converted from the safetensors in the file's own order, shards joined as the page joins them), and 16 greedy
#     tokens of each for the chat prompt.
# transformers computes in float32 here whatever the weights are stored as: the model is loaded as stored (bfloat16:
# Granite 4.2 3B is 7.3 GB so, and 14.6 GB as float32, which a runner of 16 GB has not), every nn.Linear widens its
# weight for its own product, and the embeddings are widened before they go in. A bfloat16 is a float32 to the bit, so
# the arithmetic is the float32 one of the same weights; the modeling code is transformers' own.
# The lines the engine is held to are at the end of each part; anything past them is exit 1.
import gc
import json
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
import llama2_convert  # noqa: E402
from llama2_convert import Conversion, joined_shards  # noqa: E402
from llama2_numpy import Llama  # noqa: E402

# more than 64 tokens, English and Japanese, with numbers after one space and after several (T254: where MiniCPM5's
# two-stage split is not Llama 3's), a number of more than three digits and a contraction
TEXT = ("Mount Fuji is the highest mountain in Japan, standing 3,776 metres above sea level on the island of Honshu. "
        "It's an active stratovolcano that last erupted from 1707 to  1708, about   100 kilometres from Tokyo; in 2013 "
        "it became a World Heritage Site. 富士山は日本で最も高い山で、標高は 3776 メートルです。古くから信仰の対象とされ、"
        "多くの絵画や文学作品に描かれてきました。夏には多くの登山者が山頂を目指します。")
PROMPT = "What is the capital of Japan? Answer in one sentence."
NEW_TOKENS = 16
STEPPED = 32  # positions transformers also computes token by token with its cache, for the floor of the line
CHUNK = 8 << 20

say = lambda *parts: print("reference:", *parts, flush=True)


def differences(ours, theirs):
    """(largest difference, mean difference, positions whose most likely token is the same, the largest gap between
    the two most likely tokens of theirs where the most likely token differs)"""
    ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    gap = np.abs(ours - theirs)
    same = ours.argmax(axis=1) == theirs.argmax(axis=1)
    top = np.sort(theirs, axis=1)[:, -2:]
    margins = (top[:, 1] - top[:, 0])[~same]
    return float(gap.max()), float(gap.mean()), int(same.sum()), float(margins.max()) if margins.size else 0.0


def float32_arithmetic():
    """Every nn.Linear of transformers computes in float32 from here on (see the head of this file)."""
    import torch

    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(torch.float32)
        return torch.nn.functional.linear(x.to(torch.float32), self.weight.to(torch.float32), bias)

    torch.nn.Linear.forward = forward


def logits_of(model, ids, stepped=0):
    """transformers' logits for these ids at once, and for the first `stepped` of them token by token with its cache"""
    import torch
    with torch.no_grad():
        embedded = model.get_input_embeddings()(torch.tensor([ids])).to(torch.float32)
        whole = model(inputs_embeds=embedded).logits[0].to(torch.float32).numpy()
        past, steps = None, []
        for at in range(min(stepped, len(ids))):
            out = model(inputs_embeds=embedded[:, at:at + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            steps.append(out.logits[0, -1].to(torch.float32).numpy())
    return whole, steps


def greedy(model, ids, count, stops):
    """count tokens after ids, the most likely one each time, by transformers with its cache; stops at a stop token"""
    import torch
    written = []
    with torch.no_grad():
        embed = lambda tokens: model.get_input_embeddings()(torch.tensor([tokens])).to(torch.float32)
        out = model(inputs_embeds=embed(ids), use_cache=True)
        for _ in range(count):
            token = int(out.logits[0, -1].argmax())
            if token in stops:
                break
            written.append(token)
            out = model(inputs_embeds=embed([token]), past_key_values=out.past_key_values, use_cache=True)
    return written


# --------------------------------------------------------------------------------------------- made-up Granites
# heads of 16 in a hidden size of 64: 0.3 * 4 and 0.0078125 * 4 are what q is multiplied by, the second a power of two
MADE_UP = {
    "grouped queries": dict(attention_multiplier=0.3, num_key_value_heads=2),
    "a power of two, tied": dict(attention_multiplier=0.0078125, num_key_value_heads=4, tie_word_embeddings=True),
    "one key head": dict(attention_multiplier=0.11, num_key_value_heads=1, num_hidden_layers=4),
}


def made_up_conversion(tensors, published):
    from test_convert import safetensors_file
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(published["vocab_size"])]}}).encode()
    conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary, "tokenizer.json",
                            dtype="float32", max_seq_len=256, start=8 + size)
    conversion.feed(file[8 + size:])
    conversion.finish()
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    return Llama(bytes(conversion.checkpoint), conversion.tokenizer, **options), options


def made_up(name, settings, positions=80):
    import torch
    from transformers import GraniteConfig, GraniteForCausalLM

    config = GraniteConfig(**{**dict(vocab_size=320, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                                     num_attention_heads=4, max_position_embeddings=256, rms_norm_eps=1e-5,
                                     rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                                     bos_token_id=1, eos_token_id=2), **settings})
    torch.manual_seed(253)
    model = GraniteForCausalLM(config).to(torch.float32).eval()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key == "lm_head.weight" and config.tie_word_embeddings:
                continue
            scale = 0.1 if key.endswith("norm.weight") else 0.3
            parameter.copy_((1.0 if key.endswith("norm.weight") else 0.0) + scale * torch.randn_like(parameter))
    rng = np.random.default_rng(253)
    tokens = [int(token) for token in rng.integers(0, 320, positions)]
    whole, stepped = logits_of(model, tokens, positions)
    tensors = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    if config.tie_word_embeddings:
        tensors.pop("lm_head.weight", None)
    published = json.loads(config.to_json_string(use_diff=False))
    assert published["model_type"] == "granite" and published["attention_multiplier"] == settings["attention_multiplier"]

    llama, options = made_up_conversion(tensors, published)
    ours = [llama.forward(token, pos).copy() for pos, token in enumerate(tokens)]
    plain, _ = made_up_conversion(tensors, {**published, "model_type": "llama"})
    as_a_llama = [plain.forward(token, pos).copy() for pos, token in enumerate(tokens)]

    failed = False
    floor = differences(stepped, whole)[0]
    line = max(2e-3, 10 * floor)
    say(f"made-up ({name}), transformers token by token against transformers at once: largest difference {floor:.2e}")
    for what, reference in (("transformers at once", whole), ("transformers token by token", stepped)):
        largest, mean, same, margin = differences(ours, reference)
        ok = largest <= line and (same == positions or margin <= 2 * largest)
        failed |= not ok
        say(f"made-up ({name}), the engine against {what}: largest difference {largest:.2e}, mean {mean:.2e}, the same "
            f"most likely token at {same} of {positions} positions{'' if ok else f' — FAILED (the line: {line:.1e})'}")
    largest, mean, same, _ = differences(as_a_llama, whole)
    seen = largest > 100 * line
    failed |= not seen
    say(f"made-up ({name}), the same tensors converted as a Llama's against transformers: largest difference "
        f"{largest:.2e}, the same most likely token at {same} of {positions}"
        f"{' (another model, as it must be)' if seen else ' — FAILED: the comparison does not see the multiplier'}")
    say(f"made-up ({name}): what q was multiplied by {llama2_convert.query_scale(published):.6g}, options "
        f"{json.dumps({key: options[key] for key in ('arch', 'bias', 'head_dim', 'rope_theta', 'rms_norm_eps') if key in options})}")
    # Granite 4.1's kind: the three multipliers the engine has not
    for key, value in (("embedding_multiplier", 12.0), ("residual_multiplier", 0.22), ("logits_scaling", 10.0)):
        try:
            made_up_conversion(tensors, {**published, key: value})
            failed = True
            say(f"made-up ({name}): {key} {value} was converted — FAILED")
        except ValueError:
            pass
    return failed


# ------------------------------------------------------------------------------------------------- a real model
def entries(ids):
    script = ("import('./src/models.js').then(({ MODELS }) => console.log(JSON.stringify("
              f"MODELS.filter((m) => {json.dumps(ids)}.includes(m.id)))))")
    found = {entry["id"]: entry for entry in json.loads(subprocess.check_output(["node", "-e", script], cwd=HERE.parent))}
    missing = [id for id in ids if id not in found]
    if missing:
        sys.exit(f"reference: the list has no {', '.join(missing)}")
    return [found[id] for id in ids]


def fetch(repo, revision, name, folder, optional=False):
    target = folder / name
    if target.exists():
        return target
    folder.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=60) as response, \
                    open(partial, "wb") as out:
                while block := response.read(CHUNK):
                    out.write(block)
            break
        except urllib.error.HTTPError as error:
            if error.code < 500 and optional:
                return None
            if error.code < 500 or attempt == 2:
                raise
        except OSError:
            if attempt == 2:
                raise
    partial.rename(target)
    return target


class File:
    """The converter's sink: the float32 checkpoint into a memory-mapped file, never whole into memory."""

    def __init__(self, path):
        self.path = path

    def open(self, size, header, dtype, form):
        self.data = np.memmap(self.path, dtype=np.uint8, mode="w+", shape=(size,))

    def write(self, offset, raw):
        self.data[offset:offset + raw.size] = raw


def real(entry, directory, positions):
    import torch
    import transformers
    from format_check import filled
    from transformers import AutoModelForCausalLM, AutoTokenizer

    id = entry["id"]
    original = entry["hf"].get("vocabulary") or entry["hf"]
    repo, revision = original["repo"], original["revision"]
    named = original.get("tokenizer", "tokenizer.json")
    named = named if isinstance(named, str) else named[0]
    folder = directory / repo.replace("/", "--") / revision
    say(f"{id}: {repo}@{revision}, transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}")
    for name in ("config.json", named):
        fetch(repo, revision, name, folder)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "generation_config.json"):
        fetch(repo, revision, name, folder, optional=True)
    names = ["model.safetensors"]
    if fetch(repo, revision, "model.safetensors", folder, optional=True) is None:
        index = json.loads(fetch(repo, revision, "model.safetensors.index.json", folder).read_text())
        names = sorted(set(index["weight_map"].values()))
        for name in names:
            fetch(repo, revision, name, folder)
    config = (folder / "config.json").read_text()

    # the conversion, the way the page does it: the shards joined, each file in its own order, float32
    shards = []
    for name in names:
        shard = np.memmap(folder / name, dtype=np.uint8, mode="r")
        size = struct.unpack("<Q", bytes(shard[:8]))[0]
        shards.append((shard, bytes(shard[8:8 + size]).decode(), 8 + size))
    if len(shards) == 1:
        header, base = shards[0][1], shards[0][2]
        pieces = [(shards[0][0], base, len(shards[0][0]))]
    else:
        header, lengths = joined_shards([text for _, text, _ in shards])
        base = 0
        pieces = [(shard, begin, begin + length) for (shard, _, begin), length in zip(shards, lengths)]
    read = lambda name: (folder / name).read_text() if (folder / name).exists() else None
    tokenizer_config = read("tokenizer_config.json") or ""
    has_template = bool(json.loads(tokenizer_config).get("chat_template")) if tokenizer_config else False
    sink = File(directory / f"{id}.float32.bin")
    began = time.perf_counter()
    conversion = Conversion(header, base, config, (folder / named).read_bytes(), named, dtype="float32",
                            tokenizer_config=tokenizer_config, chat_template=None if has_template else read("chat_template.jinja"),
                            sink=sink, start=base)
    for data, begin, end in pieces:
        for start in range(begin, end, CHUNK):
            conversion.feed(bytes(data[start:min(start + CHUNK, end)]))
    conversion.finish()
    sink.data.flush()
    del sink.data, shards, pieces
    options = {**conversion.options, **entry.get("options", {})}
    template = entry.get("template") or options.get("template")
    options.pop("template", None)
    shown = {key: (value if key != "specials" else f"{len(value)} of them") for key, value in options.items()}
    say(f"{id}: converted in {time.perf_counter() - began:.1f} s, {sink.path.stat().st_size} bytes, what q was multiplied "
        f"by {llama2_convert.query_scale(llama2_convert.normalize(json.loads(config))):.6g}, options {json.dumps(shown)}")
    say(f"{id}: the format of {'the list' if entry.get('template') else 'the converter'}: {json.dumps(template)}")
    llama = Llama(np.memmap(sink.path, dtype=np.uint8, mode="r"), conversion.tokenizer, kernels=None, **options)
    specials = llama.specials

    # the ids: the text as the real tokenizer splits it, with whatever it puts in front, and the chat prompt as the
    # page sends it ([bos] + the list's format) against the real template's
    reference = AutoTokenizer.from_pretrained(folder)
    ids = list(reference(TEXT)["input_ids"])
    mine = llama.tokenizer.encode(TEXT, specials)
    front = ids[:len(ids) - len(mine)]
    failed = ids[len(front):] != mine or len(front) > 1
    say(f"{id}: {len(ids)} tokens of text, the real tokenizer puts {front or 'nothing'} in front "
        f"({reference.convert_ids_to_tokens(front)}); after it the engine's tokenizer gives "
        f"{'the same ids' if not failed else 'OTHER IDS — FAILED'}")
    if failed:
        first = next((at for at, (a, b) in enumerate(zip(ids[len(front):], mine)) if a != b), min(len(mine), len(ids)))
        say(f"{id}: first at {first}: real {reference.convert_ids_to_tokens(ids[len(front):][first:first + 8])}, "
            f"the engine {reference.convert_ids_to_tokens(mine[first:first + 8])}")
    if front and front != [conversion.options["bos"]]:
        failed = True
        say(f"{id}: the converter's BOS is {conversion.options['bos']}, not what the real tokenizer begins with — FAILED")
    thinking = {"enable_thinking": False} if "(no thinking)" in entry["name"] else \
        {"enable_thinking": True} if "(thinking)" in entry["name"] else {}
    chat_real = reference.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                              tokenize=True, **thinking)
    chat_real = list(chat_real["input_ids"] if hasattr(chat_real, "keys") else chat_real)
    typed = filled(template, PROMPT) if template else PROMPT
    chat = [options["bos"]] + llama.tokenizer.encode(typed, specials)
    same_chat = chat == chat_real
    failed |= not same_chat
    say(f"{id}: the chat prompt, {len(chat)} ids of the page ([bos {options['bos']}] and the format) against "
        f"apply_chat_template({thinking or ''}): {'the same ids' if same_chat else 'OTHER IDS — FAILED'}")
    if not same_chat:
        say(f"{id}: page {reference.convert_ids_to_tokens(chat)[:40]}\nreference: {id}: real {reference.convert_ids_to_tokens(chat_real)[:40]}")
    ids = ids[:positions]

    float32_arithmetic()
    stored = json.loads(config).get("dtype") or json.loads(config).get("torch_dtype") or "float32"
    model = AutoModelForCausalLM.from_pretrained(str(folder), dtype=getattr(torch, stored)).eval()
    say(f"{id}: transformers' {type(model).__name__}, weights held as {stored}, arithmetic in float32")
    began = time.perf_counter()
    whole, stepped = logits_of(model, ids, STEPPED)
    say(f"{id}: transformers, {len(ids)} positions at once and {len(stepped)} token by token in {time.perf_counter() - began:.1f} s")
    theirs = greedy(model, chat, NEW_TOKENS, set(options["stop_tokens"]))
    del model
    gc.collect()

    began = time.perf_counter()
    ours = [llama.forward(token, pos).copy() for pos, token in enumerate(ids)]
    say(f"{id}: the engine (NumPy, float32), {len(ids)} positions in {time.perf_counter() - began:.1f} s")
    for what, logits in (("transformers", whole), ("the engine", ours)):
        logits = np.asarray(logits)
        say(f"{id}: logits of {what}: min {logits.min():.3f}, max {logits.max():.3f}, mean {logits.mean():.4f}, std {logits.std():.4f}")
    floor = differences(stepped, whole[:len(stepped)])[0]
    say(f"{id}: transformers token by token against transformers at once ({len(stepped)} positions): largest difference "
        f"{floor:.2e} (what float32 leaves between two right computations)")
    # the line: ten times what transformers' own two computations differ by, and no less than 2e-2 (logits of the
    # order of 10 through dozens of layers in float32); the most likely token the same wherever the reference's best
    # two are further apart than twice the difference
    line = max(2e-2, 10 * floor)
    for what, reference_logits, mine_logits in (("transformers at once", whole, ours),
                                                ("transformers token by token", stepped, ours[:len(stepped)])):
        largest, mean, same, margin = differences(mine_logits, reference_logits)
        ok = largest <= line and (same == len(mine_logits) or margin <= 2 * largest)
        failed |= not ok
        say(f"{id}: the engine against {what}: largest difference {largest:.2e}, mean {mean:.2e}, the same most likely "
            f"token at {same} of {len(mine_logits)} positions (the largest gap between the reference's best two where "
            f"it is not: {margin:.2e}){'' if ok else f' — FAILED (the line: {line:.1e})'}")

    # 16 greedy tokens after the page's ids of the chat prompt, by transformers and by the engine's generate()
    text = "".join(llama.generate(typed, steps=len(chat) - 1 + NEW_TOKENS, temperature=0.0, echo=False))
    wrote = reference.decode(theirs, skip_special_tokens=False)
    say(f"{id}: transformers wrote {json.dumps(wrote, ensure_ascii=False)} {theirs}")
    say(f"{id}: the engine wrote {json.dumps(text, ensure_ascii=False)}")
    failed |= wrote != text
    say(f"{id}: the two wrote {'the same' if wrote == text else 'OTHER TEXTS — FAILED'}")
    del llama
    gc.collect()
    sink.path.unlink()
    return failed


def main():
    directory = Path(sys.argv[1])
    only = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--only=")), None)
    positions = int(next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--positions=")), 96))
    ids = [arg for arg in sys.argv[2:] if not arg.startswith("--")]
    failed = False
    if only in (None, "made-up"):
        for name, settings in MADE_UP.items():
            failed |= made_up(name, settings)
    for entry in entries(ids) if only is None else []:
        failed |= real(entry, directory, positions)
    say("FAILED" if failed else "the engine computes what transformers computes")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
