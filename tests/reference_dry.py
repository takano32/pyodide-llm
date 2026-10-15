# reference_dry.py
# T357.4: the three tools that hold the engine to transformers in CI (tests/reference_llama.py, tests/reference_qwen35.py,
# tests/reference_lfm2.py), run from their arguments to their exit where there is no torch: stand-ins for torch and
# transformers (tests/reference_standin/) answer with the engine itself, on made-up models and made-up directories of
# what Hugging Face publishes. Nothing is held to transformers here (CI does that, 10 to 50 minutes a tool); what this
# shows in a quarter of a minute is that the tools still run: the made-up models, the conversions, the tokenizers'
# comparison, both ways of transformers, the lines, the states, the long text, the greedy texts, every fault, the exits.
#
#   python tests/reference_dry.py            the check: every part ends as it must (the light suite)
#   python tests/reference_dry.py --lines    every line the tools say, the times masked: to compare two trees with
#                                            (python <a tree>/tests/reference_dry.py --lines > a; diff a b)
#   ... [--work <directory>] [part ...]      the made-up directories (default .tmp/reference-dry), only these parts
#
# What must be so (anything else is exit 1):
#   right: every part of every tool ends with nothing FAILED: its exit is 0, its result False;
#   the faults are all there and all seen: the 19 of the LFM2, the weak ones of a Granite (those the check must see),
#     the three of the layers RoPE leaves alone;
#   wrong: against a transformers whose logits are 1e-2 off, one whose states are 1% off and one whose long pass alone
#     is off (the stand-in's WRONG), every tool says FAILED and ends True: a verdict nobody reads would pass the parts
#     above. And of the LFM2's faults: an engine with nothing wrong in their place is said NOT CAUGHT, and one whose
#     logits are right and whose states are not is caught, by the states.
# What it cannot see: anything of transformers itself (its classes' arguments, its cache's shapes, its tokenizer), the
# real models, reference_llama.py's --gguf, --layers and ByLayer, reference_qwen35.py's --model= (2B, 4B).
import contextlib
import io
import json
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "reference_standin"), str(HERE)]
from tree import python_folder  # noqa: E402
sys.path.insert(0, python_folder(HERE.parent))
import fetching  # noqa: E402


def no_network(url, target, tries=3, timeout=60, optional=False):
    """the files are there or they are not: the dry run asks nobody"""
    if Path(target).exists():
        return Path(target)
    if optional:
        return None
    raise FileNotFoundError(f"the dry run has no {target}")


fetching.download = no_network  # (before the tools take the name)
import made_up  # noqa: E402
import reference  # noqa: E402
import reference_gemma3  # noqa: E402
import reference_lfm2  # noqa: E402
import reference_llama  # noqa: E402
import reference_qwen35  # noqa: E402
import transformers  # noqa: E402  (the stand-in)

VOCAB, CONTEXT = 800, 512  # (the LFM2's fault "the state not cleared" runs tokens 500 to 700)
WEAK = 7  # the weak faults of a Granite whose q is multiplied (reference_llama.py's weak_errors): a line each
LONG = 300  # of the long text here: past 256, where the engine's keys and values double (the tools' own: 1024)
REVISION = "0" * 40
CHATML = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}" \
         "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
SPECIALS = ["<|pad|>", "<|startoftext|>", "<|endoftext|>", "<s>", "</s>", "<unk>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def tokenizer_json(begins=False):
    """A small byte-level BPE trained on the tools' texts, as the real tokenizers library saves it. begins: a text begins
    with <|startoftext|>, as the LFM2's tokenizer has it."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    real = Tokenizer(models.BPE())
    real.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    real.decoder = decoders.ByteLevel()
    corpus = [reference_llama.TEXT, reference_qwen35.TEXT, reference_llama.PROMPT, "user assistant system the of and to in"]
    real.train_from_iterator(corpus * 4, trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIALS, show_progress=False,
                                                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    real.add_special_tokens([f"<extra_{at}>" for at in range(VOCAB - real.get_vocab_size())])
    assert real.get_vocab_size() == VOCAB, real.get_vocab_size()
    if begins:
        real.post_processor = processors.TemplateProcessing(single="<|startoftext|> $A", special_tokens=[("<|startoftext|>", 1)])
    return real.to_str()


def put(folder, files):
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (folder / name).write_bytes(data.encode() if isinstance(data, str) else data)


# the directories directories() makes, in the mark: a name more here and every checkout makes them again
MADE = ("qwen35", "lfm2", "gemma3", "llama")


def directories(work):
    """What Hugging Face would publish, where each tool looks for it: a Qwen3.5 (one shard under the 0.8B's name), an
    LFM2, and two Llamas for reference_llama.py: a Granite (q multiplied) and a SmolLM3 (layers RoPE leaves alone)."""
    said = lambda **more: json.dumps({"bos_token": "<|startoftext|>", "eos_token": "<|im_end|>", **more})
    tensors, config = made_up.qwen35_model(n_layers=8, every=4, vocab_size=VOCAB, seq_len=CONTEXT, key_heads=2, value_heads=4)
    put(work / "qwen35", {"config.json": json.dumps(config), "tokenizer.json": tokenizer_json(), "tokenizer_config.json": said(),
                          "model.safetensors.index.json": json.dumps({"weight_map": {name: reference_qwen35.WEIGHTS for name in tensors}}),
                          reference_qwen35.WEIGHTS: made_up.safetensors_file(tensors)})
    tensors, config = made_up.lfm2_model(vocab_size=VOCAB, seq_len=CONTEXT)
    put(work / "lfm2" / "dry", {"config.json": json.dumps(config), "tokenizer.json": tokenizer_json(True),
                                "tokenizer_config.json": said(chat_template="{{ bos_token }}" + CHATML),
                                "generation_config.json": "{}", "chat_template.jinja": "{{ bos_token }}" + CHATML,
                                "model.safetensors": made_up.safetensors_file(tensors)})
    # (T261) a Gemma 3 whose window the long text here is past four times
    tensors, config = made_up.gemma3_model(vocab_size=VOCAB, seq_len=CONTEXT, window=64, kinds="ssssslssl")
    put(work / "gemma3" / "dry", {"config.json": json.dumps(config), "tokenizer.json": tokenizer_json(True),
                                  "tokenizer_config.json": said(chat_template="{{ bos_token }}" + CHATML),
                                  "generation_config.json": "{}", "chat_template.jinja": "{{ bos_token }}" + CHATML,
                                  "model.safetensors": made_up.safetensors_file(tensors)})
    for name, more in (("granite", dict(model_type="granite", attention_multiplier=0.3, embedding_multiplier=1.0, residual_multiplier=1.0,
                                        logits_scaling=1.0)),
                       ("smollm3", dict(model_type="smollm3", no_rope_layer_interval=4, use_sliding_window=False))):
        settings, weights = made_up.synthetic_weights(dim=64, hidden_dim=96, n_layers=8, n_heads=4, n_kv_heads=2, vocab_size=VOCAB,
                                                      seq_len=CONTEXT, shared=False)
        tensors, config = made_up.hugging_face(settings, weights, False)
        put(work / "llama" / f"made--{name}" / REVISION,
            {"config.json": json.dumps({**config, **more, "bos_token_id": 1, "eos_token_id": 7, "rms_norm_eps": 1e-5}),
             "tokenizer.json": tokenizer_json(), "tokenizer_config.json": said(chat_template=CHATML),
             "model.safetensors": made_up.safetensors_file(tensors)})


def masked(text):
    return re.sub(r"in \d+ s\b", "in N s", re.sub(r"\d+\.\d s\b", "N s", text))


def run(call):
    """(the lines it said, how it ended: ("returned", value), ("exit", code) or ("raised", the end of the traceback))"""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            ended = ("returned", call())
    except SystemExit as left:
        ended = ("exit", left.code)
    except BaseException as error:  # noqa: B902 (whatever a tool raises is what the dry run is here to show)
        ended = ("raised", f"{type(error).__name__}: {str(error)[:300]}\n" + "".join(traceback.format_exc().splitlines(keepends=True)[-6:]))
    return masked(out.getvalue()), ended


def entry(name):
    """an entry of the list (src/models.js) for a made-up Llama"""
    return {"id": f"dry-{name}", "name": f"Dry {name}", "options": {},
            "hf": {"repo": f"made/{name}", "revision": REVISION, "weights": "model.safetensors"}}


def nothing_wrong(llama, make):
    return llama


def only_in_the_states(llama, make):
    """right logits, and the newest row of the first convolution layer's state 1% off after every token (put back before
    the next reads it): what only the states can see"""
    right, kept = llama.forward, {}

    def forward(token, pos, **more):
        if kept:
            llama.conv_state[0][-1] = kept.pop("row")
        out = right(token, pos, **more)
        kept["row"] = llama.conv_state[0][-1].copy()
        llama.conv_state[0][-1] *= 1.01
        return out

    llama.forward = forward
    return llama


def parts(work):
    """[(name, what to call, how it must end, what its lines must hold: [(a pattern, how many lines)], the stand-in's WRONG)]"""
    def main(tool, *argv):
        def call():
            sys.argv = [f"{tool.__name__}.py", *argv]
            return tool.main()
        return call

    def llama_real(name, weak):
        def call():
            reference_llama.WEAK = weak
            try:
                return reference_llama.real(entry(name), work / "llama", 96)
            finally:
                reference_llama.WEAK = False
        return call

    def llama_made_up(name, weak=True):
        def call():
            reference_llama.WEAK = weak
            try:
                return reference_llama.made_up(name, reference_llama.MADE_UP[name])
            finally:
                reference_llama.WEAK = False
        return call

    def lfm2_faults(faults):
        def call():
            real, reference_lfm2.FAULTS = reference_lfm2.FAULTS, faults
            try:
                return reference_lfm2.real(work / "lfm2" / "dry", 96, "dry", True)
            finally:
                reference_lfm2.FAULTS = real
        return call

    right, fine = ("returned", False), [("FAILED", 0)]
    faults = len(reference_lfm2.FAULTS)
    out = []
    # the made-up models of each tool (q is multiplied by something other than 1 in all three Granites)
    out += [(f"llama made-up, {name}", llama_made_up(name), right, fine + [(": a fault, ", WEAK)], {})
            for name in reference_llama.MADE_UP]
    out += [(f"qwen35 made-up, {name}", lambda name=name: reference_qwen35.made_up(name, reference_qwen35.MADE_UP[name]), right,
             fine + [("the states after 80 positions", 1)], {}) for name in reference_qwen35.MADE_UP]
    out += [(f"lfm2 made-up, {name}", lambda name=name: reference_lfm2.made_up(name, reference_lfm2.MADE_UP[name]), right,
             fine + [("the states after 80 positions", 1)], {}) for name in reference_lfm2.MADE_UP]
    # the real parts, on the made-up directories
    out.append(("qwen35 real", lambda: reference_qwen35.real(work / "qwen35", 96), right,
                fine + [("the same ids", 1), ("the two wrote the same", 1), (f"long: the engine ran {LONG} positions", 1)], {}))
    out.append(("lfm2 real, with the faults", lambda: reference_lfm2.real(work / "lfm2" / "dry", 96, "dry", True), right,
                fine + [("the same ids", 1), ("the two wrote the same", 1), (f"long: the engine ran {LONG} positions", 1),
                        (r"fault \(", faults), (f"faults: {faults} of {faults} are past the line", 1)], {}))
    gemma = len(reference_gemma3.FAULTS) + len(reference_gemma3.LONG_FAULTS)
    out += [(f"gemma3 made-up, {name}", lambda name=name: reference_gemma3.made_up(name, reference_gemma3.MADE_UP[name]), right,
             fine + [("the states after 80 positions", 1), ("the local layers", 1)], {}) for name in reference_gemma3.MADE_UP]
    out.append(("gemma3 real, with the faults", lambda: reference_gemma3.real(work / "gemma3" / "dry", 96, "dry", True), right,
                fine + [("the same ids", 1), ("the two wrote the same", 2), (f"long: the engine ran {LONG} positions", 1),
                        (r"fault \(", gemma), (r"noted \(", len(reference_gemma3.NOTED)),
                        (f"faults: {gemma} of {gemma} are past the line", 1)], {}))
    out.append(("llama real, a Granite with the weak faults", llama_real("granite", True), right,
                fine + [("the same ids", 2), ("the two wrote the same", 1), (": a fault, ", WEAK), ("must see and did not: none", 1)], {}))
    out.append(("llama real, a SmolLM3", llama_real("smollm3", False), right,
                fine + [("the same ids", 2), (": a fault, ", 3), ("seen by neither", 0), ("the check did not see: none", 1)], {}))
    # the tools by their arguments
    ok = [("FAILED", 0), ("the engine computes what transformers computes", 1)]
    out.append(("reference_llama.py --only=made-up", main(reference_llama, str(work / "llama"), "--only=made-up"), ("exit", 0), ok, {}))
    out.append(("reference_qwen35.py --only=made-up", main(reference_qwen35, str(work / "qwen35"), "--only=made-up"), ("exit", 0), ok, {}))
    out.append(("reference_lfm2.py --only=made-up", main(reference_lfm2, str(work / "lfm2"), "--only=made-up"), ("exit", 0), ok, {}))
    out.append(("reference_gemma3.py --only=made-up", main(reference_gemma3, str(work / "gemma3"), "--only=made-up"), ("exit", 0), ok, {}))
    out.append(("reference_gemma3.py --only=real --model=dry --positions=40 --no-faults",
                main(reference_gemma3, str(work / "gemma3"), "--only=real", "--model=dry", "--positions=40", "--no-faults"), ("exit", 0),
                ok + [("the states after 40 positions", 1), (r"fault \(", 0)], {}))
    out.append(("reference_qwen35.py --only=fetch", main(reference_qwen35, str(work / "qwen35"), "--only=fetch"), ("returned", None), [], {}))
    out.append(("reference_lfm2.py --only=fetch --model=dry", main(reference_lfm2, str(work / "lfm2"), "--only=fetch", "--model=dry"),
                ("returned", None), [], {}))
    out.append(("reference_lfm2.py --only=real --model=dry,dry --positions=40 --no-faults",
                main(reference_lfm2, str(work / "lfm2"), "--only=real", "--model=dry,dry", "--positions=40", "--no-faults"), ("exit", 0),
                ok + [("the states after 40 positions", 2), (r"fault \(", 0)], {}))
    # the LFM2's faults: one that only the states see is caught, and no fault at all is not
    out.append(("lfm2 real, a fault only the states see", lfm2_faults({"only in the states": only_in_the_states}), right,
                fine + [(r"fault \(only in the states\): logits 0.00e\+00 .* the same most likely token at 65 of 65", 1),
                        ("faults: 1 of 1 are past the line", 1)], {}))
    out.append(("lfm2 real, no fault where a fault must be", lfm2_faults({"nothing wrong": nothing_wrong}), ("returned", True),
                [("NOT CAUGHT, FAILED", 1), ("faults: 0 of 1 are past the line", 1), ("FAILED", 2)], {}))
    # a transformers the engine is not within the line of: every verdict must say so, and the long text's must not
    said = [("FAILED", None), ("long: .* FAILED", 0)]
    for what, wrong in (("logits 1e-2 off", {"logits": 1e-2}), ("states 1% off", {"states": 1e-2})):
        first = next(iter(reference_llama.MADE_UP))
        out.append((f"WRONG ({what}): llama made-up", llama_made_up(first, False), ("returned", True), said, wrong))
        for tool, name in ((reference_qwen35, "qwen35"), (reference_lfm2, "lfm2"), (reference_gemma3, "gemma3")):
            first = next(iter(tool.MADE_UP))
            out.append((f"WRONG ({what}): {name} made-up", lambda tool=tool, first=first: tool.made_up(first, tool.MADE_UP[first]),
                        ("returned", True), said, wrong))
        out.append((f"WRONG ({what}): llama real", llama_real("smollm3", False), ("returned", True), said, wrong))
        out.append((f"WRONG ({what}): qwen35 real", lambda: reference_qwen35.real(work / "qwen35", 96), ("returned", True), said, wrong))
        out.append((f"WRONG ({what}): lfm2 real", lambda: reference_lfm2.real(work / "lfm2" / "dry", 96, "dry", False), ("returned", True), said, wrong))
        out.append((f"WRONG ({what}): gemma3 real", lambda: reference_gemma3.real(work / "gemma3" / "dry", 96, "dry", False), ("returned", True), said, wrong))
    # the long pass alone: one line says FAILED, the long text's
    said, wrong = [("FAILED", 1), ("^[a-z0-9]+: long: .* FAILED", 1)], {"kept": 1e-2}
    out.append(("WRONG (the long pass 1e-2 off): qwen35 real", lambda: reference_qwen35.real(work / "qwen35", 96), ("returned", True), said, wrong))
    out.append(("WRONG (the long pass 1e-2 off): lfm2 real", lambda: reference_lfm2.real(work / "lfm2" / "dry", 96, "dry", False), ("returned", True), said, wrong))
    out.append(("WRONG (the long pass 1e-2 off): gemma3 real", lambda: reference_gemma3.real(work / "gemma3" / "dry", 96, "dry", False), ("returned", True), said, wrong))
    out.append(("WRONG (logits 1e-2 off): reference_lfm2.py --only=made-up", main(reference_lfm2, str(work / "lfm2"), "--only=made-up"),
                ("exit", 1), [("^lfm2: FAILED$", 1)], {"logits": 1e-2}))
    return out


def main():
    arguments = sys.argv[1:]
    lines = "--lines" in arguments
    work = Path(arguments[arguments.index("--work") + 1]) if "--work" in arguments else HERE.parent / ".tmp" / "reference-dry"
    only = [arg for at, arg in enumerate(arguments) if not arg.startswith("--") and (at == 0 or arguments[at - 1] != "--work")]
    work = work.resolve()
    # T515: the mark says which directories were made, so that one this file has since grown (gemma3, T261) is made
    # in a checkout whose .tmp/ was made before it: the mark alone left it out, and the dry run asked the network
    made = "the made-up directories of tests/reference_dry.py: " + " ".join(MADE) + "\n"
    if not (work / "made").exists() or (work / "made").read_text() != made:
        directories(work)
        (work / "made").write_text(made)
    reference_qwen35.BOS = 2  # (the real one, 248044, is past the made-up vocabulary)
    reference.LONG = LONG
    reference_lfm2.MODELS["dry"] = ("made/lfm2", REVISION)
    # (T261: the made-up directory's tokenizer and template are the LFM2's here, a text beginning with token 1)
    reference_gemma3.MODELS["dry"] = ("made/gemma3", REVISION)
    reference_gemma3.BOS, reference_gemma3.BEGINS, reference_gemma3.CHAT = 1, "<|startoftext|>", reference_lfm2.CHAT
    reference_gemma3.SPECIALS, reference_gemma3.STOPS = reference_lfm2.SPECIALS, (1, 7)
    failed = []
    for name, call, must_end, must_hold, wrong in parts(work):
        if only and not any(part in name for part in only):
            continue
        transformers.WRONG.clear()
        transformers.WRONG.update(wrong)
        said, ended = run(call)
        transformers.WRONG.clear()
        if lines:
            print(f"==== {name}\n{said}---- {ended[0]} {ended[1]!r}" if ended[0] != "raised" else f"==== {name}\n{said}---- RAISED {ended[1]}", flush=True)
            continue
        wrongs = [] if ended == must_end else [f"ended {ended[0]} {ended[1]}, not {must_end[0]} {must_end[1]}"]
        for pattern, count in must_hold:
            found = sum(1 for line in said.splitlines() if re.search(pattern, line))
            if (found == 0) if count is None else (found != count):
                wrongs.append(f"{found} lines with {pattern!r}, not {'some' if count is None else count}")
        print(f"reference_dry: {name}: {len(said.splitlines())} lines, {ended[0]} {ended[1] if ended[0] != 'raised' else ''}"
              f"{'' if not wrongs else ' — NOT AS IT MUST: ' + '; '.join(wrongs)}", flush=True)
        if wrongs:
            failed.append(name)
            print("\n".join("    " + line[:400] for line in said.splitlines()[-12:]), flush=True)
    if not lines:
        print(f"reference_dry: {'ok' if not failed else 'FAILED: ' + '; '.join(failed)}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
