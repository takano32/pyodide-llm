# llamacpp_tiny.py (the review of T245, a probe for CI, not for main): llama.cpp's own convert_hf_to_gguf.py (at the commit
# the converter's reading of a Qwen3.5 GGUF was written from) on small made-up Qwen3.5 models that hold what the 4B, 9B and
# 27B have and the unit tests' synthetic GGUFs only imitate (more value heads than key heads: 2 and 3 to a key head; a
# classifier of its own or none), and what the page's reader makes of that GGUF against what it makes of the same
# model's safetensors:
#
#   python tests/llamacpp_tiny.py <llama.cpp directory> <work directory> [--outtypes f32,q8_0]
#
# The checkpoint of a GGUF written as F32 must be the safetensors' to the bit but for ssm_a (llama.cpp's torch exp
# against NumPy's, a unit in the last place); as Q8_0 within Q8_0's rounding of every tensor (a head in the wrong place is
# 1.4 apart); and the engine's logits on the two the same.
import argparse
import json
import math
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
import reference_qwen35 as ref  # noqa: E402
from llama2_numpy import Llama  # noqa: E402

say = lambda *parts: print("T245LLAMACPP", *parts, flush=True)

# Qwen3.5-0.8B's config.json and tokenizer files: what the converter reads besides the weights (the tokenizer's hash
# says it is qwen35's pre-tokenizer)
REPO, REVISION = ref.REPO, ref.REVISION
SHAPES = {
    # key heads, value heads, key size, value size, layers, tied classifier
    "two to one, tied (the 4B's way)": (4, 8, 32, 32, 8, True),
    "three to one, a classifier of its own (the 27B's ratio)": (4, 12, 32, 32, 8, False),
    "one to one, tied (the 0.8B's)": (4, 4, 32, 32, 4, True),
}


def build(name, shape, directory, seed):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from safetensors.torch import save_file

    keys, values, key_dim, value_dim, layers, tied = shape
    directory.mkdir(parents=True, exist_ok=True)
    for file in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        ref.fetch(file, directory.parent / "real", REPO, REVISION)
        shutil.copy(directory.parent / "real" / file, directory / file)
    config = json.loads((directory / "config.json").read_text())
    text = config["text_config"]
    text.update(hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                linear_num_key_heads=keys, linear_num_value_heads=values, linear_key_head_dim=key_dim,
                linear_value_head_dim=value_dim, max_position_embeddings=4096, tie_word_embeddings=tied, mtp_num_hidden_layers=0,
                layer_types=["linear_attention" if (layer + 1) % 4 else "full_attention" for layer in range(layers)])
    config["tie_word_embeddings"] = tied
    config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    (directory / "config.json").write_text(json.dumps(config, indent=1))
    model_config = Qwen3_5TextConfig(**{k: v for k, v in text.items() if k not in ("model_type", "dtype", "torch_dtype")})
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(model_config).to(torch.float32).eval()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key.endswith("linear_attn.norm.weight"):
                parameter.copy_(1.0 + 0.2 * torch.randn_like(parameter))
            elif key.endswith("A_log"):
                parameter.copy_(torch.log(torch.rand_like(parameter) * 8 + 0.5))
            elif key.endswith("dt_bias"):
                parameter.copy_(0.5 * torch.randn_like(parameter))
            elif key.endswith("norm.weight") or key.endswith("layernorm.weight"):
                parameter.copy_(0.1 * torch.randn_like(parameter))  # (stored around 0: the model adds 1)
            elif key == "lm_head.weight" and tied:
                continue
            else:
                parameter.copy_(0.1 * torch.randn_like(parameter))
    state = {}
    for key, tensor in model.state_dict().items():
        if key == "lm_head.weight":
            if tied:
                continue
            state[key] = tensor.to(torch.bfloat16).contiguous()
        else:
            state["model.language_model." + key[len("model."):]] = tensor.to(torch.bfloat16).contiguous()
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
    say(f"{name}: {len(state)} tensors written ({sum(t.numel() for t in state.values()) / 1e6:.1f} M values)")
    return config


def convert(llama_cpp, directory, outtype):
    out = directory / f"tiny-{outtype}.gguf"
    run = subprocess.run([sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(directory), "--outfile", str(out),
                          "--outtype", outtype], capture_output=True, text=True, cwd=llama_cpp)
    if run.returncode:
        say(f"llama.cpp's converter failed: {run.stderr[-3000:]}")
        raise SystemExit(1)
    say(f"llama.cpp's convert_hf_to_gguf.py --outtype {outtype}: {out.stat().st_size} bytes")
    return out


def prepared(source, out):
    run = subprocess.run([sys.executable, str(HERE / "perplexity_prepare.py"), str(source), str(out), "float32"],
                         capture_output=True, text=True)
    if run.returncode:
        say(f"perplexity_prepare.py failed: {run.stderr[-3000:]}")
        raise SystemExit(1)
    return json.loads(Path(f"{out}.json").read_text())


def logits_of(out, options, ids):
    llama = Llama(np.memmap(f"{out}.bin", dtype=np.uint8, mode="r"), Path(f"{out}.tokenizer.bin").read_bytes(), kernels=None, **options)
    return np.stack([llama.forward(token, pos).copy() for pos, token in enumerate(ids)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("llama_cpp")
    parser.add_argument("work")
    parser.add_argument("--outtypes", default="f32,q8_0")
    arguments = parser.parse_args()
    llama_cpp, work = Path(arguments.llama_cpp), Path(arguments.work)
    failed = False
    for seed, (name, shape) in enumerate(SHAPES.items(), 245):
        folder = work / name.split(",")[0].replace(" ", "-")
        build(name, shape, folder, seed)
        # the safetensors' way: the original's own files (no GGUF in the folder)
        original = prepared(folder, work / f"{folder.name}-safetensors")
        for outtype in arguments.outtypes.split(","):
            gguf_folder = work / f"{folder.name}-{outtype}-gguf"
            gguf_folder.mkdir(exist_ok=True)
            for file in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                shutil.copy(folder / file, gguf_folder / file)
            file = convert(llama_cpp, folder, outtype)
            shutil.move(str(file), gguf_folder / file.name)
            out = work / f"{folder.name}-{outtype}-from-gguf"
            options = prepared(gguf_folder, out)
            same = options == original
            apart, wrong = ref.tensors_apart(Path(f"{out}.bin"), Path(f"{work / (folder.name + '-safetensors')}.bin"), original)
            worst = max(apart, key=lambda entry: entry[2])
            bits = sum(entry[3] for entry in apart)
            line = 1e-6 if outtype == "f32" else 2e-2
            ids = [248045] + list(range(100, 100 + 31))
            a = logits_of(out, options, ids)
            b = logits_of(work / f"{folder.name}-safetensors", original, ids)
            gap = float(np.abs(a - b).max())
            ok = same and not wrong and worst[2] <= line and (outtype != "f32" or gap < 1e-4) and (outtype == "f32" or gap < 0.5)
            failed |= not ok
            say(f"{name}, llama.cpp's {outtype} GGUF read as the page reads it: options {'the same' if same else 'OTHER — FAILED'}; "
                f"{bits} of {len(apart)} tensors the same to the bit, the furthest {worst[2]:.2e} apart (tensor {worst[0]} {list(worst[1])}; "
                f"the line {line:.0e}); the engine's logits over 32 positions {gap:.2e} apart{'' if ok else ' — FAILED'}")
            for index, shape_, relative, bit in apart:
                if not bit and relative > 1e-7:
                    say(f"    tensor {index} {list(shape_)}: {relative:.2e} apart")
    say("FAILED" if failed else "llama.cpp's own GGUFs are read as the safetensors of the same model")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
