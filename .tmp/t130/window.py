import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "public"))
import llama2_convert as C
# models a visitor could open with ?hf= that sit near the 16 GiB line: (name, header, form)
models = [
    ("pythia-12b", [5120, 20480, 36, 40, 40, -50688, 2048], {"bias": False, "arch": "neox", "qk_norm": False, "head_dim": 0}),
    ("pythia-6.9b", [4096, 16384, 32, 32, 32, -50432, 2048], {"bias": False, "arch": "neox", "qk_norm": False, "head_dim": 0}),
    ("llama-2-13b", [5120, 13824, 40, 40, 40, -32000, 4096], {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 0}),
    ("llama-2-7b", [4096, 11008, 32, 32, 32, -32000, 4096], {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 0}),
    ("gpt-j-6b-like llama MHA 28 layers 4096", [4096, 16384, 28, 32, 32, -50400, 2048], {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 0}),
    ("qwen2.5-14b", [5120, 13824, 48, 40, 8, -152064, 4096], {"bias": True, "arch": "llama", "qk_norm": False, "head_dim": 0}),
    ("qwen3-14b", [5120, 17408, 40, 40, 8, -151936, 4096], {"bias": False, "arch": "llama", "qk_norm": True, "head_dim": 128}),
    ("mistral-nemo-12b (head 128, 40 layers, 8 kv)", [5120, 14336, 40, 32, 8, -131072, 4096], {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 128}),
]
out = []
for name, header, form in models:
    out.append({"name": name, "header": header, "form": form, "size": C.checkpoint_size(header, "int8", form), "size6": C.checkpoint_size(header, "int6", form)})
json.dump(out, open(os.path.join(HERE, "window.json"), "w"))
print(len(out))
