"""The header, the form and the checkpoint sizes of every listed model, from its config.json with the converter's own
functions (public/llama2_convert.py): what T130's review feeds forward.js's footprint() with."""
import json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "public"))
import llama2_convert as C

cache = os.path.join(HERE, "configs")
os.makedirs(cache, exist_ok=True)


def config_of(repo, rev):
    path = os.path.join(cache, f"{repo.replace('/', '__')}@{rev}.json")
    if not os.path.exists(path):
        url = f"https://huggingface.co/{repo}/raw/{rev}/config.json"
        with urllib.request.urlopen(url, timeout=30) as response:
            open(path, "wb").write(response.read())
    return json.load(open(path))


# llama2.c's own TinyStories (headers of their legacy files; llama2.c's README)
STORIES = {
    "stories260K": (64, 172, 5, 8, 4, 512, 512),
    "stories3_5M": (256, 768, 2, 8, 8, 4096, 512),
    "stories15M": (288, 768, 6, 6, 6, 32000, 256),
    "stories15M-f32": (288, 768, 6, 6, 6, 32000, 256),
    "stories42M": (512, 1376, 8, 8, 8, 32000, 1024),
    "stories42M-f32": (512, 1376, 8, 8, 8, 32000, 1024),
}

rows = []
for entry in json.load(open(os.path.join(HERE, "list.json"))):
    if entry["id"] in STORIES:
        header = STORIES[entry["id"]]
        form = {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 0}
        config = None
    else:
        config = config_of(entry["repo"], entry["rev"])
        normal = C.normalize(config)
        names = set()
        if not normal.get("tie_word_embeddings", False):
            names.add("embed_out.weight" if C.architecture(normal) == "neox" else "lm_head.weight")
        if normal.get("model_type") == "qwen2":
            names.add("model.layers.0.self_attn.q_proj.bias")
        if normal.get("model_type") == "qwen3":
            names.add("model.layers.0.self_attn.q_norm.weight")
        header = C.checkpoint_header(config, names, 4096)
        form = C.checkpoint_form(config, names)
    dtype = entry["dtype"] or "int8"
    row = {**entry, "header": list(header), "form": form}
    for name in ("int8", "int6", "float32", "float16"):
        row[f"size_{name}"] = C.checkpoint_size(header, name, form)
    row["listed_dtype"] = dtype
    rows.append(row)
json.dump(rows, open(os.path.join(HERE, "sizes.json"), "w"), indent=1)
for r in rows:
    print(f"{r['id']:40} {str(r['header']):48} {r['form']['arch']:5} bias={int(r['form']['bias'])} qkn={int(r['form']['qk_norm'])} "
          f"hd={r['form']['head_dim']:<4} int8={r['size_int8'] / 2**30:6.3f} GiB  note/bytes={r['bytes']}")
