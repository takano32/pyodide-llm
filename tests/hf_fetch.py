# hf_fetch.py: the files of a Hugging Face model of src/models.js, at the revision the list pins, for measurements in
# CI that need the original (T100): the weights, config.json, the tokenizer and tokenizer_config.json. Prints what to
# hand to tests/perplexity_prepare.py: the directory, or the .gguf file.
#
#   python3 tests/hf_fetch.py <model id, or hf:<owner>/<repository>@<revision> (as tests/e2e.mjs takes it)> <directory>
import json
import subprocess
import sys
from pathlib import Path

from fixed_outputs import HERE, fetch

model_id, directory = sys.argv[1], Path(sys.argv[2])
script = f"import('./src/models.js').then(({{ MODELS }}) => console.log(JSON.stringify(MODELS.find((m) => m.id === {json.dumps(model_id)}))))"
if model_id.startswith("hf:"):
    # a repository the list does not name as it is (the original of a model the list takes from a GGUF, T98)
    repo, _, revision = model_id[3:].partition("@")
    entry = {"hf": {"repo": repo, "revision": revision or "main", "weights": "model.safetensors",
                    "config": "config.json", "tokenizer": "tokenizer.json"}}
else:
    entry = json.loads(subprocess.check_output(["node", "-e", script], cwd=HERE.parent))
hf = entry["hf"]
weights = fetch(entry, hf["weights"], directory)
if not hf["weights"].endswith(".gguf"):
    for name in [hf["config"], *([hf["tokenizer"]] if isinstance(hf["tokenizer"], str) else hf["tokenizer"][:1]), "tokenizer_config.json"]:
        try:
            fetch(entry, name, directory)
        except OSError:
            pass  # tokenizer_config.json is optional
print(weights if hf["weights"].endswith(".gguf") else weights.parent)
