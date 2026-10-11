# hf_fetch.py: the files of a Hugging Face model of src/models.js, at the revision the list pins, for measurements in
# CI that need the original (T100): the weights, config.json, the tokenizer and tokenizer_config.json. Prints what to
# hand to tests/perplexity_prepare.py: the directory, or the .gguf file. For a GGUF that takes the vocabulary of its
# original (T136's second stage, hf.vocabulary): the directory of the original's files, with the GGUF linked into it
# (T145: int4.yml and draft.yml named the originals instead).
# T374.4: which files those are is the conduct's of a conversion (src/python/convert/conduct.py), as it is for the page: it
# is answered until it asks for the weights' stream, and each file it asked for on the way was fetched whole
# (tests/conducting.py's Fetched): the one file of the weights or the shards its index names (T192), config.json,
# tokenizer_config.json, chat_template.jinja where that has no template, and the tokenizer the converter takes. The
# conversion made for that writes nothing and is let go.
#
#   python3 tests/hf_fetch.py <model id, or hf:<owner>/<repository>@<revision> (as tests/e2e.mjs takes it)> <directory>
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from conducting import Fetched, Nothing, converted  # noqa: E402

model_id, directory = sys.argv[1], Path(sys.argv[2])
script = f"import('./src/models.js').then(({{ MODELS }}) => console.log(JSON.stringify(MODELS.find((m) => m.id === {json.dumps(model_id)}))))"
if model_id.startswith("hf:"):
    # a repository the list does not name as it is (the original of a model the list takes from a GGUF, T98), as ?hf=
    # opens one: no tokenizer is named, so the conduct's candidates (T374.4: tokenizer.json alone before)
    repo, _, revision = model_id[3:].partition("@")
    entry = {"hf": {"repo": repo, "revision": revision or "main", "weights": "model.safetensors", "config": "config.json"}}
else:
    entry = json.loads(subprocess.check_output(["node", "-e", script], cwd=HERE.parent))
hf = entry["hf"]
answerer = Fetched.of(hf, directory)
converted(answerer, hf, weights=False, sink=Nothing())
weights = answerer.folder("weights") / hf["weights"]
if hf["weights"].endswith(".gguf") and not hf.get("vocabulary"):
    print(weights)  # T74: the GGUF says its configuration and vocabulary itself
    sys.exit(0)
# the configuration and the tokenizer: the model's, or for a GGUF those of the original it takes them from, with the
# GGUF linked beside them
folder = answerer.folder("vocabulary" if hf.get("vocabulary") else "weights")
if hf.get("vocabulary"):
    link = folder / hf["weights"]
    if not link.exists():
        link.symlink_to(weights.resolve())
print(folder)
