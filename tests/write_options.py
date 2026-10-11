# write_options.py (T249): what the page gives the engine for a Hugging Face model of the list, as JSON: the converter's
# options (read with tokenizer_config.json and the chat template, so with the special tokens of the format) under the
# entry's own, as the worker merges them, the format of one turn (the entry's, else the converter's) and the entry's
# prompt. Only the small files and the heads of the weights are fetched (tests/format_check.py's converted()).
#
#   python3 tests/write_options.py <model id> <directory for the downloads>
import json
import sys
from pathlib import Path

import format_check

model_id, directory = sys.argv[1], Path(sys.argv[2])
entry = next(entry for entry in format_check.entries() if entry["id"] == model_id)
_, made = format_check.converted(entry, directory)
options = {**made.options, **entry.get("options", {})}
template = entry.get("template") or options.get("template")
options.pop("template", None)
print(json.dumps({"id": model_id, "name": entry["name"], "options": options, "template": template,
                  "prompt": entry["prompt"], "generation": entry["generation"], "header": [int(value) for value in made.stream.header]}, ensure_ascii=False))
