# perplexity_prepare.py
# A Hugging Face model converted the way the page converts it (the conduct of a conversion, fed the file in order),
# for tests/perplexity.mjs and tests/perplexity_native.py (T85): <out>.bin, <out>.tokenizer.bin and <out>.json,
# the options the page would give Llama(). convert_hf.py does not say those options; the page's path does.
#
#   python3 tests/perplexity_prepare.py <directory with config.json, model.safetensors and the tokenizer | a .gguf> <out> [int8|float32] [--entry <id>]
#
# --entry <id> (the review of T247): the options of that entry of src/models.js under the converter's, as the worker merges them
# ({...engineOptions, ...model.options}), as tests/write_options.py makes them. Without it the options are the converter's alone,
# which is what ?hf= opens: of the files the folder has (T374.4: tokenizer_config.json and chat_template.jinja are read where
# they are there, as the page asks for them, so the BOS and the stops are those the converter takes from the model's template,
# T369; without them, or without jinja2, a Qwen3.5 has a BOS of <|endoftext|>, where the list's entries begin with <|im_start|>,
# T236, and a text 18% to 45% likelier to read with it, 2B and 4B). A measurement of a model of the list is of the page's way
# with --entry.
#
# A directory with config.json, the tokenizer and a .gguf (tests/hf_fetch.py makes it for T136's second stage): the
# GGUF's weights with the original's vocabulary and configuration, as the page reads them. What a path stands for is
# tests/conducting.py's listed().
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conducting import Mapped, converted, listed  # noqa: E402

arguments = sys.argv[1:]
entry_id = None
if "--entry" in arguments:
    at = arguments.index("--entry")
    entry_id = arguments[at + 1]
    del arguments[at:at + 2]
directory, out = Path(arguments[0]), arguments[1]
dtype = arguments[2] if len(arguments) > 2 else "int8"


def entry_options(entry_id):
    """What the list gives the engine for this entry besides what the converter makes (src/models.js's options)."""
    script = (f"import('./src/models.js').then(({{ MODELS }}) => {{ const m = MODELS.find((m) => m.id === {json.dumps(entry_id)}); "
              "console.log(JSON.stringify(m ? m.options ?? {} : null)); })")
    options = json.loads(subprocess.check_output(["node", "-e", script], cwd=Path(__file__).resolve().parent.parent))
    if options is None:
        raise SystemExit(f"{entry_id} is no entry of src/models.js")
    return options


# T374.4: by the conduct of a conversion (public/convert/conduct.py), answered from the folder as the page's worker
# answers it from huggingface.co: which files are read, the tokenizer (the conduct's candidates, the first the converter
# reads) and the template (tokenizer_config.json, or chat_template.jinja where that has none) are the page's, of whatever
# of them the folder has. The checkpoint goes straight into <out>.bin
hf, answerer = listed(directory)
sink = Mapped(f"{out}.bin")
conversion = converted(answerer, hf, dtype=dtype, sink=sink)
sink.data.flush()
Path(f"{out}.tokenizer.bin").write_bytes(conversion.tokenizer)
options = {key: value for key, value in conversion.options.items() if key != "template"}
if entry_id:
    options = {**options, **entry_options(entry_id)}
Path(f"{out}.json").write_text(json.dumps(options))
print(f"{out}.bin: {Path(f'{out}.bin').stat().st_size:,} bytes, options {options}")
