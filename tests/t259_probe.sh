#!/usr/bin/env bash
# tests/t259_probe.sh (T259, a probe: not for main): DeepSeek-R1-0528-Qwen3-8B's perplexity on the page's forward
# pass (int8, the kernels) with three readings of its yarn RoPE, which differ in the options alone (the engine makes
# the RoPE tables from them):
#   yarn:  config.json's yarn without attn_factor, as transformers and llama.cpp read it (cos and sin times 0.1 ln 4 + 1)
#   attn:  the same angles with cos and sin times attn_factor (0.1 ln 4 + 1), as vLLM 0.9 and SGLang read it (1.0)
#   plain: no scaling (rope_scaling {}), as the page has the Ternary Bonsai
# The weights are bartowski's Q8_0 (int8 again without a loss), the configuration and vocabulary the original's.
#
#   bash tests/t259_probe.sh <tokens> <window> <languages> <readings>
set -euo pipefail
cd "$(dirname "$0")/.."
tokens="${1:-1500}"
window="${2:-512}"
languages="${3:-ja en}"
readings="${4:-yarn attn plain}"
dir="${RUNNER_TEMP:-.tmp}/t259"
mkdir -p "$dir/model"
echo "t259: $(node -p 'require("os").cpus()[0].model') ($(uname -m)), tokens $tokens, window $window"
df -h "$dir" | tail -1
python3 - "$dir/model" <<'PYTHON'
import json, sys
from pathlib import Path
sys.path.insert(0, "tests")
from fetching import download
room = Path(sys.argv[1])
original = "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/resolve/6e8885a6ff5c1dc5201574c8fd700323f23c25fa/"
for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
    download(original + name, room / name)
config = json.loads((room / "config.json").read_text())
print("t259: config.json says rope_scaling", json.dumps(config["rope_scaling"]))
# the converter refuses attn_factor (check_config()): converted without it, and the readings are the options'
config["rope_scaling"].pop("attn_factor", None)
(room / "config.json").write_text(json.dumps(config))
download("https://huggingface.co/bartowski/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-GGUF/resolve/"
         "75b8e3d4ecafa21b1f9a06209712f726d0ed77bc/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
         room / "model.gguf")
PYTHON
if [ ! -f "$dir/made.bin" ]; then
  python3 tests/perplexity_prepare.py "$dir/model" "$dir/made" int8 | cut -c1-300
fi
rm -f "$dir/model/model.gguf"
df -h "$dir" | tail -1
python3 - "$dir" <<'PYTHON'
import json, math, os, sys
from pathlib import Path
room = Path(sys.argv[1])
made = json.loads((room / "made.json").read_text())
yarn = made["rope_scaling"]
print("t259: the conversion's options: rope_scaling", json.dumps(yarn), "bos", made.get("bos"), "dtype", made["dtype"])
attn = {**yarn, "attention_factor": 0.8782488562869419 * (0.1 * math.log(yarn["factor"]) + 1)}
for name, scaling in (("yarn", yarn), ("attn", attn), ("plain", {})):
    (room / f"{name}.json").write_text(json.dumps({**made, "rope_scaling": scaling}))
    for end in ("bin", "tokenizer.bin"):
        link = room / f"{name}.{end}"
        link.unlink(missing_ok=True)
        os.symlink((room / f"made.{end}").resolve(), link)
    print(f"t259: {name}: rope_scaling {json.dumps(scaling)}")
PYTHON
for language in $languages; do
  node tests/wikipedia.ts "$language" "$dir/$language.txt"
  for which in $readings; do
    node tests/perplexity.ts "$dir/$which" "$tokens" "$dir/$language.txt" --file --rows=0 --wide --window="$window" \
      | tee "$dir/$which.$language.txt"
    echo "t259: $language $which window $window tokens $tokens: $(grep -m1 '^| kernels' "$dir/$which.$language.txt")"
  done
done
