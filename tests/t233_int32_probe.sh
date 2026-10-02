#!/usr/bin/env bash
# T233 review probe: the old arithmetic (T129's review: int(np.prod(...)), 32 bits in Pyodide) put back in llama2_convert.py one
# place at a time; tests/t233_int32_probe.mjs (the 27B's section of tests/smoke.mjs) must fail on each, and pass on the original
cd "$(dirname "$0")/.."
cp public/llama2_convert.py .tmp/llama2_convert.py.orig
echo "probe: the original"
node tests/t233_int32_probe.mjs || echo "probe: the ORIGINAL FAILS"
mutate() {  # <name> <old text> <new text>
  cp .tmp/llama2_convert.py.orig public/llama2_convert.py
  python - "$2" "$3" <<'PYTHON'
import sys
from pathlib import Path
old, new = sys.argv[1:3]
path = Path("public/llama2_convert.py")
text = path.read_text()
assert text.count(old) == 1, (text.count(old), old)
path.write_text(text.replace(old, new))
PYTHON
  echo "probe: mutant $1"
  if node tests/t233_int32_probe.mjs; then echo "probe: mutant $1 NOT CAUGHT"; else echo "probe: mutant $1 caught"; fi
}
mutate "tensor_bytes" "count, dtype = math.prod(shape), dtype_name(dtype)" "count, dtype = int(np.prod(shape)), dtype_name(dtype)"
mutate "the place of a ternary matrix's scales" "self.put(offset + math.prod(shape) // 4 + 4 * (first // TERNARY_GROUP), scales)" "self.put(offset + int(np.prod(shape)) // 4 + 4 * (first // TERNARY_GROUP), scales)"
cp .tmp/llama2_convert.py.orig public/llama2_convert.py
