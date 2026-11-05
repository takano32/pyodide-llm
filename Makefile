

.PHONY: run models kernels python clean
.SECONDARY:

TINYLLAMAS = https://huggingface.co/karpathy/tinyllamas/resolve/main
TINY_LM = https://huggingface.co/sbintuitions/tiny-lm/resolve/main
LLM_JP = https://huggingface.co/llm-jp/llm-jp-3-150m/resolve/main

# No binary is committed: every model is downloaded, converted and quantized here, also when the site is deployed.
# The larger models are distributed as int8; the unquantized originals (*.f32, *.f16) can be selected as well.
CHECKPOINTS = llm-jp-3-150m.bin tiny-lm.bin stories15M.bin stories42M.bin stories3_5M-v4k.bin stories260K.bin \
              llm-jp-3-150m.f16 tiny-lm.f16 stories15M.f32 stories42M.f32
TOKENIZERS = llm-jp-3-150m.tokenizer.bin tiny-lm.tokenizer.bin tokenizer.bin tok4096.bin tok512.bin

# http://localhost:8080/pyodide-llm/
run:	models kernels node_modules
	npm run dev

# public/models/ is what the page fetches: the checkpoints are cut into parts of 8 MiB, which worker.js downloads
# in parallel
models:	public/models/.done

public/models/.done:	$(CHECKPOINTS) $(TOKENIZERS) translations/.done
	rm -rf public/models && mkdir -p public/models
	for f in $(CHECKPOINTS); do split -b 8388608 -d -a 3 $$f public/models/$$f. || exit 1; done
	cp $(TOKENIZERS) tiny-lm.LICENSE.txt public/models/
	# T648.5: the translation models as the checkpoints, a model and a shortlist in parts of 8 MiB, the vocabularies
	# whole (as the tokenizers): public/models/translations/<model>/<the record's name>[.000…]
	for d in translations/*/; do mkdir -p public/models/$$d && \
		for f in $$d*.bin; do split -b 8388608 -d -a 3 $$f public/models/$$f. || exit 1; done && \
		cp $$d*.spm public/models/$$d || exit 1; done
	cp translations/LICENSE.txt public/models/translations/
	touch $@

# T648.5: Mozilla's Firefox Translations models, their vocabularies and shortlists (fetch_translations.py names each
# file with its address, length and sha256, and makes a folder of each model: en-ja-3.0, ja-en-3.0, ja-en-3.1)
translations/.done:	fetch_translations.py LICENSE
	python3 fetch_translations.py translations
	touch $@

# The Python that makes a model: the two windows and every part under them (T434: the families' files and the
# engine's parts were not named, and a change to one of them left the old models in place)
MODEL_PYTHON = src/python/llama2_convert.py src/python/llama2_numpy.py \
	$(wildcard src/python/convert/*.py src/python/convert/families/*.py src/python/engine/*.py)
# (defined before the rules that name it: make expands a rule's prerequisites as it reads the rule; T628.1's review)

# The WASM SIMD kernels that llama2_numpy.py loads with ctypes, compiled by AssemblyScript (no binary is committed).
# T628.1: and the facts of the Python that the TypeScript reads before any Python runs (python_facts.py), made here
# because everything that runs the TypeScript makes the kernels first (the deploy: `make models kernels`, then the checks)
kernels:	src/runtime/built/simdkernel.so src/runtime/built/facts/.made

src/runtime/built/facts/.made:	python_facts.py $(MODEL_PYTHON)
	python3 python_facts.py
	touch $@

# (every source of kernels/ is a prerequisite, read off the folder: a file the rule did not name left the old kernels in
# src/runtime/built/ when it changed, and the checks then tested those, and nothing said so; T644.4. kernel.ts is the
# window over kernel/*.ts, T356; blocks.ts and blocks_relaxed.ts over blocks/*.ts, T363.4)
KERNEL_SOURCES = $(wildcard kernels/*.ts kernels/*/*.ts kernels/*.mjs) kernels/build.py
src/runtime/built/simdkernel.so:	$(KERNEL_SOURCES) node_modules
	python3 kernels/build.py src/runtime/built

# The site's Python (src/python/) as two archives, engine.zip and converter.zip, which the worker fetches whole and
# unpacks in Pyodide (python_archive.py, T367.2; no binary is committed). Built every time it is asked for: it takes a
# tenth of a second, and an archive older than its sources is then nowhere. `npm run build` and `npm run dev` build
# them first (package.json).
python:
	python3 python_archive.py

# int8, 3.5x smaller than float32 (quantize.py)
%.bin:	%.f32 quantize.py $(MODEL_PYTHON)
	python3 quantize.py $< $@

# Hugging Face checkpoints: convert_hf.py writes <out>.bin and <out>.tokenizer.bin
llm-jp-3-150m/model.safetensors:
	mkdir -p llm-jp-3-150m
	for f in config.json tokenizer.json model.safetensors; do wget -q -O llm-jp-3-150m/$$f $(LLM_JP)/$$f || exit 1; done

# llm-jp-3-150m keeps its whole context of 4096 tokens: the engine lets the KV cache (200 MB at that length) grow
# with the text instead of reserving it.
# convert_hf.py writes int8 directly (the same bytes as float32 followed by quantize.py, without the 600 MB between)
CONVERTER = convert_hf.py $(MODEL_PYTHON)

llm-jp-3-150m.bin llm-jp-3-150m.tokenizer.bin &:	llm-jp-3-150m/model.safetensors $(CONVERTER)
	python3 convert_hf.py llm-jp-3-150m llm-jp-3-150m int8 4096

# both Hugging Face models are published as bfloat16, so float16 is their original precision
llm-jp-3-150m.f16:	llm-jp-3-150m/model.safetensors $(CONVERTER)
	python3 convert_hf.py llm-jp-3-150m llm-jp-3-150m-f16 float16 4096
	mv llm-jp-3-150m-f16.bin llm-jp-3-150m.f16
	rm llm-jp-3-150m-f16.tokenizer.bin

tiny-lm/pytorch_model.bin:
	mkdir -p tiny-lm
	for f in config.json spiece.model LICENSE pytorch_model.bin; do wget -q -O tiny-lm/$$f $(TINY_LM)/$$f || exit 1; done

tiny-lm.bin tiny-lm.tokenizer.bin tiny-lm.LICENSE.txt &:	tiny-lm/pytorch_model.bin $(CONVERTER)
	python3 convert_hf.py tiny-lm tiny-lm int8
	cp tiny-lm/LICENSE tiny-lm.LICENSE.txt

tiny-lm.f16:	tiny-lm/pytorch_model.bin $(CONVERTER)
	python3 convert_hf.py tiny-lm tiny-lm-f16 float16
	mv tiny-lm-f16.bin tiny-lm.f16
	rm tiny-lm-f16.tokenizer.bin

stories15M.f32 stories42M.f32:
	wget -q -O $@ $(TINYLLAMAS)/$(basename $@).bin

stories260K.bin tok512.bin:
	wget -q $(TINYLLAMAS)/stories260K/$@

stories3_5M-v4k.bin tok4096.bin:
	wget -q https://huggingface.co/ellishg/tinyllamas/resolve/main/$@

tokenizer.bin:
	wget -q https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin

node_modules:	package-lock.json
	npm ci

clean:
	rm -f *.bin *.f32 *.f16 tiny-lm.LICENSE.txt
	rm -rf public/models src/runtime/built dist .astro tiny-lm llm-jp-3-150m translations node_modules

