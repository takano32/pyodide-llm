

.PHONY: run models clean
.SECONDARY:

TINYLLAMAS = https://huggingface.co/karpathy/tinyllamas/resolve/main
TINY_LM = https://huggingface.co/sbintuitions/tiny-lm/resolve/main
LLM_JP = https://huggingface.co/llm-jp/llm-jp-3-150m/resolve/main

# No binary is committed: every model is downloaded, converted and quantized here, also when the site is deployed.
# The larger models are distributed as int8; the unquantized originals (*.f32, *.f16) can be selected as well.
CHECKPOINTS = llm-jp-3-150m.bin tiny-lm.bin stories15M.bin stories42M.bin stories3_5M-v4k.bin stories260K.bin \
              llm-jp-3-150m.f16 tiny-lm.f16 stories15M.f32 stories42M.f32
TOKENIZERS = llm-jp-3-150m.tokenizer.bin tiny-lm.tokenizer.bin tokenizer.bin tok4096.bin tok512.bin

run:	models node_modules/.bin/http-server
	npx http-server -a 0.0.0.0 -p 8080 --cors

# models/ is what the page fetches: the checkpoints are cut into parts of 8 MiB, which worker.js downloads in parallel
models:	models/.done

models/.done:	$(CHECKPOINTS) $(TOKENIZERS)
	rm -rf models && mkdir models
	for f in $(CHECKPOINTS); do split -b 8388608 -d -a 3 $$f models/$$f. || exit 1; done
	cp $(TOKENIZERS) tiny-lm.LICENSE.txt models/
	touch $@

# int8, 3.5x smaller than float32 (quantize.py)
%.bin:	%.f32 quantize.py
	python3 quantize.py $< $@

# Hugging Face checkpoints: convert_hf.py writes <out>.bin and <out>.tokenizer.bin
llm-jp-3-150m/model.safetensors:
	mkdir -p llm-jp-3-150m
	for f in config.json tokenizer.json model.safetensors; do wget -q -O llm-jp-3-150m/$$f $(LLM_JP)/$$f || exit 1; done

# its context of 4096 tokens is cut to 512: the KV cache of the full length alone would take 200 MB
llm-jp-3-150m.f32 llm-jp-3-150m.tokenizer.bin &:	llm-jp-3-150m/model.safetensors convert_hf.py
	python3 convert_hf.py llm-jp-3-150m llm-jp-3-150m float32 512
	mv llm-jp-3-150m.bin llm-jp-3-150m.f32

# both Hugging Face models are published as bfloat16, so float16 is their original precision
llm-jp-3-150m.f16:	llm-jp-3-150m/model.safetensors convert_hf.py
	python3 convert_hf.py llm-jp-3-150m llm-jp-3-150m-f16 float16 512
	mv llm-jp-3-150m-f16.bin llm-jp-3-150m.f16
	rm llm-jp-3-150m-f16.tokenizer.bin

tiny-lm/pytorch_model.bin:
	mkdir -p tiny-lm
	for f in config.json spiece.model LICENSE pytorch_model.bin; do wget -q -O tiny-lm/$$f $(TINY_LM)/$$f || exit 1; done

tiny-lm.f32 tiny-lm.tokenizer.bin tiny-lm.LICENSE.txt &:	tiny-lm/pytorch_model.bin convert_hf.py
	python3 convert_hf.py tiny-lm tiny-lm
	mv tiny-lm.bin tiny-lm.f32
	cp tiny-lm/LICENSE tiny-lm.LICENSE.txt

tiny-lm.f16:	tiny-lm/pytorch_model.bin convert_hf.py
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

node_modules/.bin/http-server:
	yarn

clean:
	rm -f *.bin *.f32 *.f16 tiny-lm.LICENSE.txt
	rm -rf models tiny-lm llm-jp-3-150m node_modules

