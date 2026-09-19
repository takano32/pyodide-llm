

.PHONY: run models clean
.SECONDARY:

TINYLLAMAS = https://huggingface.co/karpathy/tinyllamas/resolve/main
TINY_LM = https://huggingface.co/sbintuitions/tiny-lm/resolve/main

# No binary is committed: every model is downloaded (and tiny-lm converted) here, also when the site is deployed
MODELS = tiny-lm.bin stories15M.bin tokenizer.bin stories3_5M-v4k.bin tok4096.bin stories260K.bin tok512.bin stories42M.bin

run:	models node_modules/.bin/http-server
	npx http-server -a 0.0.0.0 -p 8080 --cors

models:	$(MODELS)

# The larger models are distributed as int8, 3.5x smaller than float32 (quantize.py); *.f32 files are not deployed
%.bin:	%.f32 quantize.py
	python3 quantize.py $< $@

# Japanese / English model in Hugging Face format: convert_hf.py writes tiny-lm.bin and tiny-lm.tokenizer.bin
tiny-lm.f32:	convert_hf.py
	mkdir -p tiny-lm
	for f in config.json pytorch_model.bin spiece.model LICENSE; do wget -q -O tiny-lm/$$f $(TINY_LM)/$$f || exit 1; done
	python3 convert_hf.py tiny-lm tiny-lm
	mv tiny-lm.bin tiny-lm.f32
	cp tiny-lm/LICENSE tiny-lm.LICENSE.txt

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
	rm -f *.bin *.f32 tiny-lm.LICENSE.txt
	rm -rf tiny-lm node_modules

