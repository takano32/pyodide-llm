

.PHONY: run models clean

TINYLLAMAS = https://huggingface.co/karpathy/tinyllamas/resolve/main
TINY_LM = https://huggingface.co/sbintuitions/tiny-lm/resolve/main

# No binary is committed: every model is downloaded (and tiny-lm converted) here, also when the site is deployed
MODELS = tiny-lm.bin stories15M.bin tokenizer.bin stories3_5M-v4k.bin tok4096.bin stories260K.bin tok512.bin stories42M.bin

run:	models node_modules/.bin/http-server
	npx http-server -a 0.0.0.0 -p 8080 --cors

models:	$(MODELS)

# Japanese / English model in Hugging Face format: convert_hf.py writes tiny-lm.bin and tiny-lm.tokenizer.bin
tiny-lm.bin:	convert_hf.py
	mkdir -p tiny-lm
	for f in config.json pytorch_model.bin spiece.model LICENSE; do wget -q -O tiny-lm/$$f $(TINY_LM)/$$f || exit 1; done
	python3 convert_hf.py tiny-lm tiny-lm float16
	cp tiny-lm/LICENSE tiny-lm.LICENSE.txt

stories15M.bin stories42M.bin:
	wget -q $(TINYLLAMAS)/$@

stories260K.bin tok512.bin:
	wget -q $(TINYLLAMAS)/stories260K/$@

stories3_5M-v4k.bin tok4096.bin:
	wget -q https://huggingface.co/ellishg/tinyllamas/resolve/main/$@

tokenizer.bin:
	wget -q https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin

node_modules/.bin/http-server:
	yarn

clean:
	rm -f *.bin tiny-lm.LICENSE.txt
	rm -rf tiny-lm node_modules

