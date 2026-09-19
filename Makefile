

.PHONY: run clean

run:	stories15M.bin tokenizer.bin node_modules/.bin/http-server
	npx http-server -a 0.0.0.0 -p 8080 --cors

stories15M.bin:
	wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

tokenizer.bin:
	wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin

node_modules/.bin/http-server:
	yarn

clean:
	rm -f stories15M.bin tokenizer.bin
	rm -rf node_modules

