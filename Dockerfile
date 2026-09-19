FROM node:21

WORKDIR /app
COPY . .
RUN wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
RUN wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
RUN yarn

EXPOSE 8080
CMD ["npx", "http-server", "-a", "0.0.0.0", "-p", "8080", "--cors"]

