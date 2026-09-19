FROM node:21

WORKDIR /app
# convert_hf.py needs NumPy to convert the Hugging Face checkpoint
RUN apt-get update && apt-get install -y --no-install-recommends python3-numpy && rm -rf /var/lib/apt/lists/*
COPY . .
RUN make models
RUN yarn

EXPOSE 8080
CMD ["npx", "http-server", "-a", "0.0.0.0", "-p", "8080", "--cors"]
