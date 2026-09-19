FROM node:24

WORKDIR /app
# convert_hf.py and quantize.py need NumPy to convert the checkpoints
RUN apt-get update && apt-get install -y --no-install-recommends python3-numpy && rm -rf /var/lib/apt/lists/*
COPY . .
RUN npm ci
RUN make models kernels
RUN npm run build

EXPOSE 8080
CMD ["npm", "run", "preview"]
