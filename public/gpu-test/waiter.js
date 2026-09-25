// T94, stage 0: the model's worker as stage 1 would have it: it cannot return to its event loop while Python calls
// forward(), so it hands the token to the GPU's worker and blocks in Atomics.wait until the answer is there. Here the
// answer is nothing but a counter: this measures the round trip alone. words[0]: the requests, words[1]: the answers.
onmessage = ({ data: { memory, rounds } }) => {
  const words = new Int32Array(memory);
  const began = performance.now();
  for (let i = 1; i <= rounds; i++) {
    Atomics.store(words, 0, i);
    Atomics.notify(words, 0);
    for (let seen = Atomics.load(words, 1); seen !== i; seen = Atomics.load(words, 1)) Atomics.wait(words, 1, seen);
  }
  postMessage({ microseconds: ((performance.now() - began) * 1000) / rounds });
};
