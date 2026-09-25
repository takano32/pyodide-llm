// fetch-node.mjs (T136): how fast this machine fetches a file of Hugging Face by range requests, the way the worker
// does (inOrder(): parts of one size, several connections, each taking the next part), for each part size and
// number of connections. The bytes are counted and dropped: nothing is converted and little is held. For measuring
// the line of the machine it runs on (the owner's, in Japan), where the CI's (fetch.yml, T107) is in the United States.
// Node's fetch is not a browser's (HTTP/1.1 connections of its own, no HTTP/2 to share), so this measures the
// line and the server, not the page.
//
//   node tests/fetch-node.mjs [repo@revision/file] [megabytes] [parts MiB,...] [connections,...] [rounds]
//   node tests/fetch-node.mjs Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors 256 8,16,32 4,6,8 1
const [target = "Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors", megabytes = "256",
  partList = "8,16,32", connectionList = "4,6,8", rounds = "1"] = process.argv.slice(2);
const [, repo, revision, file] = target.match(/^([^@]+)@([^/]+)\/(.+)$/);
const url = `https://huggingface.co/${repo}/resolve/${revision}/${file}`;
const total = Number(megabytes) * 1e6;

async function fetchPart(begin, end) {
  for (let attempt = 1; ; attempt++) {
    try {
      const res = await fetch(url, { headers: { Range: `bytes=${begin}-${end - 1}` } });
      if (res.status !== 206) throw new Error(`status ${res.status}`);
      let got = 0;
      for await (const chunk of res.body) got += chunk.length;
      if (got !== end - begin) throw new Error(`${got} of ${end - begin} bytes`);
      return;
    } catch (error) {
      if (attempt === 3) throw error;
    }
  }
}

// every connection takes the next part until the stretch is fetched; the seconds from the first request to the last byte
async function run(partBytes, connections) {
  let next = 0;
  const started = performance.now();
  await Promise.all(Array.from({ length: connections }, async () => {
    while (next < total) {
      const begin = next, end = Math.min(begin + partBytes, total);
      next = end;
      await fetchPart(begin, end);
    }
  }));
  return (performance.now() - started) / 1000;
}

const first = performance.now();
await fetchPart(0, 1);
console.log(`${url}\none request of one byte: ${((performance.now() - first) / 1000).toFixed(2)} s (the wait of every part)`);
console.log(`${(total / 1e6).toFixed(0)} MB each run\n\n| part MiB | connections | seconds | MB/s |\n|---:|---:|---:|---:|`);
for (let round = 0; round < Number(rounds); round++) {
  for (const part of partList.split(",").map(Number)) {
    for (const connections of connectionList.split(",").map(Number)) {
      const seconds = await run(part * 1024 * 1024, connections);
      console.log(`| ${part} | ${connections} | ${seconds.toFixed(1)} | ${(total / 1e6 / seconds).toFixed(1)} |`);
    }
  }
}
