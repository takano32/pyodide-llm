// t236_loops.mjs (T236's review, a probe for CI, not for main): a model of the list as the page runs it (forward.js on
// the int8 kernels, the entry's options, template and sampling, the context 4096 as steps: 0 says), over several prompts,
// to count what the owner asked after the card's warning that Qwen3.5 0.8B "is more prone to entering thinking loops":
// how many answers reach </think>, and how many then stop (a stop token) within the context.
//
//   node tests/t236_loops.mjs '{"out": ".tmp/g8", "id": "hf-qwen3.5-0.8b-thinking", "prompts": [...], "seed": 1000, "penalty": null}'
// penalty: a repetition penalty in place of the entry's (the page's own way against a loop; the card's presence
// penalty the sampler does not have).
import fs from "node:fs";
import { pyodideWithEngine } from "./engine.mjs";
import { MODELS, filled } from "../src/models.js";

const spec = JSON.parse(process.argv[2]);
const entry = MODELS.find((model) => model.id === spec.id);
if (!entry) throw new Error(`no entry ${spec.id}`);
// spec.tiny: a made-up model of tests/make_qwen35.py (the entry's tokens do not exist in it), only to run this script's calls
const options = { ...JSON.parse(fs.readFileSync(`${spec.out}.json`, "utf8")), ...(spec.tiny ? {} : entry.options) };
const sampling = { ...(spec.tiny ? { steps: 30, temperature: 0.7, topp: 0.8, repetition_penalty: 1.0 } : entry.generation),
  ...(spec.penalty ? { repetition_penalty: spec.penalty } : {}) };
const template = spec.tiny ? "{prompt}" : entry.template;
console.log(`T236LOOPS ${spec.id}: sampling ${JSON.stringify(sampling)}, options ${JSON.stringify({ ...options, specials: `${options.specials?.length} of them` })}, template ${JSON.stringify(template)}`);
const { pyodide } = await pyodideWithEngine();
pyodide.FS.writeFile("tokenizer.bin", fs.readFileSync(`${spec.out}.tokenizer.bin`));
pyodide.globals.set("SPEC", JSON.stringify({ file: `${process.cwd()}/${spec.out}.bin`, options, sampling, seed: spec.seed ?? 1000,
  prompts: spec.prompts.map((prompt) => filled(template, prompt)), asked: spec.prompts, id: spec.id }));
pyodide.runPython(`
import json, math, time, re
spec = json.loads(SPEC)
llama = kernel_llama_file(spec["file"], open("tokenizer.bin", "rb").read(), **spec["options"])
s = spec["sampling"]
print("T236LOOPS", spec["id"], "seq_len", llama.seq_len, "backend", llama.backend, flush=True)

def repeats(text):
    """the most often the same 40 characters occur in the last 3000 characters: a loop says itself here"""
    tail = text[-3000:]
    best = 0
    for i in range(0, max(1, len(tail) - 40), 20):
        best = max(best, tail.count(tail[i:i + 40]))
    return best

for i, prompt in enumerate(spec["prompts"]):
    began = time.perf_counter()
    pieces = []
    for piece in llama.generate(prompt, steps=s.get("steps", 0), temperature=s["temperature"], topp=s["topp"],
                                repetition_penalty=s.get("repetition_penalty", 1.0), seed=spec["seed"] + i, echo=False):
        pieces.append(piece)
    text = "".join(pieces)
    stats = llama.stats
    sampled, count, forced = stats["sampled"], stats["tokens"], stats["prompt_tokens"]
    stopped = sampled > count - forced  # a stop token ended it: one more was sampled than was written
    closed = "</think>" in text
    before = text.split("</think>")[0] if closed else text
    row = {"i": i, "prompt": spec["asked"][i], "prompt_tokens": forced, "written": count - forced, "stopped": stopped,
           "closed_think": closed, "thought_tokens": len(llama.tokenizer.encode(before)) if closed else None,
           "answer_chars": len(text.split("</think>", 1)[1].strip()) if closed else None, "max_repeat_of_40_chars": repeats(text),
           "seconds": round(time.perf_counter() - began), "tok_s": round(stats["tokens_per_second"], 1),
           "head": text[:160], "tail": text[-260:]}
    print("T236LOOPS", spec["id"], json.dumps(row, ensure_ascii=False), flush=True)
`);
process.exit(0);
