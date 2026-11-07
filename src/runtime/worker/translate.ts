// worker/translate.ts (T648.4): the load of a translation model of this site (Firefox Translations', `make models`),
// which answers a text with its translation. Its model's file comes as a checkpoint of this site does (sources.ts's
// download(): parts of 8 MiB in parallel, kept in the Cache API, while Pyodide loads) into a bytearray of Python's, and
// its vocabularies whole; Python's engine/translating.py makes the translator of the bytes (engine/marian.py reads the
// model where it is, engine/translation.py translates, its affines on the SIMD kernel affine_q8), which answers by a
// Llama's names (contracts/python.ts's Answering): worker.ts's generate() runs it as it runs a Llama.
// The vocabularies are sentencepiece models, which the converter's Python reads: its archive is placed first
// (by convert.ts, as a conversion places it). The source's sentences are cut here, by the browser's
// Intl.Segmenter, as Firefox cuts a text before it translates; Python asks for them with every answer. T650: an entry
// that names its shortlist has it come as the model does (its progress unsaid) and handed to Python, which then
// writes the output layer's rows of the source's shortlist alone.
// A module of worker.ts; it takes its neighbours' names by static imports, and a bundler links them (T367.3).

import { pyodide, pyodideAt, pythonKernels, runtimeReady } from "./runtime.ts";
import { root, switchesNow } from "./request.ts";
import { adopted } from "./loaded.ts";
import { since, took } from "./clock.ts";
import { pythonBuffer } from "./weights.ts";
import { download, tokenizerAt } from "./sources.ts";
import { converter } from "./convert.ts";
import type { OfTranslation, Status } from "../contracts/messages.ts";
import type { Proxy } from "../contracts/python.ts";

/** The sentences of a text in a language, every character of it in one of them (Intl.Segmenter's; where a browser has
 * none, the text is one) */
export function sentencesOf(language: string) {
  if (typeof Intl.Segmenter !== "function") return (text: string) => [text];
  const segmenter = new Intl.Segmenter(language, { granularity: "sentence" });
  return (text: string) => Array.from(segmenter.segment(text), ({ segment }) => segment);
}

/** A translation model loaded and adopted as the model loaded now; what its ready says besides (no GPU: "") */
export async function translated(model: OfTranslation, signal: AbortSignal, id: number) {
  postMessage({ type: "status", load: id, text: `Downloading ${model.name}...` } satisfies Status);
  const { folder, model: file, source, target, from, to, shortlist } = model.translation;
  const started = performance.now();
  const checkpoint = download({ checkpoint: `translations/${folder}/${file}`, bytes: model.bytes }, signal, id);
  const listed = shortlist && download({ checkpoint: `translations/${folder}/${shortlist.file}`, bytes: shortlist.bytes }, signal, id, false);
  const vocabulary = (name: string) => tokenizerAt(new URL(`models/translations/${folder}/${name}`, root), name, signal);
  const vocabularies = Promise.all([vocabulary(source), source === target ? undefined : vocabulary(target)]);
  vocabularies.catch(() => {});
  let weights;
  try {
    await runtimeReady();
    signal.throwIfAborted();
    await converter(signal);
    weights = pythonBuffer(model.bytes);
  } catch (error) {
    checkpoint.stop(error);
    listed?.stop(error);
    throw error;
  }
  const proxies: Proxy[] = [weights.buffer];
  try {
    await checkpoint.into(weights.write);
    const lex = shortlist && pythonBuffer(shortlist.bytes);
    if (lex) {
      proxies.push(lex.buffer);
      await listed!.into(lex.write);
    }
    const [sourceBytes, targetBytes] = await vocabularies;
    signal.throwIfAborted();
    took("download", checkpoint.seconds ?? since(started));
    const made = performance.now();
    // (the vocabularies as bytearrays of Python's: one for both where the model has one)
    const bytearray = (bytes: ArrayBuffer) => {
      const placed = pythonBuffer(bytes.byteLength);
      placed.write(0, new Uint8Array(bytes));
      proxies.push(placed.buffer);
      return placed.buffer;
    };
    const words = bytearray(sourceBytes);
    const others = targetBytes ? bytearray(targetBytes) : words;
    const translating = pyodide.pyimport("engine.translating");
    proxies.push(translating);
    adopted({ llama: translating.translation.callKwargs(weights.buffer, words, others, to,
      { kernels: pythonKernels, disable: switchesNow(), sentences: sentencesOf(from), ...(lex && { shortlist: lex.buffer }) }) });
    took("construct", since(made));
  } catch (error) {
    listed?.stop(error);  // (the shortlist's download stops with the model's that failed)
    throw error;
  } finally {
    for (const proxy of proxies) proxy.destroy();
  }
  return { overlapped: pyodideAt > started, gpu: "" };
}
