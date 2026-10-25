// src/page/repository.ts (T.505.3): what the page makes of a repository of huggingface.co nobody has looked at, before
// the worker hears of it: which of its files is the model. Nothing of the page is read here (no element, no address):
// tests/repository-check.mjs imports this file as it is, in Node, and tests/worker-fetches-check.mjs makes the models
// it hands the worker by these very functions.
//
// A repository with model.safetensors (or the index of its shards) is opened as it always was. One without is looked
// through for a GGUF, which holds a model whole (its configuration and vocabulary before its tensors): the best of
// those it has, or the one the visitor named (?hf=owner/name&file=name.gguf, the third field of the sheet).
// Which requests a conversion then makes, and in which order, is the conduct's to say (src/python/convert/conduct.py):
// what is decided here is the one name it is handed, hf.weights.

/** The file the conversion of a repository asks for where nobody named one, and the index that stands for its shards. */
export const SAFETENSORS = "model.safetensors";
const INDEX = `${SAFETENSORS}.index.json`;

// The quantizations a GGUF's name may say, the best to take first (the owner, 2026-10-13: quality first, Q8_0 where
// there is one, else Q6_K, Q5_K_M, Q4_K_M). Each row: the spellings of one, in capitals. Every one of them is widened
// and converted to int8 here, so what is gained by a smaller one is the fetch alone, and what is lost is what its
// quantization lost. The floats come after Q8_0: the same int8 at two to four times the fetch (and before them the two
// types a ternary model is published in whole, T.235: the same weights again, at an eighth of the float's fetch). The rest follow by what
// they keep of a weight; the variants with larger embeddings (_L, _XL) after the plain one of the same bits.
// tests/repository-check.mjs has the second copy of this order.
export const QUANTIZATIONS = [
  ["Q8_0"], ["PQ2_0"], ["PTQ1_0"], ["F16", "FP16"], ["BF16"], ["F32", "FP32"],
  ["Q6_K"], ["Q6_K_L"], ["Q6_K_XL"], ["Q5_K_M"], ["Q5_K_L"], ["Q5_K_XL"], ["Q4_K_M"],
  ["Q5_K_S"], ["Q5_K"], ["Q5_1"], ["Q5_0"],
  ["Q4_K_L"], ["Q4_K_XL"], ["Q4_K_S"], ["Q4_K"], ["Q4_1"], ["Q4_0"], ["IQ4_NL"], ["IQ4_XS"],
  ["Q3_K_L"], ["Q3_K_XL"], ["Q3_K_M"], ["Q3_K"], ["Q3_K_S"],
  ["IQ3_M"], ["IQ3_S"], ["IQ3_XS"], ["IQ3_XXS"],
  ["Q2_K_L"], ["Q2_K_XL"], ["Q2_K"], ["Q2_K_S"], ["IQ2_M"], ["IQ2_S"], ["IQ2_XS"], ["IQ2_XXS"], ["IQ1_M"], ["IQ1_S"],
];
// the place of a name in that order: by the quantization it says as a word of its own (Model-Q4_K_M.gguf,
// model.q4_k_m.gguf, ggml-model-f16.gguf; Q4_K is no word of Q4_K_M), whatever the case of its letters. A name that
// says none comes after them all.
const PLACES = new Map(QUANTIZATIONS.flatMap((spellings, place) => spellings.map((spelling): [string, number] => [spelling, place])));
const WORDS = new RegExp(`(?<![A-Z0-9])(${[...PLACES.keys()].sort((a, b) => b.length - a.length).join("|")})(?![A-Z0-9_])`, "g");
export function quantization(name: string) {
  const said = [...name.toUpperCase().matchAll(WORDS)].map(([, word]) => PLACES.get(word)!);
  // (the last of them: a model's own name may hold one, as the quantization follows it)
  return said.length ? said[said.length - 1] : QUANTIZATIONS.length;
}

/** How long the list of a repository is waited for. */
export const LISTING_SECONDS = 10;
// T.508.1: the pages of a repository's list asked for at most (a thousand entries each)
export const PAGES = 10;
export type Listed = { path: string; size?: number };
// A GGUF this page may open: one file that is a model. Not a part of a model split over several files
// (-00001-of-00003.gguf), not the projector of a model that reads images (mmproj), and not the importance matrix a
// quantization was made with (imatrix.gguf: a file named for it, and not a model whose name ends in it).
const opens = (path: string) => /\.gguf$/.test(path) && !/-\d{5}-of-\d{5}\.gguf$/.test(path) && !/mmproj/i.test(path) && !/^imatrix/i.test(path);
/** The GGUF to take of a repository's files: the first in the order of QUANTIZATIONS (of two of one place, the first
 * by its name), or undefined where it has none this page can read. */
export function bestGguf(files: Listed[]) {
  const ranked = files.filter(({ path }) => opens(path)).map((file) => ({ file, place: quantization(file.path) }));
  ranked.sort((a, b) => a.place - b.place || (a.file.path < b.file.path ? -1 : a.file.path > b.file.path ? 1 : 0));
  return ranked[0]?.file;
}

type Entry = { name: string; note?: string; download?: number; hf: { repo?: string; revision?: string; weights?: string } };
/** Whether a model is one nobody named a file of: its conversion asks for model.safetensors. */
export const unnamed = (entry: Entry) => entry.hf.weights === SAFETENSORS;
/** What the page says of a model of a repository beside its name: where it is from, the revision that is not main,
 * the file that is not model.safetensors. */
export const noteOf = (revision: string, file: string) => ["Hugging Face", ...(revision === "main" ? [] : [revision.slice(0, 12)]), ...(file === SAFETENSORS ? [] : [file])].join(" · ");

/** The model the worker is handed, of an entry as hfEntry() makes it and the files its repository has (the top of
 * it, at the entry's revision). With a file named, that file (found or not: a name the repository does not have is the
 * worker's to say, in the words of every file that is not there). With none: the safetensors where the repository has
 * them, as it always was; else its best GGUF; else the sentence a visitor reads.
 * A GGUF that is found says how much will be fetched (download): the page asks before it fetches a large one. */
export function settled<Of extends Entry>(entry: Of, files: Listed[]): Of {
  const top = files.filter(({ path }) => !path.includes("/"));
  if (unnamed(entry) && top.some(({ path }) => path === SAFETENSORS || path === INDEX)) return entry;
  const file = unnamed(entry) ? bestGguf(top) : top.find(({ path }) => path === entry.hf.weights);
  if (!file) {
    if (!unnamed(entry)) return entry;
    // (the owner's sentence, 2026-10-13: written whole)
    throw new Error(`${entry.hf.repo} has no model.safetensors and no GGUF file this page can read.`);
  }
  const fetched = /\.gguf$/.test(file.path) && Number.isFinite(file.size) ? { download: file.size } : {};
  return { ...entry, ...fetched, note: noteOf(entry.hf.revision!, file.path), hf: { ...entry.hf, weights: file.path } };
}

/** The files at the top of a repository at a revision, as huggingface.co lists them, or undefined where it does not say
 * (no such repository, a gated one, no line): the model is then opened as named, and whatever is wrong with it is said
 * by the worker, which meets the same answer. T.508.1: a list of more than a page (1,000 entries) comes in pages, each
 * naming the next in its Link header (which huggingface.co lets a page read), up to PAGES of them. A list not had whole
 * (a later page that is no list, more pages than that, a next page elsewhere) is none: a part of it would have
 * settled() say a file is not there that a page not read has. */
export async function filesOf(repository: string, revision: string, fetching: typeof fetch = fetch): Promise<Listed[] | undefined> {
  try {
    // (it waits ten seconds at most, all the pages: a line that does not answer must not hold the model back, and the
    // same answer is then the worker's to meet)
    const signal = AbortSignal.timeout(LISTING_SECONDS * 1000), files: Listed[] = [];
    let url: string | null = `https://huggingface.co/api/models/${repository}/tree/${revision}`;
    for (let page = 0; url; page++) {
      if (page === PAGES || !url.startsWith("https://huggingface.co/")) return undefined;
      const res: Response = await fetching(url, { signal });
      const listed: unknown = res.ok ? await res.json() : undefined;
      if (!Array.isArray(listed)) return undefined;
      files.push(...listed.filter((one) => one?.type === "file" && typeof one.path === "string").map(({ path, size }) => ({ path, size })));
      url = /<([^>]+)>;\s*rel="next"/.exec(res.headers.get("Link") ?? "")?.[1] ?? null;
    }
    return files;
  } catch (error) {
    console.warn(`repository: the files of ${repository} could not be listed (${String(error)}): opened as named`);
    return undefined;
  }
}
/** An entry of hfEntry(), looked at: settled() by the files of its repository, or itself where they are not said. */
export async function looked<Of extends Entry>(entry: Of, fetching: typeof fetch = fetch): Promise<Of> {
  const files = await filesOf(entry.hf.repo!, entry.hf.revision!, fetching);
  return files ? settled(entry, files) : entry;
}
