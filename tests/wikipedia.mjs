// wikipedia.mjs: the beginnings of Wikipedia articles as plain text, for measurements on a real text (T55, T85,
// T100). Fetched when a measurement runs: nothing of it is stored in this repository.
//
//   import { wikipediaText } from "./wikipedia.mjs";  await wikipediaText("ja", ["富士山", "夏目漱石", "新幹線"])
//   node tests/wikipedia.mjs <en | ja> <out file> [title ...]   (the three articles of T85 by default)
import fs from "node:fs";
import { fetched } from "./fetching.mjs";

// T85's articles: the same three subjects in both languages
export const ARTICLES = { en: ["Mount Fuji", "Natsume Sōseki", "Shinkansen"], ja: ["富士山", "夏目漱石", "新幹線"] };

// (the review of T236: a CI runner's connection to Wikipedia timed out once, and a measurement an hour in began with
// nothing: a few tries, 10, 20, 30 and 40 seconds apart. T357.6: the tries are tests/fetching.mjs's, which asks again
// for what another try may mend, and not for what Wikipedia refuses (a 4xx))
const ASKING = {
  tries: 5, wait: 10000, timeout: 60000, headers: { "User-Agent": "pyodide-llm perplexity measurement" },
  said: (error, attempt) => console.error(`wikipedia.mjs: attempt ${attempt} failed (${error.cause?.code ?? error.message}); trying again`),
};
const pagesOf = async (url, asking) => JSON.parse(await fetched(url, { ...ASKING, ...asking })).query.pages;

/** (asking: tests/fetching.mjs's options over the ones above: a check's own fetch and sleep) */
export async function wikipediaText(language, titles = ARTICLES[language], asking = {}) {
  let text = "";
  for (const title of titles) {
    const url = `https://${language}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&exsectionformat=plain&format=json&titles=${encodeURIComponent(title)}`;
    const pages = await pagesOf(url, asking);
    // the beginning of each article: prose, before the lists and tables of the later sections
    text += Object.values(pages)[0].extract.slice(0, 6000) + "\n";
  }
  return text;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [language, out, ...titles] = process.argv.slice(2);
  fs.writeFileSync(out, await wikipediaText(language, titles.length ? titles : undefined));
}
