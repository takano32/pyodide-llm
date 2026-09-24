// wikipedia.mjs: the beginnings of Wikipedia articles as plain text, for measurements on a real text (T55, T85,
// T100). Fetched when a measurement runs: nothing of it is stored in this repository.
//
//   import { wikipediaText } from "./wikipedia.mjs";  await wikipediaText("ja", ["富士山", "夏目漱石", "新幹線"])
//   node tests/wikipedia.mjs <en | ja> <out file> [title ...]   (the three articles of T85 by default)
import fs from "node:fs";

// T85's articles: the same three subjects in both languages
export const ARTICLES = { en: ["Mount Fuji", "Natsume Sōseki", "Shinkansen"], ja: ["富士山", "夏目漱石", "新幹線"] };

export async function wikipediaText(language, titles = ARTICLES[language]) {
  let text = "";
  for (const title of titles) {
    const url = `https://${language}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&exsectionformat=plain&format=json&titles=${encodeURIComponent(title)}`;
    const pages = (await (await fetch(url, { headers: { "User-Agent": "pyodide-llm perplexity measurement" } })).json()).query.pages;
    // the beginning of each article: prose, before the lists and tables of the later sections
    text += Object.values(pages)[0].extract.slice(0, 6000) + "\n";
  }
  return text;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [language, out, ...titles] = process.argv.slice(2);
  fs.writeFileSync(out, await wikipediaText(language, titles.length ? titles : undefined));
}
