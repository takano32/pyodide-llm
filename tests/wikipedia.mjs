// wikipedia.mjs: the beginnings of Wikipedia articles as plain text, for measurements on a real text (T55, T85,
// T100). Fetched when a measurement runs: nothing of it is stored in this repository.
//
//   import { wikipediaText } from "./wikipedia.mjs";  await wikipediaText("ja", ["富士山", "夏目漱石", "新幹線"])
//   node tests/wikipedia.mjs <en | ja> <out file> [title ...]   (the three articles of T85 by default)
import fs from "node:fs";

// T85's articles: the same three subjects in both languages
export const ARTICLES = { en: ["Mount Fuji", "Natsume Sōseki", "Shinkansen"], ja: ["富士山", "夏目漱石", "新幹線"] };

// (the review of T236: a CI runner's connection to Wikipedia timed out once, and a measurement an hour in began with
// nothing: a few tries, 10, 20, 30 and 40 seconds apart)
async function pagesOf(url) {
  for (let attempt = 1; ; attempt++) {
    try {
      const response = await fetch(url, { headers: { "User-Agent": "pyodide-llm perplexity measurement" }, signal: AbortSignal.timeout(60000) });
      if (!response.ok) throw new Error(`Wikipedia answered ${response.status}`);
      return (await response.json()).query.pages;
    } catch (error) {
      if (attempt >= 5) throw error;
      console.error(`wikipedia.mjs: attempt ${attempt} failed (${error.cause?.code ?? error.message}); trying again`);
      await new Promise((resolve) => setTimeout(resolve, 10000 * attempt));
    }
  }
}

export async function wikipediaText(language, titles = ARTICLES[language]) {
  let text = "";
  for (const title of titles) {
    const url = `https://${language}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&exsectionformat=plain&format=json&titles=${encodeURIComponent(title)}`;
    const pages = await pagesOf(url);
    // the beginning of each article: prose, before the lists and tables of the later sections
    text += Object.values(pages)[0].extract.slice(0, 6000) + "\n";
  }
  return text;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [language, out, ...titles] = process.argv.slice(2);
  fs.writeFileSync(out, await wikipediaText(language, titles.length ? titles : undefined));
}
