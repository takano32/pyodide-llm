// src/page/markdown.ts (T.556): an answer drawn as Markdown, by two libraries the page takes at run time, as the worker
// takes Pyodide (policy 2): the latest version of each is resolved on jsDelivr when it is first wanted and imported from
// there, and no version is written here (the owner, 2026-10-16: "人気あって更新されてるやつを常に最新で使う"). marked
// (MIT) reads the Markdown, and DOMPurify (Apache-2.0 or MPL-2.0) lets through the elements below and nothing else.
// Neither is in the bundle: no line of theirs is in this repository. Where they do not come (offline without a copy of
// them, a CDN that does not answer, a version that changed what is used here), the answers stay plain text.
//
// What is drawn: paragraphs, lists, bold, italic, headings, code (fenced and inline) and tables. HTML, links, images,
// quotes, rules, strikethrough and task boxes are said as the text the model wrote (the recommendation the owner chose
// for T.556: the text follows the visitor's prompt, so what it may put on the page is kept narrow). The form of drawing
// it block by block, the last one again as it grows, is naidan's src/components/block-markdown/ (852bdeab, MIT, Ryo
// Ota): no line of it is copied.

/** What is used of marked's Marked and DOMPurify's default export (both resolved at run time, typed here alone). */
type Token = { type: string; raw: string; text?: string; checked?: boolean };
type Tokens = Token[] & { links?: unknown };
type MarkedModule = { Marked: new (options: object) => { use(extension: object): unknown; lexer(text: string): Tokens; parser(tokens: Tokens): string } };
type Purify = { sanitize(html: string, options: object): DocumentFragment };
type PurifyModule = { default: Purify };
/** What the page draws an answer's Markdown with. */
export type Markdown = { blocks(text: string): Tokens; html(block: Token, links: unknown): DocumentFragment };

// the only elements and attributes an answer may draw (table cells say their alignment, a list its first number)
const ELEMENTS = ["p", "br", "strong", "em", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "code", "pre", "table", "thead", "tbody", "tr", "th", "td"];
const ATTRIBUTES = ["align", "start"];

const escaped = (text: string) => text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
/** marked's options and the renderer's parts that say what they read as text. Separate, so that Node can try it with the
 * real marked and no DOMPurify (tests/markdown-check.mjs). */
export function markedFor(marked: MarkedModule) {
  const reader = new marked.Marked({ gfm: true, breaks: true });
  const said = ({ raw }: Token) => escaped(raw);
  const saidAsBlock = ({ raw }: Token) => `<p>${escaped(raw.replace(/\n+$/, "")).replace(/\n/g, "<br>")}</p>\n`;
  reader.use({
    renderer: {
      html: said, link: said, image: said, del: said,
      blockquote: saidAsBlock, hr: saidAsBlock,
      checkbox: ({ checked }: Token) => (checked ? "[x] " : "[ ] "),
    },
  });
  return reader;
}
/** The page's Markdown, of the two modules. */
export function markdownOf(marked: MarkedModule, purify: Purify): Markdown {
  const reader = markedFor(marked);
  return {
    blocks: (text) => reader.lexer(text),
    // (T.559: and no data-* or aria-* either, which DOMPurify lets through beside ALLOWED_ATTR unless told not to)
    html: (block, links) => purify.sanitize(reader.parser(Object.assign([block], { links })), { ALLOWED_TAGS: ELEMENTS, ALLOWED_ATTR: ATTRIBUTES,
      ALLOW_DATA_ATTR: false, ALLOW_ARIA_ATTR: false, RETURN_DOM_FRAGMENT: true }),
  };
}

// a version becomes part of an address: nothing but a plain version number (pyodide.ts's form)
const VERSION = /^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$/;
const QUIET_SECONDS = 30;
/** The latest version of an npm package on jsDelivr, given up after QUIET_SECONDS. */
async function latest(name: string) {
  const answer = await fetch(`https://data.jsdelivr.com/v1/packages/npm/${name}/resolved?specifier=latest`, { signal: AbortSignal.timeout(QUIET_SECONDS * 1000) });
  const { version } = await answer.json();
  if (!VERSION.test(version)) throw new Error(`jsDelivr said ${JSON.stringify(version)} for the version of ${name}`);
  return version as string;
}
/** A package's module as jsDelivr bundles it (one file, whatever the package's own files are named in that version).
 * The address is made before import(): the bundler wraps an import() in a function that is not async. */
async function imported(name: string) {
  const address = `https://cdn.jsdelivr.net/npm/${name}@${await latest(name)}/+esm`;
  return import(/* @vite-ignore */ address);
}

let librariesAsked: Promise<Markdown | undefined> | undefined;
/** The Markdown of this page once it has come (undefined until then, and for the visit where it did not). */
export let markdownCame: Markdown | undefined;
/** Takes the two libraries, once a visit: the Markdown, or undefined where they did not come. */
export function markdownWanted() {
  librariesAsked ??= (async () => {
    try {
      const [marked, purify] = await Promise.all([imported("marked"), imported("dompurify")]) as [MarkedModule, PurifyModule];
      markdownCame = markdownOf(marked, purify.default);
    } catch (error) {
      console.warn(`markdown: its libraries did not come from jsDelivr (${String(error)}): the answers are plain text`);
    }
    return markdownCame;
  })();
  return librariesAsked;
}
/** Drawing with them failed (draw.ts): the answers of this visit are plain text from now on. */
export function markdownBroke(error: unknown) {
  console.warn(`markdown: an answer could not be drawn with its libraries (${String(error)}): the answers are plain text`);
  markdownCame = undefined;
}
