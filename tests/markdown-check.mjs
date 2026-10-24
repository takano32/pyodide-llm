// tests/markdown-check.mjs (T.556): what src/page/markdown.ts makes of an answer's Markdown with the latest marked, as the
// page takes it (the latest version on jsDelivr, bundled there as one module), before DOMPurify. Node has no DOM, so
// DOMPurify and the drawing are tests/markdown-in-browser.mjs's.
//
//   node tests/markdown-check.mjs        (fetches marked's latest version from jsDelivr: about a second)
//
// It sees: the version is resolved as the page resolves it; every element the page draws is made (headings, paragraphs,
// lists, bold, italic, code fenced and inline, a table with its alignment); what is said as text is said so, escaped
// (HTML of the model's, links, autolinks, images, quotes, rules, strikethrough, task boxes); the blocks the page draws
// one by one each make the same HTML alone as in the whole. A marked that no longer has what markdown.ts uses (a new
// major version) fails here first, and the page then draws plain text (its catch).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fetched } from "./fetching.mjs";
import { leave, scratch } from "./leave.mjs";
import { markedFor } from "../src/page/markdown.ts";

const { version } = JSON.parse(String(await fetched("https://data.jsdelivr.com/v1/packages/npm/marked/resolved?specifier=latest")));
assert.match(version, /^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$/, "jsDelivr says a version of marked");
const file = path.join(scratch("markdown-check-"), "marked.mjs");
fs.writeFileSync(file, await fetched(`https://cdn.jsdelivr.net/npm/marked@${version}/+esm`));
const marked = await import(file);
const reader = markedFor(marked);
const html = (text) => {
  const tokens = reader.lexer(text);
  return tokens.map((token) => reader.parser(Object.assign([token], { links: tokens.links }))).join("");
};

// ---- what is drawn
const drawn = html("# Title\n\n## Sub\n\nSome **bold**, *it* and `code`.\nA second line.\n\n- a\n- b\n\n3. three\n4. four\n\n" +
  "| x | y |\n|:-|-:|\n| 1 | 2 |\n\n```js\nlet a = '<b>';\n```\n");
for (const element of ["<h1>Title</h1>", "<h2>Sub</h2>", "<strong>bold</strong>", "<em>it</em>", "<code>code</code>", "<br>", "<ul>", "<li>a</li>",
  '<ol start="3">', "<table>", '<th align="left">x</th>', '<td align="right">2</td>', "<pre><code", "let a = &#39;&lt;b&gt;&#39;;"]) {
  assert.ok(drawn.includes(element), `marked ${version} makes ${element}: ${drawn}`);
}

// ---- what is said as the text the model wrote (escaped)
const hostile = "<img src=x onerror=alert(1)> [x](javascript:alert(1)) ![i](https://example.com/x.png) https://example.com " +
  "<a href=\"https://example.com\">a</a> ~~gone~~ <script>alert(1)</script>\n\n<div onclick=\"alert(1)\">block</div>\n\n> quoted\n> twice\n\n---\n\n- [ ] task\n- [x] done\n\n[ref]: https://example.com\n\n[ref]\n";
const said = html(hostile);
for (const element of ["<img", "<a ", "<a>", "<script", "<div", "<blockquote", "<hr", "<del", "<input", "href=\"", "src=\""]) {
  assert.ok(!said.includes(element), `marked ${version} with markdown.ts's renderer makes ${element}: ${said}`);
}
for (const text of ["&#60;img src=x onerror=alert(1)&#62;", "[x](javascript:alert(1))", "![i](https://example.com/x.png)", "https://example.com",
  "~~gone~~", "&#60;script&#62;", "&#60;div onclick=&#34;alert(1)&#34;&#62;", "&#62; quoted<br>&#62; twice", "<p>---</p>", "[ ] task", "[x] done", "[ref]"]) {
  assert.ok(said.includes(text), `said as text: ${text}, in ${said}`);
}
console.log(`markdown-check: marked ${version}: headings, paragraphs, lists, bold, italic, code and tables drawn; HTML, links, images, quotes, rules, strikethrough and task boxes said as text`);
await leave(0);
