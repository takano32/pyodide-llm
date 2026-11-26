// T309: what the page and the visitors' documents say of what opens, held to the tables it is made of.
//
//   (a) the model list's last entry (src/page/repository.ts's OTHER_REPOSITORY) names the families of the converter's
//       table (convert/families' FAMILIES, whose titles python_facts.py writes into src/runtime/built/facts/families.ts),
//       each once and in its order, and nothing else; one more family in the table is one more in the entry;
//   (b) README's ?hf= says the same families, as the converter's refusals say them ("A, B and C": convert/families'
//       named()). README is a document nobody builds, so it is not made from the table: it is held to it here, and a
//       family added to the converter fails this until README names it;
//   (c) the numbers README and docs/architecture.md say of the list's entries of Hugging Face (src/models.ts's MODELS,
//       the group "hf") and of those fetched as a GGUF (hf.weights a .gguf; as Q8_0 where its name says so).
//
//   node tests/docs-check.ts
import assert from "node:assert/strict";
import fs from "node:fs";
import { MODELS } from "../src/models.ts";
import { OTHER_REPOSITORY, anyOf } from "../src/page/repository.ts";
import { PYTHON_FAMILIES } from "../src/runtime/built/facts/families.ts";

const text = (name: string) => fs.readFileSync(new URL(`../${name}`, import.meta.url), "utf8").replace(/\s+/g, " ");
const families: readonly string[] = PYTHON_FAMILIES;

// (a) the entry, by the one function that lists them
assert.equal(anyOf([]), "");
assert.equal(anyOf(["A"]), "A");
assert.equal(anyOf(["A", "B"]), "A or B");
assert.equal(anyOf(["A", "B", "C"]), "A, B or C");
const [head, listed] = OTHER_REPOSITORY.split(" — any ");
assert.equal(head, "Other repository…");
const named = listed.replace(" or ", ", ").split(", ");
assert.deepEqual(named, [...families], `the model list's last entry names ${named.join(", ")}, the converter's table ${families.join(", ")}`);
assert.ok(families.length > 1 && new Set(families).size === families.length, `the families' titles are many and each once: ${families}`);
const more = anyOf([...families, "Made-up"]);
assert.ok(more !== anyOf(families) && more.endsWith(`${families.at(-1)} or Made-up`), `one more family is not said: ${more}`);

// (b) README's ?hf=
const readme = text("README.md");
const said = `${families.slice(0, -1).join(", ")} and ${families.at(-1)}`;
assert.ok(readme.includes(`What it reads: the families ${said} (`),
  `README's ?hf= is to name the converter's families as "What it reads: the families ${said} (…": python_facts.py's families`);

// (c) the numbers of the entries
const hf = MODELS.filter((entry) => entry.group === "hf");
const weights = (entry: (typeof hf)[number]) => ("hf" in entry && entry.hf && "weights" in entry.hf ? String(entry.hf.weights) : "");
const gguf = hf.filter((entry) => /\.gguf$/i.test(weights(entry)));
const q8 = gguf.filter((entry) => /Q8_0/i.test(weights(entry)));
for (const [where, line] of [
  ["README.md", `(${hf.length} entries, from`],
  ["README.md", `(for ${gguf.length} of them a GGUF: Q8_0,`],
  ["docs/architecture.md", `${gguf.length} of the ${hf.length} Hugging Face entries in the list are fetched as a GGUF (${q8.length} as Q8_0,`],
]) {
  assert.ok(text(where).includes(line), `${where} is to say "${line}" (the list's entries of Hugging Face, those of a GGUF, of Q8_0)`);
}

console.log(`docs-check: the model list's last entry and README name the converter's ${families.length} families; ` +
  `README and docs/architecture.md say ${hf.length} entries of Hugging Face, ${gguf.length} of a GGUF, ${q8.length} of Q8_0`);
