// the listed models and where each one's config.json comes from (the converter's header comes from it)
import fs from "node:fs";
import { MODELS } from "../../src/models.js";
const out = MODELS.map((m) => {
  let repo, rev = "main", kind;
  if (m.hf?.vocabulary) { repo = m.hf.vocabulary.repo; rev = m.hf.vocabulary.revision; kind = "vocabulary"; }
  else if (m.hf?.config) { repo = m.hf.repo; rev = m.hf.revision; kind = "own"; }
  else if (m.hf) { repo = m.original; kind = "original at main"; }
  else { repo = m.source; kind = "site"; }
  return { id: m.id, name: m.name, group: m.group ?? "site", repo, rev, kind, bytes: m.bytes ?? null, download: m.download ?? null,
    dtype: m.options?.dtype ?? m.conversion?.dtype ?? null, note: m.note };
});
fs.writeFileSync(new URL("list.json", import.meta.url), JSON.stringify(out, null, 1));
console.log(out.length, "models");
for (const o of out) console.log(o.id.padEnd(40), o.kind.padEnd(18), o.repo, o.rev.slice(0, 8));
