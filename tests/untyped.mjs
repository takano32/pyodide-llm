// tests/untyped.mjs (T.368.1): a file of the runtime as JavaScript, whatever it is written in. The tools that read the
// sources as text (the syntax trees of the *-modules-check.mjs, worker-source.mjs's one script for a vm's context) read
// a .ts file with its types blanked: every type is replaced by spaces, so the text has the same length and the same
// lines, a node's place in one is its place in the other, and a line a vm's error names is the source's line. (Node's
// own way of running a .ts file, node:module's stripTypeScriptTypes(): what it cannot blank, an enum, a namespace, a
// parameter property, it refuses, and so does tsconfig.base.json's erasableSyntaxOnly.) T.584: where the runtime has no
// such function (Bun), amaro's, the package Node's is made of, which blanks the same way.
//
//   untyped(text, name)     text as it is for a .js name; with the types blanked for a .ts name
//   typed(name)             whether the name is a TypeScript source's (not a declaration file's)
import module from "node:module";

// (T.584: Bun runs the tools too, the owner's choice of 2026-10-25; its node:module has no stripTypeScriptTypes)
const amaro = module.stripTypeScriptTypes ? null : await import("amaro");
const blanked = (text) => (amaro ? amaro.transformSync(text, { mode: "strip-only" }).code : module.stripTypeScriptTypes(text, { mode: "strip" }));
export { blanked };

export const typed = (name) => /\.ts$/.test(name) && !/\.d\.ts$/.test(name);

export function untyped(text, name) {
  if (!typed(name)) return text;
  // (stripTypeScriptTypes says once a process that it is experimental: the tools' output is read by people and by tests)
  const emit = process.emitWarning;
  process.emitWarning = (warning, ...rest) => (/stripTypeScriptTypes/.test(String(warning?.message ?? warning)) ? undefined : emit.call(process, warning, ...rest));
  try {
    const out = blanked(text);
    if (out.length !== text.length) throw new Error(`${name}: blanking its types changed its length (${text.length} to ${out.length})`);
    return out;
  } finally {
    process.emitWarning = emit;
  }
}
