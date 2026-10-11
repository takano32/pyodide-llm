// built.js (T367.3): the files of this site that are built and not written (`make kernels`: kernels/build.py; `make
// python`: python_archive.py), in src/runtime/built/, each by its own literal name. That is how a bundler is told of a
// file it is to serve: Vite gives each a name with a hash of its bytes (_astro/simdkernel_plain.<hash>.wasm), so a file
// that did not change keeps its name from one deployment to the next, and a page never reads one of another build.
// One by one and never by a name put together (`./built/${name}`): a URL with a variable in it makes the bundler copy
// the whole folder, and files of less than 4 KB went into the worker's text as data: URLs (T365).
export const BUILT = {
  "simdkernel.so": new URL("./built/simdkernel.so", import.meta.url),
  "simdkernel_relaxed.wasmlib": new URL("./built/simdkernel_relaxed.wasmlib", import.meta.url),
  "simdkernel_plain.wasm": new URL("./built/simdkernel_plain.wasm", import.meta.url),
  "simdkernel_relaxed_plain.wasm": new URL("./built/simdkernel_relaxed_plain.wasm", import.meta.url),
  "simdkernel_shared.wasm": new URL("./built/simdkernel_shared.wasm", import.meta.url),
  "simdkernel_relaxed_shared.wasm": new URL("./built/simdkernel_relaxed_shared.wasm", import.meta.url),
  "simdkernel_plain64.wasm": new URL("./built/simdkernel_plain64.wasm", import.meta.url),
  "simdkernel_relaxed_plain64.wasm": new URL("./built/simdkernel_relaxed_plain64.wasm", import.meta.url),
  "simdkernel_shared64.wasm": new URL("./built/simdkernel_shared64.wasm", import.meta.url),
  "simdkernel_relaxed_shared64.wasm": new URL("./built/simdkernel_relaxed_shared64.wasm", import.meta.url),
  "ceilings_plain.wasm": new URL("./built/ceilings_plain.wasm", import.meta.url),
  "ceilings_relaxed_plain.wasm": new URL("./built/ceilings_relaxed_plain.wasm", import.meta.url),
  "ceilings_shared.wasm": new URL("./built/ceilings_shared.wasm", import.meta.url),
  "ceilings_relaxed_shared.wasm": new URL("./built/ceilings_relaxed_shared.wasm", import.meta.url),
  "engine.zip": new URL("./built/engine.zip", import.meta.url),
  "converter.zip": new URL("./built/converter.zip", import.meta.url),
};
