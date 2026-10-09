// worker/told.js (T350): a caught value in words, for whatever of the worker reports an error.
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

// T242: a caught value in words. An Error reads as it always did ("TypeError: Load failed"). What is thrown is not always
// one: Emscripten's ExitStatus (Pyodide's runtime ending) is an object with a name and a message, and String() of it, or
// of any plain object, is "[object Object]", which is what /benchmark/ showed a visitor. Such a value is told by its name
// and message, else by what kind of thing it is and its own fields
export const isError = (err) => err instanceof Error || Object.prototype.toString.call(err) === "[object Error]";
export function told(err) {
  if (isError(err) || typeof err !== "object" || err === null) {
    return String(err);
  }
  const text = (value) => (typeof value === "string" && value ? value : undefined);
  const name = text(err.name) ?? text(err.constructor?.name === "Object" ? undefined : err.constructor?.name);
  if (text(err.message)) {
    return name ? `${name}: ${err.message}` : err.message;
  }
  let fields;
  try {
    fields = JSON.stringify(err);
  } catch {
    fields = undefined;  // it refers to itself
  }
  return `${name ?? "something that is not an error"} was thrown${fields && fields !== "{}" ? `: ${fields.slice(0, 300)}` : ""}`;
}
