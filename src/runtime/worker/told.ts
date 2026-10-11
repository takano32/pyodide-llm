// worker/told.js (T350): a caught value in words, for whatever of the worker reports an error.
// A module of worker.js; it takes its neighbours' names by static imports, and a bundler links them (T367.3).

// T242: a caught value in words. An Error reads as it always did ("TypeError: Load failed"). What is thrown is not always
// one: Emscripten's ExitStatus (Pyodide's runtime ending) is an object with a name and a message, and String() of it, or
// of any plain object, is "[object Object]", which is what /benchmark/ showed a visitor. Such a value is told by its name
// and message, else by what kind of thing it is and its own fields
// T368.3: what the worker reads of a caught value, whatever it is: an Error's own, and what its makers put on one
// (type: a PythonError's, the name of Python's exception; status: ranges.ts's refused(), the server's answer; final:
// sources.ts, no other try cures it; pyodide: pyodide.ts, the runtime did not load)
export type Thrown = { name?: string; message?: string; stack?: string; constructor?: { name?: string }; type?: string; status?: number; final?: boolean; pyodide?: boolean };
export const isError = (err: unknown): err is Error => err instanceof Error || Object.prototype.toString.call(err) === "[object Error]";
export function told(err: unknown) {
  if (isError(err) || typeof err !== "object" || err === null) {
    return String(err);
  }
  const text = (value: unknown) => (typeof value === "string" && value ? value : undefined);
  const name = text((err as Thrown).name) ?? text((err as Thrown).constructor?.name === "Object" ? undefined : (err as Thrown).constructor?.name);
  if (text((err as Thrown).message)) {
    return name ? `${name}: ${(err as Thrown).message}` : (err as Thrown).message!;
  }
  let fields;
  try {
    fields = JSON.stringify(err);
  } catch {
    fields = undefined;  // it refers to itself
  }
  return `${name ?? "something that is not an error"} was thrown${fields && fields !== "{}" ? `: ${fields.slice(0, 300)}` : ""}`;
}
