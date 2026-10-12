// failure.ts (T380.1): the one way a failure is carried and told apart in what a browser runs here.
//
// Three things are thrown in this site's JavaScript, and each is read in one way:
// - an Error with a sentence (new Error("…")): a message for the reader, as Python's ValueError is one. Its name is
//   "Error", and it is said by its sentence alone. Nothing more is to be known of it, so nothing more is put on it.
// - a Failure (below): the same, and one thing more that somebody reads: its kind, whether another try may cure it,
//   the status a server answered with. Never a property put on an Error afterwards, and never the wording.
// - whatever else arrives (the browser's TypeError of a line that dropped, Pyodide's PythonError, its JsException
//   around an error of this site's own that crossed Python, Emscripten's ExitStatus, which is no Error at all):
//   failureOf() reads it, here and nowhere else, into a Failure.
// A load that was cancelled is none of these: whoever catches asks the load's signal, not the error.
//
// Imports nothing: the worker, the GPU's worker, /benchmark/'s workers and the pages each bundle their own copy (and a
// Failure is known by its kind, not by its class: one that another copy of this module made is one). forward.ts and
// its parts do not read this file: a module that the worker's entry and the chunk it fetches later both import is
// put into the entry by the bundler, and the chunk then imports the worker's own file (measured, T380.1).

// The kinds, each made somewhere and read somewhere else (a kind nobody reads is not one):
//   refused    a sentence for the reader and nothing else to tell: a server's answer, a file that is not a model, a
//              write the memory would not take. What failureOf() calls whatever it has no other kind for.
//   memory     the memory ran out (T90): the page says so in its own words, with how much it had (worker.ts → the page).
//   runtime    Pyodide did not load (worker/pyodide.ts): the page may try once more without the service worker.
//   updated    a file of this page's build is gone from the site (updated.ts): said by its one sentence, wherever caught.
//   late       a step of putting a model on the GPU took too long (gpu/device.ts): not a shader that is wrong, which
//              the next one is tried after, but the end of the whole attempt.
//   cut-short  a loop of /benchmark/'s ceilings did not do its work (benchmark/sections.ts): its section is wrong.
const KINDS = ["refused", "memory", "runtime", "updated", "late", "cut-short"] as const;
export type Kind = (typeof KINDS)[number];

// What a Failure says besides its kind and sentence. again: false where another try is answered the same (a status of
// the 400s, a write that was refused); true or unsaid where one may help. status: a server's answer. cause: what was
// caught, where it was made of something caught. stack: where it happened, where that is not where this was made.
type More = { again?: boolean; status?: number; cause?: unknown; stack?: string };

// What is read of a caught value, whatever it is: an Error's own, of Pyodide's PythonError its type (the name of
// Python's exception), and a Failure's kind
export type Thrown = { name?: string; message?: string; stack?: string; constructor?: { name?: string }; type?: string; kind?: unknown };

/** A failure of a kind. An Error by name ("Error"), so that whatever says an error by its message says the sentence alone. */
export class Failure extends Error {
  readonly kind: Kind;
  readonly again: boolean | undefined;
  readonly status: number | undefined;
  constructor(kind: Kind, message: string, { again, status, cause, stack }: More = {}) {
    super(message, cause === undefined ? undefined : { cause });
    this.kind = kind;
    this.again = again;
    this.status = status;
    if (stack !== undefined) this.stack = stack;
  }
}

/** The kind of a caught value, or undefined where it is no Failure (what failureOf() would read of it is not asked). */
export function kindOf(caught: unknown): Kind | undefined {
  const kind = isError(caught) ? (caught as Thrown).kind : undefined;
  return KINDS.find((one) => one === kind);
}

// T242: a caught value in words. An Error reads as it always did ("TypeError: Load failed"). What is thrown is not always
// one: Emscripten's ExitStatus (Pyodide's runtime ending) is an object with a name and a message, and String() of it, or
// of any plain object, is "[object Object]", which is what /benchmark/ showed a visitor. Such a value is told by its name
// and message, else by what kind of thing it is and its own fields
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

// A JavaScript error thrown inside a call from Python comes back as a PythonError of type JsException, its own name
// and message on the last line: the memory of the weights, which the converter opens through the sink (Safari's
// refusal of a model past 4 GB, a memory the browser will not make or grow), and forward.ts growing the keys and
// values. It is told as itself (the review of T133: it was a traceback)
const JS_EXCEPTION = /^pyodide\.ffi\.JsException: (\w+): (.*)$/;
// (T380.2, exported: the one other reader of a last line, a tokenizer that does not fit its checkpoint, worker/load.ts,
// says Python's own last line with the exception's name, which failureOf() does not)
export const lastLine = (text: unknown) => String(text).trim().split("\n").pop()!;

/** A caught value as a Failure: itself where it is one, else read here, the one place that reads what the others throw.
 * Its sentence is what a visitor is told:
 * - a ValueError of the engine or the converter is a message for the reader (wrong file, prompt too long): its last
 *   line, no traceback;
 * - an Error by that name (also one of this site's that crossed Python) is its sentence;
 * - T90, the memory ran out, in Python (MemoryError: malloc could not grow the WebAssembly memory) or in JavaScript
 *   (RangeError: an ArrayBuffer or WebAssembly.Memory.grow was refused; Firefox says InternalError: out of memory;
 *   V8 also raises RangeError for a stack overflow, which is not this): kind "memory", and the error's last line;
 * - anything else by its name and words (told(): "TypeError: Load failed", and never "[object Object]").
 * Where it happened (stack) stays the caught value's. */
export function failureOf(caught: unknown): Failure {
  if (kindOf(caught)) return caught as Failure;
  const thrown = caught as Thrown | null | undefined;
  const [, name = thrown?.name, text = thrown?.message] = thrown?.type === "JsException" ? JS_EXCEPTION.exec(lastLine(thrown.message)) ?? [] : [];
  const memory = thrown?.type === "MemoryError" || name === "InternalError" || (name === "RangeError" && !/call stack/i.test(text ?? ""));
  const message = memory ? lastLine(text ?? caught)
    : thrown?.type === "ValueError" ? lastLine(thrown.message).replace(/^ValueError: /, "")
    : name === "Error" ? text ?? "" : told(caught);
  return new Failure(memory ? "memory" : "refused", message, { cause: caught, stack: String(thrown?.stack ?? told(caught)) });
}
