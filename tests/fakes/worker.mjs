// tests/fakes/worker.mjs (T357.5)
// A stand-in for a Worker as the code that made it sees it: its face (postMessage, onmessage and onerror or
// addEventListener, terminate), what it was posted written down, and the check speaking for it. For the checks of
// what talks to the GPU's worker (forward/gpuside.ts through createForward()'s `gpu`, a factory: gpu-part-check.mjs,
// gpu-alone-check.mjs, memory-check.mjs, thread-search-check.mjs, gpu-hybrid-check.mjs) and of what makes workers with
// `new Worker(url, options)` (worker/weights.ts: worker-sink-check.mjs, worker-outdated-check.mjs).
//
//   const { Worker, factory, made, posted } = workers({ answer })
//
//   Worker       the class, for a context's global: new Worker(url, options)
//   factory      () => a new one, for createForward({ gpu: factory })
//   made         every worker made, in order; posted: everything posted to any of them, in order
//   answer       answer(data, worker): what the worker does with a message as it is posted, on this very thread
//                (by default nothing: a worker that never answers). The answer is the check's own: what a GPU's worker
//                writes into the control area for a request is what such a check is about
//
//   a worker:    url, options (as it was made), posted (what it was posted), terminated (true once terminate() was
//                called), say(data) (it posts that to its maker: a message event to onmessage and the listeners),
//                fail(event) (the browser's error event of it: to onerror and the listeners; { message } where its
//                script threw, {} where the script did not arrive)
//
//   ended        an answer for a GPU's worker that says { type: "ended" } a turn after it is stopped, as gpu.ts's
//                end() does, and nothing else; endedThen(answer): that, and answer for every other message
//   READY        what a GPU's worker says once it is ready, with made-up values: READY({ blocks, tokens, ... })
//
// Where it is simpler than a Worker: there is no other thread. An answer runs inside postMessage(), before the maker
// goes on, so a maker that waits (Atomics.wait on a word of the control area) finds its answer there already and
// never sleeps; nothing is ever in flight, and no two things happen at once. Messages are handed over as they are,
// not cloned (an object posted and changed afterwards is changed in `posted` too). say() and fail() with nobody
// listening do nothing; a handler and the listeners hear in the order they were first set. removeEventListener and the options of addEventListener are not there. A worker that sleeps
// for as long as its work would take is tests/gpu-default-check.mjs's own (a real thread).

/** The GPU's worker says it ended, a turn after it is stopped. */
export const ended = (data, worker) => {
  if (data.type === "stop") queueMicrotask(() => worker.say({ type: "ended" }));
};
/** That, and answer(data, worker) for every other message. */
export const endedThen = (answer) => (data, worker) => (data.type === "stop" ? ended(data, worker) : answer(data, worker));
/** What a GPU's worker says when it is ready (gpu/start.ts), made up; more: what a check sets of it. */
export const READY = (more = {}) => ({ type: "ready", adapter: "a stand-in", key: "k", bytes: 1e6, seconds: 0.1, form: "tiles", attention: "flash",
  forms: [], remembered: false, blocks: [], ...more });

export function workers({ answer = () => {} } = {}) {
  const made = [], posted = [];
  class Worker {
    #handlers = {};
    constructor(url, options) {
      Object.assign(this, { url: url === undefined ? undefined : String(url), options, posted: [], terminated: false, listeners: [] });
      made.push(this);
    }
    // onmessage and onerror: a handler set this way is heard in its turn among the listeners, the turn of the first
    // time it was set (as an event handler attribute is)
    #handler(type, handler) {
      if (!(type in this.#handlers)) this.listeners.push([type, (event) => this.#handlers[type]?.(event)]);
      this.#handlers[type] = handler;
    }
    get onmessage() { return this.#handlers.message ?? null; }
    set onmessage(handler) { this.#handler("message", handler); }
    get onerror() { return this.#handlers.error ?? null; }
    set onerror(handler) { this.#handler("error", handler); }
    addEventListener(type, listener) {
      this.listeners.push([type, listener]);
    }
    postMessage(data) {
      this.posted.push(data);
      posted.push(data);
      answer(data, this);
    }
    terminate() {
      this.terminated = true;
    }
    #tell(type, event) {
      for (const [heard, listener] of [...this.listeners]) if (heard === type) listener(event);
    }
    say(data) {
      this.#tell("message", { data });
    }
    fail(event = {}) {
      this.#tell("error", event);
    }
  }
  return { Worker, factory: () => new Worker(), made, posted };
}
