// tests/fakes/pyodide.mjs (T357.5; out of tests/worker-conduct-check.mjs, where T374.2.1 wrote it)
// A stand-in for what Pyodide hands JavaScript of Python's objects: proxies, which live until they are destroyed, and
// the face it gives a Python generator. For the checks of what answers a conversion's conduct (a generator of
// src/python/convert/conduct.py) without Pyodide: the worker's loop (worker-conduct-check.mjs) and the tools'
// (conducting-check.mjs).
//
//   const { proxy, readable, generator, proxies, allDestroyed } = pyProxies()
//
//   proxy(name, fields)        a proxy: fields as they are written (a getter stays one, and runs when it is read), with
//                              destroyed (how often) and destroy(). Every proxy made is in proxies
//   readable(name, value)      a proxy with toJs(options): value, or value(options); it refuses to be read once destroyed
//   generator(running, more)   the face of a Python generator over a JavaScript one (running: the conduct, played): a
//                              proxy with next(answer) -> { done, value } and return(). Each value yielded (an array:
//                              a request) comes as a readable proxy that is to be read one level deep ({ depth: 1 })
//                              and destroyed; next() and return() refuse a generator that was destroyed. It writes
//                              down answers (every answer sent, the first next()'s undefined too), returned (how often
//                              return() was called) and afterReturn (the next()s after a return()). more.closing():
//                              called as return() closes it; what it throws, return() throws (a finally of the
//                              conduct's that fails)
//   allDestroyed()             that every proxy made was destroyed exactly once; how many there are
//
// What it does as Pyodide does, because the code under test leans on it: a generator's proxy is a JavaScript iterator
// (next(answer) is Python's send(answer); there is no send), a step's value is a proxy of its own every time, and a
// proxy that is not destroyed is a leak nobody is told of (so the stand-in counts, and reading one that was destroyed
// fails here, as a destroyed PyProxy throws).
// Where it is simpler: nothing is converted (a request is handed over as the array it is, whatever the depth asked:
// only that { depth: 1 } was asked is seen; an undefined answer stays undefined, where Pyodide makes it None, and a
// null stays null, where Pyodide makes it jsnull: that an answerer never answers null is the checks' own to see);
// there is no Python, so no exception of Python's and no finally but closing(); destroying a proxy twice only counts.
import assert from "node:assert/strict";

export function pyProxies() {
  const proxies = [];
  const proxy = (name, fields = {}) => {
    const made = Object.defineProperties({ name, destroyed: 0, destroy() { made.destroyed++; } }, Object.getOwnPropertyDescriptors(fields));
    proxies.push(made);
    return made;
  };
  const readable = (name, value) => proxy(name, { toJs(options) {
    assert.equal(this.destroyed, 0, `${name} was read after it was destroyed`);
    return typeof value === "function" ? value(options) : value;
  } });
  const generator = (running, { closing } = {}) => {
    const steps = proxy("the generator", { returned: 0, afterReturn: 0, answers: [],
      next(answer) {
        assert.equal(steps.destroyed, 0, "next() of a generator that was destroyed");
        if (steps.returned) steps.afterReturn++;
        steps.answers.push(answer);
        const step = running.next(answer);
        return step.done ? step : { done: false, value: readable(`the request ${step.value.filter((part) => typeof part !== "object").join(" ")}`, (options) => {
          assert.equal(JSON.stringify(options), "{\"depth\":1}", "a request is read one level deep: the conversion of the last one stays a proxy");
          return step.value;
        }) };
      },
      return() {
        assert.equal(steps.destroyed, 0, "return() of a generator that was destroyed");
        steps.returned++;
        running.return();
        closing?.();
        return { done: true, value: undefined };
      } });
    return steps;
  };
  const allDestroyed = () => {
    for (const made of proxies) assert.equal(made.destroyed, 1, `${made.name} was destroyed ${made.destroyed} times`);
    return proxies.length;
  };
  return { proxies, proxy, readable, generator, allDestroyed };
}
