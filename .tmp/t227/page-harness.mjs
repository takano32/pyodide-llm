// The benchmark page's own script (page-under-test.ts, cut out of src/pages/benchmark.astro by make-page.mjs) run in
// Node with stubs for the DOM, the storage and the workers; the workers answer from a scenario. The page's report(),
// warnings(), shortReport() and link, end to end, for failures no CI runner makes (a lost device, a failing step,
// logits that are not finite, a watchdog) and WRONG verdicts (the real rows of T225's probe).
import * as F from "./fixtures.mjs";

class El {
  constructor(name = "") {
    Object.assign(this, { name, children: [], hidden: false, textContent: "", innerHTML: "", value: "", checked: false, dataset: {}, style: {}, options: [], href: "",
      disabled: false, className: "", attributes: {}, classList: { toggle() {}, add() {}, remove() {} }, onclick: null });
    this.queries = new Map();
  }
  append(...c) { this.children.push(...c); }
  prepend(...c) { this.children.unshift(...c); }
  querySelector(sel) {
    if (!this.queries.has(sel)) this.queries.set(sel, new El(sel));
    return this.queries.get(sel);
  }
  querySelectorAll() { return []; }
  setAttribute(k, v) { this.attributes[k] = v; }
  removeAttribute(k) { delete this.attributes[k]; }
  addEventListener() {}
  closest() { return new El("closest"); }
  scrollIntoView() {}
  add() {}
}
const ids = new Map();
const byId = (id) => {
  if (!ids.has(id)) ids.set(id, new El(id));
  return ids.get(id);
};
const NAMES = ["device", "cpu", "model", "gpu", "storage", "line", "memory"];
const TITLES = { device: "This browser", cpu: "CPU", model: "Model", gpu: "GPU", storage: "Storage", line: "Line", memory: "Page memory" };
const sections = NAMES.map((name) => {
  const section = new El(`section ${name}`);
  section.dataset.section = name;
  section.querySelector("h2").textContent = TITLES[name];
  return section;
});
const store = () => {
  const map = new Map();
  return { getItem: (k) => map.get(k) ?? null, setItem: (k, v) => map.set(k, String(v)), removeItem: (k) => map.delete(k) };
};

// the scenario's answers, by worker file and the message's step or type
export const scenario = { model: undefined, workers: [] };

class FakeWorker {
  constructor(url) {
    this.url = String(url);
    this.listeners = [];
    scenario.workers.push(this);
  }
  addEventListener(type, fn) { if (type === "message") this.listeners.push(fn); }
  terminate() { this.ended = true; }
  emit(data) {
    for (const fn of this.listeners) fn({ data });
    this.onmessage?.({ data });
  }
  postMessage(message) {
    setTimeout(async () => {
      try {
        await scenario.answer(this, message);
      } catch (error) {
        this.emit({ error: `the fake worker threw: ${error}` });
      }
    }, 0);
  }
}

export function install(extra = {}) {
  globalThis.window = globalThis;
  globalThis.self = Object.assign(globalThis, { crossOriginIsolated: true });
  globalThis.__BUILD__ = "review";
  globalThis.document = { getElementById: byId, querySelectorAll: (sel) => (sel === "section[data-section]" ? sections : []),
    createElement: (tag) => new El(tag), querySelector: (sel) => { const m = /data-section="(\w+)"/.exec(sel); return m ? sections[NAMES.indexOf(m[1])].querySelector("h2") : byId(sel); },
    addEventListener() {}, removeEventListener() {}, visibilityState: "visible", body: new El("body") };
  globalThis.location = { search: "", origin: "http://localhost", pathname: "/pyodide-llm/benchmark/" };
  globalThis.localStorage = store();
  globalThis.sessionStorage = store();
  globalThis.Worker = FakeWorker;
  globalThis.getSelection = () => ({ removeAllRanges() {}, selectAllChildren() {} });
  globalThis.addEventListener = () => {};
  globalThis.removeEventListener = () => {};
  globalThis.Option = class { constructor(text, value) { this.text = text; this.value = value; } };
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { hardwareConcurrency: 16, deviceMemory: 8,
    userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36", ...extra } });
  byId("model").value = "llm-jp-3-150m";
  byId("model").options = [{ value: "llm-jp-3-150m" }];
  byId("size").value = "64";
  byId("size").options = [{ value: "64" }];
}

let loads = 0;
export async function load() {
  ids.clear();
  scenario.workers.length = 0;
  byId("model").value = "llm-jp-3-150m";
  byId("model").options = [{ value: "llm-jp-3-150m" }];
  byId("size").value = "64";
  byId("size").options = [{ value: "64" }];
  await import(`./page-under-test.ts?n=${++loads}`);
  return globalThis.__page;
}
export { byId, F };
