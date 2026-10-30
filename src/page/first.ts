// src/page/first.ts (T.474, out of src/pages/index.astro's script, where tsc did not read it): the first load of the
// page, which index.astro names last, after every other part has run. A model chosen while this looks (the one kept
// gone, the repository of ?hf= looked at) has started the worker itself, and wins: the review of T.116.
import { DEFAULT_MODEL } from "../models.js";
import { weighed } from "./address.ts";
import { gpuFor, recalled, remember, threadsFor } from "./remembered.ts";
import { loads, model, opensWith } from "./model.ts";
import { stillKept } from "./list.ts";
import { show } from "./settings.ts";
import { root, worker } from "./receiver.ts";
import { urlLooked } from "./choose.ts";
import type { ToWorker } from "../runtime/contracts/messages.ts";

(async () => {
  if (model === recalled && !(await stillKept(model)) && opensWith(DEFAULT_MODEL)) {
    remember(undefined);
    show(model);
  }
  // (T.505.3: a repository of ?hf= is looked at for the file that is its model, where the URL names none)
  if (await urlLooked() && loads === 1) {
    worker.postMessage({ type: "init", root, search: location.search, model: weighed(model), load: loads, threads: threadsFor(model), gpu: gpuFor(model) } satisfies ToWorker);
  }
})();
