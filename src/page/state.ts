// src/page/state.ts (T355): the one object the parts of the page's script share.
import { MODELS } from "../models.js";
import { fromUrl, requested } from "./address.ts";
import { recalled } from "./remembered.ts";

// What the page is doing now: what more than one part of this script reads or sets. A field is no variable, and a
// wrong name reads undefined without a word: tests/page-modules-check.mjs holds every page.<name> to this list.
export const page = {
  model: fromUrl ?? MODELS.find(({ id }) => id === requested) ?? recalled ?? MODELS[0],
  ready: false,
  answer: null as HTMLElement | null, // the bubble that is being written
  used: {} as Record<string, any>, // the generation settings of that run, for the line under it
  // the format of one turn the conversion read from the model's own chat_template, when there was one
  fromTemplate: undefined as string | undefined,
  benchmarking: false,  // the rounds are running: the run button stays off (the worker is busy loading)
  generating: false,
  longest: 256, // the context of the model, known once it is ready
  // the status line of a ready model: how it runs, and (T148) what the GPU takes of its prompts now
  readyLine: undefined as { backend: string; threads: number; pyodide: string } | undefined,
  gpuNote: undefined as string | undefined,
  // The model can be changed while one is loading: the worker cancels that download and starts the new one.
  // loads numbers them, and the worker's reports carry the number.
  loads: 1,
};
