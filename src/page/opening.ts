// src/page/opening.ts (T378.5): the model this visit opens with. What the URL names comes first (a model of no list,
// then one of the list), then the one chosen on an earlier visit, then the first of the list. It runs where
// index.astro imports it: before settings.ts, whose top level shows the model.
import { MODELS } from "../models.js";
import { fromUrl, requested } from "./address.ts";
import { recalled } from "./remembered.ts";
import { opensWith } from "./model.ts";

opensWith(fromUrl ?? MODELS.find(({ id }) => id === requested) ?? recalled ?? MODELS[0]);
