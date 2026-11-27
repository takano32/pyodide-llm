// src/page/opening.ts (T378.5): the model this visit opens with. What the URL names comes first (a model of no list,
// then one of the list, T669: or one the list had under another id), then the one chosen on an earlier visit, then
// the default (T511.6). It runs where
// index.astro imports it: before settings.ts, whose top level shows the model.
import { DEFAULT_MODEL, entryOf } from "../models.ts";
import { fromUrl, requested } from "./address.ts";
import { recalled } from "./remembered.ts";
import { opensWith } from "./model.ts";

opensWith(fromUrl ?? entryOf(requested) ?? recalled ?? DEFAULT_MODEL);
