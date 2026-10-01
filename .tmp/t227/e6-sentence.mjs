import { warnings, layerStepsTable } from "../../src/bench.js";
import * as F from "./fixtures.mjs";
const { stepsStep, layerRight, layerCeilings } = F;
const unsteadyForm = { ...stepsStep, result: { ...stepsStep.result, forms: stepsStep.result.forms.map((f, i) => (i ? { ...f, unsteady: true } : f)) } };
const lines = layerStepsTable(unsteadyForm, layerRight, layerCeilings);
console.log(lines.filter((l) => l.includes("(unsteady)")).map((l) => l.length));
console.log(warnings([{ title: "GPU", status: "ok", markdown: lines.join("\n") }]));
