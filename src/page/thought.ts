// src/page/thought.ts (T520): an answer's thought, as the page folds it above the answer: where it ends, and the one
// line its fold says. It imports nothing and reads nothing of a page: Node imports this file as it is
// (tests/page-parts-check.mjs).

const THOUGHT_OPENS = "<think>", THOUGHT_CLOSES = "</think>";
/** T520: what was written, as the thought (null: none; its marks and the white space around it left out), whether the
 * thought has closed, and the answer after it. `opened`: the format left the answer in <think> (else the model may open
 * the thought itself, as Qwen3's thinking form does). The form is naidan's splitAssistantThinking() of
 * src/logic/assistant-thinking.ts (852bdeab, MIT, Ryo Ota), where a format that ends in the opening mark has it put
 * before what was written: no line of it is copied. */
export function thoughtOf(text: string, opened: boolean) {
  const begun = text.trimStart();
  if (!opened && !begun.startsWith(THOUGHT_OPENS)) return { thought: null, closed: false, answer: text };
  const body = begun.startsWith(THOUGHT_OPENS) ? begun.slice(THOUGHT_OPENS.length) : begun;
  const end = body.indexOf(THOUGHT_CLOSES);
  if (end < 0) return { thought: body.trimStart(), closed: false, answer: "" };
  // (a thought its budget closed may be closed by the model once more, right after: T520's Qwen3.5 0.8B)
  const answer = body.slice(end + THOUGHT_CLOSES.length).trimStart();
  return { thought: body.slice(0, end).trim(), closed: true, answer: answer.startsWith(THOUGHT_CLOSES) ? answer.slice(THOUGHT_CLOSES.length).trimStart() : answer };
}
/** T520: the one line a folded thought says: while it goes on, once it closed, and at the answer's end (stats: the
 * engine's count of its tokens and whether its budget closed it; thought null where the engine saw no marks). */
export function thoughtLine(closed: boolean, stats?: { thought: number | null; thought_cut: boolean }) {
  if (!stats || stats.thought === null) return closed ? "Thought" : "Thinking…";
  const tokens = `${stats.thought} token${stats.thought === 1 ? "" : "s"}`;
  return stats.thought_cut ? `Thought for ${tokens}, up to its limit` : closed ? `Thought for ${tokens}` : `Thought for ${tokens}, not finished`;
}
