// src/page/thought.ts (T.520): an answer's thought, as the page folds it above the answer: where it ends, and the one
// line its fold says. It imports nothing and reads nothing of a page: Node imports this file as it is
// (tests/page-parts-check.ts).

// T.592.1: the marks of every kind of thought: its opening and closing marks, and what a model writes after the closing
// one before its answer. Harmony's (llm-jp-4's thinking forms) is its analysis channel, which the format opens, and
// the answer its final channel's message. The engine follows the same (src/python/engine/generation.py' THOUGHT_MARKS)
const MARKS = [{ opens: "<think>", closes: "</think>", answers: "" },
  { opens: "<|channel|>analysis<|message|>", closes: "<|end|>", answers: "<|start|>assistant<|channel|>final<|message|>" }];
/** T.592.1: whether a format leaves the answer inside a thought (the text sent ends in an opening mark). */
export const opensThought = (sent: string) => MARKS.some(({ opens }) => sent.trimEnd().endsWith(opens));
/** T.520: what was written, as the thought (null: none; its marks and the white space around it left out), whether the
 * thought has closed, and the answer after it. `opened`: the format left the answer in a thought (opensThought(); else
 * the model may open the thought itself, as Qwen3's thinking form does). The form is naidan's splitAssistantThinking() of
 * src/logic/assistant-thinking.ts (852bdeab, MIT, Ryo Ota), where a format that ends in the opening mark has it put
 * before what was written: no line of it is copied. */
export function thoughtOf(text: string, opened: boolean) {
  const begun = text.trimStart();
  const own = MARKS.find(({ opens }) => begun.startsWith(opens));
  if (!opened && !own) return { thought: null, closed: false, answer: text };
  const body = own ? begun.slice(own.opens.length) : begun;
  // (where the format opened it: the kind whose closing mark comes first, a mark not written yet after all the text)
  const at = (closes: string) => { const end = body.indexOf(closes); return end < 0 ? body.length : end; };
  const marks = own ?? MARKS.reduce((first, kind) => at(kind.closes) < at(first.closes) ? kind : first);
  const end = body.indexOf(marks.closes);
  if (end < 0) return { thought: body.trimStart(), closed: false, answer: "" };
  // (a thought its budget closed may be closed by the model once more, right after: T.520's Qwen3.5 0.8B)
  let answer = body.slice(end + marks.closes.length).trimStart();
  if (answer.startsWith(marks.closes)) answer = answer.slice(marks.closes.length).trimStart();
  return { thought: body.slice(0, end).trim(), closed: true, answer: past(answer, marks.answers) };
}
/** T.592.2: the answer after what a model writes before it (`answers`), "" while that is being written, and all of it where
 * it is not there. Each mark and word of it may follow a space (llm-jp-4 8B thinking writes
 * "<|start|> assistant<|channel|> final<|message|> "). */
function past(answer: string, answers: string) {
  let rest = answer;
  for (const part of answers.match(/<\|[^|]*\|>|[^<]+/g) ?? []) {
    rest = rest.trimStart();
    if (!rest.startsWith(part)) return part.startsWith(rest) ? "" : answer;
    rest = rest.slice(part.length);
  }
  return rest.trimStart();
}
/** T.520: the one line a folded thought says: while it goes on, once it closed, and at the answer's end (stats: the
 * engine's count of its tokens and whether its budget closed it; thought null where the engine saw no marks). */
export function thoughtLine(closed: boolean, stats?: { thought: number | null; thought_cut: boolean }) {
  if (!stats || stats.thought === null) return closed ? "Thought" : "Thinking…";
  const tokens = `${stats.thought} token${stats.thought === 1 ? "" : "s"}`;
  return stats.thought_cut ? `Thought for ${tokens}, up to its limit` : closed ? `Thought for ${tokens}` : `Thought for ${tokens}, not finished`;
}
