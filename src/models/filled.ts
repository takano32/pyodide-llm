// A format filled with a prompt and today's date, as the page sends it.

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
  "November", "December"];
const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
/** The directives of strftime the chat templates use (%d %b %Y is Llama 3's), for a date */
function strftime(format: string, date: Date): string {
  const two = (n: number) => String(n).padStart(2, "0");
  const values: Record<string, string> = { d: two(date.getDate()), m: two(date.getMonth() + 1), Y: String(date.getFullYear()), y: two(date.getFullYear() % 100),
    b: MONTHS[date.getMonth()].slice(0, 3), B: MONTHS[date.getMonth()], a: DAYS[date.getDay()].slice(0, 3), A: DAYS[date.getDay()],
    H: two(date.getHours()), M: two(date.getMinutes()), S: two(date.getSeconds()), "%": "%" };
  return format.replace(/%(.)/g, (directive: string, letter: string) => values[letter] ?? directive);
}
// Jinja's trim is Python's str.strip(): the white space of str.isspace() at either end. JavaScript's trim() takes
// another set, U+FEFF too and not U+001C to U+001F nor U+0085 (the review of T.138, T.144: a prompt with one of them at
// an end differed from the real template's). A loop from either end, not a pattern: [...]+$ tries again from every
// white space in the middle, and a prompt of 100,000 spaces between two words held the page 45 seconds (the review of
// T.144). Every one of them is a single UTF-16 unit.
const SPACE = /[\t-\r\x1c-\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/;
function strip(text: string): string {
  let start = 0, end = text.length;
  while (start < end && SPACE.test(text[start])) start++;
  while (end > start && SPACE.test(text[end - 1])) end--;
  return text.slice(start, end);
}
/** What the page sends for a prompt in a model's template: {prompt} is what was typed, {prompt:trim} the same
 * without the white space at either end (T.138: what a template that pipes the message through Jinja's trim writes;
 * the converter says so), {date} today (YYYY-MM-DD, the visitor's own day), and {date:format} today in strftime's
 * format (what the converter writes for a template's strftime_now(): the review of T.127, the day of the conversion
 * was kept with it). What was typed goes in as it is: as a replacement string, its $$, $&, $` and $' were patterns
 * (a typed $' wrote the rest of the template, special tokens and all; the review of T.132). */
export function filled(template: string, prompt: string, today = new Date()): string {
  return template.replace(/\{date(?::([^}]*))?\}/g, (_: string, format = "%Y-%m-%d") => strftime(format, today))
    .replace(/\{prompt(:trim)?\}/, (_: string, trim: string | undefined) => (trim ? strip(prompt) : prompt));
}
