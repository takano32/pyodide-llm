// tests/unisolated.ts (T662): whether a run of tests/e2e.ts whose page was not cross-origin isolated fails. The page's
// software threads need the isolation the site's service worker gives it (public/coi.js, T93); without it the page
// still answers, on one thread, and the weekly browsers.yml stayed green where a browser had lost it. So a page that
// is not isolated is a failure, unless:
//   - the visit asked for no isolation: ?coi=off (the page takes the service worker away, src/page/isolation.ts), or
//   - the system and the browser are a pair known not to isolate, in UNISOLATED below, with why.
//
// UNISOLATED is empty: the two weekly browsers.yml runs whose results were still kept when T662 counted them
// (43856274024 and 42609531338, 2026-11-09 and 11-16) had all their 398 runs isolated, every system of the five with
// every browser of each (Chromium, Firefox, WebKit, Chrome, Edge), and the models of Hugging Face in Chromium and
// WebKit. A pair goes in only with a run that shows it, and the reason.
//
// Read by tests/e2e.ts; held by tests/summary-check.ts (Node alone, no browser).

/** "<os.platform()> <os.arch()> <engine>" (as e2e.ts's record writes os, and the browser as it was asked for) → why it
 *  is not isolated there */
export const UNISOLATED = {};

/** The failure to record for a run, or null: `isolated` is the page's crossOriginIsolated, `os` is
 *  "<platform> <arch>", `engine` the browser asked for, `visited` the URL the run opened. */
export function unisolated({ isolated, os, engine, visited }, known = UNISOLATED) {
  if (isolated) return null;
  if (new URL(visited).searchParams.get("coi") === "off") return null;
  if (Object.hasOwn(known, `${os} ${engine}`)) return null;
  return `the page was not cross-origin isolated (${os}, ${engine}): the service worker did not take it, and it ran on one thread`;
}
