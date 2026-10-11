// updated.js (T367.4): the one place that says "the site was updated while this page was open".
// The site's runtime is served under names that carry a hash of each file's bytes (_astro/<name>.<hash>.js, .wasm,
// .zip: T367.3). A deployment replaces them all, so a page left open across one still runs its own build, and a file
// of that build which it had not fetched yet (the converter's archive at a first conversion, a module asked for
// later, a section's worker of /benchmark/) is no longer there: the server answers 404. That, and nothing else, is
// what the sentence is for. A line that dropped, an answer of the 500s, a request something blocked, a page that is
// offline, a model's part or a file of another site keep the words they had.
//
// Two ways a file is found gone:
// - its own fetch was answered 404 (built.js's built(): the kernels and the Python's archives), which is the answer;
// - the failure says no status (a dynamic import() that threw, a worker that did not start): then the file that NAMES
//   the missing one is asked for, by HEAD, and the site was updated where that one is gone. The missing file's own
//   address is not known here (the bundler writes its hashed name into the text, and only Chrome's error says it),
//   but the namer's is (its import.meta.url), and a name is a hash of the contents: the namer's text holds the named
//   file's name, so where the named file changed, the namer's own name changed with it, and it is gone as well.
//   (The other way round it does not hold: the namer may have changed alone. Then the named file failed for a reason
//   of its own while the site was in fact updated, and the sentence is said: what it says is true and what it asks for
//   is right.) A HEAD is never answered from the service worker's copies (public/coi.js keeps GETs only), and
//   no-store keeps the browser's own cache out of it. It is asked once, after a failure, never before one.

/** The sentence, as the owner wrote it (T367's decision 4). Nothing is put before or after it. */
export const UPDATED = "The site was updated while this page was open. Reload the page to go on.";

/** The error that says so: an Error by name, so that whatever reports an error by its message says the sentence alone. */
export const updated = (cause) => Object.assign(new Error(UPDATED, { cause }), { updated: true });
export const isUpdated = (error) => error?.updated === true;
/** What is said of a caught value: the sentence alone where it is updated()'s, else what the caller says of it. */
export const saidOf = (error, otherwise) => (isUpdated(error) ? UPDATED : otherwise);

/** Whether an address is a file of this site's build: this origin's, under _astro/ (where Astro puts what it builds).
 * A model's part (models/), Pyodide's CDN, huggingface.co and a development server's sources are not. */
export function ofThisBuild(address, here = globalThis.location) {
  try {
    const url = new URL(address);
    return Boolean(here?.origin) && url.origin === here.origin && url.pathname.includes("/_astro/");
  } catch {
    return false;
  }
}

/** Whether an answer says that a file of this build is gone: a 404 for an address of the build. */
export const answeredGone = (address, status, here) => status === 404 && ofThisBuild(address, here);

const ASKING_SECONDS = 10;
/** Whether a file of this build is gone now: the server answers its HEAD with 404. Anything else is false: another
 * status, no answer (offline, a line that dropped, a blocked request), none within ASKING_SECONDS, another site. */
export async function gone(address, { fetch = globalThis.fetch, here = globalThis.location, seconds = ASKING_SECONDS } = {}) {
  if (!ofThisBuild(address, here)) return false;
  const asking = new AbortController();
  const timer = setTimeout(() => asking.abort(), seconds * 1000);
  try {
    const answer = await fetch(String(address), { method: "HEAD", cache: "no-store", signal: asking.signal });
    return answeredGone(address, answer.status, here);
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/** What `asking()` resolves to (a dynamic import, by the module that writes its name: namer is that module's
 * import.meta.url). Where it fails and the namer is gone, the failure is updated(); else it is the failure it was. */
export async function named(namer, asking, probe) {
  try {
    return await asking();
  } catch (error) {
    throw isUpdated(error) || !(await gone(namer, probe)) ? error : updated(error);
  }
}

/** The error of a worker's `error` event, for the module that made the worker (namer: its import.meta.url). A script
 * that threw says so (the event has a message), and that is `otherwise`; one that did not load says nothing, and
 * where the namer is gone that is updated(). */
export async function unstarted(event, namer, otherwise, probe) {
  return !event?.message && (await gone(namer, probe)) ? updated() : new Error(otherwise);
}
