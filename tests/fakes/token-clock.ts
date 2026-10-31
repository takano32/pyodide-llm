// tests/fakes/token-clock.ts (T.357.5)
// A stand-in for the clock the engine times a token by (createForward()'s `clock`, which is performance.now in a
// page), for the checks of the search for the number of software threads (thread-search-check.ts, gpu-default-check.ts).
//
//   const clock = tokenClock(() => <the ms the token that just ended took>)
//   createForward({ ..., clock })
//
// The engine reads the clock twice a token: as it starts and as it ends. The first reading of a pair gives the time
// as it stands; the second asks msOfToken() what the token took (with whatever the check knows by then: the count of
// threads the engine ran it on, which comparison the search is in), moves the time on by that, and gives it. So a
// token takes exactly what the check says, and nothing else takes any time.
//
// Where it is simpler than a real clock: time stands still between a token's end and the next one's start, and it
// takes the readings to come in pairs. Something else that reads the engine's clock (one reading, not two) would put
// every later token out of step, its start taken for an end: the check's times would then be of the wrong tokens.
// (T.455: the GPU's side reads the same clock where forwardMany() and generateMany() run, and not always in pairs: a
// block of the prompt that is not timed, a GPU that fails or stops early read it once. So no check runs those two
// under this clock: the checks of the search call forward() alone, whose readings are none or a pair.)

/** A clock a token takes msOfToken() on: read as a token starts and as it ends. */
export function tokenClock(msOfToken) {
  let starting = false, time = 0;
  return () => {
    starting = !starting;
    if (!starting) time += msOfToken();
    return time;
  };
}
