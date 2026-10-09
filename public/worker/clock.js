// worker/clock.js (T350): the seconds since a moment, and a turn of the event loop.
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

export const since = (started) => (performance.now() - started) / 1000;

// Hand the event loop a turn, so that a message sent meanwhile is delivered. setTimeout would cost 4ms per
// call (the browsers clamp it), a MessageChannel comes back in the same millisecond.
export function breathe() {
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(0);
  });
}
