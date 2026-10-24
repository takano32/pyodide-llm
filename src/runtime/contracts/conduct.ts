// contracts/conduct.ts (T.368.2): what the conduct of a conversion asks, and what answers it. Types alone: nothing of
// this file is in what a browser gets (it is read with `import type`).
//
// The conduct is Python's (src/python/convert/conduct.py: a generator that asks for a file, a part of one, a stream,
// and ends). The worker answers it (src/runtime/worker/conduct.ts: the loop, and an answerer for where the files are,
// huggingface.co or a folder of the visitor's disk); the tools answer it from a folder (tests/conducting.mjs).
// A request comes as a tuple, its kind first. tests/contracts-check.mjs holds the kinds and their lengths here to
// the tuples conduct.py yields and to the answers of both answerers.

// which repository a file is asked of: the model's, or the one it names for its vocabulary (T.136)
export type Where = "weights" | "vocabulary";

// a part of the weights handed to the conversion (Python's Conversion.feed, a proxy): the share converted so far
export type Feed = (part: Uint8Array) => number;

export type Request =
  // a file as text, or as bytes: undefined where it is not there
  | ["text", where: Where, name: string]
  | ["bytes", where: Where, name: string]
  // the bytes [begin, end) of a file: [the bytes, the size of the whole file (undefined where the answer did not show it)]
  | ["range", where: Where, name: string, begin: number, end: number]
  // the size of a file
  | ["size", where: Where, name: string]
  // the bytes [begin, end) of the weights, in order, to feed; of total bytes, before were in when this stream began
  | ["stream", where: Where, name: string, begin: number, end: number, before: number, total: number, feed: Feed]
  // the end: the conversion (a proxy, the caller's to destroy), or a file the model needs that is not there
  | ["done", conversion: unknown]
  | ["missing", where: Where, name: string];

// What each request is answered with (a file that is not there: undefined, which is None in Python; null is not)
export type Answers = {
  text(where: Where, name: string): Promise<string | undefined> | string | undefined;
  bytes(where: Where, name: string): Promise<Uint8Array | undefined> | undefined;
  range(where: Where, name: string, begin: number, end: number): Promise<[Uint8Array, number | undefined] | undefined> | undefined;
  size(where: Where, name: string): Promise<number> | number | undefined;
  stream(where: Where, name: string, begin: number, end: number, before: number, total: number, feed: Feed): Promise<undefined> | undefined;
};

// An answerer: the answers, and the error of a conduct that ended for want of a file, in the words of whoever knows
// the place (undefined: the loop has a sentence of its own)
export type Answerer = { answers: Answers; missing(where: Where, name: string): Error | undefined };

// What an answerer tells of a stream: the size of what is streamed once it is known, the bytes in so far, and a part
// handed to the conversion (feed() returns the share converted)
export type Progress = { of(total: number): void; arriving(bytes: number): void; converting(feed: () => number): void };
