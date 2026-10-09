// worker/state.js (T350): what the modules of the worker read and set together. A module cannot assign a variable of
// another, so what more than one of them sets is a field of this one object.
// A module of public/worker.js, which asks for it with its own ?v=<build>; it reads its neighbours the same way.

const HF_CONNECTIONS = 6;
export const state = {
  // T156: the model on the GPU alone that is loading or loaded now (what gpuOnlyBuffer() made), or undefined
  gpuOnlyNow: undefined,
  // T107: ?hfParts=<MiB>&hfConnections=<N> fix the two, to measure; the page offers no way to them
  hfPartBytes: 0, hfConnections: HF_CONNECTIONS,  // 0: not fixed, decided per file from its first part
};

// T156: where the loops that write the weights can wait for the GPU's worker of a model on the GPU alone (room), and
// before its engine is built (drained)
export const weightsRoom = () => state.gpuOnlyNow?.room?.();
export const weightsDrained = () => state.gpuOnlyNow?.drained?.();
