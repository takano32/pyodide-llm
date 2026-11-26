// The step "nan" (T263): whether this device carries a NaN and an infinity where the engine stands on it. Each case of
// shaders/bench.ts's NAN_CASES runs once, a thread a case (the float32 ones in one dispatch, the f16 ones in another
// where the device has shader-f16), and what it made is read back as it is: src/bench/nan.ts says what each should be
// and which of the engine's marks stands on it. Runs on a fallback adapter too (a dispatch each: CI's software
// adapters say what they do, which is not a real GPU's).
// (T353: a module of /benchmark/'s GPU worker, benchmark/gpu.ts, which a bundler links it into: T367.3)
import { shared, gpu, STORAGE, COPY_DST, COPY_SRC, buffer, run, readBack, scoped } from "./device.ts";
import type { Thrown } from "./device.ts";
import type { NanFound, NanGot } from "../../contracts/benchmark.ts";

// an f32's bits (a NaN as the quiet one, 0x7fc00000: the bits a typed array would write are each engine's own)
const bitsOf = (value: number) => (Number.isNaN(value) ? 0x7fc00000 : new Uint32Array(new Float32Array([value]).buffer)[0]);

async function nan(): Promise<NanFound> {
  await gpu();
  const cases: NanGot[] = [];
  for (const half of [false, true]) {
    const these = shared.WGSL.NAN_CASES.filter((one) => Boolean(one.half) === half);
    if (half && !shared.features.half) {
      cases.push(...these.map(({ name, kind }): NanGot => ({ name, kind, none: "no shader-f16 here" })));
      continue;
    }
    try {
      const { got, stored } = await scoped(async (owned) => {
        const inputs = new Uint32Array(these.flatMap(({ a, b, word }) => [word ?? bitsOf(a), bitsOf(b)]));
        const floatInputs = new Uint32Array(these.flatMap(({ a, b }) => [bitsOf(a), bitsOf(b)]));
        const words = buffer(inputs.byteLength), floats = buffer(floatInputs.byteLength);
        const bits = buffer(these.length * 4, STORAGE | COPY_SRC), kept = buffer(these.length * 4, STORAGE | COPY_SRC | COPY_DST);
        owned.push(words, floats, bits, kept);
        shared.device.queue.writeBuffer(words, 0, inputs);
        shared.device.queue.writeBuffer(floats, 0, floatInputs);
        const pipeline = await shared.device.createComputePipelineAsync({ layout: "auto",
          compute: { module: shared.device.createShaderModule({ code: shared.WGSL.nanShader(half) }), entryPoint: "main" } });
        const group = shared.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
          entries: [words, floats, bits, kept].map((b, binding) => ({ binding, resource: { buffer: b } })) });
        const encoder = shared.device.createCommandEncoder(), pass = encoder.beginComputePass();
        run(pass, [pipeline, group, Math.ceil(these.length / shared.WGSL.NAN_WORKGROUP), 1]);
        pass.end();
        const got = new Uint32Array(await readBack(encoder, bits, these.length * 4));
        const stored = new Uint32Array(await readBack(shared.device.createCommandEncoder(), kept, these.length * 4));
        return { got, stored };
      });
      cases.push(...these.map(({ name, kind }, i): NanGot => ({ name, kind, got: got[i], ...(kind === "float" ? { stored: stored[i] } : {}) })));
    } catch (error) {
      const said = String((error as Thrown)?.message ?? error);
      cases.push(...these.map(({ name, kind }): NanGot => ({ name, kind, error: said })));
    }
  }
  return { cases };
}

export { nan };
