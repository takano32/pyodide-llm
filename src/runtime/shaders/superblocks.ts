// shaders/superblocks.ts (T642.1): llama.cpp's blocks of 256 weights on the GPU, one table of the types, SUPERBLOCKS: of
// each, the WGSL that reads its blocks as the checkpoint holds them (with what every type shares: BLOCK_COMMON) and
// JavaScript's reading of them (the checks' answer). The GPU's reader of the types src/python/convert/superblocks.py
// reads for the converter. Moved out of prompt.ts (T642.1): the prompt's DP4A tile (prompt.ts), a token's matrix
// (fused.ts) and the embedding's rows (run.ts) take the texts from here, the GPU worker's checks the reading.
// A part of src/runtime/shaders.ts, which is the window: everything outside src/runtime/shaders/ imports that file and no part.
// A part takes its neighbours' names by static imports (T367.3: a bundler links them).
import type { BlockPacking } from "./common.ts";
// ---- T547.2.3, T547.3.3: llama.cpp's blocks of 256 weights (as src/python/convert/superblocks.py and
// tests/blocks-model.ts read them) on the GPU, as the checkpoint holds them: a type a row of SUPERBLOCKS. A block is
// eight groups of 32 weights (the activations' groups), each two halves of 16; a weight is a half's scale times its
// small integer q, less the half's min (Q4_K: d s q - dmin m; Q2_K: d (sc & 15) q - dmin (sc >> 4)). So a row's group
// adds, for each half, the scale times the integers' dot (DP4A) less the min times the activations' sum of the half,
// which a token takes once (the dot of its 16 int8 with ones, an integer: never a float16, as d s Σqx and dmin m Σx
// cancel). Written from the layouts here (the arithmetic of ggml-vulkan's mul_mmq_funcs.glsl and of ggml-webgpu's
// blocks, a group's scales beside the integers' dot, without their lines). What every type shares, BLOCK_COMMON (b, the
// weights' words; a block and what is read of it need not start at a word: Q2_K's dmin at byte 82 does not, nor do
// most IQ types' blocks), and each type's two
// functions: blk_values(block, g, col), half col of group g as packed int8 (q, 0 up), and blk_scales(block, g), the
// group's (scale, min) of each half, of the block whose first byte is block
const BLK_READ = /* wgsl */ `
// a byte of the weights
fn blk_byte(at: u32) -> u32 {
  return (b[at / 4u] >> (8u * (at % 4u))) & 255u;
}
// the four bytes from byte at
fn blk_word(at: u32) -> u32 {
  let shift = 8u * (at % 4u);
  if (shift == 0u) {
    return b[at / 4u];
  }
  return (b[at / 4u] >> shift) | (b[at / 4u + 1u] << (32u - shift));
}`;
// (T547.3.6: and what a matrix's text adds, which an embedded row's does not read)
const BLK_DOT = /* wgsl */ `
// the sum of 16 packed int8
fn blk_sum(a: vec4<u32>) -> i32 {
  return dot4I8Packed(a[0], 0x01010101u) + dot4I8Packed(a[1], 0x01010101u) + dot4I8Packed(a[2], 0x01010101u) + dot4I8Packed(a[3], 0x01010101u);
}
// the dot of 16 packed int8
fn blk_dot(a: vec4<u32>, w: vec4<u32>) -> i32 {
  return dot4I8Packed(a[0], w[0]) + dot4I8Packed(a[1], w[1]) + dot4I8Packed(a[2], w[2]) + dot4I8Packed(a[3], w[3]);
}
// a group of 32: the token's two halves (a0, a1) and their sums, the row's (w0, w1) and their scales and mins (s: the
// first half's scale and min, then the second's), to be times the token's scale
fn BLK(a0: vec4<u32>, w0: vec4<u32>, a1: vec4<u32>, w1: vec4<u32>, s: vec4<f32>, sums: vec2<f32>) -> f32 {
  return f32(blk_dot(a0, w0)) * s.x + f32(blk_dot(a1, w1)) * s.z - s.y * sums.x - s.w * sums.y;
}`;
const BLOCK_COMMON = BLK_READ + BLK_DOT;
// (T547.3.5: the float16 scale at a byte, which every type reads but IQ1_M, whose d is in pieces)
const BLK_F16 = /* wgsl */ `
// the float16 at byte at
fn blk_f16(at: u32) -> f32 {
  return unpack2x16float(blk_word(at)).x;
}`;
// (T547.3.4: what the K types' readings share, which no IQ type reads)
const K_BITS = /* wgsl */ `
// 16 values of 2 or 4 bits from byte at on, each the bits at shift of its byte
fn blk_bits(at: u32, shift: u32, mask: u32) -> vec4<u32> {
  return vec4<u32>((blk_word(at) >> shift) & mask, (blk_word(at + 4u) >> shift) & mask, (blk_word(at + 8u) >> shift) & mask, (blk_word(at + 12u) >> shift) & mask);
}`;
// T547.3.4: the IQ types read their values from the codebooks, which the checkpoint carries (the row "codebooks":
// convert/grids.py's codebooks(), a value a byte, each grid's entries one after the other, 8 or 4 values an entry) and the
// GPU holds as they are, bound where the K types' texts bind nothing (binding 1, a matrix's scales of int8): an entry of
// 8 is two words of packed int8 (DP4A's). IQ_WORDS: the word each grid begins at (codebooks()' places over 4; a second
// copy, which tests/gpu-worker-parts-check.ts holds to Python's). Signs: a byte, bit j the minus of the j-th value; of
// 7 bits, the eighth is their parity (llama.cpp's ksigns_iq2xs, this formula)
// (of the grids a type the GPU reads reads: every word of these is in a text, and the key's check holds that)
export const IQ_WORDS = { IQ2XXS: 0, IQ2XS: 512, IQ2S: 1536, IQ3XXS: 3584, IQ3S: 3840, IQ1S: 4352, IQ4NL: 8448 };
// (the bytes of the codebooks: engine/dtypes.py's CODEBOOKS, IQ4NL's 16 values last)
export const CODEBOOK_BYTES = 33808;
// (the types that hold 7 bits of signs: IQ_GRID with iq_seven)
const IQ_SEVEN = /* wgsl */ `
// 7 bits of signs with the eighth, their parity
fn iq_seven(bits: u32) -> u32 {
  return bits | ((countOneBits(bits) & 1u) << 7u);
}`;
// (T547.3.5: in parts, each type's text the ones it reads: the binding, the signs' factors, and an entry of 8 with its
// signs (IQ2), of 4 (IQ3), of 8 shifted (IQ1), or IQ4NL's values (IQ4_XS))
const IQ_BIND = /* wgsl */ `
@group(0) @binding(1) var<storage, read> grid: array<u32>;`;
const IQ_SIGNS = /* wgsl */ `
// the factors of four signs, bit j of bits the j-th
fn iq_signs(bits: u32) -> vec4<i32> {
  return vec4<i32>(1) - 2 * vec4<i32>((vec4<u32>(bits) >> vec4<u32>(0u, 1u, 2u, 3u)) & vec4<u32>(1u));
}`;
const IQ_EIGHT = /* wgsl */ `
// the 8 values of the grid from word at, with their signs: two words of packed int8
fn iq_eight(at: u32, signs: u32) -> vec2<u32> {
  return vec2<u32>(pack4xI8(unpack4xI8(grid[at]) * iq_signs(signs)), pack4xI8(unpack4xI8(grid[at + 1u]) * iq_signs(signs >> 4u)));
}`;
const IQ_GRID = IQ_BIND + IQ_SIGNS + IQ_EIGHT;
const IQ_FOUR = /* wgsl */ `
// the 4 values of the grid at word at, with their signs: a word of packed int8
fn iq_four(at: u32, signs: u32) -> u32 {
  return pack4xI8(unpack4xI8(grid[at]) * iq_signs(signs));
}`;
// (IQ1's: a weight is d (2 s + 1) (v +- 1/8) for its grid's value v of -1, 0 or 1, which is d (2 s + 1) / 8 times the
// integer 8 v +- 1: the same number, rounded once either way, and no minimum apart)
const IQ_SHIFTED = /* wgsl */ `
// the 8 values of the grid from word at, each 8 v + 1, or 8 v - 1 where minus: two words of packed int8
fn iq_shifted(at: u32, minus: u32) -> vec2<u32> {
  let shift = vec4<i32>(1 - 2 * i32(minus));
  return vec2<u32>(pack4xI8(unpack4xI8(grid[at]) * 8 + shift), pack4xI8(unpack4xI8(grid[at + 1u]) * 8 + shift));
}`;
const IQ_NL = /* wgsl */ `
// IQ4NL's values of four 4-bit codes, a code a byte of q: a word of packed int8
fn iq_nl(q: u32) -> u32 {
  var out = 0u;
  for (var j = 0u; j < 4u; j++) {
    let code = (q >> (8u * j)) & 15u;
    out |= ((grid[${IQ_WORDS.IQ4NL}u + code / 4u] >> (8u * (code % 4u))) & 255u) << (8u * j);
  }
  return out;
}`;
// T547.3.3: JavaScript's reading of each type's blocks (the checks' answer, tests/blocks-model.ts's read(): the weights,
// float32 at each step), and where its float16 scales are (d and dmin: where a check draws them small)
const halfAt = (bytes: Uint8Array, at: number) => {
  const bits = bytes[at] | (bytes[at + 1] << 8), sign = bits & 0x8000 ? -1 : 1, e = (bits >> 10) & 31, m = bits & 1023;
  return e === 31 ? (m ? NaN : sign * Infinity) : e === 0 ? sign * m * 2 ** -24 : sign * (1 + m / 1024) * 2 ** (e - 15);
};
// (T547.3.4: of the IQ types of 8 values an entry: entryOf(p, entry) -> [the entry's first byte in the codebooks, its
// signs, its scale s], for each of a block's 32 entries)
const word = (bytes: Uint8Array, at: number) => (bytes[at] | (bytes[at + 1] << 8) | (bytes[at + 2] << 16) | (bytes[at + 3] << 24)) >>> 0;
const seven = (bits: number) => {
  let ones = 0;
  for (let b = bits; b; b >>= 1) ones += b & 1;
  return bits | ((ones & 1) << 7);
};
function iqWeights(blocks: Uint8Array, bytes: number, entryOf: (p: number, entry: number) => [number, number, number], codebooks: Uint8Array) {
  const out = new Float32Array((blocks.length / bytes) * 256), grid = new Int8Array(codebooks.buffer, codebooks.byteOffset, codebooks.length);
  for (let p = 0, t0 = 0; p < blocks.length; p += bytes, t0 += 256) {
    const d = halfAt(blocks, p);
    for (let entry = 0; entry < 32; entry++) {
      const [at, signs, s] = entryOf(p, entry), scale = f(f(d * (0.5 + s)) * 0.25);
      for (let j = 0; j < 8; j++) out[t0 + 8 * entry + j] = f(scale * grid[at + j] * (signs & (1 << j) ? -1 : 1));
    }
  }
  return out;
}
// (scale(block, at, bits): a check's drawn float16 bits put where the block at `at` has its float16 scales: at16(the
// bytes' offsets))
const at16 = (...offsets: number[]) => (block: Uint8Array, at: number, bits: number) => {
  for (const offset of offsets) {
    block[at + offset] = bits & 255;
    block[at + offset + 1] = bits >> 8;
  }
};
// (T547.3.5: the weights of blocks of `bytes` each, a block's 256 by read(p, the codebooks as int8, its 256 of out))
const f = Math.fround;
function readBlocks(blocks: Uint8Array, bytes: number, codebooks: Uint8Array, read: (p: number, grid: Int8Array, out: Float32Array) => void) {
  const out = new Float32Array((blocks.length / bytes) * 256), grid = new Int8Array(codebooks.buffer, codebooks.byteOffset, codebooks.length);
  for (let p = 0, t0 = 0; p < blocks.length; p += bytes, t0 += 256) read(p, grid, out.subarray(t0, t0 + 256));
  return out;
}
// a type of llama.cpp's blocks the GPU reads: its bytes, its WGSL and JavaScript's reading of it (scale: a check's
// drawn float16 bits put where a block has its float16 scales; weights: the blocks' weights, float32 at each step)
type Superblock = { bytes: number; helpers: string; codebooks: boolean; code: string;
  scale(block: Uint8Array, at: number, bits: number): void; weights(blocks: Uint8Array, codebooks?: Uint8Array): Float32Array };
// Each type: its bytes, the WGSL its reading takes besides (helpers, T547.3.4: the K types' K_BITS, or the IQ types'
// IQ_GRID, which reads the codebooks: codebooks), its WGSL (code) and JavaScript's reading of its blocks (T547.3.3:
// scale and weights, the checks' answer, above). The key hashes the texts the makers make of code and helpers alone;
// tests/device-key-check.ts holds that no piece of scale and weights moves a key (T642.1: until then the two were
// two tables, BLOCK_READS and BLOCK_WEIGHTS).
// Q4_K: 144 bytes. Bytes 0 to 3: d and dmin; 4 to 15: the 6-bit s and m of the eight groups (llama.cpp's
// get_scale_min_k4: group g < 4 has s = byte g & 63 and m = byte g + 4 & 63, the others have their low four bits in byte
// g + 4 and their top two in bytes g - 4 and g); 16 on: the 4-bit q, of every 32 bytes the low halves the weights of a
// group and the high halves the next group's. Both halves of a group have its s and m.
// Q2_K: 84 bytes. Bytes 0 to 15: a byte a half (16 weights), its scale the low four bits and its min the top four; 16
// to 79: the 2-bit q, of every 32 bytes the bits at 0 the weights of a group, at 2 the next group's, at 4 and at 6 the
// two after (128 weights); 80 to 83: d and dmin.
export const SUPERBLOCKS = {
  Q4_K: { bytes: 144, helpers: BLK_F16 + K_BITS, codebooks: false, code: /* wgsl */ `
fn blk_values(block: u32, g: u32, col: u32) -> vec4<u32> {
  return blk_bits(block + 16u + 32u * (g / 2u) + 16u * col, 4u * (g % 2u), 0x0F0F0F0Fu);
}
fn blk_scales(block: u32, g: u32) -> vec4<f32> {
  var s: u32;
  var m: u32;
  if (g < 4u) {
    s = blk_byte(block + 4u + g) & 63u;
    m = blk_byte(block + 8u + g) & 63u;
  } else {
    s = (blk_byte(block + 8u + g) & 15u) | ((blk_byte(block + g) >> 6u) << 4u);
    m = (blk_byte(block + 8u + g) >> 4u) | ((blk_byte(block + 4u + g) >> 6u) << 4u);
  }
  let scale = blk_f16(block) * f32(s);
  let least = blk_f16(block + 2u) * f32(m);
  return vec4<f32>(scale, least, scale, least);
}`,
    scale: at16(0, 2),
    weights(blocks: Uint8Array) {
      const out = new Float32Array((blocks.length / 144) * 256);
      for (let p = 0, t0 = 0; p < blocks.length; p += 144, t0 += 256) {
        const d = halfAt(blocks, p), dmin = halfAt(blocks, p + 2), h = p + 4;
        for (let g = 0; g < 8; g++) {
          const s = g < 4 ? blocks[h + g] & 63 : (blocks[h + 4 + g] & 15) | ((blocks[h + g - 4] >> 6) << 4);
          const m = g < 4 ? blocks[h + 4 + g] & 63 : (blocks[h + 4 + g] >> 4) | ((blocks[h + g] >> 6) << 4);
          for (let j = 0; j < 32; j++) {
            const q = (blocks[p + 16 + 32 * (g >> 1) + j] >> (4 * (g & 1))) & 15;
            out[t0 + 32 * g + j] = f(f(f(d * s) * q) - f(dmin * m));
          }
        }
      }
      return out;
    },
  },
  Q2_K: { bytes: 84, helpers: BLK_F16 + K_BITS, codebooks: false, code: /* wgsl */ `
fn blk_values(block: u32, g: u32, col: u32) -> vec4<u32> {
  return blk_bits(block + 16u + 32u * (g / 4u) + 16u * col, 2u * (g % 4u), 0x03030303u);
}
fn blk_scales(block: u32, g: u32) -> vec4<f32> {
  let d = blk_f16(block + 80u);
  let dmin = blk_f16(block + 82u);
  let first = blk_byte(block + 2u * g);
  let second = blk_byte(block + 2u * g + 1u);
  return vec4<f32>(d * f32(first & 15u), dmin * f32(first >> 4u), d * f32(second & 15u), dmin * f32(second >> 4u));
}`,
    scale: at16(80, 82),
    weights(blocks: Uint8Array) {
      const out = new Float32Array((blocks.length / 84) * 256);
      for (let p = 0, t0 = 0; p < blocks.length; p += 84, t0 += 256) {
        const d = halfAt(blocks, p + 80), dmin = halfAt(blocks, p + 82);
        for (let i = 0; i < 256; i++) {
          const g = i >> 5, l = i & 31, sc = blocks[p + 2 * g + (l >> 4)];
          const q = (blocks[p + 16 + 32 * (g >> 2) + l] >> (2 * (g & 3))) & 3;
          out[t0 + i] = f(f(f(d * (sc & 15)) * q) - f(dmin * (sc >> 4)));
        }
      }
      return out;
    },
  },
  // T547.3.4: IQ2_XXS, 66 bytes: d, then 8 bytes for each group of 32: four indices of 8 bits into the grid IQ2XXS, each
  // 8 weights, and a word whose low 28 bits are their signs (7 bits each: IQ_GRID's iq_seven) and whose top four the
  // group's scale s; a weight is d (s + 1/2) / 4 times the grid's value, with its sign
  IQ2_XXS: { bytes: 66, helpers: BLK_F16 + IQ_GRID + IQ_SEVEN, codebooks: true, code: /* wgsl */ `
fn blk_values(block: u32, g: u32, col: u32) -> vec4<u32> {
  let at = block + 2u + 8u * g;
  let aux = blk_word(at + 4u) >> (14u * col);
  let first = iq_eight(${IQ_WORDS.IQ2XXS}u + 2u * blk_byte(at + 2u * col), iq_seven(aux & 127u));
  let second = iq_eight(${IQ_WORDS.IQ2XXS}u + 2u * blk_byte(at + 2u * col + 1u), iq_seven((aux >> 7u) & 127u));
  return vec4<u32>(first.x, first.y, second.x, second.y);
}
fn blk_scales(block: u32, g: u32) -> vec4<f32> {
  let scale = blk_f16(block) * (0.5 + f32(blk_word(block + 6u + 8u * g) >> 28u)) * 0.25;
  return vec4<f32>(scale, 0.0, scale, 0.0);
}`,
    // (T547.3.4: the IQ types', of the codebooks: d (s + 1/2) / 4 times the grid's value with its sign; the eighth sign
    // of 7 their parity. No minimum: d's bytes alone take a drawn scale)
    scale: at16(0),
    weights: (blocks: Uint8Array, codebooks?: Uint8Array) => iqWeights(blocks, 66, (p, entry) => {
      const g = entry >> 2, j = entry & 3, at = p + 2 + 8 * g, aux = word(blocks, at + 4);
      return [IQ_WORDS.IQ2XXS * 4 + 8 * blocks[at + j], seven((aux >>> (7 * j)) & 127), (aux >>> 28) & 15];
    }, codebooks!),
  },
  // IQ2_XS, 74 bytes: d, 32 words of 16 bits (each 8 weights: an index of 9 bits into the grid IQ2XS and 7 bits of
  // signs), a byte of scales for each group of 32 (the low four bits the first half's s, the high four the second's)
  IQ2_XS: { bytes: 74, helpers: BLK_F16 + IQ_GRID + IQ_SEVEN, codebooks: true, code: /* wgsl */ `
fn blk_values(block: u32, g: u32, col: u32) -> vec4<u32> {
  let words = blk_word(block + 2u + 8u * g + 4u * col);
  let first = iq_eight(${IQ_WORDS.IQ2XS}u + 2u * (words & 511u), iq_seven((words >> 9u) & 127u));
  let second = iq_eight(${IQ_WORDS.IQ2XS}u + 2u * ((words >> 16u) & 511u), iq_seven(words >> 25u));
  return vec4<u32>(first.x, first.y, second.x, second.y);
}
fn blk_scales(block: u32, g: u32) -> vec4<f32> {
  let d = blk_f16(block);
  let scales = blk_byte(block + 66u + g);
  return vec4<f32>(d * (0.5 + f32(scales & 15u)) * 0.25, 0.0, d * (0.5 + f32(scales >> 4u)) * 0.25, 0.0);
}`,
    scale: at16(0),
    weights: (blocks: Uint8Array, codebooks?: Uint8Array) => iqWeights(blocks, 74, (p, entry) => {
      const q = blocks[p + 2 + 2 * entry] | (blocks[p + 3 + 2 * entry] << 8), scales = blocks[p + 66 + (entry >> 2)];
      return [IQ_WORDS.IQ2XS * 4 + 8 * (q & 511), seven(q >> 9), entry & 2 ? scales >> 4 : scales & 15];
    }, codebooks!),
  },
  // IQ2_S, 82 bytes: d, qs[32], signs[32], qh[8], scales[8]: 8 weights are an index of 10 bits into the grid IQ2S (its
  // low 8 in qs, its high 2 in qh: a byte for each group, bits 2 l and 2 l + 1 for its l-th index) and a byte of signs;
  // the scales as IQ2_XS's
  IQ2_S: { bytes: 82, helpers: BLK_F16 + IQ_GRID, codebooks: true, code: /* wgsl */ `
fn blk_values(block: u32, g: u32, col: u32) -> vec4<u32> {
  let entry = 4u * g + 2u * col;
  let high = blk_byte(block + 66u + g) >> (4u * col);
  let first = iq_eight(${IQ_WORDS.IQ2S}u + 2u * (blk_byte(block + 2u + entry) | ((high & 3u) << 8u)), blk_byte(block + 34u + entry));
  let second = iq_eight(${IQ_WORDS.IQ2S}u + 2u * (blk_byte(block + 3u + entry) | (((high >> 2u) & 3u) << 8u)), blk_byte(block + 35u + entry));
  return vec4<u32>(first.x, first.y, second.x, second.y);
}
fn blk_scales(block: u32, g: u32) -> vec4<f32> {
  let d = blk_f16(block);
  let scales = blk_byte(block + 74u + g);
  return vec4<f32>(d * (0.5 + f32(scales & 15u)) * 0.25, 0.0, d * (0.5 + f32(scales >> 4u)) * 0.25, 0.0);
}`,
    scale: at16(0),
    weights: (blocks: Uint8Array, codebooks?: Uint8Array) => iqWeights(blocks, 82, (p, entry) => {
      const index = blocks[p + 2 + entry] | (((blocks[p + 66 + (entry >> 2)] >> (2 * (entry & 3))) & 3) << 8), scales = blocks[p + 74 + (entry >> 2)];
      return [IQ_WORDS.IQ2S * 4 + 8 * index, blocks[p + 34 + entry], entry & 2 ? scales >> 4 : scales & 15];
    }, codebooks!),
  },
  // T547.3.5: IQ3_XXS, 98 bytes: d, qs[64] (a byte an index into the grid IQ3XXS, 4 weights), 8 words for the groups
  // of 32 as IQ2_XXS's (the signs of four groups of 8, the scale s); a weight is d (s + 1/2) / 2 times the grid's value,
  // with its sign
  IQ3_XXS: { bytes: 98, helpers: BLK_F16 + IQ_BIND + IQ_SIGNS + IQ_FOUR + IQ_SEVEN, codebooks: true, code: /* wgsl */ `
fn blk_values(block: u32, g: u32, col: u32) -> vec4<u32> {
  let at = block + 2u + 8u * g + 4u * col;
  let aux = blk_word(block + 66u + 4u * g) >> (14u * col);
  let first = iq_seven(aux & 127u);
  let second = iq_seven((aux >> 7u) & 127u);
  return vec4<u32>(iq_four(${IQ_WORDS.IQ3XXS}u + blk_byte(at), first), iq_four(${IQ_WORDS.IQ3XXS}u + blk_byte(at + 1u), first >> 4u),
    iq_four(${IQ_WORDS.IQ3XXS}u + blk_byte(at + 2u), second), iq_four(${IQ_WORDS.IQ3XXS}u + blk_byte(at + 3u), second >> 4u));
}
fn blk_scales(block: u32, g: u32) -> vec4<f32> {
  let scale = blk_f16(block) * (0.5 + f32(blk_word(block + 66u + 4u * g) >> 28u)) * 0.5;
  return vec4<f32>(scale, 0.0, scale, 0.0);
}`,
    // (T547.3.5: and the rest of the IQ types, as convert/superblocks.py reads them)
    scale: at16(0),
    weights: (blocks: Uint8Array, codebooks?: Uint8Array) => readBlocks(blocks, 98, codebooks!, (p, grid, out) => {
      const d = halfAt(blocks, p);
      for (let g = 0; g < 8; g++) {
        const aux = word(blocks, p + 66 + 4 * g), scale = f(f(d * (0.5 + (aux >>> 28))) * 0.5);
        for (let e = 0; e < 8; e++) {
          const signs = seven((aux >>> (7 * (e >> 1))) & 127) >> (4 * (e & 1)), at = IQ_WORDS.IQ3XXS * 4 + 4 * blocks[p + 2 + 8 * g + e];
          for (let j = 0; j < 4; j++) out[32 * g + 4 * e + j] = f(f(scale * grid[at + j]) * (signs & (1 << j) ? -1 : 1));
        }
      }
    }),
  },
  // IQ3_S, 110 bytes: d, qs[64], qh[8], signs[32], scales[4]: 4 weights are an index of 9 bits into the grid IQ3S (a
  // byte of qs, and bit k of the byte of qh of its 32 weights for the k-th index of them), 8 a byte of signs; a byte of
  // scales for 64 weights (the low four bits the first 32's s); a weight is d (2 s + 1) times the grid's value, with its
  // sign
  IQ3_S: { bytes: 110, helpers: BLK_F16 + IQ_BIND + IQ_SIGNS + IQ_FOUR, codebooks: true, code: /* wgsl */ `
fn blk_values(block: u32, g: u32, col: u32) -> vec4<u32> {
  let at = block + 2u + 8u * g + 4u * col;
  let high = blk_byte(block + 66u + g) >> (4u * col);
  let first = blk_byte(block + 74u + 4u * g + 2u * col);
  let second = blk_byte(block + 75u + 4u * g + 2u * col);
  return vec4<u32>(iq_four(${IQ_WORDS.IQ3S}u + (blk_byte(at) | ((high & 1u) << 8u)), first),
    iq_four(${IQ_WORDS.IQ3S}u + (blk_byte(at + 1u) | (((high >> 1u) & 1u) << 8u)), first >> 4u),
    iq_four(${IQ_WORDS.IQ3S}u + (blk_byte(at + 2u) | (((high >> 2u) & 1u) << 8u)), second),
    iq_four(${IQ_WORDS.IQ3S}u + (blk_byte(at + 3u) | (((high >> 3u) & 1u) << 8u)), second >> 4u));
}
fn blk_scales(block: u32, g: u32) -> vec4<f32> {
  let scale = blk_f16(block) * f32(2u * ((blk_byte(block + 106u + g / 2u) >> (4u * (g % 2u))) & 15u) + 1u);
  return vec4<f32>(scale, 0.0, scale, 0.0);
}`,
    scale: at16(0),
    weights: (blocks: Uint8Array, codebooks?: Uint8Array) => readBlocks(blocks, 110, codebooks!, (p, grid, out) => {
      const d = halfAt(blocks, p);
      for (let g = 0; g < 8; g++) {
        const scale = f(d * (2 * ((blocks[p + 106 + (g >> 1)] >> (4 * (g & 1))) & 15) + 1));
        for (let e = 0; e < 8; e++) {
          const index = blocks[p + 2 + 8 * g + e] | (((blocks[p + 66 + g] >> e) & 1) << 8);
          const signs = blocks[p + 74 + 4 * g + (e >> 1)] >> (4 * (e & 1)), at = IQ_WORDS.IQ3S * 4 + 4 * index;
          for (let j = 0; j < 4; j++) out[32 * g + 4 * e + j] = f(f(scale * grid[at + j]) * (signs & (1 << j) ? -1 : 1));
        }
      }
    }),
  },
  // IQ1_S, 50 bytes: d, qs[32], 8 words of 16 bits: 8 weights are an index of 11 bits into the grid IQ1S (a byte of qs,
  // and bits 3 l to 3 l + 2 of its 32 weights' word for the l-th index); the word's bits 12 to 14 the scale s and its
  // top bit the sign of the shift (IQ_SHIFTED)
  IQ1_S: { bytes: 50, helpers: BLK_F16 + IQ_BIND + IQ_SHIFTED, codebooks: true, code: /* wgsl */ `
fn blk_values(block: u32, g: u32, col: u32) -> vec4<u32> {
  let qh = blk_word(block + 34u + 2u * g) & 65535u;
  let at = block + 2u + 4u * g + 2u * col;
  let first = iq_shifted(${IQ_WORDS.IQ1S}u + 2u * (blk_byte(at) | (((qh >> (6u * col)) & 7u) << 8u)), qh >> 15u);
  let second = iq_shifted(${IQ_WORDS.IQ1S}u + 2u * (blk_byte(at + 1u) | (((qh >> (6u * col + 3u)) & 7u) << 8u)), qh >> 15u);
  return vec4<u32>(first.x, first.y, second.x, second.y);
}
fn blk_scales(block: u32, g: u32) -> vec4<f32> {
  let scale = blk_f16(block) * f32(2u * ((blk_word(block + 34u + 2u * g) >> 12u) & 7u) + 1u) * 0.125;
  return vec4<f32>(scale, 0.0, scale, 0.0);
}`,
    scale: at16(0),
    weights: (blocks: Uint8Array, codebooks?: Uint8Array) => readBlocks(blocks, 50, codebooks!, (p, grid, out) => {
      const d = halfAt(blocks, p);
      for (let g = 0; g < 8; g++) {
        const qh = blocks[p + 34 + 2 * g] | (blocks[p + 35 + 2 * g] << 8), scale = f(d * (2 * ((qh >> 12) & 7) + 1)), shift = qh & 0x8000 ? -0.125 : 0.125;
        for (let l = 0; l < 4; l++) {
          const at = IQ_WORDS.IQ1S * 4 + 8 * (blocks[p + 2 + 4 * g + l] | (((qh >> (3 * l)) & 7) << 8));
          for (let j = 0; j < 8; j++) out[32 * g + 8 * l + j] = f(scale * (grid[at + j] + shift));
        }
      }
    }),
  },
  // IQ1_M, 56 bytes: qs[32], qh[16], 4 words of 16 bits, and no d of its own (the top four bits of the four words, the
  // first word's the lowest): 8 weights are an index of 11 bits into the grid IQ1S (a byte of qs and the low three bits
  // of its four of qh: a byte for 16 weights, the low half the first 8), whose fourth bit is the sign of its shift
  // (IQ_SHIFTED); a word's low 12 bits the scales s of its four groups of 16
  IQ1_M: { bytes: 56, helpers: IQ_BIND + IQ_SHIFTED, codebooks: true, code: /* wgsl */ `
fn blk_values(block: u32, g: u32, col: u32) -> vec4<u32> {
  let four = blk_byte(block + 32u + 2u * g + col);
  let at = block + 4u * g + 2u * col;
  let first = iq_shifted(${IQ_WORDS.IQ1S}u + 2u * (blk_byte(at) | ((four & 7u) << 8u)), (four >> 3u) & 1u);
  let second = iq_shifted(${IQ_WORDS.IQ1S}u + 2u * (blk_byte(at + 1u) | (((four >> 4u) & 7u) << 8u)), four >> 7u);
  return vec4<u32>(first.x, first.y, second.x, second.y);
}
fn blk_scales(block: u32, g: u32) -> vec4<f32> {
  let low = blk_word(block + 48u);
  let high = blk_word(block + 52u);
  let bits = ((low & 65535u) >> 12u) | ((low >> 24u) & 0xf0u) | (((high & 65535u) >> 4u) & 0xf00u) | ((high >> 16u) & 0xf000u);
  let d = unpack2x16float(bits).x * 0.125;
  let word = (select(low, high, g >= 4u) >> (16u * ((g / 2u) % 2u))) & 65535u;
  let shift = 6u * (g % 2u);
  return vec4<f32>(d * f32(2u * ((word >> shift) & 7u) + 1u), 0.0, d * f32(2u * ((word >> (shift + 3u)) & 7u) + 1u), 0.0);
}`,
    // (its d in the top four bits of its four words of scales, the first word's the lowest)
    scale: (block: Uint8Array, at: number, bits: number) => {
      for (let i = 0; i < 4; i++) block[at + 49 + 2 * i] = (block[at + 49 + 2 * i] & 15) | (((bits >> (4 * i)) & 15) << 4);
    },
    weights: (blocks: Uint8Array, codebooks?: Uint8Array) => readBlocks(blocks, 56, codebooks!, (p, grid, out) => {
      const w = [0, 1, 2, 3].map((i) => blocks[p + 48 + 2 * i] | (blocks[p + 49 + 2 * i] << 8));
      const bits = (w[0] >> 12) | ((w[1] >> 8) & 0xf0) | ((w[2] >> 4) & 0xf00) | (w[3] & 0xf000);
      const d = halfAt(Uint8Array.of(bits & 255, bits >> 8), 0);
      for (let h = 0; h < 16; h++) {
        const scale = f(d * (2 * ((w[h >> 2] >> (3 * (h & 3))) & 7) + 1));
        for (let k = 0; k < 2; k++) {
          const four = (blocks[p + 32 + h] >> (4 * k)) & 15, shift = four & 8 ? -0.125 : 0.125;
          const at = IQ_WORDS.IQ1S * 4 + 8 * (blocks[p + 2 * h + k] | ((four & 7) << 8));
          for (let j = 0; j < 8; j++) out[16 * h + 8 * k + j] = f(scale * (grid[at + j] + shift));
        }
      }
    }),
  },
  // IQ4_XS, 136 bytes: d, a word of 16 bits (the top two bits of the groups' scales: bits 2 g and 2 g + 1), scales_l[4]
  // (their low four, a byte for two groups, the low half first), qs[128] (a group's 16 bytes: the low halves its first 16
  // weights, the high its last); a weight is d (ls - 32) times IQ4NL's value of its four bits
  IQ4_XS: { bytes: 136, helpers: BLK_F16 + IQ_BIND + IQ_NL, codebooks: true, code: /* wgsl */ `
fn blk_values(block: u32, g: u32, col: u32) -> vec4<u32> {
  let at = block + 8u + 16u * g;
  let shift = 4u * col;
  return vec4<u32>(iq_nl((blk_word(at) >> shift) & 0x0F0F0F0Fu), iq_nl((blk_word(at + 4u) >> shift) & 0x0F0F0F0Fu),
    iq_nl((blk_word(at + 8u) >> shift) & 0x0F0F0F0Fu), iq_nl((blk_word(at + 12u) >> shift) & 0x0F0F0F0Fu));
}
fn blk_scales(block: u32, g: u32) -> vec4<f32> {
  let low = (blk_byte(block + 4u + g / 2u) >> (4u * (g % 2u))) & 15u;
  let high = (blk_word(block + 2u) >> (2u * g)) & 3u;
  let scale = blk_f16(block) * (f32(low | (high << 4u)) - 32.0);
  return vec4<f32>(scale, 0.0, scale, 0.0);
}`,
    scale: at16(0),
    weights: (blocks: Uint8Array, codebooks?: Uint8Array) => readBlocks(blocks, 136, codebooks!, (p, grid, out) => {
      const d = halfAt(blocks, p), high = blocks[p + 2] | (blocks[p + 3] << 8);
      for (let g = 0; g < 8; g++) {
        const ls = ((blocks[p + 4 + (g >> 1)] >> (4 * (g & 1))) & 15) | (((high >> (2 * g)) & 3) << 4), scale = f(d * (ls - 32));
        for (let i = 0; i < 32; i++) {
          const byte = blocks[p + 8 + 16 * g + (i & 15)], code = i < 16 ? byte & 15 : byte >> 4;
          out[32 * g + i] = f(scale * grid[IQ_WORDS.IQ4NL * 4 + code]);
        }
      }
    }),
  },
} satisfies Record<BlockPacking, Superblock>;

export { BLOCK_COMMON, BLK_READ };
