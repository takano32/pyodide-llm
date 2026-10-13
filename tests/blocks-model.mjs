// blocks-model.mjs (T363.4): what the kernels of kernels/blocks.ts must give, written out value by value. One function
// with nothing outside it, so that tests/kernels-in-browser.mjs can hand its text to a page: tests/blocks-check.mjs
// and that check call the same model.
//
// The blocks are read here a second time, weight by weight, from the forms src/python/convert/superblocks.py
// documents (T362), and not as the kernels read them (16 weights a vector, the signs as masks): blocks-check holds
// read()'s float32 values to that reader's, to the bit, for every weight it checks. What a kernel returns is then the
// arithmetic at the head of kernels/blocks/types.ts, followed with Math.fround: a weight is F S W - Fm M, the integers
// G and M of a group of 32 are exact, and the float32 steps have one order.
//
// books: a checkpoint's row "codebooks" (convert/grids.py's codebooks(): the six grids, then IQ4_XS's 16 values), as
// an Int8Array. The page's check makes one up (the kernels do not care what a grid holds).
export function blocksModel() {
  const f = Math.fround;
  // the type's name: its number in a GGUF (the kernels' argument) and the bytes of a block
  const TYPES = { Q2_K: [10, 84], Q4_K: [12, 144], IQ2_XXS: [16, 66], IQ2_XS: [17, 74], IQ3_XXS: [18, 98], IQ1_S: [19, 50],
    IQ3_S: [21, 110], IQ2_S: [22, 82], IQ4_XS: [23, 136], IQ1_M: [29, 56] };
  const BOOKS = { IQ2XXS: 0, IQ2XS: 2048, IQ2S: 6144, IQ3XXS: 14336, IQ3S: 15360, IQ1S: 17408, IQ4NL: 33792, bytes: 33808 };
  const TABLES = 34976;  // the bytes sb_tables writes
  const half = (bits) => {
    const sign = bits & 0x8000 ? -1 : 1, e = (bits >> 10) & 31, m = bits & 1023;
    if (e === 31) return m ? NaN : sign * Infinity;
    return e === 0 ? sign * m * 2 ** -24 : sign * (1 + m / 1024) * 2 ** (e - 15);
  };
  const u16 = (U, p) => U[p] | (U[p + 1] << 8);
  const u32 = (U, p) => (U[p] | (U[p + 1] << 8) | (U[p + 2] << 16) | (U[p + 3] << 24)) >>> 0;
  // the eight signs of an index of 7 bits: the eighth makes the number of minus signs even
  const eight = (index) => {
    let ones = 0;
    for (let b = 0; b < 7; b++) ones += (index >> b) & 1;
    return index | ((ones & 1) << 7);
  };
  const nibble = (U, p, i) => (U[p + (i >> 1)] >> (4 * (i & 1))) & 15;

  /** The block of the type `name` at U[p]: { w: W of its 256 weights, s: S of each 16, m: M of each 16 or null, F,
   * Fm, values: the 256 as the reader gives them, float32 } */
  function read(name, U, p, books) {
    const w = new Int32Array(256), s = new Int32Array(16), values = new Float32Array(256);
    let m = null, F = 0, Fm = 0;
    const d = half(u16(U, p));
    const put = (t, magnitude, minus, scale) => { w[t] = minus ? -magnitude : magnitude; values[t] = f(f(scale * magnitude) * (minus ? -1 : 1)); };
    if (name === "IQ2_XXS" || name === "IQ3_XXS") {
      const fine = name === "IQ3_XXS";
      F = f(d * (fine ? 0.25 : 0.125));
      for (let g = 0; g < 8; g++) {
        const aux = u32(U, fine ? p + 66 + 4 * g : p + 6 + 8 * g), sc = aux >>> 28;
        const scale = f(f(d * f(0.5 + sc)) * (fine ? 0.5 : 0.25));
        s[2 * g] = s[2 * g + 1] = 2 * sc + 1;
        for (let l = 0; l < 4; l++) {
          const signs = eight((aux >>> (7 * l)) & 127);
          for (let j = 0; j < 8; j++) {
            const t = 32 * g + 8 * l + j;
            const magnitude = fine ? books[BOOKS.IQ3XXS + 4 * U[p + 2 + (t >> 2)] + (t & 3)] : books[BOOKS.IQ2XXS + 8 * U[p + 2 + 8 * g + l] + j];
            put(t, magnitude, (signs >> j) & 1, scale);
          }
        }
      }
    } else if (name === "IQ2_XS" || name === "IQ2_S") {
      const small = name === "IQ2_S";
      F = f(d * 0.125);
      for (let e = 0; e < 32; e++) {  // 8 weights
        const sc = nibble(U, p + (small ? 74 : 66), e >> 1), scale = f(f(d * f(0.5 + sc)) * 0.25);
        s[e >> 1] = 2 * sc + 1;
        const word = u16(U, p + 2 + 2 * e);
        const index = small ? U[p + 2 + e] | (((U[p + 66 + (e >> 2)] >> (2 * (e & 3))) & 3) << 8) : word & 511;
        const signs = small ? U[p + 34 + e] : eight(word >> 9);
        for (let j = 0; j < 8; j++) put(8 * e + j, books[(small ? BOOKS.IQ2S : BOOKS.IQ2XS) + 8 * index + j], (signs >> j) & 1, scale);
      }
    } else if (name === "IQ3_S") {
      F = d;
      for (let t = 0; t < 256; t++) {
        const sc = nibble(U, p + 106, t >> 5);
        s[t >> 4] = 2 * sc + 1;
        const index = U[p + 2 + (t >> 2)] | (((U[p + 66 + (t >> 5)] >> ((t >> 2) & 7)) & 1) << 8);
        put(t, books[BOOKS.IQ3S + 4 * index + (t & 3)], (U[p + 74 + (t >> 3)] >> (t & 7)) & 1, f(d * (2 * sc + 1)));
      }
    } else if (name === "IQ1_S" || name === "IQ1_M") {
      const mixed = name === "IQ1_M";
      const held = mixed ? [0, 1, 2, 3].map((k) => u16(U, p + 48 + 2 * k)) : null;
      const whole = mixed ? half((held[0] >> 12) | ((held[1] >> 8) & 0x00f0) | ((held[2] >> 4) & 0x0f00) | (held[3] & 0xf000)) : d;
      F = f(whole * 0.125);
      for (let e = 0; e < 32; e++) {  // 8 weights
        let index, sc, down;
        if (mixed) {
          const four = nibble(U, p + 32, e);
          index = U[p + e] | ((four & 7) << 8);
          sc = (held[e >> 3] >> (3 * ((e >> 1) & 3))) & 7;
          down = four & 8;
        } else {
          const qh = u16(U, p + 34 + 2 * (e >> 2));
          index = U[p + 2 + e] | (((qh >> (3 * (e & 3))) & 7) << 8);
          sc = (qh >> 12) & 7;
          down = qh & 0x8000;
        }
        s[e >> 1] = 2 * sc + 1;
        const scale = f(whole * (2 * sc + 1));
        for (let j = 0; j < 8; j++) {
          const g = books[BOOKS.IQ1S + 8 * index + j];
          w[8 * e + j] = 8 * g + (down ? -1 : 1);
          values[8 * e + j] = f(scale * f(g + (down ? -0.125 : 0.125)));
        }
      }
    } else if (name === "IQ4_XS") {
      F = d;
      const high = u16(U, p + 2);
      for (let g = 0; g < 8; g++) {
        const scale = (nibble(U, p + 4, g) | (((high >> (2 * g)) & 3) << 4)) - 32;
        s[2 * g] = s[2 * g + 1] = scale;
        for (let j = 0; j < 32; j++) {
          const t = 32 * g + j;
          w[t] = books[BOOKS.IQ4NL + nibble(U, p + 8 + 16 * g, j < 16 ? 2 * j : 2 * (j - 16) + 1)];
          values[t] = f(f(d * scale) * w[t]);
        }
      }
    } else if (name === "Q2_K" || name === "Q4_K") {
      const wide = name === "Q4_K";
      F = wide ? d : half(u16(U, p + 80));
      Fm = half(u16(U, p + (wide ? 2 : 82)));
      m = new Int32Array(16);
      for (let u = 0; u < 16; u++) {
        if (wide) {
          const g = u >> 1, h = p + 4;
          s[u] = g < 4 ? U[h + g] & 63 : (U[h + 4 + g] & 15) | ((U[h + g - 4] >> 6) << 4);
          m[u] = g < 4 ? U[h + 4 + g] & 63 : (U[h + 4 + g] >> 4) | ((U[h + g] >> 6) << 4);
        } else {
          s[u] = U[p + u] & 15;
          m[u] = U[p + u] >> 4;
        }
        for (let j = 0; j < 16; j++) {
          const t = 16 * u + j;
          // Q4_K: of every 32 bytes the low halves are 32 weights, the high halves the next 32. Q2_K: bits 2 s and
          // 2 s + 1 of byte b of a half's 32 are weight 32 s + b of the half
          w[t] = wide ? (U[p + 16 + 32 * (t >> 6) + (t & 31)] >> (4 * ((t >> 5) & 1))) & 15 : (U[p + 16 + 32 * (t >> 7) + (t & 31)] >> (2 * ((t & 127) >> 5))) & 3;
          values[t] = f(f(f(F * s[u]) * w[t]) - f(Fm * m[u]));
        }
      }
    } else {
      throw new Error(`no type of blocks called ${name}`);
    }
    return { w, s, m, F, Fm, values };
  }

  /** rows of n weights of the type at U[at], as the reader gives them: float32 */
  function values(name, U, at, n, rows, books) {
    const out = new Float32Array(rows * n), size = TYPES[name][1];
    for (let b = 0; b < (rows * n) / 256; b++) out.set(read(name, U, at + b * size, books).values, 256 * b);
    return out;
  }

  /** what matmul_sb gives for those rows against one token's activations: a (int8, -127 to 127) and their scales xs
   * (float32, one for each 32). Float32, to the bit. */
  function expected(name, U, at, n, rows, books, a, xs) {
    const out = new Float32Array(rows), size = TYPES[name][1], blocks = n / 256;
    for (let i = 0; i < rows; i++) {
      const lanes = [0, 0, 0, 0];
      for (let b = 0; b < blocks; b++) {
        const block = read(name, U, at + (i * blocks + b) * size, books);
        const G = new Array(8).fill(0), M = new Array(8).fill(0);
        for (let t = 0; t < 256; t++) {
          const x = a[256 * b + t];
          G[t >> 5] += block.s[t >> 4] * block.w[t] * x;
          if (block.m) M[t >> 5] += block.m[t >> 4] * x;
        }
        for (let k = 0; k < 4; k++) {
          const one = xs[8 * b + k], other = xs[8 * b + k + 4];
          let part = f(f(f(G[k] * one) + f(G[k + 4] * other)) * block.F);
          if (block.m) part = f(part - f(f(f(M[k] * one) + f(M[k + 4] * other)) * block.Fm));
          lanes[k] = f(lanes[k] + part);
        }
      }
      out[i] = f(f(f(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
    }
    return out;
  }

  /** what sums_x leaves after the scales: the int16 sums of each 16 of a, then of each 8 */
  function sums(a) {
    const out = new Int16Array(a.length / 16 + a.length / 8);
    a.forEach((value, j) => { out[j >> 4] += value; out[a.length / 16 + (j >> 3)] += value; });
    return out;
  }

  /** Whether the relaxed dot product of this engine multiplies signed bytes by signed bytes, exactly, for every pair of
   * bytes: what matmul_sbw_tile needs. dots(out, a, b): the kernel signed_dots; I8 and I32: the memory; at: 48 bytes
   * of it to use. (The four lanes are added up: which products a lane holds is the engine's.) */
  function signedDots(dots, I8, I32, at) {
    for (let x = -128; x < 128; x++) {
      for (let y0 = -128; y0 < 128; y0 += 16) {
        let want = 0;
        for (let j = 0; j < 16; j++) {
          I8[at + j] = x;
          I8[at + 16 + j] = y0 + j;
          want += x * (y0 + j);
        }
        dots(at + 32, at, at + 16);
        if (I32[at / 4 + 8] + I32[at / 4 + 9] + I32[at / 4 + 10] + I32[at / 4 + 11] !== want) return false;
      }
    }
    return true;
  }

  return { TYPES, BOOKS, TABLES, half, read, values, expected, sums, signedDots };
}
