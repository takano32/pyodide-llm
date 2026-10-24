# reference_saluki_patch.py (T.364.1): adds one thing to a checkout of stock llama.cpp (the pinned commit of
# tests/reference_saluki.sh, MIT): a way to multiply the matrices of Underdog Saluki 27B by float32 activations. The CPU
# of llama.cpp rounds the activation of a k- or i-quantized matrix to Q8_K first (blocks of 256, one float scale, int8
# values) and that of a Q8_0 matrix to Q8_0, so what it computes is 0.3 to 11 of a logit from its own other path and
# from another CPU's (T.361), and nothing can be held to it narrowly. With the activations left as they are, what stands
# between llama.cpp and the engine's NumPy forward pass (tests/reference_saluki_numpy.py) is float32's order of sums:
# the reading of the blocks, the order of the value heads, the attention, the delta rule and the norms are llama.cpp's
# own. It is tests/reference_27b_patch.py's idea for a stock llama.cpp; a switch of the environment,
# BLOCKS_F32_ACTIVATIONS: without it llama.cpp is what it was.
#
#   python tests/reference_saluki_patch.py <the checkout of llama.cpp>
#
# The text of llama.cpp is not in this file. What it changes is found by name: the table of what each type's product
# takes (type_traits_cpu of ggml/src/ggml-cpu/ggml-cpu.c: every product of the CPU backend and the planning of its work
# buffer read the activation's type, the dot product and the rows a call takes from it) is made writable, and
# ggml_cpu_init(), which runs before any graph is computed, sets, where the switch is on, for the ten types of the
# file's matrices and for Q8_0: activations of GGML_TYPE_F32, one row a call, and the function below. That function is
# written for this: it widens the row with llama.cpp's own dequantize_row_<type> (the reading the converter's readers
# are held to, to the bit: tests/fork_readers.py) and adds the products up: sixteen float32 lanes over a block of 256
# (the lanes are separate sums, so that the compiler can vectorize them without reordering any), then a float64 sum
# of the blocks. A row is widened for every product with it: slower than llama.cpp as it is.
# (The driver, tests/reference_27b_fork.cpp, turns llama.cpp's repacking of Q4_K and Q8_0 weights off for this run,
# NO_REPACK: a repacked tensor has another type, which this table does not reach.)
import sys
from pathlib import Path

# (the name of the type in ggml's enum, of its block, of its dequantize_row function)
TYPES = (("Q2_K", "q2_K"), ("Q4_K", "q4_K"), ("IQ2_XXS", "iq2_xxs"), ("IQ2_XS", "iq2_xs"), ("IQ2_S", "iq2_s"),
         ("IQ3_XXS", "iq3_xxs"), ("IQ3_S", "iq3_s"), ("IQ1_S", "iq1_s"), ("IQ1_M", "iq1_m"), ("IQ4_XS", "iq4_xs"), ("Q8_0", "q8_0"))

SIGNATURE = ("(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, "
             "const void * GGML_RESTRICT vy, size_t by, int nrc)")

DOT = r'''
// T.364.1 (tests/reference_saluki_patch.py): a row of llama.cpp's blocks against float32 activations, for the switch
// BLOCKS_F32_ACTIVATIONS of ggml_cpu_init()
static float ggml_blocks_f32_dot(int n, const float * GGML_RESTRICT w, const float * GGML_RESTRICT y) {
    double total = 0.0;
    int i = 0;
    for (; i + 256 <= n; i += 256) {
        float lane[16] = {0};
        for (int j = 0; j < 256; j += 16) {
            for (int m = 0; m < 16; m++) {
                lane[m] += w[i + j + m] * y[i + j + m];
            }
        }
        float sum = 0.0f;
        for (int m = 0; m < 16; m++) {
            sum += lane[m];
        }
        total += (double) sum;
    }
    for (; i < n; i++) {
        total += (double) (w[i] * y[i]);
    }
    return (float) total;
}

#define GGML_BLOCKS_F32(name, block) \
void ggml_vec_dot_##name##_f32''' + SIGNATURE.replace("\n", "") + r''' { \
    assert(nrc == 1); \
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc); \
    float * w = (float *) malloc((size_t) n * sizeof(float)); \
    dequantize_row_##name((const block *) vx, w, n); \
    *s = ggml_blocks_f32_dot(n, w, (const float *) vy); \
    free(w); \
}

'''

SWITCH = '''        // T.364.1 (tests/reference_saluki_patch.py): the matrices of llama.cpp's blocks take float32 activations where the environment says so
        if (getenv("BLOCKS_F32_ACTIVATIONS") != NULL) {
%s        }
'''


def once(text, anchor, name):
    if text.count(anchor) != 1:
        raise SystemExit(f"reference_saluki_patch: {name}: {text.count(anchor)} places like {anchor!r} (llama.cpp is not the pinned commit?)")
    return text.index(anchor)


def patch(root):
    root = Path(root)
    header = root / "ggml/src/ggml-cpu/quants.h"
    source = root / "ggml/src/ggml-cpu/quants.c"
    cpu = root / "ggml/src/ggml-cpu/ggml-cpu.c"
    texts = {path: path.read_text() for path in (header, source, cpu)}
    for path, text in texts.items():
        if "BLOCKS_F32" in text or "ggml_blocks_f32_dot" in text:
            raise SystemExit(f"reference_saluki_patch: {path} is patched already")

    # quants.h: the declarations after the one of the Q8_0 dot product
    text = texts[header]
    anchor = f"void ggml_vec_dot_q8_0_q8_0{SIGNATURE};\n"
    at = once(text, anchor, "quants.h") + len(anchor)
    texts[header] = text[:at] + "".join(f"void ggml_vec_dot_{lower}_f32{SIGNATURE};\n" for _, lower in TYPES) + text[at:]

    # quants.c: the functions at the end (dequantize_row_* are ggml-quants.h's, which it includes)
    texts[source] = texts[source] + DOT + "".join(f"GGML_BLOCKS_F32({lower}, block_{lower})\n" for _, lower in TYPES)

    # ggml-cpu.c: the table writable, and set where the switch is on
    text = texts[cpu]
    anchor = "static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {\n"
    at = once(text, anchor, "ggml-cpu.c (the table)")
    text = text[:at] + anchor.replace("static const struct", "static struct") + text[at + len(anchor):]
    anchor = "    static bool is_first_call = true;\n\n    if (is_first_call) {\n"
    at = once(text, anchor, "ggml-cpu.c (ggml_cpu_init)") + len(anchor)
    lines = "".join(f"            type_traits_cpu[GGML_TYPE_{upper}].vec_dot = ggml_vec_dot_{lower}_f32;\n"
                    f"            type_traits_cpu[GGML_TYPE_{upper}].vec_dot_type = GGML_TYPE_F32;\n"
                    f"            type_traits_cpu[GGML_TYPE_{upper}].nrows = 1;\n" for upper, lower in TYPES)
    texts[cpu] = text[:at] + SWITCH % lines + text[at:]

    for path, text in texts.items():
        path.write_text(text)
    print(f"reference_saluki_patch: the matrices of {len(TYPES)} types take float32 activations where BLOCKS_F32_ACTIVATIONS is set")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tests/reference_saluki_patch.py <the checkout of llama.cpp>")
    patch(sys.argv[1])
