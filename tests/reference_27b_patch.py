# reference_27b_patch.py
# T237's review: adds one thing to a checkout of Prism ML's fork of llama.cpp (the pinned commit of tests/reference_27b.sh):
# a way to multiply the ternary matrices (PTQ1_0) by float32 activations. The fork's CPU rounds the activation of such a
# matrix to Q8_0 first (blocks of 32, a float16 scale, int8 values), so what it computes is itself a few tenths of a
# logit away from any other way of computing the model, its own batched and one-token paths included (0.10 to 0.13: T238),
# and a comparison with the engine can be no closer than that. With the activations left as they are, what stands between
# the fork and the engine's NumPy forward pass is float32's rounding (the rotation, the order of the value heads, the
# attention and the norms are the fork's own). It is a switch of the environment, PTQ1_0_F32_ACTIVATIONS: without it the
# fork is what it was.
#
#   python tests/reference_27b_patch.py <the checkout of the fork>
#
# The text of the fork is not in this file. What it changes is found by the names of its two functions that say which type of
# activation a matrix takes and which function multiplies (ggml_cpu_vec_dot_type and ggml_cpu_vec_dot, which every product
# of the CPU backend asks, the planning of its work buffer too), and what it adds is the function below, written for this
# from the layout of a PTQ1_0 block (docs/notes/t228-bonsai-2-2026-10-01.md, 3): 128 weights in 28 bytes, qs[24] holding
# five ternary digits a byte (the first 16 bytes the weights 0 to 79, digit by digit, the next 8 the weights 80 to 119) and
# qh[2] four a byte (the weights 120 to 127), then the float16 scale d; digit n of a byte b is ((b * 3^n mod 256) * 3) >> 8
# (0, 1 or 2: -d, 0 or +d). The lanes of a run are separate sums, so that the compiler can vectorize them without reordering
# any: the dot is float32 sums of 16 lanes, then a float64 sum of the blocks.
import sys
from pathlib import Path

DOT = r'''
// T237's review (tests/reference_27b_patch.py): a ternary row against float32 activations, for the switch
// PTQ1_0_F32_ACTIVATIONS of ggml_cpu_vec_dot_type() and ggml_cpu_vec_dot()
void ggml_vec_dot_ptq1_0_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_PTQ1_0 == 0);
    assert(nrc == 1);
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const block_ptq1_0 * GGML_RESTRICT x = vx;
    const float * GGML_RESTRICT y = vy;
    static const uint8_t power[5] = {1, 3, 9, 27, 81};

    double total = 0.0;
    for (int i = 0; i < n / QK_PTQ1_0; i++) {
        const float * v = y + i * QK_PTQ1_0;
        float lane[16] = {0};
        for (int digit = 0; digit < 5; digit++) {
            for (int m = 0; m < 16; m++) {
                const int t = ((uint16_t) (uint8_t) (x[i].qs[m] * power[digit]) * 3) >> 8;
                lane[m] += (float) (t - 1) * v[digit * 16 + m];
            }
            for (int m = 0; m < 8; m++) {
                const int t = ((uint16_t) (uint8_t) (x[i].qs[16 + m] * power[digit]) * 3) >> 8;
                lane[m] += (float) (t - 1) * v[80 + digit * 8 + m];
            }
        }
        for (int digit = 0; digit < 4; digit++) {
            for (int m = 0; m < 2; m++) {
                const int t = ((uint16_t) (uint8_t) (x[i].qh[m] * power[digit]) * 3) >> 8;
                lane[m] += (float) (t - 1) * v[120 + digit * 2 + m];
            }
        }
        float sum = 0.0f;
        for (int m = 0; m < 16; m++) {
            sum += lane[m];
        }
        total += (double) GGML_CPU_FP16_TO_FP32(x[i].d) * (double) sum;
    }
    *s = (float) total;
}

'''

DECLARATION = ("void ggml_vec_dot_ptq1_0_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, "
               "size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);\n")

WANTS = '''// T237's review (tests/reference_27b_patch.py): the ternary matrices take float32 activations where the environment says so
static inline bool ggml_cpu_ptq1_wants_f32(const struct ggml_tensor * src0) {
    static int wanted = -1;
    if (wanted < 0) {
        wanted = getenv("PTQ1_0_F32_ACTIVATIONS") != NULL ? 1 : 0;
    }
    return wanted == 1 && src0->type == GGML_TYPE_PTQ1_0;
}

'''


def once(text, anchor, name):
    if text.count(anchor) != 1:
        raise SystemExit(f"reference_27b_patch: {name}: {text.count(anchor)} places like {anchor!r} (the fork is not the pinned commit?)")
    return text.index(anchor)


def patch(root):
    root = Path(root)
    header = root / "ggml/src/ggml-cpu/quants.h"
    source = root / "ggml/src/ggml-cpu/quants.c"
    cpu = root / "ggml/src/ggml-cpu/ggml-cpu.c"
    texts = {path: path.read_text() for path in (header, source, cpu)}
    for path, text in texts.items():
        if "ptq1_0_f32" in text or "ptq1_wants_f32" in text:
            raise SystemExit(f"reference_27b_patch: {path} is patched already")

    # quants.h: the declaration after the one of the Q8_0 dot product of the same type
    text = texts[header]
    anchor = "void ggml_vec_dot_ptq1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);\n"
    at = once(text, anchor, "quants.h") + len(anchor)
    texts[header] = text[:at] + DECLARATION + text[at:]

    # quants.c: the function before the Q4_0 dot product's
    text = texts[source]
    anchor = "void ggml_vec_dot_q4_0_q8_0_generic("
    at = once(text, anchor, "quants.c")
    texts[source] = text[:at] + DOT.lstrip("\n") + text[at:]

    # ggml-cpu.c: the two functions the product asks
    text = texts[cpu]
    anchor = "static inline enum ggml_type ggml_cpu_vec_dot_type(const struct ggml_tensor * src0) {\n"
    at = once(text, anchor, "ggml-cpu.c (the type of activation)")
    text = text[:at] + WANTS + anchor + "    if (ggml_cpu_ptq1_wants_f32(src0)) {\n        return GGML_TYPE_F32;\n    }\n" + text[at + len(anchor):]
    anchor = "static inline ggml_vec_dot_t ggml_cpu_vec_dot(const struct ggml_tensor * src0) {\n"
    at = once(text, anchor, "ggml-cpu.c (the dot product)")
    text = text[:at] + anchor + "    if (ggml_cpu_ptq1_wants_f32(src0)) {\n        return ggml_vec_dot_ptq1_0_f32;\n    }\n" + text[at + len(anchor):]
    texts[cpu] = text

    for path, text in texts.items():
        path.write_text(text)
    print("reference_27b_patch: the ternary matrices take float32 activations where PTQ1_0_F32_ACTIVATIONS is set")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tests/reference_27b_patch.py <the checkout of the fork>")
    patch(sys.argv[1])
