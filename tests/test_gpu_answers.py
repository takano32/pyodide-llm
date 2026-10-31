# test_gpu_answers.py (T.465.6.4's review): tests/gpu_answers.py's answers of the kinds that keep a state, where the
# checks that run them (gpu-tokens-check.ts) do not reach: a token's first layer is one token at place 0, so a state at
# an `at` past the first place, a block of more tokens than the taps (and its `batch`) and an attending layer's output
# gate after linear ones are never asked there. Held here in NumPy alone (the answers are the same Python the bridge calls):
#   - the same numbers at another `at` give the same answer there (the convolutions' rows, the delta rule's state);
#   - a block of two tokens is one token and then the next, its state carried (the convolutions and the delta rule);
#   - OUTPUT_GATE and SWIGLU to their formulas; (T.465.6.5's review) the outlier channels, a prompt's rows and RoPE to
#     numbers worked by hand.
import numpy as np

import gpu_answers as G


def u32(*words):
    return np.array(words, dtype=np.uint32).tobytes()


def f32(values):
    return np.asarray(values, dtype=np.float32).tobytes()


def floats(raw):
    return np.frombuffer(raw, dtype=np.float32)


rng = np.random.default_rng(465)


def drawn(n):
    return rng.uniform(-1, 1, n).astype(np.float32)


def test_short_conv_at_another_place_and_two_tokens_as_one_then_the_next():
    dim, taps, at = 8, 3, 2 * 3 * 8  # (two places of rows before this one's)
    mixed, weights, rows = drawn(2 * 3 * dim), drawn(at + taps * dim), drawn(at + taps * dim)
    zero = f32(np.zeros(2 * dim))
    # the same numbers at place 0
    alone = G.short_conv(f32(mixed), f32(weights[at:]), f32(rows[at:]), zero, u32(dim, taps, 0, 0), u32(2, 0, 0, 0))
    there = G.short_conv(f32(mixed), f32(weights), f32(rows), zero, u32(dim, taps, at, 0), u32(2, 0, 0, 0))
    assert floats(alone).tolist() == floats(there).tolist()
    # two tokens at once, and one then the other with the rows moved on by CONV_STATE in between
    first = G.short_conv(f32(mixed[: 3 * dim]), f32(weights), f32(rows), f32(np.zeros(dim)), u32(dim, taps, at, 0), u32(1, 0, 0, 0))
    moved = G.conv_state(f32(mixed[: 3 * dim]), f32(rows), u32(dim, taps, at, 0), u32(1, 0, 0, 0))
    second = G.short_conv(f32(mixed[3 * dim:]), f32(weights), moved, f32(np.zeros(dim)), u32(dim, taps, at, 0), u32(1, 0, 0, 0))
    np.testing.assert_allclose(floats(there), np.concatenate([floats(first), floats(second)]), rtol=1e-6, atol=1e-7)


def test_delta_conv_at_another_place_and_its_three_places():
    n, taps, keys, at = 12, 4, 2, 2 * 4 * 12
    mixed, weights, rows = drawn(2 * n), drawn(at + taps * n), drawn(at + taps * n)
    out = f32(np.zeros(2 * n))
    alone = G.delta_conv(f32(mixed), f32(weights[at:]), f32(rows[at:]), out, u32(n, taps, 0, keys, 2, 0, 0, 0), u32(2, 0, 0, 0))
    there = G.delta_conv(f32(mixed), f32(weights), f32(rows), out, u32(n, taps, at, keys, 2, 0, 0, 0), u32(2, 0, 0, 0))
    assert floats(alone).tolist() == floats(there).tolist()
    # (one token then the next: q, k and v of each in its own place of a block of one)
    first = G.delta_conv(f32(mixed[:n]), f32(weights), f32(rows), f32(np.zeros(n)), u32(n, taps, at, keys, 1, 0, 0, 0), u32(1, 0, 0, 0))
    moved = G.delta_conv_state(f32(mixed[:n]), f32(rows), u32(n, taps, at, keys, 1, 0, 0, 0), u32(1, 0, 0, 0))
    second = G.delta_conv(f32(mixed[n:]), f32(weights), moved, f32(np.zeros(n)), u32(n, taps, at, keys, 1, 0, 0, 0), u32(1, 0, 0, 0))
    one, two, both = floats(first), floats(second), floats(there)
    # the block of two: q of both, then k of both, then v of both
    np.testing.assert_allclose(both[: 2 * keys], np.concatenate([one[:keys], two[:keys]]), rtol=1e-6)
    np.testing.assert_allclose(both[2 * keys: 4 * keys], np.concatenate([one[keys: 2 * keys], two[keys: 2 * keys]]), rtol=1e-6)
    np.testing.assert_allclose(both[4 * keys:], np.concatenate([one[2 * keys:], two[2 * keys:]]), rtol=1e-6)


def test_delta_net_at_another_place_and_two_tokens_as_one_then_the_next():
    key_heads, value_heads, kd, vd = 1, 2, 4, 4
    keys, read, places = key_heads * kd, value_heads * vd, value_heads * kd * vd
    at = places  # (one place's state before this one's)
    conv = drawn(2 * 2 * keys + 2 * read)  # q of both tokens, k of both, v of both
    work = np.abs(drawn(2 * 2 * value_heads)) * 0.5 + 0.25
    state = drawn(at + places)
    out = f32(np.zeros(2 * read))
    s_alone, o_alone = G.delta_net(f32(conv), f32(work), f32(state[at:]), out, u32(key_heads, value_heads, kd, vd, 2, 0, 0, 0), u32(2, 0, 0, 0))
    s_there, o_there = G.delta_net(f32(conv), f32(work), f32(state), out, u32(key_heads, value_heads, kd, vd, 2, at, 0, 0), u32(2, 0, 0, 0))
    assert floats(o_alone).tolist() == floats(o_there).tolist()
    assert floats(s_alone).tolist() == floats(s_there)[at:].tolist()
    assert floats(s_there)[:at].tolist() == state[:at].tolist(), "another place's state as it was"
    # one token at a time, the state carried
    q, k, v = conv[: 2 * keys], conv[2 * keys: 4 * keys], conv[4 * keys:]
    one = np.concatenate([q[:keys], k[:keys], v[:read]])
    two = np.concatenate([q[keys:], k[keys:], v[read:]])
    s1, o1 = G.delta_net(f32(one), f32(work[: 2 * value_heads]), f32(state), f32(np.zeros(read)), u32(key_heads, value_heads, kd, vd, 1, at, 0, 0), u32(1, 0, 0, 0))
    s2, o2 = G.delta_net(f32(two), f32(work[2 * value_heads:]), s1, f32(np.zeros(read)), u32(key_heads, value_heads, kd, vd, 1, at, 0, 0), u32(1, 0, 0, 0))
    np.testing.assert_allclose(floats(o_there), np.concatenate([floats(o1), floats(o2)]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(floats(s_there), floats(s2), rtol=1e-5, atol=1e-6)


def test_output_gate_and_swiglu_to_their_formulas():
    x, g = drawn(6), drawn(6)
    gated = floats(G.output_gate(f32(x), f32(g), u32(3, 0, 0, 0), u32(2, 0, 0, 0)))
    np.testing.assert_allclose(gated, x / (1 + np.exp(-g.astype(np.float64))), rtol=1e-6)
    swi = floats(G.swiglu(f32(x), f32(g), u32(3, 0, 0, 0), u32(2, 0, 0, 0)))
    np.testing.assert_allclose(swi, x / (1 + np.exp(-x.astype(np.float64))) * g, rtol=1e-6)
    # (a token past the step's are left as they were)
    once = floats(G.output_gate(f32(x), f32(g), u32(3, 0, 0, 0), u32(1, 0, 0, 0)))
    assert once[3:].tolist() == x[3:].tolist()


# (T.465.6.5's review) the answers of the outlier channels, a prompt's rows and RoPE to numbers worked by hand
def test_outliers_rows_and_rope_to_numbers_worked_by_hand():
    # TAKE_OUTLIERS: channels 1 and 3 taken into picked, 0 in their places
    x, picked = G.take_outliers(f32([1, 2, 3, 4]), f32([0, 0, 0, 0]), u32(2, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0))
    assert floats(x).tolist() == [1, 0, 3, 0] and floats(picked).tolist() == [2, 4, 0, 0]
    # TERNARY_COLUMNS: rows of 128 codes; row 0's code of channel 1 is 2 (+1), row 1's is 0 (-1); scales 2 and 3
    table = np.full(2 * 128 // 4, 0x55, dtype=np.uint8)  # (every code 1: 0)
    table[0] = 0x55 & ~0x0c | (2 << 2)  # row 0, channel 1: code 2
    table[32] = 0x55 & ~0x0c | (0 << 2)  # row 1, channel 1: code 0
    out = G.ternary_columns(table.tobytes(), f32([2, 3]), f32([5, 0]), f32([1, 1]), u32(1, 2, 128, 0, 1, 0, 0, 0, 0, 0, 0, 0))
    assert floats(out).tolist() == [1 + 5 * 2, 1 - 5 * 3]
    # EMBED_ROWS: ids 0 and 2 of a piece of 2 rows from row 0: row 0 written, id 2 outside the piece and not
    rows, written = G.embed_rows(np.full(2 * 32, 1, dtype=np.int8).tobytes(), f32([0.5, 0.25]), u32(0, 2), 32, 0, 2, 2)
    assert written == [0] and floats(rows).tolist() == [0.5] * 32
    # ROPE: a head of 4 turned whole at a quarter turn (cos 0, sin 1), the key into the cache at pos + token, the value as it is
    angles = f32([0, 0, 1, 1])  # (a token's cos of its 2 angles, then their sin)
    q, keys, values = G.rope(f32([1, 0, 0, 1]), f32([1, 0, 0, 1]), f32([7, 8, 9, 10]), np.zeros(3 * 4, dtype=np.float16).tobytes(),
                             np.zeros(3 * 4, dtype=np.float16).tobytes(), angles, u32(1, 1, 4, 4), u32(1, 2, 0, 0))
    assert np.allclose(floats(q), [0, 1, -1, 0])
    assert np.frombuffer(keys, dtype=np.float16)[8:].tolist() == [0, 1, -1, 0]
    assert np.frombuffer(values, dtype=np.float16)[8:].tolist() == [7, 8, 9, 10]
