# t272_probe.py (the review of T260, T272): the real LFM2.5 350M (int8 as the page converts Liquid's Q8_0) computed a window at a
# time in NumPy float32, with the input of each kind of matrix rounded as the kernels round activations (per group of 32:
# scale = largest |value| / qmax, round half to even; qmax 63 for the 7-bit relaxed path, 127 for the 8-bit one) or left
# alone, for each kind of input apart, for the first position (the BOS) and the others apart, for each layer, with other
# groupings and with outlier channels taken out. Every line it prints begins with "t272:".
#
#   python tests/t272_probe.py <prefix of tests/perplexity_prepare.py's output> <text file> <tokens> <experiment> ...
#
# A throwaway of the review: it holds nothing the repository keeps.
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
from llama2_numpy import Llama, rope_frequencies  # noqa: E402

ROLES = ("qkv", "o", "win", "wout", "gate_up", "down", "cls")
say = lambda **fields: print("t272: " + json.dumps(fields, default=float), flush=True)
qmax_of = lambda bits: 2 ** (bits - 1) - 1


def load(prefix):
    options = json.loads(Path(f"{prefix}.json").read_text())
    return Llama(Path(f"{prefix}.bin").read_bytes(), Path(f"{prefix}.tokenizer.bin").read_bytes(), kernels=None, **options)


def chosen_rows(rows, T):
    if rows is None:
        return None
    if rows == "rest":
        return list(range(1, T))
    if rows == "first":
        return [0]
    return list(rows)


def fake_quantize(a, qmax, rows=None, group=32, outliers=None, top=0, shift=0):
    """a: (T, n) float32, every row's groups rounded as quantize_x does. rows: only these rows (the others as they are; "first"
    and "rest": the first position of a window, and the others). outliers: channels (static) and top: how many of the
    largest of a row (dynamic) that are taken out of the groups and kept as they are (the way T92's outlier channels are
    multiplied apart). shift: the groups begin that many values on (the same rounding at another place)."""
    a = np.asarray(a, dtype=np.float32)
    T, n = a.shape
    original = a
    if shift:
        a = np.roll(a, -shift, axis=1)
    work = a
    kept = None
    if outliers is not None and len(outliers):
        kept = np.zeros(a.shape, dtype=bool)
        kept[:, [(c - shift) % n for c in outliers]] = True
    if top:
        biggest = np.argsort(-np.abs(a), axis=1)[:, :top]
        dynamic = np.zeros(a.shape, dtype=bool)
        np.put_along_axis(dynamic, biggest, True, axis=1)
        kept = dynamic if kept is None else kept | dynamic
    if kept is not None:
        work = np.where(kept, np.float32(0), a)
    g = work.reshape(T, n // group, group)
    amax = np.abs(g).max(axis=2, keepdims=True)
    scale = amax / np.float32(qmax)
    inv = np.where(scale > 0, np.float32(1.0) / np.where(scale > 0, scale, 1), np.float32(0))
    out = (np.rint(g * inv) * scale).reshape(T, n).astype(np.float32)
    if kept is not None:
        out = np.where(kept, a, out)
    if shift:
        out = np.roll(out, shift, axis=1)
    picked = chosen_rows(rows, T)
    if picked is not None:
        keep = np.ones(T, dtype=bool)
        keep[picked] = False
        out[keep] = original[keep]
    return out


def rms_rows(x, w, eps):
    ms = (x * x).mean(axis=1, keepdims=True, dtype=np.float32)
    return (w * (x / np.sqrt(ms + np.float32(eps)))).astype(np.float32)


def head_norm_rows(x, w, heads, eps):
    h = x.reshape(x.shape[0], heads, -1)
    ms = (h * h).mean(axis=2, keepdims=True, dtype=np.float32)
    return (w * h / np.sqrt(ms + np.float32(eps))).astype(np.float32)


def softmax_rows(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


class Probe:
    def __init__(self, llama):
        self.l = llama
        self.hs = llama.head_size
        self.heads, self.kv = llama.n_heads, llama.n_kv_heads
        angles = np.arange(llama.seq_len)[:, None] * rope_frequencies(self.hs, 1e6)
        self.cos, self.sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        self.taps = llama.convolution["taps"]
        self.eps = {"op": 1e-5, "ffn": 1e-5, "q": 1e-5, "k": 1e-5, "final": 1e-5}

    def rope_rows(self, x):
        T, h, hs = x.shape
        c, s = self.cos[:T, None, : hs // 2], self.sin[:T, None, : hs // 2]
        pairs = x.reshape(T, h, hs // 2, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        out = np.empty_like(pairs)
        out[..., 0] = x0 * c - x1 * s
        out[..., 1] = x0 * s + x1 * c
        return out.reshape(T, h, hs)

    def window(self, ids, quant=None, record=None):
        """logits (T, vocab). quant: {role: dict(qmax=, rows=, layers=, group=, outliers=, top=, shift=, noise=, seed=)}"""
        L = self.l
        quant = quant or {}
        T = len(ids)

        def Q(a, role, layer):
            spec = quant.get(role)
            if spec is None:
                return a
            if spec.get("layers") is not None and layer not in spec["layers"]:
                return a
            if spec.get("noise"):
                rng = np.random.default_rng([spec.get("seed", 0), 99 if layer is None else layer, zlib.crc32(role.encode())])
                return (a * (1.0 + spec["noise"] * rng.standard_normal(a.shape))).astype(np.float32)
            outliers = spec.get("outliers")
            if isinstance(outliers, dict):
                outliers = outliers.get(layer)
            return fake_quantize(a, spec["qmax"], spec.get("rows"), spec.get("group", 32), outliers, spec.get("top", 0), spec.get("shift", 0))

        x = np.asarray(L.token_embedding_table[np.asarray(ids)], dtype=np.float32)
        for l, (short, a) in enumerate(L.slots):
            xb = rms_rows(x, L.rms_att_weight[l], self.eps["op"])
            if record is not None:
                record.setdefault("xb", {})[l] = xb
            if not short:
                xin = Q(xb, "qkv", l)
                q = (Q(xb, "q_in", l) if "q_in" in quant else xin) @ L.wq[a].T
                k = (Q(xb, "k_in", l) if "k_in" in quant else xin) @ L.wk[a].T
                v = (Q(xb, "v_in", l) if "v_in" in quant else xin) @ L.wv[a].T
                if record is not None:
                    record.setdefault("q_raw", {})[l] = q
                    record.setdefault("k_raw", {})[l] = k
                    record.setdefault("v_raw", {})[l] = v
                q = self.rope_rows(head_norm_rows(q, L.q_norm[a], self.heads, self.eps["q"]))
                k = self.rope_rows(head_norm_rows(k, L.k_norm[a], self.kv, self.eps["k"]))
                v = v.reshape(T, self.kv, self.hs)
                mul = self.heads // self.kv
                kk = np.repeat(k, mul, axis=1).transpose(1, 2, 0)  # (heads, head, T)
                vv = np.repeat(v, mul, axis=1).transpose(1, 0, 2)  # (heads, T, head)
                scores = (q.transpose(1, 0, 2) @ kk) / np.float32(math.sqrt(self.hs))
                scores[:, np.triu(np.ones((T, T), dtype=bool), 1)] = -np.inf
                att = softmax_rows(scores).astype(np.float32)
                if record is not None:
                    record.setdefault("att", {})[l] = att
                    record.setdefault("k_final", {})[l] = k
                    record.setdefault("v_final", {})[l] = v
                out = Q((att @ vv).transpose(1, 0, 2).reshape(T, -1), "o", l) @ L.wo[a].T
            else:
                mixed = Q(xb, "win", l) @ L.win[a].T
                B, C, z = mixed[:, :L.dim], mixed[:, L.dim:2 * L.dim], mixed[:, 2 * L.dim:]
                h = B * z
                conv = np.zeros_like(h)
                for j in range(self.taps):
                    shift = self.taps - 1 - j
                    if shift == 0:
                        conv += L.conv[a][j] * h
                    else:
                        conv[shift:] += L.conv[a][j] * h[:-shift]
                out = Q((C * conv).astype(np.float32), "wout", l) @ L.wout[a].T
            x = (x + out).astype(np.float32)
            if record is not None:
                record.setdefault("x", {})[l] = x
            xn = rms_rows(x, L.rms_ffn_weight[l], self.eps["ffn"])
            xin = Q(xn, "gate_up", l)
            g, u = xin @ L.w1[l].T, xin @ L.w3[l].T
            x = (x + Q((g / (np.float32(1.0) + np.exp(-g)) * u).astype(np.float32), "down", l) @ L.w2[l].T).astype(np.float32)
        return Q(rms_rows(x, L.rms_final_weight, self.eps["final"]), "cls", None) @ L.wcls.T


def log_softmax(logits):
    z = logits.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    return (z - np.log(np.exp(z).sum(axis=1, keepdims=True))).astype(np.float32)


def windows_of(tokens, bos, window=512):
    return [[bos] + tokens[start:start + window - 1] for start in range(0, len(tokens), window - 1)]


BUCKETS = [("0", 0, 1), ("1-8", 1, 9), ("9-32", 9, 33), ("33-128", 33, 129), ("129+", 129, 10 ** 9)]


class Runner:
    def __init__(self, probe, windows):
        self.probe, self.windows = probe, windows
        self.count = sum(len(w) - 1 for w in windows)
        began = time.time()
        self.base_lp = [log_softmax(probe.window(ids)) for ids in windows]
        self.base_target = np.concatenate([lp[np.arange(len(ids) - 1), np.asarray(ids[1:])] for lp, ids in zip(self.base_lp, windows)])
        self.base = float(-self.base_target.mean())
        say(experiment="base", ppl=math.exp(self.base), tokens=self.count, seconds=round(time.time() - began, 1))

    def run(self, name, quant=None, setup=None):
        began = time.time()
        if setup:
            setup(self.probe)
        lps = [log_softmax(self.probe.window(ids, quant)) for ids in self.windows]
        target = np.concatenate([lp[np.arange(len(ids) - 1), np.asarray(ids[1:])] for lp, ids in zip(lps, self.windows)])
        kls = np.concatenate([(np.exp(b) * (b - lp)).sum(axis=1)[:-1] for b, lp in zip(self.base_lp, lps)])
        diff = max(float(np.abs(b - lp).max()) for b, lp in zip(self.base_lp, lps))
        within = np.concatenate([np.arange(len(ids) - 1) for ids in self.windows])
        delta = target - self.base_target  # the change of every position's log probability of the token that came
        nll = float(-target.mean())
        row = {"variant": name, "ppl": math.exp(nll), "change_vs_base_pct": 100 * (math.exp(nll - self.base) - 1),
               "kl": float(kls.mean()), "max_logprob_diff": diff,
               # how common to all positions the change is: its mean over its spread / sqrt(positions) (independent noise: about 1)
               "coherence_z": float(delta.mean() / (delta.std() / math.sqrt(len(delta)) + 1e-12)), "mean_dlogp": float(delta.mean()),
               "std_dlogp": float(delta.std()), "seconds": round(time.time() - began, 1)}
        for label, low, high in BUCKETS:
            picked = (within >= low) & (within < high)
            row[f"kl@{label}"] = float(kls[picked].mean()) if picked.any() else None
        say(**row)
        return row


def quant_all(qmax, **more):
    return {role: dict(qmax=qmax, **more) for role in ROLES}


def main():
    prefix, text_file, count, *experiments = sys.argv[1:]
    llama = load(prefix)
    tokens = llama.tokenizer.encode(Path(text_file).read_text())[:int(count)]
    windows = windows_of(tokens, llama.bos)
    probe = Probe(llama)
    say(experiment="start", windows=[len(w) for w in windows], layers=llama.convolution["layers"], first=windows[0][:6])
    run = Runner(probe, windows)
    attention = [l for l, (short, _) in enumerate(llama.slots) if not short]
    for experiment in experiments:
        say(experiment=experiment, begins=True)
        if experiment == "all":
            for bits in (8, 7):
                run.run(f"{bits}-bit, every input", quant_all(qmax_of(bits)))
        elif experiment == "roles":
            for bits in (8, 7):
                for role in ROLES:
                    run.run(f"{bits}-bit, {role} only", {role: dict(qmax=qmax_of(bits))})
        elif experiment == "leave":
            for bits in (8, 7):
                for role in ROLES:
                    run.run(f"{bits}-bit, every input but {role}", {r: dict(qmax=qmax_of(bits)) for r in ROLES if r != role})
        elif experiment == "positions":
            for bits in (8, 7):
                qmax = qmax_of(bits)
                run.run(f"{bits}-bit, qkv, position 0 only", {"qkv": dict(qmax=qmax, rows="first")})
                run.run(f"{bits}-bit, qkv, positions 1.. only", {"qkv": dict(qmax=qmax, rows="rest")})
                run.run(f"{bits}-bit, every input, position 0 only", quant_all(qmax, rows="first"))
                run.run(f"{bits}-bit, every input, positions 1.. only", quant_all(qmax, rows="rest"))
        elif experiment == "layers":
            for layer in attention:
                run.run(f"8-bit, qkv, position 0 only, layer {layer} only", {"qkv": dict(qmax=127, rows="first", layers={layer})})
            for layer in attention:
                run.run(f"8-bit, qkv, every position, layer {layer} only", {"qkv": dict(qmax=127, layers={layer})})
        elif experiment == "split":
            for role in ("q_in", "k_in", "v_in"):
                run.run(f"8-bit, {role}, every position", {role: dict(qmax=127)})
                run.run(f"8-bit, {role}, position 0 only", {role: dict(qmax=127, rows="first")})
        elif experiment == "bos":
            # what the first position's rounding alone does, by the bits it is rounded to
            for bits in (5, 6, 7, 8, 9, 10, 11, 12, 14, 16):
                run.run(f"{bits}-bit, qkv, position 0 only", {"qkv": dict(qmax=qmax_of(bits), rows="first")})
            for bits in (6, 7, 8, 9, 10, 12):
                run.run(f"{bits}-bit, every input, position 0 only", quant_all(qmax_of(bits), rows="first"))
            for bits in (7, 8, 10):
                run.run(f"{bits}-bit, every input but the first position's", quant_all(qmax_of(bits), rows="rest"))
        elif experiment == "bosparts":
            for bits in (8, 7):
                for layer in attention:
                    for part in ("q_in", "k_in", "v_in"):
                        run.run(f"{bits}-bit, {part}, position 0 only, layer {layer}", {part: dict(qmax=qmax_of(bits), rows="first", layers={layer})})
        elif experiment == "shifts":
            # the same rounding with the groups beginning elsewhere: other draws of the same width
            for bits in (8, 7):
                for shift in (0, 3, 7, 11, 16, 21, 25, 29):
                    run.run(f"{bits}-bit, every input, groups shifted by {shift}", quant_all(qmax_of(bits), shift=shift))
            for bits in (8, 7):
                for shift in (0, 3, 7, 11, 16, 21, 25, 29):
                    run.run(f"{bits}-bit, qkv, position 0 only, groups shifted by {shift}", {"qkv": dict(qmax=qmax_of(bits), rows="first", shift=shift)})
        elif experiment == "bits":
            for bits in (6, 7, 8, 9, 10, 12, 14, 16):
                run.run(f"{bits}-bit, every input", quant_all(qmax_of(bits)))
        elif experiment == "noise":
            for noise in (1e-4, 3e-4, 1e-3, 3e-3):
                for seed in (1, 2, 3, 4):
                    run.run(f"relative noise {noise} on every input, seed {seed}", {r: dict(noise=noise, seed=seed) for r in ROLES})
            for noise in (1e-3, 1e-2):
                for seed in (1, 2, 3, 4):
                    run.run(f"relative noise {noise} on qkv, seed {seed}", {"qkv": dict(noise=noise, seed=seed)})
        elif experiment == "eps":
            for kind in ("op", "ffn", "q", "k", "final"):
                for value in (1e-6, 1e-4):
                    def setup(p, kind=kind, value=value):
                        p.eps = {"op": 1e-5, "ffn": 1e-5, "q": 1e-5, "k": 1e-5, "final": 1e-5}
                        p.eps[kind] = value
                    run.run(f"epsilon of the {kind} norm {value}", None, setup)
            probe.eps = {"op": 1e-5, "ffn": 1e-5, "q": 1e-5, "k": 1e-5, "final": 1e-5}
        elif experiment == "mitigate":
            for bits in (8, 7):
                qmax = qmax_of(bits)
                every = quant_all(qmax)
                run.run(f"{bits}-bit, every input but qkv", {r: dict(qmax=qmax) for r in ROLES if r != "qkv"})
                run.run(f"{bits}-bit, every input, the first position not rounded", quant_all(qmax, rows="rest"))
                run.run(f"{bits}-bit, every input, the qkv input of the first position not rounded", {**every, "qkv": dict(qmax=qmax, rows="rest")})
                static = {l: np.argsort(-np.abs(llama.rms_att_weight[l]))[:8] for l in attention}
                for top in (1, 2, 4, 8):
                    run.run(f"{bits}-bit, every input, the {top} largest channels of a row taken out of qkv's groups", {**every, "qkv": dict(qmax=qmax, top=top)})
                run.run(f"{bits}-bit, every input, the 8 largest norm weights taken out of qkv's groups", {**every, "qkv": dict(qmax=qmax, outliers=static)})
                for group in (16, 8, 4):
                    run.run(f"{bits}-bit, every input, groups of {group} for qkv", {**every, "qkv": dict(qmax=qmax, group=group)})
        elif experiment == "stats":
            rec = {}
            probe.window(windows[0], None, rec)
            x0 = np.asarray(llama.token_embedding_table[np.asarray(windows[0])], dtype=np.float32)
            say(experiment="stats", what="embedding rows", bos_rms=float(np.sqrt((x0[0] ** 2).mean())), others_rms_median=float(np.median(np.sqrt((x0[1:] ** 2).mean(axis=1)))))
            for l in attention:
                xb, qr, kr = rec["xb"][l], rec["q_raw"][l], rec["k_raw"][l]
                peak = np.abs(xb).max(axis=1) / np.sqrt((xb * xb).mean(axis=1))
                g = xb.reshape(xb.shape[0], -1, 32)
                amax = np.abs(g).max(axis=2, keepdims=True)
                zero8 = (np.rint(g / np.where(amax > 0, amax / 127, 1)) == 0).mean(axis=(1, 2))
                zero7 = (np.rint(g / np.where(amax > 0, amax / 63, 1)) == 0).mean(axis=(1, 2))
                qh, kh = qr.reshape(qr.shape[0], probe.heads, -1), kr.reshape(kr.shape[0], probe.kv, -1)
                qms, kms = (qh * qh).mean(axis=2), (kh * kh).mean(axis=2)
                w = np.abs(llama.rms_att_weight[l])
                att = rec["att"][l]
                top = np.argsort(-np.abs(xb[0]))[:3]
                say(experiment="stats", layer=l, norm_weight_max_over_median=float(w.max() / np.median(w)),
                    peak_over_rms_pos0=float(peak[0]), peak_over_rms_rest_mean=float(peak[1:].mean()), peak_over_rms_rest_max=float(peak[1:].max()),
                    biggest_channels_pos0=[int(c) for c in top], biggest_values_pos0=[float(xb[0][c]) for c in top],
                    zero8_pos0=float(zero8[0]), zero8_rest=float(zero8[1:].mean()), zero7_pos0=float(zero7[0]), zero7_rest=float(zero7[1:].mean()),
                    q_ms_min=float(qms.min()), q_ms_median=float(np.median(qms)), k_ms_min=float(kms.min()), k_ms_median=float(np.median(kms)),
                    q_heads_ms_below_10eps=float((qms < 1e-4).mean()), k_heads_ms_below_10eps=float((kms < 1e-4).mean()),
                    q_ms_pos0_min=float(qms[0].min()), k_ms_pos0_min=float(kms[0].min()),
                    attention_on_pos0_mean=float(att[:, 1:, 0].mean()), attention_on_pos0_max=float(att[:, 1:, 0].max()),
                    v_pos0_rms=float(np.sqrt((rec["v_raw"][l][0] ** 2).mean())), v_rest_rms=float(np.sqrt((rec["v_raw"][l][1:] ** 2).mean())),
                    k_pos0_rms_raw=float(np.sqrt(kms[0].mean())), k_rest_rms_raw=float(np.sqrt(kms[1:].mean())))
            for l in range(len(llama.slots)):
                x = rec["x"][l]
                say(experiment="stats", residual_after_layer=l, pos0_rms=float(np.sqrt((x[0] ** 2).mean())), pos0_peak=float(np.abs(x[0]).max()),
                    rest_rms_median=float(np.median(np.sqrt((x[1:] ** 2).mean(axis=1)))))
        else:
            say(experiment=experiment, unknown=True)


if __name__ == "__main__":
    main()
