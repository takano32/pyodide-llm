# unfold_27b.py
# T237's review: Ternary Bonsai 2 27B's rotated basis, and the small tensors llama.cpp changed on the way to the GGUF, against
# the model it was made from (Qwen/Qwen3.8-27B, 55.6 GB of bfloat16, neither ternary nor rotated), without the fork and without
# the 5.95 GB: a few hundred rows of each kind of tensor by Range requests (about 30 MB all told, 90 s).
#
# The file holds W R^-1 for every matrix, ternary, and the rows of the embedding as R e (R = H S: the signs of the input's
# values, then the normalized Walsh-Hadamard transform of every block of 1024). So the engine's R^-1 of a stored row,
# S (H row), is a row of W again, up to what the ternary quantization and the training since cost: the row of the original
# model, to which it is highly correlated (and R^-1 of any other reading of the row is not: a correlation near 0). What the
# engine's NumPy forward pass and the fork agree on (tests/reference_27b.py) is that both compute what the file says; this
# says the file is what the engine reads it as, against a third thing that is neither: the signs are those of the width
# (5120, 6144, 17408) the file's own metadata names, to every one of the 28,672 (the sign of a column the sampled rows say is
# the other way round is counted, by a t-statistic), the order is signs after the transform, the blocks are 1024 along the
# input, the output matrix of a linear layer is in Hugging Face's order of value heads (prism.hadamard.gdn_v_grouped), q and
# its gate and k are not turned, and the rows of the embedding and the classifier are rows of the original's.
# The small tensors are not rotated and were not quantized, but Prism's training went on over every parameter, so they are the
# original's to a few percent and no closer to the bit: each is held to be that near as the engine reads it (the norms with
# the 1 the model adds, A_log as -exp(A_log), every tensor of the value heads moved as llama.cpp moves them, a head's norm
# unturned) and at least twice as far the other way.
#
#   python tests/unfold_27b.py [--head <the GGUF's first 12 MiB, fetched into that file if it is not there>]
#
# Nothing here needs a GPU, a fork or more of the 27B than a few megabytes. Exit 1 where a reading fails.
import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
import llama2_convert  # noqa: E402
import llama2_numpy  # noqa: E402
from reference_27b import grouped, widen_ptq1_0  # noqa: E402
from fetching import ranged  # noqa: E402

ORIGINAL = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/"
BONSAI = ("https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/b072e1d3b35a0a630cece372c2127528e0994386/"
          "Ternary-Bonsai-2-27B-PTQ1_0.gguf")
HEAD_BYTES = 12 * 1024 * 1024
PREFIX = "model.language_model."
KEY_HEADS, VALUE_HEADS, VALUE_DIM = 16, 48, 128


def fetch(url, start, length, tries=5):
    """length bytes of a file from start, by a Range request (the redirect to the CDN is followed)."""
    try:
        return ranged(url, start, length, tries=tries, wait=2)  # (tests/fetching.py, T357)
    except OSError as error:
        raise SystemExit(f"unfold: {url} from {start}: {error}")


class Original:
    """The original model's tensors by name: the index, the header of the shard that holds one, and rows of it."""

    def __init__(self):
        self.weights = json.loads(fetch_text(ORIGINAL + "model.safetensors.index.json"))["weight_map"]
        self.headers = {}

    def info(self, name):
        shard = self.weights[name]
        if shard not in self.headers:
            size = struct.unpack("<Q", fetch(ORIGINAL + shard, 0, 8))[0]
            self.headers[shard] = (json.loads(fetch(ORIGINAL + shard, 8, size)), 8 + size)
        header, base = self.headers[shard]
        return shard, header[name], base

    def rows(self, name, first, count):
        shard, info, base = self.info(name)
        assert info["dtype"] == "BF16", info["dtype"]
        width = info["shape"][-1]
        raw = fetch(ORIGINAL + shard, base + info["data_offsets"][0] + first * width * 2, count * width * 2)
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(count, width), info["shape"]

    def whole(self, name):
        """A small tensor, all of it, as float32."""
        shard, info, base = self.info(name)
        first, last = info["data_offsets"]
        raw = fetch(ORIGINAL + shard, base + first, last - first)
        if info["dtype"] == "BF16":
            return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
        if info["dtype"] == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])
        raise SystemExit(f"unfold: {name} is {info['dtype']}")


def fetch_text(url):
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode()


class Stored:
    """The GGUF's tensors: the table of its head, and rows of a PTQ1_0 matrix by Range requests."""

    def __init__(self, head):
        path = Path(head)
        if not path.exists() or path.stat().st_size < HEAD_BYTES:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(fetch(BONSAI, 0, HEAD_BYTES))
        self.metadata, self.tensors, self.base = llama2_convert.gguf_read(path.read_bytes())

    def rows(self, name, first, count):
        info = self.tensors[name]
        assert info["type"] == 143, info["type"]
        width = info["shape"][-1]
        row = width // 128 * 28
        raw = fetch(BONSAI, self.base + info["offset"] + first * row, count * row)
        return widen_ptq1_0(np.frombuffer(raw, dtype=np.uint8)).reshape(count, width)

    def whole(self, name):
        """A small tensor, all of it, as float32 (F32 or BF16)."""
        info = self.tensors[name]
        count = int(np.prod(info["shape"]))
        if info["type"] == 0:
            raw = fetch(BONSAI, self.base + info["offset"], count * 4)
            return np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])
        assert info["type"] == 30, info["type"]
        raw = fetch(BONSAI, self.base + info["offset"], count * 2)
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])


def correlations(a, b):
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def evidence(unfolded, original):
    """For every input position, how sure the rows are that the sign there is right: the t-statistic of the product of the
    two rows' values there (positive: the same sign)."""
    product = unfolded * original
    spread = product.std(axis=0, ddof=1)
    return product.mean(axis=0) / np.where(spread > 0, spread / np.sqrt(len(product)), np.inf)


def small_tensors(stored, original):
    """The tensors that are no matrix of the rotated basis, and that llama.cpp changed on the way to the GGUF (T236): the norms
    with the 1 the model adds to them (not a linear layer's own), A_log as -exp(A_log), and every tensor over the value heads
    in llama.cpp's order of them (conversion/qwen.py's _reorder_v_heads), against the original, whole, for one linear layer and
    one full one and the model's last norm. They are not the original's to the bit: Prism's training went on over every
    parameter (a norm's weights differ by a few percent). So each is held to the original within a few percent as the
    engine reads it, and to be as far from it as the other reading would be: the value heads left as they are, no 1 added,
    q's norm turned (the engine's own order of a head, which the GGUF does not have). Returns (what, the difference as the
    engine reads it, as the other reading would, the number of values), the differences relative to the original's size."""
    value_order = grouped(KEY_HEADS, VALUE_HEADS, VALUE_DIM)
    head_order = grouped(KEY_HEADS, VALUE_HEADS)
    channels = np.concatenate([np.arange(2 * KEY_HEADS * 128), 2 * KEY_HEADS * 128 + value_order])
    unmoved = np.concatenate([np.arange(2 * KEY_HEADS * 128), 2 * KEY_HEADS * 128 + np.arange(VALUE_HEADS * VALUE_DIM)])
    prefix = PREFIX + "layers."
    results = []

    def relative(mine, theirs):
        return float(np.sqrt(np.mean((mine - theirs) ** 2)) / np.sqrt(np.mean(theirs ** 2)))

    def compare(label, right, wrong, theirs):
        results.append((label, relative(right, theirs), relative(wrong, theirs), theirs.size))

    # a linear layer (the first): -exp(A_log) and the value heads in llama.cpp's order, as they are
    a_log = -np.exp(original.whole(prefix + "0.linear_attn.A_log"))
    ssm_a = stored.whole("blk.0.ssm_a")
    compare("ssm_a is -exp(A_log), the value heads moved", ssm_a[head_order], ssm_a, a_log)
    dt_bias = stored.whole("blk.0.ssm_dt.bias")
    compare("ssm_dt.bias is dt_bias, the value heads moved", dt_bias[head_order], dt_bias, original.whole(prefix + "0.linear_attn.dt_bias"))
    for mine, theirs in (("ssm_alpha", "in_proj_a"), ("ssm_beta", "in_proj_b")):
        gate = stored.whole(f"blk.0.{mine}.weight")
        compare(f"{mine} is {theirs}, the value heads moved", gate[head_order], gate, original.whole(prefix + f"0.linear_attn.{theirs}.weight"))
    taps = stored.whole("blk.0.ssm_conv1d.weight")
    compare("ssm_conv1d is the convolution, the channels of the value heads moved", taps[channels], taps[unmoved],
            original.whole(prefix + "0.linear_attn.conv1d.weight").reshape(10240, -1))
    norm = stored.whole("blk.0.ssm_norm.weight")
    compare("ssm_norm is the norm of a value head, with no 1 added", norm, norm + 1, original.whole(prefix + "0.linear_attn.norm.weight"))
    # the norms with the 1 the model adds
    for label, name, theirs in (("attn_norm is input_layernorm + 1", "blk.0.attn_norm.weight", prefix + "0.input_layernorm.weight"),
                                ("post_attention_norm is post_attention_layernorm + 1", "blk.63.post_attention_norm.weight", prefix + "63.post_attention_layernorm.weight"),
                                ("output_norm is norm + 1", "output_norm.weight", PREFIX + "norm.weight"),
                                ("attn_q_norm is q_norm + 1", "blk.3.attn_q_norm.weight", prefix + "3.self_attn.q_norm.weight"),
                                ("attn_k_norm is k_norm + 1", "blk.3.attn_k_norm.weight", prefix + "3.self_attn.k_norm.weight")):
        mine = stored.whole(name)
        compare(label, mine, mine - 1, original.whole(theirs) + 1)
    # a head's norm in Hugging Face's order of its 256 (llama.cpp does not turn it; the engine does, where RoPE turns)
    turned = llama2_convert.transformed(np.arange(256)[:, None], ("heads", 1, 0, 1, 64), 256).reshape(-1)
    mine = stored.whole("blk.3.attn_q_norm.weight")
    compare("attn_q_norm is in Hugging Face's order of a head", mine, mine[turned], original.whole(prefix + "3.self_attn.q_norm.weight") + 1)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", default=str(HERE.parent / ".tmp" / "27b-head.bin"))
    arguments = parser.parse_args()
    stored, original = Stored(arguments.head), Original()
    basis = llama2_convert.gguf_rotated(stored.metadata, stored.tensors, True)
    block = basis["block"]
    signs = {int(width): (1 - 2 * np.unpackbits(np.frombuffer(bytes.fromhex(bits), dtype=np.uint8))[:int(width)].astype(np.float32))
             for width, bits in basis["signs"].items()}
    print(f"unfold: blocks of {block}, signs for widths {sorted(signs)}")

    value_rows = grouped(KEY_HEADS, VALUE_HEADS, VALUE_DIM)  # Hugging Face's row of the value heads -> the GGUF's
    # (the GGUF's name, the original's, the rows of the original (first, count), the GGUF's first row for them, the
    # column order of the GGUF's rows against the original's (None: the same), what it is)
    heads = 128
    cases = [
        ("blk.3.attn_q.weight", "layers.3.self_attn.q_proj.weight", [(0, 48), (6000, 48)], None, "q and its gate, a full layer"),
        ("blk.3.attn_k.weight", "layers.3.self_attn.k_proj.weight", [(0, 48), (512, 48)], None, "k"),
        ("blk.3.attn_v.weight", "layers.3.self_attn.v_proj.weight", [(0, 48), (512, 48)], None, "v"),
        ("blk.3.attn_output.weight", "layers.3.self_attn.o_proj.weight", [(0, 48), (4000, 48)], None, "o (6144 wide)"),
        ("blk.3.ffn_gate.weight", "layers.3.mlp.gate_proj.weight", [(0, 48), (9000, 48)], None, "FFN gate"),
        ("blk.3.ffn_up.weight", "layers.3.mlp.up_proj.weight", [(0, 48), (9000, 48)], None, "FFN up"),
        ("blk.3.ffn_down.weight", "layers.3.mlp.down_proj.weight", [(0, 48), (3000, 48)], None, "FFN down (17408 wide)"),
        ("blk.60.ffn_down.weight", "layers.60.mlp.down_proj.weight", [(100, 48)], None, "FFN down of a late layer"),
        ("blk.0.attn_qkv.weight", "layers.0.linear_attn.in_proj_qkv.weight", [(0, 48), (2048, 48)], None, "a linear layer's q and k"),
        ("blk.0.attn_qkv.weight", "layers.0.linear_attn.in_proj_qkv.weight", [(4096 + 5 * heads, heads)], "value", "a linear layer's v (a head of 128)"),
        ("blk.0.attn_gate.weight", "layers.0.linear_attn.in_proj_z.weight", [(7 * heads, heads)], "z", "a linear layer's z (a head of 128)"),
        ("blk.0.ssm_out.weight", "layers.0.linear_attn.out_proj.weight", [(0, 48), (3000, 48)], None, "a linear layer's output (6144 wide, Hugging Face's order of heads)"),
        ("token_embd.weight", "embed_tokens.weight", [(0, 48), (100000, 48), (248000, 48)], None, "the embedding's rows"),
        ("output.weight", "lm_head.weight", [(0, 48), (100000, 48)], None, "the classifier"),
    ]
    # more rows of the inputs 6144 and 17408 wide from other layers (a width's signs are one vector for every matrix that reads
    # it: the pooled evidence below is over all of them), for their signs are the least sure of what 96 rows say
    more = [(f"blk.{layer}.attn_output.weight", f"layers.{layer}.self_attn.o_proj.weight", [(500 + 97 * layer, 48)]) for layer in (11, 27, 63)]
    more += [(f"blk.{layer}.ssm_out.weight", f"layers.{layer}.linear_attn.out_proj.weight", [(300 + 131 * layer, 48)]) for layer in (1, 20, 40, 62)]
    more += [(f"blk.{layer}.ffn_down.weight", f"layers.{layer}.mlp.down_proj.weight", [(700 + 53 * layer, 48)]) for layer in (11, 20, 40, 63)]
    pooled = {}
    failed = []
    for gguf_name, name, places, mapping, what in cases:
        original_name = name if name == "lm_head.weight" else PREFIX + name
        right, wrong = [], {}
        columns = None
        for first, count in places:
            theirs, shape = original.rows(original_name, first, count)
            if mapping is None:
                mine = stored.rows(gguf_name, first, count)
            elif mapping == "value":
                head = (first - 4096) // heads
                gguf_first = 4096 + int(value_rows[head * heads])
                mine = stored.rows(gguf_name, gguf_first, count)
            else:  # z: the rows are the value heads alone
                head = first // heads
                mine = stored.rows(gguf_name, int(value_rows[head * heads]), count)
            width = mine.shape[1]
            s = signs[width]
            unfolded = llama2_numpy.unrotate(mine, s, block)  # a matrix's row is W's row R^-1 of it; so is the embedding's
            right.append((unfolded, theirs))
            pooled.setdefault(width, []).append((unfolded, theirs))
            # other readings of the same stored rows
            others = {
                "signs before the transform": llama2_numpy.rotate(mine, s, block),
                "no signs": llama2_numpy.hadamard(mine, block),
                "no transform": mine * s,
                "as stored": mine,
                "blocks of 512": llama2_numpy.unrotate(mine, s, 512),
                "blocks of 2048": llama2_numpy.unrotate(mine, s, 2048) if width % 2048 == 0 else None,
            }
            if width == 6144:
                others["5120's signs on the first 5120"] = np.concatenate(
                    [llama2_numpy.unrotate(mine, np.concatenate([signs[5120], signs[6144][5120:]]), block)], axis=0)
            if gguf_name.endswith("ssm_out.weight"):
                # the columns in llama.cpp's own order of the value heads: the other order
                order = grouped(KEY_HEADS, VALUE_HEADS, VALUE_DIM)
                others["the value heads in llama.cpp's order"] = unfolded[:, order]
            for label, value in others.items():
                if value is not None:
                    wrong.setdefault(label, []).append((value, theirs))
        unfolded = np.concatenate([a for a, _ in right])
        theirs = np.concatenate([b for _, b in right])
        rho = correlations(unfolded, theirs)
        relative = np.linalg.norm(unfolded - theirs, axis=1) / np.linalg.norm(theirs, axis=1)
        per_block = correlations(unfolded.reshape(-1, block), theirs.reshape(-1, block)).reshape(len(unfolded), -1)
        t = evidence(unfolded, theirs)
        line = (f"unfold: {what}: {len(unfolded)} rows of {unfolded.shape[1]}: correlation {rho.mean():.3f} (lowest {rho.min():.3f}), "
                f"relative difference {relative.mean():.3f}, the lowest block's correlation {per_block.min():.3f}")
        if len(unfolded) >= 40:
            line += f"; {int((t < 0).sum())} of {t.size} signs the wrong way by these rows (lowest t {t.min():.1f}, {int((t < 3).sum())} under 3)"
        print(line, flush=True)
        for label, parts in wrong.items():
            value = np.concatenate([a for a, _ in parts])
            other = np.concatenate([b for _, b in parts])
            print(f"unfold:     {label}: correlation {correlations(value, other).mean():+.3f}", flush=True)
            # (the transform of smaller or larger blocks is a third or a half of the way to the right one: 0.707 of it)
            if correlations(value, other).mean() > 0.85 * rho.mean():
                failed.append(f"{what} read as {label}")
        if rho.mean() < 0.5 or per_block.min() < 0.2:
            failed.append(what)
        if len(unfolded) >= 40 and (t < 0).sum() > 0:
            failed.append(f"{what}: signs the wrong way")
    for gguf_name, name, places in more:
        for first, count in places:
            theirs, _ = original.rows(PREFIX + name, first, count)
            mine = stored.rows(gguf_name, first, count)
            pooled.setdefault(mine.shape[1], []).append((llama2_numpy.unrotate(mine, signs[mine.shape[1]], block), theirs))
    for width, parts in sorted(pooled.items()):
        t = evidence(np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]))
        rows = sum(len(a) for a, _ in parts)
        print(f"unfold: the {width} signs of an input {width} wide, all {rows} rows of {len(parts)} runs of rows pooled: {int((t < 0).sum())} the "
              f"wrong way, the lowest t {t.min():.1f}, {int((t < 3).sum())} under 3", flush=True)
        if (t < 0).sum() > 0:
            failed.append(f"signs of {width} the wrong way")
    for label, right, wrong, count in small_tensors(stored, original):
        print(f"unfold: {label}: {right:.3f} of the original's size away as the engine reads it, {wrong:.3f} the other way ({count} values)", flush=True)
        if not (right <= 0.35 and wrong >= 2 * right):
            failed.append(label)
    if failed:
        print(f"unfold: FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("unfold: ok: R^-1 of the stored rows are the original's, and no other reading of them is; the small tensors are the original's, "
          "moved and added to as llama.cpp does")


if __name__ == "__main__":
    main()
