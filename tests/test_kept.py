# T.363.3: a GGUF whose matrices are in llama.cpp's blocks (the K- and i-quantized types), converted with the blocks left
# the bytes they are (Conversion(blocks=True), convert/stream.py's Kept): what the form then says ("kinds", a type to
# each layer; "tiled"), where the bytes go (the rows llama.cpp and the plan move, moved as bytes; a linear-attention
# layer's output matrix left as the file has it, its input put in the file's order by the engine), the codebooks, and
# the engine in NumPy on the checkpoint. The reference is the conversion that widens the same file to float32, which
# tests/test_superblocks.py holds to a safetensors file of the same values: the same rows to the bit, and the same
# logits but for the order one matrix's sums are taken in. On a small Qwen3.5 of Underdog Saluki 27B's shape (the ten
# types over layers and roles, three value heads to a key head, a classifier of its own, Q8_0 gates), and on the real
# model's sizes without its weights (tests/saluki_form.py).
import json

import numpy as np
import pytest
import saluki_form
from conducting import converted, listed
from made_up import hugging_face, synthetic_weights
from test_gguf import QWEN35_LAYER, QWEN35_NAMES, QWEN35_WHOLE, gguf_file, q8_0_blocks, qwen35_gguf, unigram
from test_superblocks import SALUKI, SLOW, as_saluki, blocks_of

import llama2_convert
from convert import grids
from convert.stream import kept_form, order, stays
from llama2_convert import SOURCES, Conversion, Kept, Stream, Writer, transformed, unsplit, untiled, unturned
from llama2_numpy import (BLOCKS, CODEBOOKS, FORM, Llama, checkpoint_dtype, external_tensors, file_size, placed,
                          tensor_rows)

TOKENS = (3, 9, 27, 5, 1, 30)
# the engine's rows of a Qwen3.5 that a quantized file quantizes, and the GGUF's tensor each is made of (written out a
# second time: wq and wg are one tensor there)
MATRICES = {"token_embedding_table": "token_embd", "wcls": "output", "wq": "attn_q", "wg": "attn_q", "wk": "attn_k",
            "wv": "attn_v", "wo": "attn_output", "wqkv": "attn_qkv", "wz": "attn_gate", "wout": "ssm_out",
            "w1": "ffn_gate", "w2": "ffn_down", "w3": "ffn_up"}
STATEFUL = ("wqkv", "wz", "wout")


class Model:
    """The small GGUF, its original's files, and its conversions."""

    def __init__(self, matrices=None, change=None, **shape):
        self.seen = {}
        self.config, self.file, self.same = qwen35_gguf(matrices=matrices or as_saluki(self.seen), change=change,
                                                        **{**SALUKI, **shape})
        self.text = self.config["text_config"]
        self.vocabulary = unigram(self.text["vocab_size"])
        header, self.base = llama2_convert.gguf_weights(self.file, json.dumps(self.config))
        self.header_text, self.header = header, json.loads(header)

    def conversion(self, blocks=True, dtype="int8", **more):
        return Conversion(self.header_text, self.base, json.dumps(self.config), self.vocabulary, "tokenizer.json",
                          dtype=dtype, max_seq_len=1 << 20, start=self.base, blocks=blocks, **more)

    def converted(self, blocks=True, dtype="int8", cuts=None, **more):
        """cuts: where the pieces fed end, positions in the file (4096 bytes a piece without)."""
        conversion = self.conversion(blocks, dtype, **more)
        cuts = sorted({self.base, len(self.file), *(cuts or range(self.base, len(self.file), 4096))})
        done = 0.0
        for begin, end in zip(cuts, cuts[1:]):
            now = conversion.feed(self.file[begin:end])
            assert done <= now <= 1.0
            done = now
        conversion.finish()
        assert done == 1.0
        return conversion

    def types(self, row):
        """The GGUF's types of the tensors a row is made of, a layer after the other (read off the header here)."""
        of = lambda name: self.header[name]["dtype"]
        if row in ("token_embedding_table", "wcls"):
            return [of("model.embed_tokens.weight" if row == "token_embedding_table" else "lm_head.weight")]
        hugging_face = {gguf: name for name, gguf in QWEN35_LAYER.items()}[MATRICES[row]]
        layers = range(self.text["num_hidden_layers"])
        linear = [(layer + 1) % self.text["full_attention_interval"] != 0 for layer in layers]
        which = layers if row in ("w1", "w2", "w3") else [l for l in layers if linear[l] == (row in STATEFUL)]
        return [of(f"model.layers.{layer}.{hugging_face}.weight") for layer in which]


@pytest.fixture(scope="module")
def saluki():
    model = Model()
    model.kept, model.wide = model.converted(), model.converted(blocks=False, dtype="float32")
    model.read = lambda conversion, **more: Llama(bytes(conversion.checkpoint), bytes(conversion.tokenizer),
                                                  **{**conversion.options, **more})
    return model


def logits(model, tokens=TOKENS):
    return np.stack([model.forward(token, position) for position, token in enumerate(tokens)])


# ---- what the form says
def test_the_form_says_the_type_of_every_matrix_a_layer_after_the_other(saluki):
    """Every row a quantized file quantizes is kept, with the types the GGUF has for its tensors: a stack's by its
    layers. The file's dtype says how the rest is stored, and nothing else of the options is another's."""
    kinds = saluki.kept.options["kinds"]
    assert set(kinds) == set(MATRICES)
    for row in MATRICES:
        types = saluki.types(row)
        assert kinds[row] == (types[0] if len(set(types)) == 1 else types), row
    assert {kind for row in kinds.values() for kind in ([row] if isinstance(row, str) else row)} == set(SLOW) == set(BLOCKS)
    assert sum(isinstance(row, list) for row in kinds.values()) >= 9, "no stack has layers of several types: this shows little"
    assert kinds["wq"] == kinds["wg"], "q and its gate are one tensor of the file"
    assert saluki.kept.options["dtype"] == "int8" and saluki.kept.options["tiled"] is True
    today = saluki.converted(blocks=False).options
    assert {key: value for key, value in saluki.kept.options.items() if key not in ("kinds", "tiled")} == today
    assert "kinds" not in today and "tiled" not in today
    assert bytes(saluki.kept.tokenizer) == bytes(saluki.wide.tokenizer)
    assert json.loads(json.dumps(saluki.kept.options)) == saluki.kept.options, "the options are no JSON"


def test_the_checkpoint_is_the_size_its_header_and_its_form_say(saluki):
    header, form = saluki.kept.stream.header, saluki.kept.options
    size = len(saluki.kept.checkpoint)
    assert size == file_size(tensor_rows(header, form), "int8") == saluki.kept.stream.size("int8")
    assert checkpoint_dtype(header, size, form) == "int8"
    blocks = sum(len(self_bytes) for self_bytes in block_tensors(saluki).values())
    # the header, the blocks as the file has them, the gates' two small matrices and the vectors in float32, the codebooks
    floats = sum(info["data_offsets"][1] - info["data_offsets"][0] for info in saluki.header.values()
                 if isinstance(info, dict) and info.get("dtype") == "F32")
    gates = sum(4 * int(np.prod(info["shape"])) for info in saluki.header.values() if isinstance(info, dict) and info.get("dtype") == "Q8_0")
    assert size == 28 + blocks + gates + floats + CODEBOOKS


def block_tensors(saluki):
    """{the Hugging Face name: the bytes} of the GGUF's tensors in blocks, cut out of the file here."""
    return {name: saluki.file[saluki.base + info["data_offsets"][0]:saluki.base + info["data_offsets"][1]]
            for name, info in saluki.header.items() if name != "__metadata__" and info["dtype"] in BLOCKS}


def test_a_tensor_that_does_not_move_is_in_the_checkpoint_byte_for_byte(saluki):
    """The embedding, the classifier, v, the two output matrices and the FFN's three: the file's bytes at the row's
    place (a layer's, in a stack). The output matrix of a linear-attention layer among them: its value heads are tiled
    along the columns in the file, and stay so."""
    tensors, checkpoint = block_tensors(saluki), bytes(saluki.kept.checkpoint)
    places = {place.row.name: place for place in placed(tensor_rows(saluki.kept.stream.header, saluki.kept.options), "int8")}
    names = {"token_embedding_table": ["model.embed_tokens.weight"], "wcls": ["lm_head.weight"]}
    layers = range(saluki.text["num_hidden_layers"])
    linear = [(layer + 1) % 4 != 0 for layer in layers]
    for row, part in (("wv", "self_attn.v_proj"), ("wo", "self_attn.o_proj"), ("wout", "linear_attn.out_proj"),
                      ("w1", "mlp.gate_proj"), ("w2", "mlp.down_proj"), ("w3", "mlp.up_proj")):
        which = layers if row[0] == "w" and row[1:].isdigit() else [l for l in layers if linear[l] == (row == "wout")]
        names[row] = [f"model.layers.{layer}.{part}.weight" for layer in which]
    seen = 0
    for row, of in names.items():
        place = places[row]
        parts = place.parts if place.layers else [place]
        at = [part.offset for part in parts] if place.layers else None
        for layer, name in enumerate(of):
            raw = tensors[name]
            begin = at[layer] if at else place.offset + layer * len(raw)
            assert checkpoint[begin:begin + len(raw)] == raw, f"{row}, layer {layer}"
            seen += 1
    assert seen == 2 + 2 + 2 + 6 + 3 * 8
    assert any("tiled" in saluki.header[name] and saluki.header[name]["tiled"][-1] == 1 for name in names["wout"])


def test_the_codebooks_are_the_last_row_and_the_grids(saluki):
    assert bytes(saluki.kept.checkpoint)[-CODEBOOKS:] == grids.codebooks()[0]
    assert saluki.read(saluki.kept).codebooks.tobytes() == grids.codebooks()[0]


# ---- what the engine reads of it
def test_every_row_reads_as_the_widened_conversions_row_to_the_bit(saluki):
    """The conversion that widens the file to float32 puts the value heads back, takes q and its gate apart and turns
    the pairs of q and k as it always did. Kept, the same rows are the same values, moved as bytes; and the output
    matrix of a linear-attention layer is the widened one with its columns in the file's order."""
    kept, wide = saluki.read(saluki.kept), saluki.read(saluki.wide)
    for row in MATRICES:
        if row == "token_embedding_table":
            ours, theirs = (np.stack([model.embedding(token) for token in range(40)]) for model in (kept, wide))
        else:
            ours, theirs = getattr(kept, row), getattr(wide, row)
        assert ours.dtype == np.float32 and ours.shape == theirs.shape
        if row == "wout":
            assert not np.array_equal(ours, theirs), "three value heads to a key head are tiled along the columns"
            keys, per, size = SALUKI["key_heads"], SALUKI["value_heads"] // SALUKI["key_heads"], SALUKI["value_dim"]
            # by hand: the file's column (j * keys + h) * size + c is Hugging Face's value head h * per + j
            columns = [(h * per + j) * size + c for j in range(per) for h in range(keys) for c in range(size)]
            theirs = theirs[:, :, columns]
        assert np.array_equal(ours.view(np.uint32), theirs.view(np.uint32)), row
    for row in ("wb", "wa", "conv", "dt_bias", "decay", "delta_norm", "rms_att_weight", "rms_ffn_weight", "rms_final_weight",
                "q_norm", "k_norm"):
        assert np.array_equal(getattr(kept, row), getattr(wide, row)), row
    assert kept.wb.dtype == np.float32, "the gates' small matrices are float32, as in every quantized file"


def test_its_logits_are_the_widened_conversions_but_for_the_order_of_one_matrixs_sums(saluki):
    """To the bit where the widened model is given the same matrix (its columns in the file's order, and its input
    put in that order); within what the order of a float32 sum leaves otherwise; and far off without the reordering."""
    kept, wide = saluki.read(saluki.kept), saluki.read(saluki.wide)
    assert kept.tiled is True and wide.tiled is False
    ours, theirs = logits(kept), logits(wide)
    assert np.isfinite(ours).all() and np.ptp(ours) > 1
    apart = float(np.abs(ours - theirs).max())
    assert apart < 2e-3, apart  # (measured: 2.8e-4, of logits up to 12)
    assert all(row.argmax() == other.argmax() for row, other in zip(ours, theirs))
    same = saluki.read(saluki.wide)
    same.wout, same.tiled = kept.wout, True
    assert np.array_equal(logits(same).view(np.uint32), ours.view(np.uint32))
    unordered = saluki.read(saluki.kept, tiled=None)
    assert float(np.abs(logits(unordered) - theirs).max()) > 100 * apart and unordered.tiled is False


def test_a_float32_file_keeps_the_blocks_too(saluki):
    """dtype says how the other rows are stored: float32 has the RoPE tables in the file and the vectors as they were."""
    floats = saluki.converted(dtype="float32")
    assert floats.options["dtype"] == "float32" and floats.options["kinds"] == saluki.kept.options["kinds"]
    assert len(floats.checkpoint) > len(saluki.kept.checkpoint)
    assert np.array_equal(logits(saluki.read(floats)).view(np.uint32), logits(saluki.read(saluki.kept)).view(np.uint32))


def test_forward_js_is_given_such_a_checkpoint_with_its_kinds_and_its_order(saluki):
    """T.363.5: the plan says every layer's type where it is and that the output matrix is tiled; one that says "tiled"
    alone says that too (forward.js would multiply the output matrix by an input in the other order without)."""
    from test_external import Outside
    for conversion, more in ((saluki.kept, {}), (saluki.wide, {"tiled": True})):
        options = {**conversion.options, **more}
        outside = Outside(bytes(conversion.checkpoint))
        Llama(None, bytes(conversion.tokenizer), external=outside, **options)
        assert outside.plan["tiled"] is True
        assert outside.plan["tensors"] == external_tensors(conversion.stream.header, options["dtype"], options)
    kinds = saluki.kept.options["kinds"]
    for name, kind in kinds.items():
        said = outside.plan["tensors"] if False else external_tensors(saluki.kept.stream.header, "int8", saluki.kept.options)
        assert [layer["kind"] for layer in said[name]["layers"]] == list(kind) if isinstance(kind, list) else said[name]["kind"] == kind
    plain = Outside(bytes(saluki.wide.checkpoint))
    Llama(None, bytes(saluki.wide.tokenizer), external=plain, **saluki.wide.options)
    assert plain.plan["tiled"] is False and all("layers" not in tensor for tensor in plain.plan["tensors"].values())


# ---- however the file comes
def boundaries(saluki):
    """Where pieces are cut so that they end inside a block, on a tensor's first and last byte, one byte to either
    side, and inside the padding between two tensors."""
    cuts = set()
    for info in saluki.header.values():
        if isinstance(info, dict) and "data_offsets" in info:
            begin, end = (saluki.base + at for at in info["data_offsets"])
            cuts |= {begin - 1, begin, begin + 1, begin + 37, (begin + end) // 2, end - 1, end, end + 1}
    return {cut for cut in cuts if saluki.base < cut < len(saluki.file)}


@pytest.mark.parametrize("pieces", ["at every tensor's ends and inside its blocks", "of 1000 bytes", "of 65537 bytes", "one"])
@pytest.mark.parametrize("piece", [70000, 1 << 20])
def test_the_checkpoint_is_the_same_however_the_file_is_cut(saluki, pieces, piece, monkeypatch):
    monkeypatch.setattr("convert.stream.PIECE", piece)
    cuts = {"at every tensor's ends and inside its blocks": boundaries(saluki), "one": {len(saluki.file)}}.get(
        pieces) or range(saluki.base, len(saluki.file), int(pieces.split()[1]))
    again = saluki.converted(cuts=cuts)
    assert bytes(again.checkpoint) == bytes(saluki.kept.checkpoint) and again.options == saluki.kept.options


def test_a_sink_gets_the_same_bytes_at_the_same_places(saluki):
    """As the page takes a checkpoint: no buffer in Python, every piece written where it belongs."""
    class Sink:
        def open(self, size, header, dtype, form):
            self.out, self.opened = np.zeros(size, dtype=np.uint8), (list(header), dtype, form)

        def write(self, offset, raw):
            assert raw.dtype == np.uint8 and raw.ndim == 1
            self.out[offset:offset + raw.size] = raw

    sink = Sink()
    through = saluki.converted(sink=sink, cuts=boundaries(saluki))
    assert through.checkpoint is None and sink.out.tobytes() == bytes(saluki.kept.checkpoint)
    header, dtype, form = sink.opened
    assert dtype == "int8" and form["kinds"] == through.options["kinds"] and form["tiled"] is True
    assert file_size(tensor_rows(header, form), dtype) == sink.out.size


def test_the_gguf_alone_and_the_conduct_of_a_conversion_keep_them_too(saluki, tmp_path):
    """Conversion.from_gguf(), and the request as the page will make it: conduct(hf, blocks=True)."""
    alone = Conversion.from_gguf(saluki.file, dtype="int8", max_seq_len=1 << 20, blocks=True)
    alone.feed(saluki.file[alone.base:])
    alone.finish()
    assert bytes(alone.checkpoint) == bytes(saluki.kept.checkpoint) and alone.options["kinds"] == saluki.kept.options["kinds"]
    for name, data in (("config.json", json.dumps(saluki.config).encode()), ("tokenizer.json", saluki.vocabulary),
                       ("model.gguf", saluki.file)):
        (tmp_path / name).write_bytes(data)
    hf, answerer = listed(tmp_path)
    conducted = converted(answerer, hf, dtype="int8", max_seq_len=1 << 20, blocks=True)
    assert bytes(conducted.checkpoint) == bytes(saluki.kept.checkpoint) and conducted.options == saluki.kept.options
    plain = converted(answerer, hf, dtype="int8", max_seq_len=1 << 20)
    assert "kinds" not in plain.options and bytes(plain.checkpoint) == bytes(saluki.converted(blocks=False).checkpoint)


# ---- what is kept and what is not
def test_a_conversion_that_was_not_asked_keeps_nothing_and_one_with_no_blocks_is_as_it_was():
    """blocks=True of a Q8_0 GGUF, and of a safetensors file: the conversion of today, to the byte and the option."""
    model = Model(matrices=(q8_0_blocks, 8), dim=64, hidden_dim=128, head_dim=32, key_dim=32, value_dim=32)
    asked, plain = model.converted(), model.converted(blocks=False)
    assert bytes(asked.checkpoint) == bytes(plain.checkpoint) and asked.options == plain.options
    assert isinstance(asked.stream, Kept) and type(plain.stream) is Stream and asked.stream.stay == {}
    assert asked.stream.form["kinds"] is None and asked.stream.form["tiled"] is None


def test_a_stack_with_a_layer_that_is_no_blocks_is_widened_whole(saluki):
    """One layer's FFN matrix in Q8_0: that stack is the file's dtype's as it always was, the others stay."""
    def change(stored):
        blob, _, shape = stored["blk.5.ffn_up.weight"]
        values = SOURCES[saluki.header["model.layers.5.mlp.up_proj.weight"]["dtype"]].read(blob).reshape(shape)
        stored["blk.5.ffn_up.weight"] = [q8_0_blocks(values)[0], 8, shape]

    model = Model(change=change)
    kept = model.converted()
    assert set(saluki.kept.options["kinds"]) - set(kept.options["kinds"]) == {"w3"}
    assert all(kept.options["kinds"][row] == kinds for row, kinds in saluki.kept.options["kinds"].items() if row != "w3")
    held = Llama(bytes(kept.checkpoint), bytes(kept.tokenizer), **kept.options)
    assert held.w3.shape == (8, 512, 256) and np.isfinite(logits(held)).all()
    # (int8 rounds that stack's other layers: near the kept model's, and not it)
    assert 0 < float(np.abs(logits(held) - logits(saluki.read(saluki.kept))).max()) < 2.0


def test_as_many_value_heads_as_key_heads_are_not_tiled_and_the_logits_are_the_widened_ones_to_the_bit():
    """The 0.8B's and the 2B's shape: llama.cpp moves nothing, the form says no "tiled", and no sum changes its order."""
    model = Model(value_heads=2)
    kept, wide = model.converted(), model.converted(blocks=False, dtype="float32")
    assert "tiled" not in kept.options and set(kept.options["kinds"]) == set(MATRICES)
    read = lambda conversion: Llama(bytes(conversion.checkpoint), bytes(conversion.tokenizer), **conversion.options)
    assert np.array_equal(logits(read(kept)).view(np.uint32), logits(read(wide)).view(np.uint32))


@pytest.mark.parametrize("arch", ["llama", "qwen3"])
def test_a_llamas_q_and_k_which_llama_cpp_turns_are_kept_where_they_are(arch):
    """A Llama's GGUF has q and k with the pairs of each head side by side already (the engine's order): the plan's
    turn of what the reader turned back is no move, and the bytes go straight through. A Qwen3's has them as Hugging
    Face does: their rows are moved, as bytes. Either way the widened conversion's logits, to the bit."""
    from test_qwen3 import qwen3
    config, weights = synthetic_weights(dim=256, hidden_dim=512, n_heads=4, n_kv_heads=2, vocab_size=40, shared=False)
    if arch == "qwen3":
        tensors, published = qwen3(config, weights, False)
        more = [("qwen3.attention.key_length", 4, config["head_size"]), ("qwen3.attention.layer_norm_rms_epsilon", 6, 1e-6)]
        published["rms_norm_eps"] = 1e-6
    else:
        (tensors, published), more = hugging_face(config, weights, False), [("llama.attention.layer_norm_rms_epsilon", 6, 1e-5)]
    file, _ = gguf_file(tensors, published, config["vocab_size"], arch, pre="qwen2" if arch == "qwen3" else "gpt-2",
                        more=more, matrices=blocks_of("IQ2_XS", scale=0x1400))
    vocabulary = unigram(config["vocab_size"])
    header, base = llama2_convert.gguf_weights(file, json.dumps(published))
    made = {}
    for blocks, dtype in ((True, "int8"), (False, "float32")):
        conversion = Conversion(header, base, json.dumps(published), vocabulary, "tokenizer.json", dtype=dtype,
                                max_seq_len=1 << 20, start=base, blocks=blocks)
        for start in range(base, len(file), 5000):
            conversion.feed(file[start:start + 5000])
        conversion.finish()
        made[blocks] = conversion
    stream = made[True].stream
    moved = sorted(name.split(".")[-2] for name, orders in stream.stay.items() if orders is not None)
    layers = config["n_layers"]
    assert moved == ([] if arch == "llama" else ["k_proj"] * layers + ["q_proj"] * layers)
    assert set(made[True].options["kinds"].values()) == {"IQ2_XS"} and "tiled" not in made[True].options
    read = lambda conversion: Llama(bytes(conversion.checkpoint), bytes(conversion.tokenizer), **conversion.options)
    ours, theirs = logits(read(made[True])), logits(read(made[False]))
    assert np.isfinite(ours).all() and np.ptp(ours) > 0 and np.array_equal(ours.view(np.uint32), theirs.view(np.uint32))


def test_a_granites_q_which_the_conversion_scales_is_widened_and_the_rest_kept():
    """T.502.5: a Granite's GGUF keeps its multiplier apart and q as a Llama's (turned): the conversion scales q once
    (T.253), which no bytes can do, so q is widened and every other matrix stays as its blocks. The logits are the
    widened conversion's, to the bit."""
    from test_granite import granite_gguf
    config = dict(dim=256, hidden_dim=512, n_heads=4, vocab_size=40, n_layers=2)
    settings, published, file, _ = granite_gguf(config=config, matrices=blocks_of("IQ2_XS", scale=0x1400))
    vocabulary = unigram(settings["vocab_size"])
    header, base = llama2_convert.gguf_weights(file, json.dumps(published))
    made = {}
    # (both float32: q widened stays float32 beside the blocks, as a float32 file keeps them: an int8 one would round it)
    for blocks in (True, False):
        conversion = Conversion(header, base, json.dumps(published), vocabulary, "tokenizer.json", dtype="float32",
                                max_seq_len=1 << 20, start=base, blocks=blocks)
        for start in range(base, len(file), 5000):
            conversion.feed(file[start:start + 5000])
        conversion.finish()
        made[blocks] = conversion
    kinds = made[True].options["kinds"]
    assert "wq" not in kinds and {kinds[name] for name in ("wk", "wv", "wo", "w1", "w2", "w3")} == {"IQ2_XS"}, kinds
    read = lambda conversion: Llama(bytes(conversion.checkpoint), bytes(conversion.tokenizer), **conversion.options)
    ours, theirs = logits(read(made[True])), logits(read(made[False]))
    assert np.isfinite(ours).all() and np.ptp(ours) > 0 and np.array_equal(ours.view(np.uint32), theirs.view(np.uint32))


# ---- the parts
CASES = {
    "a Llama's q, turned in the file and turned by the plan": (dict(turned=4), ("permute", 4), 8),
    "a Qwen3's q, as Hugging Face has it": (dict(), ("permute", 4), 8),
    "a Qwen3.5's q beside its gate, part of a head turning": (dict(), ("heads", 2, 0, 2, 4), 8),
    "its gate": (dict(), ("heads", 2, 1, 2, 0), 8),
    "its k": (dict(), ("heads", 1, 0, 2, 6), 16),
    "a GPT-NeoX's k, q k and v split in the file": (dict(split=2), ("heads", 3, 1, 2, 4), 8, 48),
    "value heads tiled along the rows, q and k before them": (dict(tiled=[8, 2, 3, 4, 0]), None, 8),
    "value heads tiled along the rows": (dict(tiled=[0, 4, 2, 4, 0]), None, 8),
    "value heads tiled along the columns": (dict(tiled=[0, 4, 2, 4, 1]), None, 8),
    "nothing": (dict(), None, 8),
}


@pytest.mark.parametrize("case", CASES)
def test_the_order_of_the_rows_is_what_the_conversion_does_to_the_values(case):
    """order() names, for every row of the checkpoint's tensor, the row of the file it is: taking those rows of the
    file's values is what Stream.convert() does to them (the GGUF's own order put back, then the plan's transform)."""
    marks, transform, head, count = (*CASES[case], 32)[:4]
    info = {"shape": [count, 5], "dtype": "IQ2_XS", **marks}
    values = np.random.default_rng(1).standard_normal((count, 5)).astype(np.float32)
    back = values
    if "turned" in marks:
        back = unturned(back, marks["turned"])
    if "split" in marks:
        back = unsplit(back, marks["split"])
    if "tiled" in marks and marks["tiled"][-1] == 0:
        back = untiled(back, *marks["tiled"])
    expected = transformed(back, transform, head)
    rows = order(info, transform, head)
    assert rows.ndim == 1 and np.array_equal(values[rows], expected)
    assert stays(info, transform)
    if case in ("nothing", "value heads tiled along the columns", "a Llama's q, turned in the file and turned by the plan"):
        assert np.array_equal(rows, np.arange(count))
    else:
        assert not np.array_equal(rows, np.arange(len(rows)))


@pytest.mark.parametrize("info, transform", [
    ({"dtype": "Q8_0"}, None), ({"dtype": "F32"}, None), ({"dtype": "IQ2_XS", "transposed": True}, ("transpose",)),
    ({"dtype": "IQ2_XS", "transposed": True}, None), ({"dtype": "IQ2_XS", "transposed": True}, ("permute", 4)),
    ({"dtype": "IQ2_XS"}, ("transpose",)), ({"dtype": "IQ2_XS"}, ("part", 0, 3)), ({"dtype": "IQ2_XS"}, ("scale", 0.5)),
    ({"dtype": "IQ2_XS"}, (("permute", 4), ("scale", 0.5))), ({"dtype": "IQ2_XS", "done": "one"}, ("one",)),
    ({"dtype": "IQ2_XS"}, ("decay",)), ({"dtype": "IQ2_XS"}, ("taps",))])
def test_a_tensor_that_is_no_blocks_or_whose_values_change_does_not_stay(info, transform):
    assert not stays({"shape": [8, 256], **info}, transform)


def test_only_the_rows_a_quantized_file_quantizes_are_kept():
    """A vector's row (the gates' small matrices, a norm) stays what the file's dtype makes of it, whatever its type."""
    header, form = [256, 512, 8, 4, 2, -40, 24], {"arch": "qwen35", "head_dim": 64, "linear": {
        "every": 4, "key_heads": 2, "value_heads": 6, "key_dim": 64, "value_dim": 128, "conv": 4}}
    rows = tensor_rows(header, form)
    plan, _ = llama2_convert.conversion_plan(header, form, "model.", 16)
    tensors = {name: {"dtype": "IQ2_S", "shape": [1, 256]} for parts in plan if parts for name, _ in parts}
    said = kept_form(plan, rows, tensors)
    assert set(said["kinds"]) == set(MATRICES) and set(said["kinds"].values()) == {"IQ2_S"} and said["tiled"] is None
    assert kept_form(plan, rows, {}) == {"kinds": None, "tiled": None}
    tensors["model.layers.0.linear_attn.out_proj.weight"]["tiled"] = [0, 2, 3, 128, 1]
    assert kept_form(plan, rows, tensors)["tiled"] is True


def test_the_writer_copies_bytes_to_a_layers_place_from_any_byte_on():
    header = [256, 512, 8, 4, 2, -40, 24]
    form = {"arch": "qwen35", "head_dim": 64, "linear": {"every": 4, "key_heads": 2, "value_heads": 6, "key_dim": 64, "value_dim": 128, "conv": 4},
            "kinds": {"w1": ["IQ2_XXS", "Q4_K"] * 4, "w2": "IQ3_S"}}
    rows = tensor_rows(header, form)
    out = bytearray(file_size(rows, "int8"))
    writer = Writer(out, header, "int8", form)
    index = {row.name: number for number, row in enumerate(rows)}
    places = placed(rows, "int8")
    each = 512 * 256
    # a stack of a kind for each layer: layer 3 (Q4_K, 144 bytes a block), from its byte 1000 on
    writer.copy(index["w1"], 3 * each, np.full(77, 7, dtype=np.uint8), 1000)
    at = places[index["w1"]].layers[3].offset + 1000
    assert at == places[index["w1"]].offset + 2 * 512 * 66 + 512 * 144 + 1000
    # a stack of one kind: layer 5 is where five layers of 256 * 512 / 256 blocks of 110 bytes end
    writer.copy(index["w2"], 5 * each, np.full(5, 9, dtype=np.uint8))
    there = places[index["w2"]].offset + 5 * 512 * 110
    written = np.flatnonzero(np.frombuffer(out, dtype=np.uint8)[28:]) + 28
    assert written.tolist() == [*range(at, at + 77), *range(there, there + 5)]
    assert writer.part(index["w1"], 3 * each + 256) == (places[index["w1"]].layers[3], 256)
    assert writer.part(index["w2"], 77) == (places[index["w2"]], 77)


def test_the_plan_writes_the_codebooks_itself():
    header, form = [256, 512, 8, 4, 2, -40, 24], {"arch": "qwen35", "head_dim": 64, "linear": {
        "every": 4, "key_heads": 2, "value_heads": 6, "key_dim": 64, "value_dim": 128, "conv": 4}}
    plain, _ = llama2_convert.conversion_plan(header, form, "model.", 16)
    plan, shapes = llama2_convert.conversion_plan(header, {**form, "kinds": {"w1": "IQ2_S"}}, "model.", 16)
    assert plan[:-1] == plain and plan[-1] == "codes" and shapes[-1] == (CODEBOOKS,)
    from convert.plan import made
    assert made({}, header, plan, len(plan) - 1).tobytes() == grids.codebooks()[0]
    assert all(parts is None or isinstance(parts, list) for parts in plain)


def test_the_forms_default_is_not_tiled():
    assert FORM["tiled"] is None and Llama.tiled is False


# ---- the real model's sizes, without its weights
class Places:
    def open(self, size, header, dtype, form):
        self.size, self.form, self.writes = size, form, []

    def write(self, offset, raw):
        self.writes.append((offset, raw.size))


def test_saluki_27b_is_laid_out_with_its_blocks_kept_and_its_classifier_lands_past_4_gib():
    """The header the converter makes of the real file's head (tests/saluki_form.py's, the classifier first in the
    file), without the 7.9 GB: the form it writes is the one T.363.2 sized, the checkpoint 7,956,712,492 bytes, and the
    first bytes fed, which end inside a block, are written where the classifier is. tests/smoke.mjs asks the same inside
    Pyodide, whose NumPy has integers of 32 bits."""
    sink = Places()
    stream = Kept(saluki_form.header(), 0, saluki_form.CONFIG, "int8", 4096, sink=sink)
    assert stream.form["kinds"] == saluki_form.KINDS and stream.form["tiled"] is True
    assert sink.size == saluki_form.BYTES + CODEBOOKS == 7956712492 == stream.size("int8")
    assert sink.writes == [(0, 28), (sink.size - CODEBOOKS, CODEBOOKS)]
    moved = sorted(name.rsplit(".", 2)[-2] for name, orders in stream.stay.items() if orders is not None)
    assert moved == sorted(["in_proj_qkv"] * 48 + ["in_proj_z"] * 48 + ["q_proj"] * 16 + ["k_proj"] * 16)
    assert len(stream.stay) == 402 == sum(saluki_form.TENSORS.values())
    sink.writes.clear()
    classifier = sink.size - CODEBOOKS - 248320 * (5120 // 256) * 110
    assert classifier > 2 ** 32
    before = stream.done
    assert stream.feed(bytes(300))[0] - before == 2 * 256 and stream.feed(bytes(100))[0] - before == 3 * 256
    assert sink.writes == [(classifier, 300), (classifier + 300, 100)]
    assert len(json.dumps(stream.form["kinds"])) == 4189
