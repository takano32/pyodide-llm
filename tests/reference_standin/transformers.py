# A stand-in for transformers, for tests/reference_dry.py (T357.4): the classes the reference tools ask for, answering
# with the engine itself (the tensors converted by the converter), so that the tools run from their arguments to their
# exit where there is no torch. Nothing is checked against transformers here: the engine is held to itself, and what
# the dry run shows is that a tool's own steps, lines, verdicts and faults still work. CI holds the engine to the real
# transformers (the tools' heads).
#   what a model holds after a pass is the tool's own reading of the engine's states (its engine_states() or
#     engine_cache()), put into the shape the tool reads from transformers' cache;
#   a pass of one token (the stepped way) is a little off, so that "transformers' own two computations" differ and the
#     lines are ten times a floor that is not zero;
#   WRONG: {"logits": x} adds x to every logit of a pass at once and of a pass token by token, and {"states": x} scales
#     what such a pass holds by 1 + x: a transformers that agrees with itself and that the engine is not within the line
#     of, which every tool must say FAILED to. (One whose two ways disagree by as much would only raise the line: the
#     line is ten times that.) {"kept": x} adds x to the logits of a pass that keeps only some positions' (the long
#     text's), which the other two leave right: each wrong is one verdict's to see.
#   a made-up model's tensors have transformers' names and shapes by tests/made_up.py's makers; their values are the
#     tool's (it writes every parameter).
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch

__version__ = "stand-in"
WRONG = {}
T = torch.Tensor
view = lambda array: np.ascontiguousarray(array).view(T)


def engine_of(tensors, published, context):
    from llama2_convert import Conversion
    from llama2_numpy import Llama
    from made_up import safetensors_file, unigram
    text = published.get("text_config", published)
    file = safetensors_file({name: np.asarray(tensor) for name, tensor in tensors.items()})
    size = struct.unpack("<Q", file[:8])[0]
    conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), unigram(text["vocab_size"]),
                            "tokenizer.json", dtype="float32", max_seq_len=context, start=8 + size)
    conversion.feed(file[8 + size:])
    conversion.finish()
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    checkpoint = bytes(conversion.checkpoint)
    def make():
        llama = Llama(checkpoint, conversion.tokenizer, **options)
        llama.standin_arch = options.get("arch", "llama")
        return llama
    return make


def held(llama, count, stepped, wrong=0.0):
    """the cache of transformers after count positions, from the engine's states as the tool of its kind reads them"""
    arch = llama.standin_arch
    scale = np.float32((1 + 2e-6 if stepped else 1) * (1 + wrong))
    pad = lambda rows: view(np.concatenate([np.zeros_like(rows[:, :1]), rows * scale], axis=1)[None])
    layers = []
    if arch in ("qwen35", "lfm2"):
        tool = sys.modules["reference_qwen35" if arch == "qwen35" else "reference_lfm2"]
        for _, (kind, arrays) in sorted(tool.engine_states(llama, count).items()):
            if kind == "linear":
                layers.append(types.SimpleNamespace(recurrent_states=[view(np.array(arrays[0])[None] * scale)], conv_states=[pad(np.array(arrays[1]))]))
            elif kind == "conv":
                layers.append(types.SimpleNamespace(conv_states=[pad(np.array(arrays[0]))]))
            else:
                layers.append(types.SimpleNamespace(keys=view(np.array(arrays[0])[None] * scale), values=view(np.array(arrays[1])[None] * scale)))
    elif arch in ("gemma3", "gemma4", "spark", "plamo3", "gemma2"):
        for _, (kind, arrays) in sorted(sys.modules[f"reference_{arch}"].engine_states(llama, count).items()):
            layers.append(types.SimpleNamespace(keys=view(np.array(arrays[0])[None] * scale), values=view(np.array(arrays[1])[None] * scale),
                                                is_sliding=kind == "sliding"))
    else:
        for keys, values in sys.modules["reference_llama"].engine_cache(llama, count):
            layers.append(types.SimpleNamespace(keys=view(np.array(keys)[None] * scale), values=view(np.array(values)[None] * scale)))
    return types.SimpleNamespace(layers=layers, llama=llama, count=count)


# (T283) transformers' integrations.bitnet, as reference_bitnet.py takes it: its ActQuant, whose apply the tool puts aside
# to leave the model's rounding out. While it is this one, a BitNet's passes are the model's rounded ones: the unit
# tests' reference (test_bitnet.naive_bitnet_logits, quantized) of every token so far, in place of the engine, which
# rounds nothing
def rounding_of_the_model(activation):
    return activation


integrations = types.ModuleType("transformers.integrations")
integrations.bitnet = types.ModuleType("transformers.integrations.bitnet")
integrations.bitnet.ActQuant = type("ActQuant", (), {"apply": staticmethod(rounding_of_the_model)})
sys.modules["transformers.integrations"], sys.modules["transformers.integrations.bitnet"] = integrations, integrations.bitnet


def rounded(model):
    return model.published.get("model_type") == "bitnet" and integrations.bitnet.ActQuant.apply is rounding_of_the_model


class Model:
    def __init__(self, tensors, published, config=None, context=None):
        self.tensors, self.published = tensors, published
        self.config = config or types.SimpleNamespace(**published.get("text_config", published))
        self.context = context or published.get("text_config", published)["max_position_embeddings"]
        self.made = None

    # ---- what a tool does to a model it made
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def tied(self):
        return bool(getattr(self.config, "tie_word_embeddings", False))

    def named_parameters(self):
        return [(key, value) for key, value in self.tensors.items()]

    def state_dict(self):
        out = dict(self.tensors)
        if self.tied():
            out["lm_head.weight"] = self.embedding()
        return out

    def embedding(self):
        # (T342: a Spark-X2.5's is model.embedding)
        return next(value for key, value in self.tensors.items() if key.endswith(("embed_tokens.weight", "model.embedding.weight")))

    @property
    def lm_head(self):
        return types.SimpleNamespace(weight=self.embedding() if self.tied() else self.tensors["lm_head.weight"])

    @property
    def model(self):
        layer = lambda at: types.SimpleNamespace(feed_forward=types.SimpleNamespace(
            w1=types.SimpleNamespace(weight=self.tensors.get(f"model.layers.{at}.feed_forward.w1.weight"))))
        return types.SimpleNamespace(embed_tokens=types.SimpleNamespace(weight=self.embedding()),
                                     layers=[layer(at) for at in range(self.config.num_hidden_layers)])

    # ---- the passes
    def engine(self):
        if self.made is None:
            self.made = engine_of(self.tensors, self.published, self.context)
        return self.made()

    def get_input_embeddings(self):
        return lambda ids: view(np.asarray(ids, dtype=np.float32)[..., None])

    def __call__(self, input_ids=None, inputs_embeds=None, past_key_values=None, use_cache=False, logits_to_keep=None, **more):
        ids = [int(token) for token in (np.asarray(input_ids)[0] if input_ids is not None else np.asarray(inputs_embeds)[0, :, 0])]
        llama = past_key_values.llama if past_key_values is not None else self.engine()
        start = past_key_values.count if past_key_values is not None else 0
        stepped = len(ids) == 1
        logits = np.stack([llama.forward(token, start + at).copy() for at, token in enumerate(ids)])
        so_far = (past_key_values.ids if past_key_values is not None else []) + ids
        if rounded(self):
            from test_bitnet import naive_bitnet_logits
            logits = np.stack(naive_bitnet_logits(self.tensors, self.published, so_far, quantized=True)[-len(ids):]).astype(np.float32)
        wave = np.cos(np.arange(logits.shape[-1], dtype=np.float32))
        some = logits_to_keep is not None
        logits = logits * (1 + 3e-6 * wave if stepped else 1) + WRONG.get("kept" if some else "logits", 0.0) * wave
        if logits_to_keep is not None:
            logits = logits[np.asarray(logits_to_keep)]
        past = held(llama, start + len(ids), stepped, 0.0 if some else WRONG.get("states", 0.0)) if use_cache else None
        if past is not None:
            past.ids = so_far
        return types.SimpleNamespace(logits=view(logits.astype(np.float32)[None]), past_key_values=past)

    def generate(self, ids, max_new_tokens=16, **more):
        ids, llama = [int(token) for token in np.asarray(ids)[0]], self.engine()
        for at, token in enumerate(ids):
            logits = llama.forward(token, at)
        for _ in range(max_new_tokens):
            ids.append(int(np.argmax(logits)))
            logits = llama.forward(ids[-1], len(ids) - 1)
        return view(np.array([ids]))

    @classmethod
    def from_pretrained(cls, directory, config=None, dtype=None, **more):
        directory = Path(directory)
        index = directory / "model.safetensors.index.json"
        names = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
        tensors = {}
        for name in names:
            data = (directory / name).read_bytes()
            size = struct.unpack("<Q", data[:8])[0]
            for key, said in json.loads(data[8:8 + size]).items():
                if key != "__metadata__":
                    begin, end = said["data_offsets"]
                    # (T283: a BitNet's packed matrices are bytes)
                    kind = np.uint8 if said["dtype"] == "U8" else np.float32
                    tensors[key] = np.frombuffer(data[8 + size + begin:8 + size + end], dtype=kind).reshape(said["shape"])
        published = json.loads((directory / "config.json").read_text())
        if config is not None and getattr(config, "num_hidden_layers", None):
            published["num_hidden_layers"] = config.num_hidden_layers
        return cls(tensors, published, context=min(published.get("text_config", published)["max_position_embeddings"], 4096))


# ------------------------------------------------------------------------------------------ made by a config
class Config:
    model_type, defaults = None, {}

    def __init__(self, **said):
        self.__dict__.update({**self.defaults, **said})

    def to_json_string(self, use_diff=False):
        return json.dumps({"model_type": self.model_type, **self.__dict__})


class GraniteConfig(Config):
    model_type = "granite"
    defaults = dict(embedding_multiplier=1.0, residual_multiplier=1.0, logits_scaling=1.0, tie_word_embeddings=False)


class Qwen3_5TextConfig(Config):
    model_type = "qwen3_5_text"
    defaults = dict(tie_word_embeddings=False, linear_conv_kernel_dim=4, rms_norm_eps=1e-6, hidden_act="silu",
                    attn_output_gate=True, mlp_only_layers=[])


class Lfm2Config(Config):
    model_type = "lfm2"
    defaults = dict(tie_word_embeddings=True, conv_L_cache=3, norm_eps=1e-5, block_auto_adjust_ff_dim=True, conv_bias=False)

    def __init__(self, **said):
        super().__init__(**said)
        if "layer_types" not in said:
            self.layer_types = ["full_attention" if at in said["full_attn_idxs"] else "conv" for at in range(said["num_hidden_layers"])]
        self.block_ff_dim, self.tie_embedding = self.intermediate_size, self.tie_word_embeddings


class Gemma3TextConfig(Config):
    model_type = "gemma3_text"
    defaults = dict(tie_word_embeddings=True, hidden_activation="gelu_pytorch_tanh")

    def __init__(self, **said):
        super().__init__(**said)
        pattern = self.__dict__.pop("sliding_window_pattern", 6)
        if "layer_types" not in said:
            self.layer_types = ["sliding_attention" if (at + 1) % pattern else "full_attention" for at in range(said["num_hidden_layers"])]


class SiglipVisionConfig(Config):
    model_type = "siglip_vision_model"


class Gemma3Config(Config):
    """(T613) the whole model's: a Gemma 3's text config and a tower of sight's"""
    model_type = "gemma3"

    def to_json_string(self, use_diff=False):
        said = {key: value for key, value in self.__dict__.items() if key not in ("text_config", "vision_config")}
        return json.dumps({"model_type": self.model_type, **said, "text_config": json.loads(self.text_config.to_json_string()),
                           "vision_config": json.loads(self.vision_config.to_json_string())})


class Gemma2Config(Config):
    """(T612) transformers' Gemma2Config: one sliding-attention layer of every two from the first where none are listed"""
    model_type = "gemma2"
    defaults = dict(tie_word_embeddings=True, hidden_activation="gelu_pytorch_tanh", attn_logit_softcapping=50.0,
                    final_logit_softcapping=30.0, query_pre_attn_scalar=256, sliding_window=4096, rope_theta=10000.0)

    def __init__(self, **said):
        said.pop("attn_implementation", None)
        super().__init__(**said)
        if "layer_types" not in said:
            self.layer_types = ["sliding_attention" if (at + 1) % 2 else "full_attention" for at in range(said["num_hidden_layers"])]


class Gemma4TextConfig(Config):
    model_type = "gemma4_text"
    defaults = dict(tie_word_embeddings=True, hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=None,
                    num_kv_shared_layers=0, use_double_wide_mlp=False, hidden_size_per_layer_input=0, global_head_dim=512)


class Spark2_5Config(Config):
    """(T342) what reference_spark.py has of the model's own code, configuration_spark.py"""
    model_type = "spark2_5"
    defaults = dict(tie_word_embeddings=True, hidden_act="gelu", headwise_attn_output_gate=True, gate_attn_act_mode="sigmoid")


class Plamo3Config(Config):
    """(T649.2) what reference_plamo3.py has of the model's own code, modeling_plamo.py's Plamo3Config"""
    model_type = "plamo3"
    defaults = dict(tie_word_embeddings=False, scale_embedding=False, linear_type="normal")


class Gemma4Model(Model):
    """(T329.3) what reference_gemma4.py reads of transformers' modules: which layers share keys and values, how wide
    each FFN is (transformers' own rules: the last num_kv_shared_layers layers, twice as wide there where
    use_double_wide_mlp)"""

    @property
    def model(self):
        c = self.config
        first = c.num_hidden_layers - c.num_kv_shared_layers
        layer = lambda at: types.SimpleNamespace(
            self_attn=types.SimpleNamespace(is_kv_shared_layer=at >= first > 0 if c.num_kv_shared_layers else False),
            mlp=types.SimpleNamespace(intermediate_size=c.intermediate_size * (2 if c.use_double_wide_mlp and c.num_kv_shared_layers and at >= first else 1)))
        return types.SimpleNamespace(embed_tokens=types.SimpleNamespace(weight=self.embedding()),
                                     layers=[layer(at) for at in range(c.num_hidden_layers)])


def writable(tensors):
    return {key: view(np.array(value, dtype=np.float32)) for key, value in tensors.items()}


def GraniteForCausalLM(config):
    import made_up
    settings, weights = made_up.synthetic_weights(dim=config.hidden_size, hidden_dim=config.intermediate_size, n_layers=config.num_hidden_layers,
                                                  n_heads=config.num_attention_heads, n_kv_heads=config.num_key_value_heads,
                                                  vocab_size=config.vocab_size, seq_len=8, shared=config.tie_word_embeddings)
    tensors, _ = made_up.hugging_face(settings, weights, config.tie_word_embeddings)
    return Model(writable(tensors), json.loads(config.to_json_string()), config)


def Qwen3_5ForCausalLM(config):
    import made_up
    every = config.layer_types.index("full_attention") + 1
    tensors, _ = made_up.qwen35_model(dim=config.hidden_size, hidden_dim=config.intermediate_size, n_layers=config.num_hidden_layers,
                                      every=every, n_heads=config.num_attention_heads, n_kv_heads=config.num_key_value_heads,
                                      head_dim=config.head_dim, key_heads=config.linear_num_key_heads, value_heads=config.linear_num_value_heads,
                                      key_dim=config.linear_key_head_dim, value_dim=config.linear_value_head_dim,
                                      conv=config.linear_conv_kernel_dim, vocab_size=config.vocab_size, seq_len=8,
                                      shared=config.tie_word_embeddings, prefix="model.")
    tensors = {key: value for key, value in tensors.items() if "visual" not in key and not key.startswith("mtp.")}
    return Model(writable(tensors), {"model_type": "qwen3_5", "text_config": json.loads(config.to_json_string())}, config)


def Lfm2ForCausalLM(config):
    import made_up
    kinds = "".join("c" if kind == "conv" else "a" for kind in config.layer_types)
    tensors, _ = made_up.lfm2_model(dim=config.hidden_size, block_ff_dim=config.intermediate_size, kinds=kinds,
                                    n_heads=config.num_attention_heads, n_kv_heads=config.num_key_value_heads, taps=config.conv_L_cache,
                                    vocab_size=config.vocab_size, seq_len=8, shared=config.tie_word_embeddings,
                                    adjust=config.block_auto_adjust_ff_dim)
    return Model(writable(tensors), json.loads(config.to_json_string()), config)


def Gemma3ForCausalLM(config):
    import made_up
    kinds = "".join("s" if kind == "sliding_attention" else "l" for kind in config.layer_types)
    tensors, _ = made_up.gemma3_model(dim=config.hidden_size, hidden_dim=config.intermediate_size, kinds=kinds,
                                      n_heads=config.num_attention_heads, n_kv_heads=config.num_key_value_heads,
                                      head_dim=config.head_dim, vocab_size=config.vocab_size, seq_len=8,
                                      shared=config.tie_word_embeddings)
    return Model(writable(tensors), json.loads(config.to_json_string()), config)


class Gemma3Whole(Model):
    """(T613) transformers' Gemma3ForConditionalGeneration: its language model is model.language_model"""

    @property
    def model(self):
        return types.SimpleNamespace(language_model=types.SimpleNamespace(embed_tokens=types.SimpleNamespace(weight=self.embedding())))


def Gemma3ForConditionalGeneration(config):
    """(T613) the whole model's tensors by transformers' names (model.language_model., model.vision_tower., ...)"""
    import made_up
    text = config.text_config
    kinds = "".join("s" if kind == "sliding_attention" else "l" for kind in text.layer_types)
    tensors, _ = made_up.gemma3_model(dim=text.hidden_size, hidden_dim=text.intermediate_size, kinds=kinds,
                                      n_heads=text.num_attention_heads, n_kv_heads=text.num_key_value_heads,
                                      head_dim=text.head_dim, vocab_size=text.vocab_size, seq_len=8,
                                      shared=text.tie_word_embeddings, whole="new")
    return Gemma3Whole(writable(tensors), json.loads(config.to_json_string()), text)


def Gemma2ForCausalLM(config):
    """(T612) as transformers' own: a Gemma 3's tensors but for the norms of the heads"""
    import made_up
    kinds = "".join("s" if kind == "sliding_attention" else "l" for kind in config.layer_types)
    tensors, _ = made_up.gemma2_model(dim=config.hidden_size, hidden_dim=config.intermediate_size, kinds=kinds,
                                      n_heads=config.num_attention_heads, n_kv_heads=config.num_key_value_heads,
                                      head_dim=config.head_dim, vocab_size=config.vocab_size, seq_len=8,
                                      shared=config.tie_word_embeddings)
    return Model(writable(tensors), json.loads(config.to_json_string()), config)


def Gemma4ForCausalLM(config):
    """(T329.3) as transformers' own: no k, v or their norms in a layer that reads another's"""
    import made_up
    kinds = "".join("s" if kind == "sliding_attention" else "l" for kind in config.layer_types)
    rope = config.rope_parameters
    tensors, _ = made_up.gemma4_model(dim=config.hidden_size, hidden_dim=config.intermediate_size, kinds=kinds,
                                      n_heads=config.num_attention_heads, n_kv_heads=config.num_key_value_heads,
                                      head_dim=config.head_dim, wide_head=config.global_head_dim, window=config.sliding_window,
                                      vocab_size=config.vocab_size, seq_len=8, shared=config.tie_word_embeddings,
                                      sharing=config.num_kv_shared_layers, double=config.use_double_wide_mlp,
                                      per_layer=config.hidden_size_per_layer_input, softcap=config.final_logit_softcapping,
                                      theta=rope["full_attention"]["rope_theta"], local_theta=rope["sliding_attention"]["rope_theta"],
                                      proportion=rope["full_attention"]["partial_rotary_factor"], eps=config.rms_norm_eps)
    first = len(kinds) - config.num_kv_shared_layers
    tensors = {key: value for key, value in tensors.items()
               if not any(f"layers.{at}.self_attn.{name}." in key for at in range(first, len(kinds)) for name in ("k_proj", "v_proj", "k_norm"))}
    return Gemma4Model(writable(tensors), json.loads(config.to_json_string()), config)


def Spark2_5ForCausalLM(config):
    """(T342) what reference_spark.py has of the model's own code, modeling_spark.py"""
    import made_up
    kinds = "".join("s" if kind == "sliding_attention" else "l" for kind in config.layer_types)
    rope = config.rope_parameters
    tensors, _ = made_up.spark_model(dim=config.hidden_size, hidden_dim=config.intermediate_size, kinds=kinds,
                                     n_heads=config.num_attention_heads, n_kv_heads=config.num_key_value_heads,
                                     head_dim=config.head_dim, window=config.sliding_window, vocab_size=config.vocab_size, seq_len=8,
                                     shared=config.tie_word_embeddings, theta=rope["full_attention"]["rope_theta"],
                                     local_theta=rope["sliding_attention"]["rope_theta"],
                                     part=rope["full_attention"]["partial_rotary_factor"], eps=config.rms_norm_eps)
    return Model(writable(tensors), json.loads(config.to_json_string()), config)


def Plamo3ForCausalLM(config):
    """(T649.2) what reference_plamo3.py has of the model's own code, modeling_plamo.py (its lm_head padded to 16 rows)"""
    import made_up
    tensors, _ = made_up.plamo3_model(dim=config.hidden_size, hidden_dim=config.intermediate_size, layers=config.num_hidden_layers,
                                      pattern=config.sliding_window_pattern, n_heads=config.num_attention_heads,
                                      n_kv_heads=config.num_key_value_heads, head_dim=config.head_dim, window=config.window_size,
                                      vocab_size=config.vocab_size, seq_len=8, shared=config.tie_word_embeddings, padded=True,
                                      eps=config.rms_norm_eps, theta=config.rope_theta, local_theta=config.rope_local_theta,
                                      factor=config.rope_scaling_factor, original=config.initial_context_length)
    return Model(writable(tensors), json.loads(config.to_json_string()), config)


Lfm2ForCausalLM.from_pretrained = Gemma3ForCausalLM.from_pretrained = Spark2_5ForCausalLM.from_pretrained = Model.from_pretrained
Plamo3ForCausalLM.from_pretrained = Model.from_pretrained
Gemma2ForCausalLM.from_pretrained = Model.from_pretrained
Qwen3_5ForConditionalGeneration = Model


class AutoModelForCausalLM:
    from_pretrained = Model.from_pretrained


class AutoConfig:
    @staticmethod
    def from_pretrained(directory):
        return types.SimpleNamespace(**json.loads((Path(directory) / "config.json").read_text()))


class AutoTokenizer:
    """the real tokenizers library on the directory's tokenizer.json, and its template through jinja2"""

    def __init__(self, directory):
        import tokenizers
        directory = Path(directory)
        self.tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
        said = json.loads((directory / "tokenizer_config.json").read_text())
        self.template, self.bos = said.get("chat_template"), said.get("bos_token")

    @classmethod
    def from_pretrained(cls, directory, **more):
        return cls(directory)

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids}

    def convert_ids_to_tokens(self, ids):
        return [self.tokenizer.id_to_token(int(token)) for token in ids]

    def decode(self, ids, skip_special_tokens=False):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **more):
        import jinja2
        text = jinja2.Environment().from_string(self.template).render(messages=messages, add_generation_prompt=add_generation_prompt,
                                                                      bos_token=self.bos or "", **more)
        return self.tokenizer.encode(text, add_special_tokens=False).ids if tokenize else text
