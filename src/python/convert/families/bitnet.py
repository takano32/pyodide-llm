# A BitNet b1.58 (T283: Microsoft's, transformers' model_type "bitnet"): a Llama whose matrices are ternary, one scale a
# matrix, with a norm in front of wo and one in front of w2 and the square of a ReLU in the gate of its FFN
# (engine/layers.py has the rule, docs/notes/t283-bitnet-2026-11-12.md the model). The file of microsoft/bitnet-b1.58-2B-4T
# (config.json's quantization_config: quantization_mode "offline") holds every matrix packed, four values a byte, and
# its scale beside it, as transformers' AutoBitLinear reads them (integrations/bitnet.py of transformers 5.16.1:
# unpack_weights(), and the output multiplied by weight_scale; the form, no line of it): the converter unpacks them
# ("planes"), multiplies them by their scale ("by") and keeps them ternary whatever the file's dtype (the form's
# "kinds": the values are -1, 0 and 1 times the one scale, which the ternary dtype holds as they are). The embedding,
# which is the classifier too, is a table of bfloat16s, and follows the file's dtype.
from convert.families.family import Family, refuse
from convert.families.llama import llama_sources

# the matrices a BitNet holds packed, by their rows in the layout and their names in the original
PACKED = {"wq": "self_attn.q_proj", "wk": "self_attn.k_proj", "wv": "self_attn.v_proj", "wo": "self_attn.o_proj",
          "w1": "mlp.gate_proj", "w2": "mlp.down_proj", "w3": "mlp.up_proj"}


def bitnet_check(config):
    """T283: what the engine has not of a BitNet: an activation other than the square of a ReLU, biases, and matrices
    other than a file of transformers' AutoBitLinear holds packed with their scales (quantization_mode "online", the
    bfloat16 weights the model rounds as it runs, is no such file; BitLinear divides by its scale where AutoBitLinear
    multiplies; a norm inside every matrix, use_rms_norm, is none of the published model's)."""
    if config.get("hidden_act") != "relu2":
        refuse(f"its activation is {config.get('hidden_act')}, not relu2")
    if config.get("attention_bias") or config.get("mlp_bias"):
        refuse("its layers have biases")
    said = config.get("quantization_config")
    said = said if isinstance(said, dict) else {}
    if said.get("quant_method") != "bitnet" or said.get("linear_class", "bitlinear") != "autobitlinear" \
            or said.get("quantization_mode", "offline") != "offline" or said.get("use_rms_norm"):
        refuse("its matrices are not the ternary ones of a file of BitNet's packed weights")


def bitnet_sources(d, prefix, rotary):
    """T283: a Llama's (without the biases and the norms of the heads, which a BitNet has not), the two norms more,
    and every matrix unpacked and multiplied by its scale, then (q and k) interleaved as a Llama's are."""
    layer = "model.layers.{}."
    sources = {name: source for name, source in llama_sources(d, prefix, rotary).items()
               if name not in ("bq", "bk", "bv", "q_norm", "k_norm")}
    sources.update({"rms_att_sub_weight": (layer + "self_attn.attn_sub_norm.weight", None),
                    "rms_ffn_sub_weight": (layer + "mlp.ffn_sub_norm.weight", None)})
    for row, name in PACKED.items():
        then = sources[row][1]
        sources[row] = (layer + name + ".weight", lambda at, name=name, then=then: (
            ("planes",), ("by", f"model.layers.{at}.{name}.weight_scale"), *((then,) if then else ())))
    return sources


BITNET = Family("BitNet", "bitnet", bitnet_sources, bitnet_check, free_heads=True, scaled=True,
                form={"kinds": lambda config: {row: "ternary" for row in PACKED}})
