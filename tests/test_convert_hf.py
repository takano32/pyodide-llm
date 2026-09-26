# The build's converter (convert_hf.py) on every architecture: it sized the output as a Llama, so a GPT-2 or a
# GPT-NeoX stopped on "the buffer has not the size of the checkpoint" (found in T85). The page's path was fine.
import json

import numpy as np
import pytest
from conftest import synthetic_weights
from test_bias import qwen2
from test_qwen3 import qwen3
from test_convert import converted, hugging_face, reader, safetensors_file
from test_gpt2 import gpt2_model
from test_neox import neox_model

import convert_hf
from llama2_convert import Safetensors


def llama_model():
    config, weights = synthetic_weights()
    return hugging_face(config, weights, True)


def qwen2_model():
    config, weights = synthetic_weights()
    return qwen2(config, weights, True)


def qwen3_model():
    # heads of 16 in a dim of 32 with 4 heads: q twice as wide as dim, as in Qwen3 0.6B (T124)
    config, weights = synthetic_weights(n_kv_heads=2, head_size=16)
    return qwen3(config, weights, True)


# all four architectures (Fable's review: the Llama path was the one that worked, and a change to it should say so)
@pytest.mark.parametrize("model", [llama_model, qwen2_model, qwen3_model, gpt2_model, neox_model],
                         ids=["llama", "qwen2", "qwen3", "gpt2", "neox"])
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_convert_hf_writes_what_the_page_writes(tmp_path, model, dtype):
    tensors, config = model()
    file = safetensors_file(tensors)
    (tmp_path / "model.safetensors").write_bytes(file)
    (tmp_path / "config.json").write_text(json.dumps(config))
    convert_hf.convert(tmp_path, tmp_path / "out.bin", np.dtype(dtype), 2048)
    assert (tmp_path / "out.bin").read_bytes() == converted(Safetensors(reader(file)), config, dtype, 2048)
