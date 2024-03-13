# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from unittest import mock

import pytest
import torch


def test_llama2_70b_conversion():
    from litgpt import Config
    from litgpt.scripts.convert_hf_checkpoint import copy_weights_hf_llama

    shapes = {
        "model.embed_tokens.weight": (32000, 8192),
        "model.layers.0.input_layernorm.weight": (8192,),
        "model.layers.0.mlp.down_proj.weight": (8192, 28672),
        "model.layers.0.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.0.mlp.up_proj.weight": (28672, 8192),
        "model.layers.0.post_attention_layernorm.weight": (8192,),
        "model.layers.0.self_attn.k_proj.weight": (1024, 8192),
        "model.layers.0.self_attn.o_proj.weight": (8192, 8192),
        "model.layers.0.self_attn.q_proj.weight": (8192, 8192),
        "model.layers.0.self_attn.v_proj.weight": (1024, 8192),
        "model.layers.1.input_layernorm.weight": (8192,),
        "model.layers.1.mlp.down_proj.weight": (8192, 28672),
        "model.layers.1.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.1.mlp.up_proj.weight": (28672, 8192),
        "model.layers.1.post_attention_layernorm.weight": (8192,),
        "model.layers.1.self_attn.o_proj.weight": (8192, 8192),
        "model.layers.2.input_layernorm.weight": (8192,),
        "model.layers.2.mlp.down_proj.weight": (8192, 28672),
        "model.layers.2.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.2.mlp.up_proj.weight": (28672, 8192),
        "model.layers.2.post_attention_layernorm.weight": (8192,),
        "model.layers.2.self_attn.o_proj.weight": (8192, 8192),
        "model.layers.3.input_layernorm.weight": (8192,),
        "model.layers.3.mlp.down_proj.weight": (8192, 28672),
        "model.layers.3.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.3.mlp.up_proj.weight": (28672, 8192),
        "model.layers.3.post_attention_layernorm.weight": (8192,),
        "model.layers.3.self_attn.o_proj.weight": (8192, 8192),
        "model.layers.4.input_layernorm.weight": (8192,),
        "model.layers.4.mlp.down_proj.weight": (8192, 28672),
        "model.layers.4.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.4.mlp.up_proj.weight": (28672, 8192),
        "model.layers.4.post_attention_layernorm.weight": (8192,),
        "model.layers.4.self_attn.o_proj.weight": (8192, 8192),
        "model.layers.5.mlp.gate_proj.weight": (28672, 8192),
        "model.layers.5.self_attn.o_proj.weight": (8192, 8192),
    }

    config = Config.from_name("Llama-2-70b-hf")
    holder = {}
    qkv_weights = {}
    with torch.device("meta"):
        weight_map = {k: torch.empty(s) for k, s in shapes.items()}
    copy_weights_hf_llama(config, qkv_weights, holder, weight_map)

    # we are only testing 5 layers
    assert len(qkv_weights) == 5
    # there are no loaded qkv weights
    assert all(v is None for qkv in qkv_weights.values() for v in qkv)
    # the shapes are correct
    holder = {k: tuple(t.shape) for k, t in holder.items()}
    assert holder == {
        "transformer.h.0.attn.attn.weight": (10240, 8192),
        "transformer.h.0.attn.proj.weight": (8192, 8192),
        "transformer.h.0.mlp.fc_1.weight": (28672, 8192),
        "transformer.h.0.mlp.fc_2.weight": (28672, 8192),
        "transformer.h.0.mlp.proj.weight": (8192, 28672),
        "transformer.h.0.norm_1.weight": (8192,),
        "transformer.h.0.norm_2.weight": (8192,),
        "transformer.h.1.attn.proj.weight": (8192, 8192),
        "transformer.h.1.mlp.fc_1.weight": (28672, 8192),
        "transformer.h.1.mlp.fc_2.weight": (28672, 8192),
        "transformer.h.1.mlp.proj.weight": (8192, 28672),
        "transformer.h.1.norm_1.weight": (8192,),
        "transformer.h.1.norm_2.weight": (8192,),
        "transformer.h.2.attn.proj.weight": (8192, 8192),
        "transformer.h.2.mlp.fc_1.weight": (28672, 8192),
        "transformer.h.2.mlp.fc_2.weight": (28672, 8192),
        "transformer.h.2.mlp.proj.weight": (8192, 28672),
        "transformer.h.2.norm_1.weight": (8192,),
        "transformer.h.2.norm_2.weight": (8192,),
        "transformer.h.3.attn.proj.weight": (8192, 8192),
        "transformer.h.3.mlp.fc_1.weight": (28672, 8192),
        "transformer.h.3.mlp.fc_2.weight": (28672, 8192),
        "transformer.h.3.mlp.proj.weight": (8192, 28672),
        "transformer.h.3.norm_1.weight": (8192,),
        "transformer.h.3.norm_2.weight": (8192,),
        "transformer.h.4.attn.proj.weight": (8192, 8192),
        "transformer.h.4.mlp.fc_1.weight": (28672, 8192),
        "transformer.h.4.mlp.fc_2.weight": (28672, 8192),
        "transformer.h.4.mlp.proj.weight": (8192, 28672),
        "transformer.h.4.norm_1.weight": (8192,),
        "transformer.h.4.norm_2.weight": (8192,),
        "transformer.h.5.attn.proj.weight": (8192, 8192),
        "transformer.h.5.mlp.fc_1.weight": (28672, 8192),
        "transformer.wte.weight": (32000, 8192),
        "lm_head.weight": (32000, 8192),  # due to weight tying lm_head is in the converted weights
    }


def test_convert_hf_checkpoint(tmp_path):
    from litgpt.scripts.convert_hf_checkpoint import convert_hf_checkpoint

    with pytest.raises(ValueError, match="to contain .bin"):
        convert_hf_checkpoint(checkpoint_dir=tmp_path, model_name="pythia-14m")

    bin_file = tmp_path / "foo.bin"
    bin_file.touch()
    with mock.patch("litgpt.scripts.convert_hf_checkpoint.lazy_load") as load:
        convert_hf_checkpoint(checkpoint_dir=tmp_path, model_name="pythia-14m")
    load.assert_called_with(bin_file)

    assert {p.name for p in tmp_path.glob("*")} == {"foo.bin", "model_config.yaml", "lit_model.pth"}

    # ensure that the config dict can be loaded
    from litgpt import Config

    config = Config.from_file(tmp_path / "model_config.yaml")
    assert isinstance(config, Config)
