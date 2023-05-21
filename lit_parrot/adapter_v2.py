#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility fucnction to extend the original Parrot-Adapter method to Parrot-Adapter v2,
    This is a port from Lit-LLaMA based on the code prepared by @rasbt aka Sebastian Raschka """

import torch
from torch import Tensor
from torch.nn import functional as F

from lit_parrot.adapter import Parrot

def mark_only_adapter_v2_as_trainable(model: Parrot) -> None:
    """Sets requires_grad=False for all non-adapter weights"""

    for name, param in model.named_parameters():
        substrings = ("adapter_wte", "gating_factor", "adapter_scale", "adapter_bias", "norm_1", "norm_2", "ln_f")
        param.requires_grad = any(s in name for s in substrings)

def adapter_v2_state_from_state_dict(state_dict:dict) -> dict:
    """Return the model state dict with only the adapter weights for saving"""
    substrings = ("adapter_wte", "gating_factor", "adapter_scale", "adapter_bias")

    return {name: param for name, param in state_dict.items() if any(s in name for s in substrings)}

def adapter_v2_new_forward(self, input: Tensor) -> Tensor:
    return self.adapter_scale * (F.linear(input, self.weight, self.bias) + self.adapter_bias)

def adapter_v2_linear_with_bias_and_scale(layer):
    layer.adapter_bias = torch.nn.Parameter(torch.zeros(layer.weight.shape[0]), requires_grad=True)
    layer.adapter_scale = torch.nn.Parameter(torch.ones(layer.weight.shape[0]), requires_grad=True)
    bound_method = adapter_v2_new_forward.__get__(layer, layer.__class__)
    setattr(layer, 'forward', bound_method)
    return layer

