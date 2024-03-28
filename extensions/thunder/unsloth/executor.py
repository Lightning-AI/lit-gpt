# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import sys
from pathlib import Path
from typing import Optional, Tuple

import thunder
import thunder.torch as ltorch
import torch
from thunder.core.proxies import TensorProxy
from thunder.core.transforms import get_grad, mean_backward, put_grads
from thunder.extend import OperatorExecutor, register_executor
from thunder.torch import ne, sum, true_divide
from torch import Tensor

import litgpt.model

sys.path.append(str(Path(__file__).parent))

import kernels

unsloth_ex = OperatorExecutor("unsloth", version="0.1")
register_executor(unsloth_ex)


"""
====================
 Cross Entropy Loss
====================
"""


def unsloth_cross_entropy_meta(logits: TensorProxy, labels: TensorProxy) -> Tuple[TensorProxy, TensorProxy]:
    return (
        TensorProxy(
            shape=(logits.shape[0],),
            # the cross entropy kernel only supports float32
            dtype=thunder.dtypes.float32,
            device=logits.device,
            requires_grad=logits.requires_grad,
        ),
        TensorProxy(shape=(logits.shape[0],), dtype=thunder.dtypes.float32, device=logits.device, requires_grad=False),
    )


unsloth_cross_entropy = unsloth_ex.register_operator(
    "unsloth_cross_entropy", meta=unsloth_cross_entropy_meta, fn=kernels.cross_entropy_loss._cross_entropy_forward_impl
)


def unsloth_cross_entropy_backward_impl(dlosses: Tensor, logits: Tensor, labels: Tensor, logsumexp: Tensor) -> Tensor:
    # clone() because the kernel writes the grads in the logits
    return kernels.cross_entropy_loss._cross_entropy_backward_impl(dlosses, logits.clone(), logsumexp, labels)


def unsloth_cross_entropy_backward_meta(
    dlosses: TensorProxy, logits: TensorProxy, logsumexp: TensorProxy, labels: TensorProxy
) -> TensorProxy:
    return thunder.TensorProxy(like=logits)


unsloth_cross_entropy_backward = unsloth_ex.register_operator(
    "unsloth_cross_entropy_backward", meta=unsloth_cross_entropy_backward_meta, fn=unsloth_cross_entropy_backward_impl
)


def unsloth_cross_entropy_checker(
    logits: TensorProxy,
    labels: TensorProxy,
    weight: Optional[TensorProxy] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> bool:
    return (
        weight is None
        and size_average is None
        and reduce is None
        and reduction in ("none", "mean")
        and ignore_index == -100
        and label_smoothing == 0.0
        and logits.device.type == "cuda"
        and labels.device.type == "cuda"
    )


def cross_entropy_to_unsloth(
    logits: TensorProxy,
    labels: TensorProxy,
    weight: Optional[TensorProxy] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tuple[TensorProxy, TensorProxy]:
    loss, logsumexp = unsloth_cross_entropy(logits, labels)
    if reduction == "mean":
        # "mean" reduction is not part of the kernel
        # TODO: this doesn't consider that all elements could be masked, causing a division by 0
        n_items = sum(ne(labels, -100))
        loss = true_divide(sum(loss), n_items)
    elif reduction != "none":
        raise NotImplementedError(reduction)
    return loss, logsumexp


def unsloth_cross_entropy_grad(
    logits: TensorProxy,
    labels: TensorProxy,
    weight: Optional[TensorProxy] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> TensorProxy:
    loss, logsumexp = cross_entropy_to_unsloth(**locals())
    grad = get_grad(loss)
    if reduction == "mean":
        grad = mean_backward(logsumexp.ndim, logsumexp.shape, (0,), grad)
    logits_grad = unsloth_cross_entropy_backward(grad, logits, labels, logsumexp)
    put_grads((logits,), (logits_grad,))
    return loss


# registers as cross entropy implementation, including the execution transform and now a grad transform
unsloth_ex.register_implementation(
    ltorch.cross_entropy,
    checker=unsloth_cross_entropy_checker,
    execution_transform=lambda *args: cross_entropy_to_unsloth(*args)[0],
    grad_transform=unsloth_cross_entropy_grad,
)


"""
=========
 RMSNorm
=========

The RMSNorm kernel is not integrated because it's not numerically equal and it doesn't compute the gradient for the
weight, just for the input.
"""


"""
========
 SwiGLU
========
"""


def swiglu(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(e) * g


from litgpt.model import LLaMAMLP as OriginalLLaMAMLP


class ThunderLLaMAMLP(OriginalLLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = swiglu(x_fc_1, x_fc_2)
        return self.proj(x)


litgpt.model.LLaMAMLP = ThunderLLaMAMLP


def swiglu_forward_meta(e: TensorProxy, g: TensorProxy) -> TensorProxy:
    return TensorProxy(like=e)


litgpt_swiglu = unsloth_ex.register_operator("litgpt_swiglu", meta=swiglu_forward_meta, fn=swiglu, replaces=swiglu)


unsloth_swiglu_forward = unsloth_ex.register_operator(
    "unsloth_swiglu_forward", meta=swiglu_forward_meta, fn=lambda *args: kernels.swiglu_fg_kernel(*args)
)


def unsloth_swiglu_backward_meta(DW: TensorProxy, e: TensorProxy, g: TensorProxy) -> Tuple[TensorProxy, TensorProxy]:
    return TensorProxy(like=g), TensorProxy(like=e)


def unsloth_swiglu_backward_fn(DW: Tensor, e: Tensor, g: Tensor) -> Tuple[Tensor, Tuple]:
    B, T, n_embd = e.shape
    e = e.view(-1, n_embd)
    g = g.view(-1, n_embd)
    DW, e, g = kernels.swiglu_DWf_DW_dfg_kernel(DW, e, g)
    e = e.view(B, T, n_embd)
    g = g.view(B, T, n_embd)
    return g, e


unsloth_swiglu_backward = unsloth_ex.register_operator(
    "unsloth_swiglu_backward", meta=unsloth_swiglu_backward_meta, fn=unsloth_swiglu_backward_fn
)


def swiglu_to_unsloth_checker(e: TensorProxy, g: TensorProxy) -> bool:
    return e.device.type == "cuda" and g.device.type == "cuda"


def unsloth_swiglu_grad(e: TensorProxy, g: TensorProxy) -> TensorProxy:
    h = unsloth_swiglu_forward(**locals())
    grad = get_grad(h)
    e_grad, g_grad = unsloth_swiglu_backward(grad, e, g)
    put_grads((e, g), (e_grad, g_grad))
    return h


unsloth_ex.register_implementation(
    litgpt_swiglu,
    checker=swiglu_to_unsloth_checker,
    execution_transform=unsloth_swiglu_forward,
    grad_transform=unsloth_swiglu_grad,
)


"""
======
 RoPE
======
"""


def apply_rope_meta(x: TensorProxy, cos: TensorProxy, sin: TensorProxy) -> TensorProxy:
    return TensorProxy(like=x)


apply_rope = unsloth_ex.register_operator(
    "litgpt_apply_rope", like=apply_rope_meta, fn=litgpt.model.apply_rope, replaces=litgpt.model.apply_rope
)


def unsloth_apply_rope_meta(
    Q: TensorProxy, cos: TensorProxy, sin: TensorProxy
) -> Tuple[TensorProxy, TensorProxy, TensorProxy, int, int, int]:
    batch, n_heads, seq_len, head_dim = Q.shape
    assert seq_len <= cos.shape[0]
    BLOCK_SIZE, num_warps = kernels.calculate_settings(head_dim // 2)
    div, mod = divmod(n_heads, kernels.rope_embedding.ROPE_GROUP_SIZE)
    n_groups = div + (mod != 0)
    return TensorProxy(like=Q), cos, sin, n_groups, BLOCK_SIZE, num_warps


unsloth_apply_rope = unsloth_ex.register_operator(
    "unsloth_apply_rope", meta=unsloth_apply_rope_meta, fn=kernels._rope_embedding_forward_impl
)


def unsloth_apply_rope_backward_meta(
    dY: TensorProxy, cos: TensorProxy, sin: TensorProxy, n_groups: int, BLOCK_SIZE: int, num_warps: int
) -> TensorProxy:
    return TensorProxy(like=dY)


unsloth_apply_rope_backward = unsloth_ex.register_operator(
    "unsloth_apply_rope_backward", meta=unsloth_apply_rope_backward_meta, fn=kernels._rope_embedding_backward_impl
)


def apply_rope_to_unsloth_checker(x: TensorProxy, cos: TensorProxy, sin: TensorProxy) -> bool:
    return len(x.shape) == 4 and x.device.type == "cuda" and cos.device.type == "cuda" and sin.device.type == "cuda"


def unsloth_apply_rope_grad(x: TensorProxy, cos: TensorProxy, sin: TensorProxy) -> TensorProxy:
    Q, cos, sin, n_groups, BLOCK_SIZE, num_warps = unsloth_apply_rope(x, cos, sin)
    dY = get_grad(Q)
    dX = unsloth_apply_rope_backward(dY, cos, sin, n_groups, BLOCK_SIZE, num_warps)
    put_grads((x,), (dX,))
    return Q


unsloth_ex.register_implementation(
    apply_rope,
    checker=apply_rope_to_unsloth_checker,
    execution_transform=lambda *args: unsloth_apply_rope(*args)[0],
    grad_transform=unsloth_apply_rope_grad,
)
