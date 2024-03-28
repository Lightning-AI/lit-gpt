# Lightning Thunder: a source-to-source compiler for PyTorch

[Lightning Thunder](https://github.com/Lightning-AI/lightning-thunder) makes PyTorch programs faster both on single accelerators or in distributed settings.

Thunder aims to be usable, understandable, and extensible and can achieve significant speedups over standard PyTorch eager code, through the compounding effects of optimizations and the use of best in class executors.

This extension directory shows how Thunder can be used with LitGPT.

## Thunder 👉👈 LitGPT: a short showcase

To try Lightning Thunder with your model simply `thunder.jit()` it.

```python
from litgpt import GPT
import thunder
import torch

# Use only two layers to keep the traces shorter for the demonstration
model = GPT.from_name("Llama-2-7b-hf", n_layer=2).cuda()
model = thunder.jit(model)
x = torch.randint(model.max_seq_length, (2, 5), device="cuda")
y = model(x)  # forward, this may take a bit
```

This will require some compilation time on the first forward call.

### Traces

The JIT is will acquire a Python program (what we call a "trace") from the Python program (`GPT`, a `torch.nn.Module` in this example) that was given.
This process targets PyTorch operators (like `Tensor.view()`, `+`, `torch.nn.functional.scaled_dot_product_atttention()`) and optionally custom operators (more about that later).

We can visualize the thunder trace generated under the hood:

```python
forward_trace = thunder.last_traces(model)[-1].python()
print(forward_trace)
```

```python
@torch.no_grad()
@no_autocast()
def augmented_forward_fn(*args):
  # args: "Collection"
  t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, \
  t18, t19, = args
  del args
  t24 = torch.nn.functional.embedding(t0, t19, None, None, 2.0, False, False)  # t24: "cuda:0 f32[2, 5, 4096]"
  t20 = torch_slice_prim_impl(t1, [0, 0], [5, 128], [1, 1])  # t20: "cuda:0 f32[5, 128]"
  t21 = torch_slice_prim_impl(t2, [0, 0], [5, 128], [1, 1])  # t21: "cuda:0 f32[5, 128]"
  t200 = torch.unsqueeze(t11, 0)  # t200: "cuda:0 f32[1, 4096]"
  t201 = torch.unsqueeze(t200, 1)  # t201: "cuda:0 f32[1, 1, 4096]"
  del t200
  t33 = Tensor.expand(t201, (2, 5, 4096))  # t33: "cuda:0 f32[2, 5, 4096]"
  del t201
  t229 = torch.unsqueeze(t13, 0)  # t229: "cuda:0 f32[1, 4096]"
  t230 = torch.unsqueeze(t229, 1)  # t230: "cuda:0 f32[1, 1, 4096]"
  del t229
  t84 = Tensor.expand(t230, (2, 5, 4096))  # t84: "cuda:0 f32[2, 5, 4096]"
  del t230
  t232 = torch.unsqueeze(t12, 0)  # t232: "cuda:0 f32[1, 4096]"
  t233 = torch.unsqueeze(t232, 1)  # t233: "cuda:0 f32[1, 1, 4096]"
  del t232
  t104 = Tensor.expand(t233, (2, 5, 4096))  # t104: "cuda:0 f32[2, 5, 4096]"
  del t233
  t253 = torch.unsqueeze(t14, 0)  # t253: "cuda:0 f32[1, 4096]"
  t254 = torch.unsqueeze(t253, 1)  # t254: "cuda:0 f32[1, 1, 4096]"
  del t253
  t155 = Tensor.expand(t254, (2, 5, 4096))  # t155: "cuda:0 f32[2, 5, 4096]"
  del t254
  t256 = torch.unsqueeze(t10, 0)  # t256: "cuda:0 f32[1, 4096]"
  t257 = torch.unsqueeze(t256, 1)  # t257: "cuda:0 f32[1, 1, 4096]"
  del t256
  t175 = Tensor.expand(t257, (2, 5, 4096))  # t175: "cuda:0 f32[2, 5, 4096]"
  del t257
  t221 = torch.unsqueeze(t20, 0)  # t221: "cuda:0 f32[1, 5, 128]"
  del t20
  t222 = torch.unsqueeze(t221, 1)  # t222: "cuda:0 f32[1, 1, 5, 128]"
  del t221
  t49 = Tensor.expand(t222, (2, 32, 5, 128))  # t49: "cuda:0 f32[2, 32, 5, 128]"
  del t222
  t224 = torch.unsqueeze(t21, 0)  # t224: "cuda:0 f32[1, 5, 128]"
  del t21
  t225 = torch.unsqueeze(t224, 1)  # t225: "cuda:0 f32[1, 1, 5, 128]"
  del t224
  t51 = Tensor.expand(t225, (2, 32, 5, 128))  # t51: "cuda:0 f32[2, 32, 5, 128]"
  del t225
  [t30, t34] = nvFusion0(t24, t33)
  t35 = torch.nn.functional.linear(t34, t3, None)  # t35: "cuda:0 f32[2, 5, 12288]"
  t36 = torch.reshape(t35, (2, 5, 32, 3, 128))  # t36: "cuda:0 f32[2, 5, 32, 3, 128]"
  del t35
  t37 = torch.permute(t36, (0, 2, 3, 1, 4))  # t37: "cuda:0 f32[2, 32, 3, 5, 128]"
  del t36
  (t38, t39, t40) = torch.split(t37, (1, 1, 1), 2)
  del t37
  t41 = torch.reshape(t38, (2, 32, 5, 128))  # t41: "cuda:0 f32[2, 32, 5, 128]"
  del t38
  t42 = torch.reshape(t39, (2, 32, 5, 128))  # t42: "cuda:0 f32[2, 32, 5, 128]"
  del t39
  t43 = torch.reshape(t40, (2, 32, 5, 128))  # t43: "cuda:0 f32[2, 32, 5, 128]"
  del t40
  t44 = torch_slice_prim_impl(t41, [0, 0, 0, 0], [2, 32, 5, 128], [1, 1, 1, 1])  # t44: "cuda:0 f32[2, 32, 5, 128]"
  t54 = torch_slice_prim_impl(t42, [0, 0, 0, 0], [2, 32, 5, 128], [1, 1, 1, 1])  # t54: "cuda:0 f32[2, 32, 5, 128]"
  t64 = torch_slice_prim_impl(t41, [0, 0, 0, 0], [2, 32, 5, 0], [1, 1, 1, 1])  # t64: "cuda:0 f32[2, 32, 5, 0]"
  del t41
  t66 = torch_slice_prim_impl(t42, [0, 0, 0, 0], [2, 32, 5, 0], [1, 1, 1, 1])  # t66: "cuda:0 f32[2, 32, 5, 0]"
  del t42
  t46 = torch_slice_prim_impl(t44, [0, 0, 0, 64], [2, 32, 5, 128], [1, 1, 1, 1])  # t46: "cuda:0 f32[2, 32, 5, 64]"
  t45 = torch_slice_prim_impl(t44, [0, 0, 0, 0], [2, 32, 5, 64], [1, 1, 1, 1])  # t45: "cuda:0 f32[2, 32, 5, 64]"
  t55 = torch_slice_prim_impl(t54, [0, 0, 0, 0], [2, 32, 5, 64], [1, 1, 1, 1])  # t55: "cuda:0 f32[2, 32, 5, 64]"
  t56 = torch_slice_prim_impl(t54, [0, 0, 0, 64], [2, 32, 5, 128], [1, 1, 1, 1])  # t56: "cuda:0 f32[2, 32, 5, 64]"
  [t47, t57] = nvFusion1(t46, t56)
  del t46, t56
  t48 = torch.cat((t47, t45), -1)  # t48: "cuda:0 f32[2, 32, 5, 128]"
  del t47, t45
  t58 = torch.cat((t57, t55), -1)  # t58: "cuda:0 f32[2, 32, 5, 128]"
  del t57, t55
  [t53, t63] = nvFusion2(t44, t48, t49, t51, t54, t58)
  del t44, t48, t54, t58
  t65 = torch.cat((t53, t64), -1)  # t65: "cuda:0 f32[2, 32, 5, 128]"
  del t53, t64
  t67 = torch.cat((t63, t66), -1)  # t67: "cuda:0 f32[2, 32, 5, 128]"
  del t63, t66
  (t68, t69, t70, t71) = sdpaex_grad_forward_scaled_dot_product_efficient_attention(t65, t67, t43, None, 0.0, True, 0.08838834764831843)
  t72 = torch.permute(t68, (0, 2, 1, 3))  # t72: "cuda:0 f32[2, 5, 32, 128]"
  t73 = torch.reshape(t72, (2, 5, 4096))  # t73: "cuda:0 f32[2, 5, 4096]"
  del t72
  t74 = torch.nn.functional.linear(t73, t15, None)  # t74: "cuda:0 f32[2, 5, 4096]"
  [t75, t81, t85] = nvFusion3(t24, t74, t84)
  del t74
  t86 = torch.nn.functional.linear(t85, t5, None)  # t86: "cuda:0 f32[2, 5, 11008]"
  t87 = torch.nn.functional.linear(t85, t7, None)  # t87: "cuda:0 f32[2, 5, 11008]"
  [t93] = nvFusion4(t86, t87)
  t94 = torch.nn.functional.linear(t93, t16, None)  # t94: "cuda:0 f32[2, 5, 4096]"
  [t101, t105, t95] = nvFusion5(t104, t75, t94)
  del t94
  t106 = torch.nn.functional.linear(t105, t4, None)  # t106: "cuda:0 f32[2, 5, 12288]"
  t107 = torch.reshape(t106, (2, 5, 32, 3, 128))  # t107: "cuda:0 f32[2, 5, 32, 3, 128]"
  del t106
  t108 = torch.permute(t107, (0, 2, 3, 1, 4))  # t108: "cuda:0 f32[2, 32, 3, 5, 128]"
  del t107
  (t109, t110, t111) = torch.split(t108, (1, 1, 1), 2)
  del t108
  t112 = torch.reshape(t109, (2, 32, 5, 128))  # t112: "cuda:0 f32[2, 32, 5, 128]"
  del t109
  t113 = torch.reshape(t110, (2, 32, 5, 128))  # t113: "cuda:0 f32[2, 32, 5, 128]"
  del t110
  t114 = torch.reshape(t111, (2, 32, 5, 128))  # t114: "cuda:0 f32[2, 32, 5, 128]"
  del t111
  t135 = torch_slice_prim_impl(t112, [0, 0, 0, 0], [2, 32, 5, 0], [1, 1, 1, 1])  # t135: "cuda:0 f32[2, 32, 5, 0]"
  t137 = torch_slice_prim_impl(t113, [0, 0, 0, 0], [2, 32, 5, 0], [1, 1, 1, 1])  # t137: "cuda:0 f32[2, 32, 5, 0]"
  t115 = torch_slice_prim_impl(t112, [0, 0, 0, 0], [2, 32, 5, 128], [1, 1, 1, 1])  # t115: "cuda:0 f32[2, 32, 5, 128]"
  del t112
  t125 = torch_slice_prim_impl(t113, [0, 0, 0, 0], [2, 32, 5, 128], [1, 1, 1, 1])  # t125: "cuda:0 f32[2, 32, 5, 128]"
  del t113
  t116 = torch_slice_prim_impl(t115, [0, 0, 0, 0], [2, 32, 5, 64], [1, 1, 1, 1])  # t116: "cuda:0 f32[2, 32, 5, 64]"
  t117 = torch_slice_prim_impl(t115, [0, 0, 0, 64], [2, 32, 5, 128], [1, 1, 1, 1])  # t117: "cuda:0 f32[2, 32, 5, 64]"
  t127 = torch_slice_prim_impl(t125, [0, 0, 0, 64], [2, 32, 5, 128], [1, 1, 1, 1])  # t127: "cuda:0 f32[2, 32, 5, 64]"
  t126 = torch_slice_prim_impl(t125, [0, 0, 0, 0], [2, 32, 5, 64], [1, 1, 1, 1])  # t126: "cuda:0 f32[2, 32, 5, 64]"
  [t118, t128] = nvFusion6(t117, t127)
  del t117, t127
  t129 = torch.cat((t128, t126), -1)  # t129: "cuda:0 f32[2, 32, 5, 128]"
  del t128, t126
  t119 = torch.cat((t118, t116), -1)  # t119: "cuda:0 f32[2, 32, 5, 128]"
  del t118, t116
  [t124, t134] = nvFusion7(t115, t119, t125, t129, t49, t51)
  del t115, t119, t125, t129
  t136 = torch.cat((t124, t135), -1)  # t136: "cuda:0 f32[2, 32, 5, 128]"
  del t124, t135
  t138 = torch.cat((t134, t137), -1)  # t138: "cuda:0 f32[2, 32, 5, 128]"
  del t134, t137
  (t139, t140, t141, t142) = sdpaex_grad_forward_scaled_dot_product_efficient_attention(t136, t138, t114, None, 0.0, True, 0.08838834764831843)
  t143 = torch.permute(t139, (0, 2, 1, 3))  # t143: "cuda:0 f32[2, 5, 32, 128]"
  t144 = torch.reshape(t143, (2, 5, 4096))  # t144: "cuda:0 f32[2, 5, 4096]"
  del t143
  t145 = torch.nn.functional.linear(t144, t17, None)  # t145: "cuda:0 f32[2, 5, 4096]"
  [t146, t152, t156] = nvFusion8(t145, t155, t95)
  del t145
  t158 = torch.nn.functional.linear(t156, t8, None)  # t158: "cuda:0 f32[2, 5, 11008]"
  t157 = torch.nn.functional.linear(t156, t6, None)  # t157: "cuda:0 f32[2, 5, 11008]"
  [t164] = nvFusion9(t157, t158)
  t165 = torch.nn.functional.linear(t164, t18, None)  # t165: "cuda:0 f32[2, 5, 4096]"
  [t166, t172, t176] = nvFusion10(t146, t165, t175)
  del t165
  t177 = torch.nn.functional.linear(t176, t9, None)  # t177: "cuda:0 f32[2, 5, 32000]"
  return {'output': t177, 'flat_args': [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19], 'flat_output': (t177,)}, ((t0, t101, t104, t105, t114, t136, t138, t139, t140, t141, t142, t144, t146, t15, t152, t155, t156, t157, t158, t16, t164, t166, t17, t172, t175, t176, t18, t24, t3, t30, t33, t34, t4, t43, t49, t5, t51, t6, t65, t67, t68, t69, t7, t70, t71, t73, t75, t8, t81, t84, t85, t86, t87, t9, t93, t95), (False, False, True, True, 4096.0, 4096.0, 0.0, 0.08838834764831843, 4096.0, 4096.0, 4096.0, 0.0, 0.08838834764831843, 32000, 2, 2))
```

This is a straight-lined version of `GPT.forward` that has been optimized. Since it's running on CUDA, the [NvFuser](https://github.com/NVIDIA/Fuser) executor has created regions (look for "nvFusion") that fuse multiple operators together.

Operator fusion is very desirable with modern hardware and helps out in overhead-bound or device-bound settings by:
- Launching less kernels, thus reducing the kernel launch overhead.
- Reducing the number of memory accesses performed by reusing them in a fused operation
- Minimizing host-device communications

Thunder also uses a multi-level intermediate representation. If we let it print all levels

```python
forward_trace = thunder.last_traces(model)[-1]
print(forward_trace)
```

We can see as comments the primitives that compose the fusion regions. For instance, this is the region associated to [the `RMSNorm` implementation](https://github.com/Lightning-AI/litgpt/blob/9b6475dabf90c7acee506a026bd9fa86251835bf/litgpt/model.py#L409-L420)

```python
  [t146, t152, t156] = nvFusion8(t145, t155, t95)
    # t146 = prims.add(t145, t95)  # t146: "cuda:0 f32[2, 5, 4096]"
    # t147 = prims.mul(t146, t146)  # t147: "cuda:0 f32[2, 5, 4096]"
    # t148 = prims.sum(t147, (2,))  # t148: "cuda:0 f32[2, 5]"
    # t149 = prims.broadcast_in_dim(t148, [2, 5, 1], [0, 1])  # t149: "cuda:0 f32[2, 5, 1]"
    # t150 = prims.div(t149, 4096.0)  # t150: "cuda:0 f32[2, 5, 1]"
    # t151 = prims.add(t150, 1e-05)  # t151: "cuda:0 f32[2, 5, 1]"
    # t152 = prims.rsqrt(t151)  # t152: "cuda:0 f32[2, 5, 1]"
    # t153 = prims.broadcast_in_dim(t152, (2, 5, 4096), (0, 1, 2))  # t153: "cuda:0 f32[2, 5, 4096]"
    # t154 = prims.mul(t146, t153)  # t154: "cuda:0 f32[2, 5, 4096]"
    # t156 = prims.mul(t154, t155)  # t156: "cuda:0 f32[2, 5, 4096]"
```

Similarly, we can visualize the backward trace:

```python
backward_trace = thunder.last_backward_traces(model)[-1].python()
print(backward_trace)
```

```python
@torch.no_grad()
@no_autocast()
def backward_fn(saved_for_backward, cotangents):
  # saved_for_backward: "Collection"
  # cotangents: "Collection"
  C0, C1, = saved_for_backward
  clear_collection(saved_for_backward)
  del saved_for_backward
  t178, = cotangents
  clear_collection(cotangents)
  del cotangents
  t0, t101, t104, t105, t114, t136, t138, t139, t140, t141, t142, t144, t146, \
  t15, t152, t155, t156, t157, t158, t16, t164, t166, t17, t172, t175, t176, t18, \
  t24, t3, t30, t33, t34, t4, t43, t49, t5, t51, t6, t65, t67, t68, t69, t7, t70, \
  t71, t73, t75, t8, t81, t84, t85, t86, t87, t9, t93, t95, = C0
  clear_collection(C0)
  del C0
  b1, b2, b41, b91, f101, f106, f40, f42, f51, f56, f6, f90, f92, i0, i23, i73, \
  = C1
  clear_collection(C1)
  del C1
  t639 = torch.reshape(t178, (-1, 32000))  # t639: "cuda:0 f32[10, 32000]"
  del t178
  t643 = torch.permute(t639, (1, 0))  # t643: "cuda:0 f32[32000, 10]"
  t644 = torch.reshape(t176, (-1, 4096))  # t644: "cuda:0 f32[10, 4096]"
  del t176
  t669 = torch.reshape(t164, (-1, 11008))  # t669: "cuda:0 f32[10, 11008]"
  del t164
  t686 = torch.reshape(t156, (-1, 4096))  # t686: "cuda:0 f32[10, 4096]"
  del t156
  t720 = torch.reshape(t144, (-1, 4096))  # t720: "cuda:0 f32[10, 4096]"
  del t144
  t776 = torch.reshape(t105, (-1, 4096))  # t776: "cuda:0 f32[10, 4096]"
  del t105
  t802 = torch.reshape(t93, (-1, 11008))  # t802: "cuda:0 f32[10, 11008]"
  del t93
  t819 = torch.reshape(t85, (-1, 4096))  # t819: "cuda:0 f32[10, 4096]"
  del t85
  t853 = torch.reshape(t73, (-1, 4096))  # t853: "cuda:0 f32[10, 4096]"
  del t73
  t911 = torch.reshape(t34, (-1, 4096))  # t911: "cuda:0 f32[10, 4096]"
  del t34
  t640 = torch.matmul(t639, t9)  # t640: "cuda:0 f32[10, 4096]"
  del t639, t9
  t645 = torch.matmul(t643, t644)  # t645: "cuda:0 f32[32000, 4096]"
  del t643, t644
  t641 = torch.reshape(t640, (2, 5, 4096))  # t641: "cuda:0 f32[2, 5, 4096]"
  del t640
  [t648, t663] = nvFusion0(f106, t166, t172, t175, t641)
  del f106, t166, t172, t175, t641
  t664 = torch.reshape(t663, (-1, 4096))  # t664: "cuda:0 f32[10, 4096]"
  t668 = torch.permute(t664, (1, 0))  # t668: "cuda:0 f32[4096, 10]"
  t665 = torch.matmul(t664, t18)  # t665: "cuda:0 f32[10, 11008]"
  del t664, t18
  t670 = torch.matmul(t668, t669)  # t670: "cuda:0 f32[4096, 11008]"
  del t668, t669
  t666 = torch.reshape(t665, (2, 5, 11008))  # t666: "cuda:0 f32[2, 5, 11008]"
  del t665
  [t672, t680] = nvFusion1(t157, t158, t666)
  del t157, t158, t666
  t681 = torch.reshape(t672, (-1, 11008))  # t681: "cuda:0 f32[10, 11008]"
  del t672
  t685 = torch.permute(t681, (1, 0))  # t685: "cuda:0 f32[11008, 10]"
  t688 = torch.reshape(t680, (-1, 11008))  # t688: "cuda:0 f32[10, 11008]"
  del t680
  t692 = torch.permute(t688, (1, 0))  # t692: "cuda:0 f32[11008, 10]"
  t689 = torch.matmul(t688, t6)  # t689: "cuda:0 f32[10, 4096]"
  del t688, t6
  t682 = torch.matmul(t681, t8)  # t682: "cuda:0 f32[10, 4096]"
  del t681, t8
  t694 = torch.matmul(t692, t686)  # t694: "cuda:0 f32[11008, 4096]"
  del t692
  t687 = torch.matmul(t685, t686)  # t687: "cuda:0 f32[11008, 4096]"
  del t685, t686
  t683 = torch.reshape(t682, (2, 5, 4096))  # t683: "cuda:0 f32[2, 5, 4096]"
  del t682
  t690 = torch.reshape(t689, (2, 5, 4096))  # t690: "cuda:0 f32[2, 5, 4096]"
  del t689
  [t698, t714] = nvFusion2(f101, t146, t152, t155, t663, t683, t690)
  del f101, t146, t152, t155, t663, t683, t690
  t715 = torch.reshape(t714, (-1, 4096))  # t715: "cuda:0 f32[10, 4096]"
  t719 = torch.permute(t715, (1, 0))  # t719: "cuda:0 f32[4096, 10]"
  t716 = torch.matmul(t715, t17)  # t716: "cuda:0 f32[10, 4096]"
  del t715, t17
  t721 = torch.matmul(t719, t720)  # t721: "cuda:0 f32[4096, 4096]"
  del t719, t720
  t717 = torch.reshape(t716, (2, 5, 4096))  # t717: "cuda:0 f32[2, 5, 4096]"
  del t716
  t722 = torch.reshape(t717, (2, 5, 32, 128))  # t722: "cuda:0 f32[2, 5, 32, 128]"
  del t717
  t723 = torch.permute(t722, (0, 2, 1, 3))  # t723: "cuda:0 f32[2, 32, 5, 128]"
  del t722
  (t724, t725, t726, _) = sdpaex_scaled_dot_product_efficient_attention_backward(t723, t136, t138, t114, None, t139, t140, t141, t142, f90, b91, scale=f92)
  del t723, t136, t138, t114, t139, t140, t141, t142, f90, b91, f92
  t765 = torch.reshape(t726, (2, 32, 1, 5, 128))  # t765: "cuda:0 f32[2, 32, 1, 5, 128]"
  del t726
  t727 = torch_slice_prim_impl(t725, [0, 0, 0, 0], [2, 32, 5, 128], [1, 1, 1, 1])  # t727: "cuda:0 f32[2, 32, 5, 128]"
  del t725
  t730 = torch_slice_prim_impl(t724, [0, 0, 0, 0], [2, 32, 5, 128], [1, 1, 1, 1])  # t730: "cuda:0 f32[2, 32, 5, 128]"
  del t724
  [t747, t764] = nvFusion3(t49, t51, t727, t730)
  del t727, t730
  t766 = torch.reshape(t747, (2, 32, 1, 5, 128))  # t766: "cuda:0 f32[2, 32, 1, 5, 128]"
  del t747
  t767 = torch.reshape(t764, (2, 32, 1, 5, 128))  # t767: "cuda:0 f32[2, 32, 1, 5, 128]"
  del t764
  t768 = torch.cat((t767, t766, t765), i73)  # t768: "cuda:0 f32[2, 32, 3, 5, 128]"
  del t767, t766, t765, i73
  t769 = torch.permute(t768, (0, 3, 1, 2, 4))  # t769: "cuda:0 f32[2, 5, 32, 3, 128]"
  del t768
  t770 = torch.reshape(t769, (2, 5, 12288))  # t770: "cuda:0 f32[2, 5, 12288]"
  del t769
  t771 = torch.reshape(t770, (-1, 12288))  # t771: "cuda:0 f32[10, 12288]"
  del t770
  t775 = torch.permute(t771, (1, 0))  # t775: "cuda:0 f32[12288, 10]"
  t777 = torch.matmul(t775, t776)  # t777: "cuda:0 f32[12288, 4096]"
  del t775, t776
  t772 = torch.matmul(t771, t4)  # t772: "cuda:0 f32[10, 4096]"
  del t771, t4
  t773 = torch.reshape(t772, (2, 5, 4096))  # t773: "cuda:0 f32[2, 5, 4096]"
  del t772
  [t780, t796] = nvFusion4(f56, t101, t104, t714, t773, t95)
  del f56, t101, t104, t714, t773, t95
  t797 = torch.reshape(t796, (-1, 4096))  # t797: "cuda:0 f32[10, 4096]"
  t801 = torch.permute(t797, (1, 0))  # t801: "cuda:0 f32[4096, 10]"
  t798 = torch.matmul(t797, t16)  # t798: "cuda:0 f32[10, 11008]"
  del t797, t16
  t803 = torch.matmul(t801, t802)  # t803: "cuda:0 f32[4096, 11008]"
  del t801, t802
  t799 = torch.reshape(t798, (2, 5, 11008))  # t799: "cuda:0 f32[2, 5, 11008]"
  del t798
  [t805, t813] = nvFusion5(t799, t86, t87)
  del t799, t86, t87
  t814 = torch.reshape(t805, (-1, 11008))  # t814: "cuda:0 f32[10, 11008]"
  del t805
  t818 = torch.permute(t814, (1, 0))  # t818: "cuda:0 f32[11008, 10]"
  t821 = torch.reshape(t813, (-1, 11008))  # t821: "cuda:0 f32[10, 11008]"
  del t813
  t825 = torch.permute(t821, (1, 0))  # t825: "cuda:0 f32[11008, 10]"
  t822 = torch.matmul(t821, t5)  # t822: "cuda:0 f32[10, 4096]"
  del t821, t5
  t815 = torch.matmul(t814, t7)  # t815: "cuda:0 f32[10, 4096]"
  del t814, t7
  t827 = torch.matmul(t825, t819)  # t827: "cuda:0 f32[11008, 4096]"
  del t825
  t820 = torch.matmul(t818, t819)  # t820: "cuda:0 f32[11008, 4096]"
  del t818, t819
  t816 = torch.reshape(t815, (2, 5, 4096))  # t816: "cuda:0 f32[2, 5, 4096]"
  del t815
  t823 = torch.reshape(t822, (2, 5, 4096))  # t823: "cuda:0 f32[2, 5, 4096]"
  del t822
  [t831, t847] = nvFusion6(f51, t75, t796, t81, t816, t823, t84)
  del f51, t75, t796, t81, t816, t823, t84
  t848 = torch.reshape(t847, (-1, 4096))  # t848: "cuda:0 f32[10, 4096]"
  t852 = torch.permute(t848, (1, 0))  # t852: "cuda:0 f32[4096, 10]"
  t849 = torch.matmul(t848, t15)  # t849: "cuda:0 f32[10, 4096]"
  del t848, t15
  t854 = torch.matmul(t852, t853)  # t854: "cuda:0 f32[4096, 4096]"
  del t852, t853
  t850 = torch.reshape(t849, (2, 5, 4096))  # t850: "cuda:0 f32[2, 5, 4096]"
  del t849
  t855 = torch.reshape(t850, (2, 5, 32, 128))  # t855: "cuda:0 f32[2, 5, 32, 128]"
  del t850
  t856 = torch.permute(t855, (0, 2, 1, 3))  # t856: "cuda:0 f32[2, 32, 5, 128]"
  del t855
  (t857, t858, t859, _) = sdpaex_scaled_dot_product_efficient_attention_backward(t856, t65, t67, t43, None, t68, t69, t70, t71, f40, b41, scale=f42)
  del t856, t65, t67, t43, t68, t69, t70, t71, f40, b41, f42
  t900 = torch.reshape(t859, (2, 32, 1, 5, 128))  # t900: "cuda:0 f32[2, 32, 1, 5, 128]"
  del t859
  t863 = torch_slice_prim_impl(t857, [0, 0, 0, 0], [2, 32, 5, 128], [1, 1, 1, 1])  # t863: "cuda:0 f32[2, 32, 5, 128]"
  del t857
  t860 = torch_slice_prim_impl(t858, [0, 0, 0, 0], [2, 32, 5, 128], [1, 1, 1, 1])  # t860: "cuda:0 f32[2, 32, 5, 128]"
  del t858
  [t882, t899] = nvFusion7(t49, t51, t860, t863)
  del t49, t51, t860, t863
  t902 = torch.reshape(t899, (2, 32, 1, 5, 128))  # t902: "cuda:0 f32[2, 32, 1, 5, 128]"
  del t899
  t901 = torch.reshape(t882, (2, 32, 1, 5, 128))  # t901: "cuda:0 f32[2, 32, 1, 5, 128]"
  del t882
  t903 = torch.cat((t902, t901, t900), i23)  # t903: "cuda:0 f32[2, 32, 3, 5, 128]"
  del t902, t901, t900, i23
  t904 = torch.permute(t903, (0, 3, 1, 2, 4))  # t904: "cuda:0 f32[2, 5, 32, 3, 128]"
  del t903
  t905 = torch.reshape(t904, (2, 5, 12288))  # t905: "cuda:0 f32[2, 5, 12288]"
  del t904
  t906 = torch.reshape(t905, (-1, 12288))  # t906: "cuda:0 f32[10, 12288]"
  del t905
  t910 = torch.permute(t906, (1, 0))  # t910: "cuda:0 f32[12288, 10]"
  t907 = torch.matmul(t906, t3)  # t907: "cuda:0 f32[10, 4096]"
  del t906, t3
  t912 = torch.matmul(t910, t911)  # t912: "cuda:0 f32[12288, 4096]"
  del t910, t911
  t908 = torch.reshape(t907, (2, 5, 4096))  # t908: "cuda:0 f32[2, 5, 4096]"
  del t907
  [t915, t931] = nvFusion8(f6, t24, t30, t33, t847, t908)
  del f6, t24, t30, t33, t847, t908
  t932 = torch.torch.ops.aten.embedding_backward(t931, t0, i0, -1, b1, b2)  # t932: "cuda:0 f32[32000, 4096]"
  del t931, t0, i0, b1, b2
  return (None, None, None, t912, t777, t827, t694, t820, t687, t645, t648, t915, t780, t831, t698, t854, t803, t721, t670, t932)
```

These traces are long, and require some familiarity with the model implementation to follow them, but they allow you to:
- Inspect exactly what operations are run including their decompositions.
- Inspect the sizes of tensors, their device, data type and conversions.
- Apply transformations to the traces since the computations are completely decoupled from the data.
- Inspect the backward operations generated for each forward operation to understand what autograd is doing.

### Transforms

Transforms are one of the core features of Thunder. For example, they enable easy data parallel distribution. That is replicated data parallelism (DDP) and fully-sharded data parallelism (FSDP).

We provide ready-to-use Fabric strategies that integrate Thunder DDP|FSDP. Under the hood, the code is quite straightforward:

```python
model = thunder.distributed.ddp(model)
# or
# model = thunder.distributed.fsdp(model)

model = thunder.jit(model)
```

After applying the DDP transformation, the backward trace will include the expected all-reduce collectives:

```python
  p1022 = torch_all_reduce_prim_impl(t1021, _DistributedReduceOps_0, _torch_distributed_distributed_c10d_ProcessGroup_1, True, False)  # p1022: "FUTURE cuda:0 f32[16797696]"
  ...
  t1059 = torch_wait_prim_impl(p1025)  # t1059: "cuda:0 f32[131072000]"
```

With `L.Fabric`, this is how to use them:

```python
from extensions.thunder.strategies import ThunderFSDPStrategy, ThunderDDPStrategy

# fully-sharded data parallel
strategy = ThunderFSDPStrategy(
    sharding_strategy="ZERO3",
    bucketing_strategy="BLOCK",
    executors=("sdpa", "torchcompile", "nvfuser", "torch"),
    state_dict_type="full",
)

# replicated data parallel
strategy = ThunderDDPStrategy(executors=("sdpa", "torchcompile", "nvfuser", "torch"))

fabric = L.Fabric(devices=devices, strategy=strategy)
fabric.launch()
model = fabric.setup(model)  # JIT is called here
```

And in the case of FSDP all-gathers in forward and reduce-scatters in backward.
Meaning that Thunder automatically introduced the necessary collective operations to support data parallelism.

### Executors

Thunder allows you to define a priority list of executors that can map operators:

```python
import thunder
from thunder.executors.sdpaex import sdpa_ex
from thunder.executors.torch_compile import torch_compile_executor

model = thunder.jit(
    model,
    executors=[sdpa_ex, torch_compile_executor, thunder.nvfuser_executor, thunder.pytorch_executor]
)
```

Notice how `torch.compile` is a valid executor. This executor registers a few operators with improved performance so that you can utilize the fastest set of operator implementations possible.

### Custom executors

Lightning Thunder provides extension points to integrate fast kernels for operators in your model without having to modify your implementation.

For instance, the [Unsloth project](https://github.com/unslothai/unsloth/) provides several Triton kernels that can be used with LitGPT:
- Cross entropy loss
- SwiGLU (part of `LLaMAMLP`)
- RoPE

The [`unsloth` directory](unsloth) contains a [custom executor](unsloth/executor.py) that registers these operators for LitGPT.
We can enable this executor by passing it to the list of executors available. The order matters because we want to run its custom operators before
`NvFuser` creates its fusion regions.

```python
from unsloth.executor import unsloth_ex

model = thunder.jit(
    model,
    executors=[sdpa_ex, unsloth_ex, torch_compile_executor, thunder.nvfuser_executor, thunder.pytorch_executor]
)
```

Doing this, the model trace now includes the Unsloth kernel calls:

```python
def augmented_forward_fn(*args):
    ...
    (t121, _, _, _, _, _) = unsloth_apply_rope(t120, t21, t22)
    ...
    (t189, t190) = unsloth_cross_entropy(t187, t188)
    ...

def backward_fn(saved_for_backward, cotangents):
    ...
    t652 = unsloth_cross_entropy_backward(t651, t187, t188, t190)  # t652: "cuda:0 f32[6, 320]"
    ...
    t763 = unsloth_apply_rope_backward(t757, t21, t22, 1, 8, 4)  # t763: "cuda:0 f32[2, 4, 3, 16]"
```

We provide a specific [pre-training script copy](unsloth/pretrain.py) that uses this executor.
Given the Unsloth results below, these hand-written kernels do not seem to be worth it, showcasing the power of automated fusion compilers like [NvFuser](https://github.com/NVIDIA/Fuser).

## Examples and benchmarks:

> [!WARNING]
> Lightning Thunder is alpha and not ready for production runs. Feel free to try it out, expect a few bumps along the way.
> We expect speed and memory usage to improve as we continue to develop it.

We provide a version of the main pre-training script [that integrates Thunder](pretrain.py) that uses TinyLlama, a 1.1B parameter LLM.

| Setting              | Compiler/JIT | Devices | ms/iter @ step 10 | Memory (GB) |
|----------------------|--------------|---------|-------------------|-------------|
| Fully-sharded ZeRO 3 | Eager        | 8       | 460.88            | 22.13       |
| Fully-sharded ZeRO 3 | Inductor     | 8       | Error             | Error       |
| Fully-sharded ZeRO 3 | Thunder      | 8       | 333.22            | 21.40       |
|                      |              |         |                   |             |
| Replicated           | Eager        | 8       | 535.28            | 32.05       |
| Replicated           | Inductor     | 8       | Error             | Error       |
| Replicated           | Thunder      | 8       | OOM               | OOM         |
|                      |              |         |                   |             |
| -                    | Eager        | 1       | 449.88            | 29.85       |
| -                    | Inductor     | 1       | Error             | Error       |
| -                    | Thunder      | 1       | 322.83            | 26.37       |
|                      |              |         |                   |             |
| Unsloth              | Thunder      | 1       | 331.93            | 25.19       |

<details>
<summary>Reproduction details</summary>

Config:

```yaml
out_dir: out/pretrain-thunder
data: TinyStories
tokenizer_dir: checkpoints/meta-llama/Llama-2-7b-hf
logger_name: csv
```

Commands:

```bash
python extensions/thunder/pretrain.py --config config.yaml --compiler null --train.global_batch_size 32
python extensions/thunder/pretrain.py --config config.yaml --compiler torch --train.global_batch_size 32
python extensions/thunder/pretrain.py --config config.yaml --executors '[sdpa, torchcompile, nvfuser, torch]' --train.global_batch_size 32

python extensions/thunder/pretrain.py --config config.yaml --compiler null --strategy ddp
python extensions/thunder/pretrain.py --config config.yaml --compiler torch --strategy ddp
python extensions/thunder/pretrain.py --config config.yaml --executors '[sdpa, torchcompile, nvfuser, torch]' --strategy ddp

python extensions/thunder/pretrain.py --config config.yaml --compiler null --devices 1
python extensions/thunder/pretrain.py --config config.yaml --compiler torch --devices 1
python extensions/thunder/pretrain.py --config config.yaml --executors '[sdpa, torchcompile, nvfuser, torch]' --devices 1

python extensions/thunder/pretrain.py --config config.yaml --executors '[sdpa, unsloth, torchcompile, nvfuser, torch]' --devices 1
```

Gradient accumulation is disabled in the FSDP setting because Thunder does not support skipping the backward synchronization yet.

`torch.compile` fails to compile the `_FabricModule` due to this bug: https://github.com/pytorch/pytorch/issues/112787#issuecomment-1986827601 

The CUDA devices are all NVIDIA A100-SXM4-40GB.

```text
Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Is debug build: False
CUDA used to build PyTorch: 12.1
CUDA runtime version: 12.3.107
Nvidia driver version: 545.23.08
pytorch-triton==3.0.0+989adb9a29
torch==2.4.0.dev20240326+cu121
lightning-thunder==84ac1b0f5352f9a5388bd207aab7a0580c29cbfc
nvfuser_cu121==0.2.0.dev20240327
```

</details>
