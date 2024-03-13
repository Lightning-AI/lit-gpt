# Quantize the model

This document provides different strategies for quantizing the various models available in LitGPT to reduce GPU memory usage, which is useful for running larger models on certain GPU hardware.

**All the examples below were run on an A100 40GB GPU with CUDA 12.1.**

> [!NOTE]
> Quantization also supports finetuning via [QLoRA](finetune_lora.md)

## Baseline

It's useful to start with a baseline to have a reference point for memory savings via the various quantization methods.

```bash
litgpt generate base --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision 32-true --max_new_tokens 256
...
Time for inference 1: 6.93 sec total, 36.96 tokens/sec.
Memory used: 28.95 GB
```

First, using a lower precision compared to 32-bit float can result in two times reduced memory consumption. You can either try setting `--precision 16-true` for regular 16-bit precision or  `--precision bf16-true` if your GPU supports brain-float 16-bit precision. ([This brief video](https://lightning.ai/courses/deep-learning-fundamentals/9.0-overview-techniques-for-speeding-up-model-training/unit-9.1-accelerated-model-training-via-mixed-precision-training/) explains the difference between regular 16-bit and bf16-bit precision.)

In short, when `--precision bf16-true` or `--precision 16-true` is used, the model weights will automatically be converted and consume less memory.
However, this might not be enough for large models or when using GPUs with limited memory.

```bash
litgpt generate base --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 5.37 sec total, 47.66 tokens/sec.
Memory used: 14.50 GB
```

To reduce the memory requirements further, LitGPT supports several quantization techniques, which are shown below.

> [!TIP]
> Most quantization examples below also use the `--precision bf16-true` setting explained above. If your GPU does not support `bfloat16`, you can change it to `--precision 16-true`.

## `bnb.nf4`

Enabled with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

> [!IMPORTANT]
> `bitsandbytes` only supports `CUDA` devices and the `Linux` operating system.
> Windows users should use [WSL2](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).

Uses the normalized float 4 (nf4) data type. This is recommended over "fp4" based on the paper's experimental results and theoretical analysis.

```bash
pip install scipy bitsandbytes  # scipy is required until https://github.com/TimDettmers/bitsandbytes/pull/525 is released

litgpt generate base --quantize bnb.nf4 --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 6.80 sec total, 37.62 tokens/sec
Memory used: 5.72 GB
```

## `bnb.nf4-dq`

Enabled with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

"dq" stands for "Double Quantization" which reduces the average memory footprint by quantizing the quantization constants.
In average, this amounts to about 0.37 bits per parameter (approximately 3 GB for a 65B model).

```bash
pip install scipy bitsandbytes  # scipy is required until https://github.com/TimDettmers/bitsandbytes/pull/525 is released

litgpt generate base --quantize bnb.nf4-dq --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 8.09 sec total, 30.87 tokens/sec
Memory used: 5.38 GB
```

## `bnb.fp4`

Enabled with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

Uses pure FP4 quantization.

```bash
pip install scipy bitsandbytes  # scipy is required until https://github.com/TimDettmers/bitsandbytes/pull/525 is released

litgpt generate base --quantize bnb.fp4 --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 6.92 sec total, 36.98 tokens/sec
Memory used: 5.72 GB
```

## `bnb.fp4-dq`

Enabled with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

"dq" stands for "Double Quantization" which reduces the average memory footprint by quantizing the quantization constants.
In average, this amounts to about 0.37 bits per parameter (approximately 3 GB for a 65B model).

```bash
pip install scipy bitsandbytes  # scipy is required until https://github.com/TimDettmers/bitsandbytes/pull/525 is released

litgpt generate base --quantize bnb.fp4-dq --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 10.02 sec total, 25.54 tokens/sec
Memory used: 5.38 GB
```

## `bnb.int8`

Enabled with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2110.02861) to learn more about how it works.

```bash
pip install scipy bitsandbytes  # scipy is required until https://github.com/TimDettmers/bitsandbytes/pull/525 is released

litgpt generate base --quantize bnb.int8 --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision 16-true --max_new_tokens 256
...
Time for inference 1: 20.22 sec total, 12.66 tokens/sec
Memory used: 8.70 GB
```
