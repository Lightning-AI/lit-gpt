# Inference

We demonstrate how to run inference (next token prediction) with the GPT base model in the [`generate.py`](generate.py) script:

```bash
litgpt generate base --prompt "Hello, my name is" --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

Output:

```text
Hello, my name is Levi Durrer, I'm an Austrian journalist - Chairman of the Press Blair Party, with 37 years in the Press Blair International, and two years in the Spectre of Austerity for the other. I'm crossing my fingers that you will feel
```

The script assumes you have downloaded and converted the weights as described [here](download_model_weights.md).

This will run the 3B pre-trained model and require ~7 GB of GPU memory using the `bfloat16` datatype.

## Run interactively

You can also chat with the model interactively:

```bash
litgpt chat --checkpoint_dir checkpoints/stabilityai/stablelm-tuned-alpha-3b
```

This script can work with any checkpoint. For the best chat-like experience, we recommend using it with a checkpoints
fine-tuned for chatting such as `stabilityai/stablelm-tuned-alpha-3b` or `togethercomputer/RedPajama-INCITE-Chat-3B-v1`.

## Run a large model on one smaller device

Check out our [quantization tutorial](quantize.md).

## Run a large model on multiple smaller devices

We offer two scripts to leverage multiple devices for inference.

### [`litgpt/generate/sequentially.py`](../litgpt/generate/sequentially.py)

Allows you to run models that wouldn't fit in a single card by partitioning the transformer blocks across all your devices and running them sequentially.

For instance, `meta-llama/Llama-2-70b-chat-hf` would require ~140 GB of GPU memory to load on a single device, plus the memory for activations.
With 80 transformer layers, we could partition them across 8, 5, 4, or 2 devices.

```shell
litgpt generate sequentially \
  --checkpoint_dir checkpoints/meta-llama/Llama-2-70b-chat-hf \
  --max_new_tokens 256 \
  --num_samples 2
```

Using A100 40GB GPUs, we need to use at least 4. You can control the number of devices by setting the `CUDA_VISIBLE_DEVICES=` environment variable.

| Devices | Max GPU RAM | Token/sec |
|---------|-------------|-----------|
| 2       | OOM         | -         |
| 4       | 35.64 GB    | 7.55      |
| 5       | 28.72 GB    | 7.49      |
| 8       | 18.35 GB    | 7.47      |

Note that the memory usage will also depend on the `max_new_tokens` value used.

The script also supports quantization, using 4-bit precision, we can now use 2 GPUs

```shell
litgpt generate sequentially \
  --checkpoint_dir checkpoints/meta-llama/Llama-2-70b-chat-hf \
  --max_new_tokens 256 \
  --num_samples 2 \
  --quantize bnb.nf4-dq
```

| Devices | Max GPU RAM | Token/sec |
|---------|-------------|-----------|
| 2       | 20.00 GB    | 8.63      |
| 4       | 10.80 GB    | 8.23      |
| 5       | 8.96 GB     | 8.10      |
| 8       | 6.23 GB     | 8.18      |

Smaller devices can also be used to run inference with this technique.

### [`litgpt/generate/tp.py`](../litgpt/generate/tp.py)

Uses tensor parallelism (TP) to run models that wouldn't fit in a single card by sharding the MLP and Attention QKV linear layers across all your devices.

For instance, `meta-llama/Llama-2-70b-chat-hf` would require ~140 GB of GPU memory to load on a single device, plus the memory for activations.
The requirement is that the intermediate size (for the MLP) and the QKV size (for attention) is divisible by the number of devices.
With an intermediate size of 28672, we can use 2, 4, 7, or 8 devices. With a QKV size of 10240 we can use 2, 4, 5, or 8 devices.
Since the script is configured to shard both, the intersection is used: we can only use 2, 4, or 8 devices.

```shell
litgpt generate tp \
  --checkpoint_dir checkpoints/meta-llama/Llama-2-70b-chat-hf \
  --max_new_tokens 256 \
  --num_samples 2
```

Using A100 40GB GPUs, we need to use at least 4. You can control the number of devices by setting the `CUDA_VISIBLE_DEVICES=` environment variable.

| Devices | Max GPU RAM | Token/sec |
|---------|-------------|-----------|
| 2       | OOM         | -         |
| 4       | 35.46 GB    | 9.33      |
| 8       | 18.19 GB    | 8.61      |

Note that the memory usage will also depend on the `max_new_tokens` value used.

The script also supports quantization, using 4-bit precision, we can now use 2 GPUs

```shell
litgpt generate tp \
  --checkpoint_dir checkpoints/meta-llama/Llama-2-70b-chat-hf \
  --max_new_tokens 256 \
  --num_samples 2 \
  --quantize bnb.nf4-dq
```

| Devices | Max GPU RAM | Token/sec |
|---------|-------------|-----------|
| 2       | 19.79 GB    | 6.72      |
| 4       | 10.73 GB    | 6.48      |
| 8       | 6.15 GB     | 6.20      |

Smaller devices can also be used to run inference with this technique.
