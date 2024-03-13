# Finetuning with Adapter

Adapter, first introduced for the LLaMA model as [LLaMA-Adapter](https://arxiv.org/abs/2303.16199), is a form of prefix-tuning that prepends a learnable adaption-prompt to the inputs of the attention blocks in an LLM. In total, there are only ~500k parameters to update during finetuning in StableLM 3B, which significantly reduces the memory footprint and speeds up training.

We are able to demonstrate instruction-finetuning LitGPT StableLM 3B on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset on a **single RTX 3060 GPU**. If using 8 GPUs, finetuning can be completed in under 1 hour.

If you are new to Adapter and are interested to learn more about how it works before proceeding with the finetuning guide below, you might find our article [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/) helpful.

LLaMA-Adapter v2 extends the original LLaMA-Adapter idea by adding trainable bias and scale parameters to each linear layer in the transformer. Furthermore, LLaMA-Adapter v2 makes the normalization layers trainable. Where the StableLM 3B model has 500k trainable parameters with GPT v1, GPT-Adapter v2 adds an additional 1.5 M trainable parameter for the bias and scale parameters and ~300k trainable parameters for the normalization layers. So, adapter v2 has ~2.3 M trainable parameters in total.

## Preparation

The steps here only need to be done once:

1. Follow the instructions in the [README](../README.md) to install the dependencies.
2. Download and convert the weights following our [guide](download_model_weights.md).

LitGPT provides common datasets for finetuning, such as Alpaca, LIMA, Dolly, and more.
You can optionally [prepare your own dataset](#tune-on-your-dataset).
For more information about dataset preparation, also see the [prepare_dataset.md](./prepare_dataset.md) tutorial.

## Running the finetuning

```bash
litgpt finetune adapter \
  --data Alpaca \
  --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

or for Adapter V2

```bash
litgpt finetune adapter_v2 \
  --data Alpaca \
  --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

The finetuning requires at least one GPU with ~12 GB memory.
You can speed up training by passing the `devices` argument to the script to utilize more GPUs if available.
Depending on the available GPU memory, you can also tune the `micro_batch_size` parameter to utilize the GPU efficiently.
To fit Adapter V2 to 12GB memory set `--train.micro_batch_size 2`.

For example, the following settings will let you finetune the model in under 1 hour:

```bash
--devices 4 --train.micro_batch_size 4
```

This script will save checkpoints periodically to the `out_dir` directory. If you are finetuning different models or on your own dataset, you can specify an output directory with your preferred name:

```bash
litgpt finetune adapter \
  --data Alpaca \
  --out_dir out/adapter/my-model-finetuned
```

or for Adapter V2

```bash
litgpt finetune adapter_v2 \
  --data Alpaca \
  --out_dir out/adapter_v2/my-model-finetuned
```

If your GPU does not support `bfloat16`, you can pass the `--precision 32-true` argument.
For instance, to fine-tune on MPS (the GPU on modern Macs), you can run

```bash
litgpt finetune adapter \
  --data Alpaca \
  --out_dir out/adapter/my-model-finetuned \
  --precision 32-true
```

Note that `mps` as the accelerator will be picked up automatically by Fabric when running on a modern Mac.

### Quantization

Optionally, finetuning using quantization can be enabled via the `--quantize` flag, for example using the 4-bit NormalFloat data type:

```bash
litgpt finetune adapter --quantize "bnb.nf4"
```

or using adapter_v2 with double-quantization:

```bash
litgpt finetune adapter_v2 --quantize "bnb.nf4-dq"
```

For additional benchmarks and resource requirements, please see the [Resource Tables](resource-tables.md).

## Test the model

You can test the finetuned model with your own instructions by running:

```bash
litgpt generate adapter \
    --prompt "Recommend a movie to watch on the weekend." \
    --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

or for Adapter V2

```bash
litgpt generate adapter_v2 \
    --prompt "Recommend a movie to watch on the weekend." \
    --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

Output:

```text
A good movie to watch on the weekend would be The Lion King, since it's a classic family film that everyone can enjoy...
```

If your GPU supports `bfloat16`, the script will automatically use it.

## Tune on your dataset

You can easily train on your own instruction dataset saved in JSON format.

1. Create a JSON file in which each row holds one instruction-response pair.
   A row has an entry for 'instruction', 'input', and 'output', where 'input' is optional and can be
   the empty string if the instruction doesn't require a context. Below is an example json file:

    ```text
    [
        {
            "instruction": "Arrange the given numbers in ascending order.",
            "input": "2, 4, 0, 8, 3",
            "output": "0, 2, 3, 4, 8"
        },
        ...
    ]
    ```

2. Run `litgpt/finetune/adapter.py` or `litgpt/finetune/adapter_v2.py` by passing in the location of your data (and optionally other parameters):

    ```bash
    litgpt finetune adapter \
        --data JSON \
        --data.json_path data/mydata.json \
        --checkpoint_dir checkpoints/tiiuae/falcon-7b \
        --out_dir data/mydata-finetuned
    ```
