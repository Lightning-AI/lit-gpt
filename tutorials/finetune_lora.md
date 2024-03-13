# Finetuning with LoRA / QLoRA

[Low-rank adaption (LoRA)](https://arxiv.org/abs/2106.09685) is a technique to approximate the update to the linear layers in a LLM with a low-rank matrix factorization. This significantly reduces the number of trainable parameters and speeds up training with little impact on the final performance of the model.
We demonstrate this method by instruction-finetuning LitGPT StableLM 3B on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset on a **single RTX 3090 (24GB) GPU** with CUDA 11.8.

&nbsp;

## Preparation

The steps here only need to be done once:

1. Follow the instructions in the [README](../README.md) to install the dependencies.
2. Download and convert the weights and save them in the `./checkpoints` folder.
   Weights can be downloaded following the instructions in the [download_model_weights](download_model_weights.md) documentation:

LitGPT provides common datasets for finetuning, such as Alpaca, LIMA, Dolly, and more.
You can optionally [prepare your own dataset](#tune-on-your-dataset).
For more information about dataset preparation, also see the [prepare_dataset.md](./prepare_dataset.md) tutorial.

&nbsp;

## Running the Finetuning

```bash
litgpt finetune lora --data Alpaca
```

The finetuning requires at least one GPU with ~24 GB memory (RTX 3090).

This script will save checkpoints periodically to the folder `out/`.

> [!NOTE]
> LoRA can be applied to not only `query`, `key` or `value` matrices, but also to `projection`, `mlp` and classification `head`.
> According to [QLoRA](https://arxiv.org/abs/2305.14314) paper (section 4): "LoRA on all linear transformer block layers are required to match full finetuning performance".
> By default LoRA is applied only to the `query` and `value` matrices. In order to apply LoRA to other weight matrices - change the arguments to `litgpt/finetune/lora.py` accordingly.

Optionally, finetuning using 4-bit quantization (as in QLoRA) can be enabled via the `--quantize` flag, for example using the 4-bit NormalFloat data type:

```bash
litgpt finetune lora --quantize "bnb.nf4"
```

and optionally with double-quantization:

```bash
litgpt finetune lora --quantize "bnb.nf4-dq"
```

The table below lists a comparison with different settings on a StableLM 3B model finetuned with LoRA on Alpaca for 1,000 iterations using a microbatch size of 1:

| Settings                                    | Training Memory | Training Time |  Inference Memory |
|---------------------------------------------|-----------------|---------------|-------------------|
| Default (bf16-mixed)                        | 26.92 GB        | 1.34 min      | 21.43 GB          |
| --precision bf16-true                       | 9.69 GB         | 1.24 min      | 7.30 GB           |
| --precision bf16-true --quantize bnb.nf4    | 6.35 GB         | 1.82 min      | 3.20 GB           |
| --precision bf16-true --quantize bnb.nf4-dq | 6.19 GB         | 1.87 min      | 3.04 GB           |

The advantages of QLoRA-style quantization are more pronounced in larger models, such as Llama 2 7B. The table below summarizes the results for Llama 2 7B on Alpaca for 1,000 iterations using a microbatch size of 1:

| Settings                                    | Training Memory  | Training Time | Inference Memory |
|---------------------------------------------|------------------|---------------|------------------|
| Default (bf16-mixed)                        | OutOfMemoryError | N/A           | 40.21 GB         |
| --precision bf16-true                       | 21.30 GB         | 2.36 min      | 13.52 GB         |
| --precision bf16-true --quantize bnb.nf4    | 14.14 GB         | 3.68 min      | 4.57 GB          |
| --precision bf16-true --quantize bnb.nf4-dq | 13.84 GB         | 3.83 min      | 4.26 GB          |

For additional benchmarks and resource requirements, please see the [Resource Tables](resource-tables.md).

&nbsp;

## Test the Model

You can test the finetuned model with your own instructions by running:

```bash
litgpt generate lora \
  --prompt "Recommend a movie to watch on the weekend."
```

Output:

```text
I would recommend the movie The Martian (2015). It is a sci-fi movie starring Matt Damon that follows the story of...
```

If your GPU supports `bfloat16`, you can additionally pass `--precision "bf16-true"` to bring the memory consumption down to ~7.6 GB for StableLM-3B (versus ~15.2  GB for `--precision "32-full"`). In addition, you may use quantization methods, for example `--precision "bf16-true" --quantize "bnb.nf4"` brings the memory consumption further down to ~4.4 GB for StableLM-3B.

&nbsp;

## Tune on Your Dataset

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

2. Run `litgpt/finetune/lora.py` by passing in the location of your data (and optionally other parameters):

    ```bash
    litgpt finetune lora \
        --data JSON \
        --data.json_path data/mydata.json \
        --checkpoint_dir checkpoints/tiiuae/falcon-7b \
        --out_dir data/mydata-finetuned
    ```


&nbsp;

## Merging LoRA Weights

Finetuning a model with LoRA generates a `lit_model.pth.lora` file. This file exclusively contains the LoRA weights, which has is much smaller than the original model checkpoint to conserve storage space. If desired, there is the option to merge these LoRA weights directly into the original model's checkpoint, which creates a full `lit_model.pth` checkpoint. The advantage of this merging process is to streamline inference operations, as it eliminates the need to dynamically incorporate the LoRA weights during runtime, which can improve inference speed.

For example, after finetuning a model using LoRA with the following command:

```bash
litgpt finetune lora \
  --checkpoint_dir "checkpoints/stabilityai/stablelm-base-alpha-3b/" \
  --train_data_dir data/mydata --val_data_dir data/mydata/ \
  --out_dir "out/lora/stablelm-base-alpha-3b/"
```

This code will produce a `lit_model.pth.lora` file in the specified output directory, containing only the LoRA weights. To merge these LoRA weights with the original model checkpoint, you can use the `merge_lora.py` script as follows:

```bash
litgpt merge_lora \
  --checkpoint_dir "out/lora/stablelm-base-alpha-3b/final"
```

Executing this script results in the creation of a full `lit_model.pth` checkpoint that can be used with the `generate/base.py` or `chat/base.py` scripts for inference:

```bash
litgpt generate base \
  --checkpoint_dir "out/lora/stablelm-base-alpha-3b/final"
```

Similarly, you can evaluate the model using the `eval/lm_eval_harness.py` script (see the [evaluation](evaluation.md) tutorial for more information):

```bash
python eval/lm_eval_harness.py \
  --checkpoint_dir "out/lora/stablelm-base-alpha-3b/final" \
  --precision "bf16-true" \
  --save_filepath "results.json"
```
