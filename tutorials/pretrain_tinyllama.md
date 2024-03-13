# Pretrain TinyLlama

This tutorial will walk you through pretraining [TinyLlama](https://github.com/jzhang38/TinyLlama/).

> [!TIP]
> To get started with zero setup, clone the [TinyLlama studio on Lightning AI](https://lightning.ai/lightning-ai/studios/llm-pretrain-tinyllama-1-1b).

## What's TinyLlama?

[TinyLlama](https://github.com/jzhang38/TinyLlama/) is architecturally the same as Meta AI's LLama 2, but only has 1.1B parameters and is instead trained on multiple epochs on a mix of [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) and [Starcoder](https://huggingface.co/datasets/bigcode/starcoderdata) datasets.

Here is a quick fact sheet:

| Name                          | Description                                                                                                                                                  |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Parameters                    | 1.1B                                                                                                                                                         |
| Model Size                    | Layers: 22, Heads: 32, Query Groups: 4, Embedding Size: 2048, Intermediate Size: 5632                                                                        |
| Sequence Length               | 2048                                                                                                                                                         |
| Learning Rate                 | 4e-4                                                                                                                                                         |
| Learning Rate Schedule        | Cosine with 2000 warmup steps                                                                                                                                |
| Training Data                 | [SlimPajama](https://huggingface.co/datasets/cerebras/slimpajama-627b) (893 GB), [Starcoder](https://huggingface.co/datasets/bigcode/starcoderdata) (290 GB) |
| Combined Dataset Size         | Around 950B tokens                                                                                                                                           |
| Total Tokens During Training  | 3 trillion (3 epochs)                                                                                                                                        |
| Time to complete training     | ~ 4 weeks with 64 A100 GPUs                                                                                                                                  |
| Model FLOPs Utilization (MFU) | 52%                                                                                                                                                          |

(this table was sourced from the author's [README](https://github.com/jzhang38/TinyLlama/))

## Download datasets

You can download the data using git lfs:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com):
sudo apt install git-lfs
```

```bash
git clone https://huggingface.co/datasets/cerebras/slimpajama-627b data/slimpajama-raw
git clone https://huggingface.co/datasets/bigcode/starcoderdata data/starcoderdata-raw
```

Around 1.2 TB of disk space is required to store both datasets.

## Prepare the datasets for training

In order to start pretraining litgpt on it, you need to read, tokenize, and write the data in binary chunks. This will leverage the `litdata` optimization pipeline and streaming dataset.

First, install additional dependencies for preprocessing:

```bash
pip install '.[all]'
```

You will need to have the tokenizer config available:

```bash
litgpt download \
   --repo_id meta-llama/Llama-2-7b-hf \
   --access_token your_hf_token \
   --tokenizer_only true
```

Then, run the preprocessing script for each dataset and split.
You will require **1.1 TB** of disk space for Starcoder and **2.5** TB of space for the SlimPajama dataset.

**Starcoder:**

```bash
python litgpt/data/prepare_starcoder.py \
  --input_dir data/starcoderdata-raw \
  --output_dir data/starcoder \
  --tokenizer_path checkpoints/meta-llama/Llama-2-7b-hf
```

**SlimPajama:**

```bash
python litgpt/data/prepare_slimpajama.py \
  --input_dir data/slimpajama-raw/validation \
  --output_dir data/slimpajama/val \
  --tokenizer_path checkpoints/meta-llama/Llama-2-7b-hf

python litgpt/data/prepare_slimpajama.py \
  --input_dir data/slimpajama-raw/test \
  --output_dir data/slimpajama/test \
  --tokenizer_path checkpoints/meta-llama/Llama-2-7b-hf

python litgpt/data/prepare_slimpajama.py \
  --input_dir data/slimpajama-raw/train \
  --output_dir data/slimpajama/train \
  --tokenizer_path checkpoints/meta-llama/Llama-2-7b-hf
```

If you want to run on a small slice of the datasets first, pass the flag `--fast_dev_run=true` to the commands above.
In the above we are assuming that you will be using the same tokenizer as used in LlaMA/TinyLlama, but any trained [SentencePiece](https://github.com/google/sentencepiece) tokenizer with a 32000 vocabulary size will do here.

## Pretraining

Running the pretraining script with its default settings requires at least 8 A100 GPUs.

```bash
litgpt pretrain --config config_hub/pretrain/tinyllama.yaml
```

The script will save checkpoints periodically to the folder `out/`.
By default, the `pretrain` script will pretrain the model with FSDP in
`bfloat16` mixed precision and gradient accumulation.

Note that `pretrain` is not actually a model-specific training script, so feel free [try other configurations](../config_hub)
or change the model type and size by passing a different string to the model name argument, for example:

```shell
litgpt pretrain --model_name Gemma-2b
```

The currently supported model names are contained in the [config.py](https://github.com/Lightning-AI/litgpt/litgpt/config.py) file.
You can

1) either search this file for lines containing "name =",
2) or run `litgpt download` without additional command line arguments

Keep in mind that training with a single machine will take weeks. To speed up the process, you'll need access to a cluster.
Once you're in a cluster, you can follow [these instructions](https://lightning.ai/docs/fabric/stable/fundamentals/launch.html#launch-on-a-cluster)
to launch the script across machines:

- [Lightning AI](https://lightning.ai/docs/fabric/stable/guide/multi_node/cloud.html)
- [SLURM cluster](https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html)
- [Barebones cluster](https://lightning.ai/docs/fabric/stable/guide/multi_node/barebones.html)
- [MPI](https://lightning.ai/docs/fabric/stable/guide/multi_node/other.html)

The script exposes several hyperparameters you can tweak through the command line.

For instance, `--train.micro_batch_size` should be adjusted so the process will use the available
GPU memory. For more tips to avoid out-of-memory issues, please also see the more detailed
[Dealing with out-of-memory (OOM) errors](oom.md) guide.

Last, logging is kept minimal in the script, but for long-running experiments we recommend switching to a proper experiment tracker.
As an example, we included WandB (set `--logger_name=wandb`) to show how you can integrate any experiment tracking framework.
For reference, [here are the loss curves for our reproduction](https://api.wandb.ai/links/awaelchli/y7pzdpwy).

## Resume training

The checkpoints saved during pretraining contain all the information to resume if needed.
Simply rerun the script with the `--resume` argument added:

```bash
litgpt pretrain \
  --config config_hub/pretrain/tinyllama.yaml \
  --resume out/pretrain/tiny-llama/step-00060500
```
**Important:** Each checkpoint is a directory. Point to the directory, not the 'lit_model.pth' file inside of it.

## Export checkpoints

After training is completed, you can convert the checkpoint to a format that can be loaded for evaluation, inference, finetuning etc.

```bash
litgpt convert pretrained_checkpoint \
  --checkpoint_dir out/pretrain/tiny-llama/step-00060500 \
  --output_dir checkpoints/tiny-llama/final
```

After conversion, the output folder will contain these files:
```
checkpoints/tiny-llama/final
├── model_config.yaml
├── lit_model.pth
├── tokenizer_config.json
├── tokenizer.json
└── tokenizer.model
```

You can then use this checkpoint folder to run [evaluation](evaluation.md), [inference](inference.md), [finetuning](finetune_lora.md) or [process the checkpoint further](convert_lit_models.md).
