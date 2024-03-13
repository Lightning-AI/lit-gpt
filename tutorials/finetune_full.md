# Finetuning the whole model

If you are interested in parameter-efficient finetuning, check out [finetune_adapter.md](finetune_adapter.md). In contrast to parameter-efficient finetuning, this "full" approach finetunes all model parameters, which is substantially more expensive. It may only be recommended as a baseline for comparison studies.

## Preparation

The steps here only need to be done once:

1. Follow the instructions in the [README](../README.md) to install the dependencies.
2. Download and convert the weights following our [guide](download_model_weights.md).

LitGPT provides common datasets for finetuning, such as Alpaca, LIMA, Dolly, and more.
You can optionally [prepare your own dataset](#tune-on-your-dataset).
For more information about dataset preparation, also see the [prepare_dataset.md](./prepare_dataset.md) tutorial.

## Running the finetuning

```bash
litgpt finetune full \
  --data Alpaca \
  --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

Finetuning the falcon-7b model requires at least 8 GPUs with ~40 GB memory each.

You can speed up training by passing the `devices` argument to the script to utilize more GPUs if available.
Depending on the available GPU memory, you can also tune the `micro_batch_size` parameter to utilize the GPU efficiently.

This script will save checkpoints periodically to the `out_dir` directory. If you are finetuning different models or on your own dataset, you can specify an output directory with your preferred name:

```bash
litgpt finetune full \
  --data Alpaca \
  --out_dir out/full/my-model-finetuned
```

If your GPU does not support `bfloat16`, you can pass the `--precision 32-true` argument.
For instance, to fine-tune on MPS (the GPU on modern Macs), you can run

```bash
litgpt finetune full \
  --data Alpaca \
  --out_dir out/full/my-model-finetuned \
  --precision 32-true
```

Note that `mps` as the accelerator will be picked up automatically by Fabric when running on a modern Mac.

## Test the model

You can test the finetuned model with your own instructions by running:

```bash
litgpt generate full \
    --prompt "Recommend a movie to watch on the weekend." \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b \
    --finetuned_path out/full/my-model-finetuned/lit_model_finetuned.pth
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

2. Run `litgpt/finetune/full.py` by passing in the location of your data (and optionally other parameters):

    ```bash
    litgpt finetune full \
        --data JSON \
        --data.json_path data/mydata.json \
        --checkpoint_dir checkpoints/tiiuae/falcon-7b \
        --out_dir data/mydata-finetuned
    ```
