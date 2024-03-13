## Converting LitGPT weights to Hugging Face Transformers

LitGPT weights need to be converted to a format that Hugging Face understands with a [conversion script](../litgpt/scripts/convert_lit_checkpoint.py) before our scripts can run.

We provide a helpful script to convert models LitGPT models back to their equivalent Hugging Face Transformers format:

```sh
litgpt convert from_litgpt \
    --checkpoint_dir checkpoint_dir \
    --output_dir converted_dir
```

These paths are just placeholders, you will need to customize them based on which finetuning or pretraining script you ran and its configuration.

### Loading converted LitGPT checkpoints into transformers


For example,

```bash
cp checkpoints/repo_id/config.json converted/config.json
```

Then, you can load the checkpoint file in a Python session as follows:

```python
import torch
from transformers import AutoModel


state_dict = torch.load("output_dir/model.pth")
model = AutoModel.from_pretrained(
    "output_dir/", local_files_only=True, state_dict=state_dict
)
```

Alternatively, you can also load the model without copying the `config.json` file as follows:

```python
model = AutoModel.from_pretrained("online_repo_id", state_dict=state_dict)
```



### Merging LoRA weights

Please note that if you want to convert a model that has been fine-tuned using an adapter like LoRA, these weights should be [merged](../litgpt/scripts/merge_lora.py) to the checkpoint prior to converting.

```sh
litgpt merge_lora \
    --checkpoint_dir path/to/lora/checkpoint_dir
```

<br>
<br>

# A finetuning and conversion tutorial

This section contains a reproducible example for finetuning a LitGPT model and converting it back into a HF `transformer` model.

1. Download a model of interest:

For convenience, we first specify an environment variable (optional) to avoid copy and pasting the whole path:

```bash
export repo_id=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

Instead of using TinyLlama, you can replace the `repo_id` target with any other model repository 
specifier that is currently supported by LitGPT. You can get a list of supported repository specifier
by running `litgpt/scripts/download.py` without any additional arguments.

Then, we download the model we specified via `$repo_id` above:

```bash
litgpt download --repo_id $repo_id
```

2. Finetune the model:


```bash
export finetuned_dir=out/lit-finetuned-model

litgpt finetune lora \
   --checkpoint_dir checkpoints/$repo_id \
   --out_dir $finetuned_dir \
   --train.epochs 1 \
   --data Alpaca
```

3. Merge LoRA weights:

Note that this step only applies if the model was finetuned with `lora.py` above and not when `full.py` was used for finetuning.

```bash
litgpt merge_lora \
    --checkpoint_dir $finetuned_dir/final
```


4. Convert the finetuning model back into a HF format:

```bash
litgpt convert from_litgpt \
   --checkpoint_dir $finetuned_dir/final/ \
   --output_dir out/hf-tinyllama/converted \
```


5. Load the model into a `transformers` model:

```python
import torch
from transformers import AutoModel

state_dict = torch.load('out/hf-tinyllama/converted/model.pth')
model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", state_dict=state_dict)
```
