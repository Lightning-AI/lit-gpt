# LLM Evaluation

&nbsp;

## Using lm-evaluation-harness

You can evaluate LitGPT using [EleutherAI's lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) framework with a large number of different evaluation tasks.

You need to install the `lm-eval` framework first:

```bash
pip install lm_eval
```

&nbsp;

### Evaluating LitGPT base models

Suppose you downloaded a base model that we want to evaluate. Here, we use the `microsoft/phi-2` model:

```bash
litgpt download --repo_id microsoft/phi-2
```

The download command above will save the model to the `checkoints/microsoft/phi-2` directory, which we can
specify in the following evaluation command:


```
litgpt evaluate \
  --checkpoint_dir checkpoints/microsoft/phi-2/ \
  --batch_size 4 \
  --out_dir evaluate_model/
```

Please note that the `litgpt evaluate` command run an internal model conversion. 
This is only necessary the first time you want to evaluate a model. To skip the conversion, 
when you want to evaluate a model a second time, you can pass the `--skip_conversion true` argument:

```
litgpt evaluate \
  --checkpoint_dir checkpoints/microsoft/phi-2/ \
  --batch_size 4 \
  --out_dir evaluate_model/ \
  --skip_conversion true
```

&nbsp;

> [!TIP]
> By default, `ligpt evaluate` will evaluate a model on all Open LM Leaderboard tasks, which corresponds
> to the setting `--tasks 
> "hellaswag,gsm8k,truthfulqa_mc2,mmlu,winogrande,arc_challenge"`. 

> [!TIP]
> The evaluation may take a long time, and for testing purpoes, you may want to reduce the number of tasks
> or set a limit for the number of examples per task, for example, `--limit 10`.

A list of supported tasks can be found [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md).




&nbsp;

### Evaluating LoRA-finetuned LLMs

No further conversion is necessary when evaluating LoRA-finetuned models as the `finetune lora` command already prepares the necessary merged model files:

```bash
litgpt finetune lora \
  --checkpoint_dir checkpoints/microsoft/phi-2 \
  --out_dir lora_model
```

&nbsp;

```bash
litgpt evaluate \
  --checkpoint_dir lora_model/final \
  --batch_size 4 \
  --out_dir evaluate_model/ \
```
