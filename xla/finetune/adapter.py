# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightning as L
import torch
import torch_xla.core.xla_model as xm
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import XLAFSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor, measure_flops

from litgpt.adapter import GPT, Block, Config, adapter_filter, mark_only_adapter_as_trainable
from litgpt.tokenizer import Tokenizer
from litgpt.utils import check_valid_checkpoint_dir, chunked_cross_entropy, estimate_flops, lazy_load, num_parameters

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from xla.scripts.prepare_alpaca import generate_prompt
from xla.generate.base import generate
from xla.utils import rank_print, sequential_load_and_fsdp_wrap

eval_interval = 200
save_interval = 200
eval_iters = 100
eval_max_new_tokens = 100
log_interval = 1
devices = XLAAccelerator.auto_device_count()
# the state of very large models will not fit on the system RAM, this flag can alleviate it by loading it on each rank
# sequentially
reduce_cpu_memory_usage_during_load = False

# Hyperparameters
learning_rate = 3e-3
batch_size = 4
micro_batch_size = batch_size
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
epoch_size = 50000  # train dataset size
num_epochs = 5
max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.02
warmup_steps = 2 * (epoch_size // micro_batch_size) // devices // gradient_accumulation_iters  # 2 epochs

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    *,
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    out_dir: Path = Path("out/adapter/alpaca"),
    precision: str = "bf16-true",
) -> None:
    if devices > 1:
        strategy = XLAFSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",  # change to "sharded" in multi-host environments where the filesystem is not shared
            sequential_save=True,
        )
    else:
        strategy = "auto"
    logger = CSVLogger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger)
    rank_print(fabric, hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir)


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path) -> None:
    check_valid_checkpoint_dir(checkpoint_dir)

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")

    config = Config.from_name(name=checkpoint_dir.name, adapter_start_layer=0)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    rank_print(fabric, f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")

    if reduce_cpu_memory_usage_during_load:
        model = sequential_load_and_fsdp_wrap(fabric, lambda: GPT(config), checkpoint_path)
    else:
        with fabric.init_module(empty_init=False):
            model = GPT(config)
        checkpoint = lazy_load(checkpoint_path)
        # strict=False because missing keys due to adapter weights not contained in state dict
        model.load_state_dict(checkpoint, strict=False)

    model = fabric.setup_module(model)
    # mark as trainable only after sharding due to https://github.com/pytorch/xla/pull/5484
    mark_only_adapter_as_trainable(model)
    # these are not correct in the sharding case
    rank_print(fabric, f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    rank_print(fabric, f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=learning_rate)
    optimizer = fabric.setup_optimizers(optimizer)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, model, optimizer, train_data, val_data, checkpoint_dir, out_dir)
    rank_print(fabric, f"Training time: {(time.perf_counter()-train_time):.2f}s")

    # Save the final checkpoint at the end of training
    save_path = out_dir / "lit_model_adapter_finetuned.pth"
    save_adapter_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length = get_longest_seq_length(train_data)
    model.max_seq_length = longest_seq_length
    # to avoid recompilation, this script is configured to pad batches to the `longest_seq_length`
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    with torch.device("meta"):
        meta_model = GPT(model.config)
        mark_only_adapter_as_trainable(meta_model)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `flops_per_batch=estimated_flops` instead
        estimated_flops = estimate_flops(meta_model, training=True) * micro_batch_size
        rank_print(fabric, f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        # this assumes that all samples have a fixed length equal to the longest sequence length
        # which is most likely false during finetuning
        x = torch.randint(0, 1, (micro_batch_size, longest_seq_length))
        forward_fn = lambda: meta_model(x)
        loss_fn = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, forward_fn, loss_fn)
        rank_print(fabric, f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    throughput = ThroughputMonitor(fabric, window_size=50)
    step_count = 0
    total_t0 = time.perf_counter()

    xm.mark_step()
    for iter_num in range(1, max_iters + 1):
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(fabric, train_data, longest_seq_length)

        is_accumulating = iter_num % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            xm.mark_step()
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / gradient_accumulation_iters)
        xm.mark_step()

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
        else:
            xm.mark_step()

        if iter_num % log_interval == 0:
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0,
                batches=iter_num,
                samples=iter_num * micro_batch_size,
                lengths=iter_num * micro_batch_size * longest_seq_length,
                flops=measured_flops * log_interval,
            )
            throughput.compute_and_log(step=iter_num)
            rank_print(
                fabric,
                f"iter {iter_num} step {step_count}:"
                # uncomment to print the loss. this will considerably slow down the iteration times
                # + f" loss {loss.item():.4f},"
                + f" iter time: {(t1 - iter_t0) * 1000:.2f}ms" + (" (optimizer.step)" if not is_accumulating else ""),
            )

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, longest_seq_length)
            t1 = time.perf_counter() - t0
            rank_print(fabric, f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_adapter_checkpoint(fabric, model, checkpoint_path)


# xla does not support `inference_mode`: RuntimeError: Cannot set version_counter for inference tensor
@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, longest_seq_length: int
) -> torch.Tensor:
    rank_print(fabric, "Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    xm.mark_step()
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data, longest_seq_length)
        logits = model(input_ids)
        xm.mark_step()
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
    val_loss = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    rank_print(fabric, instruction)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)
    output = generate(model, encoded, max_returned_tokens=len(encoded) + eval_max_new_tokens, temperature=0.8)
    model.clear_kv_cache()
    output = tokenizer.decode(output)
    rank_print(fabric, output)

    model.train()
    return val_loss


def get_batch(fabric: L.Fabric, data: List[Dict], longest_seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    def pad_right(x, pad_id):
        # pad right using a fixed longest sequence length to avoid recompilation
        n = longest_seq_length - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    x, y = fabric.to_device((x, y))
    return x, y


def get_longest_seq_length(data: List[Dict]) -> int:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    return max(len(d["input_ids"]) for d in data)


def save_adapter_checkpoint(fabric: L.Fabric, model: torch.nn.Module, file_path: Path) -> None:
    rank_print(fabric, f"Saving adapter weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": adapter_filter})


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
