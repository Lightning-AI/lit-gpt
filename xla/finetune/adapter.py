import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightning as L
import torch
import torch_xla.core.xla_model as xm
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.strategies import XLAFSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from xla.generate.base import generate
from lit_gpt.adapter import GPT, Config, adapter_filter, mark_only_adapter_as_trainable, Block
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import check_valid_checkpoint_dir, chunked_cross_entropy, lazy_load, num_parameters, step_csv_logger
from scripts.prepare_alpaca import generate_prompt
from xla.utils import rank_print

eval_interval = 200
save_interval = 200
eval_iters = 50
log_interval = 1
devices = XLAAccelerator.auto_device_count()
# change this value to force a maximum sequence length
override_max_seq_length = None

# Hyperparameters
learning_rate = 3e-3
batch_size = 1
micro_batch_size = 1
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
):
    if devices > 1:
        strategy = XLAFSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="sharded",  # change to "sharded" in multi-host environments where the filesystem is not shared
            sequential_save=True,
        )
    else:
        strategy = "auto"
    logger = step_csv_logger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger)
    rank_print(fabric, hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir)


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path):
    check_valid_checkpoint_dir(checkpoint_dir)

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")

    config = Config.from_name(name=checkpoint_dir.name, adapter_start_layer=0)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    rank_print(fabric, f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=False):
        model = GPT(config)
    with lazy_load(checkpoint_path) as checkpoint:
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
    train(fabric, model, optimizer, train_data, val_data, checkpoint_dir, out_dir, speed_monitor)
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
    speed_monitor: SpeedMonitor,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    max_seq_length, longest_seq_length, longest_seq_ix = get_max_seq_length(train_data)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        mark_only_adapter_as_trainable(meta_model)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        rank_print(fabric, f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        # this assumes that all samples have a fixed length equal to the longest sequence length
        # which is most likely false during finetuning
        x = torch.randint(0, 1, (micro_batch_size, longest_seq_length))
        measured_flops = measure_flops(meta_model, x)
        rank_print(fabric, f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    xm.mark_step()
    for iter_num in range(max_iters):
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(
            fabric, train_data, longest_seq_length, longest_seq_ix if iter_num == 0 else None
        )

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, max_seq_length=max_seq_length, lm_head_chunk_size=128)
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

        t1 = time.perf_counter()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (iter_num + 1) * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
        )
        if iter_num % log_interval == 0:
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
            speed_monitor.eval_end(t1)
            rank_print(fabric, f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_adapter_checkpoint(fabric, model, checkpoint_path)


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
    max_returned_tokens = len(encoded) + 100
    output = generate(
        model, idx=encoded, max_returned_tokens=max_returned_tokens, max_seq_length=max_returned_tokens, temperature=0.8
    )
    output = tokenizer.decode(output)
    rank_print(fabric, output)

    model.reset_cache()

    model.train()
    return val_loss


def get_batch(
    fabric: L.Fabric, data: List[Dict], longest_seq_length: int, longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

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


def get_max_seq_length(data: List[Dict]) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # support easy override at the top of the file
    return (
        override_max_seq_length if isinstance(override_max_seq_length, int) else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )


def save_adapter_checkpoint(fabric, model, file_path: Path):
    rank_print(fabric, f"Saving adapter weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": adapter_filter})


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
