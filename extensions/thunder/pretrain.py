# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import os
import pprint
import sys
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union, List

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from typing_extensions import Literal

from litgpt import Tokenizer
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import DataModule, TinyLlama
from litgpt.model import GPT, CausalSelfAttention, Config, LLaMAMLP, Block
from litgpt.utils import (
    CLI,
    CycleIterator,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
    save_hyperparameters,
)

# support running without installing as a package
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from unsloth.executor import unsloth_ex  # import for registration


def setup(
    model_name: Optional[str] = None,
    model_config: Optional[Config] = None,
    out_dir: Path = Path("out/pretrain"),
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Path] = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        learning_rate=4e-4,
        weight_decay=1e-1,
        beta1=0.9,
        beta2=0.95,
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    devices: Union[int, str] = "auto",
    tokenizer_dir: Optional[Path] = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "tensorboard",
    seed: int = 42,
    compiler: Optional[Literal["thunder", "torch"]] = "thunder",
    executors: Optional[List[str]] = ("sdpa", "torchcompile", "nvfuser", "torch"),
    strategy: Literal["auto", "ddp", "fsdp"] = "fsdp",
):
    """Pretrain a model.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Mutually exclusive with
            ``model_config``.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
            Useful for continued pretraining. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        devices: How many devices/GPUs to use. Uses all GPUs by default.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        compiler: If desired, the compiler/JIT to use.
        executors: If using Thunder, the executors to enable.
        strategy: If desired, the strategy to use.
    """
    hparams = locals()
    data = TinyLlama() if data is None else data
    if model_config is not None and model_name is not None:
        raise ValueError("Only one of `model_name` or `model_config` can be set.")
    elif model_config is None and model_name is None:
        model_name = "tiny-llama-1.1b"
    config = Config.from_name(model_name) if model_config is None else model_config
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    logger = choose_logger(
        logger_name, out_dir, name=f"pretrain-{config.name}", resume=resume, log_interval=train.log_interval
    )

    if devices > 1:
        if compiler == "thunder":
            if strategy == "fsdp":
                from extensions.thunder.strategies import ThunderFSDPStrategy

                strategy = ThunderFSDPStrategy(
                    sharding_strategy="ZERO3", bucketing_strategy="BLOCK", state_dict_type="full", jit=False,
                )
            elif strategy == "ddp":
                from extensions.thunder.strategies import ThunderDDPStrategy

                strategy = ThunderDDPStrategy(jit=False)
        else:
            if strategy == "fsdp":
                strategy = FSDPStrategy(
                    auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="FULL_SHARD"
                )
    else:
        strategy = "auto"
    fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-true", loggers=[logger])
    fabric.launch()

    if compiler is not None:
        global forward_and_loss
        forward_and_loss = jit(forward_and_loss, executors) if compiler == "thunder" else torch.compile(forward_and_loss)

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(hparams)

    main(
        fabric,
        devices,
        seed,
        initial_checkpoint_dir,
        resume,
        config,
        data,
        out_dir,
        tokenizer_dir,
        tokenizer,
        train,
        eval,
        compiler,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Optional[Path],
    resume: Union[bool, Path],
    config: Config,
    data: DataModule,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    tokenizer: Optional[Tokenizer],
    train: TrainArgs,
    eval: EvalArgs,
    compiler: Optional[Literal["thunder", "torch"]],
) -> None:
    validate_args(train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
    if train.max_seq_length:
        model.max_seq_length = train.max_seq_length

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    model = fabric.setup(model)
    if compiler == "thunder":
        # avoid `Tensor.register_hook` which is unsupported
        model._register_backward_hook = lambda *_: None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train.learning_rate,
        weight_decay=train.weight_decay,
        betas=(train.beta1, train.beta2),
        fused=True,
    )
    optimizer = fabric.setup_optimizers(optimizer)

    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train, model.max_seq_length)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    state = {
        "model": model,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }

    if resume is True:
        resume = max(out_dir.rglob("step-*/*.pth"), key=(lambda p: int(p.parent.name.split("-")[1])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    fit(fabric, devices, state, train_dataloader, val_dataloader, out_dir, tokenizer_dir, train, eval)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    train: TrainArgs,
    eval: EvalArgs,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    validate(fabric, model, val_dataloader, max_iters=2)  # sanity check
    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))
        model_fwd = lambda: meta_model(x)
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * model.max_seq_length
    max_iters = max_tokens_per_device // tokens_per_iter
    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices)
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices), sync_on_compute=False).to(
        fabric.device
    )
    fabric.barrier()
    total_t0 = time.perf_counter()
    val_loss = "n/a"

    warmup_iters = train.lr_warmup_steps * train.gradient_accumulation_iters(devices)
    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(train.learning_rate, state["iter_num"], warmup_iters, max_iters, train.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.max_seq_length].contiguous().long()
        targets = train_data[:, 1 : (model.max_seq_length + 1)].contiguous().long()

        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        # FIXME: need the jitted model reference to pass it here. How do I get it?
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            loss = forward_and_loss(model, input_ids, targets)
            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        running_loss.update(loss.detach())

        if not is_accumulating:
            # THUNDER unsupported: https://github.com/Lightning-AI/lightning-thunder/issues/2357
            # fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        if state["iter_num"] % log_iter_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(state["iter_num"] * train.micro_batch_size * model.max_seq_length),
            )
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * train.micro_batch_size * model.max_seq_length,
                "total_tokens": (state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size),
                "learning_rate": lr,
            }
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
            val_loss = val_loss.item()
            td = time.perf_counter() - t0

            fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=state["iter_num"] - 1)
            fabric.barrier()

        if train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0:
            checkpoint_file = out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth"
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
            fabric.save(checkpoint_file, state)
            if fabric.global_rank == 0:
                save_hyperparameters(setup, checkpoint_file.parent)
                if tokenizer_dir is not None:
                    copy_config_files(tokenizer_dir, checkpoint_file.parent)
                save_config(model.config, checkpoint_file.parent)


def forward_and_loss(model: nn.Module, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids)
    # disable chunk_size to enable the unsloth cross entropy kernel
    loss = chunked_cross_entropy(logits, targets, chunk_size=0)
    return loss


@torch.no_grad()
def validate(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int) -> torch.Tensor:
    fabric.barrier()
    fabric.print("Validating ...")
    model.eval()

    losses = []
    for k, batch in enumerate(val_dataloader):
        if k >= max_iters:
            break
        input_ids = batch[:, 0 : model.max_seq_length].contiguous().long()
        targets = batch[:, 1 : (model.max_seq_length + 1)].contiguous().long()
        loss = forward_and_loss(model, input_ids, targets)
        losses.append(loss)

    val_loss = torch.stack(losses).mean()
    model.train()
    fabric.barrier()
    return val_loss


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs, block_size: int
) -> Tuple[DataLoader, DataLoader]:
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=block_size)
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)


def init_out_dir(out_dir: Path) -> Path:
    if not out_dir.is_absolute() and "LIGHTNING_ARTIFACTS_DIR" in os.environ:
        return Path(os.getenv("LIGHTNING_ARTIFACTS_DIR")) / out_dir
    return out_dir


def validate_args(train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume) -> None:
    issues = []
    unsupported = [(train, ["max_steps", "epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if initial_checkpoint_dir and resume:
        issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")
    if issues:
        raise ValueError("\n".join(issues))


def jit(fn: Callable, executors: List[str]) -> Any:
    assert executors is not None
    import thunder
    from strategies.utils import _validate_executors

    executors = _validate_executors(executors)
    return thunder.jit(fn, executors=executors)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    CLI(setup)
