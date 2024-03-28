# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Utility functions for training and inference."""
import math
import pickle
import shutil
import sys
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Mapping, Optional, TypeVar, Union

import lightning as L
import torch
import torch.nn as nn
import torch.utils._device
import yaml
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.pytorch.loggers import WandbLogger
from torch.serialization import normalize_storage_type
from typing_extensions import Self

if TYPE_CHECKING:
    from litgpt import GPT, Config


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state.shape)
            else:
                total += p.numel()
    return total


def reset_parameters(module: nn.Module) -> None:
    """Calls `reset_parameters` on the module and all its submodules."""
    for mod in module.modules():
        if callable(getattr(mod, "reset_parameters", None)):
            mod.reset_parameters()


def check_valid_checkpoint_dir(checkpoint_dir: Path, model_filename: str = "lit_model.pth") -> None:
    files = {
        model_filename: (checkpoint_dir / model_filename).is_file(),
        "model_config.yaml": (checkpoint_dir / "model_config.yaml").is_file(),
        "tokenizer.json OR tokenizer.model": (checkpoint_dir / "tokenizer.json").is_file()
        or (checkpoint_dir / "tokenizer.model").is_file(),
        "tokenizer_config.json": (checkpoint_dir / "tokenizer_config.json").is_file(),
    }
    if checkpoint_dir.is_dir():
        if all(files.values()):
            # we're good
            return
        problem = f" is missing the files: {[f for f, exists in files.items() if not exists]!r}"
    else:
        problem = " is not a checkpoint directory"

    # list locally available checkpoints
    available = list(Path("checkpoints").glob("*/*"))
    if available:
        options = "\n --checkpoint_dir ".join([""] + [repr(str(p.resolve())) for p in available])
        extra = f"\nYou have downloaded locally:{options}\n"
    else:
        extra = ""

    error_message = (
        f"--checkpoint_dir {str(checkpoint_dir.absolute())!r}{problem}."
        "\nFind download instructions at https://github.com/Lightning-AI/litgpt/blob/main/tutorials\n"
        f"{extra}\nSee all download options by running:\n litgpt download"
    )
    print(error_message, file=sys.stderr)
    raise SystemExit(1)


class SavingProxyForStorage:
    def __init__(self, obj, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.saver = saver
        if not (isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj)):
            raise TypeError(f"expected storage, not {type(obj)}")

        # this logic is taken from PyTorch 2.0+ torch/serialization.py
        if isinstance(obj, torch.storage.TypedStorage):
            # PT upstream wants to deprecate this eventually...
            storage = obj._untyped_storage
            storage_type_str = obj._pickle_storage_type()
            storage_type = getattr(torch, storage_type_str)
            storage_numel = obj._size()
        else:
            storage = obj
            storage_type = normalize_storage_type(type(obj))
            storage_numel = storage.nbytes()

        storage_key = saver._write_storage_and_return_key(storage)
        location = torch.serialization.location_tag(storage)

        self.storage_info = ("storage", storage_type, storage_key, location, storage_numel)

    def __reduce_ex__(self, protocol_version):
        assert False, "this should be handled with out of band"


class SavingProxyForTensor:
    def __init__(self, tensor, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.reduce_ret_fn, reduce_args = tensor.__reduce_ex__(protocol_version)
        if reduce_args[0] == torch._utils._rebuild_tensor_v2:
            # for Tensors with Python attributes
            (a0, a1, (storage, *a2_other), *other_reduce_args) = reduce_args
            assert isinstance(storage, torch.storage.TypedStorage), "Please check for updates"
            storage_proxy = SavingProxyForStorage(storage, saver, protocol_version=protocol_version)
            self.reduce_args = (a0, a1, (storage_proxy, *a2_other), *other_reduce_args)
        else:
            (storage, *other_reduce_args) = reduce_args
            assert isinstance(storage, torch.storage.TypedStorage), "Please check for updates"
            storage_proxy = SavingProxyForStorage(storage, saver, protocol_version=protocol_version)
            self.reduce_args = (storage_proxy, *other_reduce_args)

    def __reduce_ex__(self, protocol_version):
        if protocol_version != self.protocol_version:
            raise RuntimeError(f"Unexpected protocol version: expected {self.protocol_version}, got {protocol_version}")
        return self.reduce_ret_fn, self.reduce_args


class IncrementalPyTorchPickler(pickle.Pickler):
    def __init__(self, saver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_dtypes = {}
        self.saver = saver
        self.id_map = {}

    # this logic is taken from PyTorch 2.0+ torch/serialization.py
    def persistent_id(self, obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, SavingProxyForStorage):
            return obj.storage_info

        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in self.storage_dtypes:
                    if storage_dtype != self.storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that view the same data as different types"
                        )
                else:
                    self.storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = self.id_map.get(storage._cdata)
            if storage_key is None:
                storage_key = self.saver._write_storage_and_return_key(storage)
                self.id_map[storage._cdata] = storage_key
            location = torch.serialization.location_tag(storage)

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None


class incremental_save:
    def __init__(self, name):
        self.name = name
        self.zipfile = torch._C.PyTorchFileWriter(str(name))
        self.has_saved = False
        self.next_key = 0

    def __enter__(self):
        return self

    def store_early(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return SavingProxyForTensor(tensor, self)
        raise TypeError(f"can only store tensors early, not {type(tensor)}")

    def save(self, obj):
        if self.has_saved:
            raise RuntimeError("have already saved")
        # Write the pickle data for `obj`
        data_buf = BytesIO()
        pickler = IncrementalPyTorchPickler(self, data_buf, protocol=5)
        pickler.dump(obj)
        data_value = data_buf.getvalue()
        self.zipfile.write_record("data.pkl", data_value, len(data_value))
        self.has_saved = True

    def _write_storage_and_return_key(self, storage):
        if self.has_saved:
            raise RuntimeError("have already saved")
        key = self.next_key
        self.next_key += 1
        name = f"data/{key}"
        if storage.device.type != "cpu":
            storage = storage.cpu()
        num_bytes = storage.nbytes()
        self.zipfile.write_record(name, storage.data_ptr(), num_bytes)
        return key

    def __exit__(self, type, value, traceback):
        self.zipfile.write_end_of_file()


T = TypeVar("T")


def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_index: int = -100,
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude

    # lm_head was chunked (we are fine-tuning)
    if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

        # chunk cross entropy
        logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        non_masked_elems = (targets != ignore_index).sum()
        return torch.cat(loss_chunks).sum() / max(1, non_masked_elems)

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

    # lm_head wasn't chunked, chunk cross entropy
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
        for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
    ]
    non_masked_elems = (targets != ignore_index).sum()
    return torch.cat(loss_chunks).sum() / max(1, non_masked_elems)


def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str) -> Dict:
    for checkpoint_name, attribute_name in mapping.items():
        full_checkpoint_name = prefix + checkpoint_name
        if full_checkpoint_name in state_dict:
            full_attribute_name = prefix + attribute_name
            state_dict[full_attribute_name] = state_dict.pop(full_checkpoint_name)
    return state_dict


def get_default_supported_precision(training: bool) -> str:
    """Return default precision that is supported by the hardware: either `bf16` or `16`.

    Args:
        training: `-mixed` or `-true` version of the precision to use

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    from lightning.fabric.accelerators import MPSAccelerator

    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


def load_checkpoint(fabric: L.Fabric, model: nn.Module, checkpoint_path: Path, strict: bool = True) -> None:
    if isinstance(fabric.strategy, FSDPStrategy):
        fabric.load_raw(checkpoint_path, model, strict=strict)
    else:
        state_dict = lazy_load(checkpoint_path)
        state_dict = state_dict.get("model", state_dict)
        model.load_state_dict(state_dict, strict=strict)


def flops_per_param(max_seq_length: int, n_layer: int, n_embd: int, n_params: int) -> int:
    flops_per_token = 2 * n_params  # each parameter is used for a MAC (2 FLOPS) per network operation
    # this assumes that all samples have a fixed length equal to the block size
    # which is most likely false during finetuning
    flops_per_seq = flops_per_token * max_seq_length
    attn_flops_per_seq = n_layer * 2 * 2 * (n_embd * (max_seq_length**2))
    return flops_per_seq + attn_flops_per_seq


def estimate_flops(model: "GPT", training: bool) -> int:
    """Measures estimated FLOPs for MFU.

    Refs:
        * https://ar5iv.labs.arxiv.org/html/2205.05198#A1
        * https://ar5iv.labs.arxiv.org/html/2204.02311#A2
    """
    # using all parameters for this is a naive over estimation because not all model parameters actually contribute to
    # this FLOP computation (e.g. embedding, norm). For this reason, the result will be higher by a fixed percentage
    # (~10%) compared to the measured FLOPs, making those lower but more realistic.
    # For a proper estimate, this needs a more fine-grained calculation as in Appendix A of the paper.
    n_trainable_params = num_parameters(model, requires_grad=True)
    trainable_flops = flops_per_param(
        model.max_seq_length, model.config.n_layer, model.config.n_embd, n_trainable_params
    )
    # forward + backward + gradients (assumes no gradient accumulation)
    ops_per_step = 3 if training else 1
    n_frozen_params = num_parameters(model, requires_grad=False)
    frozen_flops = flops_per_param(model.max_seq_length, model.config.n_layer, model.config.n_embd, n_frozen_params)
    # forward + backward
    frozen_ops_per_step = 2 if training else 1
    return ops_per_step * trainable_flops + frozen_ops_per_step * frozen_flops


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> Self:
        return self


def copy_config_files(source_dir: Path, out_dir: Path) -> None:
    """Copies the specified configuration and tokenizer files into the output directory."""

    config_files = ["generation_config.json", "model_config.yaml"]
    tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]

    for file_name in config_files + tokenizer_files:
        src_path = source_dir / file_name
        if src_path.exists():
            shutil.copy(src_path, out_dir)


def CLI(*args: Any, **kwargs: Any) -> Any:
    from jsonargparse import CLI, set_config_read_mode, set_docstring_parse_options

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    kwargs.setdefault("as_positional", False)

    return CLI(*args, **kwargs)


def save_hyperparameters(function: callable, checkpoint_dir: Path) -> None:
    """Captures the CLI parameters passed to `function` without running `function` and saves them to the checkpoint."""
    from jsonargparse import capture_parser

    # TODO: Make this more robust
    # This hack strips away the subcommands from the top-level CLI
    # to parse the file as if it was called as a script
    known_commands = [
        ("finetune", "full"),
        ("finetune", "lora"),
        ("finetune", "adapter"),
        ("finetune", "adapter_v2"),
        ("pretrain",),
    ]
    for known_command in known_commands:
        unwanted = slice(1, 1 + len(known_command))
        if tuple(sys.argv[unwanted]) == known_command:
            sys.argv[unwanted] = []

    parser = capture_parser(lambda: CLI(function))
    config = parser.parse_args()
    parser.save(config, checkpoint_dir / "hyperparameters.yaml", overwrite=True)


def save_config(config: "Config", checkpoint_dir: Path) -> None:
    config_dict = asdict(config)
    with open(checkpoint_dir / "model_config.yaml", "w") as fp:
        yaml.dump(config_dict, fp)


def parse_devices(devices: Union[str, int]) -> int:
    if devices in (-1, "auto"):
        return torch.cuda.device_count() or 1
    if isinstance(devices, int) and devices > 0:
        return devices
    raise ValueError(f"Devices must be 'auto' or a positive integer, got: {devices!r}")


def choose_logger(
    logger_name: Literal["csv", "tensorboard", "wandb"],
    out_dir: Path,
    name: str,
    log_interval: int = 1,
    resume: Optional[bool] = None,
    **kwargs: Any,
):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name="csv", flush_logs_every_n_steps=log_interval, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name="tensorboard", **kwargs)
    if logger_name == "wandb":
        return WandbLogger(project=name, resume=resume, **kwargs)
    raise ValueError(f"`--logger_name={logger_name}` is not a valid option. Choose from 'csv', 'tensorboard', 'wandb'.")


def get_linear_nonlinear_params(model):
    linear_params = []
    nonlinear_params = []

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            linear_params.extend(list(model.parameters()))
        else:
            nonlinear_params.extend(list(model.parameters()))

    # Make extra sure that there is no overlap
    linear_params = list(set(linear_params) - set(nonlinear_params))
    return linear_params, nonlinear_params
