import datetime
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import lightning as L
import torch
from auto_gptq.modeling._base import BaseGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.modeling._utils import autogptq_post_init
from lightning.fabric.plugins.precision.utils import _ClassReplacementContextManager

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config
from lit_gpt.model import Block
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import check_valid_checkpoint_dir, load_checkpoint


# `GPTForAutoGPTQ` and `BlockForAutoGPTQ` are only needed for quantization process.
# These classes have changed `forward` methods, so they can accept and provide such arguments as `attention_mask`
# to "behave" like HuggingFace models and thus become compatible with AutoGPTQ.
class GPTForAutoGPTQ(GPT):
    # Changes to make it compatible with AutoGPTQ:
    # - "input_ids" instead of "idx"
    # - **kwargs store "attention_mask" (that we don't need), but AutoGPTQ provides
    def forward(self, input_ids: torch.Tensor, input_pos: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        """For AutoGPTQ this forward pass is needed only to capture arguments (through a forward hook attached to the
        first layer) that will be send to the first Transformer block. That means it will be called only once."""

        T = input_ids.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")
        x = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)

        # - LayerHijacker (inside the quantize method) doesn't expect all these arguments, but expects kwargs,
        #   so we can provide them as keyword arguments
        self.transformer.h[0](x, cos=self.cos[:T], sin=self.sin[:T], mask=None, input_pos=input_pos, **kwargs)


class BlockForAutoGPTQ(Block):
    # Changes to make it compatible with AutoGPTQ:
    # - accept "attention_mask" that is provided by AutoGPTQ, but don't use it
    def forward(self, *args: Any, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> tuple[torch.Tensor]:
        output = super().forward(*args, **kwargs)
        # HF TransformerBlock returns at least (hidden_state,)
        return (output,)


@dataclass
class QuantizeConfig(BaseQuantizeConfig):
    kernel: Optional[str] = None
    marlin_cached: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.validate_config()

    def validate_config(self, kernel: Optional[str] = None) -> None:
        """Check if the selected config is supported with the kernel.

         Kernel    Supported precisions
                   ┌───────────────┐
                   ┆ 2 ┆ 3 ┆ 4 ┆ 8 ┆
                   ├---------------┤
         cuda_old  ┆ ✔ ┆ ✔ ┆ ✔ ┆ ✔ ┆
         cuda      ┆ ✔ ┆ ✔ ┆ ✔ ┆ ✔ ┆
         exllama   ┆   ┆   ┆ ✔ ┆   ┆
         exllamav2 ┆   ┆   ┆ ✔ ┆   ┆
         triton    ┆ ✔ ┆   ┆ ✔ ┆ ✔ ┆
         marlin    ┆   ┆   ┆ ✔ ┆   ┆
                   └───────────────┘

        Args:
            kernel: override the kernel from the config.
        """
        kernel = kernel or self.kernel
        if (kernel == "triton" and self.bits == 3) or (kernel in ("exllama", "exllamav2", "marlin") and self.bits != 4):
            raise ValueError(f"Kernel '{kernel}' doesn't support {self.bits}bit precision.")
        if kernel == "marlin":
            if self.group_size not in (-1, 128):
                raise ValueError(f"Kernel Marlin doesn't support group_size of {self.group_size}, only -1 or 128.")
            if self.desc_act:
                raise ValueError("Kernel Marlin doesn't support enabled desc_act.")
            if not self.sym:
                raise ValueError("Kernel Marlin doesn't support disable `sym` (symmetric quantization).")

    @classmethod
    def load_config(cls, path: Path) -> "QuantizeConfig":
        if not path.is_file():
            raise ValueError(f"Quantize config is not in `{path.parent}`.")

        return cls(**json.loads(path.read_text()))

    def save_config(self, path: Path) -> None:
        with open(path, "w") as fp:
            json.dump(asdict(self), fp, indent=4)

    def __str__(self) -> str:
        return str(
            {
                "bits": self.bits,
                "group_size": self.group_size,
                "damp_percent": self.damp_percent,
                "desc_act": self.desc_act,
                "static_groups": self.static_groups,
                "sym": self.sym,
                "true_sequential": self.true_sequential,
                "kernel": self.kernel,
            }
        )


class AutoGPTQ(BaseGPTQForCausalLM):
    def __init__(
        self,
        model: GPTForAutoGPTQ,
        quantize_config: BaseQuantizeConfig,
        quantized: bool = False,
        is_triton_backend: bool = False,
        **kwargs: Any,
    ):
        """A wrapper around Lit-GPT model to perform AutoGPTQ quantization.

        Only layers inside a transformer block can be quantized.

        Args:
            model: model to quantize.
            quantized: whether the model is already quantized.
            quantize_config: config containing parameters for quantization.
            is_triton_backend: if True - Triton kernel is used for "packing" weights.
        """
        model.config.model_type = None  # compatibility with AutoGPTQ
        super().__init__(model, quantized, quantize_config, is_triton_backend, **kwargs)

        # name of a layer with transformer blocks
        self.layers_block_name = "transformer.h"

        # names of the layers that are executed before a transformer blocks
        # (won't be quantized)
        self.outside_layer_modules = ["transformer.wte"]

        # Names of linear layers inside of a transformer block that will be quantized.
        # There are four sub lists, the modules in each of them can be seen as one operation.
        # The order should be the order they are truly executed.
        # Usually they are:
        # - Attention QKV projection
        # - Attention output projection
        # - MLP project input
        # - MLP project output
        # Each layer undergoes quantization using inputs that have passed through the previously quantized layers.

        self.inside_layer_modules = [["attn.attn"], ["attn.proj"]]

        mlp_class = self.model.config._mlp_class
        if mlp_class == "GptNeoxMLP":
            self.inside_layer_modules.extend([["mlp.fc"], ["mlp.proj"]])
        elif mlp_class == "LLaMAMLP":
            self.inside_layer_modules.extend([["mlp.fc_1", "mlp.fc_2"], ["mlp.proj"]])
        elif mlp_class == "LLaMAMoE":
            # AutoGPTQ doesn't quantize "gate" layer
            # [
            #   [mlp.experts.0.fc_1, ..., mlp.experts.0.fc2, ...],
            #   [mlp.experts.0.proj],
            # ]
            self.inside_layer_modules.extend(
                [
                    [
                        f"mlp.experts.{expert_idx}.{attr}"
                        for attr in ("fc_1", "fc_2")
                        for expert_idx in range(self.model.config.n_expert)
                    ],
                    [f"mlp.experts.{expert_idx}.proj" for expert_idx in range(self.model.config.n_expert)],
                ]
            )
        else:
            raise ValueError(f"MLP class `{mlp_class}` is not yet supported by AutoGPTQ")

        self.lm_head_name = "lm_head"

    def convert_to_quantized(
        self,
        kernel: Literal["cuda_old", "cuda", "exllama", "exllamav2", "triton", "marlin"],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Replace linear layers with QuantLinear from the selected kernel.

        Only layers inside a transformer block that are specified in `inside_layer_modules` will be replaced.

        Args:
            kernel: from which kernel use QuantLinear class for replacement.
            device: if provided the QuantLinear class will be placed there.
        """

        # Check if the kernel is compatible with the config used for quantization
        self.quantize_config.validate_config(kernel)

        QuantLinear = importlib.import_module(f"auto_gptq.nn_modules.qlinear.qlinear_{kernel}").QuantLinear

        inside_layer_modules = tuple(sum(self.inside_layer_modules, []))
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.endswith(inside_layer_modules):
                parent_name, _, attribute = name.rpartition(".")
                parent_module = self.model.get_submodule(parent_name)
                new_layer = QuantLinear(
                    infeatures=module.in_features,
                    outfeatures=module.out_features,
                    bits=self.quantize_config.bits,
                    group_size=self.quantize_config.group_size,
                    bias=(module.bias is not None),
                    weight_dtype=module.weight.dtype,
                )
                device = device or next(module.parameters()).device
                new_layer = new_layer.to(device)
                new_layer.device = device
                parent_module.__setattr__(attribute, new_layer)

    def convert_quantized_to_marlin(self, quantized_model_dir: Path) -> None:
        """Convert model with QuantLinear layers to layers from Marlin kernel.

        Marlin kernel has a different structure of weights, hence we need to convert
        quantized weights into a supported format.

        Since the process of conversion might take a while, this method saves (caches)
        converted weights in the same folder with the quantized weights.

        Args:
            quantized_model_dir: Marlin cache will be put in the same folder with the quantized weights.
        """

        # select QuantLinear that was used during quantization
        QuantLinear = importlib.import_module(
            f"auto_gptq.nn_modules.qlinear.qlinear_{self.quantize_config.kernel}"
        ).QuantLinear

        # if there are no modules with QuantLinear that was used during quantization, then
        # most likely the model has already been converted
        if not any(isinstance(module, QuantLinear) for module in self.model.modules()):
            return

        from auto_gptq.utils.marlin_utils import _validate_marlin_compatibility, convert_to_marlin

        unsupported_reason = _validate_marlin_compatibility(self.quantize_config)
        if unsupported_reason is not None:
            raise ValueError(
                "The model can not be converted to use the Marlin kernel for the following reason: "
                f"{unsupported_reason}, which is not supported by Marlin kernel."
            )

        self.model.config.quantization_config = {}  # required by AutoGPTQ
        convert_to_marlin(self.model, QuantLinear, self.quantize_config, repack=True)

        marlin_cache_path = quantized_model_dir / "marlin_cache.pth"
        torch.save(self.model.state_dict(), marlin_cache_path)
        self.quantize_config.marlin_cached = True
        self.quantize_config.save_config(quantized_model_dir / "quantize_config.json")

    def post_init(self, max_input_length: Union[None, int] = None) -> None:
        """Obligatory initialization of kernel's buffers."""
        autogptq_post_init(self.model, use_act_order=self.quantize_config.desc_act, max_input_length=max_input_length)

    def strip_bias(self) -> None:
        """Drop `bias` if it's not selected in the model's config.

        AutoGPTQ always creates bias.
        """

        if not self.model.config.bias:
            for module in self.model.transformer.modules():
                if hasattr(module, "QUANT_TYPE"):  # all QuantLinear have this attribute
                    module.bias = None

    # in AutoGPTQ method `to` only expects `device`
    def to(self, device: Union[str, torch.device], dtype: torch.dtype) -> "AutoGPTQ":
        self.model.to(device=device, dtype=dtype)
        return self


def main(
    *,
    data_dir: Path = Path("data/alpaca"),
    n_samples: int = 1024,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    output_path: Optional[Path] = None,
    bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.01,
    desc_act: bool = True,
    static_groups: bool = True,
    sym: bool = True,
    true_sequential: bool = True,
    batch_size: int = 32,
    use_triton: bool = False,
) -> None:
    check_valid_checkpoint_dir(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    if output_path is None:
        output_path = checkpoint_dir / f"quantized/{bits}bit/lit_model_gptq.pth"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load and prepare calibration data ---
    calibration_data = torch.load(data_dir / "test.pt")
    # AutoGPTQ expects a list of dicts with two keys in each: "input_ids" and "attention_mask".
    # Since Lit model doesn't need "attention_mask", we can "fake" it.
    calibration_data = [
        {"input_ids": record["input_ids"], "attention_mask": [1] * len(record["input_ids"])}
        for record in calibration_data[:n_samples]
    ]

    # --- Create a model ---
    config = Config.from_json(checkpoint_dir / "lit_config.json")
    # The model is loaded into a CPU RAM and each layer, that is about to be quantized,
    # is moved to a GPU by the AutoGPTQ itself.
    # "torch.float16" precision is preferred during the inference, so it's better to
    # have the model in this precision during the quantization.
    fabric = L.Fabric(accelerator="cpu", precision="16-true")
    with fabric.init_module(empty_init=False), _ClassReplacementContextManager(
        {"lit_gpt.model.Block": BlockForAutoGPTQ}
    ):
        model = GPTForAutoGPTQ(config)
    model.eval()
    load_checkpoint(fabric, model, checkpoint_path)

    # --- Update model's config ---
    # AutoGPTQ wants to retrieve these 3 parameters, but Lit's config doesn't have
    model.config.pad_token_id = None  # ._prepare_examples_for_quantization
    model.config.eos_token_id = Tokenizer(checkpoint_dir).eos_id  # _prepare_examples_for_quantization
    model.config.use_cache = False  # for quantization it's disabled anyway

    # --- Quantize the model ---
    quantize_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=desc_act,
        static_groups=static_groups,
        sym=sym,
        true_sequential=true_sequential,
        kernel="triton" if use_triton else None,
    )
    autogptq = AutoGPTQ(model, quantize_config=quantize_config)

    quantize_time = time.perf_counter()
    autogptq.quantize(calibration_data, batch_size=batch_size, use_triton=use_triton)
    quantize_time = int(time.perf_counter() - quantize_time)
    fabric.print(f"Quantization time: {datetime.timedelta(seconds=quantize_time)}")

    # AutoGPTQ creates bias weights even if they are not needed.
    # Trim them before saving (per config)
    autogptq.strip_bias()
    # Save the model
    torch.save(model.state_dict(), output_path)

    # Save quantize config with the used kernel - will be reused during inference
    quantize_config.kernel = next(module.QUANT_TYPE for module in model.modules() if hasattr(module, "QUANT_TYPE"))
    # guard for "cuda-old"
    quantize_config.kernel = quantize_config.kernel.replace("-", "_")
    quantize_config.save_config(output_path.with_name("quantize_config.json"))


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")

    CLI(main)
