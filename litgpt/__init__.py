# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import re
import logging

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer

from lightning_utilities.core.imports import RequirementCache

_LIGHTNING_AVAILABLE = RequirementCache("lightning>=2.2.0.dev0")
if not bool(_LIGHTNING_AVAILABLE):
    raise ImportError(
        "LitGPT requires lightning nightly. Please run:\n"
        f" pip uninstall -y lightning; pip install litgpt\n{str(_LIGHTNING_AVAILABLE)}"
    )

# Suppress excessive warnings, see https://github.com/pytorch/pytorch/issues/111632
pattern = re.compile(".*Profiler function .* will be ignored")
logging.getLogger("torch._dynamo.variables.torch").addFilter(lambda record: not pattern.search(record.getMessage()))

# Avoid printing state-dict profiling output at the WARNING level when saving a checkpoint
logging.getLogger("torch.distributed.fsdp._optim_utils").disabled = True
logging.getLogger("torch.distributed.fsdp._debug_utils").disabled = True

__all__ = ["GPT", "Config", "PromptStyle", "Tokenizer"]
