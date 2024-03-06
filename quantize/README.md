<p>
    <h1 align="center">Post-training weights-only quantization</h1>
    <h3 align="center">with AutoGPTQ</h3>
</p>

In Lit-GPT post-training quantization is done via integrating [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ): an easy-to-use LLM quantization package with user-friendly APIs, based on GPTQ algorithm (weight-only quantization).

By using GPTQ quantization, you can compress your preferred language model down to 8, 4, 3, or even 2 bits. This compression does not significantly affect performance.

> [!IMPORTANT]
> AutoGPTQ only runs on a GPU.

To use it first install the package:

```bash
pip install auto-gptq
```

### Calibration data

To understand the importance of the calibration data we need to first take a look at what GPTQ algorithm does at a high level.

From the [GPTQ](https://arxiv.org/pdf/2210.17323.pdf) paper we can see that the objective of quantization is to find a matrix of quantized weights $\widehat{W}$ which minimizes the squared error, relative to the full precision layer output.

$$\arg\min_{\widehat{W}} \lVert \mathbf{W}\mathbf{X} - \mathbf{\widehat{W}}\mathbf{X}\rVert_2^2 \tag{1}$$

where:

- $\mathbf{X}$ - The inputs (from the calibration dataset).
- $\mathbf{W}$ - The full precision weights of the model.
- $\mathbf{\widehat{W}}$ - The quantized weights of the model.

The core idea is to take a weight matrix, gradually quantize it block by block, evaluate the difference between the quantized and the original weights and adjust the remaining unquantized weights to make up for the error that was introduced by quantization.

Since the model is quantized by minimizing the difference between the expected output of the original model and the output of the quantized one, you would want to use the data that is representative of what the model is expected to see during inference.

If the model is fine-tuned on a specific domain, then you should use in-domain dataset for quantization. But if the models is expected to handle multiple tasks (a typical case for LLMs), then you should pick a dataset that contains cross-domain tasks, such as instruction-following dataset.

> [!NOTE]
> In the [prepare_dataset.md](../tutorials/prepare_dataset.md) you can find examples of how to create such a dataset.

### Quantization

Quantization is done via

```bash
python quantize/autogptq.py --data_dir data/... --checkpoint_dir checkpoints/... --quantize gptq.int4
```

which accepts following arguments:

| Argument          | Description                                                                                                                                                                                                                                                                                                |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *data_dir*        | Directory where the calibration data resides. Default: "data/alpaca"                                                                                                                                                                                                                                       |
| *n_samples*       | How many samples to use for calibration. Default: 1024                                                                                                                                                                                                                                                     |
| *checkpoint_dir*  | Directory where the original model weights are located. Default: "checkpoints/stabilityai/stablelm-base-alpha-3b"                                                                                                                                                                                          |
| *output_path*     | Path where the quantized weights will be saved. If not provided will be saved in `quantized` folder at the same level as original weights. Default: None                                                                                                                                                   |
| *bits*            | The number of bits to quantize to. Supports 2,3,4 and 8 bits. Default: 4                                                                                                                                                                                                                                   |
| *group_size*      | The less the group size the more precise the model, but requires more memory. Recommended value is `128`, and `-1` uses per-column quantization. Default: 128                                                                                                                                              |
| *damp_percent*    | The percent of the average Hessian diagonal to use for dampening. `0.01` is the default, but `0.1` might result in slightly better accuracy. Default: 0.01                                                                                                                                                 |
| *desc_act*        | (aka `act_order`) Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly speed up inference but the perplexity may become slightly worse. Default: True                                                                                                 |
| *static_groups*   | Determines all group-grids in advance rather than dynamically during quantization. Recommended when using `--desc_act` for more efficient inference. Default: True                                                                                                                                         |
| *sym*             | Whether to use [symmetric quantization](https://huggingface.co/docs/optimum/concept_guides/quantization#symmetric-and-affine-quantization-schemes). Default: False                                                                                                                                         |
| *true_sequential* | Whether to perform sequential quantization even within a single Transformer block. Instead of quantizing the entire block at once, we perform layer-wise quantization. As a result, each layer undergoes quantization using inputs that have passed through the previously quantized layers. Default: True |
| *batch_size*      | The number of samples of calibration data to pass into a model. Default: 32                                                                                                                                                                                                                                |
| *use_triton*      | Set it as True if during inference `Triton` kernel will be use. Default: False                                                                                                                                                                                                                             |

> [!TIP]
> If you encounter such an error during quantitation: </br>
> `The factorization could not be completed because the input is not positive-definite` </br>
> then try to increase `damp_percent` or `num_samples`.

> [!NOTE]
> An explanation of how to load the model can be found in one of the `tutorials/download_*.md` files.

#### Kernels compatibility

If there is a need to use a specific kernel during inference, the arguments above should be selected in a such way that they conform kernel's requirements, since not all kernels support all the precisions.

| Kernel     | 2bit | 3bit | 4bit | 8bit |
|------------|------|------|------|------|
| cuda_old   |  ✔   |  ✔   |  ✔   |  ✔   |
| cuda       |  ✔   |  ✔   |  ✔   |  ✔   |
| exllama    |      |      |  ✔   |      |
| exllamav2  |      |      |  ✔   |      |
| triton     |  ✔   |      |  ✔   |  ✔   |
| marlin     |      |      |  ✔   |      |

Additionally, `Marlin` kernel adds these requirements:

- `Group_size` should be either `128` or `-1`.
- `Desc_act` should be False.
- `Sym` should be True.
- For each weight matrix that will be quantized, the number of `input features` should be divisible by `128` and the number of `output features` - by `256`.

Out of all kernels Marlin is the newest and the fastest, but supports only compute capability >= 8.0 (Ampere generation and newer). On this [Wiki](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) page one can find a table with graphics cards and their compute capabilities.

Benchmarks with kernels can be found [here](https://github.com/huggingface/optimum/tree/main/tests/benchmark#gptq-benchmark).

## Inference

To generate new tokens with the quantized model we can use `generate/base.py` script by providing GPTQ-specific args:

```bash
python generate/base.py --checkpoint_dir ... --quantize gptq.int4 --kernel ...
```

| Argument         | Description                                                                                              |
|------------------|----------------------------------------------------------------------------------------------------------|
| *checkpoint_dir* | Directory where the quantized weight are located.                                                        |
| *quantize*       | `gptq.intX`, where X - number of bits. Select the same number of bits that was used during quantization. |
| *gptq_kernel*    | You can override the kernel that was used during quantization (in most cases it's `exllama`).            |

If kernel is `Marlin`, then the quantized weights will be repacked. Since the process might take a while we utilize caching, which will speed up subsequent script executions.

> [!NOTE]
> GPTQ inference doesn't support model compilation.
