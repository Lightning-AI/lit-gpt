# TPU support

This project utilizes [`Fabric`](https://lightning.ai/docs/fabric/stable), which supports TPUs via [PyTorch XLA](https://github.com/pytorch/xla).

> [!NOTE]
> This guide assumes that you have already set-up your [Google Cloud environment](https://cloud.google.com/run/docs/setup).

To set up a Google Cloud instance with a TPU v4 VM, run the following commands:

```shell
gcloud compute tpus tpu-vm create litgpt --version=tpu-vm-v4-base --accelerator-type=v4-8 --zone=us-central2-b
gcloud compute tpus tpu-vm ssh litgpt --zone=us-central2-b
```

You can also choose a different TPU type. To do so, change the `version`, `accelerator-type`, and `zone` arguments. Find all regions and zones [here](https://cloud.google.com/tpu/docs/regions-zones).

<details>
<summary>Multihost caveats</summary>

TPU v4-8 uses a single host. SSH'ing into the machine and running commands manually will only work when using a single host (1 slice in the TPU pod).
In multi-host environments, such as larger TPU pod slices, it's necessary to launch all commands on all hosts simultaneously to avoid hangs.
For local development, it is advisable to upload a zip file containing all your current changes and execute it inside the VM from your personal computer:

```shell
# Zip the local directory, excluding large directories from the zip. You may want to keep them.
zip -r local_changes.zip . -x  ".git/*" "checkpoints/*" "data/*" "out/*"
# Copy the .zip file to the TPU VM
gcloud compute tpus tpu-vm scp --worker=all local_changes.zip "litgpt:~"
# Unzip on each host
gcloud compute tpus tpu-vm ssh litgpt --worker=all --command="cd ~; unzip -q -o local_changes.zip"

# Example of a typical workflow
gcloud compute tpus tpu-vm ssh tmp --worker=all --command="cd ~; bash install_dependencies.sh"
gcloud compute tpus tpu-vm ssh tmp --worker=all --command="cd ~; bash prepare_checkpoints.sh"
gcloud compute tpus tpu-vm ssh tmp --worker=all --command="cd ~; bash run_desired_script.sh"

# This will allow you to kill all python processes on all workers
gcloud compute tpus tpu-vm ssh tmp --worker=all --command="pkill -e python"
```

Notice how the commands to install the environment and prepare checkpoints need to be run on all workers, since the filesystem
for each worker (host) is not shared.

For the rest of this tutorial, it will be assumed that it is being run on a single host for simplicity.

</details>

Once inside the machine, clone the repository and install the dependencies:

```shell
git clone https://github.com/Lightning-AI/litgpt
cd litgpt
pip install .
```

Install Optimized BLAS:

```shell
sudo apt update
sudo apt install libopenblas-dev
```

Since LitGPT requires a torch version newer than torch 2.0.0, manually install nightly builds of torch and torch_xla:

```shell
pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl
pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl
```

While computations will run by default using the new PjRT runtime, it is recommended to set the following environment variables:

```shell
export ALLOW_MULTIPLE_LIBTPU_LOAD=1
export PJRT_DEVICE=TPU
```

> [!NOTE]
> An extensive guide on setup and available options can be found [here](https://cloud.google.com/tpu/docs/v4-users-guide).

Since a new machine was created, you may need to download pretrained weights.
They can be copied to the machine using `gcloud compute tpus tpu-vm scp`, or you can follow the steps described in our [downloading guide](download_model_weights.md).

It is also recommended to set up a persistent disk from which to load checkpoints.
Follow [this guide](https://cloud.google.com/tpu/docs/setup-persistent-disk#setting_up_a_tpu_vm_and_a_persistent_disk) to do so.
Read-write disks are not supported in multihost VM setups, so persistent disks cannot be used to save checkpoints in that case.
Persistent disks can still be useful in read-only mode to load pretrained weights before finetuning or inference.
In multihost settings, FSDP will save checkpoint shards per host and consolidate them into a single checkpoint.
For safekeeping, it is recommended to upload the consolidated checkpoints to a Google Cloud bucket.
Alternatively, you can use the `scp` command to transfer these checkpoints from the TPU VM periodically, although this is not implemented in our scripts.

## Inference

This project provides custom versions of the regular recipes to run with XLA in the `xla` directory.
To generate text, use the following command:

```shell
python3 xla/generate/base.py --prompt "Hello, my name is" --num_samples 3
```

For the first generation, this command will take around 17 seconds as XLA needs to compile the graph.
Subsequent generations will take around 2 seconds.

## Fine-tuning

To get started fine-tuning Falcon 7B with adapter, run the following command:

```shell
python3 xla/scripts/prepare_alpaca.py --checkpoint_dir checkpoints/tiiuae/falcon-7b

python3 xla/finetune/adapter.py --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true
```

<details>
<summary>Multihost caveats</summary>

This script is configured to save "full" checkpoints, which isn't possible on multihost TPU VMs.
Here's how you can consolidate them together into a single one after training with `state_dict_type="sharded"`:

```shell
path_to_shards="out/adapter/alpaca/lit_model_adapter_finetuned"
mkdir -p $path_to_shards
workers=4  # 4 hosts
for ((i = 0; i < workers; i++)); do
  # aggregate all shards locally
  gcloud compute tpus tpu-vm scp --worker=$i "litgpt:${path_to_shards}/*" "${path_to_shards}/" --zone us-central2-b
done
# copy all shards to all workers
gcloud compute tpus tpu-vm scp --worker=all ${path_to_shards}/* "litgpt:${path_to_shards}/" --zone us-central2-b
# consolidate the shards in each worker
gcloud compute tpus tpu-vm ssh tmp --worker=all --command="python -m torch_xla.distributed.fsdp.consolidate_sharded_ckpts --ckpt_prefix ${path_to_shards}/checkpoint --ckpt_suffix '_rank-*-of-*.pth' --save_path ${path_to_shards}.pth" --zone us-central2-b
```

</details>

Since the TPU VM host RAM is limited (200 GB), we implement a technique to sequentially load and shard the checkpoint that can be enabled by
setting `reduce_cpu_memory_usage_during_load = True`. This is necessary to load falcon-40b.

To generate text with the adapter fine-tuned model weights, use the following command:

```shell
python3 xla/generate/adapter.py --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --adapter_path out/adapter/alpaca/lit_model_adapter_finetuned.pth
```

> **Warning**
> Remember to delete your instance when you are done.
>
> ```shell
> gcloud compute tpus tpu-vm delete litgpt --zone=us-central2-b
> ```

## Computational Performance

Using the [adapter finetuning script](finetune/adapter.py) and XLA's FSDP implementation, a 49.57% MFU was achieved with Falcon 7B on a v4-32 (micro batch size 7), and a 39.67% MFU was achieved with Falcon 40B on a v4-512 (micro batch size 3) at a fixed 1034 maximum sequence length.

Since the TPU VM host has limited system memory (RAM) compared to device memory (HBM), specific techniques were implemented to limit peak RAM usage when loading the model and pretrained weights before sharding, as well as when saving sharded checkpoints.
A v4 chip has 32 GiB HBM, so with 4 devices per host (4 * 32 = 128 GiB HBM), each host has 188 GiB RAM, which is shared across the devices.
Therefore, any RAM allocation over 188/4 = 47 GiB would exceed the host's RAM capacity.
A ~24B parameter model on CPU (with half precision) would be the largest possible model under this setup without the techniques used in our scripts.
