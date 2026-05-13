# diffusion-distillation

Few-step distillation for flow-matching text-to-image models.

## Contents

- **DMD**: decoupled CA + DM loss, fake critic with TTUR (LoRA student and fake adapters)

## Structure

```
train.py                 entrypoint, --trainer picks the trainer class
args/                    argparse groups, YAML --config sets defaults
configs/                 dmd.yaml
trainers/base.py         accelerator, optimizer, loop, checkpoints
trainers/dmd.py          DecoupledDMDT2ITrainer
losses/dmd.py            decoupled DMD loss, fake score loss
schedulers/              inference scheduler and timesteps
models/                  Fibo 2.0 transformer, Qwen3-VL text encoder, VAE
lora/adapters.py         student and fake LoRA adapters
train_utils/             packing, checkpoints, LR schedule, runtime
data/                    webdataset latent shards, per-rank allocation JSON
```

## Run

```bash
pip install -r requirements.txt
accelerate launch train.py --config configs/dmd.yaml --json_data_input_path data_input.json
```
