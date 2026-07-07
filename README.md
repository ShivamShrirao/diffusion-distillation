# diffusion-distillation

Few-step distillation for flow-matching text-to-image models.

## Contents

- **DMD**: decoupled CA + DM loss, fake critic with TTUR (LoRA adapters or full fine-tune)
- **TDM**: per-segment training, segment-bridge renoise
- **DMDR**: Qwen3-VL pairwise judge reward on the student's x0
- **Edit**: Fibo 2.0 edits through reference images, read by Qwen3-VL and fed to every forward as clean context tokens

## Structure

```
train.py                 entrypoint, --trainer picks the trainer class
args/                    argparse groups, YAML --config sets defaults
configs/                 dmd.yaml, dmd_edit.yaml
trainers/base.py         accelerator, optimizer, loop, checkpoints
trainers/dmd.py          DecoupledDMDT2ITrainer
trainers/dmdr.py         DDMDRTrainer (DMD + judge reward)
trainers/dmd_edit.py     DecoupledDMDEditTrainer (edit records)
reward/                  stock Qwen3-VL pairwise judge
losses/dmd.py            decoupled DMD loss, fake score loss
schedulers/              inference scheduler and timesteps
models/                  Fibo 2.0 transformer, Qwen3-VL text encoder, VAE
lora/adapters.py         student and fake LoRA adapters
train_utils/             packing, checkpoints, LR schedule, runtime
data/                    webdataset latent shards, per-rank allocation JSON
```

## Data

`data_input.json` lists webdataset shards of VAE latents per resolution. An edit record also holds
`reference_images` (the encoded images to edit) and `instruction`; its `caption`, of the result, is optional.

## Run

```bash
pip install -r requirements.txt
accelerate launch train.py --config configs/dmd.yaml --json_data_input_path data_input.json
accelerate launch train.py --config configs/dmd_edit.yaml --json_data_input_path data_input.json
```
