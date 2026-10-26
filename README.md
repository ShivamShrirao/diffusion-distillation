# diffusion-distillation

Few-step distillation for flow-matching text-to-image models.

## Contents

- **DMD**: decoupled CA + DM loss, fake critic with TTUR, full fine-tune, bf16 frozen teachers; per-record teacher CFG; optional second teacher, by coin or routed per segment / below a noise level
- **PDMD**: project the fake critic's error out of the DM direction
- **TDM**: per-segment training, segment-bridge renoise, pseudo-Huber loss, min-SNR and importance-weighted critic, teacher forcing
- **DMDR**: Qwen3-VL pairwise judge reward on the student's x0, both pair orders, exact-text check
- **Edit**: Fibo 2.0 edits through reference images, read by Qwen3-VL and fed to every forward as clean context tokens; edit + T2I runs, one kind per micro-step, per-dataset weights, EDIT/T2I micro-step order; DMDR pairwise edit prompt; DMD and SMD
- **SMD**: segment matching to the teacher's CFG velocities inside the student's segment, no fake critic; pivot walk or a learned Pivot head bends the query path; DMDR optional

## Structure

```
train.py                 entrypoint, --trainer picks the trainer class
args/                    argparse groups, YAML --config sets defaults
configs/                 dmd.yaml, dmd_edit.yaml, smd.yaml, smd_edit.yaml
trainers/base.py         accelerator, optimizer, loop, checkpoints
trainers/dmd.py          DecoupledDMDT2ITrainer
trainers/dmdr.py         DDMDRTrainer (DMD + judge reward)
trainers/dmd_edit.py     DecoupledDMDEditTrainer, DDMDREditTrainer (edit + T2I)
trainers/smd.py          SMDTrainer, SMDEditTrainer (SMD, optional judge reward)
reward/                  stock Qwen3-VL pairwise judge
losses/                  decoupled DMD loss, fake score loss, SMD loss, pseudo-Huber
schedulers/              inference scheduler and timesteps
models/                  Fibo 2.0 transformer (padded and packed forward), SMD Pivot head, Qwen3-VL text encoder, VAE
train_utils/             packing, varlen attention, checkpoints, LR schedule, runtime
data/                    webdataset latent shards, per-rank allocation JSON
```

## Data

`data_input.json` lists webdataset shards of VAE latents per resolution. An edit record also holds
`reference_images` (the encoded images to edit) and `instruction`; its `caption`, of the result, is optional.
A dataset with `"edit": true` holds edit records; `"weight"` scales its share.

## Run

```bash
pip install -r requirements.txt
accelerate launch train.py --config configs/dmd.yaml --json_data_input_path data_input.json
accelerate launch train.py --config configs/smd.yaml --student_init checkpoints/dmd/checkpoint_000400/student
accelerate launch train.py --config configs/dmd_edit.yaml --json_data_input_path data_input.json
accelerate launch train.py --config configs/smd_edit.yaml --student_init checkpoints/dmd_edit/checkpoint_000400/student
```
