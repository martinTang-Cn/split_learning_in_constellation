# Paired Radar-Optical CROMA Satellite Demo

This project simulates four Walker-Delta orbital planes. Slot 1 in every plane
is a radar satellite and slot 2 is an optical satellite. Each pair observes
co-registered samples with one common semantic-segmentation mask.

## Model placement

```text
Radar satellite                 Optical satellite
radar_encoder                   optical_encoder
projR -> radar auxiliary head   projO -> optical auxiliary head
        |                               |
        +---------- patch tokens -------+
                        |
                 Ground station
                  cross_encoder
                fused seg head
             projR / projO distillation
```

The encoders and cross encoder come directly from `pretrain_croma.CROMA`.
`PatchSegmentationHead` is added for downstream segmentation. The pretraining
MAE decoder, random masking, and distributed contrastive loss are not used in
the downstream task.

## Code layout

- `multimodal_croma_demo.py`: command-line entry point and experiment assembly
- `croma_models.py`: CROMA component construction, checkpoint loading, and heads
- `multimodal_data.py`: dataset adapters, paired partitions, and lazy batches
- `datasets.py`: WHU Opt-SAR and Houston 2013 patch datasets
- `pair_contact_scheduler.py`: individual-to-pair contact-window conversion
- `multimodal_sfl.py`: satellite state, local updates, server updates, aggregation
- `multimodal_evaluation.py`: segmentation metrics and CSV output helpers

`pretrain_croma.py` remains unchanged and is treated as the CROMA source model.

## Contact and ISL assumption

For every orbital plane, the pair is considered connected whenever either
satellite has a direct ground-station window. The other satellite uses the
assumed inter-satellite link. Overlapping direct windows are merged.

Both branches transmit simultaneously. With the default
`paired_uplink_mode=parallel_full_rate`, transaction time uses the slower of
the two parallel transfers, not their sum. The ground station processes one
orbital pair at a time.

Features are keyed by `batch_number` and contain `sample_ids`. The ground
station invokes `cross_encoder` only when radar and optical packets have the
same batch and sample IDs.

## Disconnected training

During invisible intervals, each satellite runs
`encoder -> downloaded projection -> auxiliary segmentation head` on the same
paired sample batch. The local optimizer always updates the encoder and the
two-layer convolutional segmentation head, and can optionally update the
downloaded projection copy. This is
controlled by `segmentation_training.freeze_projection_during_disconnection`,
which defaults to `true`; when enabled, the projection remains in the forward
chain but only the encoder and segmentation head are updated. The next
connection overwrites that
local projection copy with the current ground-station projection. The two
physical satellites are logically parallel; a single GPU
evaluates their tensor operations sequentially while their shared pair virtual
clock advances by the slower branch's modeled compute time.

At reconnection, the pair uploads both encoder states, both auxiliary-head
states, and the most recent matched feature batches. The ground station runs
`cross_encoder` on the paired features and uses its output as a detached
teacher: `projR(radar_feature)` and `projO(optical_feature)` are trained with
MSE, while the fused segmentation head keeps its supervised segmentation loss.
After the transaction, the current `projR` and `projO` states are copied to the
radar and optical satellites and are included in downlink-size accounting.
Client aggregation remains an equal arithmetic mean for radar/optical encoder
and auxiliary-head parameters; the ground projection layers are maintained at
the server and are not overwritten by satellite-local updates.

## Demo profile and real data

The checked-in configuration uses a small CROMA profile so the full pipeline
can be validated on one GPU or CPU:

```text
image size: 32 x 32
patch size: 8
encoder dimension: 64
encoder layers: 2
attention heads: 4
```

The demo consumes paired real remote-sensing patches. Configure the dataset
root and modality channel counts before starting a run; no fallback dataset is
generated automatically.

For the original-sized CROMA, change the `croma` section to the dimensions
used by the checkpoint and set `pretrained_checkpoint`. The image side length
must equal:

```text
patch_size * sqrt(num_patches)
```

The `dataset.name` setting accepts `whu_opt_sar` or `houston2013`. Set
`dataset.root_dir` to the source folder; relative paths are resolved from this
project directory. The adapter maps WHU's `sar_patch` and Houston's `lidar` to
radar, and WHU's `optical_patch` and Houston's `hsi` to optical. It uses the
source datasets' `train`/`val` splits, shuffles the train patch indices with
`segmentation_training.seed`, and distributes them across the four plane pairs
with counts differing by at most one. The modalities and mask of a patch always
stay together. The validation split is never assigned to satellites. A batch
is read from disk only when its local training step runs.

For WHU, set `dataset.name` to `whu_opt_sar` and set its `root_dir` to the
folder containing `optical/`, `sar/`, and `lbl/`. The number of segmentation
classes and modality channels are obtained from the selected dataset
automatically: WHU uses 8 classes with 1 radar and 4 optical channels. For
Houston, use `dataset.name: "houston2013"` and point `root_dir` to the folder
with the four contest TIFF files; it uses 15 classes with 1 radar and 144
optical channels.
Houston's unlabeled pixels (`-1`) are ignored in training loss and validation
metrics. The default `dataset.drop_empty: true` excludes fully unlabeled
patches. `dataset.stride` controls Houston patch spacing (`null` means one
patch side); `stride_ratio` and
`num_ratio` control WHU patch sampling.

The patch image side must equal `croma.patch_size * sqrt(croma.num_patches)`;
the default tiny profile is 32 pixels. For a 256-pixel image with 8-pixel
CROMA patches, set `num_patches` to `1024`. This can be expensive on a single
GPU. Checkpoint weights must match the configured image size and modality
channel counts; no spectral-band projection or channel truncation is applied.
The dataset adapters validate these settings before model construction.

Satellite encoder training is controlled by
`segmentation_training.satellite_encoder_training_mode`:
`frozen` keeps both satellite encoders frozen for the whole run, `staged`
freezes them for `pretrained_encoder_warmup_aggregations` aggregations and
then unfreezes only the last `pretrained_encoder_trainable_blocks` Transformer
blocks, and `full` trains all satellite encoder parameters throughout. The
default is `staged`.

## Run

```powershell
cd "D:\Files\code\卫星轨道仿真与训练"
python -m pip install -r requirements.txt
python run_demo.py
```

To run the standard split-learning baseline with the same orbit contacts and
dataset configuration, use:

```powershell
python standard_split_learning_demo.py
```

The baseline reads `croma.pretrained_checkpoint` and supports CROMA
pretraining checkpoints saved with `model_state_dict`. A checkpoint can also
be supplied for one run without editing the config:

```powershell
python standard_split_learning_demo.py --pretrained-checkpoint "D:\path\to\checkpoint.pt"
```

The checkpoint must match the configured `patch_size`, `num_patches`,
`encoder_dim`, and radar/optical channel counts. The segmentation head is
initialized by the downstream experiment because it is not part of CROMA
pretraining.

This baseline performs no training while a satellite pair is disconnected.
At each paired contact it trains the two satellite encoders for
`segmentation_training.recent_smashed_batches` batches, while the ground
station trains the shared cross encoder and segmentation head. Its timestamped
results are written under `../standard_split_learning_demo/`.

`device=auto` selects `cuda:0` when CUDA PyTorch is available and otherwise
uses CPU.

## Outputs

- `outputs/contact_windows.csv`: individual direct satellite contacts
- `outputs/orbit_states.csv`: generated satellite orbit states
- `../multimodal_croma_demo/<timestamp>/config.json`: exact config copy for one run
- `../multimodal_croma_demo/<timestamp>/pair_contact_windows.csv`: ISL-assisted plane-pair contacts
- `../multimodal_croma_demo/<timestamp>/multimodal_local_training_log.csv`: paired offline local updates
- `../multimodal_croma_demo/<timestamp>/multimodal_training_log.csv`: paired transfers and server updates
- `../multimodal_croma_demo/<timestamp>/multimodal_aggregation_log.csv`: equal aggregation records
- `../multimodal_croma_demo/<timestamp>/multimodal_training_summary.json`: pixel accuracy, mIoU, and counts
- `../multimodal_croma_demo/<timestamp>/multimodal_final_checkpoint.pt`: encoders, projections, cross encoder, and heads

Every invocation of `multimodal_croma_demo.py` creates a new timestamped child
directory under the parent directory of this project, so previous multimodal
training results are not overwritten. The
orbit generator continues to write its current input files to `outputs/`.
