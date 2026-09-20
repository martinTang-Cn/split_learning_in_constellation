"""Paired radar-optical datasets, deterministic partitioning, and lazy batching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


DATASET_SPEC = {
    "whu_opt_sar": {"num_classes": 8, "radar_channels": 1, "optical_channels": 4},
    "houston2013": {"num_classes": 15, "radar_channels": 1, "optical_channels": 144},
}


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    num_classes: int
    ignore_index: int
    radar_channels: int
    optical_channels: int
    image_size: int


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    validation: Dataset
    metadata: DatasetMetadata


class RealPairedDatasetAdapter(Dataset):
    """Map each real dataset's modality order to ``radar, optical, mask, id``."""

    def __init__(self, source: Dataset, dataset_name: str) -> None:
        self.source = source
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        sample = self.source[index]
        if not isinstance(sample, (tuple, list)) or len(sample) < 3:
            raise ValueError(
                f"{self.dataset_name} sample {index} must contain at least three values"
            )
        first, second, mask = sample[:3]
        if self.dataset_name == "whu_opt_sar":
            optical, radar = first, second
        elif self.dataset_name == "houston2013":
            optical, radar = first, second  # hsi is optical; lidar is radar.
        else:
            raise ValueError(f"Unsupported adapter dataset: {self.dataset_name}")
        return radar.float(), optical.float(), mask.long(), index


class PairedBatchSequence(Sequence):
    """A deterministic batch plan that reads image tensors only when requested."""

    def __init__(
        self, dataset: Dataset, indices: Sequence[int], batch_size: int,
        epochs: int, seed: int, shuffle: bool = True,
    ) -> None:
        if batch_size <= 0 or epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        self.dataset = dataset
        self.batch_specs: list[tuple[int, int, list[int]]] = []
        generator = torch.Generator().manual_seed(seed)
        batch_number = 0
        base_indices = [int(index) for index in indices]
        for epoch in range(1, epochs + 1):
            if shuffle and base_indices:
                order = torch.randperm(len(base_indices), generator=generator).tolist()
                epoch_indices = [base_indices[position] for position in order]
            else:
                epoch_indices = base_indices
            for start in range(0, len(epoch_indices), batch_size):
                self.batch_specs.append(
                    (epoch, batch_number, epoch_indices[start:start + batch_size])
                )
                batch_number += 1

    def __len__(self) -> int:
        return len(self.batch_specs)

    def __getitem__(self, position: int):
        epoch, batch_number, indices = self.batch_specs[position]
        samples = [self.dataset[index] for index in indices]
        if not samples:
            raise IndexError(position)
        radar, optical, masks, sample_ids = zip(*samples)
        return (
            epoch,
            batch_number,
            torch.tensor(sample_ids, dtype=torch.long),
            torch.stack(radar),
            torch.stack(optical),
            torch.stack(masks),
        )


def _resolve_root(root_value, project_dir: Path) -> Path:
    if not root_value:
        raise ValueError("dataset.root_dir must be set when using a real dataset")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = project_dir / root
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")
    return root


def _dataset_name(config) -> str:
    dataset_config = config.get("dataset", {})
    if not dataset_config.get("name"):
        raise ValueError("dataset.name must be set to whu_opt_sar or houston2013")
    raw_name = str(dataset_config["name"]).lower()
    aliases = {
        "whu": "whu_opt_sar",
        "whuoptsarpatchdataset": "whu_opt_sar",
        "houston": "houston2013",
        "houston2013patchdataset": "houston2013",
    }
    return aliases.get(raw_name, raw_name)


def build_dataset_bundle(config, project_dir: Path, pair_count: int) -> DatasetBundle:
    """Construct a real train/validation dataset and inspect one sample."""
    training, model_config = config["segmentation_training"], config["croma"]
    dataset_config = config.get("dataset", {})
    name = _dataset_name(config)
    if name not in {"whu_opt_sar", "houston2013"}:
        raise ValueError(
            "dataset.name must be one of: whu_opt_sar, houston2013"
        )
    if pair_count <= 0:
        raise ValueError("No satellite pairs have contact windows in the simulation")
    image_size = int(model_config["patch_size"]) * math.isqrt(int(model_config["num_patches"]))
    seed = int(training["seed"])

    root = _resolve_root(dataset_config.get("root_dir"), project_dir)
    normalize = bool(dataset_config.get("normalize", True))
    norm_type = str(dataset_config.get("norm_type", "standard"))
    from datasets import Houston2013PatchDataset, WHUOptSarPatchDataset

    if name == "whu_opt_sar":
        common = dict(
            root_dir=str(root),
            train_ratio=float(dataset_config.get("train_ratio", 0.8)),
            patch_size=image_size,
            stride_ratio=float(dataset_config.get("stride_ratio", 0.9)),
            random_seed=seed,
            num_ratio=float(dataset_config.get("num_ratio", 1.0)),
            normalize=normalize,
            norm_type=norm_type,
        )
        train = RealPairedDatasetAdapter(
            WHUOptSarPatchDataset(split="train", **common), name
        )
        validation = RealPairedDatasetAdapter(
            WHUOptSarPatchDataset(split="val", **common), name
        )
        ignore_index = -100
    else:
        stride_value = dataset_config.get("stride")
        common = dict(
            root_dir=str(root),
            patch_size=image_size,
            stride=image_size if stride_value is None else int(stride_value),
            drop_empty=bool(dataset_config.get("drop_empty", True)),
            normalize=normalize,
            norm_type=norm_type,
            return_coords=False,
        )
        train = RealPairedDatasetAdapter(
            Houston2013PatchDataset(split="train", **common), name
        )
        validation = RealPairedDatasetAdapter(
            Houston2013PatchDataset(split="val", **common), name
        )
        ignore_index = -1
    if len(train) < pair_count:
        raise ValueError(
            f"Training dataset has {len(train)} samples, fewer than {pair_count} satellite pairs"
        )
    if len(validation) == 0:
        raise ValueError("Validation dataset is empty")
    radar, optical, mask, _ = train[0]
    _validate_sample(radar, optical, mask, image_size)
    validation_radar, validation_optical, validation_mask, _ = validation[0]
    _validate_sample(validation_radar, validation_optical, validation_mask, image_size)
    specification = DATASET_SPEC[name]
    metadata = DatasetMetadata(
        name=name,
        num_classes=specification["num_classes"],
        ignore_index=ignore_index,
        radar_channels=specification["radar_channels"],
        optical_channels=specification["optical_channels"],
        image_size=image_size,
    )
    _validate_channels(radar, optical, metadata, "training")
    _validate_channels(validation_radar, validation_optical, metadata, "validation")
    return DatasetBundle(train=train, validation=validation, metadata=metadata)


def _validate_sample(radar, optical, mask, image_size: int) -> None:
    if radar.ndim != 3 or optical.ndim != 3 or mask.ndim != 2:
        raise ValueError(
            "Each sample must be radar (C,H,W), optical (C,H,W), and mask (H,W); "
            f"got {tuple(radar.shape)}, {tuple(optical.shape)}, {tuple(mask.shape)}"
        )
    expected = (image_size, image_size)
    if tuple(radar.shape[-2:]) != expected or tuple(optical.shape[-2:]) != expected:
        raise ValueError(
            f"Dataset patch size must match CROMA image size {expected}; got "
            f"radar={tuple(radar.shape[-2:])}, optical={tuple(optical.shape[-2:])}"
        )
    if tuple(mask.shape) != expected:
        raise ValueError(f"Mask size must be {expected}; got {tuple(mask.shape)}")
    if not torch.is_floating_point(radar) or not torch.is_floating_point(optical):
        raise ValueError("Radar and optical tensors must be floating point")


def _validate_channels(
    radar: torch.Tensor,
    optical: torch.Tensor,
    metadata: DatasetMetadata,
    split: str,
) -> None:
    actual = (int(radar.shape[0]), int(optical.shape[0]))
    expected = (metadata.radar_channels, metadata.optical_channels)
    if actual != expected:
        raise ValueError(
            f"{metadata.name} {split} sample has radar/optical channels "
            f"{actual}, expected {expected}"
        )


def partition_dataset_indices(
    dataset_size: int, pair_ids: Sequence[str], seed: int,
) -> dict[str, list[int]]:
    """Randomly split sample indices as evenly as possible across satellite pairs."""
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(dataset_size, generator=generator)
    splits = torch.tensor_split(shuffled, len(pair_ids))
    return {
        pair_id: [int(index) for index in split.tolist()]
        for pair_id, split in zip(pair_ids, splits)
    }
