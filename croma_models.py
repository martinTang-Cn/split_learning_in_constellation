"""Downstream segmentation modules built from the pretrained CROMA model."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from pretrain_croma import CROMA


class FeatureProjection(nn.Module):
    """Two-layer token-wise MLP used to distill ground cross-modal features."""

    def __init__(self, feature_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Expected token features with shape (B, N, C), got {tuple(features.shape)}"
            )
        return self.network(features)


class PatchSegmentationHead(nn.Module):
    """Map CROMA patch tokens to a full-resolution mask with two convolutions."""

    def __init__(self, encoder_dim: int, num_classes: int, num_patches: int) -> None:
        super().__init__()
        self.grid_size = math.isqrt(num_patches)
        if self.grid_size * self.grid_size != num_patches:
            raise ValueError("num_patches must be a perfect square")
        self.num_patches = num_patches
        self.convolutional_head = nn.Sequential(
            nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(encoder_dim, num_classes, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, image_size: int) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != self.num_patches:
            raise ValueError(
                f"Expected tokens of shape (B, {self.num_patches}, C), got {tuple(tokens.shape)}"
            )
        feature_map = tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[-1], self.grid_size, self.grid_size
        )
        feature_map = F.interpolate(
            feature_map, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        return self.convolutional_head(feature_map)


def load_checkpoint_if_configured(croma: CROMA, config, project_dir: Path) -> str:
    checkpoint_value = config["croma"].get("pretrained_checkpoint")
    if not checkpoint_value:
        return "random_initialization"
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_dir / checkpoint_path
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = croma.load_state_dict(state, strict=False)
    return (
        f"{checkpoint_path} (missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)})"
    )


def pretrained_encoder_schedule(config) -> dict[str, int | str]:
    """Return the configured satellite encoder training policy."""
    training = config["segmentation_training"]
    mode = str(training.get("satellite_encoder_training_mode", "staged")).lower()
    if mode not in {"frozen", "staged", "full"}:
        raise ValueError(
            "satellite_encoder_training_mode must be one of: frozen, staged, full"
        )
    return {
        "mode": mode,
        "warmup_aggregations": max(
            0, int(training.get("pretrained_encoder_warmup_aggregations", 5))
        ),
        "trainable_blocks": max(
            0, int(training.get("pretrained_encoder_trainable_blocks", 1))
        ),
    }


def configure_satellite_encoder_trainability(
    radar_encoder: nn.Module,
    optical_encoder: nn.Module,
    config,
    global_version: int,
) -> str:
    """Apply the configured satellite encoder freeze/unfreeze policy."""
    schedule = pretrained_encoder_schedule(config)
    mode = str(schedule["mode"])
    if mode == "full":
        radar_encoder.requires_grad_(True)
        optical_encoder.requires_grad_(True)
        return "full_train"

    if mode == "frozen":
        radar_encoder.requires_grad_(False)
        optical_encoder.requires_grad_(False)
        return "frozen"

    warmup_aggregations = int(schedule["warmup_aggregations"])
    trainable_blocks = int(schedule["trainable_blocks"])
    if global_version < warmup_aggregations or trainable_blocks == 0:
        radar_encoder.requires_grad_(False)
        optical_encoder.requires_grad_(False)
        return "pretrained_frozen"

    for encoder in (radar_encoder, optical_encoder):
        encoder.requires_grad_(False)
        layers = getattr(getattr(encoder, "transformer", None), "layers", None)
        if layers is None:
            raise AttributeError("CROMA ViT encoder does not expose transformer.layers")
        for block in layers[-trainable_blocks:]:
            block.requires_grad_(True)
    return f"pretrained_last_{trainable_blocks}_blocks"


def build_croma_components(config, device, project_dir: Path):
    model_config = config["croma"]
    dataset_metadata = config["dataset_metadata"]
    radar_channels = int(dataset_metadata["radar_channels"])
    optical_channels = int(dataset_metadata["optical_channels"])
    croma = CROMA(
        patch_size=model_config["patch_size"],
        encoder_dim=model_config["encoder_dim"],
        encoder_layers=model_config["encoder_layers"],
        attention_heads=model_config["attention_heads"],
        decoder_dim=model_config["decoder_dim"],
        decoder_layers=model_config["decoder_layers"],
        total_channels=radar_channels + optical_channels,
        num_patches=model_config["num_patches"],
        opt_channels=optical_channels,
        radar_channels=radar_channels,
    )
    checkpoint_status = load_checkpoint_if_configured(croma, config, project_dir)
    return (
        croma.radar_encoder.to(device),
        croma.optical_encoder.to(device),
        croma.cross_encoder.to(device),
        croma.attn_bias.to(device),
        checkpoint_status,
    )
