"""Evaluation and output helpers for the paired CROMA experiment."""

from __future__ import annotations

import csv

import torch

from multimodal_data import PairedBatchSequence


def write_csv(path, rows, fieldnames=None) -> None:
    if not rows and fieldnames is None:
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_global(
    test_data, radar_worker, optical_worker, cross_encoder, ground_head,
    attention_bias, global_states, config, device,
):
    radar_worker.load_state_dict(global_states["radar_encoder"])
    optical_worker.load_state_dict(global_states["optical_encoder"])
    radar_worker.eval()
    optical_worker.eval()
    cross_encoder.eval()
    ground_head.eval()
    training = config["segmentation_training"]
    batch_size = int(training["batch_size"])
    metadata = config.get("dataset_metadata", {})
    num_classes = int(metadata["num_classes"])
    ignore_index = int(metadata.get("ignore_index", -100))
    batches = PairedBatchSequence(
        test_data, range(len(test_data)), batch_size, 1, 0, shuffle=False
    )
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for _, _, _, radar, optical, targets in batches:
        image_size = int(radar.shape[-1])
        radar_tokens = radar_worker(radar.to(device), attention_bias, None)
        optical_tokens = optical_worker(optical.to(device), attention_bias, None)
        predictions = ground_head(cross_encoder(radar_tokens, optical_tokens, attention_bias), image_size).argmax(dim=1).cpu()
        valid = targets != ignore_index
        indices = targets[valid].reshape(-1) * num_classes + predictions[valid].reshape(-1)
        if indices.numel() == 0:
            continue
        confusion += torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    intersection = confusion.diag().float()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    if confusion.sum() == 0:
        raise ValueError("Validation dataset contains no labeled pixels")
    valid = union > 0
    mean_iou = float((intersection[valid] / union[valid]).mean().item())
    pixel_accuracy = float(intersection.sum().item() / max(1, confusion.sum().item()))
    radar_worker.train()
    optical_worker.train()
    cross_encoder.train()
    ground_head.train()
    return pixel_accuracy, mean_iou, confusion.tolist()
