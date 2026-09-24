"""Centralized CROMA segmentation baseline paced by satellite contact windows.

One shared model sees the same per-plane data partitions as the distributed
experiments. Each paired contact schedules a fixed number of full-model updates;
there are no satellite models, communication transactions, or aggregations.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import torch
from torch import nn

from croma_models import PatchSegmentationHead, build_croma_components
from multimodal_data import PairedBatchSequence, build_dataset_bundle, partition_dataset_indices
from multimodal_evaluation import evaluate_global, write_csv
from multimodal_sfl import clone_state
from orbit_model import parse_utc
from pair_contact_scheduler import build_pair_contacts, load_raw_contacts


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_DIR.parent / "centralized_multimodal_demo"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but is unavailable")
    return device


def make_pair_batches(pair_contacts, dataset, training):
    """Match the original experiment's deterministic per-plane sample streams."""
    planes = sorted({row["plane"] for row in pair_contacts})
    pair_ids = [next(row["pair_id"] for row in pair_contacts if row["plane"] == plane) for plane in planes]
    indices = partition_dataset_indices(len(dataset), pair_ids, int(training["seed"]))
    return {
        pair_id: PairedBatchSequence(
            dataset, indices[pair_id], int(training["batch_size"]),
            int(training["epochs"]), int(training["seed"]) + plane,
        )
        for plane, pair_id in zip(planes, pair_ids)
    }


def run_training(config, raw_contacts, output_dir: Path):
    started = time.perf_counter()
    training, model_config = config["segmentation_training"], config["croma"]
    torch.manual_seed(int(training["seed"]))
    torch.set_num_threads(1)
    device = select_device(training["device"])
    pair_contacts = build_pair_contacts(raw_contacts, parse_utc(config["simulation"]["epoch_utc"]))
    pair_count = len({row["pair_id"] for row in pair_contacts})
    dataset_bundle = build_dataset_bundle(config, PROJECT_DIR, pair_count)
    metadata = dataset_bundle.metadata
    config["dataset_metadata"] = {
        "name": metadata.name,
        "ignore_index": metadata.ignore_index,
        "num_classes": metadata.num_classes,
        "radar_channels": metadata.radar_channels,
        "optical_channels": metadata.optical_channels,
    }
    radar_encoder, optical_encoder, cross_encoder, attention_bias, checkpoint_status = (
        build_croma_components(config, device, PROJECT_DIR)
    )
    head = PatchSegmentationHead(
        model_config["encoder_dim"], metadata.num_classes, model_config["num_patches"]
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=metadata.ignore_index)
    weight_decay = float(training.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW([
        {"params": radar_encoder.parameters(), "lr": float(training["encoder_learning_rate"])},
        {"params": optical_encoder.parameters(), "lr": float(training["encoder_learning_rate"])},
        {"params": cross_encoder.parameters(), "lr": float(training["server_learning_rate"])},
        {"params": head.parameters(), "lr": float(training["server_learning_rate"])},
    ], weight_decay=weight_decay)

    budget = int(training["local_steps_per_disconnection"]) + int(training["recent_smashed_batches"])
    if budget <= 0:
        raise ValueError("The sum of local_steps_per_disconnection and recent_smashed_batches must be positive")
    streams = make_pair_batches(pair_contacts, dataset_bundle.train, training)
    contacts_per_pair = {
        pair_id: sum(row["pair_id"] == pair_id for row in pair_contacts)
        for pair_id in streams
    }
    for pair_id, stream in streams.items():
        needed = budget * contacts_per_pair[pair_id]
        if len(stream) < needed:
            raise ValueError(
                f"{pair_id} has only {len(stream)} planned batches for {needed} "
                "required updates; increase segmentation_training.epochs"
            )
    cursors = {pair_id: 0 for pair_id in streams}
    contact_log, batch_log = [], []

    for contact in pair_contacts:
        pair_id = contact["pair_id"]
        stream = streams[pair_id]
        losses = []
        ignored_batches = 0
        while len(losses) < budget and cursors[pair_id] < len(stream):
            batch_epoch, batch_number, sample_ids, radar, optical, labels = stream[cursors[pair_id]]
            cursors[pair_id] += 1
            if not torch.any(labels != metadata.ignore_index):
                ignored_batches += 1
                continue
            radar, optical, labels = radar.to(device), optical.to(device), labels.to(device)
            optimizer.zero_grad()
            radar_tokens = radar_encoder(radar, attention_bias, mask_info=None)
            optical_tokens = optical_encoder(optical, attention_bias, mask_info=None)
            fused = cross_encoder(radar_tokens, optical_tokens, attention_bias)
            loss = criterion(head(fused, metadata.image_size), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            batch_log.append({
                "pair_contact_id": contact["pair_contact_id"],
                "pair_id": pair_id,
                "epoch": batch_epoch,
                "batch_number": batch_number,
                "sample_ids": ";".join(map(str, sample_ids.tolist())),
                "loss": round(losses[-1], 8),
            })

        if len(losses) != budget:
            raise RuntimeError(
                f"{pair_id} exhausted usable batches at {contact['pair_contact_id']} "
                f"after {len(losses)}/{budget} updates; increase "
                "segmentation_training.epochs or inspect ignored labels"
            )
        accuracy, miou, _ = evaluate_global(
            dataset_bundle.validation, radar_encoder, optical_encoder,
            cross_encoder, head, attention_bias,
            {"radar_encoder": clone_state(radar_encoder),
             "optical_encoder": clone_state(optical_encoder)},
            config, device,
        )
        elapsed = time.perf_counter() - started
        contact_log.append({
            "pair_contact_id": contact["pair_contact_id"],
            "pair_id": pair_id,
            "contact_start_utc": contact["start_utc"],
            "contact_end_utc": contact["end_utc"],
            "status": "completed",
            "requested_batches": budget,
            "trained_batches": len(losses),
            "ignored_batches": ignored_batches,
            "mean_loss": round(sum(losses) / len(losses), 8),
            "test_accuracy": accuracy,
            "test_miou": miou,
            "wall_runtime": f"[{int(elapsed // 60):02d}:{int(elapsed % 60):02d}]",
        })
        print(
            f"[centralized-contact] window_end_utc={contact['end_utc']} "
            f"runtime=[{int(elapsed // 60):02d}:{int(elapsed % 60):02d}] "
            f"pair={pair_id} batches={len(losses)}/{budget} "
            f"mean_loss={sum(losses) / len(losses):.8f} "
            f"test_accuracy={accuracy:.8f} test_miou={miou:.8f}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "centralized_pair_contact_windows.csv", pair_contacts)
    write_csv(output_dir / "centralized_training_log.csv", contact_log)
    write_csv(output_dir / "centralized_batch_log.csv", batch_log)
    final_accuracy, final_miou, _ = evaluate_global(
        dataset_bundle.validation, radar_encoder, optical_encoder, cross_encoder,
        head, attention_bias,
        {"radar_encoder": clone_state(radar_encoder),
         "optical_encoder": clone_state(optical_encoder)},
        config, device,
    )
    summary = {
        "algorithm": "centralized multimodal CROMA segmentation paced by paired contacts",
        "device": str(device),
        "checkpoint": checkpoint_status,
        "dataset": metadata.name,
        "train_samples": len(dataset_bundle.train),
        "validation_samples": len(dataset_bundle.validation),
        "pair_count": pair_count,
        "pair_contact_windows": len(pair_contacts),
        "requested_batches_per_contact": budget,
        "requested_total_batches": budget * len(pair_contacts),
        "trained_total_batches": len(batch_log),
        "full_budget_contacts": sum(row["status"] == "completed" for row in contact_log),
        "final_test_accuracy": final_accuracy,
        "final_test_miou": final_miou,
    }
    with (output_dir / "centralized_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    torch.save({
        "config": config,
        "radar_encoder_state": clone_state(radar_encoder),
        "optical_encoder_state": clone_state(optical_encoder),
        "cross_encoder_state": clone_state(cross_encoder),
        "segmentation_head_state": clone_state(head),
    }, output_dir / "centralized_final_checkpoint.pt")
    print(json.dumps(summary, indent=2))
    print(f"Training log: {output_dir / 'centralized_training_log.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.json")
    parser.add_argument("--contacts", type=Path, default=PROJECT_DIR / "outputs" / "contact_windows.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    if args.pretrained_checkpoint is not None:
        config["croma"]["pretrained_checkpoint"] = str(args.pretrained_checkpoint.resolve())
    run_dir = args.output_dir / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    run_training(config, load_raw_contacts(args.contacts), run_dir)
    print(f"Run output directory: {run_dir}")


if __name__ == "__main__":
    main()
