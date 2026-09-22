"""Standard paired split-learning baseline for the LEO contact simulation.

Unlike the multimodal SFL demo, this baseline performs no computation while a
pair is disconnected.  During each paired contact, the radar and optical
encoders run on the satellite side, while the cross encoder and segmentation
head run on the ground-station side.  The single-device implementation keeps
the split-learning boundary explicit by detaching smashed features and then
backpropagating the server-produced feature gradients through the encoders.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

import torch
from torch import nn

from croma_models import PatchSegmentationHead, build_croma_components
from multimodal_data import PairedBatchSequence, build_dataset_bundle, partition_dataset_indices
from multimodal_evaluation import write_csv
from multimodal_sfl import ModalityState, PlanePair, clone_state
from orbit_model import parse_utc
from pair_contact_scheduler import build_pair_contacts, load_raw_contacts, utc_at


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_DIR.parent / "standard_split_learning_demo"


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


def create_timestamped_run_dir(base_dir: Path, config_path: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.json")
    return run_dir


def make_split_pairs(pair_contacts, global_encoder_states, config, dataset_bundle):
    """Create one persistent client encoder state and batch stream per plane."""
    training = config["segmentation_training"]
    pair_ids_by_plane = {
        plane: next(row["pair_id"] for row in pair_contacts if row["plane"] == plane)
        for plane in sorted({row["plane"] for row in pair_contacts})
    }
    pair_indices = partition_dataset_indices(
        len(dataset_bundle.train), list(pair_ids_by_plane.values()), int(training["seed"])
    )
    pairs = {}
    for plane, pair_id in pair_ids_by_plane.items():
        example = next(row for row in pair_contacts if row["plane"] == plane)
        pairs[pair_id] = PlanePair(
            pair_id=pair_id,
            plane=plane,
            radar=ModalityState(
                example["radar_satellite_id"],
                "radar",
                {key: value.clone() for key, value in global_encoder_states["radar"].items()},
                {},
                {},
            ),
            optical=ModalityState(
                example["optical_satellite_id"],
                "optical",
                {key: value.clone() for key, value in global_encoder_states["optical"].items()},
                {},
                {},
            ),
            batches=PairedBatchSequence(
                dataset_bundle.train,
                pair_indices[pair_id],
                int(training["batch_size"]),
                int(training["epochs"]),
                int(training["seed"]) + plane,
            ),
        )
    return pairs


def train_split_batch(
    pair,
    radar_worker,
    optical_worker,
    cross_encoder,
    segmentation_head,
    attention_bias,
    radar_optimizer,
    optical_optimizer,
    server_optimizer,
    criterion,
    image_size,
    device,
):
    """Run one split-learning step across the satellite/server boundary."""
    _, batch_number, sample_ids, radar, optical, labels = pair.take_batch()
    if not torch.any(labels != criterion.ignore_index):
        return None

    radar = radar.to(device)
    optical = optical.to(device)
    labels = labels.to(device)

    radar_optimizer.zero_grad()
    optical_optimizer.zero_grad()
    server_optimizer.zero_grad()

    # Satellite side: create features, then expose only detached smashed data
    # to the server-side modules.  The original feature graphs are retained so
    # the server feature gradients can be propagated back below.
    radar_tokens = radar_worker(radar, attention_bias, mask_info=None)
    optical_tokens = optical_worker(optical, attention_bias, mask_info=None)
    radar_smashed = radar_tokens.detach().requires_grad_(True)
    optical_smashed = optical_tokens.detach().requires_grad_(True)

    # Ground-station side: train the cross encoder and segmentation head.
    fused = cross_encoder(radar_smashed, optical_smashed, attention_bias)
    logits = segmentation_head(fused, image_size)
    loss = criterion(logits, labels)
    loss.backward()
    radar_feature_grad = radar_smashed.grad.detach()
    optical_feature_grad = optical_smashed.grad.detach()
    server_optimizer.step()

    # Satellite side receives only the gradients at the split boundary.
    torch.autograd.backward(
        (radar_tokens, optical_tokens),
        (radar_feature_grad, optical_feature_grad),
    )
    radar_optimizer.step()
    optical_optimizer.step()

    return {
        "batch_number": batch_number,
        "sample_ids": ";".join(map(str, sample_ids.tolist())),
        "loss": float(loss.item()),
    }


@torch.no_grad()
def evaluate_split_clients(
    pairs,
    test_data,
    radar_worker,
    optical_worker,
    cross_encoder,
    segmentation_head,
    attention_bias,
    config,
    device,
):
    """Evaluate every client encoder with the shared server model."""
    training = config["segmentation_training"]
    metadata = config["dataset_metadata"]
    num_classes = int(metadata["num_classes"])
    ignore_index = int(metadata["ignore_index"])
    batch_size = int(training["batch_size"])
    batches = PairedBatchSequence(test_data, range(len(test_data)), batch_size, 1, 0, shuffle=False)
    pair_metrics = []
    radar_worker.eval()
    optical_worker.eval()
    cross_encoder.eval()
    segmentation_head.eval()

    for pair in pairs.values():
        radar_worker.load_state_dict(pair.radar.encoder_state)
        optical_worker.load_state_dict(pair.optical.encoder_state)
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        for _, _, _, radar, optical, targets in batches:
            radar_tokens = radar_worker(radar.to(device), attention_bias, None)
            optical_tokens = optical_worker(optical.to(device), attention_bias, None)
            logits = segmentation_head(
                cross_encoder(radar_tokens, optical_tokens, attention_bias),
                int(radar.shape[-1]),
            )
            predictions = logits.argmax(dim=1).cpu()
            valid = targets != ignore_index
            indices = targets[valid].reshape(-1) * num_classes + predictions[valid].reshape(-1)
            if indices.numel():
                confusion += torch.bincount(
                    indices, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)
        intersection = confusion.diag().float()
        union = confusion.sum(0) + confusion.sum(1) - intersection
        valid_classes = union > 0
        pair_metrics.append({
            "accuracy": float(intersection.sum().item() / max(1, confusion.sum().item())),
            "miou": float((intersection[valid_classes] / union[valid_classes]).mean().item()),
        })

    radar_worker.train()
    optical_worker.train()
    cross_encoder.train()
    segmentation_head.train()
    if not pair_metrics:
        raise ValueError("No split-learning clients are available for evaluation")
    return (
        sum(item["accuracy"] for item in pair_metrics) / len(pair_metrics),
        sum(item["miou"] for item in pair_metrics) / len(pair_metrics),
    )


def run_training(config, raw_contacts, output_dir: Path):
    run_started_at = time.perf_counter()
    training, model_config = config["segmentation_training"], config["croma"]
    torch.manual_seed(int(training["seed"]))
    torch.set_num_threads(1)
    device = select_device(training["device"])
    epoch = parse_utc(config["simulation"]["epoch_utc"])
    pair_contacts = build_pair_contacts(raw_contacts, epoch)
    pair_count = len({row["pair_id"] for row in pair_contacts})
    dataset_bundle = build_dataset_bundle(config, PROJECT_DIR, pair_count)
    config["dataset_metadata"] = {
        "name": dataset_bundle.metadata.name,
        "ignore_index": dataset_bundle.metadata.ignore_index,
        "num_classes": dataset_bundle.metadata.num_classes,
        "radar_channels": dataset_bundle.metadata.radar_channels,
        "optical_channels": dataset_bundle.metadata.optical_channels,
    }
    num_classes = dataset_bundle.metadata.num_classes
    image_size = dataset_bundle.metadata.image_size

    radar_worker, optical_worker, cross_encoder, attention_bias, checkpoint_status = build_croma_components(
        config, device, PROJECT_DIR
    )
    print(f"[split-init] pretrained_checkpoint={checkpoint_status}", flush=True)
    segmentation_head = PatchSegmentationHead(
        model_config["encoder_dim"], num_classes, model_config["num_patches"]
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset_bundle.metadata.ignore_index)

    # Standard split learning trains all four model components.  The special
    # SFL encoder-freezing option is intentionally not used by this baseline.
    radar_worker.requires_grad_(True)
    optical_worker.requires_grad_(True)
    server_optimizer = torch.optim.AdamW(
        list(cross_encoder.parameters()) + list(segmentation_head.parameters()),
        lr=training["server_learning_rate"],
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    initial_states = {"radar": clone_state(radar_worker), "optical": clone_state(optical_worker)}
    pairs = make_split_pairs(pair_contacts, initial_states, config, dataset_bundle)
    # Each simulated client keeps its own AdamW moments even though all
    # clients reuse one physical encoder module on the single GPU.
    weight_decay = float(training.get("weight_decay", 0.01))
    radar_optimizers = {
        pair_id: torch.optim.AdamW(
            radar_worker.parameters(),
            lr=training["encoder_learning_rate"],
            weight_decay=weight_decay,
        )
        for pair_id in pairs
    }
    optical_optimizers = {
        pair_id: torch.optim.AdamW(
            optical_worker.parameters(),
            lr=training["encoder_learning_rate"],
            weight_decay=weight_decay,
        )
        for pair_id in pairs
    }
    recent_batches = int(training["recent_smashed_batches"])
    if recent_batches <= 0:
        raise ValueError("segmentation_training.recent_smashed_batches must be positive")

    contact_log, batch_log = [], []
    successful_contacts = 0
    for contact in pair_contacts:
        pair = pairs[contact["pair_id"]]
        radar_worker.load_state_dict(pair.radar.encoder_state)
        optical_worker.load_state_dict(pair.optical.encoder_state)
        steps = 0
        losses = []
        contact_start = time.perf_counter()
        while steps < recent_batches and pair.has_local_work:
            result = train_split_batch(
                pair, radar_worker, optical_worker, cross_encoder, segmentation_head,
                attention_bias, radar_optimizers[pair.pair_id],
                optical_optimizers[pair.pair_id], server_optimizer,
                criterion, image_size, device,
            )
            if result is None:
                continue
            steps += 1
            losses.append(result["loss"])
            batch_log.append({
                "pair_contact_id": contact["pair_contact_id"],
                "pair_id": pair.pair_id,
                "batch_number": result["batch_number"],
                "sample_ids": result["sample_ids"],
                "loss": round(result["loss"], 8),
            })

        pair.radar.encoder_state = clone_state(radar_worker)
        pair.optical.encoder_state = clone_state(optical_worker)
        if steps:
            successful_contacts += 1
            status, reason = "completed", ""
        else:
            status, reason = "skipped", "dataset_exhausted"
        accuracy, miou = evaluate_split_clients(
            pairs, dataset_bundle.validation, radar_worker, optical_worker,
            cross_encoder, segmentation_head, attention_bias, config, device,
        ) if steps else ("", "")
        elapsed_s = time.perf_counter() - run_started_at
        contact_log.append({
            "pair_contact_id": contact["pair_contact_id"],
            "pair_id": pair.pair_id,
            "contact_start_utc": contact["start_utc"],
            "contact_end_utc": contact["end_utc"],
            "status": status,
            "reason": reason,
            "trained_batches": steps,
            "requested_batches": recent_batches,
            "mean_loss": round(sum(losses) / len(losses), 8) if losses else "",
            "test_accuracy": accuracy,
            "test_miou": miou,
            "wall_runtime": f"[{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}]",
            "processing_s": round(time.perf_counter() - contact_start, 6),
        })
        if steps:
            print(
                f"[split-contact] window_end_utc={contact['end_utc']} "
                f"pair={pair.pair_id} batches={steps} "
                f"mean_loss={sum(losses) / len(losses):.8f} "
                f"test_accuracy={accuracy:.8f} test_miou={miou:.8f}",
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "standard_split_training_log.csv", contact_log)
    write_csv(output_dir / "standard_split_batch_log.csv", batch_log)
    summary = {
        "algorithm": "standard paired split learning without disconnected local training",
        "device": str(device),
        "checkpoint": checkpoint_status,
        "dataset": dataset_bundle.metadata.name,
        "pair_count": len(pairs),
        "pair_contact_windows": len(pair_contacts),
        "successful_contacts": successful_contacts,
        "skipped_contacts": len(pair_contacts) - successful_contacts,
        "recent_smashed_batches": recent_batches,
        "satellite_encoder_training_mode": "full",
        "server_model": "cross_encoder + segmentation_head",
        "final_test_accuracy": contact_log[-1]["test_accuracy"] if contact_log else "",
        "final_test_miou": contact_log[-1]["test_miou"] if contact_log else "",
    }
    with (output_dir / "standard_split_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    torch.save(
        {
            "config": config,
            "radar_encoder_states": {pair_id: pair.radar.encoder_state for pair_id, pair in pairs.items()},
            "optical_encoder_states": {pair_id: pair.optical.encoder_state for pair_id, pair in pairs.items()},
            "cross_encoder_state": clone_state(cross_encoder),
            "segmentation_head_state": clone_state(segmentation_head),
        },
        output_dir / "standard_split_final_checkpoint.pt",
    )
    print(json.dumps(summary, indent=2))
    print(f"Training log: {output_dir / 'standard_split_training_log.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.json")
    parser.add_argument("--contacts", type=Path, default=PROJECT_DIR / "outputs" / "contact_windows.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path overriding croma.pretrained_checkpoint.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_dir = create_timestamped_run_dir(args.output_dir, config_path)
    config = load_json(config_path)
    if args.pretrained_checkpoint is not None:
        config["croma"]["pretrained_checkpoint"] = str(args.pretrained_checkpoint)
    run_training(config, load_raw_contacts(args.contacts), run_dir)
    print(f"Run output directory: {run_dir}")


if __name__ == "__main__":
    main()
