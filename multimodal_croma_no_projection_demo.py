"""Ablation study: paired multimodal CROMA split-federated learning without
satellite projection layers.

Relation to ``multimodal_croma_demo.py``:

* No ``projR``/``projO`` modules exist anywhere in this pipeline.
* During a disconnected interval each satellite trains
  ``encoder -> auxiliary segmentation head`` directly on the encoder tokens.
* The ground station no longer computes the projection distillation MSE; it
  trains ``cross_encoder + ground_head`` with the segmentation loss only.
* Orbit scheduling, dataset partitioning, contact-window bookkeeping, the
  equal-weight aggregation of encoder/auxiliary states, and the evaluation
  protocol are unchanged, so the projection layer is the only experimental
  variable.

``segmentation_training.freeze_projection_during_disconnection`` is ignored by
this script because no projection layer exists.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch import nn

from croma_models import (
    PatchSegmentationHead,
    build_croma_components,
    configure_satellite_encoder_trainability,
    pretrained_encoder_schedule,
)
from multimodal_data import (
    PairedBatchSequence,
    build_dataset_bundle,
    partition_dataset_indices,
)
from multimodal_evaluation import evaluate_global, write_csv
from multimodal_sfl import (
    FeaturePacket,
    ModalityState,
    PairContribution,
    PlanePair,
    average_states,
    clone_state,
    estimate_transaction,
)
from orbit_model import parse_utc
from pair_contact_scheduler import build_pair_contacts, load_raw_contacts, utc_at


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_DIR.parent / "multimodal_croma_no_projection_demo"

# Same columns as the projection-based demo so the two logs can be compared
# line by line; the distillation columns stay empty by construction.
CONTACT_LOG_FIELDS = [
    "pair_contact_id", "pair_id", "radar_satellite_id", "optical_satellite_id",
    "direct_satellites", "contact_start_utc", "contact_end_utc",
    "transaction_start_utc", "transaction_finish_utc", "status", "reason",
    "matched_batches", "radar_upload_bytes", "optical_upload_bytes",
    "server_updates", "server_mean_loss",
    "radar_distillation_mean_loss", "optical_distillation_mean_loss",
    "test_accuracy", "test_miou", "aggregation_performed",
    "aggregation_members", "global_version", "modeled_transaction_s",
]


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
    """Create one immutable output directory and preserve the input config."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.json")
    return run_dir


def _trainable_parameters(module):
    """Return only the parameters enabled by the current freeze schedule."""
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _prune_buffer(buffer: dict[int, FeaturePacket], limit: int) -> None:
    while len(buffer) > limit:
        del buffer[next(iter(buffer))]


def train_pair_offline_no_projection(
    pair: PlanePair, stop_time_s, config, radar_worker, optical_worker,
    radar_auxiliary, optical_auxiliary, attention_bias, criterion, device,
    epoch, log_rows,
):
    """Satellite-side training with the auxiliary head on raw encoder tokens."""
    training = config["segmentation_training"]
    if not pair.has_local_work:
        return
    radar_worker.load_state_dict(pair.radar.encoder_state)
    optical_worker.load_state_dict(pair.optical.encoder_state)
    radar_auxiliary.load_state_dict(pair.radar.auxiliary_state)
    optical_auxiliary.load_state_dict(pair.optical.auxiliary_state)
    radar_parameters = [
        {"params": _trainable_parameters(radar_worker), "lr": training["encoder_learning_rate"]},
        {"params": _trainable_parameters(radar_auxiliary), "lr": training["auxiliary_learning_rate"]},
    ]
    optical_parameters = [
        {"params": _trainable_parameters(optical_worker), "lr": training["encoder_learning_rate"]},
        {"params": _trainable_parameters(optical_auxiliary), "lr": training["auxiliary_learning_rate"]},
    ]
    weight_decay = float(training.get("weight_decay", 0.01))
    radar_optimizer = torch.optim.AdamW(radar_parameters, weight_decay=weight_decay)
    optical_optimizer = torch.optim.AdamW(optical_parameters, weight_decay=weight_decay)
    step_s = max(float(training["radar_local_compute_s"]), float(training["optical_local_compute_s"]))
    buffer_limit = int(training["recent_smashed_batches"])
    image_size = int(config["croma"]["patch_size"]) * math.isqrt(int(config["croma"]["num_patches"]))
    completed = 0
    while (
        completed < int(training["local_steps_per_disconnection"])
        and pair.has_local_work
        and pair.local_clock_s + step_s <= stop_time_s
    ):
        epoch_number, batch_number, sample_ids, radar, optical, labels = pair.take_batch()
        if not torch.any(labels != criterion.ignore_index):
            continue
        radar, optical, labels_device = radar.to(device), optical.to(device), labels.to(device)
        step_start = pair.local_clock_s
        radar_optimizer.zero_grad()
        radar_tokens = radar_worker(radar, attention_bias, mask_info=None)
        radar_loss = criterion(radar_auxiliary(radar_tokens, image_size), labels_device)
        radar_loss.backward()
        radar_optimizer.step()
        pair.radar.local_version += 1
        optical_optimizer.zero_grad()
        optical_tokens = optical_worker(optical, attention_bias, mask_info=None)
        optical_loss = criterion(optical_auxiliary(optical_tokens, image_size), labels_device)
        optical_loss.backward()
        optical_optimizer.step()
        pair.optical.local_version += 1
        pair.radar_buffer[batch_number] = FeaturePacket(
            radar_tokens.detach().cpu().clone(), labels.clone(), sample_ids.clone(),
            batch_number, pair.radar.local_version,
        )
        pair.optical_buffer[batch_number] = FeaturePacket(
            optical_tokens.detach().cpu().clone(), labels.clone(), sample_ids.clone(),
            batch_number, pair.optical.local_version,
        )
        _prune_buffer(pair.radar_buffer, buffer_limit)
        _prune_buffer(pair.optical_buffer, buffer_limit)
        pair.local_clock_s += step_s
        completed += 1
        log_rows.append({
            "pair_id": pair.pair_id, "radar_satellite_id": pair.radar.satellite_id,
            "optical_satellite_id": pair.optical.satellite_id, "epoch": epoch_number,
            "batch_number": batch_number, "sample_ids": ";".join(map(str, sample_ids.tolist())),
            "start_utc": utc_at(epoch, step_start), "finish_utc": utc_at(epoch, pair.local_clock_s),
            "radar_auxiliary_loss": round(float(radar_loss.item()), 8),
            "optical_auxiliary_loss": round(float(optical_loss.item()), 8),
            "radar_local_version": pair.radar.local_version,
            "optical_local_version": pair.optical.local_version,
            "matched_buffered_batches": len(pair.matched_batch_ids()),
        })
    pair.radar.encoder_state, pair.radar.auxiliary_state = clone_state(radar_worker), clone_state(radar_auxiliary)
    pair.optical.encoder_state, pair.optical.auxiliary_state = clone_state(optical_worker), clone_state(optical_auxiliary)


def train_server_no_projection(
    pair, matched_ids, cross_encoder, ground_head, attention_bias,
    optimizer, criterion, image_size, device,
):
    """Ground-station update with no projection distillation term."""
    losses = {"segmentation": []}
    for batch_id in matched_ids:
        radar_packet, optical_packet = pair.radar_buffer[batch_id], pair.optical_buffer[batch_id]
        if not torch.equal(radar_packet.sample_ids, optical_packet.sample_ids):
            raise RuntimeError(f"Unmatched sample IDs in pair {pair.pair_id}")
        optimizer.zero_grad()
        fused = cross_encoder(
            radar_packet.tokens.to(device), optical_packet.tokens.to(device), attention_bias
        )
        segmentation_loss = criterion(ground_head(fused, image_size), radar_packet.labels.to(device))
        segmentation_loss.backward()
        optimizer.step()
        losses["segmentation"].append(float(segmentation_loss.item()))
    return losses


def reset_pair_from_global_no_projection(pair, global_states, global_version):
    pair.radar.encoder_state = {key: value.clone() for key, value in global_states["radar_encoder"].items()}
    pair.radar.auxiliary_state = {key: value.clone() for key, value in global_states["radar_auxiliary"].items()}
    pair.optical.encoder_state = {key: value.clone() for key, value in global_states["optical_encoder"].items()}
    pair.optical.auxiliary_state = {key: value.clone() for key, value in global_states["optical_auxiliary"].items()}
    pair.radar.downloaded_global_version = global_version
    pair.optical.downloaded_global_version = global_version


def make_plane_pairs(pair_contacts, global_states, config, dataset_bundle):
    """One radar/optical pair per plane; no projection state is carried."""
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
        pairs[example["pair_id"]] = PlanePair(
            pair_id=example["pair_id"], plane=plane,
            radar=ModalityState(
                example["radar_satellite_id"], "radar",
                {key: value.clone() for key, value in global_states["radar_encoder"].items()},
                {key: value.clone() for key, value in global_states["radar_auxiliary"].items()},
                {},
            ),
            optical=ModalityState(
                example["optical_satellite_id"], "optical",
                {key: value.clone() for key, value in global_states["optical_encoder"].items()},
                {key: value.clone() for key, value in global_states["optical_auxiliary"].items()},
                {},
            ),
            batches=PairedBatchSequence(
                dataset_bundle.train,
                pair_indices[pair_id],
                training["batch_size"],
                training["epochs"],
                training["seed"] + plane,
            ),
        )
    return pairs


def run_training(config, raw_contacts, output_dir: Path):
    """No-projection ablation of the paired multimodal CROMA SFL pipeline."""
    run_started_at = time.perf_counter()
    training, model_config = config["segmentation_training"], config["croma"]
    torch.manual_seed(training["seed"])
    torch.set_num_threads(1)
    device = select_device(training["device"])
    epoch = parse_utc(config["simulation"]["epoch_utc"])
    horizon_s = float(config["simulation"]["duration_hours"]) * 3600.0
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
    # The auxiliary heads receive encoder tokens directly: no projR/projO.
    radar_auxiliary = PatchSegmentationHead(
        model_config["encoder_dim"], num_classes, model_config["num_patches"]
    ).to(device)
    optical_auxiliary = PatchSegmentationHead(
        model_config["encoder_dim"], num_classes, model_config["num_patches"]
    ).to(device)
    ground_head = PatchSegmentationHead(
        model_config["encoder_dim"], num_classes, model_config["num_patches"]
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset_bundle.metadata.ignore_index)
    # No projection modules exist, so the station trains only the cross encoder
    # and the ground segmentation head.
    server_optimizer = torch.optim.AdamW(
        list(cross_encoder.parameters()) + list(ground_head.parameters()),
        lr=training["server_learning_rate"],
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    global_states = {
        "radar_encoder": clone_state(radar_worker),
        "radar_auxiliary": clone_state(radar_auxiliary),
        "optical_encoder": clone_state(optical_worker),
        "optical_auxiliary": clone_state(optical_auxiliary),
    }
    pairs = make_plane_pairs(pair_contacts, global_states, config, dataset_bundle)
    test_data = dataset_bundle.validation
    encoder_schedule = pretrained_encoder_schedule(config)
    encoder_stage = configure_satellite_encoder_trainability(
        radar_worker, optical_worker, config, global_version=0
    )
    if encoder_schedule["mode"] != "full":
        print(
            "[pretrained-encoder] "
            f"mode={encoder_schedule['mode']} stage={encoder_stage} "
            f"warmup_aggregations={encoder_schedule['warmup_aggregations']} "
            f"trainable_blocks={encoder_schedule['trainable_blocks']}",
            flush=True,
        )
    aggregation_k = int(training["aggregation_k"])
    if not 1 <= aggregation_k <= len(pairs):
        raise ValueError("aggregation_k must be between 1 and the plane count")
    pending, local_log, contact_log, aggregation_log = [], [], [], []
    ground_available_s, global_version, server_updates = 0.0, 0, 0
    for contact in pair_contacts:
        pair: PlanePair = pairs[contact["pair_id"]]
        configure_satellite_encoder_trainability(
            radar_worker, optical_worker, config, global_version
        )
        train_pair_offline_no_projection(
            pair, contact["start_offset_s"], config, radar_worker, optical_worker,
            radar_auxiliary, optical_auxiliary, attention_bias, criterion, device,
            epoch, local_log,
        )
        matched_ids = pair.matched_batch_ids()
        will_aggregate = len(pending) + 1 >= aggregation_k
        estimate = estimate_transaction(pair, matched_ids, contact, config, will_aggregate)
        start_s = max(contact["start_offset_s"], ground_available_s)
        finish_s = start_s + estimate["duration_s"]
        deadline_s = contact["end_offset_s"] - float(config["link"]["safety_margin_s"])
        status, reason = "completed", ""
        losses = {"segmentation": []}
        aggregation_members, aggregation_performed = "", False
        test_accuracy, test_miou = "", ""
        server_mean_loss = ""
        if not matched_ids:
            status, reason = "skipped", "no_matched_multimodal_features"
        elif start_s >= contact["end_offset_s"]:
            status, reason = "skipped", "ground_station_busy_until_disconnect"
        elif finish_s > deadline_s:
            status, reason = "skipped", "paired_transaction_does_not_fit_contact"
        else:
            losses = train_server_no_projection(
                pair, matched_ids, cross_encoder, ground_head, attention_bias,
                server_optimizer, criterion, image_size, device,
            )
            server_updates += len(losses["segmentation"])
            server_mean_loss = (
                sum(losses["segmentation"]) / len(losses["segmentation"])
                if losses["segmentation"] else ""
            )
            pending.append(PairContribution(
                pair.pair_id, pair.radar.encoder_state, pair.radar.auxiliary_state,
                pair.optical.encoder_state, pair.optical.auxiliary_state,
            ))
            if len(pending) == aggregation_k:
                aggregation_members = ";".join(item.pair_id for item in pending)
                global_states = {
                    "radar_encoder": average_states([item.radar_encoder_state for item in pending]),
                    "radar_auxiliary": average_states([item.radar_auxiliary_state for item in pending]),
                    "optical_encoder": average_states([item.optical_encoder_state for item in pending]),
                    "optical_auxiliary": average_states([item.optical_auxiliary_state for item in pending]),
                }
                global_version, aggregation_performed = global_version + 1, True
                encoder_stage = configure_satellite_encoder_trainability(
                    radar_worker, optical_worker, config, global_version
                )
                aggregation_log.append({
                    "global_version": global_version, "finish_utc": utc_at(epoch, finish_s),
                    "plane_pairs": aggregation_members, "pair_count": aggregation_k,
                    "weight_per_pair": round(1.0 / aggregation_k, 8),
                })
                test_accuracy, test_miou, _ = evaluate_global(
                    test_data, radar_worker, optical_worker, cross_encoder,
                    ground_head, attention_bias, global_states, config, device,
                )
                elapsed_s = time.perf_counter() - run_started_at
                runtime_text = f"[{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}]"
                server_loss_text = f"{server_mean_loss:.8f}" if server_mean_loss != "" else "n/a"
                print(
                    "[aggregation] "
                    f"window_end_utc={contact['end_utc']} runtime={runtime_text} "
                    f"server_mean_loss={server_loss_text} "
                    f"test_accuracy={test_accuracy:.8f} test_miou={test_miou:.8f}",
                    flush=True,
                )
                pending.clear()
            reset_pair_from_global_no_projection(pair, global_states, global_version)
            pair.radar_buffer.clear()
            pair.optical_buffer.clear()
            ground_available_s = finish_s
        contact_log.append({
            "pair_contact_id": contact["pair_contact_id"], "pair_id": pair.pair_id,
            "radar_satellite_id": pair.radar.satellite_id,
            "optical_satellite_id": pair.optical.satellite_id,
            "direct_satellites": contact["direct_satellites"],
            "contact_start_utc": contact["start_utc"], "contact_end_utc": contact["end_utc"],
            "transaction_start_utc": utc_at(epoch, start_s) if status == "completed" else "",
            "transaction_finish_utc": utc_at(epoch, finish_s) if status == "completed" else "",
            "status": status, "reason": reason, "matched_batches": len(matched_ids),
            "radar_upload_bytes": estimate["radar_upload_bytes"],
            "optical_upload_bytes": estimate["optical_upload_bytes"],
            "server_updates": len(losses["segmentation"]),
            "server_mean_loss": round(server_mean_loss, 8) if server_mean_loss != "" else "",
            "radar_distillation_mean_loss": "",
            "optical_distillation_mean_loss": "",
            "test_accuracy": test_accuracy, "test_miou": test_miou,
            "aggregation_performed": int(aggregation_performed),
            "aggregation_members": aggregation_members,
            "global_version": global_version,
            "modeled_transaction_s": round(estimate["duration_s"], 8),
        })
        pair.local_clock_s = max(pair.local_clock_s, contact["end_offset_s"])

    for pair in pairs.values():
        train_pair_offline_no_projection(
            pair, horizon_s, config, radar_worker, optical_worker,
            radar_auxiliary, optical_auxiliary, attention_bias, criterion, device,
            epoch, local_log,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pair_contact_windows.csv", pair_contacts)
    local_log.sort(key=lambda row: (row["start_utc"], row["pair_id"]))
    write_csv(output_dir / "multimodal_local_training_log.csv", local_log)
    write_csv(output_dir / "multimodal_training_log.csv", contact_log, CONTACT_LOG_FIELDS)
    write_csv(
        output_dir / "multimodal_aggregation_log.csv", aggregation_log,
        ["global_version", "finish_utc", "plane_pairs", "pair_count", "weight_per_pair"],
    )
    pixel_accuracy, mean_iou, _ = evaluate_global(
        test_data, radar_worker, optical_worker, cross_encoder, ground_head,
        attention_bias, global_states, config, device,
    )
    successful = sum(row["status"] == "completed" for row in contact_log)
    summary = {
        "algorithm": "paired multimodal CROMA SFL without projection layers and without staleness weighting",
        "projection_layers": "removed; satellite auxiliary heads consume encoder tokens directly",
        "ground_distillation": "disabled (no projR/projO MSE)",
        "device": str(device),
        "croma_profile": model_config["profile"], "checkpoint": checkpoint_status,
        "image_size": image_size,
        "dataset": dataset_bundle.metadata.name,
        "train_samples": len(dataset_bundle.train),
        "validation_samples": len(dataset_bundle.validation),
        "samples_per_pair": {
            pair_id: len({index for _, _, indices in pair.batches.batch_specs for index in indices})
            for pair_id, pair in pairs.items()
        },
        "plane_pairs": len(pairs), "pair_contact_windows": len(pair_contacts),
        "local_paired_steps": len(local_log),
        "server_updates": server_updates, "successful_pair_transactions": successful,
        "skipped_pair_contacts": len(contact_log) - successful,
        "aggregations": len(aggregation_log),
        "aggregation_k": aggregation_k, "aggregation": "uniform arithmetic mean per modality",
        "pretrained_encoder_schedule": {
            "mode": str(encoder_schedule["mode"]),
            "warmup_aggregations": int(encoder_schedule["warmup_aggregations"]),
            "trainable_blocks_after_warmup": int(encoder_schedule["trainable_blocks"]),
            "final_stage": encoder_stage,
        },
        "pixel_accuracy": pixel_accuracy, "mean_iou": mean_iou,
        "last_ground_transaction_utc": utc_at(epoch, ground_available_s),
        "pending_matched_batches": {
            pair_id: len(pair.matched_batch_ids())
            for pair_id, pair in pairs.items() if pair.matched_batch_ids()
        },
    }
    with (output_dir / "multimodal_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    torch.save({
        "config": config, "global_version": global_version, "global_states": global_states,
        "cross_encoder_state": clone_state(cross_encoder),
        "ground_segmentation_head_state": clone_state(ground_head),
    }, output_dir / "multimodal_final_checkpoint.pt")
    print(json.dumps(summary, indent=2))
    print(f"Pair contacts: {output_dir / 'pair_contact_windows.csv'}")
    print(f"Training log: {output_dir / 'multimodal_training_log.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.json")
    parser.add_argument("--contacts", type=Path, default=PROJECT_DIR / "outputs" / "contact_windows.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RUNS_DIR,
        help="Base directory; each invocation creates a timestamped child directory.",
    )
    parser.add_argument(
        "--pretrained-checkpoint", type=Path, default=None,
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
