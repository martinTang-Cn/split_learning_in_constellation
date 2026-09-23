"""Multimodal FedAvg baseline for the LEO contact simulation.

Each orbital plane's radar/optical pair is one multimodal client.  The client
keeps a complete local model: radar encoder, optical encoder, cross encoder,
and segmentation head.  Local training happens while the pair is disconnected
and the complete model is exchanged when a paired contact is available.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

import torch
from torch import nn

from croma_models import PatchSegmentationHead, build_croma_components
from multimodal_data import PairedBatchSequence, build_dataset_bundle, partition_dataset_indices
from multimodal_evaluation import evaluate_global, write_csv
from multimodal_sfl import average_states, clone_state, state_nbytes
from orbit_model import SPEED_OF_LIGHT_KM_S, parse_utc
from pair_contact_scheduler import build_pair_contacts, load_raw_contacts, utc_at


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_DIR.parent / "fedavg_multimodal_demo"


@dataclass
class FedAvgClient:
    pair_id: str
    plane: int
    radar_satellite_id: str
    optical_satellite_id: str
    batches: PairedBatchSequence
    radar_encoder_state: dict[str, torch.Tensor]
    optical_encoder_state: dict[str, torch.Tensor]
    cross_encoder_state: dict[str, torch.Tensor]
    segmentation_head_state: dict[str, torch.Tensor]
    next_batch_index: int = 0
    local_clock_s: float = 0.0
    global_version: int = 0

    @property
    def has_local_work(self) -> bool:
        return self.next_batch_index < len(self.batches)

    def take_batch(self):
        batch = self.batches[self.next_batch_index]
        self.next_batch_index += 1
        return batch


@dataclass
class FedAvgContribution:
    client_id: str
    radar_encoder_state: dict[str, torch.Tensor]
    optical_encoder_state: dict[str, torch.Tensor]
    cross_encoder_state: dict[str, torch.Tensor]
    segmentation_head_state: dict[str, torch.Tensor]


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


def model_state_nbytes(states: dict[str, dict[str, torch.Tensor]]) -> int:
    return sum(state_nbytes(state) for state in states.values())


def make_clients(pair_contacts, initial_states, config, dataset_bundle):
    training = config["segmentation_training"]
    pair_ids_by_plane = {
        plane: next(row["pair_id"] for row in pair_contacts if row["plane"] == plane)
        for plane in sorted({row["plane"] for row in pair_contacts})
    }
    pair_indices = partition_dataset_indices(
        len(dataset_bundle.train), list(pair_ids_by_plane.values()), int(training["seed"])
    )
    clients = {}
    for plane, pair_id in pair_ids_by_plane.items():
        row = next(item for item in pair_contacts if item["pair_id"] == pair_id)
        clients[pair_id] = FedAvgClient(
            pair_id=pair_id,
            plane=plane,
            radar_satellite_id=row["radar_satellite_id"],
            optical_satellite_id=row["optical_satellite_id"],
            batches=PairedBatchSequence(
                dataset_bundle.train,
                pair_indices[pair_id],
                int(training["batch_size"]),
                int(training["epochs"]),
                int(training["seed"]) + plane,
            ),
            radar_encoder_state={key: value.clone() for key, value in initial_states["radar"].items()},
            optical_encoder_state={key: value.clone() for key, value in initial_states["optical"].items()},
            cross_encoder_state={key: value.clone() for key, value in initial_states["cross"].items()},
            segmentation_head_state={key: value.clone() for key, value in initial_states["head"].items()},
        )
    return clients


def load_client_state(client, radar_encoder, optical_encoder, cross_encoder, head):
    radar_encoder.load_state_dict(client.radar_encoder_state)
    optical_encoder.load_state_dict(client.optical_encoder_state)
    cross_encoder.load_state_dict(client.cross_encoder_state)
    head.load_state_dict(client.segmentation_head_state)


def save_client_state(client, radar_encoder, optical_encoder, cross_encoder, head):
    client.radar_encoder_state = clone_state(radar_encoder)
    client.optical_encoder_state = clone_state(optical_encoder)
    client.cross_encoder_state = clone_state(cross_encoder)
    client.segmentation_head_state = clone_state(head)


def train_client_offline(
    client,
    stop_time_s,
    config,
    radar_encoder,
    optical_encoder,
    cross_encoder,
    head,
    attention_bias,
    radar_optimizer,
    optical_optimizer,
    server_optimizer,
    criterion,
    image_size,
    device,
):
    """Train the client's complete model during its disconnected interval."""
    if not client.has_local_work:
        return []
    load_client_state(client, radar_encoder, optical_encoder, cross_encoder, head)
    training = config["segmentation_training"]
    step_s = max(
        float(training["radar_local_compute_s"]),
        float(training["optical_local_compute_s"]),
    )
    local_steps = int(training["local_steps_per_disconnection"])
    losses = []
    completed = 0
    while (
        completed < local_steps
        and client.has_local_work
        and client.local_clock_s + step_s <= stop_time_s
    ):
        _, batch_number, sample_ids, radar, optical, labels = client.take_batch()
        if not torch.any(labels != criterion.ignore_index):
            continue
        radar, optical, labels = radar.to(device), optical.to(device), labels.to(device)
        radar_optimizer.zero_grad()
        optical_optimizer.zero_grad()
        server_optimizer.zero_grad()
        radar_tokens = radar_encoder(radar, attention_bias, mask_info=None)
        optical_tokens = optical_encoder(optical, attention_bias, mask_info=None)
        fused = cross_encoder(radar_tokens, optical_tokens, attention_bias)
        loss = criterion(head(fused, image_size), labels)
        loss.backward()
        radar_optimizer.step()
        optical_optimizer.step()
        server_optimizer.step()
        losses.append(float(loss.item()))
        completed += 1
        client.local_clock_s += step_s
    save_client_state(client, radar_encoder, optical_encoder, cross_encoder, head)
    return losses


def reset_client_from_global(client, global_states, global_version):
    client.radar_encoder_state = {key: value.clone() for key, value in global_states["radar"].items()}
    client.optical_encoder_state = {key: value.clone() for key, value in global_states["optical"].items()}
    client.cross_encoder_state = {key: value.clone() for key, value in global_states["cross"].items()}
    client.segmentation_head_state = {key: value.clone() for key, value in global_states["head"].items()}
    client.global_version = global_version


def contribution_from_client(client):
    return FedAvgContribution(
        client_id=client.pair_id,
        radar_encoder_state=client.radar_encoder_state,
        optical_encoder_state=client.optical_encoder_state,
        cross_encoder_state=client.cross_encoder_state,
        segmentation_head_state=client.segmentation_head_state,
    )


def estimate_transaction(client, contact, config, global_states, will_aggregate):
    """Estimate complete-model upload/download time for a paired contact."""
    link = config["link"]
    training = config["segmentation_training"]
    overhead = int(link["protocol_overhead_bytes"])
    model_bytes = model_state_nbytes(global_states)
    upload_s = (model_bytes + overhead) * 8 / (contact["uplink_mbps"] * 1e6)
    download_s = (model_bytes + overhead) * 8 / (contact["downlink_mbps"] * 1e6)
    propagation_s = 2 * contact["min_slant_range_km"] / SPEED_OF_LIGHT_KM_S
    aggregation_s = float(training["aggregation_compute_s"]) if will_aggregate else 0.0
    return {
        "model_bytes": model_bytes,
        "upload_s": upload_s,
        "download_s": download_s,
        "propagation_s": propagation_s,
        "aggregation_s": aggregation_s,
        "duration_s": upload_s + download_s + propagation_s + aggregation_s,
    }


def run_training(config, raw_contacts, output_dir: Path):
    run_started_at = time.perf_counter()
    training, model_config = config["segmentation_training"], config["croma"]
    torch.manual_seed(int(training["seed"]))
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

    radar_encoder, optical_encoder, cross_encoder, attention_bias, checkpoint_status = build_croma_components(
        config, device, PROJECT_DIR
    )
    head = PatchSegmentationHead(
        model_config["encoder_dim"], num_classes, model_config["num_patches"]
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset_bundle.metadata.ignore_index)
    initial_states = {
        "radar": clone_state(radar_encoder),
        "optical": clone_state(optical_encoder),
        "cross": clone_state(cross_encoder),
        "head": clone_state(head),
    }
    global_states = {key: {name: value.clone() for name, value in state.items()} for key, state in initial_states.items()}
    clients = make_clients(pair_contacts, initial_states, config, dataset_bundle)
    clients_per_aggregation = int(training["aggregation_k"])
    if not 1 <= clients_per_aggregation <= len(clients):
        raise ValueError("aggregation_k must be between 1 and the plane count")

    weight_decay = float(training.get("weight_decay", 0.01))
    optimizer_by_client = {}
    for client_id in clients:
        optimizer_by_client[client_id] = {
            "radar": torch.optim.AdamW(radar_encoder.parameters(), lr=training["encoder_learning_rate"], weight_decay=weight_decay),
            "optical": torch.optim.AdamW(optical_encoder.parameters(), lr=training["encoder_learning_rate"], weight_decay=weight_decay),
            "server": torch.optim.AdamW(
                list(cross_encoder.parameters()) + list(head.parameters()),
                lr=training["server_learning_rate"],
                weight_decay=weight_decay,
            ),
        }
    pending, contact_log, local_log, aggregation_log = [], [], [], []
    ground_available_s = 0.0
    global_version = 0

    for contact in pair_contacts:
        client = clients[contact["pair_id"]]
        optimizers = optimizer_by_client[client.pair_id]
        losses = train_client_offline(
            client,
            contact["start_offset_s"],
            config,
            radar_encoder,
            optical_encoder,
            cross_encoder,
            head,
            attention_bias,
            optimizers["radar"],
            optimizers["optical"],
            optimizers["server"],
            criterion,
            image_size,
            device,
        )
        if losses:
            local_log.append({
                "pair_id": client.pair_id,
                "contact_end_utc": contact["end_utc"],
                "local_steps": len(losses),
                "mean_loss": round(sum(losses) / len(losses), 8),
            })

        will_aggregate = len(pending) + 1 >= clients_per_aggregation
        estimate = estimate_transaction(client, contact, config, global_states, will_aggregate)
        start_s = max(contact["start_offset_s"], ground_available_s)
        finish_s = start_s + estimate["duration_s"]
        deadline_s = contact["end_offset_s"] - float(config["link"]["safety_margin_s"])
        status, reason = "completed", ""
        aggregation_performed, aggregation_members = False, ""
        test_accuracy, test_miou = "", ""
        if not losses:
            status, reason = "skipped", "dataset_exhausted"
        elif start_s >= contact["end_offset_s"]:
            status, reason = "skipped", "ground_station_busy_until_disconnect"
        elif finish_s > deadline_s:
            status, reason = "skipped", "fedavg_transaction_does_not_fit_contact"
        else:
            pending.append(contribution_from_client(client))
            if len(pending) == clients_per_aggregation:
                aggregation_members = ";".join(item.client_id for item in pending)
                global_states = {
                    "radar": average_states([item.radar_encoder_state for item in pending]),
                    "optical": average_states([item.optical_encoder_state for item in pending]),
                    "cross": average_states([item.cross_encoder_state for item in pending]),
                    "head": average_states([item.segmentation_head_state for item in pending]),
                }
                global_version += 1
                aggregation_performed = True
                aggregation_log.append({
                    "global_version": global_version,
                    "finish_utc": utc_at(epoch, finish_s),
                    "clients": aggregation_members,
                    "client_count": clients_per_aggregation,
                    "weight_per_client": round(1.0 / clients_per_aggregation, 8),
                })
                radar_encoder.load_state_dict(global_states["radar"])
                optical_encoder.load_state_dict(global_states["optical"])
                cross_encoder.load_state_dict(global_states["cross"])
                head.load_state_dict(global_states["head"])
                accuracy, miou, _ = evaluate_global(
                    dataset_bundle.validation,
                    radar_encoder,
                    optical_encoder,
                    cross_encoder,
                    head,
                    attention_bias,
                    {"radar_encoder": global_states["radar"], "optical_encoder": global_states["optical"]},
                    config,
                    device,
                )
                for item in pending:
                    reset_client_from_global(clients[item.client_id], global_states, global_version)
                pending.clear()
                test_accuracy, test_miou = accuracy, miou
                elapsed_s = time.perf_counter() - run_started_at
                print(
                    f"[fedavg-aggregation] window_end_utc={contact['end_utc']} "
                    f"runtime=[{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}] "
                    f"test_accuracy={accuracy:.8f} test_miou={miou:.8f}",
                    flush=True,
                )
            ground_available_s = finish_s
        contact_log.append({
            "pair_contact_id": contact["pair_contact_id"],
            "pair_id": client.pair_id,
            "contact_start_utc": contact["start_utc"],
            "contact_end_utc": contact["end_utc"],
            "status": status,
            "reason": reason,
            "local_steps": len(losses),
            "model_bytes": estimate["model_bytes"],
            "aggregation_performed": int(aggregation_performed),
            "aggregation_members": aggregation_members,
            "global_version": global_version,
            "test_accuracy": test_accuracy,
            "test_miou": test_miou,
            "modeled_transaction_s": round(estimate["duration_s"], 8),
        })
        client.local_clock_s = max(client.local_clock_s, contact["end_offset_s"])

    for client in clients.values():
        train_client_offline(
            client,
            horizon_s,
            config,
            radar_encoder,
            optical_encoder,
            cross_encoder,
            head,
            attention_bias,
            optimizer_by_client[client.pair_id]["radar"],
            optimizer_by_client[client.pair_id]["optical"],
            optimizer_by_client[client.pair_id]["server"],
            criterion,
            image_size,
            device,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fedavg_training_log.csv", contact_log)
    write_csv(output_dir / "fedavg_local_training_log.csv", local_log)
    write_csv(output_dir / "fedavg_aggregation_log.csv", aggregation_log)
    radar_encoder.load_state_dict(global_states["radar"])
    optical_encoder.load_state_dict(global_states["optical"])
    cross_encoder.load_state_dict(global_states["cross"])
    head.load_state_dict(global_states["head"])
    final_accuracy, final_miou, _ = evaluate_global(
        dataset_bundle.validation,
        radar_encoder,
        optical_encoder,
        cross_encoder,
        head,
        attention_bias,
        {"radar_encoder": global_states["radar"], "optical_encoder": global_states["optical"]},
        config,
        device,
    )
    summary = {
        "algorithm": "multimodal FedAvg with complete local models and contact windows",
        "device": str(device),
        "checkpoint": checkpoint_status,
        "dataset": dataset_bundle.metadata.name,
        "client_definition": "one radar-optical pair per orbital plane",
        "pair_count": len(clients),
        "pair_contact_windows": len(pair_contacts),
        "successful_contacts": sum(row["status"] == "completed" for row in contact_log),
        "skipped_contacts": sum(row["status"] == "skipped" for row in contact_log),
        "local_steps_per_disconnection": int(training["local_steps_per_disconnection"]),
        "aggregation_k": clients_per_aggregation,
        "global_versions": global_version,
        "final_test_accuracy": final_accuracy,
        "final_test_miou": final_miou,
    }
    with (output_dir / "fedavg_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    torch.save(
        {
            "config": config,
            "global_version": global_version,
            "global_states": global_states,
            "client_states": {
                client_id: {
                    "radar_encoder": client.radar_encoder_state,
                    "optical_encoder": client.optical_encoder_state,
                    "cross_encoder": client.cross_encoder_state,
                    "segmentation_head": client.segmentation_head_state,
                }
                for client_id, client in clients.items()
            },
        },
        output_dir / "fedavg_final_checkpoint.pt",
    )
    print(json.dumps(summary, indent=2))
    print(f"Training log: {output_dir / 'fedavg_training_log.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.json")
    parser.add_argument("--contacts", type=Path, default=PROJECT_DIR / "outputs" / "contact_windows.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_dir = create_timestamped_run_dir(args.output_dir, config_path)
    run_training(load_json(config_path), load_raw_contacts(args.contacts), run_dir)
    print(f"Run output directory: {run_dir}")


if __name__ == "__main__":
    main()
