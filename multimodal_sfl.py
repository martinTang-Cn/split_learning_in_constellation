"""Single-device SFL state, local training, fusion training, and aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from orbit_model import SPEED_OF_LIGHT_KM_S
from pair_contact_scheduler import utc_at


@dataclass
class FeaturePacket:
    tokens: torch.Tensor
    labels: torch.Tensor
    sample_ids: torch.Tensor
    batch_number: int
    local_version: int


@dataclass
class ModalityState:
    satellite_id: str
    modality: str
    encoder_state: dict[str, torch.Tensor]
    auxiliary_state: dict[str, torch.Tensor]
    projection_state: dict[str, torch.Tensor]
    local_version: int = 0
    downloaded_global_version: int = 0


@dataclass
class PlanePair:
    pair_id: str
    plane: int
    radar: ModalityState
    optical: ModalityState
    batches: list[tuple]
    next_batch_index: int = 0
    local_clock_s: float = 0.0
    radar_buffer: dict[int, FeaturePacket] = field(default_factory=dict)
    optical_buffer: dict[int, FeaturePacket] = field(default_factory=dict)

    @property
    def has_local_work(self) -> bool:
        return self.next_batch_index < len(self.batches)

    def take_batch(self):
        batch = self.batches[self.next_batch_index]
        self.next_batch_index += 1
        return batch

    def matched_batch_ids(self) -> list[int]:
        return sorted(set(self.radar_buffer) & set(self.optical_buffer))


@dataclass
class PairContribution:
    pair_id: str
    radar_encoder_state: dict[str, torch.Tensor]
    radar_auxiliary_state: dict[str, torch.Tensor]
    optical_encoder_state: dict[str, torch.Tensor]
    optical_auxiliary_state: dict[str, torch.Tensor]


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def state_nbytes(state: dict[str, torch.Tensor]) -> int:
    return sum(tensor_nbytes(value) for value in state.values())


def clone_state(module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def average_states(states: list[dict[str, torch.Tensor]]):
    if not states:
        raise ValueError("Cannot aggregate an empty list")
    result = {}
    for key in states[0]:
        tensors = [state[key] for state in states]
        result[key] = torch.stack(tensors).mean(dim=0) if tensors[0].is_floating_point() else tensors[0].clone()
    return result


def _prune_buffer(buffer: dict[int, FeaturePacket], limit: int) -> None:
    while len(buffer) > limit:
        del buffer[next(iter(buffer))]


def train_pair_offline(
    pair: PlanePair, stop_time_s, config, radar_worker, optical_worker, radar_auxiliary,
    optical_auxiliary, radar_projection, optical_projection, attention_bias,
    criterion, device, epoch, log_rows,
):
    """Train both satellite branches during one pair's invisible interval."""
    training = config["segmentation_training"]
    if not pair.has_local_work:
        return
    radar_worker.load_state_dict(pair.radar.encoder_state)
    optical_worker.load_state_dict(pair.optical.encoder_state)
    radar_auxiliary.load_state_dict(pair.radar.auxiliary_state)
    optical_auxiliary.load_state_dict(pair.optical.auxiliary_state)
    radar_projection.load_state_dict(pair.radar.projection_state)
    optical_projection.load_state_dict(pair.optical.projection_state)
    freeze_projection = bool(training.get("freeze_projection_during_disconnection", True))
    radar_projection.requires_grad_(not freeze_projection)
    optical_projection.requires_grad_(not freeze_projection)
    radar_parameters = [
        {"params": radar_worker.parameters(), "lr": training["encoder_learning_rate"]},
        {"params": radar_auxiliary.parameters(), "lr": training["auxiliary_learning_rate"]},
    ]
    optical_parameters = [
        {"params": optical_worker.parameters(), "lr": training["encoder_learning_rate"]},
        {"params": optical_auxiliary.parameters(), "lr": training["auxiliary_learning_rate"]},
    ]
    if not freeze_projection:
        radar_parameters.insert(1, {"params": radar_projection.parameters(), "lr": training["auxiliary_learning_rate"]})
        optical_parameters.insert(1, {"params": optical_projection.parameters(), "lr": training["auxiliary_learning_rate"]})
    weight_decay = float(training.get("weight_decay", 0.01))
    radar_optimizer = torch.optim.AdamW(radar_parameters, weight_decay=weight_decay)
    optical_optimizer = torch.optim.AdamW(optical_parameters, weight_decay=weight_decay)
    step_s = max(float(training["radar_local_compute_s"]), float(training["optical_local_compute_s"]))
    buffer_limit = int(training["recent_smashed_batches"])
    image_size = int(config["croma"]["patch_size"]) * math.isqrt(int(config["croma"]["num_patches"]))
    completed = 0
    while completed < int(training["local_steps_per_disconnection"]) and pair.has_local_work and pair.local_clock_s + step_s <= stop_time_s:
        epoch_number, batch_number, sample_ids, radar, optical, labels = pair.take_batch()
        if not torch.any(labels != criterion.ignore_index):
            continue
        radar, optical, labels_device = radar.to(device), optical.to(device), labels.to(device)
        step_start = pair.local_clock_s
        radar_optimizer.zero_grad()
        radar_tokens = radar_worker(radar, attention_bias, mask_info=None)
        radar_projected = radar_projection(radar_tokens)
        radar_loss = criterion(radar_auxiliary(radar_projected, image_size), labels_device)
        radar_loss.backward()
        radar_optimizer.step()
        pair.radar.local_version += 1
        optical_optimizer.zero_grad()
        optical_tokens = optical_worker(optical, attention_bias, mask_info=None)
        optical_projected = optical_projection(optical_tokens)
        optical_loss = criterion(optical_auxiliary(optical_projected, image_size), labels_device)
        optical_loss.backward()
        optical_optimizer.step()
        pair.optical.local_version += 1
        pair.radar_buffer[batch_number] = FeaturePacket(radar_tokens.detach().cpu().clone(), labels.clone(), sample_ids.clone(), batch_number, pair.radar.local_version)
        pair.optical_buffer[batch_number] = FeaturePacket(optical_tokens.detach().cpu().clone(), labels.clone(), sample_ids.clone(), batch_number, pair.optical.local_version)
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
            "radar_local_version": pair.radar.local_version, "optical_local_version": pair.optical.local_version,
            "matched_buffered_batches": len(pair.matched_batch_ids()),
        })
    pair.radar.encoder_state, pair.radar.auxiliary_state = clone_state(radar_worker), clone_state(radar_auxiliary)
    pair.optical.encoder_state, pair.optical.auxiliary_state = clone_state(optical_worker), clone_state(optical_auxiliary)
    pair.radar.projection_state = clone_state(radar_projection)
    pair.optical.projection_state = clone_state(optical_projection)


def _packet_bytes(packet: FeaturePacket, include_labels: bool) -> int:
    total = tensor_nbytes(packet.tokens) + tensor_nbytes(packet.sample_ids)
    return total + tensor_nbytes(packet.labels) if include_labels else total


def estimate_transaction(pair, matched_ids, contact, config, will_aggregate):
    """Calculate one paired satellite-ground transaction on parallel links."""
    link, training = config["link"], config["segmentation_training"]
    overhead = int(link["protocol_overhead_bytes"])
    radar_up = state_nbytes(pair.radar.encoder_state) + state_nbytes(pair.radar.auxiliary_state) + sum(_packet_bytes(pair.radar_buffer[key], True) for key in matched_ids) + overhead
    optical_up = state_nbytes(pair.optical.encoder_state) + state_nbytes(pair.optical.auxiliary_state) + sum(_packet_bytes(pair.optical_buffer[key], False) for key in matched_ids) + overhead
    radar_down = state_nbytes(pair.radar.encoder_state) + state_nbytes(pair.radar.auxiliary_state) + state_nbytes(pair.radar.projection_state) + overhead
    optical_down = state_nbytes(pair.optical.encoder_state) + state_nbytes(pair.optical.auxiliary_state) + state_nbytes(pair.optical.projection_state) + overhead
    radar_upload_s, optical_upload_s = radar_up * 8 / (contact["uplink_mbps"] * 1e6), optical_up * 8 / (contact["uplink_mbps"] * 1e6)
    radar_download_s, optical_download_s = radar_down * 8 / (contact["downlink_mbps"] * 1e6), optical_down * 8 / (contact["downlink_mbps"] * 1e6)
    if training["paired_uplink_mode"] == "parallel_full_rate":
        upload_s, download_s = max(radar_upload_s, optical_upload_s), max(radar_download_s, optical_download_s)
    else:
        upload_s, download_s = radar_upload_s + optical_upload_s, radar_download_s + optical_download_s
    propagation_s = 2 * contact["min_slant_range_km"] / SPEED_OF_LIGHT_KM_S
    server_s = len(matched_ids) * float(training["server_compute_s_per_batch"])
    aggregation_s = float(training["aggregation_compute_s"]) if will_aggregate else 0.0
    return {
        "radar_upload_bytes": radar_up, "optical_upload_bytes": optical_up,
        "radar_download_bytes": radar_down, "optical_download_bytes": optical_down,
        "upload_s": upload_s, "download_s": download_s, "propagation_s": propagation_s,
        "server_s": server_s, "aggregation_s": aggregation_s,
        "duration_s": upload_s + propagation_s + server_s + aggregation_s + download_s,
    }


def train_server_on_matched_features(
    pair, matched_ids, cross_encoder, ground_head, radar_projection,
    optical_projection, attention_bias, optimizer, criterion, image_size, device,
):
    losses = {"segmentation": [], "radar_distillation": [], "optical_distillation": []}
    radar_projection.requires_grad_(True)
    optical_projection.requires_grad_(True)
    for batch_id in matched_ids:
        radar_packet, optical_packet = pair.radar_buffer[batch_id], pair.optical_buffer[batch_id]
        if not torch.equal(radar_packet.sample_ids, optical_packet.sample_ids):
            raise RuntimeError(f"Unmatched sample IDs in pair {pair.pair_id}")
        optimizer.zero_grad()
        fused = cross_encoder(radar_packet.tokens.to(device), optical_packet.tokens.to(device), attention_bias)
        radar_features = radar_projection(radar_packet.tokens.to(device))
        optical_features = optical_projection(optical_packet.tokens.to(device))
        segmentation_loss = criterion(ground_head(fused, image_size), radar_packet.labels.to(device))
        radar_distillation_loss = torch.nn.functional.mse_loss(radar_features, fused.detach())
        optical_distillation_loss = torch.nn.functional.mse_loss(optical_features, fused.detach())
        loss = segmentation_loss + radar_distillation_loss + optical_distillation_loss
        loss.backward()
        optimizer.step()
        losses["segmentation"].append(float(segmentation_loss.item()))
        losses["radar_distillation"].append(float(radar_distillation_loss.item()))
        losses["optical_distillation"].append(float(optical_distillation_loss.item()))
    return losses


def reset_pair_from_global(pair, global_states, global_version):
    pair.radar.encoder_state = {key: value.clone() for key, value in global_states["radar_encoder"].items()}
    pair.radar.auxiliary_state = {key: value.clone() for key, value in global_states["radar_auxiliary"].items()}
    pair.radar.projection_state = {key: value.clone() for key, value in global_states["radar_projection"].items()}
    pair.optical.encoder_state = {key: value.clone() for key, value in global_states["optical_encoder"].items()}
    pair.optical.auxiliary_state = {key: value.clone() for key, value in global_states["optical_auxiliary"].items()}
    pair.optical.projection_state = {key: value.clone() for key, value in global_states["optical_projection"].items()}
    pair.radar.downloaded_global_version = global_version
    pair.optical.downloaded_global_version = global_version
