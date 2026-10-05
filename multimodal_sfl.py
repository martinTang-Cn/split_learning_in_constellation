"""Single-device SFL state, local training, fusion training, and aggregation."""

from __future__ import annotations

import copy
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
    contrastive_state: dict[str, torch.Tensor] = field(default_factory=dict)


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
    radar_contrastive_state: dict[str, torch.Tensor] = field(default_factory=dict)
    optical_contrastive_state: dict[str, torch.Tensor] = field(default_factory=dict)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def state_nbytes(state: dict[str, torch.Tensor]) -> int:
    return sum(tensor_nbytes(value) for value in state.values())


def clone_state(module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def make_ema_teacher(student):
    """Create a frozen teacher initialized from the current student weights."""
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher, student, decay: float) -> None:
    """Move teacher parameters toward the updated student parameters."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("ema_teacher_decay must be in the interval [0, 1)")
    teacher_parameters = dict(teacher.named_parameters())
    for name, student_parameter in student.named_parameters():
        teacher_parameter = teacher_parameters[name]
        teacher_parameter.mul_(decay).add_(student_parameter.detach(), alpha=1.0 - decay)
    # Keep non-parameter buffers (if a future encoder adds any) synchronized.
    teacher_buffers = dict(teacher.named_buffers())
    for name, student_buffer in student.named_buffers():
        teacher_buffers[name].copy_(student_buffer.detach())
    teacher.eval()


def annealed_value(start: float, end: float, step: int, anneal_steps: int) -> float:
    """Linearly move from start to end over server update steps."""
    if anneal_steps <= 0:
        return float(end)
    progress = min(max(float(step) / float(anneal_steps), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * progress


def symmetric_contrastive_loss(
    radar_embeddings: torch.Tensor,
    optical_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute bidirectional in-batch InfoNCE for paired satellite embeddings."""
    if temperature <= 0.0:
        raise ValueError("contrastive_temperature must be positive")
    if radar_embeddings.ndim != 2 or optical_embeddings.ndim != 2:
        raise ValueError("Contrastive embeddings must have shape (batch, dimension)")
    if radar_embeddings.shape != optical_embeddings.shape:
        raise ValueError("Radar and optical contrastive embeddings must have the same shape")
    if radar_embeddings.shape[0] < 2:
        # A one-sample batch has no in-batch negative. Keep a connected zero so
        # the caller can include this term without special-casing backward().
        return (radar_embeddings.sum() + optical_embeddings.sum()) * 0.0
    logits = radar_embeddings @ optical_embeddings.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels)
    )


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


def _trainable_parameters(module):
    """Return only parameters enabled by the current freeze schedule."""
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def train_pair_offline(
    pair: PlanePair, stop_time_s, config, radar_worker, optical_worker, radar_auxiliary,
    optical_auxiliary, radar_projection, optical_projection, attention_bias,
    criterion, device, epoch, log_rows, radar_contrastive=None,
    optical_contrastive=None,
):
    """Train both satellite branches during one pair's invisible interval."""
    training = config["segmentation_training"]
    if not pair.has_local_work:
        return
    radar_worker.load_state_dict(pair.radar.encoder_state)
    optical_worker.load_state_dict(pair.optical.encoder_state)
    radar_auxiliary.load_state_dict(pair.radar.auxiliary_state)
    # A shared auxiliary head is represented by passing the same module for
    # both arguments.  Load it once so the optical branch does not overwrite
    # the radar branch's latest shared-head state.
    if optical_auxiliary is not radar_auxiliary:
        optical_auxiliary.load_state_dict(pair.optical.auxiliary_state)
    radar_projection.load_state_dict(pair.radar.projection_state)
    optical_projection.load_state_dict(pair.optical.projection_state)
    if radar_contrastive is not None and pair.radar.contrastive_state:
        radar_contrastive.load_state_dict(pair.radar.contrastive_state)
    if optical_contrastive is not None and pair.optical.contrastive_state:
        optical_contrastive.load_state_dict(pair.optical.contrastive_state)
    freeze_projection = bool(training.get("freeze_projection_during_disconnection", True))
    radar_projection.requires_grad_(not freeze_projection)
    optical_projection.requires_grad_(not freeze_projection)
    local_parameters = [
        {"params": _trainable_parameters(radar_worker), "lr": training["encoder_learning_rate"]},
        {"params": _trainable_parameters(optical_worker), "lr": training["encoder_learning_rate"]},
        {"params": _trainable_parameters(radar_auxiliary), "lr": training["auxiliary_learning_rate"]},
    ]
    if optical_auxiliary is not radar_auxiliary:
        local_parameters.append(
            {"params": _trainable_parameters(optical_auxiliary), "lr": training["auxiliary_learning_rate"]}
        )
    if not freeze_projection:
        local_parameters.extend([
            {"params": _trainable_parameters(radar_projection), "lr": training["auxiliary_learning_rate"]},
            {"params": _trainable_parameters(optical_projection), "lr": training["auxiliary_learning_rate"]},
        ])
    contrastive_enabled = bool(training.get("contrastive_learning_enabled", radar_contrastive is not None and optical_contrastive is not None))
    if contrastive_enabled and (radar_contrastive is None or optical_contrastive is None):
        raise ValueError("Contrastive learning requires both radar and optical projection heads")
    if contrastive_enabled:
        local_parameters.extend([
            {"params": _trainable_parameters(radar_contrastive), "lr": training["auxiliary_learning_rate"]},
            {"params": _trainable_parameters(optical_contrastive), "lr": training["auxiliary_learning_rate"]},
        ])
    weight_decay = float(training.get("weight_decay", 0.01))
    local_optimizer = torch.optim.AdamW(local_parameters, weight_decay=weight_decay)
    step_s = max(float(training["radar_local_compute_s"]), float(training["optical_local_compute_s"]))
    buffer_limit = int(training["recent_smashed_batches"])
    image_size = int(config["croma"]["patch_size"]) * math.isqrt(int(config["croma"]["num_patches"]))
    contrastive_temperature = float(training.get("contrastive_temperature", 0.07))
    contrastive_weight = float(training.get("contrastive_loss_weight", 0.1))
    if contrastive_enabled and contrastive_temperature <= 0.0:
        raise ValueError("contrastive_temperature must be positive")
    if contrastive_weight < 0.0:
        raise ValueError("contrastive_loss_weight must be non-negative")
    completed = 0
    while completed < int(training["local_steps_per_disconnection"]) and pair.has_local_work and pair.local_clock_s + step_s <= stop_time_s:
        epoch_number, batch_number, sample_ids, radar, optical, labels = pair.take_batch()
        if not torch.any(labels != criterion.ignore_index):
            continue
        radar, optical, labels_device = radar.to(device), optical.to(device), labels.to(device)
        step_start = pair.local_clock_s
        local_optimizer.zero_grad()
        radar_tokens = radar_worker(radar, attention_bias, mask_info=None)
        optical_tokens = optical_worker(optical, attention_bias, mask_info=None)
        radar_projected = radar_projection(radar_tokens)
        optical_projected = optical_projection(optical_tokens)
        radar_loss = criterion(radar_auxiliary(radar_projected, image_size), labels_device)
        optical_loss = criterion(optical_auxiliary(optical_projected, image_size), labels_device)
        if contrastive_enabled:
            radar_embedding = radar_contrastive(radar_tokens)
            optical_embedding = optical_contrastive(optical_tokens)
            contrastive_loss = symmetric_contrastive_loss(
                radar_embedding, optical_embedding, contrastive_temperature
            )
        else:
            contrastive_loss = radar_loss.new_zeros(())
        total_loss = radar_loss + optical_loss + contrastive_weight * contrastive_loss
        total_loss.backward()
        local_optimizer.step()
        pair.radar.local_version += 1
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
            "contrastive_loss": round(float(contrastive_loss.item()), 8),
            "radar_local_version": pair.radar.local_version, "optical_local_version": pair.optical.local_version,
            "matched_buffered_batches": len(pair.matched_batch_ids()),
        })
    shared_auxiliary_state = clone_state(radar_auxiliary)
    pair.radar.encoder_state, pair.radar.auxiliary_state = clone_state(radar_worker), shared_auxiliary_state
    pair.optical.encoder_state = clone_state(optical_worker)
    pair.optical.auxiliary_state = clone_state(optical_auxiliary)
    pair.radar.projection_state = clone_state(radar_projection)
    pair.optical.projection_state = clone_state(optical_projection)
    if radar_contrastive is not None:
        pair.radar.contrastive_state = clone_state(radar_contrastive)
    if optical_contrastive is not None:
        pair.optical.contrastive_state = clone_state(optical_contrastive)


def _packet_bytes(packet: FeaturePacket, include_labels: bool) -> int:
    total = tensor_nbytes(packet.tokens) + tensor_nbytes(packet.sample_ids)
    return total + tensor_nbytes(packet.labels) if include_labels else total


def estimate_transaction(pair, matched_ids, contact, config, will_aggregate):
    """Calculate one paired satellite-ground transaction on parallel links."""
    link, training = config["link"], config["segmentation_training"]
    overhead = int(link["protocol_overhead_bytes"])
    radar_up = state_nbytes(pair.radar.encoder_state) + state_nbytes(pair.radar.auxiliary_state) + sum(_packet_bytes(pair.radar_buffer[key], True) for key in matched_ids) + overhead
    optical_up = state_nbytes(pair.optical.encoder_state) + state_nbytes(pair.optical.auxiliary_state) + sum(_packet_bytes(pair.optical_buffer[key], False) for key in matched_ids) + overhead
    radar_up += state_nbytes(pair.radar.contrastive_state)
    optical_up += state_nbytes(pair.optical.contrastive_state)
    radar_down = state_nbytes(pair.radar.encoder_state) + state_nbytes(pair.radar.auxiliary_state) + state_nbytes(pair.radar.projection_state) + overhead
    optical_down = state_nbytes(pair.optical.encoder_state) + state_nbytes(pair.optical.auxiliary_state) + state_nbytes(pair.optical.projection_state) + overhead
    radar_down += state_nbytes(pair.radar.contrastive_state)
    optical_down += state_nbytes(pair.optical.contrastive_state)
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
    ema_teacher=None, ema_decay=0.99, ema_decay_start=None,
    ema_decay_anneal_steps=0, distillation_weight_start=1.0,
    distillation_weight_end=1.0, distillation_anneal_steps=0, server_step=0,
):
    losses = {
        "segmentation": [], "radar_distillation": [], "optical_distillation": [],
        "ema_teacher_decay": [], "distillation_weight": [],
    }
    radar_projection.requires_grad_(True)
    optical_projection.requires_grad_(True)
    decay_start = float(ema_decay if ema_decay_start is None else ema_decay_start)
    decay_end = float(ema_decay)
    for batch_offset, batch_id in enumerate(matched_ids):
        update_step = int(server_step) + batch_offset
        current_decay = annealed_value(
            decay_start, decay_end, update_step, int(ema_decay_anneal_steps)
        )
        current_distillation_weight = annealed_value(
            float(distillation_weight_start), float(distillation_weight_end),
            update_step, int(distillation_anneal_steps),
        )
        radar_packet, optical_packet = pair.radar_buffer[batch_id], pair.optical_buffer[batch_id]
        if not torch.equal(radar_packet.sample_ids, optical_packet.sample_ids):
            raise RuntimeError(f"Unmatched sample IDs in pair {pair.pair_id}")
        optimizer.zero_grad()
        fused = cross_encoder(radar_packet.tokens.to(device), optical_packet.tokens.to(device), attention_bias)
        if ema_teacher is None:
            distillation_target = fused.detach()
        else:
            with torch.no_grad():
                distillation_target = ema_teacher(
                    radar_packet.tokens.to(device), optical_packet.tokens.to(device), attention_bias
                )
        radar_features = radar_projection(radar_packet.tokens.to(device))
        optical_features = optical_projection(optical_packet.tokens.to(device))
        segmentation_loss = criterion(ground_head(fused, image_size), radar_packet.labels.to(device))
        radar_distillation_loss = torch.nn.functional.mse_loss(radar_features, distillation_target)
        optical_distillation_loss = torch.nn.functional.mse_loss(optical_features, distillation_target)
        loss = segmentation_loss + current_distillation_weight * (
            radar_distillation_loss + optical_distillation_loss
        )
        loss.backward()
        optimizer.step()
        if ema_teacher is not None:
            update_ema_teacher(ema_teacher, cross_encoder, current_decay)
        losses["segmentation"].append(float(segmentation_loss.item()))
        losses["radar_distillation"].append(float(radar_distillation_loss.item()))
        losses["optical_distillation"].append(float(optical_distillation_loss.item()))
        losses["ema_teacher_decay"].append(current_decay)
        losses["distillation_weight"].append(current_distillation_weight)
    return losses


def reset_pair_from_global(pair, global_states, global_version):
    """Sync newer global encoders/heads and always download ground projections."""
    for modality in (pair.radar, pair.optical):
        prefix = modality.modality
        # Preserve local progress until the server has a newer global version.
        if global_version > modality.downloaded_global_version:
            modality.encoder_state = {
                key: value.clone() for key, value in global_states[f"{prefix}_encoder"].items()
            }
            modality.auxiliary_state = {
                key: value.clone() for key, value in global_states[f"{prefix}_auxiliary"].items()
            }
            modality.downloaded_global_version = global_version
        # Ground distillation changes projections independently of aggregation.
        modality.projection_state = {
            key: value.clone() for key, value in global_states[f"{prefix}_projection"].items()
        }
    if "radar_contrastive" in global_states:
        pair.radar.contrastive_state = {key: value.clone() for key, value in global_states["radar_contrastive"].items()}
    if "optical_contrastive" in global_states:
        pair.optical.contrastive_state = {key: value.clone() for key, value in global_states["optical_contrastive"].items()}
        
