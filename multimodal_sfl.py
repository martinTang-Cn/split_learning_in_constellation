"""Single-device SFL state, local training, fusion training, and aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F

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
class TeacherReplayPacket:
    sample_ids: torch.Tensor
    logits: torch.Tensor


@dataclass
class ModalityState:
    satellite_id: str
    modality: str
    encoder_state: dict[str, torch.Tensor]
    auxiliary_state: dict[str, torch.Tensor]
    projection_state: dict[str, torch.Tensor]
    teacher_encoder_state: dict[str, torch.Tensor] = field(default_factory=dict)
    teacher_auxiliary_state: dict[str, torch.Tensor] = field(default_factory=dict)
    teacher_projection_state: dict[str, torch.Tensor] = field(default_factory=dict)
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
    ground_teacher_replay: dict[int, TeacherReplayPacket] = field(default_factory=dict)
    teacher_replay_cursor: int = 0

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


def _trainable_parameters(module):
    """Return only parameters enabled by the current freeze schedule."""
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _disconnection_settings(config):
    settings = config.get("disconnection_training", {})
    return {
        "temperature": float(settings.get("temperature", 2.0)),
        "ground_teacher_weight": float(settings.get("ground_teacher_weight", 0.5)),
        "isl_mutual_weight": float(settings.get("isl_mutual_weight", 0.25)),
        "confidence_threshold": float(settings.get("confidence_threshold", 0.7)),
        "proximal_mu": float(settings.get("proximal_mu", 1e-4)),
        "teacher_replay_weight": float(settings.get("teacher_replay_weight", 0.5)),
        "teacher_replay_batches": max(0, int(settings.get("teacher_replay_batches", 4))),
        "teacher_replay_frequency": max(1, int(settings.get("teacher_replay_frequency", 4))),
    }


def _masked_kl(student_logits, teacher_logits, labels, temperature, confidence_threshold, ignore_index):
    """Compute confidence-filtered pixelwise KL for logits with shape (B,C,H,W)."""
    teacher_probabilities = F.softmax(teacher_logits.detach() / temperature, dim=1)
    confidence = teacher_probabilities.max(dim=1).values
    valid = confidence >= confidence_threshold
    if labels is not None:
        valid &= labels != ignore_index
    per_pixel = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        teacher_probabilities,
        reduction="none",
    ).sum(dim=1) * (temperature * temperature)
    if not torch.any(valid):
        return per_pixel.mean() * 0.0
    return per_pixel[valid].mean()


def _proximal_penalty(module, reference_state, device):
    if not reference_state:
        return torch.zeros((), device=device)
    penalty = torch.zeros((), device=device)
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or name not in reference_state:
            continue
        reference = reference_state[name].to(device=device, dtype=parameter.dtype)
        penalty = penalty + (parameter - reference).pow(2).mean()
    return penalty


def _next_teacher_replay(pair: PlanePair, completed: int, frequency: int):
    if not pair.ground_teacher_replay or completed % frequency != 0:
        return None
    keys = sorted(pair.ground_teacher_replay)
    key = keys[pair.teacher_replay_cursor % len(keys)]
    pair.teacher_replay_cursor += 1
    return pair.ground_teacher_replay[key]


def _load_replay_batch(pair: PlanePair, packet: TeacherReplayPacket, device):
    samples = [pair.batches.dataset[int(index)] for index in packet.sample_ids.tolist()]
    radar, optical, labels, _ = zip(*samples)
    return (
        torch.stack(radar).to(device),
        torch.stack(optical).to(device),
        torch.stack(labels).to(device),
        packet.logits.to(device=device, dtype=torch.float32),
    )


def train_pair_offline(
    pair: PlanePair, stop_time_s, config, radar_worker, optical_worker, radar_auxiliary,
    optical_auxiliary, radar_projection, optical_projection,
    radar_teacher_worker, optical_teacher_worker, radar_teacher_auxiliary,
    optical_teacher_auxiliary, radar_teacher_projection, optical_teacher_projection,
    attention_bias,
    criterion, device, epoch, log_rows,
):
    """Train both branches with ground-teacher, ISL, and proximal losses."""
    training = config["segmentation_training"]
    distillation = _disconnection_settings(config)
    if not pair.has_local_work:
        return
    radar_worker.load_state_dict(pair.radar.encoder_state)
    optical_worker.load_state_dict(pair.optical.encoder_state)
    radar_auxiliary.load_state_dict(pair.radar.auxiliary_state)
    optical_auxiliary.load_state_dict(pair.optical.auxiliary_state)
    radar_projection.load_state_dict(pair.radar.projection_state)
    optical_projection.load_state_dict(pair.optical.projection_state)
    radar_teacher_worker.load_state_dict(pair.radar.teacher_encoder_state or pair.radar.encoder_state)
    optical_teacher_worker.load_state_dict(pair.optical.teacher_encoder_state or pair.optical.encoder_state)
    radar_teacher_auxiliary.load_state_dict(pair.radar.teacher_auxiliary_state or pair.radar.auxiliary_state)
    optical_teacher_auxiliary.load_state_dict(pair.optical.teacher_auxiliary_state or pair.optical.auxiliary_state)
    radar_teacher_projection.load_state_dict(pair.radar.teacher_projection_state or pair.radar.projection_state)
    optical_teacher_projection.load_state_dict(pair.optical.teacher_projection_state or pair.optical.projection_state)
    for module in (
        radar_teacher_worker, optical_teacher_worker, radar_teacher_auxiliary,
        optical_teacher_auxiliary, radar_teacher_projection, optical_teacher_projection,
    ):
        module.eval()
        module.requires_grad_(False)
    radar_worker.train()
    optical_worker.train()
    radar_auxiliary.train()
    optical_auxiliary.train()
    freeze_projection = bool(training.get("freeze_projection_during_disconnection", True))
    radar_projection.requires_grad_(not freeze_projection)
    optical_projection.requires_grad_(not freeze_projection)
    radar_parameters = [
        {"params": _trainable_parameters(radar_worker), "lr": training["encoder_learning_rate"]},
        {"params": _trainable_parameters(radar_auxiliary), "lr": training["auxiliary_learning_rate"]},
    ]
    optical_parameters = [
        {"params": _trainable_parameters(optical_worker), "lr": training["encoder_learning_rate"]},
        {"params": _trainable_parameters(optical_auxiliary), "lr": training["auxiliary_learning_rate"]},
    ]
    if not freeze_projection:
        radar_parameters.insert(1, {"params": _trainable_parameters(radar_projection), "lr": training["auxiliary_learning_rate"]})
        optical_parameters.insert(1, {"params": _trainable_parameters(optical_projection), "lr": training["auxiliary_learning_rate"]})
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

        with torch.no_grad():
            teacher_r_tokens = radar_teacher_worker(radar, attention_bias, mask_info=None)
            teacher_o_tokens = optical_teacher_worker(optical, attention_bias, mask_info=None)
            teacher_r_logits = radar_teacher_auxiliary(
                radar_teacher_projection(teacher_r_tokens), image_size
            )
            teacher_o_logits = optical_teacher_auxiliary(
                optical_teacher_projection(teacher_o_tokens), image_size
            )

        # Compute both student predictions before either optimizer step so the
        # mutual-distillation terms use the same paired samples.
        optical_optimizer.zero_grad()
        optical_tokens = optical_worker(optical, attention_bias, mask_info=None)
        optical_projected = optical_projection(optical_tokens)
        optical_logits = optical_auxiliary(optical_projected, image_size)

        radar_optimizer.zero_grad()
        radar_tokens = radar_worker(radar, attention_bias, mask_info=None)
        radar_projected = radar_projection(radar_tokens)
        radar_logits = radar_auxiliary(radar_projected, image_size)
        replay_packet = _next_teacher_replay(
            pair, completed, distillation["teacher_replay_frequency"]
        )
        replay_radar_logits = replay_optical_logits = replay_labels = replay_teacher_logits = None
        if replay_packet is not None:
            replay_radar, replay_optical, replay_labels, replay_teacher_logits = _load_replay_batch(
                pair, replay_packet, device
            )
            replay_radar_tokens = radar_worker(replay_radar, attention_bias, mask_info=None)
            replay_optical_tokens = optical_worker(replay_optical, attention_bias, mask_info=None)
            replay_radar_logits = radar_auxiliary(
                radar_projection(replay_radar_tokens), image_size
            )
            replay_optical_logits = optical_auxiliary(
                optical_projection(replay_optical_tokens), image_size
            )
        radar_segmentation_loss = criterion(radar_logits, labels_device)
        radar_teacher_loss = _masked_kl(
            radar_logits, teacher_r_logits, labels_device,
            distillation["temperature"], distillation["confidence_threshold"], criterion.ignore_index,
        )
        radar_replay_loss = (
            _masked_kl(
                replay_radar_logits, replay_teacher_logits, replay_labels,
                distillation["temperature"], distillation["confidence_threshold"], criterion.ignore_index,
            ) if replay_packet is not None else radar_segmentation_loss * 0.0
        )
        radar_mutual_loss = _masked_kl(
            radar_logits, optical_logits.detach(),
            labels_device, distillation["temperature"], distillation["confidence_threshold"], criterion.ignore_index,
        )
        radar_proximal_loss = (
            _proximal_penalty(radar_worker, pair.radar.encoder_state, device)
            + _proximal_penalty(radar_auxiliary, pair.radar.auxiliary_state, device)
            + _proximal_penalty(radar_projection, pair.radar.projection_state, device)
        )
        radar_loss = (
            radar_segmentation_loss
            + distillation["ground_teacher_weight"] * radar_teacher_loss
            + distillation["teacher_replay_weight"] * radar_replay_loss
            + distillation["isl_mutual_weight"] * radar_mutual_loss
            + distillation["proximal_mu"] * radar_proximal_loss
        )
        radar_loss.backward()
        radar_optimizer.step()
        pair.radar.local_version += 1
        optical_segmentation_loss = criterion(optical_logits, labels_device)
        optical_teacher_loss = _masked_kl(
            optical_logits, teacher_o_logits, labels_device,
            distillation["temperature"], distillation["confidence_threshold"], criterion.ignore_index,
        )
        optical_replay_loss = (
            _masked_kl(
                replay_optical_logits, replay_teacher_logits, replay_labels,
                distillation["temperature"], distillation["confidence_threshold"], criterion.ignore_index,
            ) if replay_packet is not None else optical_segmentation_loss * 0.0
        )
        optical_mutual_loss = _masked_kl(
            optical_logits, radar_logits.detach(), labels_device,
            distillation["temperature"], distillation["confidence_threshold"], criterion.ignore_index,
        )
        optical_proximal_loss = (
            _proximal_penalty(optical_worker, pair.optical.encoder_state, device)
            + _proximal_penalty(optical_auxiliary, pair.optical.auxiliary_state, device)
            + _proximal_penalty(optical_projection, pair.optical.projection_state, device)
        )
        optical_loss = (
            optical_segmentation_loss
            + distillation["ground_teacher_weight"] * optical_teacher_loss
            + distillation["teacher_replay_weight"] * optical_replay_loss
            + distillation["isl_mutual_weight"] * optical_mutual_loss
            + distillation["proximal_mu"] * optical_proximal_loss
        )
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
            "radar_auxiliary_loss": round(float(radar_segmentation_loss.item()), 8),
            "optical_auxiliary_loss": round(float(optical_segmentation_loss.item()), 8),
            "radar_ground_teacher_loss": round(float(radar_teacher_loss.item()), 8),
            "optical_ground_teacher_loss": round(float(optical_teacher_loss.item()), 8),
            "radar_mutual_loss": round(float(radar_mutual_loss.item()), 8),
            "optical_mutual_loss": round(float(optical_mutual_loss.item()), 8),
            "radar_proximal_loss": round(float(radar_proximal_loss.item()), 8),
            "optical_proximal_loss": round(float(optical_proximal_loss.item()), 8),
            "ground_teacher_replay_used": int(replay_packet is not None),
            "radar_local_version": pair.radar.local_version, "optical_local_version": pair.optical.local_version,
            "matched_buffered_batches": len(pair.matched_batch_ids()),
        })
    pair.radar.encoder_state, pair.radar.auxiliary_state = clone_state(radar_worker), clone_state(radar_auxiliary)
    pair.optical.encoder_state, pair.optical.auxiliary_state = clone_state(optical_worker), clone_state(optical_auxiliary)
    pair.radar.projection_state = clone_state(radar_projection)
    pair.optical.projection_state = clone_state(optical_projection)


def _prune_teacher_replay(pair: PlanePair, limit: int) -> None:
    while len(pair.ground_teacher_replay) > limit:
        del pair.ground_teacher_replay[next(iter(pair.ground_teacher_replay))]


def _packet_bytes(packet: FeaturePacket, include_labels: bool) -> int:
    total = tensor_nbytes(packet.tokens) + tensor_nbytes(packet.sample_ids)
    return total + tensor_nbytes(packet.labels) if include_labels else total


def estimate_transaction(pair, matched_ids, contact, config, will_aggregate, include_teacher_packets=False):
    """Calculate one paired satellite-ground transaction on parallel links."""
    link, training = config["link"], config["segmentation_training"]
    overhead = int(link["protocol_overhead_bytes"])
    radar_up = state_nbytes(pair.radar.encoder_state) + state_nbytes(pair.radar.auxiliary_state) + sum(_packet_bytes(pair.radar_buffer[key], True) for key in matched_ids) + overhead
    optical_up = state_nbytes(pair.optical.encoder_state) + state_nbytes(pair.optical.auxiliary_state) + sum(_packet_bytes(pair.optical_buffer[key], False) for key in matched_ids) + overhead
    radar_down = state_nbytes(pair.radar.encoder_state) + state_nbytes(pair.radar.auxiliary_state) + state_nbytes(pair.radar.projection_state) + overhead
    optical_down = state_nbytes(pair.optical.encoder_state) + state_nbytes(pair.optical.auxiliary_state) + state_nbytes(pair.optical.projection_state) + overhead
    teacher_down = 0
    if include_teacher_packets:
        teacher_settings = _disconnection_settings(config)
        teacher_count = min(len(matched_ids), teacher_settings["teacher_replay_batches"])
        num_classes = int(config.get("dataset_metadata", {}).get("num_classes", 0))
        if teacher_count and num_classes:
            for batch_id in matched_ids[-teacher_count:]:
                packet = pair.radar_buffer[batch_id]
                teacher_down += packet.labels.numel() * num_classes * 2 + tensor_nbytes(packet.sample_ids)
            teacher_down += overhead
    radar_down += teacher_down
    optical_down += teacher_down
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
    pair, matched_ids, cross_encoder, ground_head, radar_auxiliary,
    optical_auxiliary, radar_projection, optical_projection, attention_bias,
    optimizer, criterion, image_size, device, config,
):
    losses = {
        "segmentation": [], "radar_distillation": [], "optical_distillation": [],
        "radar_ground_teacher": [], "optical_ground_teacher": [],
    }
    distillation = _disconnection_settings(config)
    radar_projection.requires_grad_(True)
    optical_projection.requires_grad_(True)
    radar_auxiliary.requires_grad_(True)
    optical_auxiliary.requires_grad_(True)
    for batch_id in matched_ids:
        radar_packet, optical_packet = pair.radar_buffer[batch_id], pair.optical_buffer[batch_id]
        if not torch.equal(radar_packet.sample_ids, optical_packet.sample_ids):
            raise RuntimeError(f"Unmatched sample IDs in pair {pair.pair_id}")
        optimizer.zero_grad()
        fused = cross_encoder(radar_packet.tokens.to(device), optical_packet.tokens.to(device), attention_bias)
        radar_features = radar_projection(radar_packet.tokens.to(device))
        optical_features = optical_projection(optical_packet.tokens.to(device))
        labels = radar_packet.labels.to(device)
        ground_logits = ground_head(fused, image_size)
        segmentation_loss = criterion(ground_logits, labels)
        radar_distillation_loss = torch.nn.functional.mse_loss(radar_features, fused.detach())
        optical_distillation_loss = torch.nn.functional.mse_loss(optical_features, fused.detach())
        radar_teacher_logits = radar_auxiliary(radar_features, image_size)
        optical_teacher_logits = optical_auxiliary(optical_features, image_size)
        radar_ground_teacher_loss = (
            criterion(radar_teacher_logits, labels)
            + _masked_kl(
                radar_teacher_logits, ground_logits.detach(), labels,
                distillation["temperature"], distillation["confidence_threshold"], criterion.ignore_index,
            )
        )
        optical_ground_teacher_loss = (
            criterion(optical_teacher_logits, labels)
            + _masked_kl(
                optical_teacher_logits, ground_logits.detach(), labels,
                distillation["temperature"], distillation["confidence_threshold"], criterion.ignore_index,
            )
        )
        loss = (
            segmentation_loss + radar_distillation_loss + optical_distillation_loss
            + distillation["ground_teacher_weight"]
            * (radar_ground_teacher_loss + optical_ground_teacher_loss)
        )
        loss.backward()
        optimizer.step()
        losses["segmentation"].append(float(segmentation_loss.item()))
        losses["radar_distillation"].append(float(radar_distillation_loss.item()))
        losses["optical_distillation"].append(float(optical_distillation_loss.item()))
        losses["radar_ground_teacher"].append(float(radar_ground_teacher_loss.item()))
        losses["optical_ground_teacher"].append(float(optical_ground_teacher_loss.item()))
        replay_limit = distillation["teacher_replay_batches"]
        if replay_limit:
            pair.ground_teacher_replay[batch_id] = TeacherReplayPacket(
                radar_packet.sample_ids.clone(), ground_logits.detach().cpu().half()
            )
            _prune_teacher_replay(pair, replay_limit)
    return losses


def reset_pair_from_global(pair, global_states, global_version):
    pair.radar.encoder_state = {key: value.clone() for key, value in global_states["radar_encoder"].items()}
    pair.radar.auxiliary_state = {key: value.clone() for key, value in global_states["radar_auxiliary"].items()}
    pair.radar.projection_state = {key: value.clone() for key, value in global_states["radar_projection"].items()}
    pair.radar.teacher_encoder_state = {key: value.clone() for key, value in global_states["radar_encoder"].items()}
    pair.radar.teacher_auxiliary_state = {key: value.clone() for key, value in global_states["radar_auxiliary"].items()}
    pair.radar.teacher_projection_state = {key: value.clone() for key, value in global_states["radar_projection"].items()}
    pair.optical.encoder_state = {key: value.clone() for key, value in global_states["optical_encoder"].items()}
    pair.optical.auxiliary_state = {key: value.clone() for key, value in global_states["optical_auxiliary"].items()}
    pair.optical.projection_state = {key: value.clone() for key, value in global_states["optical_projection"].items()}
    pair.optical.teacher_encoder_state = {key: value.clone() for key, value in global_states["optical_encoder"].items()}
    pair.optical.teacher_auxiliary_state = {key: value.clone() for key, value in global_states["optical_auxiliary"].items()}
    pair.optical.teacher_projection_state = {key: value.clone() for key, value in global_states["optical_projection"].items()}
    pair.radar.downloaded_global_version = global_version
    pair.optical.downloaded_global_version = global_version
