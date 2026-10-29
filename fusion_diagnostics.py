"""Read-only probes of offline encoder updates with a fixed ground model."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch

from multimodal_sfl import clone_state


FUSION_PROBE_FIELDS = [
    "fusion_probe_status", "fusion_probe_sample_ids", "fusion_probe_sample_count",
    "fusion_probe_sampling_with_replacement", "fusion_probe_valid_pixels",
    "fusion_probe_local_steps", "fusion_probe_ground_global_version",
    "fusion_probe_ground_server_updates", "fusion_probe_loss_before",
    "fusion_probe_loss_after", "fusion_probe_loss_delta",
]


@contextmanager
def _isolated_rng(seed, device):
    """Restore training RNGs even if dataset transforms consume randomness."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class FixedGroundFusionProbe:
    """Compare pair encoders before/after offline training, without optimization.

    The caller must wrap ONLY the offline training call in ``measure``. This
    keeps cross_encoder and ground_head weights identical in the two passes.
    Sampling uses the pair's training partition, not the validation dataset.
    """

    def __init__(self, pairs, radar_worker, optical_worker, cross_encoder,
                 ground_head, attention_bias, criterion, image_size, device,
                 batch_size, seed):
        self.radar_worker, self.optical_worker = radar_worker, optical_worker
        self.cross_encoder, self.ground_head = cross_encoder, ground_head
        self.attention_bias, self.criterion = attention_bias, criterion
        self.image_size, self.device = image_size, device
        self.batch_size, self.seed = int(batch_size), int(seed)
        self.indices = {
            pair_id: sorted({index for _, _, indices in pair.batches.batch_specs for index in indices})
            for pair_id, pair in pairs.items()
        }

    def _sample(self, pair, seed):
        indices = self.indices[pair.pair_id]
        if not indices:
            raise ValueError(f"No probe samples available for {pair.pair_id}")
        generator = torch.Generator().manual_seed(seed)
        replacement = len(indices) < self.batch_size
        if replacement:
            positions = torch.randint(len(indices), (self.batch_size,), generator=generator).tolist()
        else:
            positions = torch.randperm(len(indices), generator=generator)[:self.batch_size].tolist()
        # Fetch once, so any stochastic transforms are identical for both passes.
        with _isolated_rng(seed, self.device):
            samples = [pair.batches.dataset[indices[position]] for position in positions]
        radar, optical, labels, sample_ids = zip(*samples)
        return (torch.stack(radar), torch.stack(optical), torch.stack(labels)), sample_ids, replacement

    @torch.no_grad()
    def _loss(self, pair, batch, seed):
        # Single-GPU workers may currently contain a different pair's weights.
        # Restore them afterward, including buffers, and preserve all mode flags.
        radar_state = clone_state(self.radar_worker)
        optical_state = clone_state(self.optical_worker)
        roots = (self.radar_worker, self.optical_worker, self.cross_encoder, self.ground_head)
        modes = [(module, module.training) for root in roots for module in root.modules()]
        try:
            self.radar_worker.load_state_dict(pair.radar.encoder_state)
            self.optical_worker.load_state_dict(pair.optical.encoder_state)
            for root in roots:
                root.eval()
            with _isolated_rng(seed, self.device):
                radar, optical, labels = (tensor.to(self.device) for tensor in batch)
                radar_tokens = self.radar_worker(radar, self.attention_bias, mask_info=None)
                optical_tokens = self.optical_worker(optical, self.attention_bias, mask_info=None)
                fused = self.cross_encoder(radar_tokens, optical_tokens, self.attention_bias)
                return float(self.criterion(self.ground_head(fused, self.image_size), labels).item())
        finally:
            self.radar_worker.load_state_dict(radar_state)
            self.optical_worker.load_state_dict(optical_state)
            for module, training in modes:
                module.training = training

    @contextmanager
    def measure(self, pair, contact_id, global_version, server_updates):
        # Stable across freeze/unfreeze runs and independent of Python hash randomization.
        digest = hashlib.sha256(f"{self.seed}:{pair.pair_id}:{contact_id}:fusion-probe".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        batch, sample_ids, replacement = self._sample(pair, seed)
        valid_pixels = int((batch[2] != self.criterion.ignore_index).sum().item())
        initial_local_version = pair.radar.local_version
        metrics = dict.fromkeys(FUSION_PROBE_FIELDS, "")
        metrics.update({
            "fusion_probe_status": "completed" if valid_pixels else "no_valid_labels",
            "fusion_probe_sample_ids": ";".join(str(int(index)) for index in sample_ids),
            "fusion_probe_sample_count": len(sample_ids),
            "fusion_probe_sampling_with_replacement": int(replacement),
            "fusion_probe_valid_pixels": valid_pixels,
            "fusion_probe_ground_global_version": global_version,
            "fusion_probe_ground_server_updates": server_updates,
        })
        before = self._loss(pair, batch, seed) if valid_pixels else None
        yield metrics
        metrics["fusion_probe_local_steps"] = pair.radar.local_version - initial_local_version
        if valid_pixels:
            after = self._loss(pair, batch, seed)
            metrics.update({
                "fusion_probe_loss_before": before,
                "fusion_probe_loss_after": after,
                "fusion_probe_loss_delta": after - before,
            })
