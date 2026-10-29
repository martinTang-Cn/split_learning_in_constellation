"""Read-only, per-contact diagnostics for satellite projection alignment."""

from contextlib import contextmanager
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F

from multimodal_sfl import clone_state


@contextmanager
def isolated_rng(device):
    """Diagnostic sampling/forwards must not change the training RNG stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class ProjectionDiagnostics:
    """Sample one paired batch per window; compare local states before/after.

    Samples come from that pair's training partition, not a held-out test set.
    The batch does not advance the training cursor. If the partition is smaller
    than batch_size, use every available sample once (without replacement).
    """

    def __init__(self, pairs, batch_size, seed, path):
        self.batch_size = int(batch_size)
        self.indices, self.generators = {}, {}
        for pair_id, pair in pairs.items():
            first_epoch = pair.batches.batch_specs[0][0]
            indices = []
            for epoch, _, batch_indices in pair.batches.batch_specs:
                if epoch != first_epoch:
                    break
                indices.extend(batch_indices)
            self.indices[pair_id] = sorted(set(indices))
            self.generators[pair_id] = torch.Generator().manual_seed(int(seed) + 1000003 + pair.plane)
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def sample(self, pair, device):
        indices = self.indices[pair.pair_id]
        order = torch.randperm(len(indices), generator=self.generators[pair.pair_id])
        chosen = [indices[i] for i in order[:self.batch_size].tolist()]
        with isolated_rng(device):
            samples = [pair.batches.dataset[i] for i in chosen]
            radar, optical, _, sample_ids = zip(*samples)
            return torch.stack(radar), torch.stack(optical), list(sample_ids)

    @torch.no_grad()
    def measure(self, pair, batch, radar_worker, optical_worker, cross_encoder,
                radar_projection, optical_projection, attention_bias, device):
        """Evaluate local projection vs fusion, restoring shared worker states.

        Normalize each (sample, token) vector along its feature dimension.
        MSE averages squared differences across B,N,C; cosine averages over B,N.
        Both passes in a window use the same live cross encoder, with no ground
        update in between. Teacher features are recomputed from current local
        encoders, rather than holding the teacher output tensor constant.
        """
        workers = (radar_worker, optical_worker, radar_projection, optical_projection)
        states = [clone_state(module) for module in workers]
        modes = {module: module.training for root in (*workers, cross_encoder)
                 for module in root.modules()}
        local_states = (pair.radar.encoder_state, pair.optical.encoder_state,
                        pair.radar.projection_state, pair.optical.projection_state)
        try:
            with isolated_rng(device):
                for module, state in zip(workers, local_states):
                    module.load_state_dict(state)
                    module.eval()
                cross_encoder.eval()
                radar = radar_worker(batch[0].to(device), attention_bias, mask_info=None)
                optical = optical_worker(batch[1].to(device), attention_bias, mask_info=None)
                teacher = F.normalize(cross_encoder(radar, optical, attention_bias).float(), dim=-1, eps=1e-8)
                metrics = {}
                for name, projected in (("radar", radar_projection(radar)),
                                        ("optical", optical_projection(optical))):
                    projected = F.normalize(projected.float(), dim=-1, eps=1e-8)
                    metrics[f"{name}_normalized_mse"] = F.mse_loss(projected, teacher).item()
                    metrics[f"{name}_cosine_similarity"] = F.cosine_similarity(
                        projected, teacher, dim=-1, eps=1e-8
                    ).mean().item()
                return metrics
        finally:
            for module, state in zip(workers, states):
                module.load_state_dict(state)
            for module, mode in modes.items():
                module.training = mode

    def append(self, metadata, before, after):
        row = dict(metadata)
        for name in before:
            row[f"{name}_before"] = before[name]
            row[f"{name}_after"] = after[name]
            row[f"{name}_delta"] = after[name] - before[name]
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
