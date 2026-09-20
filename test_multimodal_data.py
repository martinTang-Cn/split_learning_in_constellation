"""Focused tests for paired data routing and lazy satellite batches."""

import unittest
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from multimodal_evaluation import evaluate_global
from multimodal_data import (
    PairedBatchSequence, RealPairedDatasetAdapter, build_dataset_bundle,
    partition_dataset_indices,
)


class SourceDataset(Dataset):
    def __init__(self, length=11, channels=4):
        self.length = length
        self.channels = channels
        self.read_indices = []

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.read_indices.append(index)
        optical = torch.full((self.channels, 8, 8), float(index + 10))
        radar = torch.full((1, 8, 8), float(index + 100))
        mask = torch.full((8, 8), index % 8, dtype=torch.long)
        return optical, radar, mask


class PairedDataTests(unittest.TestCase):
    def test_adapters_route_both_modalities_with_same_sample_id(self):
        for name in ("whu_opt_sar", "houston2013"):
            with self.subTest(name=name):
                source = SourceDataset()
                radar, optical, mask, sample_id = RealPairedDatasetAdapter(source, name)[3]
                self.assertEqual(sample_id, 3)
                self.assertEqual(float(radar[0, 0, 0]), 103.0)
                self.assertEqual(float(optical[0, 0, 0]), 13.0)
                self.assertEqual(int(mask[0, 0]), 3)

    def test_partition_is_even_complete_and_deterministic(self):
        names = ["plane_1", "plane_2", "plane_3", "plane_4"]
        first = partition_dataset_indices(11, names, 42)
        self.assertEqual(first, partition_dataset_indices(11, names, 42))
        self.assertEqual(sorted(index for values in first.values() for index in values), list(range(11)))
        self.assertEqual(sorted(map(len, first.values())), [2, 3, 3, 3])
        self.assertNotEqual(first, partition_dataset_indices(11, names, 7))

    def test_batches_load_only_requested_samples_and_preserve_pairing(self):
        source = SourceDataset()
        dataset = RealPairedDatasetAdapter(source, "whu_opt_sar")
        batches = PairedBatchSequence(dataset, [2, 4, 6], 2, 2, 12)
        self.assertEqual(source.read_indices, [])
        self.assertEqual(len(batches), 4)
        epoch, number, ids, radar, optical, masks = batches[0]
        self.assertEqual((epoch, number), (1, 0))
        self.assertEqual(source.read_indices, ids.tolist())
        self.assertTrue(torch.equal(radar[:, 0, 0, 0] - optical[:, 0, 0, 0], torch.full((2,), 90.0)))
        self.assertTrue(torch.equal(masks[:, 0, 0], ids % 8))

    def test_houston_unlabeled_pixels_do_not_affect_metrics(self):
        class Encoder(nn.Module):
            def forward(self, images, attention_bias, mask_info):
                return images[:, :1]

        class Fusion(nn.Module):
            def forward(self, radar, optical, attention_bias):
                return radar

        class Head(nn.Module):
            def forward(self, tokens, image_size):
                return torch.zeros(tokens.shape[0], 2, image_size, image_size)

        class Validation(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, index):
                label = torch.full((8, 8), -1, dtype=torch.long)
                label[0, 0] = 0
                return torch.zeros(1, 8, 8), torch.zeros(3, 8, 8), label, index

        radar, optical, fusion, head = Encoder(), Encoder(), Fusion(), Head()
        config = {
            "segmentation_training": {"batch_size": 2},
            "dataset_metadata": {"ignore_index": -1, "num_classes": 2},
        }
        states = {
            "radar_encoder": radar.state_dict(),
            "optical_encoder": optical.state_dict(),
        }
        accuracy, miou, confusion = evaluate_global(
            Validation(), radar, optical, fusion, head, None, states, config,
            torch.device("cpu"),
        )
        self.assertEqual((accuracy, miou), (1.0, 1.0))
        self.assertEqual(confusion, [[1, 0], [0, 0]])

    def test_real_dataset_constructors_and_metadata(self):
        class WHU(SourceDataset):
            def __init__(self, split, **kwargs):
                super().__init__(length=9 if split == "train" else 2)

        class Houston(SourceDataset):
            def __init__(self, split, **kwargs):
                super().__init__(length=9 if split == "train" else 2, channels=144)

        fake_module = types.ModuleType("datasets")
        fake_module.WHUOptSarPatchDataset = WHU
        fake_module.Houston2013PatchDataset = Houston
        base = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
        base["croma"]["patch_size"] = 4
        base["croma"]["num_patches"] = 4
        with tempfile.TemporaryDirectory() as temporary_root:
            with patch.dict(sys.modules, {"datasets": fake_module}):
                for name, optical_channels, classes, ignore in (
                    ("whu_opt_sar", 4, 8, -100),
                    ("houston2013", 144, 15, -1),
                ):
                    with self.subTest(name=name):
                        base["dataset"] = {"name": name, "root_dir": temporary_root}
                        bundle = build_dataset_bundle(base, Path(temporary_root), 4)
                        self.assertEqual((len(bundle.train), len(bundle.validation)), (9, 2))
                        self.assertEqual(bundle.metadata.ignore_index, ignore)
                        self.assertEqual(bundle.metadata.num_classes, classes)
                        self.assertEqual(bundle.metadata.radar_channels, 1)
                        self.assertEqual(bundle.metadata.optical_channels, optical_channels)
                        self.assertEqual(bundle.train[2][3], 2)


if __name__ == "__main__":
    unittest.main()
