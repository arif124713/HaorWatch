"""predict_probs must pair each prediction with exactly one correct chip id."""

import numpy as np
import torch

from app.ml.train import predict_probs


class _IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1, :, :]  # 1 logit per pixel


class _FakeLoader:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


def _batch(ids, size=4):
    n = len(ids)
    images = torch.zeros((n, 2, size, size), dtype=torch.float32)
    floods = torch.zeros((n, 1, size, size), dtype=torch.float32)
    valids = torch.ones((n, 1, size, size), dtype=torch.bool)
    return images, floods, valids, tuple(ids)


def test_chip_ids_align_one_per_prediction():
    ids = ["alpha", "beta", "gamma"]
    loader = _FakeLoader([_batch(ids)])
    probs, floods, valids, chip_ids = predict_probs(_IdentityModel(), loader, torch.device("cpu"))
    assert len(probs) == len(chip_ids) == 3
    assert chip_ids == ids


def test_chip_ids_align_across_multiple_batches_different_sizes():
    loader = _FakeLoader([
        _batch(["a", "b"]),
        _batch(["c", "d", "e"]),
        _batch(["f"]),
    ])
    probs, floods, valids, chip_ids = predict_probs(_IdentityModel(), loader, torch.device("cpu"))
    assert len(probs) == len(chip_ids) == 6
    assert chip_ids == ["a", "b", "c", "d", "e", "f"]
    # No duplicated ids (the previous bug multiplied ids by batch size).
    assert len(set(chip_ids)) == len(chip_ids)


def test_ids_not_duplicated_for_batch_of_four():
    loader = _FakeLoader([_batch(["w", "x", "y", "z"])])
    _, _, _, chip_ids = predict_probs(_IdentityModel(), loader, torch.device("cpu"))
    assert chip_ids == ["w", "x", "y", "z"]
