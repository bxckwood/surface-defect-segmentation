"""Функции потерь для бинарных масок и логитов одного канала."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _prepare_targets(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Expected logits shaped [N, 1, H, W].")
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    if targets.shape != logits.shape:
        raise ValueError(f"Targets shape {tuple(targets.shape)} differs from logits {tuple(logits.shape)}.")

    return targets.to(device=logits.device, dtype=logits.dtype)


class BinaryBCELoss(nn.Module):
    """Пиксельная бинарная кросс-энтропия по логитам."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _prepare_targets(logits, targets)
        return F.binary_cross_entropy_with_logits(logits, targets)


class DiceLoss(nn.Module):
    """Мягкая Dice-потеря только для изображений с дефектом."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive.")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _prepare_targets(logits, targets)
        probabilities = torch.sigmoid(logits).flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        # Пустые маски исключены из Dice; в BCEDiceLoss их учитывает BCE.
        positive = targets.sum(dim=1) > 0
        if not positive.any():
            return logits.sum() * 0.0

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice[positive].mean()


class BCEDiceLoss(nn.Module):
    """Сумма BCE для всех снимков и Dice для снимков с дефектом."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = BinaryBCELoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


def get_loss(name: str) -> nn.Module:
    """Выбирает BCE или BCE + Dice по имени из конфигурации."""

    key = name.strip().lower().replace("-", "_")
    if key == "bce":
        return BinaryBCELoss()
    if key in {"bce_dice", "bce+dice"}:
        return BCEDiceLoss()
    raise ValueError(f"Unknown loss {name!r}; choose 'bce' or 'bce_dice'.")
