"""Пиксельные и поснимковые метрики сегментации дефектов."""

from __future__ import annotations

import math

import torch


class SegmentationMetrics:
    """Накапливает пиксельные и поснимковые метрики по пакетам."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.image_count = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.clean_image_count = 0
        self.clean_image_false_alarm_count = 0
        self.defect_image_count = 0
        self.missed_defect_count = 0

        self.positive_image_iou_sum = 0.0
        self.positive_image_dice_sum = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> None:
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0, 1].")

        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if logits.ndim != 4 or logits.shape[1] != 1 or targets.shape != logits.shape:
            raise ValueError("Expected matching logits and targets shaped [N, 1, H, W].")

        truth = targets.to(device=logits.device) > 0.5
        predicted = torch.sigmoid(logits) >= threshold
        tp_per_image = (predicted & truth).flatten(start_dim=1).sum(dim=1)
        fp_per_image = (predicted & ~truth).flatten(start_dim=1).sum(dim=1)
        fn_per_image = (~predicted & truth).flatten(start_dim=1).sum(dim=1)

        defect_images = (tp_per_image + fn_per_image) > 0
        clean_images = ~defect_images
        predicted_images = (tp_per_image + fp_per_image) > 0

        # TP, FP и FN суммируются по всему набору для micro foreground IoU/Dice.
        self.image_count += logits.shape[0]
        self.tp += int(tp_per_image.sum().item())
        self.fp += int(fp_per_image.sum().item())
        self.fn += int(fn_per_image.sum().item())

        self.clean_image_count += int(clean_images.sum().item())
        self.clean_image_false_alarm_count += int((clean_images & predicted_images).sum().item())
        self.defect_image_count += int(defect_images.sum().item())
        self.missed_defect_count += int((defect_images & (tp_per_image == 0)).sum().item())

        # Среднее по дефектным снимкам считаем отдельно от общей пиксельной метрики.
        if defect_images.any():
            union = tp_per_image + fp_per_image + fn_per_image
            dice_denominator = 2 * tp_per_image + fp_per_image + fn_per_image
            self.positive_image_iou_sum += float((tp_per_image[defect_images] / union[defect_images]).sum().item())
            self.positive_image_dice_sum += float(
                ((2 * tp_per_image[defect_images]) / dice_denominator[defect_images]).sum().item()
            )

    def compute(self) -> dict[str, int | float | None]:
        iou_denominator = self.tp + self.fp + self.fn
        dice_denominator = 2 * self.tp + self.fp + self.fn

        clean_false_alarm_rate = (
            self.clean_image_false_alarm_count / self.clean_image_count
            if self.clean_image_count
            else None
        )
        missed_defect_rate = (
            self.missed_defect_count / self.defect_image_count
            if self.defect_image_count
            else None
        )

        return {
            "image_count": self.image_count,
            "pixel_tp": self.tp,
            "pixel_fp": self.fp,
            "pixel_fn": self.fn,
            "foreground_iou": self.tp / iou_denominator if iou_denominator else None,
            "foreground_dice": 2 * self.tp / dice_denominator if dice_denominator else None,
            "positive_image_mean_iou": (
                self.positive_image_iou_sum / self.defect_image_count if self.defect_image_count else None
            ),
            "positive_image_mean_dice": (
                self.positive_image_dice_sum / self.defect_image_count if self.defect_image_count else None
            ),
            "clean_image_count": self.clean_image_count,
            "clean_image_false_alarm_count": self.clean_image_false_alarm_count,
            "clean_image_false_alarm_rate": clean_false_alarm_rate,
            "clean_image_specificity": 1 - clean_false_alarm_rate if clean_false_alarm_rate is not None else None,
            "defect_image_count": self.defect_image_count,
            "missed_defect_count": self.missed_defect_count,
            "missed_defect_rate": missed_defect_rate,
            "defect_image_recall": 1 - missed_defect_rate if missed_defect_rate is not None else None,
        }
