import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset import SurfaceDefectDataset
from src.losses import get_loss
from src.metrics import SegmentationMetrics
from src.predict import load_model, make_panel, predict_image


ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device, thresholds):
    scores = {threshold: SegmentationMetrics() for threshold in thresholds}
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)

        total_loss += loss_fn(logits, masks).item() * images.size(0)
        for threshold in thresholds:
            scores[threshold].update(logits, masks, threshold)

    return total_loss / len(loader.dataset), {threshold: metric.compute() for threshold, metric in scores.items()}


@torch.inference_mode()
def save_examples(model, loader, data_root, image_size, threshold, device, output_dir):
    """Сохраняет панели лучших, плохих, пропущенных и ложных случаев."""
    records = []

    for batch in loader:
        logits = model(batch["image"].to(device))
        prediction = torch.sigmoid(logits).cpu() >= threshold
        truth = batch["mask"] > 0.5

        for index, image_path in enumerate(batch["image_path"]):
            predicted = prediction[index]
            target = truth[index]
            tp = int((predicted & target).sum())
            fp = int((predicted & ~target).sum())
            fn = int((~predicted & target).sum())
            iou = tp / (tp + fp + fn) if tp + fp + fn else None

            records.append({"image": image_path, "mask": batch["mask_path"][index],
                            "tp": tp, "fp": fp, "fn": fn, "iou": iou})

    groups = {
        "good": sorted((r for r in records if r["tp"] > 0), key=lambda r: r["iou"], reverse=True)[:3],
        "bad": sorted((r for r in records if r["tp"] > 0), key=lambda r: r["iou"])[:3],
        "missed": sorted((r for r in records if r["fn"] > 0 and r["tp"] == 0),
                         key=lambda r: r["fn"], reverse=True)[:3],
        "false_positive": sorted((r for r in records if r["fn"] == 0 and r["tp"] == 0 and r["fp"] > 0),
                                 key=lambda r: r["fp"], reverse=True)[:3],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    titles = {
        "good": "Удачные",
        "bad": "Плохо выделенные",
        "missed": "Пропущенные дефекты",
        "false_positive": "Ложные срабатывания",
    }
    lines = ["# Примеры предсказаний и ошибок", "", f"Порог: {threshold}", "",
             f"TP, FP, FN и IoU рассчитаны на сетке модели {image_size[0]}×{image_size[1]} "
             "(высота×ширина). "
             "Для панелей предсказанная маска возвращена к исходному размеру через NEAREST, "
             "разметка показана в исходном размере.", "",
             "Примеры отобраны автоматически по пересечению масок и числу ошибок.", ""]

    for label, selected in groups.items():
        lines.extend([f"## {titles[label]}", ""])
        if not selected:
            lines.extend(["В этой категории примеров нет.", ""])
        for index, record in enumerate(selected, 1):
            with Image.open(data_root / record["image"]) as file:
                image = file.convert("RGB")
            with Image.open(data_root / record["mask"]) as file:
                truth = file.convert("L")

            prediction = predict_image(model, image, image_size, threshold, device)
            filename = f"{label}_{index}.png"
            make_panel(image, prediction, truth).save(output_dir / filename)
            lines.append(f"- [{record['image']}]({filename}): IoU={record['iou']}, "
                         f"TP={record['tp']}, FP={record['fp']}, FN={record['fn']}")
        lines.append("")

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--search-threshold", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--examples-dir", type=Path)
    args = parser.parse_args()

    if args.search_threshold and args.split != "val":
        parser.error("Threshold search is allowed only on validation data")

    model, config = load_model(args.checkpoint, args.device)
    data_root = ROOT / config["data_root"]
    split_csv = ROOT / config["split_dir"] / f"{args.split}.csv"
    dataset = SurfaceDefectDataset(data_root, split_csv, config["image_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    # Подбор по test смешал бы настройку порога с итоговой оценкой.
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7] if args.search_threshold else [args.threshold]
    loss, scores = evaluate(model, loader, get_loss(config["loss"]), args.device, thresholds)
    # При равном IoU берём порог, ближайший к исходному 0.5.
    threshold = max(thresholds, key=lambda value: (scores[value]["foreground_iou"], -abs(value - 0.5)))
    result = {
        "checkpoint": str(args.checkpoint), "model": config["model"], "loss_name": config["loss"],
        "split": args.split, "loss": loss, "threshold": threshold,
        "metrics": scores[threshold],
    }

    if args.search_threshold:
        result["validation_thresholds"] = {str(key): value["foreground_iou"] for key, value in scores.items()}

    output = args.output or ROOT / config["output_dir"] / f"{args.split}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    if args.examples_dir:
        save_examples(model, loader, data_root, config["image_size"], threshold, args.device, args.examples_dir)


if __name__ == "__main__":
    main()
