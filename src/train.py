import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.dataset import SurfaceDefectDataset
from src.losses import get_loss
from src.metrics import SegmentationMetrics
from src.model import build_model


ROOT = Path(__file__).resolve().parents[1]


def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch, epochs):
    model.train()
    total_loss = 0.0
    seen = 0
    progress = tqdm(loader, desc=f"Train {epoch}/{epochs}", unit="batch", leave=False, mininterval=1)

    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
        progress.set_postfix(loss=f"{total_loss / seen:.4f}", refresh=False)

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def validate(model, loader, loss_fn, device, epoch, epochs):
    model.eval()
    total_loss = 0.0
    seen = 0
    metrics = SegmentationMetrics()
    progress = tqdm(loader, desc=f"Val   {epoch}/{epochs}", unit="batch", leave=False, mininterval=1)

    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)

        total_loss += loss_fn(logits, masks).item() * images.size(0)
        seen += images.size(0)
        metrics.update(logits, masks, threshold=0.5)
        progress.set_postfix(loss=f"{total_loss / seen:.4f}", refresh=False)

    return total_loss / len(loader.dataset), metrics.compute()


def main():
    parser = argparse.ArgumentParser(description="Train one surface defect segmentation experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt, or best.pt for older runs")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device_name = config.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    data_root = ROOT / config["data_root"]
    split_dir = ROOT / config["split_dir"]
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = SurfaceDefectDataset(data_root, split_dir / "train.csv", config["image_size"], train=True)
    val_data = SurfaceDefectDataset(data_root, split_dir / "val.csv", config["image_size"])
    loader_kwargs = {"batch_size": config["batch_size"], "num_workers": 0, "pin_memory": device.type == "cuda"}

    positive_fraction = config.get("positive_sample_fraction")
    sampling_generator = None
    if positive_fraction is None:
        train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    else:
        if not isinstance(positive_fraction, (int, float)) or not 0 < positive_fraction < 1:
            parser.error("positive_sample_fraction must be between 0 and 1")

        labels = [row["has_defect"] for row in train_data.rows]
        if set(labels) != {"0", "1"}:
            parser.error("Training split must contain both 0 and 1 in has_defect")

        positive_count = labels.count("1")
        negative_count = len(labels) - positive_count
        # Долю каждой группы распределяем поровну между её снимками.
        weights = [positive_fraction / positive_count if label == "1" else
                   (1 - positive_fraction) / negative_count for label in labels]
        sampling_generator = torch.Generator()
        sampler = WeightedRandomSampler(weights, num_samples=len(train_data), replacement=True,
                                        generator=sampling_generator)
        train_loader = DataLoader(train_data, sampler=sampler, **loader_kwargs)

    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    model = build_model(config["model"], pretrained=config.get("pretrained", False) and not args.resume).to(device)
    loss_fn = get_loss(config["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best_iou = -1.0
    history_path = output_dir / "history.csv"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    fields = ["epoch", "train_loss", "val_loss", "val_iou", "val_dice"]
    start_epoch = 1

    if args.resume:
        # Берём checkpoint с наибольшей эпохой и сверяем конфиг, чтобы не смешать эксперименты.
        checkpoints = [(path, torch.load(path, map_location=device, weights_only=True))
                       for path in (last_path, best_path) if path.exists()]
        if not checkpoints:
            parser.error(f"No checkpoint in {output_dir}")
        if not best_path.exists():
            parser.error(f"Missing best.pt in {output_dir}")
        keys = ("model", "loss", "pretrained", "seed", "data_root", "split_dir",
                "image_size", "batch_size", "learning_rate", "weight_decay", "positive_sample_fraction")
        for path, saved in checkpoints:
            if any(saved["config"].get(key) != config.get(key) for key in keys):
                parser.error(f"Configuration differs from {path}; only epochs and device may change")
        checkpoint_path, checkpoint = max(checkpoints, key=lambda item: item[1]["epoch"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        saved_total_epochs = checkpoint["config"]["epochs"]
        best_iou = max(saved.get("best_val_iou", saved["val_iou"]) for _, saved in checkpoints)
        del checkpoints, checkpoint, saved

        if not history_path.exists():
            parser.error(f"Missing history.csv in {output_dir}")
        with history_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        if checkpoint_path == best_path and rows and int(rows[-1]["epoch"]) >= saved_total_epochs:
            parser.error("This completed older run has no last.pt; its final epoch cannot be restored")
        kept_rows = [row for row in rows if int(row["epoch"]) < start_epoch]
        if [int(row["epoch"]) for row in kept_rows] != list(range(1, start_epoch)):
            parser.error("history.csv does not match the checkpoint epochs")

        if start_epoch > config["epochs"]:
            print(f"Checkpoint is already at epoch {start_epoch - 1}; increase epochs in the config to continue")
            return

        # Если история зашла дальше checkpoint, сохраняем её копию перед обрезкой.
        if len(kept_rows) != len(rows):
            backup_path = output_dir / "history_before_resume.csv"
            shutil.copy2(history_path, backup_path)
            with history_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(kept_rows)
            print(f"History trimmed to epoch {start_epoch - 1}; original saved in {backup_path}")
        print(f"Resuming from {checkpoint_path}, epoch {start_epoch}/{config['epochs']}")
    elif history_path.exists() or last_path.exists() or best_path.exists():
        parser.error(f"Run files already exist in {output_dir}; use --resume or another output_dir")

    with history_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not args.resume:
            writer.writeheader()
        for epoch in range(start_epoch, config["epochs"] + 1):
            # Привязываем выбор индексов к эпохе для --resume.
            if sampling_generator is not None:
                sampling_generator.manual_seed(config["seed"] + epoch)

            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, config["epochs"])
            val_loss, val_metrics = validate(model, val_loader, loss_fn, device, epoch, config["epochs"])
            val_iou = val_metrics["foreground_iou"]
            val_dice = val_metrics["foreground_dice"]

            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                             "val_iou": val_iou, "val_dice": val_dice})
            file.flush()

            improved = val_iou > best_iou
            best_iou = max(best_iou, val_iou)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_iou": val_iou,
                "best_val_iou": best_iou,
                "config": config,
            }
            if improved:
                torch.save(checkpoint, best_path)
            torch.save(checkpoint, last_path)
            print(f"Epoch {epoch:02d}: train loss={train_loss:.4f}, val loss={val_loss:.4f}, "
                  f"val IoU={val_iou:.4f}, val Dice={val_dice:.4f}")


if __name__ == "__main__":
    main()
