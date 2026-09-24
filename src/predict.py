import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.dataset import image_to_tensor
from src.model import build_model


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]

    model = build_model(config["model"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return model, config


@torch.inference_mode()
def predict_image(model, image, image_size, threshold=0.5, device="cpu"):
    original = image.convert("RGB")
    tensor = image_to_tensor(original, image_size).unsqueeze(0).to(device)
    probability = torch.sigmoid(model(tensor))[0, 0]

    # NEAREST сохраняет бинарную маску при возврате к размеру исходного снимка.
    mask = (probability >= threshold).to(torch.uint8).cpu().numpy() * 255
    return Image.fromarray(mask).resize(original.size, Image.Resampling.NEAREST)


def make_overlay(image, mask):
    original = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    marked = np.asarray(mask.convert("L")) > 0
    original[marked] = (0.5 * original[marked] + 0.5 * np.array([255, 35, 35])).astype(np.uint8)

    return Image.fromarray(original)


def make_panel(image, predicted_mask, true_mask=None):
    image = image.convert("RGB")
    predicted_mask = predicted_mask.convert("RGB")

    panels = [("Original", image)]
    if true_mask is not None:
        panels.append(("Ground truth", true_mask.convert("RGB")))
    panels.extend([("Prediction", predicted_mask), ("Overlay", make_overlay(image, predicted_mask))])

    width, height = image.size
    result = Image.new("RGB", (width * len(panels), height + 24), "white")
    draw = ImageDraw.Draw(result)

    for index, (title, panel) in enumerate(panels):
        result.paste(panel.resize(image.size), (index * width, 24))
        draw.text((index * width + 5, 5), title, fill="black")

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict a defect mask for one image")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint, args.device)
    with Image.open(args.image) as file:
        image = file.convert("RGB")
    mask = predict_image(model, image, config["image_size"], args.threshold, args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask.save(args.output_dir / "mask.png")
    make_overlay(image, mask).save(args.output_dir / "overlay.png")
    make_panel(image, mask).save(args.output_dir / "comparison.png")
    print(f"Saved mask and visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
