import argparse
import math
from pathlib import Path

import gradio as gr

from src.predict import load_model, make_overlay, predict_image


def main():
    parser = argparse.ArgumentParser(description="Local surface defect segmentation demo")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be finite and in [0, 1].")

    model, config = load_model(args.checkpoint, args.device)

    def segment(image):
        if image is None:
            return None, None, None

        original = image.convert("RGB")
        mask = predict_image(model, original, config["image_size"], args.threshold, args.device)
        return original, mask, make_overlay(original, mask)

    demo = gr.Interface(
        fn=segment,
        inputs=gr.Image(type="pil", label="Upload surface image"),
        outputs=[
            gr.Image(type="pil", label="Original"),
            gr.Image(type="pil", label="Predicted mask"),
            gr.Image(type="pil", label="Overlay"),
        ],
        title="Surface Defect Segmentation",
        description="Local demo using a trained checkpoint. Red areas show predicted defects.",
    )
    demo.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
