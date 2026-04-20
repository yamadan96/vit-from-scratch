"""
Gradio web application for ViT CIFAR-10 image classification.

Provides an interactive demo where users can upload images and see
real-time classification results from the trained Vision Transformer.

Usage:
    CHECKPOINT_DIR=checkpoints uv run python app.py

Environment variables:
    CHECKPOINT_DIR: Directory containing best_model.pth (default: checkpoints)
"""

import logging
import os
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from predictor import CIFAR10_CLASSES, Predictor, PredictionResult

logger = logging.getLogger(__name__)

# Checkpoint resolution
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"

# Global state
_predictor_error: str | None = None


def _initialize_predictor() -> None:
    """Attempt to initialize the Predictor singleton at startup."""
    global _predictor_error

    if not CHECKPOINT_PATH.exists():
        _predictor_error = (
            f"Checkpoint not found at: {CHECKPOINT_PATH}\n"
            "Please train the model first:\n"
            "  uv run python src/train.py --epochs 100\n"
            "Then set CHECKPOINT_DIR if needed."
        )
        logger.error(_predictor_error)
        return

    try:
        predictor = Predictor()
        predictor.initialize(CHECKPOINT_PATH, device="cuda")
        logger.info("Predictor ready")
    except Exception as e:
        _predictor_error = f"Failed to initialize model: {e}"
        logger.exception("Predictor initialization failed")


def classify_image(image: Image.Image | None) -> dict[str, float]:
    """Classify an uploaded image using the ViT model.

    Args:
        image: PIL image from Gradio input (None if no image uploaded)

    Returns:
        Dictionary mapping class names to probabilities (for gr.Label)
    """
    if image is None:
        return {}

    if _predictor_error is not None:
        # Return error as a special label so the user sees the message
        return {f"Error: {_predictor_error}": 1.0}

    try:
        predictor = Predictor()
        result: PredictionResult = predictor.predict(image)
        # Build label dict from top-3 predictions
        return {name: prob for name, prob in result.top3}
    except Exception as e:
        logger.exception("Prediction failed")
        return {f"Prediction error: {e}": 1.0}


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    with gr.Blocks(
        title="ViT CIFAR-10 Classifier",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # Vision Transformer -- CIFAR-10 Classifier

            ViT (Dosovitskiy et al., 2020) "An Image is Worth 16x16 Words"
            reproduction trained on CIFAR-10.

            Upload an image to classify it into one of 10 categories:
            airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
            """
        )

        if _predictor_error is not None:
            gr.Markdown(
                f"**Warning:** {_predictor_error}"
            )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Input Image",
                )
                classify_btn = gr.Button(
                    "Classify",
                    variant="primary",
                )

            with gr.Column():
                label_output = gr.Label(
                    num_top_classes=3,
                    label="Classification Result Top-3",
                )

        # Wire up events
        classify_btn.click(
            fn=classify_image,
            inputs=image_input,
            outputs=label_output,
        )
        image_input.change(
            fn=classify_image,
            inputs=image_input,
            outputs=label_output,
        )

        gr.Markdown(
            """
            ---
            **Model:** ViT-Small (6 layers, 8 heads, D=256)
            | **Dataset:** CIFAR-10 (32x32)
            | **Patch size:** 4x4 (64 patches)
            """
        )

    return app


# Initialize predictor at module load time
_initialize_predictor()


def main() -> None:
    """Launch the Gradio application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
