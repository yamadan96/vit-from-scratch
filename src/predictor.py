"""
Inference module for Vision Transformer CIFAR-10 classifier.

Provides a singleton Predictor class that loads a trained checkpoint
and performs inference on PIL images. Designed for integration with
the Gradio web application.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import ViTConfig, VisionTransformer

logger = logging.getLogger(__name__)

# CIFAR-10 class names (index-aligned with torchvision.datasets.CIFAR10)
CIFAR10_CLASSES: list[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# CIFAR-10 channel-wise statistics (must match training normalization)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class PredictionResult:
    """Container for a single classification prediction.

    Attributes:
        class_id: Predicted class index (0-9)
        class_name: Human-readable class name
        confidence: Softmax probability of the predicted class
        top3: Top-3 predictions as (class_name, probability) pairs
    """

    class_id: int
    class_name: str
    confidence: float
    top3: list[tuple[str, float]]


class Predictor:
    """Singleton inference wrapper for the Vision Transformer.

    Uses the singleton pattern to ensure the model is loaded only once,
    even when called from multiple Gradio prediction requests.

    Usage:
        predictor = Predictor()
        predictor.initialize(Path("checkpoints/best_model.pth"))
        result = predictor.predict(pil_image)
    """

    _instance: "Predictor | None" = None
    _initialized: bool = False

    def __new__(cls) -> "Predictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(
        self, checkpoint_path: Path, device: str = "cuda"
    ) -> None:
        """Load model checkpoint and prepare for inference.

        Args:
            checkpoint_path: Path to the saved .pth checkpoint
            device: Target device ("cuda" or "cpu")

        Raises:
            FileNotFoundError: If checkpoint does not exist
            RuntimeError: If checkpoint format is invalid
        """
        if self._initialized:
            logger.info("Predictor already initialized, skipping")
            return

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        # Select device (fallback to CPU if CUDA unavailable)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA not available, falling back to CPU")

        self._device = torch.device(device)

        # Load checkpoint
        logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=self._device, weights_only=True
        )

        # Reconstruct config from checkpoint (or use default)
        config_dict = checkpoint.get("config")
        if config_dict is not None:
            config = ViTConfig(**config_dict)
        else:
            logger.warning(
                "No config found in checkpoint, using default ViTConfig"
            )
            config = ViTConfig()

        # Build model and load weights
        self._model = VisionTransformer(config).to(self._device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        # Inference preprocessing (no augmentation)
        self._transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

        self._initialized = True
        logger.info(
            "Predictor initialized: device=%s, val_acc=%.4f (epoch %d)",
            self._device,
            checkpoint.get("val_acc", -1.0),
            checkpoint.get("epoch", -1),
        )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> PredictionResult:
        """Run inference on a single PIL image.

        Args:
            image: Input PIL image (any size, will be resized to 32x32)

        Returns:
            PredictionResult with class prediction, confidence, and top-3

        Raises:
            RuntimeError: If Predictor has not been initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "Predictor not initialized. Call initialize() first."
            )

        # Convert to RGB if necessary (e.g., RGBA, grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess: resize, crop, normalize -> [1, 3, 32, 32]
        tensor = self._transform(image).unsqueeze(0).to(self._device)

        # Forward pass
        logits = self._model(tensor)  # [1, num_classes]
        probs = F.softmax(logits, dim=1).squeeze(0)  # [num_classes]

        # Top-3 predictions
        top3_probs, top3_indices = probs.topk(3)
        top3: list[tuple[str, float]] = [
            (CIFAR10_CLASSES[idx.item()], prob.item())
            for idx, prob in zip(top3_indices, top3_probs)
        ]

        # Best prediction
        best_idx = top3_indices[0].item()
        best_prob = top3_probs[0].item()

        return PredictionResult(
            class_id=best_idx,
            class_name=CIFAR10_CLASSES[best_idx],
            confidence=best_prob,
            top3=top3,
        )
