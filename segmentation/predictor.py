import torch
from pathlib import Path
import argparse
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet import UNet


class GreenAreaPredictor:
    """Loads a trained U-Net and predicts green-area masks."""

    def __init__(
        self,
        weights_path: str | Path,
        image_size: int = 256,
        threshold: float = 0.5,
        device: str = "auto",
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        features: int = 64,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.image_size = image_size
        self.threshold = threshold
        self.device = self._resolve_device(device)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.model = UNet(in_channels=3, out_channels=1, features=features).to(self.device)
        self._load_model()

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> tuple[np.ndarray, float]:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        tensor = self._preprocess(image).to(self.device)

        self.model.eval()
        logits = self.model(tensor)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask = (probability >= self.threshold).astype(np.uint8) * 255

        result_mask = Image.fromarray(mask, mode="L")
        result_mask = result_mask.resize(original_size, Image.Resampling.NEAREST)
        result_array = np.asarray(result_mask, dtype=np.uint8)
        green_percentage = float((result_array > 0).mean() * 100.0)

        return result_array, green_percentage

    def save_mask(self, mask: np.ndarray, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(output_path)

    def predict_and_save(
        self,
        image_path: str | Path,
        output_path: str | Path,
    ) -> float:
        mask, green_percentage = self.predict(image_path)
        self.save_mask(mask, output_path)
        return green_percentage

    def _load_model(self) -> None:
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - self.mean) / self.std
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()
        return tensor

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict green-area mask with U-Net.")
    parser.add_argument("--weights", required=True, help="Path to trained model weights.")
    parser.add_argument("--image", required=True, help="Path to source image.")
    parser.add_argument("--output", required=True, help="Path for saving predicted mask.")
    parser.add_argument("--image-size", type=int, default=256, help="Model input size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold.")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda.")
    parser.add_argument("--features", type=int, default=64, help="Base U-Net features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = GreenAreaPredictor(
        weights_path=args.weights,
        image_size=args.image_size,
        threshold=args.threshold,
        device=args.device,
        features=args.features,
    )
    green_percentage = predictor.predict_and_save(args.image, args.output)
    print(f"Green area: {green_percentage:.2f}%")


if __name__ == "__main__":
    main()
