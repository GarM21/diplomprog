from pathlib import Path

import cv2
import numpy as np


class ImageLoader:
    @staticmethod
    def load(file_path: str | Path) -> np.ndarray:
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Файл не найден: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Не удалось прочитать изображение. Проверьте формат файла.")

        return image
