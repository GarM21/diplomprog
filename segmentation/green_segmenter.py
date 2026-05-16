import cv2
import numpy as np

from models.segmentation_result import SegmentationResult


class GreenSegmenter:
    """Basic HSV-based green area segmentation for satellite images."""

    def __init__(
        self,
        lower_hsv: tuple[int, int, int] = (35, 35, 35),
        upper_hsv: tuple[int, int, int] = (90, 255, 255),
        kernel_size: int = 5,
    ) -> None:
        self._lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self._upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self._kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    def segment(self, image: np.ndarray) -> SegmentationResult:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, self._lower_hsv, self._upper_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        green_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        green_percentage = (green_pixels / total_pixels) * 100 if total_pixels else 0.0

        return SegmentationResult(
            mask=mask,
            green_percentage=green_percentage,
            green_pixels=green_pixels,
            total_pixels=total_pixels,
        )
