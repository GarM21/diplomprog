import cv2
import numpy as np

from models.segmentation_result import SegmentationResult


class GreenSegmenter:
    """OpenCV baseline for green area segmentation in satellite images.

    The segmenter combines HSV thresholding with an adaptive Excess Green
    index (ExG) filter. HSV alone often marks gray roofs, shadows or water as
    vegetation, so the additional color-dominance checks make the baseline
    stricter while still preserving dark green areas.
    """

    def __init__(
        self,
        lower_hsv: tuple[int, int, int] = (25, 25, 25),
        upper_hsv: tuple[int, int, int] = (95, 255, 255),
        kernel_size: int = 5,
        min_exg_threshold: int = 18,
        max_exg_threshold: int = 45,
    ) -> None:
        self._lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self._upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        self._min_exg_threshold = min_exg_threshold
        self._max_exg_threshold = max_exg_threshold

    def segment(self, image: np.ndarray) -> SegmentationResult:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv_image, self._lower_hsv, self._upper_hsv)

        b_channel, g_channel, r_channel = cv2.split(image)
        b = b_channel.astype(np.int16)
        g = g_channel.astype(np.int16)
        r = r_channel.astype(np.int16)

        exg = (2 * g) - r - b
        exg_threshold = self._estimate_exg_threshold(exg)
        exg_mask = exg >= exg_threshold

        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_a_channel = lab_image[:, :, 1]
        lab_green_mask = lab_a_channel <= 132

        green_dominance_mask = (g >= r + 4) & (g >= b - 8)
        combined_mask = (
            (hsv_mask > 0)
            & exg_mask
            & green_dominance_mask
            & lab_green_mask
        )

        mask = combined_mask.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = self._remove_small_components(mask)

        green_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        green_percentage = (green_pixels / total_pixels) * 100 if total_pixels else 0.0

        return SegmentationResult(
            mask=mask,
            green_percentage=green_percentage,
            green_pixels=green_pixels,
            total_pixels=total_pixels,
        )

    def _estimate_exg_threshold(self, exg: np.ndarray) -> int:
        exg_8bit = np.clip((exg + 255) / 2, 0, 255).astype(np.uint8)
        otsu_threshold, _ = cv2.threshold(
            exg_8bit,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        exg_threshold = int(otsu_threshold) * 2 - 255
        return int(
            np.clip(
                exg_threshold,
                self._min_exg_threshold,
                self._max_exg_threshold,
            )
        )

    @staticmethod
    def _remove_small_components(mask: np.ndarray) -> np.ndarray:
        components_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        if components_count <= 1:
            return mask

        min_area = max(16, int(mask.shape[0] * mask.shape[1] * 0.00002))
        filtered = np.zeros_like(mask)
        for component_id in range(1, components_count):
            area = stats[component_id, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered[labels == component_id] = 255

        return filtered
