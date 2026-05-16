from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    green_percentage: float
    green_pixels: int
    total_pixels: int
