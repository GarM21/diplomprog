from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy

import cv2
import numpy as np


class ImageView(QLabel):
    """Widget for displaying OpenCV images inside a Qt interface."""

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 220)
        self.setMaximumHeight(520)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setScaledContents(False)
        self.setStyleSheet(
            """
            QLabel {
                background: #f4f6f8;
                border: 1px solid #cfd7df;
                color: #55616d;
                font-size: 14px;
            }
            """
        )

    def set_cv_image(self, image: np.ndarray) -> None:
        self._pixmap = self._convert_cv_to_pixmap(image)
        self._update_scaled_pixmap()

    def clear_image(self, placeholder: str) -> None:
        self._pixmap = None
        self.clear()
        self.setText(placeholder)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt method name
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return

        available_size = self.contentsRect().size()
        if available_size.width() <= 0 or available_size.height() <= 0:
            return

        scaled = self._pixmap.scaled(
            available_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    @staticmethod
    def _convert_cv_to_pixmap(image: np.ndarray) -> QPixmap:
        if image.ndim == 2:
            height, width = image.shape
            bytes_per_line = width
            qt_image = QImage(
                image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_Grayscale8,
            )
            return QPixmap.fromImage(qt_image.copy())

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        qt_image = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        return QPixmap.fromImage(qt_image.copy())
