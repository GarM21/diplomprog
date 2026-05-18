from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from segmentation.green_segmenter import GreenSegmenter
from ui.image_view import ImageView
from utils.image_loader import ImageLoader


if TYPE_CHECKING:
    from segmentation.predictor import GreenAreaPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best_unet.pth"
METHOD_OPENCV = "OpenCV HSV"
METHOD_UNET = "U-Net"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Анализ зелёных насаждений")
        self.resize(1180, 720)

        self._image: np.ndarray | None = None
        self._image_path: Path | None = None
        self._segmenter = GreenSegmenter()
        self._predictor: "GreenAreaPredictor | None" = None

        self._source_view = ImageView("Загрузите спутниковый снимок")
        self._result_view = ImageView("Overlay-результат появится после анализа")
        self._method_combo = QComboBox()
        self._result_label = QLabel("Процент зелёных насаждений: -")

        self._build_ui()

    def _build_ui(self) -> None:
        load_button = QPushButton("Загрузить снимок")
        load_button.clicked.connect(self._load_image)

        analyze_button = QPushButton("Анализ")
        analyze_button.clicked.connect(self._analyze_image)

        method_label = QLabel("Метод анализа:")
        self._method_combo.addItems([METHOD_OPENCV, METHOD_UNET])
        self._method_combo.setCurrentText(METHOD_OPENCV)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(load_button)
        controls_layout.addWidget(method_label)
        controls_layout.addWidget(self._method_combo)
        controls_layout.addWidget(analyze_button)
        controls_layout.addStretch()

        self._result_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._result_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        controls_layout.addWidget(self._result_label)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self._source_view)
        image_layout.addWidget(self._result_view)

        root_layout = QVBoxLayout()
        root_layout.addLayout(controls_layout)
        root_layout.addLayout(image_layout, stretch=1)

        central_widget = QWidget()
        central_widget.setLayout(root_layout)
        self.setCentralWidget(central_widget)
        self.setStatusBar(QStatusBar())

    def _load_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите спутниковый снимок",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not file_name:
            return

        try:
            self._image_path = Path(file_name)
            self._image = ImageLoader.load(file_name)
        except ValueError as error:
            QMessageBox.critical(self, "Ошибка загрузки", str(error))
            return

        self._source_view.set_cv_image(self._image)
        self._result_view.clear_image("Overlay-результат появится после анализа")
        self._result_label.setText("Процент зелёных насаждений: -")
        self.statusBar().showMessage(f"Загружен файл: {self._image_path.name}")

    def _analyze_image(self) -> None:
        if self._image is None or self._image_path is None:
            QMessageBox.warning(self, "Нет изображения", "Сначала загрузите снимок.")
            return

        method = self._method_combo.currentText()
        try:
            if method == METHOD_OPENCV:
                mask, green_percentage = self._analyze_with_opencv()
            else:
                mask, green_percentage = self._analyze_with_unet()
        except Exception as error:  # noqa: BLE001 - GUI should show analysis errors.
            QMessageBox.critical(self, "Ошибка анализа", str(error))
            return

        overlay = self._create_overlay(self._image, mask)
        self._result_view.set_cv_image(overlay)
        self._result_label.setText(
            f"Процент зелёных насаждений: {green_percentage:.2f}%"
        )
        self.statusBar().showMessage(f"Анализ выполнен методом: {method}")

    def _analyze_with_opencv(self) -> tuple[np.ndarray, float]:
        if self._image is None:
            raise ValueError("Изображение не загружено.")

        result = self._segmenter.segment(self._image)
        return result.mask, result.green_percentage

    def _analyze_with_unet(self) -> tuple[np.ndarray, float]:
        if self._image_path is None:
            raise ValueError("Изображение не загружено.")
        if not DEFAULT_WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                "Файл weights/best_unet.pth не найден. "
                "Сначала обучите модель командой: python training/train.py"
            )

        predictor = self._get_predictor()
        return predictor.predict(self._image_path)

    def _get_predictor(self) -> "GreenAreaPredictor":
        from segmentation.predictor import GreenAreaPredictor

        if self._predictor is None:
            self.statusBar().showMessage("Загрузка модели U-Net...")
            self._predictor = GreenAreaPredictor(
                weights_path=DEFAULT_WEIGHTS_PATH,
                image_size=512,
                threshold=0.5,
                features=64,
            )
        return self._predictor

    @staticmethod
    def _create_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)

        mask_bool = mask > 0
        green_layer = np.zeros_like(image)
        green_layer[mask_bool] = (0, 255, 0)

        highlighted = cv2.addWeighted(image, 1.0 - alpha, green_layer, alpha, 0)
        overlay = image.copy()
        overlay[mask_bool] = highlighted[mask_bool]
        return overlay
