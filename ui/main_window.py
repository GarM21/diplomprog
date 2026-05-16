from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
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

from ui.image_view import ImageView
from utils.image_loader import ImageLoader


if TYPE_CHECKING:
    from segmentation.predictor import GreenAreaPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best_unet.pth"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Анализ зелёных насаждений")
        self.resize(1100, 680)

        self._image = None
        self._image_path: Path | None = None
        self._predictor: "GreenAreaPredictor | None" = None

        self._source_view = ImageView("Загрузите спутниковый снимок")
        self._mask_view = ImageView("Маска результата появится после анализа")
        self._result_label = QLabel("Процент зелёных насаждений: -")

        self._build_ui()

    def _build_ui(self) -> None:
        load_button = QPushButton("Загрузить снимок")
        load_button.clicked.connect(self._load_image)

        analyze_button = QPushButton("Анализ")
        analyze_button.clicked.connect(self._analyze_image)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(load_button)
        controls_layout.addWidget(analyze_button)
        controls_layout.addStretch()

        self._result_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._result_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        controls_layout.addWidget(self._result_label)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self._source_view)
        image_layout.addWidget(self._mask_view)

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
        self._mask_view.clear_image("Маска результата появится после анализа")
        self._result_label.setText("Процент зелёных насаждений: -")
        self.statusBar().showMessage(f"Загружен файл: {self._image_path.name}")

    def _analyze_image(self) -> None:
        if self._image is None or self._image_path is None:
            QMessageBox.warning(self, "Нет изображения", "Сначала загрузите снимок.")
            return

        if not DEFAULT_WEIGHTS_PATH.exists():
            QMessageBox.warning(
                self,
                "Модель не найдена",
                (
                    "Файл weights/best_unet.pth не найден.\n"
                    "Сначала подготовьте датасет и обучите модель командой:\n"
                    "python training/train.py"
                ),
            )
            return

        try:
            predictor = self._get_predictor()
            mask, green_percentage = predictor.predict(self._image_path)
        except Exception as error:  # noqa: BLE001 - GUI should show model errors to the user.
            QMessageBox.critical(self, "Ошибка анализа", str(error))
            return

        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self._mask_view.set_cv_image(colored_mask)
        self._result_label.setText(
            f"Процент зелёных насаждений: {green_percentage:.2f}%"
        )
        self.statusBar().showMessage("Анализ выполнен нейросетевой моделью U-Net")

    def _get_predictor(self) -> "GreenAreaPredictor":
        from segmentation.predictor import GreenAreaPredictor

        if self._predictor is None:
            self.statusBar().showMessage("Загрузка модели U-Net...")
            self._predictor = GreenAreaPredictor(
                weights_path=DEFAULT_WEIGHTS_PATH,
                image_size=256,
                threshold=0.5,
                features=64,
            )
        return self._predictor
