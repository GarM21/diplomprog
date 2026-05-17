# Green Area Analyzer

Desktop-приложение и модуль обучения U-Net для определения зелёных насаждений на спутниковых снимках.

## Возможности

- загрузка и отображение спутникового снимка в PyQt5;
- базовая OpenCV-сегментация зелёных зон;
- обучение U-Net для бинарной сегментации;
- расчёт IoU, Dice coefficient и pixel accuracy;
- сохранение лучшей и последней модели;
- предсказание маски и процента зелёных насаждений.

## Установка

```bash
pip install -r requirements.txt
```

## Датасет Kaggle

Для обучения используется датасет Kaggle:

`asim3000/satellite-imagery-urban-tree-segmentation-india`

В датасете 87 пар спутниковых изображений и бинарных масок 512x512 для сегментации городской растительности в Нью-Дели. Исходная структура датасета: `final_training_patches/images/` и `final_training_patches/masks/`. Изображения имеют расширение `.jpg`, а маски `.png`, поэтому скрипт подготовки сопоставляет пары по основе имени и сохраняет итоговые пары с одинаковыми `.png` именами.

Перед скачиванием нужен Kaggle API token:

1. Откройте Kaggle: `Account` -> `Create New API Token`.
2. Поместите `kaggle.json` в папку `C:\Users\<USER>\.kaggle\kaggle.json`.
3. Установите зависимости:

```bash
pip install -r requirements.txt
```

Скачайте и подготовьте датасет:

```bash
python training/prepare_kaggle_dataset.py
```

Скрипт скачает датасет, разобьёт его на `train/val/test` и сохранит в `dataset/urban_tree_india`.

Если архив уже скачан и распакован вручную в `dataset_raw/urban_tree_india`, можно пропустить скачивание:

```bash
python training/prepare_kaggle_dataset.py --skip-download
```

Для повторной подготовки существующей папки используйте:

```bash
python training/prepare_kaggle_dataset.py --overwrite
```

## Итоговая структура датасета

Файлы изображений и масок должны иметь одинаковые имена. Маска должна быть бинарной: `255` или `1` для зелёных зон, `0` для фона.

```text
dataset/urban_tree_india/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Пример пары:

```text
dataset/urban_tree_india/train/images/satellite_001.png
dataset/urban_tree_india/train/masks/satellite_001.png
```

## Запуск desktop-приложения

Кнопка **«Анализ»** в GUI использует обученную U-Net модель из `weights/best_unet.pth`. Если файла весов нет, приложение покажет предупреждение и предложит сначала запустить обучение.

```bash
python main.py
```

## Обучение U-Net

Параметры обучения находятся в `configs/train_config.yaml`: размер изображения, batch size, learning rate, количество эпох и пути сохранения весов.

```bash
python training/train.py
```

Можно передать другой конфиг:

```bash
python training/train.py --config configs/loveda_pretrain_config.yaml
```

Во время обучения после каждой эпохи выводятся `loss`, `IoU`, `Dice` и `PixelAcc`. Модели сохраняются в:

- `weights/best_unet.pth` — лучшая модель по IoU на валидации;
- `weights/last_unet.pth` — модель после последней эпохи.

## Улучшенное обучение на LoveDA

Для повышения точности рекомендуется сначала обучить модель на большом датасете LoveDA, а затем дообучить её на Urban Tree Segmentation India.

LoveDA — многоклассовый датасет. Для этой задачи скрипт подготовки преобразует его маски в бинарный формат:

- классы `6` и `7` (`forest`, `agriculture`) -> `255`;
- остальные классы -> `0`.

Подготовка LoveDA через Kaggle-зеркало:

```bash
python training/prepare_loveda_dataset.py
```

LoveDA весит несколько гигабайт, поэтому загрузка может идти долго. Скрипт выводит этапы загрузки, распаковки и конвертации. Если загрузка оборвалась и zip оказался повреждён, скачайте заново:

```bash
python training/prepare_loveda_dataset.py --force-download --overwrite
```

Если LoveDA уже скачан и распакован в `dataset_raw/loveda`:

```bash
python training/prepare_loveda_dataset.py --skip-download
```

Повторная подготовка с очисткой старой выходной папки:

```bash
python training/prepare_loveda_dataset.py --skip-download --overwrite
```

Предобучение на LoveDA:

```bash
python training/train.py --config configs/loveda_pretrain_config.yaml
```

Дообучение на Urban Tree Segmentation India:

```bash
python training/train.py --config configs/urban_finetune_config.yaml
```

После дообучения GUI будет использовать файл `weights/best_unet.pth`.

## Предсказание маски

Запуск из консоли:

```bash
python segmentation/predictor.py --weights weights/best_unet.pth --image dataset/test/images/satellite_001.png --output outputs/satellite_001_mask.png
```

Пример использования класса `GreenAreaPredictor`:

```python
from segmentation.predictor import GreenAreaPredictor

predictor = GreenAreaPredictor(
    weights_path="weights/best_unet.pth",
    image_size=256,
    threshold=0.5,
)

green_percent = predictor.predict_and_save(
    image_path="dataset/test/images/satellite_001.png",
    output_path="outputs/satellite_001_mask.png",
)

print(f"Green area: {green_percent:.2f}%")
```

## Аугментации

В обучающем датасете используются:

- resize до размера из `configs/train_config.yaml`;
- horizontal flip;
- vertical flip;
- random rotation;
- normalization.
