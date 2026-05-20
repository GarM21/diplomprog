# Urban Tree Segmentation India Training Summary

Полная история эпох для этого запуска ранее не сохранялась в файл. Из текущих checkpoint-ов доступны метрики лучшей и последней эпохи.

| Checkpoint | Epoch | Train loss | Val loss | IoU | Dice | Pixel accuracy | Comment |
|---|---:|---:|---:|---:|---:|---:|---|
| `best_unet.pth` | 26 | - | 0.800671 | 0.588077 | 0.718247 | 0.895956 | Best checkpoint by validation IoU |
| `last_unet.pth` | 30 | - | 0.878892 | 0.557208 | 0.693837 | 0.875927 | Last checkpoint after training |

После следующего запуска `python training/train.py` полная таблица по эпохам будет сохраняться в `reports/urban_tree_training_history.csv`.
