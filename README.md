# Исследование кросс-датасетной адаптации модели PhysNet методом transfer learning для дистанционной фотоплетизмографии

Этот проект исследует восстановление дистанционного фотоплетизмографического
сигнала (rPPG/BVP) из RGB-видео лица.  ель исследования
— проверить перенос модели, предварительно обученной на UBFC-rPPG, на другую
камеру и популяцию MCD-rPPG.

## Экспериментальная схема

1. Видео и синхронный контактный PPG приводятся к общей временной сетке 30 Hz
   по исходным timestamps.
2. MediaPipe Face Landmarker выделяет лицо; crop имеет размер `72 × 72`. Кадры сохраняются в RGB.
3. Для входа используется DiffNormalized-представление:

   ```text
   D_t = (I_(t+1) - I_t) / (I_(t+1) + I_t + 1e-7)
   ```

   Затем разности нормируются на стандартное отклонение всего ролика.
   Целевая PPG-метка — нормированная первая разность контактного PPG.
4. Из обработанного ролика формируются окна по 10 секунд с шагом
   150 кадров. Для MCD используются только ролики `FullHDwebcam` с исходной
   частотой не ниже 27 Hz.
5. PhysNet обучается с `DifferentialWaveformLoss`: отрицательная
   корреляция разностей и корреляция кумулятивно восстановленной волны;
   вес waveform-компонента — `0.3`.

## Оценка

Валидация выполняется на уровне целого видео, а не отдельных перекрывающихся
окон. Предсказания окон сшиваются усреднением, DiffNormalized-сигнал
интегрируется в BVP, затем к предсказанию и эталону применяется одинаковая
постобработка. ЧСС определяется FFT в полосе 0.75–2.5 Hz.

## Сравнение моделей

| Модель | Датасет | валидация | Pearson | HR MAE | HR RMSE |
|---|---|---:|---:|---:|---:|
| PhysNet | UBFC-rPPG, 2kfold | 21 субъект | 0.750 | 1.63 | 2.64 |
| PhysNet | UBFC-rPPG → MCD-rPPG | 94 субъекта | **0.822** | **1.58** | 3.03 |


## Данные

| Набор | Назначение | Использованные данные |
|---|---|---|
| UBFC-rPPG | Предварительное обучение и baseline | 42 субъекта, 490 окон,  
| MCD-rPPG | fine-tuning | Камера FullHDwebcam; 31 964 окна и 468 субъектов |

## Структура

```text
models/
  physnet.py          # PhysNet: 3D-CNN для восстановления rPPG
  loss.py             # DifferentialWaveformLoss

datasets/
  ubfc.py             # UBFC-rPPG: видео и контактный PPG
  mcd.py              # MCD-rPPG: видео, ppg_sync и timestamps камер

src/
  config.py           # пути, preprocessing и параметры эксперимента
  face_detector.py    # Face Landmarker и face crop
  preprocess.py       # синхронизация, DiffNormalized и сохранение окон
  dataset.py          # загрузка и subject-level split
  train.py            # обучение PhysNet и сохранение checkpoint'ов
  evaluate.py         # video-level оценка
  utils.py            # фильтрация, FFT и метрики
```

## Запуск

```powershell
# preprocessing
.\.venv\Scripts\python.exe -m src.preprocess

# обучение
.\.venv\Scripts\python.exe -m src.train

# оценка
.\.venv\Scripts\python.exe -m src.evaluate
```