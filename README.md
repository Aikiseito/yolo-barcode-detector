# barcode detector

Детектирование зон – грубые коробки (bbox detection). Ищем одномерные и двумерные коды на фотографиях: qr, ean13, ean8, upc, 1d (нераспознанные одномерки), dm, az, pdf, id (невозможно детектировать), hd (трудно детектировать).

# Постановка задачи

В данном проекте я решаю задачу детекции одномерных (например, штрих-кодов типа ean13) и двумерных (например, qr-коды) кодов на фотографиях. Моя система анализирует изображения и выдаёт координаты четырех углов для каждой фигуры, в которую помещается обнаруженный код, или сообщает об его отсутствии. 

__Input:__ картинка в формате .jpg или .png – фотография с мобильного устройства.

__Output:__ массив вида [(𝑥1, 𝑥2, 𝑥3, 𝑥4), ...] – координаты left top – right bottom углов bbox’а, в котором находится код. Возвращает пустой список, если штрихкод не найден. Есть возможность получить на выход картинку с bbox’ом.

# Метрики

Ключевые метрики - IoU и расстояние Хаусдорфа.

# Валидация

Для разделения выборок использую метод K-fold кросс-валидацию с 5 фолд.

# Данные 

Мой датасет: 117 фотографий 1D и 2D кодов в формате .jpg и соответствующие им файлы .json (имя-фото.jpg.json) с разметкой. Из них:
qr  | ean13 | 1d | dm | pdf | upc |  hd | id |
103 |   83  | 69 | 69 |  4  |  3  | 159 | 68 |

# Моделирование

__Бейзлайн__

Сравнение со стандартной моделью YOLO (без дообучения)

__Основная модель__

yolov5s.pt

__Внедрение__

Программный пакет состоит из нескольких модулей, отвечающих за обучение и валидацию, а также за инференс модели

# Установка

__Скачивание данных__

«`{bash}< >{python commands.py download_data}«`

__Обучение__

python commands.py train

__Инференс__

python commands.py infer

__ONNX экспорт__

python commands.py export_onnx --model_path=... --output_path=...

__TensorRT экспорт__

«`{bash}< >{python commands.py export_tensorrt --onnx_path=... --output_path=...}«`

1. Клонируйте репозиторий
    git clone https://github.com/your-username/barcode-detector.git
    cd barcode-detector
2. Установите poetry
     poetry install






# Структура репозитория

yolo-barcode-detector/
├── detector/          # Python-пакет
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── preprocess.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   ├── export.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── git_utils.py
│   └── serving/
│       ├── __init__.py
│       ├── server.py
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   ├── data.yaml
│   ├── model/
│   │   ├── yolo.yaml
│   ├── train/
│   │   ├── train.yaml
│   ├── infer/
│   │   ├── infer.yaml
│   ├── mlflow.yaml
├── data/
│   ├── train.dvc
│   ├── val.dvc
│   ├── test.dvc
├── plots/
│   └── (графики и визуализации)
├── commands.py
├── pyproject.toml
├── README.md
├── .pre-commit-config.yaml
├── .gitignore
└── dvc.yaml
