import os
from pathlib import Path

def convert_json_to_txt(json_dir, txt_dir):
    """Конвертирует разметку из JSON в YOLO TXT формат."""
    import json
    os.makedirs(txt_dir, exist_ok=True)
    for json_file in Path(json_dir).glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
        # Преобразование в YOLO формат (пример, доработайте под свою разметку)
        # Здесь предполагается, что в data['objects'] лежит список объектов с координатами
        txt_file = Path(txt_dir) / (json_file.stem + ".txt")
        with open(txt_file, "w") as out:
            for obj in data.get("objects", []):
                # Пример: класс 0, нормированные координаты центра и ширины/высоты
                coords = obj["data"]
                # ... преобразование в YOLO ...
                # out.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                pass
