import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO
import cv2
import os

@hydra.main(config_path="../../configs", config_name="default", version_base="1.1")
def predict(cfg: DictConfig):
    """
    Обнаружение штрих-кода на изображении

    Args:
        image_path: Путь к изображению

    Returns:
        Список bounding boxes (координаты левого верхнего и правого нижнего углов)
        в формате [(x1, y1, x2, y2), ...]. Возвращает пустой список, если штрих-код не найден
    """

    print(OmegaConf.to_yaml(cfg))

    # DVC Pull
    if cfg.data.use_dvc:
        import subprocess
        try:
            subprocess.run(["dvc", "pull"], check=True)
            print("Data pulled successfully using DVC.")
        except subprocess.CalledProcessError as e:
            print(f"Error pulling data with DVC: {e}")

    model = YOLO(cfg.inference.model_path) # Путь к модели из конфига

    def detect_barcode(image_path):
        results = model(image_path)
        bboxes = []
        for r in results:
            for box in r.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = box
                if confidence > cfg.inference.confidence_threshold:  # Порог confidence из конфига
                    bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        return bboxes

    # Пример использования:
    # Добавим цикл по изображениям из конфига или из папки
    if cfg.inference.image_paths:  # Если пути к изображениям заданы в конфиге
        image_paths = cfg.inference.image_paths
    else:  # Иначе берем все изображения из указанной папки
        image_paths = [os.path.join(cfg.data.test_dir, filename)
                       for filename in os.listdir(cfg.data.test_dir)
                       if filename.endswith((".jpg", ".png"))]

    for image_path in image_paths:
        bboxes = detect_barcode(image_path)
        print(f"Обнаруженные bounding boxes для {image_path}: {bboxes}")

        # Optional: Рисование и сохранение результатов
        if cfg.inference.draw_results:
            img = cv2.imread(image_path)
            for bbox in bboxes:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            output_path = os.path.join(cfg.inference.output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, img)
            print(f"Результат сохранен в {output_path}")

if __name__ == "__main__":
    predict()
