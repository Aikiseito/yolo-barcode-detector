import hydra
from omegaconf import DictConfig
from ultralytics import YOLO

def detect_barcodes(image_path, model_path, conf=0.5):
    model = YOLO(model_path)
    results = model(image_path)
    bboxes = []
    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = box
            if confidence > conf:
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
    return bboxes

@hydra.main(config_path="../../configs/infer", config_name="infer.yaml")
def infer(cfg: DictConfig):
    import os
    from barcode_detector.metrics.evaluation import evaluate_on_folder
    predictions_dir = cfg.infer.predictions_dir
    os.makedirs(predictions_dir, exist_ok=True)
    for filename in os.listdir(cfg.data.test_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(cfg.data.test_dir, filename)
            bboxes = detect_barcodes(image_path, cfg.model.model_path, cfg.infer.confidence)
            # Сохраняем или выводим результаты
    # Оценка метрик
    evaluate_on_folder(cfg.data.test_dir, predictions_dir)

if __name__ == "__main__":
    infer()
