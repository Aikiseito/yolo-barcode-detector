import hydra
from omegaconf import DictConfig
import mlflow
from barcode_detector.utils.logger import get_logger

@hydra.main(config_path="../../configs/train", config_name="train.yaml")
def train(cfg: DictConfig):
    logger = get_logger()
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    with mlflow.start_run():
        # Логируем гиперпараметры
        mlflow.log_params(dict(cfg))
        # Дообучение YOLO (ultralytics)
        from ultralytics import YOLO
        model = YOLO(cfg.model.pretrained_weights)
        results = model.train(
            data=cfg.data.yaml,
            epochs=cfg.train.epochs,
            imgsz=cfg.train.imgsz,
            batch=cfg.train.batch,
            name=cfg.train.run_name
        )
        # Логируем метрики
        mlflow.log_metric("box_map", results.box.map)
        mlflow.log_metric("box_map50", results.box.map50)
        # Сохраняем модель
        best_model_path = f"runs/detect/{cfg.train.run_name}/weights/best.pt"
        mlflow.log_artifact(best_model_path)
        logger.info(f"Модель сохранена: {best_model_path}")

if __name__ == "__main__":
    train()
