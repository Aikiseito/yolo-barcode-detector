import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import torch
from ultralytics import YOLO
import mlflow
import os

@hydra.main(config_path="../configs", config_name="default", version_base="1.1")
def train(cfg: DictConfig) -> None:
    """
    Обучение модели YOLO.

    Args:
        cfg (DictConfig): Конфигурация Hydra.
    """

    print(OmegaConf.to_yaml(cfg)) # Вывод конфигурации

    # 1. Загрузка данных (используем dvc pull внутри dataloading.py или здесь)
    #    Можно использовать API dvc или выполнить команду CLI из Python.
    #    Пример с CLI:
    # import subprocess
    # subprocess.run(["dvc", "pull"], check=True)

    # DVC Pull
    if cfg.data.use_dvc:
        import subprocess
        try:
            subprocess.run(["dvc", "pull"], check=True)
            print("Data pulled successfully using DVC.")
        except subprocess.CalledProcessError as e:
            print(f"Error pulling data with DVC: {e}")

    # 2. Определение модели
    model = YOLO(cfg.model.model_name) #  'yolov5s.pt' вынесли в конфиг

    # 3. Дообучение модели
    with mlflow.start_run() as run:
        # Логируем параметры эксперимента
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_param("git_commit_id", os.popen('git rev-parse HEAD').read().strip())

        results = model.train(
            data=os.path.join(get_original_cwd(), cfg.data.data_yaml),  # 'custom_data.yaml' вынесли в конфиг и используем абсолютный путь
            epochs=cfg.training.epochs,
            imgsz=cfg.training.imgsz,
            batch=cfg.training.batch,
            name=cfg.training.experiment_name,
        )

        # 4. Оценка результатов
        metrics = model.val()
        mlflow.log_metric("map", metrics.box.map)
        mlflow.log_metric("map50", metrics.box.map50)
        print(metrics.box.map)
        print(metrics.box.map50)

        # Логируем артефакты (модель)
        mlflow.log_artifacts("runs/detect/" + cfg.training.experiment_name)

        # 5. Конвертация в ONNX (пример)
        if cfg.training.convert_to_onnx:
            try:
                success = model.export(format="onnx",  # format="torchscript",  # или другой формат
                    # обязательно укажите путь, куда сохранить модель
                    name = cfg.training.experiment_name,
                    # path = os.path.join(get_original_cwd(), cfg.training.onnx_path)
                    )
                if success:
                    print("Model converted to ONNX successfully.")
                    # Логируем ONNX модель как артефакт
                    mlflow.log_artifact(f'{cfg.training.experiment_name}.onnx')
                else:
                    print("Failed to convert model to ONNX.")
            except Exception as e:
                print(f"Error converting model to ONNX: {e}")

if __name__ == "__main__":
    train()
