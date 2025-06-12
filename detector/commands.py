import fire
import os
from barcode_detection.training import train
from barcode_detection.inference import predict
# from barcode_detection.evaluation import evaluate # Если нужен отдельный модуль

class Commands:
    """
    CLI для запуска обучения, предсказания и оценки модели
    """

    def train(self, config_path: str = "configs/default.yaml"):
        """
        Запускает обучение модели

        Args:
            config_path (str): Путь к конфигурационному файлу Hydra.
        """
        train.train() #  config_path=config_path

    def predict(self, config_path: str = "configs/default.yaml"):
        """
        Запускает предсказание модели

        Args:
            config_path (str): Путь к конфигурационному файлу Hydra
        """
        predict.predict() # config_path=config_path

    # def evaluate(self, config_path: str = "configs/default.yaml"):
    #     """
    #     Запускает оценку модели
    #
    #     Args:
    #         config_path (str): Путь к конфигурационному файлу Hydra
    #     """
    #     evaluate.evaluate(config_path=config_path)


if __name__ == "__main__":
    fire.Fire(Commands)
