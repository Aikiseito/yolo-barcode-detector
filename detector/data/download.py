import dvc.api
import logging

def download_data():
    """Скачивает данные с помощью DVC."""
    try:
        # Используем dvc CLI из Python
        import subprocess
        subprocess.run(["dvc", "pull"], check=True)
        logging.info("Данные успешно скачаны с помощью DVC.")
    except Exception as e:
        logging.error(f"Ошибка при скачивании данных: {e}")
        raise
