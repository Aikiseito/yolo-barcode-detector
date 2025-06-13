import fire
import hydra
from barcode_detector.models.train import train
from barcode_detector.models.infer import infer
from barcode_detector.models.export import export_to_onnx, export_to_tensorrt
from barcode_detector.data.download import download_data

def main():
    fire.Fire({
        "download_data": download_data,
        "train": train,
        "infer": infer,
        "export_onnx": export_to_onnx,
        "export_tensorrt": export_to_tensorrt,
    })

if __name__ == "__main__":
    main()
