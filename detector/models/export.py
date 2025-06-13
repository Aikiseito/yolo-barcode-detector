import torch
from ultralytics import YOLO

def export_to_onnx(model_path, output_path, imgsz=640):
    model = YOLO(model_path)
    model.export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True, half=False, device="cpu", path=output_path)
    print(f"Модель экспортирована в ONNX: {output_path}")

def export_to_tensorrt(onnx_path, output_path):
    import tensorrt as trt
    # Пример: используйте trtexec или python API TensorRT для конвертации
    pass
