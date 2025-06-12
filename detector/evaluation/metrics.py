import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(bbox1, bbox2):
    """Вычисляет хаусдорфово расстояние между двумя bounding boxes."""
    # Представляем bounding boxes как наборы точек
    points1 = np.array([[bbox1[0], bbox1[1]], [bbox1[2], bbox1[1]], [bbox1[2], bbox1[3]], [bbox1[0], bbox1[3]]])
    points2 = np.array([[bbox2[0], bbox2[1]], [bbox2[2], bbox2[1]], [bbox2[2], bbox2[3]], [bbox2[0], bbox2[3]]])

    # Вычисляем одностороннее хаусдорфово расстояние d(points1, points2) и d(points2, points1)
    d1 = directed_hausdorff(points1, points2)[0]
    d2 = directed_hausdorff(points2, points1)[0]

    return max(d1, d2)

def calculate_metrics(ground_truth_bboxes, predicted_bboxes, iou_threshold=0.5):
  """
  Вычисляет метрики качества обнаружения: Precision, Recall, F1-score и среднее значение IoU (Intersection over Union)

  Args:
      ground_truth_bboxes: Список ground truth bounding boxes
      predicted_bboxes: Список predicted bounding boxes
      iou_threshold: Порог IoU для определения true positive

  Returns:
      Словарь с метриками: {'precision': precision, 'recall': recall, 'f1': f1, 'mean_iou': mean_iou, 'hausdorf: inf'}
  """

  tp = 0  # True positives
  fp = 0  # False positives
  fn = 0  # False negatives
  ious = []
  hausdorff_distances = []

  if not ground_truth_bboxes:  # Обработка случая, когда ground truth пуст
    if predicted_bboxes:
        fp = len(predicted_bboxes)
    return {'precision': 0.0 if fp else 1.0, 'recall': 0.0, 'f1': 0.0, 'mean_iou': 0.0, 'hausdorf': 0.0}

  for gt_bbox in ground_truth_bboxes:
    best_iou = 0
    best_hausdorff = float('inf')
    found_match = False
    for pred_bbox in predicted_bboxes:
      iou = calculate_iou(gt_bbox, pred_bbox)
      haus = hausdorff_distance(gt_bbox, pred_bbox)
      best_iou = max(best_iou, iou)
      best_hausdorff = min(best_hausdorff, haus)
      if iou >= iou_threshold: # Если площадь больше трешхолда, считаем, что true positive
        found_match = True
        tp += 1
        predicted_bboxes.remove(pred_bbox) # Удаляем предсказанный bounding box, чтобы избежать повторного подсчета
        ious.append(iou)
        break
    hausdorff_distances.append(best_hausdorff)

    if not found_match: # Если код был, а мы не нашли
      fn += 1

  fp += len(predicted_bboxes) # Если что-то нашли, а ничего не было (из predicted_bboxes уже удалили то, что реально было)

  precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
  mean_iou = sum(ious) / len(ious) if ious else 0.0
  mean_hausdorff = np.mean(hausdorff_distances)

  return {'precision': precision, 'recall': recall, 'f1': f1, 'mean_iou': mean_iou, 'hausdorf': mean_hausdorff} # Пусть все пока будут, дальше разберемся


def calculate_iou(bbox1, bbox2):
  """Вычисляет IoU (Intersection over Union) для двух bounding boxes"""
  x_left = max(bbox1[0], bbox2[0])
  y_top = max(bbox1[1], bbox2[1])
  x_right = min(bbox1[2], bbox2[2])
  y_bottom = min(bbox1[3], bbox2[3])

  if x_right < x_left or y_bottom < y_top:
    return 0.0

  intersection_area = (x_right - x_left) * (y_bottom - y_top)
  
  bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
  bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
  union_area = bbox1_area + bbox2_area - intersection_area
  
  return intersection_area / union_area
