import numpy as np
from scipy.spatial.distance import directed_hausdorff

def calculate_iou(bbox1, bbox2):
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

def hausdorff_distance(bbox1, bbox2):
    points1 = np.array([[bbox1[0], bbox1[1]], [bbox1[2], bbox1[1]], [bbox1[2], bbox1[3]], [bbox1[0], bbox1[3]]])
    points2 = np.array([[bbox2[0], bbox2[1]], [bbox2[2], bbox2[1]], [bbox2[2], bbox2[3]], [bbox2[0], bbox2[3]]])
    d1 = directed_hausdorff(points1, points2)[0]
    d2 = directed_hausdorff(points2, points1)[0]
    return max(d1, d2)

def evaluate_on_folder(test_dir, predictions_dir):
    # Сравнение bbox'ов и расчет метрик для всех изображений
    pass
