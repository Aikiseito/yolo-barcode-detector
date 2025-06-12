import pytorch_lightning as pl
import torch
from torch import nn
from ultralytics import YOLO
import mlflow
import torchmetrics
from typing import Any


class YOLOLightningModule(pl.LightningModule):
    def __init__(self, model_name: str = "yolov8n.pt", lr: float = 0.001, weight_decay: float = 0.0005, conf_thres=0.25, iou_thres=0.7):
        super().__init__()
        self.model = YOLO(model_name)
        self.lr = lr
        self.weight_decay = weight_decay
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Metrics
        self.train_metrics = torchmetrics.detection.MeanAveragePrecision(box_format='xywhn', iou_thresholds=[0.5], class_metrics=True)
        self.val_metrics = torchmetrics.detection.MeanAveragePrecision(box_format='xywhn', iou_thresholds=[0.5], class_metrics=True)
        self.test_metrics = torchmetrics.detection.MeanAveragePrecision(box_format='xywhn', iou_thresholds=[0.5], class_metrics=True)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        results = self.model(images)
        loss = results.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #Calculate metrics
        self.train_metrics.update(results.pred, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        results = self.model(images)
        loss = results.loss
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #Calculate metrics
        self.val_metrics.update(results.pred, targets)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        results = self.model(images)
        loss = results.loss
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #Calculate metrics
        self.test_metrics.update(results.pred, targets)
        return loss
    
    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict({'train/mAP': metrics[0], 'train/mAP50': metrics[1], 'train/mAP75': metrics[2]}, prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict({'val/mAP': metrics[0], 'val/mAP50': metrics[1], 'val/mAP75': metrics[2]}, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict({'test/mAP': metrics[0], 'test/mAP50': metrics[1], 'test/mAP75': metrics[2]}, prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
