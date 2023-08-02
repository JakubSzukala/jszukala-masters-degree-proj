from abc import ABCMeta
from enum import Enum
import sys
from typing import Any
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import (
    MetricCollection,
    Precision,
    PrecisionRecallCurve,
    Recall,
    F1Score,
    ConfusionMatrix,
)

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from pytorch_accelerated.callbacks import TrainerCallback, LogMetricsCallback

from model.utils import is_scalar, yolo_to_xyxy, detection_results_to_classification_results, LossTracker

import numpy as np

class RunType(Enum):
    TRAINING = 0
    EVALUATION = 1


class TensorboardLoggingCallback(LogMetricsCallback):
    def __init__(self, log_dir):
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir)


    def log_metrics(self, trainer, metrics: dict):
        """
        Overrridden function that will both log to console and to tensorboard.
        It will only log scalars, one at a time.
        """
        for metric_name, metric_value in metrics.items():
            if 'train' in metric_name:
                prefix = 'train'
            elif 'eval' in metric_name:
                prefix = 'eval'
            else:
                prefix = 'metric'
            trainer.print(f"\n{metric_name}: {metric_value}")
            if is_scalar(metric_value):
                self.writer.add_scalar(prefix + '/' + metric_name, metric_value, trainer.run_history.current_epoch)


class DetectionLossTrackerCallback(TrainerCallback):
    """
    Callback class used for tracking loss values for detection model.
    """
    def __init__(self):
        self.train_loss = {
            'box_loss' : LossTracker(),
            'obj_loss' : LossTracker(),
            'cls_loss' : LossTracker(),
        }

        self.eval_loss = {
            'box_loss' : LossTracker(),
            'obj_loss' : LossTracker(),
            'cls_loss' : LossTracker(),
        }


    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        """
        Update corresponding loss trackers with loss for eval step. batch_output_items_scaled
        is scaled by batch size to mimic final loss calculation in yolov7's loss.py.
        """
        batch_size = batch[0].shape[0]

        batch_output_items_scaled = batch_output['loss_items'] * batch_size

        self.eval_loss['box_loss'].update(batch_output_items_scaled[0], batch_size)
        self.eval_loss['obj_loss'].update(batch_output_items_scaled[1], batch_size)
        self.eval_loss['cls_loss'].update(batch_output_items_scaled[2], batch_size)


    def on_train_step_end(self, trainer, batch, batch_output, **kwargs):
        """
        Update corresponding loss trackers with loss for train step. batch_output_items_scaled
        is scaled by batch size to mimic final loss calculation in yolov7's loss.py.
        """
        batch_size = batch[0].shape[0]

        batch_output_items_scaled = batch_output['loss_items'] * batch_size

        self.train_loss['box_loss'].update(batch_output_items_scaled[0], batch_size)
        self.train_loss['obj_loss'].update(batch_output_items_scaled[1], batch_size)
        self.train_loss['cls_loss'].update(batch_output_items_scaled[2], batch_size)


    def on_train_epoch_end(self, trainer, **kwargs):
        """
        Update train loss series with average loss for epoch and reset loss trackers.
        """
        for loss_name, loss_tracker in self.train_loss.items():
            trainer.run_history.update_metric(f'train_{loss_name}', loss_tracker.average)
            loss_tracker.reset()


    def on_eval_epoch_end(self, trainer, **kwargs):
        """
        Update eval loss series with average loss for epoch and reset loss trackers.
        """
        for loss_name, loss_tracker in self.eval_loss.items():
            trainer.run_history.update_metric(f'eval_{loss_name}', loss_tracker.average)
            loss_tracker.reset()


class BinaryPrecisionRecallMetricsCallback(TrainerCallback):
    def __init__(self, device='cuda:0', confidence_threshold=0.4):
        super().__init__()
        self.run_type = None
        self.series_metrics = MetricCollection(
            {
                'pr_curve' : PrecisionRecallCurve(
                    task='binary',
                ).to(device),
                'confusion_matrix' : ConfusionMatrix(
                    task='binary',
                    threshold=confidence_threshold,
                    normalize='true'
                ).to(device),
            }
        )
        self.one_shot_metrics = MetricCollection({
            'precision' : Precision(
                task='binary',
                threshold=confidence_threshold
            ).to(device),
            'recall' : Recall(
                task='binary',
                threshold=confidence_threshold
            ).to(device),
            'f1' : F1Score(
                task='binary',
                threshold=confidence_threshold
            ).to(device),
        })


    def on_training_run_start(self, trainer, **kwargs):
        self.run_type = RunType.TRAINING
        self.one_shot_metrics.to(trainer.device)


    def on_evaluation_run_start(self, trainer, **kwargs):
        self.run_type = RunType.EVALUATION
        self.one_shot_metrics.to(trainer.device)
        self.series_metrics.to(trainer.device)


    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        # gt labels: [ image_id, class id, normalized cxcywh ]
        # preds: [ xyxy, score, class_id, image_id ]
        preds = batch_output['predictions'].to(trainer.device)
        images, ground_truth_labels, image_ids, original_image_sizes = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
        )

        # Isolate single image for calculation of metrics, this way no image mixing will occur
        for batch_image_id, absolute_image_id in enumerate(image_ids):
            # Get only xyxy boxes and scores from single image preds
            single_image_preds = preds[preds[:, 6] == absolute_image_id, :]
            single_image_preds_boxes_scores = single_image_preds[:, :5]

            # Get only xyxy boxes image gts
            single_image_gt = ground_truth_labels[ground_truth_labels[:, 0] == batch_image_id, :]
            single_image_gt_boxes = single_image_gt[:, 2:].clone()
            single_image_gt_boxes = yolo_to_xyxy(single_image_gt_boxes, original_image_sizes)

            # Convert them to classification results for classification metrics calculation
            classification_gt, classification_preds = detection_results_to_classification_results(
                single_image_gt_boxes,
                single_image_preds_boxes_scores,
                trainer.device
            )

            self.one_shot_metrics.update(classification_preds, classification_gt)
            if self.run_type == RunType.EVALUATION:
                self.series_metrics.update(classification_preds, classification_gt)


    def on_eval_epoch_end(self, trainer, **kwargs):
        # Metrics recorded every eval epoch
        computed_metrics = self.one_shot_metrics.compute()
        for metric_name, metric in computed_metrics.items():
            trainer.run_history.update_metric(metric_name, metric.cpu())
        self.one_shot_metrics.reset()

        # Metrics recorded only at the end of training
        # during evaluation run
        if self.run_type == RunType.EVALUATION:
            computed_curve_metrics = self.series_metrics.compute()
            trainer.run_history.update_metric(
                'confusion_matrix',
                computed_curve_metrics['confusion_matrix'].cpu()
            )

            # Prepare precision recall curve metrics and calculate f1 curve
            pr_curve_precision = computed_curve_metrics['pr_curve'][0]
            pr_curve_recall = computed_curve_metrics['pr_curve'][1]
            pr_curve_thresholds = computed_curve_metrics['pr_curve'][2].cpu()
            pr_curve_thresholds = torch.cat([torch.tensor([0.0]), pr_curve_thresholds])
            f1_curve = 2 * (pr_curve_precision * pr_curve_recall) / (pr_curve_precision + pr_curve_recall)
            trainer.run_history.update_metric('pr_curve_precision', pr_curve_precision.cpu())
            trainer.run_history.update_metric('pr_curve_recall', pr_curve_recall.cpu())
            trainer.run_history.update_metric('pr_curve_thresholds', pr_curve_thresholds)
            trainer.run_history.update_metric('f1_curve', f1_curve.cpu())
            self.series_metrics.reset()


    def on_training_run_end(self, trainer, **kwargs):
        self.run_type = None


    def on_evaluation_run_end(self, trainer, **kwargs):
        self.run_type = None


class MeanAveragePrecisionCallback(TrainerCallback):
    def __init__(self, iou_thresholds=None):
        super().__init__()
        self.iou_thresholds = iou_thresholds
        if iou_thresholds is None:
            self.iou_thresholds = np.linspace(0.5, 0.75, 6).tolist()
        self.th_string = f'{min(self.iou_thresholds())}' \
            if len(self.iou_thresholds) == 1 \
            else f'{min(self.iou_thresholds)}-{max(self.iou_thresholds)}'
        self.metric = MeanAveragePrecision(
            iou_thresholds=self.iou_thresholds
        )

    def _move_to_device(self, trainer):
        self.metric.to(trainer.device)


    def on_training_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)


    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)


    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        # gt labels: [ 1, class id, normalized cxcywh ]
        # preds: [ xyxy, score, class_id, image_id ]
        preds = batch_output['predictions'].to(trainer.device)
        images, ground_truth_labels, image_ids, original_image_sizes = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
        )

        metric_input_preds = []
        metric_input_gt = []
        for batch_image_id, absolute_image_id in enumerate(image_ids):
            # gt labels: [ image_id, class id, normalized cxcywh ]
            # preds: [ xyxy, score, class_id, image_id ]
            single_image_preds = preds[preds[:, 6] == absolute_image_id, :]
            single_image_gt = ground_truth_labels[ground_truth_labels[:, 0] == batch_image_id, :].clone()
            single_image_gt_boxes = single_image_gt[:, 2:].clone()
            single_image_gt_boxes = yolo_to_xyxy(single_image_gt_boxes, original_image_sizes)

            metric_input_preds.append(
                {
                    'boxes' : single_image_preds[:, :4],
                    'scores' : single_image_preds[:, 4],
                    'labels' : single_image_preds[:, 5]
                }
            )
            metric_input_gt.append(
                {
                    # Copy of converted boxes
                    'boxes' : single_image_gt_boxes,
                    'labels' : single_image_gt[:, 1]
                }
            )
        self.metric.update(metric_input_preds, metric_input_gt)


    def on_eval_epoch_end(self, trainer, **kwargs):
        computed_metrics = self.metric.compute() # TODO: Add range here
        trainer.run_history.update_metric(f'mAP:{self.th_string}', computed_metrics['map'].cpu())
        trainer.run_history.update_metric(f'mAP:0.5', computed_metrics['map_50'].cpu())
        self.metric.reset()
