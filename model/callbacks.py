from enum import Enum
import sys
import torch
import torchvision

from torchmetrics import (
    MetricCollection,
    Precision,
    PrecisionRecallCurve,
    Recall,
    F1Score,
    ConfusionMatrix,
)

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from pytorch_accelerated.callbacks import TrainerCallback

from model.utils import yolo_to_xyxy, detection_results_to_classification_results, LossTracker

import numpy as np

class RunType(Enum):
    TRAINING = 0
    EVALUATION = 1


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
            self.trainer.update_metric(f'train_{loss_name}', loss_tracker.average)
            loss_tracker.reset()


    def on_eval_epoch_end(self, trainer, **kwargs):
        """
        Update eval loss series with average loss for epoch and reset loss trackers.
        """
        for loss_name, loss_tracker in self.eval_loss.items():
            self.trainer.update_metric(f'eval_{loss_name}', loss_tracker.average)
            loss_tracker.reset()


class BinaryPrecisionRecallMetricsCallback(TrainerCallback):
    def __init__(self, average='macro', device='cuda:0', confidence_threshold=0.4):
        super().__init__()
        self.average = average
        self.run_type = None
        self.metrics_returns_lookup = {
            # Metric             returns
            'pr_curve' :         ['pr_curve_precision', 'pr_curve_recall', 'pr_curve_thresholds'],
            'confusion_matrix' : ['confusion_matrix'],
            'precision' :        ['precision'],
            'recall' :           ['recall'],
            'f1' :               ['f1'],
        }
        self.series_metrics = MetricCollection(
            {
                'pr_curve' : PrecisionRecallCurve(
                    task='binary',
                    average=average
                ).to(device),
                'confusion_matrix' : ConfusionMatrix(
                    task='binary',
                    average=average,
                    threshold=confidence_threshold
                ).to(device),
            }
        )
        self.one_shot_metrics = MetricCollection({
            'precision' : Precision(
                task='binary',
                average=average,
                threshold=confidence_threshold
            ).to(device),
            'recall' : Recall(
                task='binary',
                average=average,
                threshold=confidence_threshold
            ).to(device),
            'f1' : F1Score(
                task='binary',
                average=average,
                threshold=confidence_threshold
            ).to(device),
        })


    def _update_history(self, trainer, computed_metrics_collection):
        for metric_name, metric in computed_metrics_collection.items():
            metric_returns_names = self.metrics_returns_lookup[metric_name]
            if len(metric_returns_names) == 1: # TODO: This if is not necessary, loop is enough
                trainer.run_history.update_metric(metric_returns_names[0], metric.cpu())
            else:
                for i, metric_output in enumerate(metric_returns_names):
                    trainer.run_history.update_metric(metric_output, metric[i].cpu())


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
        print(f"batch size: {images.shape[0]}")
        print(f"batch output losses: {batch_output['loss_items']}")
        print(f"batch output loss: {batch_output['loss']}")

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
        self._update_history(trainer, computed_metrics)
        self.one_shot_metrics.reset()

        # Metrics recorded only at the end of training
        # during evaluation run
        if self.run_type == RunType.EVALUATION:
            computed_curve_metrics = self.series_metrics.compute()
            self._update_history(trainer, computed_curve_metrics)
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
