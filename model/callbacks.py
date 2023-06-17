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

from model.utils import yolo_to_xyxy, detection_results_to_classification_results

import numpy as np


class PrecisionRecallMetricsCallback(TrainerCallback):
    def __init__(self, task, num_classes, average='macro', device='cuda:0', confidence_threshold=0.4):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.average = average
        self.metrics = MetricCollection({
            'precision' : Precision(
                task=task,
                num_classes=num_classes,
                average=average,
                threshold=confidence_threshold
            ).to(device),
            'recall' : Recall(
                task=task,
                num_classes=num_classes,
                average=average,
                threshold=confidence_threshold
            ).to(device),
            'pr_curve' : PrecisionRecallCurve(
                task=task,
                num_classes=num_classes,
                average=average
            ).to(device),
            'f1' : F1Score(
                task=task,
                num_classes=num_classes,
                average=average,
                threshold=confidence_threshold
            ).to(device),
            'confusion_matrix' : ConfusionMatrix(
                task=task,
                num_classes=num_classes,
                average=average,
                threshold=confidence_threshold
            ).to(device),
        })

    def _move_to_device(self, trainer):
        self.metrics.to(trainer.device)


    def on_training_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)


    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)


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
            self.metrics.update(classification_preds, classification_gt)


    def on_eval_epoch_end(self, trainer, **kwargs):
        computed_metrics = self.metrics.compute()
        pr_curve_precision, pr_curve_recall, pr_curve_thresholds = computed_metrics['pr_curve']
        trainer.run_history.update_metric('precision', computed_metrics['precision'].cpu())
        trainer.run_history.update_metric('recall', computed_metrics['recall'].cpu())
        trainer.run_history.update_metric('pr_curve_precision', pr_curve_precision.cpu())
        trainer.run_history.update_metric('pr_curve_recall', pr_curve_recall.cpu())
        trainer.run_history.update_metric('pr_curve_thresholds', pr_curve_thresholds.cpu())
        trainer.run_history.update_metric('f1', computed_metrics['f1'].cpu())
        trainer.run_history.update_metric('confusion_matrix', computed_metrics['confusion_matrix'].cpu())
        self.metrics.reset()


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
