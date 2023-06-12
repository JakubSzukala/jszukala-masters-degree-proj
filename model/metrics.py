import sys
import torch
import torchvision

from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    PrecisionRecallCurve,
    Recall,
    F1Score,
    ConfusionMatrix,
)

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from pytorch_accelerated.callbacks import TrainerCallback

import numpy as np


def cxcywh_to_xyxy(box):
            x, y, w, h = box
            return torch.tensor([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


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
            ).to(device)
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

        for batch_image_id, absolute_image_id in enumerate(image_ids):
            single_image_preds = preds[preds[:, 6] == absolute_image_id, :]
            single_image_gt = ground_truth_labels[ground_truth_labels[:, 0] == batch_image_id, :]
            self.update_metrics(trainer, single_image_gt, single_image_preds, original_image_sizes)


    def update_metrics(self, trainer, ground_truth_labels, preds, original_image_sizes):
        # TODO: wrap it in another function
        # Denormalize and convert ncxncywh to xyxy
        gt_boxes = ground_truth_labels[:, 2:].clone()
        gt_boxes[:, [0, 2]] *= original_image_sizes[0, 1]
        gt_boxes[:, [1, 3]] *= original_image_sizes[0, 0]
        for i, row in enumerate(gt_boxes):
            gt_boxes[i, :] = cxcywh_to_xyxy(row)

        # No predictions and no ground truths
        # This would mean updating TN which are irrelevant for recall and precision
        if preds.shape[0] == 0 and gt_boxes.shape[0] == 0:
            return

        # Any prediction made when no gt boxes are present is a false positive
        if gt_boxes.shape[0] == 0:
            metric_input_gt = torch.zeros(preds.shape[0], dtype=torch.int, device=trainer.device)
            metric_input_preds = preds[:, 4]
            self.metrics.update(metric_input_preds, metric_input_gt)
            return

        # No predictions made when gt boxes are present is a false negative
        if preds.shape[0] == 0:
            metric_input_gt = torch.ones(gt_boxes.shape[0], dtype=torch.int, device=trainer.device)
            metric_input_preds = torch.zeros(gt_boxes.shape[0], device=trainer.device)
            self.metrics.update(metric_input_preds, metric_input_gt)
            return

        iou_matrix = torchvision.ops.box_iou(gt_boxes, preds[:, :4])

        results_dim = min(gt_boxes.shape[0], preds.shape[0])
        recorded_matches = torch.empty(results_dim, 2, device=trainer.device)
        preds_match_idices = []
        for i in range(results_dim):
            # Find best iou for each gt
            best_match_iou_vec, best_match_pred_idx_vec  = iou_matrix.max(dim=1)

            # Of these find which gt is best matched
            best_match_iou_scalar = best_match_iou_vec.max()
            best_match_gt_idx = best_match_iou_vec.argmax()
            best_match_pred_idx = best_match_pred_idx_vec[best_match_gt_idx]

            assert best_match_iou_scalar == iou_matrix[best_match_gt_idx, best_match_pred_idx]

            # It has to be adjusted, because max overlap could be zero, but confidence score could be high
            confidence_score = preds[best_match_pred_idx, 4] if best_match_iou_scalar > 0 else 0

            # Record the match
            recorded_matches[i, :] = torch.tensor([
                1.0, # Ground truth
                confidence_score
            ], device=trainer.device)

            # Record pred index of a match
            preds_match_idices.append(best_match_pred_idx)

            # Set the matched gt and pred to -1 so that they are not matched again
            iou_matrix[best_match_gt_idx, :] = -1
            iou_matrix[:, best_match_pred_idx] = -1

        # Assert that there are no duplicate pred indices
        assert len(preds_match_idices) == len(set(preds_match_idices))

        if preds.shape[0] > gt_boxes.shape[0]:
            padding_size = preds.shape[0] - gt_boxes.shape[0]
            mask = torch.ones(preds.shape[0], dtype=torch.bool, device=trainer.device)
            mask[preds_match_idices] = False
            preds_without_matches = preds[mask, :]
            confidence_scores_padding = preds_without_matches[:, 4]
            assert confidence_scores_padding.shape[0] == padding_size
            padding = torch.hstack([
                torch.zeros(padding_size, 1, device=trainer.device),
                confidence_scores_padding.unsqueeze(1)
            ])
            recorded_matches = torch.vstack([
                recorded_matches,
                padding
            ])

        elif preds.shape[0] < gt_boxes.shape[0]:
            padding_size = gt_boxes.shape[0] - preds.shape[0]
            padding = torch.hstack([
                torch.ones(padding_size, 1, device=trainer.device),
                torch.zeros(padding_size, 1, device=trainer.device)
            ])
            recorded_matches = torch.vstack([
                recorded_matches,
                padding
            ])
        metric_input_gt = recorded_matches[:, 0].type(torch.int)
        metric_input_preds = recorded_matches[:, 1]
        self.metrics.update(metric_input_preds, metric_input_gt)


    def on_eval_epoch_end(self, trainer, **kwargs):
        computed_metrics = self.metrics.compute()
        pr_curve_precision, pr_curve_recall, pr_curve_thresholds = computed_metrics['pr_curve']
        trainer.run_history.update_metric('precision', computed_metrics['precision'].cpu())
        trainer.run_history.update_metric('recall', computed_metrics['recall'].cpu())
        trainer.run_history.update_metric('pr_curve_precision', pr_curve_precision.cpu())
        trainer.run_history.update_metric('pr_curve_recall', pr_curve_recall.cpu())
        trainer.run_history.update_metric('pr_curve_thresholds', pr_curve_thresholds.cpu())
        trainer.run_history.update_metric('f1', computed_metrics['f1'].cpu())
        self.metrics.reset()


class MeanAveragePrecisionCallback(TrainerCallback):
    def __init__(self, iou_thresholds=None):
        super().__init__()
        self.iou_thresholds = iou_thresholds
        if iou_thresholds is None:
            self.iou_thresholds = list(torch.arange(0.5, 0.75, 0.05))
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

        # Denormalize and convert ncxncywh to xyxy
        gt_boxes = ground_truth_labels[:, 2:].clone()
        gt_boxes[:, [0, 2]] *= original_image_sizes[0, 1]
        gt_boxes[:, [1, 3]] *= original_image_sizes[0, 0]
        for i, row in enumerate(gt_boxes):
            gt_boxes[i, :] = cxcywh_to_xyxy(row)

        metric_input_preds = []
        metric_input_gt = []
        for batch_image_id, absolute_image_id in enumerate(image_ids):
            # gt labels: [ image_id, class id, normalized cxcywh ]
            # preds: [ xyxy, score, class_id, image_id ]
            single_image_preds = preds[preds[:, 6] == absolute_image_id, :]
            single_image_gt = ground_truth_labels[ground_truth_labels[:, 0] == batch_image_id, :].clone()
            # Denormalize and convert ncxncywh to xyxy
            single_image_gt[:, [2, 4]] *= original_image_sizes[0, 1]
            single_image_gt[:, [3, 5]] *= original_image_sizes[0, 0]
            for i, row in enumerate(single_image_gt):
                single_image_gt[i, 2:] = cxcywh_to_xyxy(row[2:])

            metric_input_preds.append(
                {
                    'boxes' : single_image_preds[:, :4],
                    'scores' : single_image_preds[:, 4],
                    'labels' : single_image_preds[:, 5]
                }
            )
            metric_input_gt.append(
                {
                    'boxes' : single_image_gt[:, 2:],
                    'labels' : single_image_gt[:, 1]
                }
            )
        self.metric.update(metric_input_preds, metric_input_gt)


    def on_eval_epoch_end(self, trainer, **kwargs):
        computed_metrics = self.metric.compute() # TODO: Add range here
        trainer.run_history.update_metric('mean_average_precision', computed_metrics['map'].cpu())
        self.metric.reset()
