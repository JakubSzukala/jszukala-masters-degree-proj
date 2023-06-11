import sys
import torch
import torchvision

from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    PrecisionRecallCurve,
    Recall,
#    F1,
    ConfusionMatrix
)

from pytorch_accelerated.callbacks import TrainerCallback

import numpy as np


def cxcywh_to_xyxy(box):
            x, y, w, h = box
            return torch.tensor([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


class PrecisionRecallMetricsCallback(TrainerCallback):
    def __init__(self, task, num_classes, average='macro', device='cuda:0'):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.average = average
        self.metrics = MetricCollection({
            'precision': Precision(task=task, num_classes=num_classes, average=average).to(device),
            'recall': Recall(task=task, num_classes=num_classes, average=average).to(device),
        })


    def _move_to_device(self, trainer):
        self.metrics.to(trainer.device)


    def on_training_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)


    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)


    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        # gt labels: [ 0, class id, normalized cxcywh ]
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

        # No predictions and no ground truths
        # This would mean updating TN which are irrelevant for recall and precision
        if preds.shape[0] == 0 and gt_boxes.shape[0] == 0:
            return

        # Any prediction made when no gt boxes are present is a false positive
        if gt_boxes.shape[0] == 0:
            metric_input_gt = torch.zeros(preds.shape[0])
            metric_input_preds = torch.ones(preds.shape[0])
            self.metrics.update(metric_input_preds, metric_input_gt)
            return

        # No predictions made when gt boxes are present is a false negative
        if preds.shape[0] == 0:
            metric_input_gt = torch.ones(gt_boxes.shape[0])
            metric_input_preds = torch.zeros(gt_boxes.shape[0])
            self.metrics.update(metric_input_preds, metric_input_gt)
            return

        # If no edge cases
        # Prepare IoU matrix between gt boxes and preds
        iou_matrix = torchvision.ops.box_iou(gt_boxes, preds[:, :4])

        # https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/src/evaluators/tube_evaluator.py#L132
        results_dim = min(gt_boxes.shape[0], preds.shape[0])
        recorded_matches = torch.empty(results_dim, 2)
        for i in range(results_dim):
            # Find best iou for each gt
            best_match_iou_vec, best_match_pred_idx_vec  = iou_matrix.max(dim=1)

            # Of these find which gt is best matched
            best_match_iou_scalar = best_match_iou_vec.max()
            best_match_gt_idx = best_match_iou_vec.argmax()
            best_match_pred_idx = best_match_pred_idx_vec[best_match_gt_idx]

            assert best_match_iou_scalar == iou_matrix[best_match_gt_idx, best_match_pred_idx]

            # Record the match
            recorded_matches[i, :] = torch.tensor([
                1.0, # Ground truth
                best_match_iou_scalar.ceil() # On hit =1 on miss =0
            ])

            # Set the matched gt and pred to -1 so that they are not matched again
            iou_matrix[best_match_gt_idx, :] = -1
            iou_matrix[:, best_match_pred_idx] = -1
        recorded_matches, _ = torch.sort(recorded_matches, dim=0)

        if preds.shape[0] > gt_boxes.shape[0]:
            padding_size = preds.shape[0] - gt_boxes.shape[0]
            padding = torch.hstack([
                torch.zeros(padding_size, 1),
                torch.ones(padding_size, 1)
            ])
            recorded_matches = torch.vstack([
                recorded_matches,
                padding
            ])

        elif preds.shape[0] < gt_boxes.shape[0]:
            padding_size = gt_boxes.shape[0] - preds.shape[0]
            padding = torch.hstack([
                torch.ones(padding_size, 1),
                torch.zeros(padding_size, 1)
            ])
            recorded_matches = torch.vstack([
                recorded_matches,
                padding
            ])

        metric_input_gt = recorded_matches[:, 0]
        metric_input_preds = recorded_matches[:, 1]
        self.metrics.update(metric_input_preds, metric_input_gt)


    def on_eval_epoch_end(self, trainer, **kwargs):
        metrics = self.metrics.compute()
        trainer.run_history.update_metric('precision', metrics['precision'].cpu())
        trainer.run_history.update_metric('recall', metrics['recall'].cpu())
        self.metrics.reset()


class PrecisionRecallCurveMetricsCallback(TrainerCallback):
    def __init__(self, task, num_classes, average='macro', device='cuda:0'):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.average = average
        self.metric = PrecisionRecallCurve(task=task, num_classes=num_classes, average=average).to(device)

    def _move_to_device(self, trainer):
        self.metrics.to(trainer.device)


    def on_training_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)


    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)


    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        # gt labels: [ 0, class id, normalized cxcywh ]
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

        # No predictions and no ground truths
        # This would mean updating TN which are irrelevant for recall and precision
        if preds.shape[0] == 0 and gt_boxes.shape[0] == 0:
            return

        # Any prediction made when no gt boxes are present is a false positive
        if gt_boxes.shape[0] == 0:
            metric_input_gt = torch.zeros(preds.shape[0], dtype=torch.int)
            metric_input_preds = preds[:, 4]
            self.metrics.update(metric_input_preds, metric_input_gt)
            return

        # No predictions made when gt boxes are present is a false negative
        if preds.shape[0] == 0:
            metric_input_gt = torch.ones(gt_boxes.shape[0], dtype=torch.int)
            metric_input_preds = torch.zeros(gt_boxes.shape[0])
            self.metrics.update(metric_input_preds, metric_input_gt)
            return

        iou_matrix = torchvision.ops.box_iou(gt_boxes, preds[:, :4])

        results_dim = min(gt_boxes.shape[0], preds.shape[0])
        recorded_matches = torch.empty(results_dim, 2)
        preds_match_idices = []
        for i in range(results_dim):
            # Find best iou for each gt
            best_match_iou_vec, best_match_pred_idx_vec  = iou_matrix.max(dim=1)

            # Of these find which gt is best matched
            best_match_iou_scalar = best_match_iou_vec.max()
            best_match_gt_idx = best_match_iou_vec.argmax()
            best_match_pred_idx = best_match_pred_idx_vec[best_match_gt_idx]

            assert best_match_iou_scalar == iou_matrix[best_match_gt_idx, best_match_pred_idx]

            # In precision recall callback, iou is ceiled so if overlap is zero, the score is zero = no match
            # Here it has to be adjusted, because max overlap could be zero, but confidence score could be high
            confidence_score = preds[best_match_pred_idx, 4] if best_match_iou_scalar > 0 else 0

            # Record the match
            recorded_matches[i, :] = torch.tensor([
                1.0, # Ground truth
                confidence_score
            ])

            # Record pred index of a match
            preds_match_idices.append(best_match_pred_idx)

            # Set the matched gt and pred to -1 so that they are not matched again
            iou_matrix[best_match_gt_idx, :] = -1
            iou_matrix[:, best_match_pred_idx] = -1
        recorded_matches, _ = torch.sort(recorded_matches, dim=0)

        # Assert that there are no duplicate pred indices
        assert len(preds_match_idices) == len(set(preds_match_idices))

        if preds.shape[0] > gt_boxes.shape[0]:
            padding_size = preds.shape[0] - gt_boxes.shape[0]
            mask = torch.ones(preds.shape[0], dtype=torch.bool)
            mask[preds_match_idices] = False
            preds_without_matches = preds[mask, :]
            confidence_scores_padding = preds_without_matches[:, 4]
            assert confidence_scores_padding.shape[0] == padding_size
            padding = torch.hstack([
                torch.zeros(padding_size, 1, device='cuda:0'),
                confidence_scores_padding.unsqueeze(1)
            ])
            recorded_matches = torch.vstack([
                recorded_matches.to(trainer.device),
                padding
            ])

        elif preds.shape[0] < gt_boxes.shape[0]:
            padding_size = gt_boxes.shape[0] - preds.shape[0]
            padding = torch.hstack([
                torch.ones(padding_size, 1),
                torch.zeros(padding_size, 1)
            ])
            recorded_matches = torch.vstack([
                recorded_matches,
                padding
            ])
        metric_input_gt = recorded_matches[:, 0].type(torch.int)
        metric_input_preds = recorded_matches[:, 1]
        self.metric.update(metric_input_preds, metric_input_gt)


    def on_eval_epoch_end(self, trainer, **kwargs):
        pr_curve_precision, pr_curve_recall, pr_curve_thresholds = self.metric.compute()
        trainer.run_history.update_metric('pr_curve_precision', pr_curve_precision.cpu())
        trainer.run_history.update_metric('pr_curve_recall', pr_curve_recall.cpu())
        trainer.run_history.update_metric('pr_curve_thresholds', pr_curve_thresholds.cpu())
        self.metrics.reset()