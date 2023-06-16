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

import numpy as np

def _yolo_to_xyxy(boxes, image_sizes):
    def cxcywh_to_xyxy(box):
        x, y, w, h = box
        return torch.tensor([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

    # Denormalize
    boxes[:, [0, 2]] *= image_sizes[0, 1]
    boxes[:, [1, 3]] *= image_sizes[0, 0]

    # Convert from denormalized yolo format to xyxy
    for i, row in enumerate(boxes):
        boxes[i, :] = cxcywh_to_xyxy(row)
    return boxes


def _detection_results_to_classification_results(gt, preds, device):
    """
    Function that takes ground truths and preds, from detection model,
    matches most likely prediction with ground truth and returns
    tensors of ground truths and confidence scores of matches.

    Args:
        gt (torch.Tensor): Ground truths [ xyxy ]
        preds (torch.Tensor): Predictions [ xyxy, score ]
        device (str): Device to move tensors to
    Returns:
        ground_truths (torch.Tensor): Ground truths
        predictions (torch.Tensor): Confidence scores of matches
    """
    # No predictions and no ground truths
    # This would mean updating TN which are irrelevant for recall and precision
    if preds.shape[0] == 0 and gt.shape[0] == 0:
        ground_truths = torch.zeros([1, 1], dtype=torch.int, device=device)
        predictions = torch.zeros([1, 1], device=device)
        return ground_truths, predictions

    # Any prediction made when no gt boxes are present is a false positive
    if gt.shape[0] == 0:
        ground_truths = torch.zeros(preds.shape[0], dtype=torch.int, device=device)
        predictions = preds[:, 4]
        return ground_truths, predictions

    # No predictions made when gt boxes are present is a false negative
    if preds.shape[0] == 0:
        ground_truths = torch.ones(gt.shape[0], dtype=torch.int, device=device)
        predictions = torch.zeros(gt.shape[0], device=device)
        return ground_truths, predictions

    iou_matrix = torchvision.ops.box_iou(gt, preds[:, :4])

    results_dim = min(gt.shape[0], preds.shape[0])
    recorded_matches = torch.empty(results_dim, 2, device=device)
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
        ], device=device)

        # Record pred index of a match
        preds_match_idices.append(best_match_pred_idx)

        # Set the matched gt and pred to -1 so that they are not matched again
        iou_matrix[best_match_gt_idx, :] = -1
        iou_matrix[:, best_match_pred_idx] = -1

    # Assert that there are no duplicate pred indices
    assert len(preds_match_idices) == len(set(preds_match_idices))

    if preds.shape[0] > gt.shape[0]:
        padding_size = preds.shape[0] - gt.shape[0]
        mask = torch.ones(preds.shape[0], dtype=torch.bool, device=device)
        mask[preds_match_idices] = False
        preds_without_matches = preds[mask, :]
        confidence_scores_padding = preds_without_matches[:, 4]
        assert confidence_scores_padding.shape[0] == padding_size
        padding = torch.hstack([
            torch.zeros(padding_size, 1, device=device),
            confidence_scores_padding.unsqueeze(1)
        ])
        recorded_matches = torch.vstack([
            recorded_matches,
            padding
        ])

    elif preds.shape[0] < gt.shape[0]:
        padding_size = gt.shape[0] - preds.shape[0]
        padding = torch.hstack([
            torch.ones(padding_size, 1, device=device),
            torch.zeros(padding_size, 1, device=device)
        ])
        recorded_matches = torch.vstack([
            recorded_matches,
            padding
        ])
    ground_truths = recorded_matches[:, 0].type(torch.int)
    predictions = recorded_matches[:, 1]
    return ground_truths, predictions





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
            single_image_gt_boxes = _yolo_to_xyxy(single_image_gt_boxes, original_image_sizes)

            # Convert them to classification results for classification metrics calculation
            classification_gt, classification_preds = _detection_results_to_classification_results(
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
            single_image_gt_boxes = _yolo_to_xyxy(single_image_gt_boxes, original_image_sizes)

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
