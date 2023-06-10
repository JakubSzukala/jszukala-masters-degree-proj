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
            #'pr_curve': PrecisionRecallCurve(task=task, num_classes=num_classes, average=average)
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
        gt_boxes = ground_truth_labels[:, 2:]
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
        matched_boxes = iou_matrix.max(dim=1).values.ceil()

        # Note: matched_boxes.shape[0] == gt_boxes.shape[0]
        if matched_boxes.shape[0] < preds.shape[0]:
            zero_padding_len = preds.shape[0] - matched_boxes.shape[0]
            metric_input_gt = torch.hstack([
                torch.ones(gt_boxes.shape[0]).to(trainer.device),
                torch.zeros(zero_padding_len).to(trainer.device)
            ])
            ones_padding_len = preds.shape[0] - matched_boxes.shape[0]
            metric_input_preds = torch.hstack([
                matched_boxes,
                torch.ones(ones_padding_len).to(trainer.device)
            ])
        # No padding needed
        else:
            metric_input_gt = torch.ones(gt_boxes.shape[0]).to(trainer.device)
            metric_input_preds = matched_boxes

        self.metrics.update(metric_input_preds, metric_input_gt)


    def on_eval_epoch_end(self, trainer, **kwargs):
        metrics = self.metrics.compute()
        # Here we can record metrics and store them in run history object.
        # They will also be printed if PrintProgressCallback is used (it is used)
        trainer.run_history.update_metric('precision', metrics['precision'].cpu())
        trainer.run_history.update_metric('recall', metrics['recall'].cpu())
        #trainer.run_history.update_metric('pr_curve', metrics['pr_curve'].cpu())

        self.metrics.reset()