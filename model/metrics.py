import torch
import torchvision

import numpy as np
import tqdm
import time

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


def intersection_over_union(box1, box2):
    ix1 = torch.max(box1[0], box2[0])
    iy1 = torch.max(box1[1], box2[1])
    ix2 = torch.max(box1[2], box2[2])
    iy2 = torch.max(box1[3], box2[3])

    # No intersection
    if ix2 < ix1 or iy2 < iy1:
        return 0.0

    intersection_area = (ix2 - ix1) * (iy2 - iy1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou


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

        # Convert ncxncywh to xyxy
        print(f"Image sizes: {original_image_sizes}")
        gt_boxes = ground_truth_labels[:, 2:]
        gt_boxes[:, [0, 2]] *= original_image_sizes[0, 1]
        gt_boxes[:, [1, 3]] *= original_image_sizes[0, 0]
        for i, row in enumerate(gt_boxes):
            gt_boxes[i, :] = cxcywh_to_xyxy(row)

        print("Compare gt boxes and preds:")
        print(f"Ground truth boxes:\n{gt_boxes[0]}")
        print(f"Predicted boxes:\n{preds[0]}")

        # Prepare IoU matrix between gt boxes and preds
        iou_matrix = torchvision.ops.box_iou(gt_boxes, preds[:, :4])
        matched_boxes = iou_matrix.max(dim=1).values.ceil()

        print(f"iou_matrix shape: {iou_matrix.shape}")
        print(f"iou_matrix:\n{iou_matrix}")
        print(f"Max values in iou_matrix / matched boxes: n={matched_boxes.shape} {matched_boxes}")
        print(f"Preds shape: {preds.shape}, ground truths shape: {gt_boxes.shape}")
        if gt_boxes.shape[0] < preds.shape[0]:
            zero_padding_len = preds.shape[0] - gt_boxes.shape[0]
            metric_input_gt = torch.hstack([
                torch.ones(gt_boxes.shape[0]),
                torch.zeros(zero_padding_len)
            ])
            ones_padding_len = preds.shape[0] - gt_boxes.shape[0]
            metric_input_preds = torch.hstack([
                matched_boxes,
                torch.ones(ones_padding_len).to(trainer.device)
            ])
        else:
            zero_padding_len = gt_boxes.shape[0] - preds.shape[0]
            metric_input_gt = torch.ones(gt_boxes.shape[0])
            metric_input_preds = torch.hstack([
                matched_boxes,
                torch.zeros(zero_padding_len).to(trainer.device)
            ])
        metric_input_gt = metric_input_gt.to(trainer.device)
        print(f"inputs on the devices: {metric_input_gt.device}, {metric_input_preds.device}")
        self.metrics.update(metric_input_preds, metric_input_gt)


    def on_eval_epoch_end(self, trainer, **kwargs):
        metrics = self.metrics.compute()
        # Here we can record metrics and store them in run history object.
        # They will also be printed if PrintProgressCallback is used (it is used)
        trainer.run_history.update_metric('precision', metrics['precision'].cpu())
        trainer.run_history.update_metric('recall', metrics['recall'].cpu())
        #trainer.run_history.update_metric('pr_curve', metrics['pr_curve'].cpu())

        self.metrics.reset()