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
        print(f"preds example: {preds[:5]}")

        # Denormalize and convert ncxncywh to xyxy
        gt_boxes = ground_truth_labels[:, 2:].clone()
        gt_boxes[:, [0, 2]] *= original_image_sizes[0, 1]
        gt_boxes[:, [1, 3]] *= original_image_sizes[0, 0]
        for i, row in enumerate(gt_boxes):
            gt_boxes[i, :] = cxcywh_to_xyxy(row)
        print(f"gt example: {gt_boxes[:5]}")

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
        # Here we can record metrics and store them in run history object.
        # They will also be printed if PrintProgressCallback is used (it is used)
        trainer.run_history.update_metric('precision', metrics['precision'].cpu())
        trainer.run_history.update_metric('recall', metrics['recall'].cpu())
        #trainer.run_history.update_metric('pr_curve', metrics['pr_curve'].cpu())

        self.metrics.reset()


class PrecisionRecallCurveMetricsCallback(TrainerCallback):
    def __init__(self, task, num_classes, average='macro', device='cuda:0'):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.average = average
        self.metrics = MetricCollection({ # TODO Remove collection
            'pr_curve': PrecisionRecallCurve(task=task, num_classes=num_classes, average=average)
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
        print(f"in curve preds example: {preds[:5]}")

        # Denormalize and convert ncxncywh to xyxy
        gt_boxes = ground_truth_labels[:, 2:].clone()
        gt_boxes[:, [0, 2]] *= original_image_sizes[0, 1]
        gt_boxes[:, [1, 3]] *= original_image_sizes[0, 0]
        for i, row in enumerate(gt_boxes):
            gt_boxes[i, :] = cxcywh_to_xyxy(row)
        print(f"in curve gt example: {gt_boxes[:5]}")

        # No predictions and no ground truths
        # This would mean updating TN which are irrelevant for recall and precision
        if preds.shape[0] == 0 and gt_boxes.shape[0] == 0:
            print("Edge case 1")
            return

        # Any prediction made when no gt boxes are present is a false positive
        if gt_boxes.shape[0] == 0:
            print("Edge case 2")
            metric_input_gt = torch.zeros(preds.shape[0])
            metric_input_preds = preds[:, 4]
            self.metrics.update(metric_input_preds, metric_input_gt)
            return

        # No predictions made when gt boxes are present is a false negative
        if preds.shape[0] == 0:
            print("Edge case 3")
            metric_input_gt = torch.ones(gt_boxes.shape[0])
            metric_input_preds = torch.zeros(gt_boxes.shape[0])
            self.metrics.update(metric_input_preds, metric_input_gt)
            return

        print("No edge case")
        # If no edge cases
        # Prepare IoU matrix between gt boxes and preds
        iou_matrix = torchvision.ops.box_iou(gt_boxes, preds[:, :4])
        gt_boxes_with_match = iou_matrix.max(dim=1).values.ceil() # 0 if no match, 1 if match
        pred_idx_of_matched_box = iou_matrix.argmax(dim=1)
        print(f"indexes of matched preds: {pred_idx_of_matched_box} and dups in them {torch.unique(pred_idx_of_matched_box, return_counts=True)}")

        print("Shapes of gt_boxes_with_match and pred_idx_of_matched_box:")
        print(gt_boxes_with_match.shape)
        print(pred_idx_of_matched_box.shape)

        # Compile a tensor, where first col indicates if gt box has a match
        # second col indicates index of matched pred box
        # third col indicates confidence score (NOT IoU!) of matched pred box
        # third col is initialized to zeros and will be updated later.
        # These all are in one tensor as it is safer with all the dim checks.
        matches = torch.hstack([
            gt_boxes_with_match.reshape(-1, 1),
            pred_idx_of_matched_box.reshape(-1, 1),
            torch.zeros(gt_boxes_with_match.shape[0], 1).to(trainer.device)
        ])

        idxs_gt_boxes_with_match = torch.where(matches[:, 0] == 1)
        gt_boxes_with_match_scores = preds[pred_idx_of_matched_box[idxs_gt_boxes_with_match], 4].reshape(-1, 1)
        print(f"match scores shape: {gt_boxes_with_match_scores.shape}")
        matches[idxs_gt_boxes_with_match] = torch.hstack([matches[idxs_gt_boxes_with_match][:, :2], gt_boxes_with_match_scores])
        print(f"matches: {matches}")

        zero_padding_len = preds.shape[0] - gt_boxes_with_match.shape[0]
        mask = torch.ones(preds.shape[0], dtype=torch.bool).to(trainer.device)
        print(f"mask to a mask: {pred_idx_of_matched_box[idxs_gt_boxes_with_match]} and len {pred_idx_of_matched_box[idxs_gt_boxes_with_match].shape[0]}")
        print(f"Duplicates in mask: {torch.unique(pred_idx_of_matched_box[idxs_gt_boxes_with_match], return_counts=True)}")
        mask[pred_idx_of_matched_box[idxs_gt_boxes_with_match]] = False
        unmatched_preds = preds[mask]
        print(f"")
        print(f"mask to a mask: {mask} and len {mask.shape} unique values counts: {torch.unique(mask, return_counts=True)}")
        print(f"unmatched_preds len shape: {unmatched_preds.shape}")
        print(f"matched preds shape: {gt_boxes_with_match_scores.shape}")
        print(f"preds shape: {preds.shape} and gt_boxes shape: {gt_boxes.shape}")
        print(f"idxs_gt_boxes_with_match: {idxs_gt_boxes_with_match[0].shape}")
        print(f"Len of zero padding: {zero_padding_len}")
        #preds_with_no_match = preds[]

        sys.exit(0)
        # Note: matched_boxes.shape[0] == gt_boxes.shape[0]
        if gt_boxes_with_match.shape[0] < preds.shape[0]:
            zero_padding_len = preds.shape[0] - gt_boxes_with_match.shape[0]
            metric_input_gt = torch.hstack([
                torch.ones(gt_boxes.shape[0]).to(trainer.device),
                torch.zeros(zero_padding_len).to(trainer.device)
            ])
            ones_padding_len = preds.shape[0] - gt_boxes_with_match.shape[0]
            metric_input_preds = torch.hstack([
                gt_boxes_with_match,
                torch.ones(ones_padding_len).to(trainer.device)
            ])
        # No padding needed
        else:
            metric_input_gt = torch.ones(gt_boxes.shape[0]).to(trainer.device)
            metric_input_preds = gt_boxes_with_match

        self.metrics.update(metric_input_preds, metric_input_gt)


    def on_eval_epoch_end(self, trainer, **kwargs):
        metrics = self.metrics.compute()
        # Here we can record metrics and store them in run history object.
        # They will also be printed if PrintProgressCallback is used (it is used)
        trainer.run_history.update_metric('precision', metrics['precision'].cpu())
        trainer.run_history.update_metric('recall', metrics['recall'].cpu())
        #trainer.run_history.update_metric('pr_curve', metrics['pr_curve'].cpu())

        self.metrics.reset()