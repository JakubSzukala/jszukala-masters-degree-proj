import torch
import torchvision


def yolo_to_xyxy(boxes, image_sizes):
    """
    Function that takes yolo format boxes and image sizes and returns
    xyxy format boxes.

    Note: this function modifies the boxes tensor.
    """
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


def detection_results_to_classification_results(gt, preds, device):
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