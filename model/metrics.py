from typing import Any
import torch
import torchvision
from torchmetrics import Metric, Precision


def get_binary_stat_scores(preds, targets, thresholds):
    """
    Calculates true positives, false positives, false negatives and true negatives
    for binary classification task. Each of the scores will have the shape [N, 1]
    where N is the number of thresholds.

    Args:
        preds (torch.Tensor): Predictions of shape [N, 1]
        targets (torch.Tensor): Targets of shape [N, 1]
        thresholds (torch.Tensor): Thresholds to use for predictions

    Returns:
        tp (torch.Tensor): True positives
        fp (torch.Tensor): False positives
        fn (torch.Tensor): False negatives
        tn (torch.Tensor): True negatives
        thresholds (torch.Tensor): Thresholds used for predictions
    """

    assert preds.shape == targets.shape

    # Prepare matrix where columns are targets and preds and
    # these are stacked in dim=0 thresholds_n times
    matrix = torch.hstack([targets, preds])
    matrix = torch.stack([matrix] * len(thresholds), dim=0)

    matrix = matrix >= thresholds.reshape(-1, 1, 1)

    #      TARGETS            PREDICTIONS         TARGETS
    tp = ((matrix[:, :, 0] == matrix[:, :, 1]) & (matrix[:, :, 0] == 1)).sum(dim=1).reshape(-1, 1)
    fp = ((matrix[:, :, 0] != matrix[:, :, 1]) & (matrix[:, :, 0] == 0)).sum(dim=1).reshape(-1, 1)
    fn = ((matrix[:, :, 0] != matrix[:, :, 1]) & (matrix[:, :, 0] == 1)).sum(dim=1).reshape(-1, 1)
    tn = ((matrix[:, :, 0] == matrix[:, :, 1]) & (matrix[:, :, 0] == 0)).sum(dim=1).reshape(-1, 1)

    return tp, fp, fn, tn, thresholds


class PrecisionCurve(Metric):
    def __init__(self, thresholds=None):
        super().__init__()
        self.thresholds = thresholds

        # Default for concatenation to work, equal to TN which does not
        # affect precision
        self.add_state(
            "preds",
            default=torch.zeros([1, 1], device=self.device),
        )
        self.add_state(
            "targets",
            default=torch.zeros([1, 1], device=self.device),
        )


    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape
        self.preds   = torch.cat([self.preds, preds.reshape(-1, 1)])
        self.targets = torch.cat([self.targets, targets.reshape(-1, 1)])


    def compute(self):
        if self.thresholds is None:
            self.thresholds = torch.sort(torch.unique(self.preds)).values

        tp, fp, _, _, thresholds = get_binary_stat_scores(self.preds, self.targets, self.thresholds)
        precisions = tp / (tp + fp)
        precisions = torch.nan_to_num(precisions)

        return precisions.squeeze(), thresholds.squeeze()


class RecallCurve(Metric):
    def __init__(self, thresholds=None):
        super().__init__()
        self.thresholds = thresholds

        # Default for concatenation to work, equal to TN which does not
        # affect precision
        self.add_state(
            "preds",
            default=torch.zeros([1, 1], device=self.device),
        )
        self.add_state(
            "targets",
            default=torch.zeros([1, 1], device=self.device),
        )


    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape
        self.preds   = torch.cat([self.preds, preds.reshape(-1, 1)])
        self.targets = torch.cat([self.targets, targets.reshape(-1, 1)])


    def compute(self):
        if self.thresholds is None:
            self.thresholds = torch.sort(torch.unique(self.preds)).values

        tp, _, fn, _, thresholds = get_binary_stat_scores(self.preds, self.targets, self.thresholds)
        recalls = tp / (tp + fn)
        recalls = torch.nan_to_num(recalls)

        return recalls.squeeze(), thresholds.squeeze()


class F1Curve(Metric):
    def __init__(self, thresholds=None):
        super().__init__()
        self.thresholds = thresholds

        # Default for concatenation to work, equal to TN which does not
        # affect precision
        self.add_state(
            "preds",
            default=torch.zeros([1, 1], device=self.device),
        )
        self.add_state(
            "targets",
            default=torch.zeros([1, 1], device=self.device),
        )


    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape
        self.preds   = torch.cat([self.preds, preds.reshape(-1, 1)])
        self.targets = torch.cat([self.targets, targets.reshape(-1, 1)])


    def compute(self):
        if self.thresholds is None:
            self.thresholds = torch.sort(torch.unique(self.preds)).values

        tp, fp, fn, _, thresholds = get_binary_stat_scores(self.preds, self.targets, self.thresholds)
        precisions = tp / (tp + fp)
        recalls = tp / (tp + fn)
        f1s = 2 * (precisions * recalls) / (precisions + recalls)
        f1s = torch.nan_to_num(f1s)

        return f1s.squeeze(), thresholds.squeeze()