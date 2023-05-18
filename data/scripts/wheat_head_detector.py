from typing import Any, Sequence, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

class WheatHeadDetector(pl.LightningModule):
    def __init__(self) -> None:
        self.network = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        num_classes = 2
        # cls_score is a linear layer calculatin class scores (logits)
        # heads are the layers that are attached to the backbone
        # backbone is the feature extractor (pretrained on coco or imagenet)
        # and heads perform actual task (here detection)
        in_features = self.network.roi_heads.box_predictor.cls_score.in_features

        # Alternatively look for implementation example in torchvision.models.detection.faster_rcnn

        # Replace the pre-trained head with a new one suitable for application
        self.network.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x: Any) -> Any:
        pass

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pass

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        return super().configure_callbacks()