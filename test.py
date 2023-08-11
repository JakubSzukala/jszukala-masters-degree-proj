import argparse
from functools import partial
import os

from yolov7.trainer import Yolov7Trainer
from yolov7 import create_yolov7_model
from yolov7.loss_factory import create_yolov7_loss
from yolov7.trainer import filter_eval_predictions
from yolov7.dataset import Yolov7Dataset
from yolov7.dataset import yolov7_collate_fn
from data.adapter import GwhdToYoloAdapter, load_gwhd_df
from model.augmentations import get_gwhd_test_augmentations
import numpy as np
from pytorch_accelerated.callbacks import get_default_callbacks

from model.callbacks import MeanAveragePrecisionCallback

def test_yolov7(**kwargs):
    model = create_yolov7_model(kwargs['model'] , num_classes=1, pretrained=False)

    test_df = load_gwhd_df(kwargs['dataset'])
    test_augmentations = get_gwhd_test_augmentations(kwargs['image_size'], kwargs['image_size'])
    test_adapter = GwhdToYoloAdapter(kwargs['data_dir'], test_df, test_augmentations)
    yolo_test_ds = Yolov7Dataset(test_adapter)

    trainer = Yolov7Trainer(
        model=model,
        optimizer=None,
        loss_func=create_yolov7_loss(model, image_size=kwargs['image_size']),
        filter_eval_predictions_fn=partial(
            filter_eval_predictions, confidence_threshold=kwargs['confidence_th'], nms_threshold=kwargs['nms_th']
        ),
        callbacks=[
            #MeanAveragePrecisionCallback([0.5]),
            MeanAveragePrecisionCallback(np.linspace(0.5, 0.75, 6).tolist()),
            *get_default_callbacks(progress_bar=True)
        ]
    )
    trainer.load_checkpoint(kwargs['weights'], load_optimizer=False)

    trainer.evaluate(
        dataset=yolo_test_ds,
        per_device_batch_size=2,
        collate_fn=yolov7_collate_fn
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Yolov7 model name')
    parser.add_argument('--weights', type=str, required=True, help='Path to weights .pt file')
    parser.add_argument('--nms_th', type=float, required=False, default=0.45, help='NMS threshold')
    parser.add_argument('--confidence_th', type=float, required=False, default=0.15, help='Confidence threshold for predictions')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset descriptor file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--image_size', type=int, required=False, default=640, help='Image size')
    parser.add_argument('--batch_size', type=int, required=False, default=2, help='Batch size')

    cli_args = parser.parse_args()
    print(cli_args)
    test_yolov7(**vars(cli_args))
