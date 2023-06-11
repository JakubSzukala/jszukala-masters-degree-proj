#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import os
import yaml
import argparse

from data.adapter import load_gwhd_df
from data.adapter import GwhdToYoloAdapter

from yolov7.dataset import Yolov7Dataset
from yolov7.plotting import show_image
from yolov7 import create_yolov7_model
from yolov7.loss_factory import create_yolov7_loss
from yolov7.dataset import yolov7_collate_fn
from yolov7.trainer import Yolov7Trainer
from yolov7.trainer import filter_eval_predictions
from yolov7.evaluation import CalculateMeanAveragePrecisionCallback

from pytorch_accelerated.schedulers import CosineLrScheduler
from pytorch_accelerated.callbacks import (
    EarlyStoppingCallback,
    SaveBestModelCallback,
    get_default_callbacks
)

from functools import partial

from model.metrics import PrecisionRecallMetricsCallback, PrecisionRecallCurveMetricsCallback

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')

hyperparameters_config_file = parser.parse_args().config
with open(hyperparameters_config_file) as f:
    config = yaml.safe_load(f)

DATASET_ROOT_DIR = config['dataset']['path']
print(f"Loading dataset from: {DATASET_ROOT_DIR}")
train_subset = os.path.join(config['dataset']['path'], config['dataset']['subsets']['train'])
test_subset = os.path.join(config['dataset']['path'], config['dataset']['subsets']['test'])
val_subset = os.path.join(config['dataset']['path'], config['dataset']['subsets']['val'])

# Load pandas frames describing datasets
train_df = load_gwhd_df(os.path.join(DATASET_ROOT_DIR, train_subset))
test_df = load_gwhd_df(os.path.join(DATASET_ROOT_DIR, test_subset))
val_df = load_gwhd_df(os.path.join(DATASET_ROOT_DIR, val_subset))


# Instantiate adapters providing interface between df descriptors and yolov7 dataset
images_dir = os.path.join(config['dataset']['path'], config['dataset']['images_dir'])
train_adapter = GwhdToYoloAdapter(images_dir, train_df, None)
test_adapter = GwhdToYoloAdapter(images_dir, test_df, None)
val_adapter = GwhdToYoloAdapter(images_dir, val_df, None)

# Instantiate yolov7 datasets
yolo_train_ds = Yolov7Dataset(train_adapter)
yolo_test_ds = Yolov7Dataset(test_adapter)
yolo_val_ds = Yolov7Dataset(val_adapter)

# Test if fetching image works properly
image_tensor, labels, image_id, image_size = yolo_train_ds[0]

# Denormalize boxes
boxes = labels[:, 2:]
boxes[:, [0, 2]] *= image_size[1]
boxes[:, [1, 3]] *= image_size[0]

#show_image(image_tensor.permute(1, 2, 0), boxes.tolist(), None, 'cxcywh')
model_name = config['model_name']
model = create_yolov7_model(model_name, num_classes=1, pretrained=True)
loss_func = create_yolov7_loss(model, image_size=image_size[0])

if config['optimizer']['name'] == 'adam':
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['optimizer']['lr'],
    )

elif config['optimizer']['name'] == 'sgd':
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['optimizer']['lr'],
        momentum=config['optimizer']['momentum'],
        nesterov=config['optimizer']['nesterov']
    )
else:
    raise ValueError('Optimizer not supported')


confidence_threshold = config['trainer_params']['filter_eval_predictions_fn_params']['confidence_threshold']
nms_threshold = config['trainer_params']['filter_eval_predictions_fn_params']['nms_threshold']
iou_threshold = config['trainer_params']['mean_average_precision_callback_params']['iou_threshold']
patience = config['trainer_params']['early_stopping_callback_params']['patience']
watch_metric_es = config['trainer_params']['early_stopping_callback_params']['watch_metric']
greater_is_better_es = config['trainer_params']['early_stopping_callback_params']['greater_is_better']
early_stopping_threshold = config['trainer_params']['early_stopping_callback_params']['early_stopping_threshold']
watch_metric_sbm = config['trainer_params']['save_best_model_callback_params']['watch_metric']
greater_is_better_sbm = config['trainer_params']['save_best_model_callback_params']['greater_is_better']

# Fix random seed
torch.manual_seed(0)
np.random.seed(0)

# Create trainer and train
trainer = Yolov7Trainer(
    model=model,
    optimizer=optimizer,
    loss_func=loss_func,
    filter_eval_predictions_fn=partial(
        filter_eval_predictions, confidence_threshold=confidence_threshold, nms_threshold=nms_threshold
    ),
    callbacks=[
        CalculateMeanAveragePrecisionCallback.create_from_targets_df(
            targets_df=val_df.query("has_annotation == True")[
                ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
            ],
            image_ids=set(val_df.image_id.unique()),
            iou_threshold=iou_threshold,
        ),
        SaveBestModelCallback(
            watch_metric=watch_metric_sbm,
            greater_is_better=greater_is_better_sbm
        ),
        #EarlyStoppingCallback(
            #early_stopping_patience=patience,
            #watch_metric=watch_metric_es,
            #greater_is_better=greater_is_better_es,
            #early_stopping_threshold=early_stopping_threshold,
        #),
        PrecisionRecallMetricsCallback(
            task='binary',
            num_classes=1,
            average='macro'
        ),
        PrecisionRecallCurveMetricsCallback(
            task='binary',
            num_classes=1,
            average='macro'
        ),
        *get_default_callbacks(progress_bar=True),
    ],
)

num_epochs = config['training_params']['num_epochs']
batch_size = config['training_params']['per_device_batch_size']
num_warmup_epochs = config['training_params']['cosine_lr_scheduler_params']['num_warmup_epochs']
num_cooldown_epochs = config['training_params']['cosine_lr_scheduler_params']['num_cooldown_epochs']
k_decay = config['training_params']['cosine_lr_scheduler_params']['k_decay']

trainer.train(
        num_epochs=num_epochs,
        train_dataset=yolo_train_ds,
        eval_dataset=yolo_val_ds,
        per_device_batch_size=batch_size,
        create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(
            num_warmup_epochs=num_warmup_epochs,
            num_cooldown_epochs=num_cooldown_epochs,
            k_decay=k_decay,
        ),
        collate_fn=yolov7_collate_fn,
)

trainer.evaluate(
    dataset=yolo_val_ds,
    per_device_batch_size=batch_size,
    collate_fn=yolov7_collate_fn
)