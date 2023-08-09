#!/usr/bin/env python
# coding: utf-8

import sys

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
import argparse
import datetime
import shutil

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

from model.callbacks import (
    BinaryPrecisionRecallMetricsCallback,
    MeanAveragePrecisionCallback,
    DetectionLossTrackerCallback,
    TensorboardLoggingCallback
)
from model.augmentations import get_gwhd_train_augmentations, get_gwhd_test_augmentations, get_gwhd_val_augmentations


def create_log_directory(log_dir):
    """
    Function that creates log directory if it does not exist and encodes it with current date and time.

    Args:
        log_dir (str): Path to log directory
    """
    current_datetime = datetime.datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(log_dir):
        time_encoded_log_dir = log_dir + current_datetime_str
        os.makedirs(time_encoded_log_dir)
        return time_encoded_log_dir
    else:
        raise Exception("Log directory already exists!")


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
img_w = config['model']['required_img_width']
img_h = config['model']['required_img_height']
train_augmentations = get_gwhd_train_augmentations(img_w, img_h)
val_augmentations = get_gwhd_train_augmentations(img_w, img_h)
test_augmentations = get_gwhd_train_augmentations(img_w, img_h)
train_adapter = GwhdToYoloAdapter(images_dir, train_df, train_augmentations)
test_adapter = GwhdToYoloAdapter(images_dir, test_df, test_augmentations)
val_adapter = GwhdToYoloAdapter(images_dir, val_df, val_augmentations)

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
model_name = config['model']['model_name']
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
best_model_name = 'best_model_' + watch_metric_sbm + '.pt'

# Fix random seed
torch.manual_seed(0)
np.random.seed(0)

# Create unique per run time encoded log directory
time_encoded_log_dir = create_log_directory(config['log_dir'])

# Save config used for training under log directory
shutil.copy(hyperparameters_config_file, time_encoded_log_dir)

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
            greater_is_better=greater_is_better_sbm,
            save_path=os.path.join(time_encoded_log_dir, best_model_name)
        ),
        #EarlyStoppingCallback(
            #early_stopping_patience=patience,
            #watch_metric=watch_metric_es,
            #greater_is_better=greater_is_better_es,
            #early_stopping_threshold=early_stopping_threshold,
        #),
        BinaryPrecisionRecallMetricsCallback(
            confidence_threshold=confidence_threshold
        ),
        #MeanAveragePrecisionCallback(np.linspace(0.5, 0.95, 10).tolist()),
        #MeanAveragePrecisionCallback([0.5]),
        DetectionLossTrackerCallback(),
        TensorboardLoggingCallback(time_encoded_log_dir),
        *get_default_callbacks(progress_bar=True)
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

# Retrieve metrics
pr_curve_precision = trainer.run_history.get_latest_metric('pr_curve_precision')
pr_curve_recall = trainer.run_history.get_latest_metric('pr_curve_recall')
pr_curve_thresholds = trainer.run_history.get_latest_metric('pr_curve_thresholds')
f1_curve = trainer.run_history.get_latest_metric('f1_curve')
confusion_matrix = trainer.run_history.get_latest_metric('confusion_matrix')

# TODO: Clean it up somehow, move it to separate function or do it in callback
# Save these metrics to csv files
save_to_csv = lambda data, filepath: np.savetxt(filepath, data.numpy(), delimiter=',')
save_to_csv(pr_curve_precision, os.path.join(time_encoded_log_dir, 'pr_curve_precision.csv'))
save_to_csv(pr_curve_recall, os.path.join(time_encoded_log_dir, 'pr_curve_recall.csv'))
save_to_csv(pr_curve_thresholds, os.path.join(time_encoded_log_dir, 'pr_curve_thresholds.csv'))
save_to_csv(f1_curve, os.path.join(time_encoded_log_dir, 'f1_curve.csv'))
save_to_csv(confusion_matrix, os.path.join(time_encoded_log_dir, 'confusion_matrix.csv'))

writer = SummaryWriter(time_encoded_log_dir)

fig = plt.figure(1)
plt.title('Precision-Recall curve')
plt.plot(pr_curve_recall, pr_curve_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig(os.path.join(time_encoded_log_dir, 'pr_curve.png'))
writer.add_figure('Precision-Recall curve', fig)

fig = plt.figure(2)
plt.title('Confidence Thresholds')
plt.plot(pr_curve_thresholds.flip(0)) # TODO: Why flip?
plt.xlabel('Index')
plt.ylabel('Confidence')
plt.savefig(os.path.join(time_encoded_log_dir, 'confidence_thresholds.png'))
writer.add_figure('Confidence Thresholds', fig)

fig = plt.figure(3)
plt.title('Precision curve')
plt.plot(pr_curve_thresholds, pr_curve_precision)
plt.xlabel('Confidence')
plt.ylabel('Precision')
plt.savefig(os.path.join(time_encoded_log_dir, 'precision_curve.png'))
writer.add_figure('Precision curve', fig)

fig = plt.figure(4)
plt.title('Recall curve')
plt.plot(pr_curve_thresholds, pr_curve_recall)
plt.xlabel('Confidence')
plt.ylabel('Recall')
plt.savefig(os.path.join(time_encoded_log_dir, 'recall_curve.png'))
writer.add_figure('Recall curve', fig)

fig = plt.figure(5)
plt.title('F1 curve')
plt.plot(pr_curve_thresholds, f1_curve)
plt.xlabel('Confidence')
plt.ylabel('F1')
plt.savefig(os.path.join(time_encoded_log_dir, 'f1_curve.png'))
writer.add_figure('F1 curve', fig)

fig = plt.figure(6)
plt.title('Confusion matrix')
plt.imshow(confusion_matrix)
plt.xlabel('Predicted') # TODO: Check if these axes are correctly subtitled
plt.ylabel('Actual')
plt.savefig(os.path.join(time_encoded_log_dir, 'confusion_matrix.png'))
writer.add_figure('Confusion matrix', fig)

#plt.show()