
import os
import yaml
import argparse

from data.adapter import GwhdToYoloAdapter, load_gwhd_df

from yolov7.dataset import Yolov7Dataset
from yolov7.plotting import show_image

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')

hyperparameters_config_file = parser.parse_args().config
with open(hyperparameters_config_file) as f:
    config = yaml.safe_load(f)

DATASET_ROOT_DIR = config['dataset']['path']

train_subset = os.path.join(DATASET_ROOT_DIR, 'competition_train.csv')
train_df = load_gwhd_df(os.path.join(DATASET_ROOT_DIR, train_subset))
images_dir = os.path.join(config['dataset']['path'], config['dataset']['images_dir'])
train_adapter = GwhdToYoloAdapter(images_dir, train_df, None)
yolo_train_ds = Yolov7Dataset(train_adapter)

for i in range(len(yolo_train_ds)):
    image_tensor, labels, image_id, image_size = yolo_train_ds[i]
    boxes = labels[:, 2:]
    boxes[:, [0, 2]] *= image_size[1]
    boxes[:, [1, 3]] *= image_size[0]

    show_image(image_tensor.permute(1, 2, 0), boxes.tolist(), None, 'cxcywh')