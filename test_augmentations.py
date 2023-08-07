
import os
import yaml
import argparse

from data.adapter import GwhdToYoloAdapter, load_gwhd_df

from yolov7.dataset import Yolov7Dataset
from yolov7.plotting import show_image

import albumentations as A
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')

hyperparameters_config_file = parser.parse_args().config
with open(hyperparameters_config_file) as f:
    config = yaml.safe_load(f)

DATASET_ROOT_DIR = config['dataset']['path']
YOLO_PAD_VALUE = [114, 114, 114] # Average value from ImageNet
GWHD_PAD_VALUE = [82, 82, 54] # Average values per channel from GWHD dataset

train_subset = os.path.join(DATASET_ROOT_DIR, 'competition_train.csv')
train_df = load_gwhd_df(os.path.join(DATASET_ROOT_DIR, train_subset))
print(train_df.head())
images_dir = os.path.join(config['dataset']['path'], config['dataset']['images_dir'])

transforms = A.Compose(
    [
        A.RandomScale(scale_limit=[-0.5, 0.2], interpolation=cv2.INTER_CUBIC, p=0.5),
        A.Flip(p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.9),
            A.GaussNoise(p=0.1),
        ], p=0.6),
        A.RandomShadow(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.FancyPCA(alpha=1),
        A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=GWHD_PAD_VALUE),
        A.RandomCrop(width=1024, height=1024),
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=0, min_visibility=0)
)

train_adapter = GwhdToYoloAdapter(images_dir, train_df, transforms)
yolo_train_ds = Yolov7Dataset(train_adapter)

for i in range(len(yolo_train_ds)):
    image_tensor, labels, image_id, image_size = yolo_train_ds[i]
    print(f"Image id: {image_id}")
    boxes = labels[:, 2:]
    boxes[:, [0, 2]] *= image_size[1]
    boxes[:, [1, 3]] *= image_size[0]

    show_image(image_tensor.permute(1, 2, 0), boxes.tolist(), None, 'cxcywh')