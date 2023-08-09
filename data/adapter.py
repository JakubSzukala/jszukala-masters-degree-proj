import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision

import os
import csv
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Any
from PIL import Image

from data.scripts.data_integrity import calculate_md5_recursive
import numpy as np


def load_gwhd_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['image_id'] = df.index # Group annotations under image id
    df['class_id'] = 0 # Wheat head
    df['class_name'] = 'wheat_head'
    df = df.assign(Box=df['BoxesString'].str.split(';')).explode('Box').reset_index(drop=True)
    df[['xmin', 'ymin', 'xmax', 'ymax']] = df['Box'].str.split(' ', expand=True)
    df['has_annotation'] = df.loc[:, 'Box'] != 'no_box'
    df.loc[df.loc[:, 'Box'] == 'no_box', ['Box', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'class_name']] = \
        [np.nan for _ in range(6)] + ['background']
    df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].astype(float)

    return df

# Source: https://towardsdatascience.com/yolov7-a-deep-dive-into-the-current-state-of-the-art-for-object-detection-ce3ffedeeaeb#0c9c 
class GwhdToYoloAdapter(torch.utils.data.Dataset):
    def __init__(
            self,
            images_dir_path: Path,
            annotations_df: pd.DataFrame,
            transforms: Optional[Callable] = None,
    ):
        self.images_dir_path = images_dir_path
        self.annotations_df = annotations_df
        self.transforms = transforms

        # Abstraction to map image ids to image indices and other way around
        self.image_idx_to_image_id = {
            idx: image_id
            for idx, image_id in enumerate(self.annotations_df.image_id.unique())
        }
        self.image_id_to_image_idx = {
            v: k for k, v, in self.image_idx_to_image_id.items()
        }


    def __len__(self) -> int:
        return len(self.image_idx_to_image_id)


    def __getitem__(self, index):
        # Compile rows representing single image from df into easier to access pieces of information
        image_id = self.image_idx_to_image_id[index]
        image_info = self.annotations_df[self.annotations_df.image_id == image_id]
        file_name = image_info.image_name.values[0]
        assert image_id == image_info.image_id.values[0]

        image = Image.open(os.path.join(self.images_dir_path, file_name)).convert("RGB")
        image = np.array(image)

        if image_info.has_annotation.any():
            xyxy_bboxes = image_info[["xmin", "ymin", "xmax", "ymax"]].values
            class_ids = image_info["class_id"].values
        else:
            xyxy_bboxes = np.array([])
            class_ids = np.array([])

        if self.transforms is not None:
            transformed = self.transforms(
                image=image, bboxes=xyxy_bboxes, labels=class_ids
            )
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        image_hw = image.shape[:2]

        return image, xyxy_bboxes, class_ids, image_id, image_hw