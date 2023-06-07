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
    df[['x_min', 'y_min', 'x_max', 'y_max']] = df['Box'].str.split(' ', expand=True)
    df['has_annotation'] = df.loc[:, 'Box'] != 'no_box'
    df.loc[df.loc[:, 'Box'] == 'no_box', ['Box', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id', 'class_name']] = \
        [np.nan for _ in range(6)] + ['background']

    return df

