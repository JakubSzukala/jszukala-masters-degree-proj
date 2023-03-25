import torch
import pytorch_lightning as pl

import os
import csv
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Any
from PIL import Image

from data.data_integrity import calculate_md5_recursive

class WheatHeadsDataset(torch.utils.data.Dataset):
    DATASET_MD5 = '359c66afd88f0983726003ffec4ab466'

    def __init__(
            self,
            data_root: Path,
            subset: str,
            transforms: Optional[Callable] = None,
            target_transforms: Optional[Callable] = None,
            download: bool = True
            ) -> None:
        super().__init__() # Unnecessary?
        self.data_root = data_root
        self.subset = subset
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.download = download
        self.data = {}
        self.img_ids = []

        # Perform check for data existence/validity
        data_exists = self._check_exists(self.data_root)

        # Take actions on the result
        if not data_exists and not self.download:
            raise RuntimeError("Dataset not found or corrupted. Use download=True to download it")
        elif not data_exists and self.download:
            # TODO: Implement download procedure
            raise NotImplementedError

        # Create a data subset from competition_<test|val|train>.csv
        if self.subset not in ['train', 'test', 'val', 'all']:
            raise ValueError('Value of subset argument must be one of: train, validation, test, full')
        subset_map = {
            'train' : 'competition_train.csv',
            'test' : 'competition_test.csv',
            'val' : 'competition_val.csv'
        }
        if self.subset != 'all':
            self.data = self._parse_csv(
                os.path.join(self.data_root, subset_map[self.subset]))
        else:
            for csv_filename in subset_map.values():
                # Merging operator of 2 dictionaries (Python 3.9.0+)
                self.data |= self._parse_csv(
                    os.path.join(self.data_root, csv_filename)
                )
        self.img_ids = list(self.data.keys())


    def _check_exists(self, data_root) -> bool:
        return calculate_md5_recursive(data_root) == self.DATASET_MD5


    def _parse_csv(self, csv_path: Path) -> dict:
        data = {}
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None) # Skip header
            for line in reader:
                img_name = line[0]
                img_domain = line[-1]
                img_bboxes = []
                bboxes_as_str = line[1:-1][0].split(';')
                for bbox_as_str in bboxes_as_str:
                    try:
                        img_bboxes.append([int(i) for i in bbox_as_str.split(' ')])
                    except ValueError:
                        img_bboxes.append([])
                data[img_name] = {'targets' : { 'bboxes' : img_bboxes}, 'domain' : img_domain}
        return data


    def _load_image(self, index: int) -> Image.Image:
        img_name = self.img_ids[index]
        return Image.open(os.path.join(self.data_root, 'images', img_name)).convert("RGB")


    def _load_target(self, index: int, target: str) -> List[int]:
        img_name = self.img_ids[index]
        return self.data[img_name]['targets'][target]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self._load_image(index)
        target = self._load_target(index, 'bboxes')

        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return img, target


    def __len__(self) -> int:
        return len(self.imgs)




#class WheatHeadsDataModule(pl.LightningDataModule):
    #def __init__(self, data_dir:str="data", batch_size:int=8) -> None:
        #super().__init__()
        #self.data_dir = data_dir
        #self.batch_size = batch_size


    ## Return loaders which are classes providing iterators for Dataset class
    ## Loader combines Dataset and Sampler
    #def _download(self):
        #pass

