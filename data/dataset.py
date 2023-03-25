import torch
import pytorch_lightning as pl

import os
import csv
from pathlib import Path
from typing import Optional, Callable, Tuple, Any

from data.data_integrity import calculate_md5_recursive

class WheatHeadsDataset(torch.utils.data.Dataset):
    dataset_md5 = '359c66afd88f0983726003ffec4ab466'

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
        self.imgs = []
        self.targets = []

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
            self.imgs, self.targets = self._parse_csv(
                os.path.join(self.data_root, subset_map[self.subset]))
        else:
            temp_imgs = []
            temp_targets = []
            for csv_filename in subset_map.values():
                temp_imgs, temp_targets = self._parse_csv(
                    os.path.join(self.data_root, csv_filename)
                )
                self.imgs.append(temp_imgs)
                self.targets.append(temp_targets)


    def _check_exists(self, data_root) -> bool:
        return calculate_md5_recursive(data_root) == self.dataset_md5


    def _parse_csv(self, csv_path: Path):
        imgs = []
        targets = []
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None) # Skip header
            for line in reader:
                img_name = line[0]
                img_target = []
                bboxes_as_str = line[1:-1][0].split(';')
                for bbox_as_str in bboxes_as_str:
                    try:
                        img_target.append([int(i) for i in bbox_as_str.split(' ')])
                    except ValueError as e:
                        pass
                imgs.append(img_name)
                targets.append(img_target)
        return tuple(imgs), tuple(targets)


    def _load_image(self):
        raise NotImplementedError


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

