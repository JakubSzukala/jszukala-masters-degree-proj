import torch
import pytorch_lightning as pl

import os
from pathlib import Path
from typing import Optional, Callable

from data.data_integrity import calculate_md5_recursive

class WheatHeadsDataset(torch.utils.data.Dataset):
    dataset_file_list = [
        #             MD5 hash           File / directory name
        ('359c66afd88f0983726003ffec4ab466', 'gwhd_2021')
        ]
    def __init__(
            self,
            data_root: Path,
            train: bool,
            transforms: Optional[Callable] = None,
            target_transforms: Optional[Callable] = None,
            download: bool = True
            ) -> None:
        super().__init__() # Unnecessary?
        self.data_root = data_root
        self.train = train
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.download = download

        # Perform check for data existence/validity
        data_exists = self._check_exists(self.data_root)

        # Take actions on the result
        if not data_exists and not self.download:
            raise RuntimeError("Dataset not found or corrupted. Use download=True to download it")
        elif not data_exists and self.download:
            # TODO: Implement download procedure
            raise NotImplementedError

        #self.imgs = 


    def _check_exists(self, data_root) -> bool:
        return calculate_md5_recursive(data_root) == self.dataset_md5

    def _load_image(self):
        raise NotImplementedError


    def _load_target(self):
        raise NotImplementedError



#class WheatHeadsDataModule(pl.LightningDataModule):
    #def __init__(self, data_dir:str="data", batch_size:int=8) -> None:
        #super().__init__()
        #self.data_dir = data_dir
        #self.batch_size = batch_size


    ## Return loaders which are classes providing iterators for Dataset class
    ## Loader combines Dataset and Sampler
    #def _download(self):
        #pass

