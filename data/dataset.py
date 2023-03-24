import torch
import pytorch_lightning as pl

import os
from pathlib import Path

from data.data_integrity import check_integrity, binary_to_dict

class WheatHeadsDataset(torch.utils.data.Dataset):
    """Entire Wheat Heads Dataset. Provide path to the entire set, train, validation and test.
    Depending on train argument different parts of the set will be returned. integrity_ref argument
    is a binary file with stored checksum for the entire dataset."""
    def __init__(self, data_root: Path, train: bool, integrity_ref: Path, download: bool = True) -> None:
        super().__init__()
        self.data_root = data_root
        self.train = train

        # Read the reference file with checksums for files in dataset
        with open(integrity_ref, 'rb') as file:
            checksum_references = binary_to_dict(file)

        data_exists = self._check_exists(checksum_references, self.data_root)
        if not data_exists and not download:
            raise RuntimeError("Dataset not found or corrupted. Use download=True to download it")
        elif not data_exists and download:
            # TODO: Implement download procedure
            raise NotImplementedError


    def _check_exists(self, checksum_references: dict, data_root) -> bool:
        return all(check_integrity(
            checksum_references[file], os.path.join(data_root, file)) for file in checksum_references.keys())



#class WheatHeadsDataModule(pl.LightningDataModule):
    #def __init__(self, data_dir:str="data", batch_size:int=8) -> None:
        #super().__init__()
        #self.data_dir = data_dir
        #self.batch_size = batch_size


    ## Return loaders which are classes providing iterators for Dataset class
    ## Loader combines Dataset and Sampler
    #def _download(self):
        #pass

