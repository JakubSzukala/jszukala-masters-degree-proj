from torch.utils.data import WeightedRandomSampler

import numpy as np
import pandas as pd

class SamplerConstructor:
    def __init__(self):
        self.dataset_df = None
        self.names = None
        self.counts = None
        self.sampler = None


    def load_dataset(self, dataset_path):
        self.dataset_df = pd.read_csv(dataset_path)


    def _calculate_weights(self, column='domain'):
        self.names, self.counts = \
            np.unique(self.dataset_df[column], return_counts=True)

        # For each domain assign weight equal to 1 / count
        weights = 1 / self.counts
        self.dataset_df['weight'] = self.dataset_df[column].apply(
            lambda x: weights[np.where(self.names == x)[0][0]]
        )


    def _create_sampler(self):
        self.sampler = WeightedRandomSampler(
            weights=self.dataset_df['weight'],
            num_samples=len(self.dataset_df['weight']),
            replacement=True
        )

    def get_sampler(self, column='domain'):
        if self.dataset_df is None:
            raise ValueError('Dataset not loaded')
        self._calculate_weights(column)
        self._create_sampler()
        return self.sampler
