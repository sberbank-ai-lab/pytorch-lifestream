# coding: utf-8
import logging

from torch.utils.data import Dataset, IterableDataset

import numpy as np

logger = logging.getLogger(__name__)


class SplittingDataset(Dataset):
    def __init__(self, base_dataset, splitter):
        self.base_dataset = base_dataset
        self.splitter = splitter

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]

        feature_arrays = row['feature_arrays']
        local_date = row['event_time']

        indexes = self.splitter.split(local_date)
        data = [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]
        return data


class IterableSplittingDataset(IterableDataset):
    style = 'iterable'

    def __init__(self, base_dataset, splitter):
        self.base_dataset = base_dataset
        self.splitter = splitter

    def __iter__(self):
        for row, uid in self.base_dataset:
            # TODO: get local_date from data
            local_date = np.arange(len(next(iter(row.values()))))

            for ix in self.splitter.split(local_date):
                yield {k: v[ix] for k, v in row.items()}, uid


class SeveralSplittingsDataset(Dataset):
    def __init__(self, base_dataset, splitters):
        self.base_dataset = base_dataset
        self.splitters = splitters

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]

        feature_arrays = row['feature_arrays']
        local_date = row['event_time']

        data = []
        for splitter in self.splitters:
            indexes = splitter.split(local_date)
            data += [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]
        return data

