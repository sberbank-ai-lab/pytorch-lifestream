import json
import logging

import pytorch_lightning as pl
from embeddings_validation.file_reader import TargetFile, ID_TYPE_MAPPING
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dltranz.data_load import padded_collate, IterableChain, IterableAugmentations
from dltranz.data_load.augmentations.build_augmentations import build_augmentations
from dltranz.data_load.data_module.map_augmentation_dataset import MapAugmentationDataset
from dltranz.data_load.iterable_processing.category_size_clip import CategorySizeClip
from dltranz.data_load.iterable_processing.feature_filter import FeatureFilter
from dltranz.data_load.iterable_processing.feature_type_cast import FeatureTypeCast
from dltranz.data_load.iterable_processing.id_filter import IdFilter
from dltranz.data_load.iterable_processing.iterable_shuffle import IterableShuffle
from dltranz.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from dltranz.data_load.iterable_processing.target_join import TargetJoin
from dltranz.data_load.parquet_dataset import ParquetFiles, ParquetDataset

logger = logging.getLogger(__name__)


class ClsDataModuleTrain(pl.LightningDataModule):
    def __init__(self, conf, pl_module, fold_id):
        super().__init__()

        self.fold_id = fold_id
        self._type = conf['type']
        assert self._type in ('map', 'iterable')

        self.setup_conf = conf['setup']
        self.train_conf = conf['train']
        self.valid_conf = conf['valid']
        self.test_conf = conf.get('test', self.valid_conf)

        self.col_id = self.setup_conf['col_id']
        self.col_id_dtype = {
            'str': str,
            'int': int,
        }[self.setup_conf['col_id_dtype']]
        self.col_target = self.setup_conf['col_target']
        self.category_names = pl_module.seq_encoder.category_names
        self.category_names.add('event_time')
        self.category_max_size = pl_module.seq_encoder.category_max_size

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self._train_targets = None
        self._valid_targets = None
        self._test_targets = None

        self._fold_info = None

    def prepare_data(self):
        if 'dataset_files' in self.setup_conf:
            self.setup_iterable_files()
        elif 'dataset_parts' in self.setup_conf:
            self.setup_iterable_parts()
        else:
            raise AttributeError(f'Missing dataset definition. One of `dataset_parts` or `dataset_files` expected')

        if self._type == 'map':
            self.setup_map()

    def setup_iterable_files(self):
        if self.setup_conf['split_by'] == 'embeddings_validation':
            train_data_files = ParquetFiles(self.setup_conf['dataset_files.train_data_path']).data_files
            test_data_files = ParquetFiles(self.setup_conf['dataset_files.test_data_path']).data_files

            self.read_external_splits()

            self.train_dataset = ParquetDataset(
                train_data_files,
                post_processing=IterableChain(*self.build_iterable_processing('train')),
                shuffle_files=True if self._type == 'iterable' else False,
            )
            self.valid_dataset = ParquetDataset(
                train_data_files,
                post_processing=IterableChain(*self.build_iterable_processing('valid')),
                shuffle_files=False,
            )
            self.test_dataset = ParquetDataset(
                test_data_files,
                post_processing=IterableChain(*self.build_iterable_processing('test')),
                shuffle_files=False,
            )

        else:
            raise AttributeError(f'Unknown split strategy: {self.setup_conf.split_by}')

    def read_external_splits(self):
        if self.setup_conf['split_by'] == 'embeddings_validation':
            with open(self.setup_conf['fold_info'], 'r') as f:
                self._fold_info = json.load(f)

        current_fold = self._fold_info[self.fold_id]
        self._train_targets = TargetFile.load(current_fold['train']['path'])
        self._valid_targets = TargetFile.load(current_fold['valid']['path'])
        self._test_targets = TargetFile.load(current_fold['test']['path'])

        labeled_amount = self.train_conf.get('labeled_amount', None)
        if labeled_amount is not None:
            _len_orig = len(self._train_targets)
            if type(labeled_amount) is float:
                labeled_amount = int(_len_orig * labeled_amount)
            self._train_targets.df = self._train_targets.df.iloc[:labeled_amount]
            logger.info(f'Reduced train amount from {_len_orig} to {len(self._train_targets)}')

    def build_iterable_processing(self, part):
        yield FeatureTypeCast({self.col_id: self.col_id_dtype})

        if 'dataset_files' in self.setup_conf and self.setup_conf['split_by'] == 'embeddings_validation':
            if part == 'train':
                yield IdFilter(id_col=self.col_id, relevant_ids=self._train_targets.df[self.col_id].values.tolist())
            elif part == 'valid':
                yield IdFilter(id_col=self.col_id, relevant_ids=self._valid_targets.df[self.col_id].values.tolist())
            elif part == 'test':
                yield IdFilter(id_col=self.col_id, relevant_ids=self._test_targets.df[self.col_id].values.tolist())
            else:
                raise AttributeError(f'Unknown part: {part}')

        if part == 'train':
            yield SeqLenFilter(min_seq_len=self.train_conf['min_seq_len'])

        if part == 'train':
            yield TargetJoin(self.col_id, self._train_targets.df.set_index(self.col_id)[self.col_target].to_dict())
        elif part == 'valid':
            yield TargetJoin(self.col_id, self._valid_targets.df.set_index(self.col_id)[self.col_target].to_dict())
        elif part == 'test':
            yield TargetJoin(self.col_id, self._test_targets.df.set_index(self.col_id)[self.col_target].to_dict())
        else:
            raise AttributeError(f'Unknown part: {part}')

        yield FeatureFilter(keep_feature_names=self.category_names)
        yield CategorySizeClip(self.category_max_size)

        if self._type == 'iterable':
            # all processing in single chain
            if part == 'train':
                yield IterableShuffle(self.train_conf['buffer_size'])

            yield IterableAugmentations(self.build_augmentations(part))

    def build_augmentations(self, part):
        if part == 'train':
            return build_augmentations(self.train_conf['augmentations'])
        elif part == 'valid':
            return build_augmentations(self.valid_conf['augmentations'])
        elif part == 'test':
            return build_augmentations(self.test_conf['augmentations'])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=padded_collate,
            shuffle=False if self._type == 'iterable' else True,
            num_workers=self.train_conf['num_workers'],
            batch_size=self.train_conf['batch_size'],
            drop_last=self.train_conf.get('drop_last', False)
        )

    def setup_map(self):
        self.train_dataset = list(tqdm(iter(self.train_dataset)))
        logger.info(f'Loaded {len(self.train_dataset)} for train')
        self.valid_dataset = list(tqdm(iter(self.valid_dataset)))
        logger.info(f'Loaded {len(self.valid_dataset)} for valid')
        self.test_dataset = list(tqdm(iter(self.test_dataset)))
        logger.info(f'Loaded {len(self.test_dataset)} for test')

        self.train_dataset = MapAugmentationDataset(
            base_dataset=self.train_dataset,
            a_chain=self.build_augmentations('train'),
        )
        self.valid_dataset = MapAugmentationDataset(
            base_dataset=self.valid_dataset,
            a_chain=self.build_augmentations('valid'),
        )
        self.test_dataset = MapAugmentationDataset(
            base_dataset=self.test_dataset,
            a_chain=self.build_augmentations('test'),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            collate_fn=padded_collate,
            num_workers=self.valid_conf['num_workers'],
            batch_size=self.valid_conf['batch_size'],
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=padded_collate,
            num_workers=self.test_conf['num_workers'],
            batch_size=self.test_conf['batch_size'],
        )

