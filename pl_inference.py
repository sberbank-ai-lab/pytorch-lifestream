import logging

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import ChainDataset
from torch.utils.data.dataloader import DataLoader

from dltranz.data_load import IterableChain, padded_collate, IterableAugmentations
from dltranz.data_load.augmentations.seq_len_limit import SeqLenLimit
from dltranz.data_load.iterable_processing.category_size_clip import CategorySizeClip
from dltranz.data_load.iterable_processing.feature_filter import FeatureFilter
from dltranz.data_load.iterable_processing.feature_type_cast import FeatureTypeCast
from dltranz.data_load.iterable_processing.target_extractor import TargetExtractor
from dltranz.data_load.parquet_dataset import ParquetDataset, ParquetFiles
from dltranz.lightning_modules.coles_module import CoLESModule
from dltranz.lightning_modules.cpc_module import CpcModule
from dltranz.lightning_modules.rtd_module import RtdModule
from dltranz.lightning_modules.sop_nsp_module import SopNspModule
from dltranz.metric_learn.inference_tools import save_scores
from dltranz.train import score_model
from dltranz.util import get_conf

logger = logging.getLogger(__name__)


def create_inference_dataloader(conf, pl_module):
    """This is inference dataloader for `experiments`
    """
    post_processing = IterableChain(
        FeatureTypeCast({conf['col_id']: int}),
        TargetExtractor(target_col=conf['col_id']),
        FeatureFilter(drop_non_iterable=True),
        CategorySizeClip(pl_module.seq_encoder.category_max_size),
        IterableAugmentations(
            SeqLenLimit(**conf['SeqLenLimit']),
        )
    )
    l_dataset = [
        ParquetDataset(
            ParquetFiles(path).data_files,
            post_processing=post_processing,
            shuffle_files=False,
        ) for path in conf['dataset_files']]
    dataset = ChainDataset(l_dataset)
    return DataLoader(
        dataset=dataset,
        collate_fn=padded_collate,
        shuffle=False,
        num_workers=conf['loader.num_workers'],
        batch_size=conf['loader.batch_size'],
    )


def main(args=None):
    conf = get_conf(args)

    pl.seed_everything(42)

    pl_module = None
    for m in [CoLESModule, CpcModule, SopNspModule, RtdModule]:
        if m.__name__ == conf['params.pl_module_name']:
            pl_module = m
            break
    if pl_module is None:
        raise NotImplementedError(f'Unknown pl module {conf.params.pl_module_name}')
    model = pl_module.load_from_checkpoint(conf['model_path'])
    model.seq_encoder.is_reduce_sequence = True

    dl = create_inference_dataloader(conf['inference_dataloader'], model)

    ids, pred = score_model(model, dl, conf['params'])

    df_scores_cols = [f'v{i:003d}' for i in range(pred.shape[1])]
    col_id = conf['inference_dataloader.col_id']
    df_scores = pd.concat([
        pd.DataFrame({col_id: ids}),
        pd.DataFrame(pred, columns=df_scores_cols),
        ], axis=1)
    logger.info(f'df_scores examples: {df_scores.shape}:')

    save_scores(df_scores, None, conf['output'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(funcName)-20s   : %(message)s')
    logging.getLogger("lightning").setLevel(logging.INFO)

    main()
