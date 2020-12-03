import pytorch_lightning as pl

from dltranz.metric_learn.metric import BatchRecallTopPL
from dltranz.models import create_head_layers
from dltranz.seq_encoder import create_encoder
from dltranz.train import get_optimizer, get_lr_scheduler, ReduceLROnPlateauWrapper
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy


class CoLESModule(pl.LightningModule):
    metric_name = 'recall_top_k'

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()

        self.sampling_strategy = get_sampling_strategy(params)
        self.loss = get_loss(params, self.sampling_strategy)

        self._seq_encoder = create_encoder(params, is_reduce_sequence=True)
        self._head = create_head_layers(params, self._seq_encoder)

        self.validation_metric = BatchRecallTopPL(**params['validation_metric_params'])

    @property
    def seq_encoder(self):
        return self._seq_encoder

    def forward(self, x):
        return self._seq_encoder(x)

    def training_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        loss = self.loss(y_h, y)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        self.validation_metric(y_h, y)

    def validation_epoch_end(self, outputs):
        self.log(self.metric_name, self.validation_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        params = self.hparams.params
        optimizer = get_optimizer(self, params)
        scheduler = get_lr_scheduler(optimizer, params)
        if isinstance(scheduler, ReduceLROnPlateauWrapper):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]
