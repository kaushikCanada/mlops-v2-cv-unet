from typing import Any
from collections.abc import Sequence
from lightning.pytorch.callbacks.callback import Callback
import lightning
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    FBetaScore,
    JaccardIndex,
    Precision,
    Recall,
)

from torchgeo.trainers import SemanticSegmentationTask


class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    # any keywords we add here between *args and **kwargs will be found in self.hparams
    def __init__(
        self, *args: Any, tmax: int = 50, eta_min: float = 1e-6, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)  # pass args and kwargs to the parent class

    def configure_optimizers(
        self,
    ) -> 'lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig':
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        tmax: int = self.hparams['tmax']
        eta_min: float = self.hparams['eta_min']

        optimizer = AdamW(self.parameters(), lr=self.hparams['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eta_min)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': self.monitor},
        }

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams['num_classes']

        self.train_metrics = MetricCollection(
            {
                'OverallAccuracy': Accuracy(
                    task='multiclass', num_classes=num_classes, average='micro'
                ),
                'OverallPrecision': Precision(
                    task='multiclass', num_classes=num_classes, average='micro'
                ),
                'OverallRecall': Recall(
                    task='multiclass', num_classes=num_classes, average='micro'
                ),
                'OverallF1Score': FBetaScore(
                    task='multiclass',
                    num_classes=num_classes,
                    beta=1.0,
                    average='micro',
                ),
                'MeanIoU': JaccardIndex(
                    num_classes=num_classes, task='multiclass', average='macro'
                ),
            },
            prefix='train_',
        )
        self.val_metrics = self.train_metrics.clone(prefix='val_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """Initialize callbacks for saving the best and latest models.

        Returns:
            List of callbacks to apply.
        """
        return [
            # ModelCheckpoint(every_n_epochs=50, save_top_k=-1, save_last=True),
            # ModelCheckpoint(monitor=self.monitor, mode=self.mode, save_top_k=5),
        ]

    def on_train_epoch_start(self) -> None:
        """Log the learning rate at the start of each training epoch."""
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            lr = optimizers[0].param_groups[0]['lr']
        else:
            lr = optimizers.param_groups[0]['lr']
        # self.logger.experiment.add_scalar('lr', lr, self.current_epoch)