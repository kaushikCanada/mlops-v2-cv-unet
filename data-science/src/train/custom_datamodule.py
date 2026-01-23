# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DeepGlobe Land Cover Classification Challenge datamodule."""

from typing import Any
import kornia.augmentation as K
import torch
from torch.utils.data import DataLoader
from torchgeo.transforms.transforms import _ExtractPatches
from custom_nongeodataset import CustomSemanticSegmentationDataset
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datamodules.geo import NonGeoDataModule


class CustomSemanticSegmentationDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the DeepGlobe Land Cover dataset.

    Uses the train/test splits from the dataset.
    """

    def __init__(
        self,
        data_dir: str,
        splits_dir: str,
        config_path: str,
        batch_size: int = 8,
        patch_size: tuple[int, int] | int = 256,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new CustomSemanticSegmentationDataset, batch_size=batch_size, num_workers=num_workers, **kwargs
 instance.
        """
        super().__init__(
            CustomSemanticSegmentationDataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.data_dir = data_dir
        self.splits_dir = splits_dir
        self.config_path = config_path
        self.patch_size = _to_tuple(patch_size)
        self.mean = torch.tensor(
                    [
                        0.485,
                        0.456,
                        0.406
                    ]
                )
        self.std = torch.tensor(
                    [
                        0.229,
                        0.224,
                        0.225
                    ]
                )
        # self.train_aug = A.Compose([
        #                 A.HorizontalFlip(p=0.5),
        #                 A.RandomBrightnessContrast(p=0.5),
        #                 A.Normalize(mean=(0.485, 0.456, 0.406),
        #                             std=(0.229, 0.224, 0.225)),
        #                 A.ToTensorV2()                       # optional—but tensors are guaranteed
        #             ])
        # self.val_aug = A.Compose([
        #                 A.Normalize(mean=(0.485, 0.456, 0.406),
        #                             std=(0.229, 0.224, 0.225)),
        #                 A.ToTensorV2()                       # optional—but tensors are guaranteed
        #             ])
        
        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _ExtractPatches(window_size=self.patch_size),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )
        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomCrop(self.patch_size, pad_if_needed=True),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """

        self.train_dataset = CustomSemanticSegmentationDataset(root = '',
                                                               data_dir = self.data_dir,
                                                               splits_dir = self.splits_dir,
                                                               config_path = self.config_path,
                                                               split = 'train',
                                                               transforms = None, # self.train_aug
                                                               **self.kwargs)
        self.val_dataset = CustomSemanticSegmentationDataset(root = '',
                                                               data_dir = self.data_dir,
                                                               splits_dir = self.splits_dir,
                                                               config_path = self.config_path,
                                                               split = 'val',
                                                               transforms = None, # self.val_aug
                                                               **self.kwargs)
        self.test_dataset = CustomSemanticSegmentationDataset(root = '',
                                                               data_dir = self.data_dir,
                                                               splits_dir = self.splits_dir,
                                                               config_path = self.config_path,
                                                               split = 'test',
                                                               transforms = None, # self.val_aug,
                                                               **self.kwargs)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    # def on_after_batch_transfer(
    #         self, batch: dict[str, torch.Tensor], dataloader_idx: int
    #     ) -> dict[str, torch.Tensor]:
    #     """Apply batch augmentations to the batch after it is transferred to the device.

    #     Args:
    #         batch: A batch of data that needs to be altered or augmented.
    #         dataloader_idx: The index of the dataloader to which the batch belongs.

    #     Returns:
    #         A batch of data.
    #     """
    #     return batch