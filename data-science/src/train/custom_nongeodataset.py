# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DeepGlobe Land Cover Classification Challenge dataset."""

import os
from collections.abc import Callable

from typing import Dict, List, Tuple, Optional, Union
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
import yaml
import pandas as pd
from torchgeo.datasets.errors import DatasetNotFoundError
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import (
    Path,
    check_integrity,
    draw_semantic_segmentation_masks,
    extract_archive,
    rgb_to_mask,
)


class CustomSemanticSegmentationDataset(NonGeoDataset):
    """
        CustomNonGeoDataset
    """

    def __init__(
        self,
        root: str,
        data_dir: str,
        splits_dir: str,
        config_path: str,
        split: str = 'train',
        overlap_percent: float = 0.0,
        patch_size: int = 256,
        transforms = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new DeepGlobeLandCover dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """

        self.root = root
        self.data_dir = data_dir
        self.splits_dir = data_dir if splits_dir is None else splits_dir
        self.config_path = config_path
        self.split = split
        self.transforms = transforms
        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.step = int(self.patch_size * (1 - self.overlap_percent))
        self.config = self._load_config()
        self.mapping_df = pd.read_csv(self.splits_dir +"/"+ "image_mask_mapping_"+ self.split +".csv")

        # print(self.config)
        self.classes, self.colormap = self.create_class_colormap(self.config['rgb_to_class'])
        # print("Classes (hex config):", self.classes)
        # print("Colormap (hex config):", self.colormap)

        # print(self.mapping_df)
        
        self.patch_index = []

        for fidx, row in self.mapping_df.iterrows():
            h, w = Image.open(self.data_dir +"/images/"+ row['image_filename']).size[::-1]          # (H,W)
            n_y = (h - self.patch_size) // self.step + 1
            n_x = (w - self.patch_size) // self.step + 1
            for iy in range(n_y):
                for ix in range(n_x):
                    self.patch_index.append((fidx, iy, ix))

        # print(self.patch_index)

    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def create_class_colormap(self,rgb_to_class):
        """
        Creates class names and colormap from RGB-to-class mapping.
        
        Args:
            rgb_to_class (dict): Dictionary mapping class names to color values (hex or RGB tuples)
            
        Returns:
            tuple: (class_names, matplotlib ListedColormap)
        """
        # Extract class names in order
        class_names = tuple(rgb_to_class.keys())
        
        # Convert colors to matplotlib format (normalized RGB tuples)
        colors = []
        colormap = []
        for class_name in class_names:
            color = rgb_to_class[class_name]
            
            if isinstance(color, str):  # Hex color
                # Remove '#' if present and convert to RGB
                hex_color = color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                # colors.append(tuple(c/255 for c in rgb))
            elif isinstance(color, (tuple, list)):  # RGB tuple
                if all(isinstance(c, int) and c <= 255 for c in color):
                    # colors.append(tuple(c/255 for c in color))
                    rgb = tuple(int(c) for c in color_value)
                else:
                    raise ValueError(f"Invalid RGB values for class {class_name}: {color}")
            else:
                raise ValueError(f"Invalid color format for class {class_name}: {color}")
            
            colormap.append(rgb)

        return class_names, tuple(colormap)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        fidx, iy, ix = self.patch_index[index]
        row = self.mapping_df.iloc[fidx]

        # --- read full image & mask (no caching) ----------------------------
        img: np.typing.NDArray[np.uint8] = np.array(Image.open(self.data_dir +"/images/"+ row["image_filename"]).convert("RGB"))
        msk: np.typing.NDArray[np.uint8] = np.array(Image.open(self.data_dir +"/masks/"+ row["mask_filename"]).convert("RGB"))
        # print(img.shape,msk.shape)
        msk = rgb_to_mask(msk, self.colormap)
        # print(img.shape,msk.shape)
        
        # --- build patch grids with patchify --------------------------------
        img_patch = img[iy:iy+self.patch_size, ix:ix+self.patch_size]
        msk_patch = msk[iy:iy+self.patch_size, ix:ix+self.patch_size]

        # print(img_patch.shape,msk_patch.shape)
         # Albumentations (if any)
        # if self.transforms is not None:
        #     out = self.transforms(image=img_patch, mask=msk_patch)
        #     img_patch, msk_patch = out["image"], out["mask"].long()

        # Ensure Torch tensors even if transforms omitted / no ToTensorV2
        if not torch.is_tensor(img_patch):
            img_patch = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
        if not torch.is_tensor(msk_patch):
            msk_patch = torch.from_numpy(msk_patch).long()               # (H,W)

        sample = {'image': img_patch, 'mask': msk_patch}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.patch_index)
    
    
