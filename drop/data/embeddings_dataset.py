from torch.utils.data import Dataset
from typing import Dict, Any, List, TypeVar, Tuple, Optional, Union, Generator, Callable
from omegaconf import DictConfig, ListConfig

DataFrame = TypeVar("pandas.core.frame.DataFrame")
import h5py
import io
from PIL import Image
import torch
import numpy as np
import logging

class H5Dataset(Dataset):
    def __init__(
        self,
        abs_wsi_path: str,
        h5_path_img: str,
        h5_path_embed: Optional[str] = None,
    ):
        self.abs_wsi_path = abs_wsi_path
        self.h5_path_img = h5_path_img
        self.h5_path_embed = h5_path_embed
        self.image_mpp = 1.0  # todo fix
        self.masked_indices = None
        with h5py.File(f"{self.h5_path_img}", "r") as f:
            try:
                self.masked_indices = f["masked_indices"][:]
                self.image_mpp = f["mpp"][()]
            except:
                logging.info('no masked_indices and mpp defined in h5')

    def get_image(self, index):
        with h5py.File(f"{self.h5_path_img}", "r") as f:
            binary_data = f["data"][index]
        try:
            return Image.open(io.BytesIO(binary_data))
        except Exception as e:
            print(f"Error reading image {index} from {self.h5_path_img}: {e}")
            breakpoint()

    def get_embedding(self, index):

        with h5py.File(f"{self.h5_path_embed}", "r") as f:
            features = f["features"][:]
            region_index = f["region_index"][:]
            feature_index = index

            # Use the indices to retrieve the features
            selected_features = torch.tensor(features[feature_index]).squeeze()

        return selected_features

    def get_metadata(self, key: str, index: int):
        with h5py.File(f"{self.h5_path_img}", "r") as f:
            binary_data = f[key][index]
        if type(binary_data) == np.ndarray:
            binary_data = tuple(binary_data)
        return binary_data

    def get_grid(self):
        with h5py.File(f"{self.h5_path_img}", "r") as f:
            return f["grid"][:]

    def __len__(self):
        with h5py.File(f"{self.h5_path_img}", "r") as f:
            try:
                num_regions = len(f["tile_indices"]) # for h5 images
            except:
                num_regions = len(f['region_index'])  # for model embeddings
        return num_regions if self.masked_indices is None else len(self.masked_indices)

    def __getitem__(self, index: int):
        region_index: int
        if self.masked_indices is not None:
            region_index = self.masked_indices[index]
        else:
            region_index = index

        if self.h5_path_embed is not None:
            # the H5Embeddingsdataset is created from the H5 image dataset, therefore we use the region_index to retrieve the embeddings
            return_dict = {"image": self.get_embedding(region_index), "path": self.abs_wsi_path}
        else:
            # the H5Imagedataset is created from the masked tile_ROI dataset, therefore we use the index to retrieve the images
            return_dict = {"image": self.get_image(index), "path": self.abs_wsi_path}

        return_dict.update(
            {
                "region_index": region_index,
                "mpp": self.image_mpp,
                "grid_local_coordinates": (0, 0),
                "grid_index": 0,
            }  # only one grid is supported
        )
        if self.h5_path_embed is None:  # this is bad
            meta = {key: self.get_metadata(key, index) for key in ["coordinates", "tile_indices"]}
            return_dict.update(meta)
        return return_dict


