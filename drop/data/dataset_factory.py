from typing import Dict, Any, List, TypeVar, Tuple, Optional, Union, Generator
from omegaconf import DictConfig, ListConfig
DataFrame = TypeVar("pandas.core.frame.DataFrame")
import logging
from pathlib import Path
import numpy as np
import pandas as pd
# mask utils
import drop.data.dataset_utils as ds_utils
import math
# macenko
from drop.transforms.custom_transforms import MacenkoAugmentation
#logging
from drop.utils.logging import setup_logging
logger = setup_logging(log_name=__name__)
# h5 embeddings
from drop.data.embeddings_dataset import H5Dataset
import drop.ahcore_tools.create_h5_dataset as h5_tools
from drop.data_proc import data_utils




class DatasetFactory:
    """
    This creates a WSI dataset for a WSI based on required parameters.
    It can also store h5 embeddings of tiles.
    Can return a Tile or a region dataset.
    """
    def __init__(self,
                 data_df: DataFrame,
                 input_type: str,
                 server_cols: List[str],
                 regions_cols: List[str],
                 cache_path: str,
                 embeddings_dir: str,
                 use_embeddings: bool,
                 derived_data_dir: str,
                 scratch_data_dir: str,
                 debug: bool = False,
                 use_h5: bool = False,
                 make_h5: bool = False,
                 icc_profile: str = 'sRGB',
                 tiling: str = 'grid'):

        self._debug = debug
        self._use_h5 = use_h5
        self._make_h5 = make_h5
        self._icc_profile = icc_profile
        self.tiling = tiling

        self.data_df = data_df
        self.input_type = input_type
        self.regions_cols = regions_cols
        self.server_cols = server_cols
        self.mpps = self.get_mpps()
        self.cache_path: Optional[Path] = (
            Path(cache_path) if type(cache_path) is str else cache_path
        )
        self.embeddings_dir = embeddings_dir
        self.use_embeddings = use_embeddings
        self.derived_data_dir = derived_data_dir
        self.scratch_data_dir = scratch_data_dir

    def get_mpp_wsi_dataset(self, slide_row, mpp_idx, tiling_subdirs):
        """ Retrieves WSI dataset at given MPP."""
        abs_wsi_path = slide_row[self.server_cols.path]
        subdir_data = slide_row[self.server_cols.subdir]
        # convert float nan to None
        mask_path = slide_row.astype(object).where(pd.notnull(slide_row), None)[
            self.server_cols.mask
        ]
        mask = None

        if not self.cache_path:
            wsi_dataset = self.get_dataset_for_input_type(abs_wsi_path, mask_path, self.mpps[mpp_idx])
        else:
            # ensure that the hashes stay the same.
            if type(self.tiling.region_mpp) == ListConfig:
                current_image_hash = ""
            else:
                # requires self.tiling.region_mpp to be a float, and not modified
                current_image_hash = ds_utils.generate_hash(abs_wsi_path.split("/")[-1], mask_path, self.mpps[mpp_idx],
                                                            self.tiling)

            dlup_ds_path, h5_path, h5_path_proc, embed_path, embed_path_proc = ds_utils.get_paths(abs_wsi_path,
                                                                                            current_image_hash,
                                                                                            tiling_subdirs[mpp_idx],
                                                                                            subdir_data,
                                                                                            self.input_type,
                                                                                            self.derived_data_dir,
                                                                                            self.scratch_data_dir,
                                                                                            self.embeddings_dir
                                                                                            )

            if self.use_embeddings:  # model embeddings
                used_h5_path = self.get_used_h5_path(embed_path, embed_path_proc)
                try:
                    wsi_dataset = H5Dataset(abs_wsi_path, h5_path_img=embed_path, h5_path_embed=used_h5_path)
                    logging.info(f"Using h5 file {used_h5_path}")
                except:
                    wsi_dataset = []  # skip wsi
                    logging.warning(
                        f"Embeddings not found for {abs_wsi_path}. Create embeddings before running model."
                    )

            elif self._use_h5 and h5_path.is_file():  # h5 embedded images
                used_h5_path = self.get_used_h5_path(h5_path, h5_path_proc)
                wsi_dataset = H5Dataset(abs_wsi_path, h5_path_img=used_h5_path)
                logging.info(f"Using h5 file {used_h5_path}")
            else:
                if dlup_ds_path.is_file():
                    wsi_dataset = ds_utils.load_from_cache(dlup_ds_path)
                    if not len(wsi_dataset) < wsi_dataset.__dict__["masked_indices"][-1]:
                        logging.warning("The mask may be wrong for this dataset")
                    wsi_dataset.__dict__["_path"] = abs_wsi_path
                    logging.info(f"Loaded cached dataset from {dlup_ds_path} for {abs_wsi_path}")
                    if self._make_h5:
                        h5_tools.h5_tiling_pipeline(
                            h5_path,
                            wsi_dataset,
                            self.mpps[mpp_idx],
                            tuple(self.tiling.region_size[0]),
                            mask_path,
                        )
                else:
                    wsi_dataset = self.get_dataset_for_input_type(abs_wsi_path, mask_path, self.mpps[mpp_idx])
                    if wsi_dataset is not None:
                        ds_utils.save_to_cache(wsi_dataset, dlup_ds_path)
                        if self._make_h5:
                            h5_tools.h5_tiling_pipeline(
                                h5_path,
                                wsi_dataset,
                                self.mpps[mpp_idx],
                                tuple(self.tiling.region_size[0]),
                                mask_path,
                            )
        return wsi_dataset, abs_wsi_path, embed_path, mask_path


    def get_dataset_for_input_type(self, abs_wsi_path: Path, mask_path: str, mpp: float):
        if self.input_type == "tiles":
            dataset = self.get_masked_tile_dataset(abs_wsi_path, mask_path, mpp)
        elif self.input_type == "regions":
            dataset = self.get_region_dataset(abs_wsi_path)
        elif self.input_type == "embeddings":
            raise NotImplementedError
        else:
            raise ValueError(f"input_type {self.input_type} not supported")

        return dataset

    def get_masked_tile_dataset(self, abs_wsi_path: Path, mask_path: str, mpp: float ):
        mask = ds_utils.get_mask(mask_path)
        tiled_wsi_dataset = ds_utils.create_tiled_wsi_dataset(abs_wsi_path, mask, self.tiling, mpp)
        return tiled_wsi_dataset

    def get_region_dataset(self, abs_wsi_path: Path):
        slide_row = self.data_df[self.data_df[self.server_cols.path] == abs_wsi_path]
        wsi_dataset = ds_utils.create_img_bag_wsi_dataset(abs_wsi_path, slide_row[self.regions_cols.sel].item())
        return wsi_dataset

    def get_tiling_subdirs(self, mpps):
        tiling_subdirs = []
        for mpp in mpps:
            tiling_subdir_tmp = f"mpp{mpp}_size{self.tiling.region_size[0][0]}"
            tiling_subdir_tmp += "_icc" if self._icc_profile else ""
            tiling_subdirs.append(tiling_subdir_tmp)
        return tiling_subdirs

    def get_used_h5_path(self, orig_path, proc_path):
        if proc_path.is_file():
            return proc_path
        elif not self._debug:
            data_utils.copy_files_to_scratch(
                [orig_path], self.derived_data_dir, self.scratch_data_dir, ".h5"
            )
            return proc_path
        else:
            return orig_path

    def get_mpps(self):
        mpps= [self.tiling.region_mpp] if type(self.tiling.region_mpp) == float else [mpp for mpp in
                                                                                      self.tiling.region_mpp]
        return mpps

