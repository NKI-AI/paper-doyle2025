from typing import Dict, Any, List, TypeVar, Tuple, Optional, Union, Generator
from omegaconf import DictConfig, ListConfig
DataFrame = TypeVar("pandas.core.frame.DataFrame")
import logging
from pathlib import Path
from torch.utils import data
from dlup.data.dataset import ConcatDataset, SlideImage, TiledWsiDataset, TiledROIsSlideImageDataset
from torch import Tensor
import torchvision
#logging
from drop.utils.logging import setup_logging
logger = setup_logging(log_name=__name__)
# mask utils
import drop.data.dataset_utils as ds_utils

from .dataset_factory import DatasetFactory


class Dataset(data.Dataset):
    """
    This dataset should work to retrieve tiles produced by DLUP preprocessing,
    along with any relevant bounding boxes.
    """

    def __init__(
        self,
        data_df: DataFrame,
        y_col: str,
        data_cols: DictConfig,
        split: str,
        input_type: str,
        transforms: ListConfig,
        tiling: DictConfig,
        derived_data_path: Union[Path, str],
        scratch_data_path: Union[Path, str],
        extra_features: Optional[List[str]] = None,
        regions_cols: Optional[DictConfig] = None,
        cache_path: Optional[Union[Path, str]] = None,  # when this is given either data is loaded or saved
        embeddings_dir: Optional[Union[Path, str]] = None,
        use_embeddings: bool = False,
        icc_profile: bool = False,
        debug: bool = False,
        use_h5: bool = True,
        make_h5: bool = True,
    ):
        """
        Initializes a Tile dataset.

        Parameters
        -----
        data_df: DataFrame
            Contains all the data for building our dataset.
        y_col: str
            Name of the column containing the labels. Making this modular, so we can change the y_col depending on the task.
        data_cols: DictConfig
            Contains Columns of our data_df relating to the slides on the server, the metadata file and the regions.
        split: str
            Split of the data_proc (train or test or None for inference).
        transforms: ListConfig
           List of transform objects to be applied to items. They are only pre_transforms,
           transforms are applied in the litmodule.
        tiling: DictConfig
            Contains configs for making tiles.
        cache_path:
            For storing datasets with masks.
        derived_data_path:
            For storing h5 tiles
        scratch_data_path:
            For storing intermediate data

        """
        self.split = split
        if self.split is None:
            logging.warning("This dataset is not split into train and test. ")
        transforms = None if transforms.pre_transform == [None] else transforms.pre_transform
        self.transforms = transforms
        self.data_df = data_df
        self.y_col = y_col
        self.extra_features = extra_features if len(extra_features) > 0 else None
        self.server_cols = data_cols.server
        self.meta_cols = data_cols.meta
        self.regions_cols = regions_cols
        self.embeddings_dir = embeddings_dir
        self.use_embeddings = use_embeddings
        if self.split is not None:
            relevant_data_df = self.data_df.loc[self.data_df[self.meta_cols.split] == self.split].copy()
        else:
            relevant_data_df = self.data_df
        self.input_type = input_type # used in logging
        self.dataset_factory = DatasetFactory(
            data_df=self.data_df,
            input_type=input_type,
            server_cols=self.server_cols,
            tiling=tiling,
            derived_data_dir=derived_data_path,
            scratch_data_dir=scratch_data_path,
            regions_cols=self.regions_cols,
            cache_path=cache_path,
            embeddings_dir=embeddings_dir,
            use_embeddings=self.use_embeddings,
            icc_profile=icc_profile,
            debug=debug,
            use_h5=use_h5,
            make_h5=make_h5,
        )

        self.mpps = self.dataset_factory.mpps
        self.dlup_dataset, relevant_data_df, mpps_regions = self.build_dataset(relevant_data_df)
        self.image_mpp = relevant_data_df.iloc[0]["wsi_mpp"]
        self.image_df = relevant_data_df  # used to access the data_df in the getitem (one row per image)
        self.sample_df = self.image_df.loc[self.image_df.index.repeat(self.image_df["num_regions"])].reset_index(
            drop=True
        )
        self.sample_df['region_mpp'] = mpps_regions


    def __len__(self) -> int:
        return len(self.dlup_dataset)



    def build_dataset(self, relevant_data_df) -> ConcatDataset[TiledWsiDataset]:
        """
        Generates a DLUP dataset
        The generated DLUP dataset can be used to retrieve individual tile samples. It is built from a collection of
        whole-slide images.
        Returns
        ------
        A list of dicts containing essential data for each sample.
        """
        single_wsi_datasets: list = []
        logging.info(f"Building dataset...")
        logging.info(f"Using embeddings: {self.use_embeddings}")
        subdirs = relevant_data_df[self.server_cols.subdir].unique().tolist()
        if self.transforms is not None:
            self.transforms = [self.transforms[0](datasets=subdirs)]
        # Iterate over all slides
        logger.info(relevant_data_df)
        num_regions = []
        slides_without_regions = []
        # Define directory where cached datasets and embeddings are stored. Is a list for each mpp
        tiling_subdirs = self.dataset_factory.get_tiling_subdirs(self.mpps)
        # keep a record of the mpp for each region
        mpps_regions = []
        for idx, slide_row in relevant_data_df.iterrows():
            mpp_datasets = []
            for mpp_idx, mpp in enumerate(self.mpps):
                wsi_dataset, abs_wsi_path, embed_path, mask_path = self.dataset_factory.get_mpp_wsi_dataset(slide_row, mpp_idx, tiling_subdirs)
                if self.embeddings_dir:
                    relevant_data_df.loc[idx, "embed_path"] = embed_path
                if (wsi_dataset is not None and len(wsi_dataset) > 0):  # len might be 0 if all regions are masked
                    ds_utils.create_macenko_stain_vectors(self.transforms, abs_wsi_path, slide_row, mask_path)
                mpp_datasets.append(wsi_dataset)
                mpps_regions.extend([mpp] * len(wsi_dataset))
            wsi_datasets = ConcatDataset(mpp_datasets)
            if len(mpp_datasets) > 0:
                single_wsi_datasets.append(wsi_datasets)
                num_regions.append(len(wsi_datasets))
            else:
                num_regions.append(0)
                slides_without_regions.append(abs_wsi_path)
        dlup_dataset = ConcatDataset(single_wsi_datasets)
        logging.info(f"Built dataset successfully")
        relevant_data_df["num_regions"] = num_regions
        if not self.input_type == "regions":
            try:
                relevant_data_df["wsi_mpp"] = wsi_dataset.slide_image.mpp
            except:
                raise ValueError
        # remove slides with no regions --> important to do this!
        relevant_data_df = relevant_data_df[relevant_data_df["num_regions"] > 0]
        store_tile_coords = False
        if store_tile_coords:
            ds_utils.store_tile_coordinates(
                dlup_dataset, relevant_data_df, self.tiling, self.cache_path, self.server_cos.subdir
            )

        return dlup_dataset, relevant_data_df, mpps_regions


    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, float]]:
        """
        Returns a dataset sample (tile) and associated metadata.
        Parameters
        ------
        index : int
            Numerical index specifying which dataset sample to retrieve. Refers to a tile.
        Returns
        ------
        return_object: dict
            'x': torch.Tensor
                A tensor representation of the retrieved image.
            'sample_index': int
                A tensor representation of the sample index of the sample with respect to the dataset.
            'imageName': str
                The server name of the slide.
            'region_index': torch.Tensor
                A tensor representation of the region index on that slide.
            'wsi_mpp': float
                The mpp of the slide at level 0 (original size).
            'mpp': float
                The mpp of the tile as returned by the dataset.
            'scaling_factor': float
                The scaling factor of the tile.
            'y': torch.Tensor, optional
                A tensor representation of the target (if y_col is specified).
            'y_pred': torch.Tensor, optional
                A tensor representation of the prediction (if available in image_df).
            Also additional metadata.
        """
        # Get tile sample from dlup dataset
        img_dict = self.dlup_dataset.__getitem__(index)
        img = img_dict.pop("image")

        required_fields = ["coordinates", "mpp", "region_index", "grid_local_coordinates"]
        tile_meta = {key: value for key, value in img_dict.items() if key in required_fields}

        # Scaling factor used to rescale bboxes to the original image size
        tile_meta["scaling_factor"] = (
            tile_meta["mpp"] / self.image_mpp
        )
        tile_meta["wsi_mpp"] = self.image_mpp  # We add this so we can calculate the box size in microns

        server_path_slide = str(img_dict["path"])
        meta_sample = self.image_df.loc[self.image_df[self.server_cols.path] == server_path_slide]
        tile_meta[self.server_cols.name] = meta_sample[self.server_cols.name].item()
        tile_meta[self.server_cols.slidescore_id] = meta_sample[self.server_cols.slidescore_id].item()
        tile_meta[self.server_cols.subdir] = meta_sample[self.server_cols.subdir].item()

        if not self.use_embeddings:
            img = img.convert("RGB")  # Image is initially in RGBA format
            if self.transforms is not None:
                for aug in self.transforms:
                    img = aug(img, img_path=server_path_slide, dataset=tile_meta[self.server_cols.subdir])
            else:
                img = torchvision.transforms.ToTensor()(img)  # Scales between 0 and 1

        return_object = {"x": img, "sample_index": index}
        return_object.update(tile_meta)

        if self.y_col is not None:
            return_object["y"] = meta_sample[self.y_col].item()

        if "y_pred" in meta_sample.columns:
            return_object["y_pred"] = meta_sample["y_pred"].item()
        if self.extra_features is not None:
            return_object["extra_features"] = [meta_sample[feature].item() for feature in self.extra_features]

        return return_object
