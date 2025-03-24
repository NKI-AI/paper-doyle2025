from typing import Dict, Any, List, TypeVar, Optional, Union
DataFrame = TypeVar("pandas.core.frame.DataFrame")
from omegaconf import DictConfig, ListConfig
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils import data
import torchvision.transforms as transforms
from dlup._exceptions import UnsupportedSlideError  # dlup version older: DlupUnsupportedSlideError
from dlup.data.dataset import ConcatDataset, SlideImage, SlideImageDatasetBase
from dlup.annotations import WsiAnnotations

import drop.data.dataset_utils as mask_utils

from drop.transforms.custom_transforms import MacenkoAugmentation


class MILRegionDataset(data.Dataset):
    """
    This DLUP dataset retrieves (augmented) region images with associated labels.
    The labels of regions (instances) are based on the label of the associated WSI (bag).
    """

    def __init__(
        self,
        data_df: DataFrame,
        y_col: str,
        data_cols: DictConfig,
        split: str,
        input_type: str,
        target_mpp: float,  # remove
        tiling: DictConfig,
        transforms: ListConfig,
        regions_cols: DictConfig,
        cache_path: Optional[Union[Path, str]] = None,  # when this is given either data is loaded or saved
    ):
        """
        Initializes a Region MIL dataset.

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
        # target_mpp: float
        #     The mpp at which regions should be read from the WSI.
        tiling: DictConfig
            Contains the parameters for creating the regions. For instance ROI size at a given mpp and target outsize.
        transforms: ListConfig
           List of transform objects to be applied to items.
        regions_cols: DictConfig
            Columns which specify regions.
        cache_path:
            For storing datasets.
        """

        super().__init__()
        self.data_df = data_df

        self.y_col = y_col
        self.split = split
        self.tiling = tiling
        self.transforms = transforms
        self.server_cols = data_cols.server
        self.meta_cols = data_cols.meta
        self.regions_cols = regions_cols
        self.cache_path = Path(cache_path) if cache_path is not None else None
        if self.cache_path is not None:
            self.cache_path.mkdir(
                parents=True, exist_ok=True
            )  # is now at Precision_NKI_89 - could potentially also be specfic to scanne
        self.relevant_data_df = None
        if self.split is not None:
            self.relevant_data_df = self.data_df.loc[self.data_df[self.meta_cols.split] == self.split].copy()
        else:
            self.relevant_data_df = self.data_df
        self.dlup_dataset = self.build_dataset()

    def build_dataset(self) -> ConcatDataset[SlideImageDatasetBase]:
        """
        Generates a MIL dataset. The generated MIL dataset can be used to retrieve individual region samples.
        It is built from a collection of whole-slide images.

        Returns
        ------
        A list of dicts containing essential data_proc for each sample.
        """
        single_wsi_datasets: list = []
        logging.info(f"Building dataset...")
        slides_without_regions = []
        num_regions = []

        for idx, slide_row in self.relevant_data_df.iterrows():
            abs_wsi_path = slide_row[self.server_cols.path]
            try:
                slide_image = SlideImage.from_file_path(Path(abs_wsi_path))
                mpp = slide_image.mpp
                slide_image.close()
            except Exception as e:
                # Print the exception message
                logging.warning("Exception:", str(e))
                if e == UnsupportedSlideError:
                    logging.warning(f"{abs_wsi_path} is unsupported. Skipping WSI.")
                num_regions.append(0)
                continue

            if self.transforms is not None:
                if type(self.transforms[0]) == MacenkoAugmentation:
                    try:
                        macenko_augmentation = self.transforms[0]
                    except KeyError:
                        logging.warning(f"Macenko normaliser not defined for {slide_row['subdir']}")

                    for template_index, template_normaliser in enumerate(macenko_augmentation.normalisers):
                        _ = template_normaliser[slide_row["subdir"]]._get_staining_vectors_from_cache_or_file(
                            [abs_wsi_path]
                        )

            if self.cache_path:
                current_image_hash = mask_utils.generate_hash(abs_wsi_path, self.tiling)
                pickle_file_path = self.cache_path / f"{current_image_hash}.pkl"
                if pickle_file_path.is_file():
                    wsi_dataset = mask_utils.load_from_cache(pickle_file_path)
                else:
                    wsi_dataset = self.create_img_bag_wsi_dataset(abs_wsi_path, slide_row[self.regions_cols.sel])
                    mask_utils.save_to_cache(wsi_dataset, pickle_file_path)
            else:
                wsi_dataset = self.create_img_bag_wsi_dataset(abs_wsi_path, slide_row[self.regions_cols.sel])

            if wsi_dataset is not None:
                single_wsi_datasets.append(wsi_dataset)
                num_regions.append(len(wsi_dataset))
            else:
                # the slides without regions need to be removed before creating the CV splits.
                slides_without_regions.append(slide_row[self.server_cols.name])
                num_regions.append(0)

        self.relevant_data_df["num_regions"] = num_regions
        logging.info(f"{len(slides_without_regions)} Slides without any detected regions: {slides_without_regions}")

        dlup_dataset = ConcatDataset(single_wsi_datasets)
        logging.info(f"Built dataset successfully")

        return dlup_dataset

    def create_img_bag_wsi_dataset(self, abs_wsi_path: str, regions: List[List[int]]) -> SlideImageDatasetBase:
        # regions specific part
        regions_coords = np.array(regions)
        print(len(regions))
        if len(regions_coords) > 0:
            regions_coords = regions_coords[:, :4].astype(int).tolist()
            res = [regions_coords[i][:4] + [regions[i][4]] for i in range(len(regions_coords))]
        else:
            res = []
        if len(res) >= 1:
            img_bag_wsi_dataset = SlideImageDatasetBase(
                path=abs_wsi_path, regions=res, crop=False, mask=None, mask_threshold=None, transform=None
            )
            # for i in range(len(img_bag_wsi_dataset)):
            #     try:
            #         _ = img_bag_wsi_dataset.__getitem__(int(i))
            #     except:
            #         print("error")
            #         breakpoint()
            return img_bag_wsi_dataset
        else:
            return None

    def __len__(self) -> int:
        return len(self.dlup_dataset)

    def __getitem__(self, index: int) -> Dict:
        """
        Samples an img_dict with keys ['image', 'coordinates', 'mpp', 'path', 'region_index'] from dlup_dataset.
        Applies transformations including resizing.
        Returns a dataset sample (region) and associated label, server_name and region index.

        Parameters
        ------
        index : int
            Numerical index specifying which dataset sample to retrieve. Refers to a tile.
        Returns
        ------
        return_object:
            x: A tensor representation of the retrieved image.
            y: A tensor representation of the target
            id: A tensor representation of the servername of the slide
            region_index: An int which is the index which was sampled from dlup_dataset
        """
        # Get a region sample from dlup dataset
        try:
            img_dict = self.dlup_dataset.__getitem__(index)
        except:
            print(index)
            print("coudlnt get item")
            img_dict = self.dlup_dataset.__getitem__(index + 1)

        server_path_slide = str(img_dict["path"])
        img = img_dict.pop("image")
        img = img.convert("RGB")  # image is RGBA initially
        import torchvision

        if self.transforms is not None:
            for aug in self.transforms:
                if type(aug) == MacenkoNormalizer:
                    transformation_probability = 1.0
                    import random

                    if random.random() < transformation_probability:
                        img = aug(img, img_path=[server_path_slide])
                    else:
                        img = torchvision.transforms.ToTensor()(img)
                else:
                    img = aug(img)
        else:
            img = torchvision.transforms.ToTensor()(img)

        meta_sample = self.relevant_data_df.loc[self.relevant_data_df[self.server_cols.path] == server_path_slide]
        meta_dict = {
            self.server_cols.name: meta_sample[self.server_cols.name].item(),
            self.server_cols.slidescore_id: int(meta_sample[self.server_cols.slidescore_id].item()),
            self.server_cols.subdir: meta_sample[self.server_cols.subdir].item(),
        }

        return_object = {"x": img, "region_index": index}
        return_object.update(meta_dict)
        if self.y_col is not None:
            target = int(meta_sample[self.y_col].item())
            return_object.update({"y": target})

        return return_object
