import os
from typing import Dict, Any, List, TypeVar, Tuple, Optional, Union
from omegaconf import DictConfig
# for dealing with caching of masks
import pickle
import hashlib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
# for loading mask
from dlup.annotations import WsiAnnotations
from dlup._exceptions import UnsupportedSlideError  # dlup version older: DlupUnsupportedSlideError
from dlup.backends import TifffileSlide, OpenSlideSlide, PyVipsSlide  # type: ignore
from dlup.data.dataset import ConcatDataset, SlideImage, TiledWsiDataset, BaseWsiDataset
from dlup.tiling import TilingMode
from drop.transforms.custom_transforms import MacenkoAugmentation
from drop.tools.json_saver import JsonSaver


def generate_hash(img_path, *args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    current_image_hash = hashlib.sha256(pickle.dumps([img_path, *args])).hexdigest()
    return current_image_hash

def get_mask(mask_path: Optional[Union[str, Path]]) -> Any:
    if mask_path is None or mask_path == "":
        return None
    else:
        mask_path = Path(mask_path)
        if mask_path.suffix == ".tiff":
            mask = SlideImage.from_file_path(
                mask_path, backend=TifffileSlide
            )
        elif mask_path.suffix == ".json":
            mask = WsiAnnotations.from_geojson(mask_path)
        else:
            logging.info("invalid mask file")
        return mask

def load_from_cache(pickle_file_path):
    with open(pickle_file_path, "rb") as file:
        img_bag_wsi_dataset = torch.load(file)
        # logging.info(f"Loaded image dataset at {pickle_file_path} from cache")
    return img_bag_wsi_dataset


def save_to_cache(img_bag_wsi_dataset, pickle_file_path):
    with open(pickle_file_path, "wb") as file:
        torch.save(img_bag_wsi_dataset, file)
        logging.info(f"Saved image dataset at {pickle_file_path} to cache")

def open_slide_image(abs_wsi_path: str,
                     backend: OpenSlideSlide,
                        apply_color_profile: bool = False) -> SlideImage:
    try:
        slide_image = SlideImage.from_file_path(
            Path(abs_wsi_path),
            backend=backend, apply_color_profile=apply_color_profile
            )
        return slide_image
    except Exception as e:
        # Print the exception message
        logging.warning("Exception:", str(e))
        if e == UnsupportedSlideError:
            logging.warning(f"{abs_wsi_path} is unsupported. Skipping WSI.")
        return None


def create_tiled_wsi_dataset( abs_wsi_path: Path, mask: Any, tiling: DictConfig, target_mpp: float):
    tiled_wsi_dataset = None
    try:
        slide_image = open_slide_image(abs_wsi_path)
    except Exception as e:
        logging.warning("Exception:", str(e))
        logging.warning(f"{abs_wsi_path} is unsupported. Skipping WSI.")
        breakpoint()
    try:
        tiled_wsi_dataset = TiledWsiDataset.from_standard_tiling(
            path=abs_wsi_path,
            mpp=target_mpp,
            tile_size=tiling.region_size[0],
            tile_overlap=(
                tiling.tile_overlap_x,
                tiling.tile_overlap_y,
            ),
            tile_mode=TilingMode[
                tiling.tile_mode
            ],  # TilingMode.overflow does not (!) work with the current implementation of the dataset
            crop=tiling.crop,
            mask=mask,
            mask_threshold=tiling.foreground_percentage_threshold,
            transform=None,
            limit_bounds=True,
        )
    except Exception as e:
        logging.warning("Exception:", str(e))
        tiled_wsi_dataset = None
        breakpoint()
    return tiled_wsi_dataset


def create_img_bag_wsi_dataset(abs_wsi_path: str, regions: List[List[int]]) -> BaseWsiDataset:
    # regions specific part
    regions_coords = np.array(regions)
    print(len(regions))
    if len(regions_coords) > 0:
        regions_coords = regions_coords[:, :4].astype(int).tolist()
        res = [regions_coords[i][:4] + [regions[i][4]] for i in range(len(regions_coords))]
    else:
        res = []
    if len(res) >= 1:
        img_bag_wsi_dataset = BaseWsiDataset(
            path=abs_wsi_path, regions=res, crop=False, mask=None, mask_threshold=None, transform=None
        )
        return img_bag_wsi_dataset
    else:
        return None

def create_macenko_stain_vectors(transforms, abs_wsi_path: Path, slide_row: pd.Series, mask_path):
    if transforms is not None:
        if type(transforms[0]) == MacenkoAugmentation:
            try:
                macenko_augmentation = transforms[0]
            except KeyError:
                logging.warning(f"Macenko normaliser not defined for {slide_row['subdir']}")

            for template_index, template_normaliser in enumerate(macenko_augmentation.normalisers):
                _ = template_normaliser[slide_row["subdir"]]._get_staining_vectors_from_cache_or_file([abs_wsi_path], [mask_path])


def store_tile_coordinates(dlup_dataset, relevant_data_df, tiling_dict, out_dir, subdir_col):
    """Store the tile coordinates in a json file for later use"""
    json_saver = JsonSaver("masked_tiles", f"{out_dir}/masked_tiles.json")
    dlup_tiles_list = []
    region_coordinates_scaled_to_wsi = []
    sel_indices = []
    for i, tiled_dataset in enumerate(dlup_dataset.__dict__["datasets"]):
        sel_indices.append(tiled_dataset.__dict__["masked_indices"])
        dlup_tiles = [tiled_dataset.__dict__["regions"][sel_i] for sel_i in tiled_dataset.__dict__["masked_indices"]]
        dlup_tiles_list.append(dlup_tiles)
        regions_per_wsi = np.array(dlup_tiles, dtype=object)
        # remove the 5th column of np.array which contains the mpp
        try:
            regions_per_wsi_scaled = regions_per_wsi[:, :4] * (
                regions_per_wsi[:, 4:5] / relevant_data_df["wsi_mpp"][i]
            )
        except IndexError:
            regions_per_wsi_scaled = np.array([])
        region_coordinates_scaled_to_wsi.append(regions_per_wsi_scaled)

    # relevant_data_df["regions_coordinates"] = dlup_tiles_list # can't store to json easily
    relevant_data_df["regions_coordinates_scaled_to_wsi"] = [
        coords.tolist() for coords in region_coordinates_scaled_to_wsi
    ]
    relevant_data_df["masked_indices"] = [coords.tolist() for coords in sel_indices]
    subdirs = relevant_data_df[subdir_col].unique().tolist()
    for subdir in subdirs:
        subdir_df = relevant_data_df[relevant_data_df[subdir_col] == subdir]
        id_dict = tiling_dict.copy()
        id_dict.update({"subdir": subdir})
        json_saver.save_selected_data(id_dict, "slide_level", subdir_df.to_dict())

def replace_hash(file_path, hash):
    return str(file_path).replace(hash, '')

def rename_file(file_path, hash):
    new_file_name = replace_hash(file_path, hash)
    # Rename the file
    os.rename(file_path, new_file_name)
    return Path(new_file_name)

def get_h5_path(abs_wsi_path: str, current_image_hash: str, output_dir: str, subdir: str = None) -> Path:

    # obtain the scanner and block path from the absolute path
    parts = abs_wsi_path.split("PRECISION/")
    scanner_path = parts[1].split("images/")[0]
    # Combine the scanner_path with the output_dir (derived_data_dir)
    output_dir = Path(output_dir) / Path(scanner_path)
    # add h5 output folder
    if subdir is not None:
        output_dir = output_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (Path(abs_wsi_path).stem + current_image_hash + ".h5")
    return output_file

def deal_with_paths(dlup_ds_path, h5_path, h5_path_proc, embed_path, embed_path_proc, current_image_hash):
    try:
        old = dlup_ds_path
        dlup_ds_path = rename_file(dlup_ds_path, current_image_hash)
        print(f"Removed hash from {dlup_ds_path}") if old != dlup_ds_path else None
    except:
        dlup_ds_path = Path(replace_hash(dlup_ds_path, current_image_hash))
    try:
        old = h5_path
        h5_path = rename_file(h5_path, current_image_hash)
        print(f"Removed hash from {h5_path}") if old != h5_path else None
        h5_path_proc = rename_file(h5_path_proc, current_image_hash)
    except:
        h5_path = Path(replace_hash(h5_path, current_image_hash))
        h5_path_proc = Path(replace_hash(h5_path_proc, current_image_hash))
    try:
        old = embed_path
        embed_path = rename_file(embed_path, current_image_hash)
        print(f"Removed hash from {embed_path}") if old != embed_path else None
        embed_path_proc = rename_file(embed_path_proc, current_image_hash)
    except:
        embed_path = Path(replace_hash(embed_path, current_image_hash))
        embed_path_proc = Path(replace_hash(embed_path_proc, current_image_hash))

    return dlup_ds_path, h5_path, h5_path_proc, embed_path, embed_path_proc


def get_paths(abs_wsi_path, current_image_hash, tiling_subdir, subdir_data, input_type, derived_data_dir, scratch_data_dir, embeddings_dir):
    # Need to know whether it's an embedding ds or not, and what the input area is (tile or region)
    input_type = input_type.replace('embeddings_', "")
    input_type = f"_{input_type}"

    h5_path = get_h5_path(
        abs_wsi_path,
        current_image_hash,
        output_dir=derived_data_dir,
        subdir=f"h5_images{input_type}/{tiling_subdir}",
    )
    h5_path_proc = get_h5_path(
        abs_wsi_path,
        current_image_hash,
        output_dir=scratch_data_dir,
        subdir=f"h5_images{input_type}/{tiling_subdir}",
    )
    dlup_ds_path = (
            Path(derived_data_dir)
            / subdir_data
            / f"cached_datasets{input_type}"
            / tiling_subdir
            / f"{Path(abs_wsi_path).stem}{current_image_hash}.pkl"
    )
    if embeddings_dir:
        embed_path = get_h5_path(
            abs_wsi_path,
            current_image_hash,
            output_dir=derived_data_dir,
            subdir=f"h5_embeddings{input_type}/{tiling_subdir}/{embeddings_dir}",
        )
        embed_path_proc = get_h5_path(
            abs_wsi_path,
            current_image_hash,
            output_dir=scratch_data_dir,
            subdir=f"h5_embeddings{input_type}/{tiling_subdir}/{embeddings_dir}",
        )
    else:
        embed_path = None
        embed_path_proc = None

    res = deal_with_paths(dlup_ds_path, h5_path, h5_path_proc, embed_path, embed_path_proc,
                                           current_image_hash)
    dlup_ds_path, h5_path, h5_path_proc, embed_path, embed_path_proc = res

    return dlup_ds_path, h5_path, h5_path_proc, embed_path, embed_path_proc