import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Dict
import pandas as pd
from typing import TypeVar
DataFrame = TypeVar("pandas.core.frame.DataFrame")
from omegaconf import DictConfig
from drop.data_proc import data_utils


class SlideMapper:
    """
    Takes the metadata file and matches it to the available slides on server.
    For this purpose it creates a slide_mapping_df of the servername of the slide and the slide_path.
    It also creates a mapping between the server_name and the slidescore_id,
    and a mapping between the servername and the mask.
    The resulting df is saved to matched_metadata_fn_out.
    """

    def __init__(
        self,
        dataset_name: str,
        orig_data_dir: str,
        derived_data_dir: str,
        data_dir: str,
        subdirs: List[str],
        img_ext: str,
        dataset_dir: str,
        metadata_fn: str,
        slidescore_mapping_fn: str,
        server_id_slide_mapping_fn: str,
        meta_data_cols_orig: DictConfig,
        matched_metadata_fn_out: str,
        server_cols: DictConfig,
    ):
        self.dataset_name = dataset_name
        self.orig_data_dir = orig_data_dir
        self.derived_data_dir = derived_data_dir
        self.data_dir = data_dir
        self.subdirs = subdirs
        self.img_ext = img_ext
        self.dataset_dir = dataset_dir
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        self.sel_metadata_fn = f"{dataset_dir}{metadata_fn}"
        self.matched_metadata_path = Path(f"{dataset_dir}{matched_metadata_fn_out}")
        self.slidescore_mapping_fn = slidescore_mapping_fn
        self.server_name_path_slidescore_id_map_fn = f"{dataset_dir}{server_id_slide_mapping_fn}"
        self.meta_cols = meta_data_cols_orig
        self.server_cols = server_cols

    def create_matched_metadata_csv(self) -> None:
        """
        1. For all available slides on the server in specified (sub-)directories, define
         a mapping between their filename (stem) and their path on the server as a dict.
        2. For these slides it also defines a dict between their slide_id on slidescore and their name.
        3. Get a mapping between server_names or paths and masks
        4. These mappings are saved to a common DataFrame slide_mapping_df
        5. Match metadata to the slide_mapping_df to create matched_metadata_fn_out
        """

        slides_paths_df = self.get_slide_paths_df()

        self.slidescore_mapping_fn = None
        if self.slidescore_mapping_fn is not None:
            slide_id_name_df = self.map_slidescore_id_to_slides(slides_paths_df)
        else:
            slide_id_name_df = slides_paths_df.copy()  # is a deep copy
            slide_id_name_df[self.server_cols.slidescore_id] = ""

        slide_mapping_df = self.map_masks_to_slides(slides_paths_df, slide_id_name_df)

        meta_df_matched = self.match_label_slides_to_available_slides(slide_mapping_df)


        slide_mapping_df.to_csv(self.server_name_path_slidescore_id_map_fn, index=False)
        meta_df_matched.to_csv(self.matched_metadata_path, index=False)

    def get_slide_paths_df(self):
        server_name_server_path_dict = data_utils.get_slide_paths_on_server(
            self.orig_data_dir, self.subdirs, self.img_ext, "images/"
        )
        slides_paths_df = (
            pd.DataFrame.from_dict(server_name_server_path_dict, orient="index")
            .reset_index()
            .set_axis([self.server_cols.name, self.server_cols.path], axis=1)
        )
        slides_paths_df = self.set_subdir_name(slides_paths_df)

        return slides_paths_df

    def set_subdir_name(self, slides_paths_df):
        if len(self.subdirs) > 1:
            slides_paths_df[self.server_cols.subdir] = slides_paths_df[self.server_cols.path].apply(
                lambda x: f"{Path(x).parts[-3]}/"
            )
        else:
            slides_paths_df[self.server_cols.subdir] = self.subdirs[0]

        return slides_paths_df

    def map_slidescore_id_to_slides(self, slides_paths_df):
        server_name_slidescore_id_dict = data_utils.get_slidename_id_dict(
            self.orig_data_dir, self.subdirs, self.slidescore_mapping_fn
        )
        slide_id_name_df = pd.DataFrame(
            server_name_slidescore_id_dict.items(), columns=[self.server_cols.name, self.server_cols.slidescore_id]
        )
        slide_id_mapping_df = slides_paths_df.merge(slide_id_name_df, how="outer", on=self.server_cols.name)
        if slide_id_mapping_df[self.server_cols.slidescore_id].isnull().sum() != 0:
            logging.info("Not all slides were mapped to slidescore id.")
            raise ValueError

        return slide_id_mapping_df

    def map_masks_to_slides(self, slides_paths_df, slide_id_name_df):
        """Create mapping of server_name, mask_name and subdir"""
        mask_name_path_dict = data_utils.get_slide_paths_on_server(
            self.derived_data_dir,
            self.subdirs,
            "tiff",
            "tiff_background_masks/",
        )
        if len(mask_name_path_dict) == 0:
            slide_id_name_df[self.server_cols.mask] = ""
            return slide_id_name_df
        else:
            mask_path_df = (
                pd.DataFrame.from_dict(mask_name_path_dict, orient="index")
                .reset_index()
                .set_axis([self.server_cols.name, self.server_cols.mask], axis=1)
            )
            slide_mask_mapping_df = slides_paths_df.merge(mask_path_df, how="outer", on=self.server_cols.name)

            if slide_mask_mapping_df[self.server_cols.mask].isnull().sum() != 0:
                null_mask_rows = slide_mask_mapping_df[slide_mask_mapping_df[self.server_cols.mask].isnull()]

                logging.warning(
                    f"\n\n Not all slides were mapped to mask. Masks should be provided for tile datasets. "
                    f"These are {null_mask_rows}\n\n."
                    f"Removing these slides from the mapping."
                )
                slide_mask_mapping_df = slide_mask_mapping_df[~slide_mask_mapping_df[self.server_cols.mask].isnull()]
            slide_mapping_df = slide_id_name_df.merge(
                slide_mask_mapping_df,
                how="inner",
                on=[self.server_cols.name, self.server_cols.path, self.server_cols.subdir],
            )
            return slide_mapping_df

    def match_label_slides_to_available_slides(self, slide_mapping_df: DataFrame) -> None:
        """
        Match server_name of slides to slidenames in metadata file.
        Store the matched metadata slides in dataframe.
        Merge these with slide_mapping df
        Save to csv.
        """
        # match metadata slide_names to server_names.
        server_slidenames = slide_mapping_df[self.server_cols.name].astype(str).to_list()
        sel_meta_df = pd.read_csv(f"{self.sel_metadata_fn}")

        # drop rows with no filename (slide_id) --> we set corrupted slides to nan when making metadata
        # Also nans can be slides only scanned on mrxs or svs
        sel_meta_df = sel_meta_df.dropna(subset=[self.meta_cols.slide_id])
        sel_meta_df[self.meta_cols.slide_id] = sel_meta_df[self.meta_cols.slide_id].astype(str)
        meta_slidenames = sel_meta_df[self.meta_cols.slide_id].to_list()

        meta_server_slidename_match_record = data_utils.match_input_slidenames_to_server_slidenames(
            server_slidenames=server_slidenames,
            meta_slidenames=meta_slidenames,
            dataset_name=self.dataset_name,
            subdirs=self.subdirs,
        )
        # filter meta_data by slidenames to which a server_slide has been mapped
        meta_server_slidename_match_df = pd.DataFrame(
            meta_server_slidename_match_record.items(), columns=[self.meta_cols.slide_id, self.server_cols.name]
        )
        # create target dataframe
        meta_df_matched = pd.merge(sel_meta_df, meta_server_slidename_match_df, on=self.meta_cols.slide_id, how="left")
        # Add slide_mapping_df columns to meta_df_matched ([ 'slide_path', 'subdir', 'imageID', 'mask'])
        meta_df_matched = pd.merge(meta_df_matched, slide_mapping_df, on=self.server_cols.name)

        return meta_df_matched

    def copy_data_to_scratch(self, slide_mapping_df: DataFrame) -> None:
        img_paths = slide_mapping_df[self.server_cols.path].to_list()
        dest_img_paths = data_utils.copy_files_to_scratch(img_paths, self.orig_data_dir, self.data_dir, self.img_ext)
        slide_mapping_df[self.server_cols.path] = dest_img_paths
        return slide_mapping_df
