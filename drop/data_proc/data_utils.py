import subprocess
import re
from typing import Generator, List, Optional, Tuple, Dict, TypeVar
DataFrame = TypeVar("pandas.core.frame.DataFrame")
from omegaconf import DictConfig
import logging
from pathlib import Path
import pandas as pd

def transform_categorical_columns(df, column_mapping):
    for column, mapping in column_mapping.items():
        df[column] = df[column].map(mapping).fillna(df[column])
    return df

def remove_duplicates(list_of_strings: List[str]) -> List[str]:
    """Remove duplicates from list vy turning it into a dict and then back into a list.
    Then check if there are samples that occur in only one of the 2 sets (the duplicates) with the symmetric difference operation.
    """
    list_duplicates_removed = list(dict.fromkeys(list_of_strings))
    duplicates = set(list_duplicates_removed) ^ set(list_of_strings)
    return list_duplicates_removed, duplicates


def copy_data_to_processing(orig_paths, scratch_img_paths, img_ext):
    """Copy all folders in orig_path to processing.
    Subprocess call waits until completion before continuing. We copy all files and directories, not just the ones
    with the correct extension because we need this for the mrxs format."""
    # include_files = f'--include="*.{img_ext}"'

    logging.info(
        f"Rsyncing {len(orig_paths)} files from {Path(orig_paths[0]).parent} to {Path(scratch_img_paths[0]).parent}."
    )
    for i, path in enumerate(orig_paths):
        subprocess.call(["rsync", "-ar", "--ignore-existing", path, scratch_img_paths[i]])


def get_slide_paths_on_server(orig_data_dir, subdirs, img_ext, img_folder) -> Dict[str, str]:
    """
      - Check available slides/masks in each image subdirectory on mount and scratch
      - If less images available in scratch than mount, rsync subdirectories (images and metadata/ masks) to scratch.
      - Collect the paths of all available files if available
      - Check for duplicates in path and remove.
      - Make slide/mask server_name server_path dict
    Parameters
       ------
       orig_data_dir : str
           Archive directory of images
      subdirs: str
          Subdirectories in orig_data_dir and data_dir which contain images and metadata folders
      img_ext: str
          Image extension of slides, depends on scanner type
      Returns
      ------
      slide_names_server_path_dict: Dict of slide names and image paths available on the server.
    """

    img_paths_server = {}
    for subdir in subdirs:
        orig_img_dir = f"{orig_data_dir}{subdir}{img_folder}"
        try:
            img_paths_orig = get_img_paths(orig_img_dir, img_ext)
        except:
            msg = f"No images with extension {img_ext} found in image directory {orig_img_dir}."
            logging.warning(msg)
            raise ValueError(msg)
        img_paths_orig, duplicates = remove_duplicates(img_paths_orig)
        logging.info(f"Server duplicates: {duplicates}.")
        img_paths_server[subdir] = img_paths_orig
    slide_names_server_path_dict = {}
    for subdir in subdirs:
        res = {Path(v).stem: v for v in img_paths_server[subdir]}
        slide_names_server_path_dict.update(res)
    return slide_names_server_path_dict


def match_orig_to_scratch_paths(img_paths_orig, orig_path, scratch_path):
    new_img_paths: List[str] = []
    for img_path_orig in img_paths_orig:
        try:
            new_img_paths.append(str(img_path_orig).replace(orig_path, scratch_path))
        except:
            breakpoint()
    return new_img_paths


def copy_files_to_scratch(img_paths_orig, orig_path, scratch_path, img_ext):
    """ "
    scratch_path: str
        Processing/ scratch directory of images (or it's the same as orig_data_dir, then files are loaded from there)
    """

    scratch_img_paths = match_orig_to_scratch_paths(img_paths_orig, orig_path, scratch_path)
    scratch_img_dir = Path(scratch_img_paths[0]).parent
    Path(scratch_img_dir).mkdir(parents=True, exist_ok=True)
    copy_data_to_processing(img_paths_orig, scratch_img_paths, img_ext=img_ext)

    # img_paths_dest = get_img_paths(scratch_img_dir, img_ext)
    # if len(img_paths_dest) != len(img_paths_orig):
    #     msg = f"Number of images in {scratch_img_dir} is smaller than in orig dir."
    #     logging.warning(msg)
    #     raise ValueError(msg)

    return scratch_img_paths


def get_img_paths(image_dir: str, img_ext: str) -> List[str]:
    """
    Collects all the image paths in a given directory in a list.
     Parameters
     ------
     image_dir : str
        Directory which contains images
    img_ext: str
        Image extension of slides, depends on scanner type
    Returns
    ------
    img_paths: List of image paths in image_dir
    """
    logging.warning(" we are exluding images in extra subfolders with maxdepth 2, so includes extra slides")
    # image_dir = image_dir + 'Extra/'
    bashCmd = ["find", image_dir, "-maxdepth", "2", "-type", "f", "-name", f"*.{img_ext}"]  # fixme now
    process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
    img_paths, error = process.communicate()
    img_paths = img_paths.decode("utf8").splitlines()
    return img_paths


def get_slidename_id_dict(data_dir: str, sub_dirs: str, fn: str) -> Dict[str, str]:
    """
    - Reads the slidescore id and name mapping from the metadata subdirectories
    - Stores all the information in a dictionary
    """

    slidescore_id_slide_name_dict = {}
    for sub_dir in sub_dirs:
        slidescore_id_slide_name_df_file = f"{data_dir}{sub_dir}metadata/{fn}"
        try:
            slide_id_name_df = pd.read_csv(
                slidescore_id_slide_name_df_file,
                sep="\t",
                names=["imageID", "imageName"],
                skipinitialspace=True,
                dtype=str,
            )
            slidescore_id_slide_name_dict.update(dict(zip(slide_id_name_df.imageName, slide_id_name_df.imageID)))
            logging.info("Slidescore mapping between image name and ID loaded for selected slides.")
        except NameError:
            logging.info("Slidescore mapping file not found")
    return slidescore_id_slide_name_dict


def is_server_name_matching(meta_fn, server_fn, max_diff):
    return re.match(meta_fn, server_fn) and (len(server_fn) - len(meta_fn)) <= max_diff


def filter_server_names(meta_fn, filtered_servernames, max_diff):
    return [server_fn for server_fn in filtered_servernames if is_server_name_matching(meta_fn, server_fn, max_diff)]


def no_matching_meta_fn(server_fn, meta_slidenames, max_diff):
    return all(not is_server_name_matching(meta_fn, server_fn, max_diff) for meta_fn in meta_slidenames)


def match_input_slidenames_to_server_slidenames(
    server_slidenames: List[str], meta_slidenames: List[str], dataset_name: str, subdirs: [str]
) -> List[str]:
    """
    - Preprocess slidenames from server.
    Remove certain slides with criteria depending on the dataset.
    For instance, for Sloane, there is a slide called '7080979 P-OUT OF FOCUS'. We split by " ", and remove if the second item is "P-OUT".
    - Preprocess slidenames from metadata. For the Precisio NKI dataset we change some things directly in the metadata slidenames for the matthcing.
    To get back to the original metadata name, we store the original meta slidename and the modified ones in a dict called meta_slide_name_dict.
    - We match meta_slidenames to server_slidenames using re (matches from beginning of strings).
    - We check which meta_slidenames could not be matched to files on the server
    - We record if there are multiple matches too.
    - Assert that the number of server slides (after removing unwanted slides) is equal to the matched slides
     + slides with multiple  matches + unmatched_slides.
     - Asser that the number of meta_slidenames is equal to the matched slidenames plus the meta_slidenames that were not found.


    Returns
    ------
    meta_server_slidename_match_record: Dict of meta_slidenames  as keys with matched server slide

    """
    orig_meta_slidenames = meta_slidenames.copy()

    if dataset_name == "Precision_Maartje":
        max_diff = 0
        filtered_servernames = server_slidenames
    elif dataset_name == "Precision_NKI_89_05":
        max_diff = 0
        filtered_servernames = server_slidenames
        # # remove file extension from slide name in metadata file
        meta_slidenames = [i.split(".")[0] for i in meta_slidenames]
    elif dataset_name == "Sloane":
        remove_slides_with_ending = ["P-OUT"]
        max_diff = 10
        # could potentially do this better by aggregating slides by their i.split(" ") value as a dictionary, then we don't need re.match
        filtered_servernames = [i for i in server_slidenames if i.split(" ")[1] not in remove_slides_with_ending]
    else:
        raise Exception(f"Incorrect datasetname")
    meta_slidenames_dict = {i: orig_meta_slidenames[meta_slidenames.index(i)] for i in meta_slidenames}

    meta_server_slidename_match_record = {
        meta_fn: filter_server_names(meta_fn, filtered_servernames, max_diff) for meta_fn in meta_slidenames
    }

    sel_slides_not_found = [
        (meta_fn, server_fn) for meta_fn, server_fn in meta_server_slidename_match_record.items() if not server_fn
    ]
    logging.info(
        f"No of slides from labels file that were not found: {len(sel_slides_not_found)}\n These are: {sel_slides_not_found}"
    )
    slides_with_multiple_matches_on_server = [
        (k, v) for k, v in meta_server_slidename_match_record.items() if len(v) > 1
    ]
    logging.info(
        f"For the following metadata slidename multiple slides on the server were found."
        f" {slides_with_multiple_matches_on_server}"
    )  # todo maybe exit if there are multiple matches? -yes

    meta_server_slidename_match_record = {
        k: v[0] for k, v in meta_server_slidename_match_record.items() if len(v) == 1
    }
    meta_slidenames_matched = list(meta_server_slidename_match_record.keys())
    extra_slides_on_server = [
        server_fn for server_fn in server_slidenames if no_matching_meta_fn(server_fn, meta_slidenames, max_diff)
    ]
    logging.info(
        f"No of slides in data folders that are not selected, so not in labels file: {len(extra_slides_on_server)}. \n These are: {extra_slides_on_server}"
    )
    assert len(meta_slidenames_matched) + len(slides_with_multiple_matches_on_server) + len(
        extra_slides_on_server
    ) == len(server_slidenames), "All server slides need to be assigned."

    assert len(meta_slidenames_matched) + len(sel_slides_not_found) == len(
        meta_slidenames
    ), "Number of selected slides should match number of found or not found slides on server."
    meta_server_slidename_match_record = {
        meta_slidenames_dict[k]: v for k, v in meta_server_slidename_match_record.items()
    }

    return meta_server_slidename_match_record



def create_meta_mapping(input_cols: DictConfig, target_cols: DictConfig) -> Dict[str, str]:
    """Creates a mapping between the input column names and the associated target column names. The target_cols object
    can also contain column mappings that are not input col names.
    """
    meta_data_col_mapping = {v: target_cols[k] for k, v in input_cols.items()}
    return meta_data_col_mapping


def rename_cols_using_match_dict(df: DataFrame, input_cols: DictConfig, target_cols: DictConfig):
    """ Requires that where the target_col value is the same as the input_col.target value,
     we remove the entry from target_col and input_col.
     """
    match_dict = create_meta_mapping(input_cols, target_cols)
    df = df.rename(match_dict, axis=1)
    return df
