import logging
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Dict, TypeVar, Any
DataFrame = TypeVar("pandas.core.frame.DataFrame")

import drop.data_proc.metadata_selection as meta_data_sel
from drop.data_proc.data_subselection import SubSelector
from drop.tools.json_saver import JsonSaver
from drop.data_proc import data_utils





def perform_slide_subselection(slide_meta_df: DataFrame, slide_col: str, slide_sel_params) -> DataFrame:
    slide_selector = SubSelector()
    sub_selection = slide_selector(list(slide_meta_df[slide_col]), slide_sel_params)
    sel_data_df = slide_meta_df[slide_meta_df[slide_col].isin(sub_selection)]
    return sel_data_df


class TaskDataPrepper:
    """
    - Filters Matched metadata file by specific selection criteria (target data and exclusion criteria)
    - Subselects slides based on slide sampling strategy
    - Saves to json
    - Collect post-processed regions
    - Prepares dataframe by only keeping relevant columns (including: Patient id, slide_Id, outcome and split if applicable, etc)
    - Saves to json
    """

    def __init__(
        self,
        dataset_name: str,
        data_name: str,
        subdirs: List[str],
        task_data_fn: str,
        task_dir: str,
        matched_metadata_path: str,
        meta_data_cols_orig: DictConfig,
        data_cols: DictConfig,
        data_sel_params: Dict[str, bool],
        data_sel_strategy: Dict[str, str],
    ) -> None:
        sel_data_path = f"{task_dir}{data_name}/{task_data_fn}"
        Path(f"{task_dir}{data_name}/").mkdir(parents=True, exist_ok=True)
        self.matched_data_path = matched_metadata_path
        self.data_sel_params = data_sel_params
        self.task_params = {"subdirs": subdirs, "data_selection": self.data_sel_params}
        self.data_sel_strategy = data_sel_strategy
        slide_params = self.data_sel_strategy.slides
        self.task_params.update({"slides": slide_params})
        self.json_saver = JsonSaver("params", sel_data_path)
        self.data_sel_strategy = data_sel_strategy
        self.dataset_name = dataset_name
        self.data_name = data_name  # if there are multiple studies in the same dataset
        self.orig_meta_cols = meta_data_cols_orig
        self.data_cols = data_cols
        self.server_cols = data_cols.server
        self.meta_cols = data_cols.meta
        self.keep_cols = list({**self.server_cols, **self.meta_cols}.values())


    def process_data(self, matched_data_df: DataFrame, criteria) -> DataFrame:
        """
        - Prepares data for further processing (dataset specific). In detail: Column values are encoded as integers, outcome info is added.
        - Renames columns to standardised column names for further processing. (e.g. "outcome" instead of "first_subseq_event"). Also
        removes the target variable as a meta variable.
        - Ensures that all required are there (split and target), as well as recurrence related columns (vital_status and time_to_event).
        columns are present.
        - Performs data filtering based on the given criteria.
        - keeps only relevant columns that are also available in the metadata.

        """
        data_df = meta_data_sel.prepare_data(matched_data_df, self.dataset_name, self.data_sel_params)
        data_df = data_utils.rename_cols_using_match_dict(data_df, self.orig_meta_cols,  self.meta_cols)
        data_df = meta_data_sel.select_data(data_df,
                                  target=criteria.target,
                                  cutoff_years=criteria.cutoff_years,
                                  exclude_radiotherapy=criteria.exclude_radiotherapy,
                                  drop_nas_in_cols=criteria.drop_nas_in_cols)

        # make column for split in data_df, if no split col is specified under split key in orig metadata cols dictionary
        if self.orig_meta_cols.split is None:
            data_df[self.meta_cols.split] = None

        if criteria.target == "outcome":
            self.keep_cols.extend(["time_to_event", "months_followup"])  # months_followup mainly for data analysis
            if 'vital_status' in data_df.columns:
                self.keep_cols.extend(["vital_status"])

        # keep only specified columns that also exist in the metadata
        remove_cols = [col for col in self.keep_cols if col not in data_df.columns]
        self.keep_cols = [col for col in self.keep_cols if col in data_df.columns]
        if 'age_slide' in data_df.columns:
            self.keep_cols.extend(['age_slide'])
        if 'year_diagnose' in data_df.columns:  # not in Sloane
            self.keep_cols.extend(['year_diagnose'])
        logging.warning(f"Removed {remove_cols} from keep_cols, as they are not in the original metadata.")
        data_df = data_df[self.keep_cols].reset_index(drop=True)
        return data_df

    def create_filtered_slide_data_df(self) -> None:
        """Reads matched datafile (containing slide info and metadata).
        It is important to read the names of
         slides as string, so no data names get shortened to integers.
        Perform filtering based on patient criteria and exclusion criteria.
        Do slide subselection based on data selection, then add selected slides to the json.
        """
        matched_data_df = pd.read_csv(
            self.matched_data_path,
            dtype={self.server_cols.name: str, self.orig_meta_cols.slide_id: str},
            keep_default_na=False,
        )
        sel_data_df = self.process_data(matched_data_df, self.data_sel_params)
        sub_selection_df = perform_slide_subselection(
            sel_data_df, self.server_cols.name, self.data_sel_strategy.slides
        )
        self.json_saver.save_selected_data(self.task_params, "data", sub_selection_df.to_dict())

    def match_to_data_df(self, new_df: DataFrame, keep_cols: List[str]) -> DataFrame:
        """Add regions to data_df based on slide name."""
        data_df = self.read_datadf_using_strategy()
        keep_cols= keep_cols + [self.server_cols.name]
        df_matched = data_df.merge(new_df[keep_cols], on=self.server_cols.name, how="left")
        df_matched = df_matched.dropna(subset=keep_cols)
        logging.warning(f"Number of rows dropped due to missing regions: {len(data_df) - len(df_matched)}")
        return df_matched


    def read_datadf_using_strategy(self) -> DataFrame:
        """
        Read the dataframe by filtering for the applied parameters for constructing the data.
        """
        relevant_params_dict, _ = self.json_saver.read_selected_data(self.task_params)
        df = pd.DataFrame(relevant_params_dict["data"])
        df.index = df.index.astype(int)
        df[self.server_cols.name] = df[self.server_cols.name].astype(str)
        if self.server_cols.slidescore_id in df.columns:
            df[self.server_cols.slidescore_id] = df[self.server_cols.slidescore_id].astype(str)
        return df
