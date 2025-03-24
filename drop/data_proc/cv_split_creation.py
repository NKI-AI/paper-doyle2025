import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
from omegaconf import DictConfig, ListConfig
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold
from drop.tools.json_saver import JsonSaver

""""
Relies on split columnn to be called split and values to be train, val or test or None.
"""

class MIL_CVSplitter:
    """Creates a CV split for the given dataset."""

    def __init__(
        self,
        output_dir: str,
        cv_split_fn: str,
        cv_params: DictConfig,
    ):
        self.cv_split_path = f"{output_dir}{cv_split_fn}"
        self.cv_params = {"cross_validatation_splits": cv_params}
        self.json_split_saver = JsonSaver("params", self.cv_split_path)
        self.strategy = cv_params.strategy
        self.kfolds = cv_params.kfolds
        self.stratify_on = cv_params.stratify_on  # is usually the target col
        self.group_on = (
            cv_params.group_on
        )
        if self.strategy == "StratifiedGroupKFold":
            self.CV_splitter = StratifiedGroupKFold(n_splits=self.kfolds)
        elif self.strategy == "StratifiedKFold":
            self.CV_splitter = StratifiedKFold(n_splits=self.kfolds)
        elif self.strategy == "GroupKFold":
            self.CV_splitter = GroupKFold(n_splits=self.kfolds)

        else:
            logging.log("Provide valid CV splitting strategy")

    def __call__(self, data_df, regions_col: Optional[str] = None, split_col: Optional[str] = 'split'):
        ''' The data prep pipeline renames the orginal split col to split.'''
        cv_df = data_df.copy()
        cv_df = self.drop_test_split_data(cv_df, split_col)

        # for stratification on multiple variables
        if isinstance(self.stratify_on, List) or isinstance(self.stratify_on, ListConfig):
            cv_df[str(self.stratify_on)] = cv_df[self.stratify_on].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

        if regions_col != None:
            logging.info("Creating CV splits based on regions.")
            targets, groups, X = self.get_targets_groups_for_regions(cv_df, regions_col)
        else:
            logging.info("Creating CV splits based on slides.")
            targets, groups, X = self.get_targets_groups_for_slides(cv_df)

        folds_dict = {self.group_on: dict(enumerate(groups))}

        for f, (tr_idx, val_idx) in enumerate(self.CV_splitter.split(X=X, y=targets, groups=groups)):
            folds_dict[str(f)] = dict.fromkeys(tr_idx.tolist(), "train")
            folds_dict[str(f)].update(dict.fromkeys(val_idx.tolist(), "val"))

        self.get_splits_stats(folds_dict)
        logging.info("\n\n Created New CV splits. \n\n")
        return folds_dict

    def drop_test_split_data(self, cv_df, split_col):
        if split_col in cv_df.columns:  # should be renamed to split for whatever it was before
            logging.info("Dropping test split from cross validation")
            cv_df = cv_df.drop(cv_df[cv_df[split_col] == "test"].index)
        else:
            logging.warning("No test split was dropped before making folds")
        return cv_df

    def get_targets_groups_for_slides(self, cv_df):

        if isinstance(self.stratify_on, List) or isinstance(self.stratify_on, ListConfig):
            targets = cv_df[str(self.stratify_on)].to_list()
            print(targets)
        elif isinstance(self.stratify_on, str) and self.stratify_on != '':
            targets = cv_df[self.stratify_on].to_list()
        else:
            targets = None

        if isinstance(self.group_on, str) and self.group_on != '':
            groups = cv_df[self.group_on].to_list()
        else:
            groups = None
        slides = groups  #given one row per slide

        return targets, groups, slides


    def get_targets_groups_for_regions(
        self, cv_df, regions_col
    ):
        """For every region, record its outcome (or target) and the associated group (or slide in our case).
        """
        cv_df["no_regions"] = cv_df[regions_col].apply(lambda x: len(x))
        cv_df["targets"] = cv_df.apply(lambda x: np.repeat(x[str(self.stratify_on)], x["no_regions"]).tolist(), axis=1)
        cv_df[self.group_on] = cv_df.apply(lambda x: np.repeat(x[self.group_on], x["no_regions"]).tolist(), axis=1)
        targets = np.array(sum(cv_df["targets"].to_list(), []))
        groups = np.array(sum(cv_df[self.group_on].to_list(), []))
        regions = sum(cv_df[regions_col].to_list(), [])

        return targets, groups, regions


    def get_splits_stats(self, folds_dict):
        # map slide back and summarise per slide_id - if this gives an error then groups have been mixed.
        folds_df = pd.DataFrame(folds_dict)
        # for data_analysis:
        sum_folds_df = folds_df.groupby([self.group_on]).agg(
            {str(k): lambda x: " ".join(map(str, set(x))) for k in range(self.CV_splitter.n_splits)}
        )
        print(sum_folds_df)

    def save_cv_splits(self, folds_dict: Dict, data_params: Optional[DictConfig] = None):
        if data_params is not None:
            print(data_params)
            self.update_cv_params(data_params)
        self.json_split_saver.save_selected_data(self.cv_params, "data", folds_dict)

    def update_cv_params(self, data_params: DictConfig):
        self.cv_params.update(data_params)
