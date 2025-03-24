from omegaconf import DictConfig, ListConfig
from typing import Union, Tuple, TypeVar, List, Optional, Any
import pandas as pd
DataFrame = TypeVar("pandas.core.frame.DataFrame")
import logging
from drop.tools.json_saver import JsonSaver

class DataFactory:
    def __init__(self, dataset_cfg: DictConfig):
        self.dataset_cfg = dataset_cfg
        self.slide_mapper = self.dataset_cfg.slide_mapping
        self.task_data_prepper = self.dataset_cfg.task_data_prep(data_sel_params=self.dataset_cfg.data_sel_params)

    def prepare_slide_data_for_task(self):
        self.slide_mapper.create_matched_metadata_csv()
        self.task_data_prepper.create_filtered_slide_data_df()

    def copy_data_to_scratch(self, df: DataFrame):
        self.slide_mapper.copy_data_to_scratch(df)
        return df


class DataPrep:
    def __init__(
        self,
        data_cfg: DictConfig,
        task_dir: str,
        data_fn: str,
        construct_dataset=False,
        input_regions=False,
        analyse_split=False,
        analyse_folds=False,
    ):
        self.construct_dataset = construct_dataset
        self.input_regions = input_regions
        self.analyse_split = analyse_split
        self.analyse_folds = analyse_folds
        self.cv_splitter = data_cfg.cv_splitter
        self.data_cols = data_cfg.data_cols
        self.dataset_name = data_cfg.dataset_name
        self.data_name = data_cfg.name
        self.data_sel_params = data_cfg.data_sel_params
        self.data_sel_strategy = data_cfg.data_sel_strategy
        # set up the json saver
        dataset_file_path = f"{task_dir}{self.data_name}_{data_fn}"
        self.json_saver = JsonSaver("params", dataset_file_path)
        self.datasets = self.create_dict_of_datasets(data_cfg)
        self.data_params = {"datasets": list(self.datasets)}  # gets keys
    """
    This function prepares the data for the task.
    If construct_dataset is True, it will create the slide mapping, task data, regions data and cv splits.
    Otherwise it will read the data based on the selection criteria from a json file. To read the correct data,
     the task preparation parameters must be set in the task_data_prepper.
     They are optionally updated if input regions are used- therefore we cannot directly read it from the cfg
    CV splits are created and stored with the cv_params, slide_params and optionally region_params in a json file.
    """

    def update_data_params(self, new_data_params):
        for key, value in new_data_params.items():
            if key not in self.data_params.keys():
                self.data_params[key] = value
            else:
                if type(value) == dict or type(value) == DictConfig:
                    self.data_params[key].update(value)
                elif type(value) == list or type(value) == ListConfig:
                    self.data_params[key].extend(value)
                else:
                    self.data_params[key] = value

    def prepare_data(self) -> Tuple[DataFrame, Union[DataFrame, dict]]:
        data_df = []
        for dataset_cfg in list(self.datasets.values()):
            dataset = DataFactory(dataset_cfg)
            if self.construct_dataset:
                dataset.prepare_slide_data_for_task()
                self.update_data_params(dataset.task_data_prepper.task_params)
            else:
                self.update_data_params(dataset.task_data_prepper.task_params)

            sel_data_df = dataset.task_data_prepper.read_datadf_using_strategy()
            data_df.append(sel_data_df)

        data_df = pd.concat(data_df).reset_index(drop=True) if len(data_df) > 1 else data_df[0]
        logging.info(data_df)
        self.json_saver.save_selected_data(self.data_params, "data", data_df.to_dict())

        if self.analyse_split:
            from drop.data_analysis_new.meta_data.analyse_splits_per_risk_group import analyse_data_splits
            analyse_data_splits(data_df,
                              stratify_on= self.cv_splitter.stratify_on,
                              split_col=dataset_cfg.meta_data_cols_orig.split,  # will contain outer split number: split_non_rt_only$OUTER_SPLIT
                              data_sel_params = self.data_sel_params
                              )

        folds_df = None
        if self.cv_splitter:
            self.cv_splitter.update_cv_params(self.data_params)  # should be the same for both datasets
            if self.construct_dataset:
                regions_col = None
                folds_dict = self.make_cv_splits(regions_col, split_col="split")
            else:
                folds_dict = self.get_folds()
            folds_df = pd.DataFrame(folds_dict)  # the index is the region_index
            logging.info(folds_df)

        if self.analyse_folds:
            from drop.data_analysis_new.meta_data.analyse_splits_per_risk_group import call_folds_analysis
            call_folds_analysis(
                folds_df,
                data_df,
                self.cv_splitter.stratify_on,
                self.cv_splitter.group_on,
                self.cv_splitter.kfolds,
                dataset_cfg.meta_data_cols_orig.split,

            )
        return data_df, folds_df

    def create_dict_of_datasets(self, data_cfg: DictConfig):
        """ Make a dict of datasets from the data_cfg, if there is only one, create a dict for the dataset
        Update the datasets with the general data params.
        """
        datasets = {
            ds_name: dataset
            for ds_name, dataset in data_cfg.items()
            if ds_name in ["Block1_Aperio", "Block1_P1000"]
        }
        if len(datasets) == 0:
            datasets = {data_cfg["name"]: data_cfg}
        return datasets

    def make_cv_splits(self, regions_col: Optional[str] = None, split_col=None):
        relevant_params_dict, _ = self.json_saver.read_selected_data(self.data_params)
        data_df = pd.DataFrame(relevant_params_dict["data"])
        folds_dict = self.cv_splitter(data_df, regions_col, split_col)  # test that this works -need to make new config
        self.cv_splitter.save_cv_splits(folds_dict, self.data_params)
        return folds_dict

    def get_folds(self) -> DataFrame:
        relevant_criterium_dict, _ = self.cv_splitter.json_split_saver.read_selected_data(self.cv_splitter.cv_params)
        return relevant_criterium_dict["data"]
