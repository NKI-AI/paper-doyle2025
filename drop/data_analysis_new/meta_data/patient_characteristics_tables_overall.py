import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import TypeVar
import pandas as pd
DataFrame = TypeVar("pandas.core.frame.DataFrame")
from drop.data_analysis_new.meta_data.analyse_splits_per_group import *
from drop.utils.env_setup import setup_environment
setup_environment()
project='drop'


def get_patient_characteristics_filename(base_path, dataset_name, subset, clinical_features_name):
    return f"{base_path}patient_characteristics_{dataset_name}_{subset}{clinical_features_name}.csv"

def get_clinical_features_name(clinical_features):

    feature_map = {
        frozenset([]): "",
        frozenset(['er', 'her2', 'pr', 'grade', 'age_diagnose']): "_basic",
        frozenset(['er', 'her2', 'grade', 'age_diagnose']): "_basic_no_pr",
        frozenset(['er', 'her2', 'grade']): "_complete_grade_her2_er",
        frozenset(['er', 'her2', 'pr', 'grade', 'age_diagnose', 'cox2', 'p16']): "_extended",
        frozenset(['er', 'her2', 'grade', 'age_diagnose', 'cox2', 'p16']): "_extended_no_pr",
    }
    clinical_features_name = feature_map.get(frozenset(clinical_features), "unknown")
    return clinical_features_name

def get_patient_characteristics_table(cfg: DictConfig, base_path, analyse_split=True, analyse_folds=True):

    clinical_features_name = get_clinical_features_name(cfg.data.data_sel_params.drop_nas_in_cols)
    subset = 'all'
    print(clinical_features_name)
    data_prep = instantiate(cfg.data_prep)
    data_df, folds_df = data_prep.prepare_data()
    if project == 'drop':
        df = get_stats_by_risk_group(data_df)
    elif project == 'biomarker':
        df = get_stats_overall(data_df, cfg.data_prep.data_cfg.dataset_name)

    out_fn = get_patient_characteristics_filename(base_path, cfg.data_prep.data_cfg.dataset_name, subset, clinical_features_name)
    df.to_csv(out_fn, index=False)
    if cfg.data_prep.data_cfg.dataset_name == "Sloane":
        # merge sloane and precision
        out_fn_p = get_patient_characteristics_filename(base_path, 'Precision_NKI_89_05', subset,
                                                      clinical_features_name)
        df_precision = pd.read_csv(out_fn_p)
        df = pd.merge(df_precision, df, on="Patient Characteristics", how="outer")
        out_fn_p_s= get_patient_characteristics_filename(base_path, 'precision_sloane', subset,
                                                      clinical_features_name)
        df.to_csv(out_fn_p_s, index=False)
        breakpoint()

    if analyse_split:  # outer fold
        analyse_single_split(data_df,
                            stratify_on=cfg.data.data_sel_params.target,
                            partition_name=cfg.data.meta_data_cols_orig.split,
                            use_percentages=True,
                            clinical_features_name=clinical_features_name,
                            base_path=base_path
                            )

        # only successful after all individual splits have been analysed, so should work when split is the last kfold split
        k_folds = cfg.data.cv_splitter['cv_params'].kfolds
        split_name =  cfg.data.meta_data_cols_orig.split
        partition_name, split_number = split_name[:-1], int(split_name[-1])
        if split_number+ 1 == k_folds:
            get_averaged_outer_splits_characteristics(base_path,
                                                      stratify_on=cfg.data.data_sel_params.target,
                                                      partition_name=partition_name,
                                                      subset_name='all',
                                                      k_folds=cfg.data.cv_splitter['cv_params'].kfolds,
                                                      clinical_features_name=clinical_features_name,
                                                      use_percentages=True
            )



    if analyse_folds: # inner folds
        cv_params = cfg.data.cv_splitter['cv_params']
        # Should only make this with the target variable specified really
        merged_folds_df = pd.merge(folds_df, data_df, on=cv_params.group_on, how="left")
        analyse_inner_folds(merged_folds_df,
                            folds= cv_params.kfolds,
                            stratify_on=cv_params.stratify_on,
                            group_on=cv_params.group_on,
                            partition_name=cfg.data.meta_data_cols_orig.split,
                            clinical_features_name=clinical_features_name,
                            use_percentages=True,
                            base_path=base_path
                            )




@hydra.main(version_base="1.2", config_path="../../../configs/", config_name="config")
def main(cfg: DictConfig):
    analyse_split = False
    analyse_folds = False
    base_path = '/projects/drop/data_biomarker/'
    base_path = '/projects/drop/data_drop/'

    if cfg.data_prep.data_cfg.dataset_name == "Precision_NKI_89_05":
        analyse_split=True  # in the biomarker paper, we show the unstratified data splits
        if cfg.data.data_sel_params.target != '':
            analyse_folds=True

    get_patient_characteristics_table(cfg, base_path, analyse_split=analyse_split, analyse_folds=analyse_folds)



if __name__ == "__main__":
    main()
