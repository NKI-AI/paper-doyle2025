import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import os
import pandas as pd
from typing import TypeVar
DataFrame = TypeVar("pandas.core.frame.DataFrame")
# load environment paths
from drop.utils.env_setup import setup_environment
setup_environment()
from drop.data_analysis_new.meta_data.utils.metadata_analysis import get_median_observation_time
from drop.data_analysis_new.meta_data.analyse_splits_per_risk_group import calculate_chars_for_risk_group


def get_stats_overall(df, group):
    # Get abs and perc stats
    results =[]
    results.append(calculate_chars_for_risk_group(df, group))
    results.append(calculate_chars_for_risk_group(df, group, use_percentages=True))
    merged_df = pd.merge(results[0], results[1], on="var", how="outer")
    merged_df.columns = ['var', 'Abs', 'perc']

    # Format the column to include the second value in brackets
    merged_df[group] = merged_df.apply(lambda row: f"{row['Abs']} ({row['perc']})", axis=1)
    merged_df = merged_df[['var', group]]

    merged_df = merged_df.rename(columns={"var": "Patient Characteristics"})
    merged_df.fillna(0, inplace=True)
    return merged_df

def get_stats_by_risk_group(df):
    # Analyse relevant portion of the data iIBC and controls.
    median_obs, iqr_obs = get_median_observation_time(df)
    results = []
    for group in ["Low risk", "High risk"]:
        risk_val = 0 if group=="Low risk" else 1
        risk_df = df[df["outcome"] == risk_val]
        risk_values = calculate_chars_for_risk_group(risk_df, group)
        results.append(risk_values)

    merged_df = pd.merge(results[0], results[1], on="var", how="outer")
    merged_df = merged_df.rename(columns={"var": "Patient Characteristics"})
    merged_df.fillna(0, inplace=True)

    return merged_df

def get_patient_characteristics_table(cfg: DictConfig, clinical_features: str, project:str, split: bool = False):
    base_path = os.getcwd().split("DROP")[0]
    cfg.data_prep.data_cfg.cv_splitter = False
    if clinical_features == "none":
        CLINICAL_FEATURES = []
    elif clinical_features == "basic":
        CLINICAL_FEATURES = ['er', 'her2', 'pr', 'grade', 'age_diagnose']
    elif clinical_features == "extended":
        CLINICAL_FEATURES = ['er', 'her2', 'pr', 'grade', 'age_diagnose', 'cox2', 'p16', ]
    cfg.data.data_sel_params.drop_nas_in_cols = CLINICAL_FEATURES
    data_prep = instantiate(cfg.data_prep)
    sel_data_df, folds_df = data_prep.prepare_data()
    if split:
        train_df = sel_data_df.loc[sel_data_df["split"] == "train"]
        test_df  = sel_data_df.loc[sel_data_df["split"] == "test"]
    if project == 'drop':
        df = get_stats_by_risk_group(sel_data_df)
    elif project == 'biomarker':
        df = get_stats_overall(sel_data_df, cfg.data_prep.data_cfg.dataset_name)
    if cfg.data_prep.data_cfg.dataset_name == "Precision_NKI_89_05":
        df.to_csv(f"{base_path}/patient_characteristics_precision_dropNAs_in_clinical_{clinical_features}.csv",
                  index=False)
        if split:  # if there is just one split
            train_df = get_stats_by_risk_group(train_df)
            train_df.to_csv(
                f"{base_path}/patient_characteristics_precision_dropNAs_in_clinical_{clinical_features}_train_part.csv",
                index=False)
            test_df = get_stats_by_risk_group(test_df)
            test_df.to_csv(
                f"{base_path}/patient_characteristics_precision_dropNAs_in_clinical_{clinical_features}_test_part.csv",
                index=False)
    elif cfg.data_prep.data_cfg.dataset_name == "Sloane":
        df.to_csv(f"{base_path}/patient_characteristics_sloane_dropNAs_in_clinical_{clinical_features}.csv",
                  index=False)


@hydra.main(version_base="1.2", config_path="../../../configs/", config_name="config")
def main(cfg: DictConfig):
    cfg.data_prep.data_cfg.cv_splitter = False
    project = "DROP"
    get_patient_characteristics_table(cfg, "none", project)
    if project == "drop":
        get_patient_characteristics_table(cfg, "basic")
        get_patient_characteristics_table(cfg, "extended")

def get_final_table(base_path):
    # merge sloane and precision
    df1 = pd.read_csv(f"{base_path}/patient_characteristics_precision_dropNAs_in_clinical_none.csv")
    df2 = pd.read_csv(f"{base_path}/patient_characteristics_sloane_dropNAs_in_clinical_none.csv")
    df = pd.merge(df1, df2, on="Patient Characteristics", how="outer")
    df.to_csv(f"{base_path}patient_characteristics_sloane_precision.csv", index=False)

    # merge sloane and precision for clinical basic
    df1 = pd.read_csv(f"{base_path}/patient_characteristics_precision_dropNAs_in_clinical_basic.csv")
    df2 = pd.read_csv(f"{base_path}/patient_characteristics_sloane_dropNAs_in_clinical_basic.csv")

    df = pd.merge(df1, df2, on="Patient Characteristics", how="outer")
    df.to_csv(f"{base_path}patient_characteristics_sloane_precision_clinical_basic.csv", index=False)

    return df


if __name__ == "__main__":
    base_path = os.getcwd().split("DROP")[0]
    main()
    get_final_table(base_path)
