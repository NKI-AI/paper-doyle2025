from typing import TypeVar, Optional, List, Tuple
from omegaconf import DictConfig
DataFrame = TypeVar("pandas.core.frame.DataFrame")
import numpy as np
import scipy.stats as stats
import pandas as pd
from drop.data_analysis_new.meta_data.utils.metadata_analysis import *

def process_dfs_with_optional_percentages(
    dfs: list,
    excluded_dfs: list,
    group: str
):
    """
    Process a list of DataFrames with optional percentage calculation, excluding specified DataFrames.

    Parameters:
    - dfs (list): List of DataFrames to process.
    - use_percentages (bool): Whether to apply percentages to group columns.
    - excluded_dfs (list): List of DataFrames to exclude from percentage calculation.
    - group (str): The column name to calculate percentages on.

    Returns:
    - list: Processed DataFrames with or without percentages.
    """
    # Ensure original order is maintained
    processed_dfs = []

    for df in dfs:
        if not any(id(df) == id(excluded_df) for excluded_df in excluded_dfs):
            try:
                df[group] = round(df[group] / sum(df[group]), 2)
            except Exception as e:
                print(f"Error processing percentages for {df}: {e}")
        # Append the DataFrame (processed or unchanged)
        processed_dfs.append(df)

    return processed_dfs


def calculate_chars_for_group(df, group, use_percentages=False):
    # Helper function to safely get stats
    def safe_get_stats(get_func, *args, default=None, **kwargs):
        try:
            return get_func(*args, **kwargs)
        except:
            return default

    # Analyse relevant portion of the data iIBC and controls.
    total_df = get_total_samples(df, group)
    try:
        fu_df = get_follow_up_stats(df, group)
    except:
        fu_df = pd.DataFrame()
    deceased_df = safe_get_stats(get_deceased_stats, df, group, default=pd.DataFrame())
    outcome_df = get_outcome_stats(df, group)
    age_df = get_age_stats(df, group)
    age_slide_df = safe_get_stats(get_age_slide_stats, df, group, default=pd.DataFrame())
    grade_df = get_grade_stats(df, "grade", group)
    her2_df = get_her2_stats(df, group)
    er_df = get_rec_stats(df, "er", group)
    pr_df = get_rec_stats(df, "pr", group)

    # Handle cases for cox2, p16, and p53
    cox2_df = safe_get_stats(get_grade_stats, df, "cox2", group, default=pd.DataFrame())
    if cox2_df.empty:  # Check fallback name
        cox2_df = safe_get_stats(get_grade_stats, df, "cox2_score", group, default=pd.DataFrame())

    p16_df = safe_get_stats(get_rec_stats, df, "p16", group, default=pd.DataFrame())
    if p16_df.empty:  # Check fallback name
        p16_df = safe_get_stats(get_rec_stats, df, "p16_conclusion", group, default=pd.DataFrame())

    p53_df = safe_get_stats(get_p53_stats, df, group, default=pd.DataFrame())

    # Combine all DataFrames
    dfs = [total_df, fu_df, outcome_df, age_df, deceased_df, age_slide_df, grade_df, her2_df, er_df, pr_df, cox2_df, p16_df,
           p53_df]

    if use_percentages:
        excluded_dfs = [fu_df, age_df, age_slide_df]
        dfs = process_dfs_with_optional_percentages(
            dfs=dfs,
            excluded_dfs=excluded_dfs,
            group=group
        )

    # Check that the number of patients in each variable adds up to the total
    check_dfs(dfs, len(df), group)

    # Concatenate all DataFrames into a single DataFrame
    char_values = pd.concat(dfs, axis=0, ignore_index=True)
    return char_values



def get_fold_char_table_path(base_path, split_type, stratify_on, partition_name, subset_name, clinical_features_name,
                             use_percentages):
    strat_on_suffix = f'_strat_on_none' if stratify_on == '' else f'_strat_on_{stratify_on}'
    percentages_suffix = '' if not use_percentages else '_percentages'
    suffix = f"_{subset_name}" if subset_name != "all" else ""
    out_fn = f'{base_path}{partition_name}_{split_type}_characteristics{strat_on_suffix}{clinical_features_name}{suffix}{percentages_suffix}.csv'
    return out_fn


def get_averaged_fold_char_table(base_path, split_type, stratify_on, partition_name, subset_name, clinical_features_name,
                             use_percentages):
    strat_on_suffix = f'_strat_on_none' if stratify_on == '' else f'_strat_on_{stratify_on}'
    percentages_suffix = '' if not use_percentages else '_percentages'
    suffix = f"_{subset_name}" if subset_name != "all" else ""
    out_fn = f'{base_path}{partition_name}_{split_type}_characteristics_averaged{strat_on_suffix}{clinical_features_name}{suffix}{percentages_suffix}.csv'
    return out_fn

def calculate_mean_and_ci(row, columns):
    values = row[columns].values
    values = pd.to_numeric(values, errors='coerce')  # convert to numeric, turn invalid strings into NaN
    mean = np.mean(values)

    std_error = stats.sem(values)  # Standard error of the mean
    if std_error > 0:
        ci_lower, ci_upper = stats.t.interval(
            0.95, len(values) - 1, loc=mean, scale=std_error
        )
    else:  # Zero standard error (all values are identical)
        ci_lower, ci_upper = mean, mean
    mean, ci_lower, ci_upper = mean.round(2), ci_lower.round(2), ci_upper.round(2)
    return mean, (ci_lower, ci_upper)

def create_average_df(merged_df):
    # Select the train and val columns
    train_columns = [col for col in merged_df.columns if "train" in col]
    val_columns = [col for col in merged_df.columns if "val" in col]

    # Apply the function to each row for train and val columns
    merged_df[['train_mean', 'train_95%CI']] = merged_df.apply( lambda row: pd.Series(calculate_mean_and_ci(row, train_columns)), axis=1 )

    merged_df[['val_mean', 'val_95%CI']] = merged_df.apply(
        lambda row: pd.Series(calculate_mean_and_ci(row, val_columns)), axis=1
    )
    res= merged_df[["Patient Characteristics", 'train_mean', 'train_95%CI', 'val_mean', 'val_95%CI']]
    return res


