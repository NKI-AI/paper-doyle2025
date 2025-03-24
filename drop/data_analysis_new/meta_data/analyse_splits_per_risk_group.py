from typing import TypeVar
import pandas as pd
DataFrame = TypeVar("pandas.core.frame.DataFrame")
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


def calculate_chars_for_risk_group(risk_df, group, use_percentages=False):
    # Helper function to safely get stats
    def safe_get_stats(get_func, *args, default=None, **kwargs):
        try:
            return get_func(*args, **kwargs)
        except:
            return default

    # Analyse relevant portion of the data iIBC and controls.
    try:
        fu_df = get_follow_up_stats(risk_df, group)
    except:
        fu_df = pd.DataFrame()
    deceased_df = safe_get_stats(get_deceased_stats, risk_df, group, default=pd.DataFrame())
    outcome_df = get_outcome_stats(risk_df, group)
    age_df = get_age_stats(risk_df, group)
    age_slide_df = safe_get_stats(get_age_slide_stats, risk_df, group, default=pd.DataFrame())
    grade_df = get_grade_stats(risk_df, "grade", group)
    her2_df = get_her2_stats(risk_df, group)
    er_df = get_rec_stats(risk_df, "er", group)
    pr_df = get_rec_stats(risk_df, "pr", group)

    # Handle cases for cox2, p16, and p53
    cox2_df = safe_get_stats(get_grade_stats, risk_df, "cox2", group, default=pd.DataFrame())
    if cox2_df.empty:  # Check fallback name
        cox2_df = safe_get_stats(get_grade_stats, risk_df, "cox2_score", group, default=pd.DataFrame())

    p16_df = safe_get_stats(get_rec_stats, risk_df, "p16", group, default=pd.DataFrame())
    if p16_df.empty:  # Check fallback name
        p16_df = safe_get_stats(get_rec_stats, risk_df, "p16_conclusion", group, default=pd.DataFrame())

    p53_df = safe_get_stats(get_p53_stats, risk_df, group, default=pd.DataFrame())

    # Combine all DataFrames
    dfs = [fu_df, outcome_df, age_df, deceased_df, age_slide_df, grade_df, her2_df, er_df, pr_df, cox2_df, p16_df,
           p53_df]

    if use_percentages:
        excluded_dfs = [fu_df, age_df, age_slide_df]
        dfs = process_dfs_with_optional_percentages(
            dfs=dfs,
            excluded_dfs=excluded_dfs,
            group=group
        )

    # Check that the number of patients in each variable adds up to the total
    check_dfs(dfs, len(risk_df), group)

    # Concatenate all DataFrames into a single DataFrame
    risk_values = pd.concat(dfs, axis=0, ignore_index=True)
    return risk_values


def get_combined_char_table_train_val_per_risk_group(df_train, df_test, use_percentages):

    # Analyse relevant portion of the data iIBC and controls.
    results_df = []
    test_col = 'test' if 'test' in df_test.columns else 'val'
    for (group, risk_df) in [('train', df_train), (test_col, df_test)]:
        risk_values = calculate_chars_for_risk_group(risk_df, group, use_percentages)
        results_df.append(risk_values)
    df = pd.merge(results_df[0], results_df[1], on="var", how="outer")
    df = df.rename(columns={"var": "Patient Characteristics"})
    df.fillna(0, inplace=True)

    # Add patient counts and combine
    n_patients = pd.DataFrame({
        'Patient Characteristics': ["n patients"],
        'train': [len(df_train)],
        'val': [len(df_test)]
    })
    df_char = pd.concat([n_patients, df], ignore_index=True)
    return df_char


def analyse_data_splits(
        df: DataFrame,
        stratify_on: str,
        split_col: str,
        data_sel_params: DictConfig,
        base_path: str = "/projects/drop/",
        subsets: dict = None,
        folds: int = None,
        partition_name: str = "split",
):
    """
    Analyze dataset characteristics for standard splits or cross-validation folds.

    Parameters:
    - df (DataFrame): Input data.
    - stratify_on (str): Column to stratify on.
    - split_col (str): Column name for splits.
    - data_sel_params (DictConfig): Configuration for column selections.
    - base_path (str): Base directory for output files.
    - subsets (dict): Dictionary of subset filters (e.g., {"all": None, "er_0": df['er'] == 0}).
    - folds (int, optional): Number of folds for cross-validation analysis. If None, assumes standard train/test split.
    - group_on (str, optional): Column for checking group overlaps (only for folds analysis).
    - partition_name (str, optional): Name of the partition (default: "split").
    - save_plot (bool, optional): Whether to save plots for fold analysis.
    """
    # Determine suffix based on data selection parameters
    clinical_sets = {
        frozenset(['er', 'her2', 'grade', 'age_diagnose', 'pr']): "_clinical_basic",
        frozenset(['er', 'her2', 'grade', 'age_diagnose', 'pr', 'p16', 'cox2']): "_clinical_extended"
    }
    add = clinical_sets.get(frozenset(data_sel_params.drop_nas_in_cols), "")

    subsets = subsets or {"all": None}
    aggregated_results = {}

    def process_subset(df_subset, suffix: str):
        """Processes a subset, computing and saving characteristics."""
        char_dfs = []

        if folds is None:
            df_fit = df_subset.loc[df_subset[split_col] == "train"]
            df_test = df_subset.loc[df_subset[split_col] == "test"]
            char_df = get_combined_char_table_train_val_per_risk_group(df_fit, df_test, use_percentages=False)
            char_df.to_csv(f'{base_path}split_characteristics_{stratify_on}{split_col}{add}{suffix}.csv', index=False)
            char_df = get_combined_char_table_train_val_per_risk_group(df_fit, df_test, use_percentages=True)
            char_df.to_csv(f'{base_path}split_characteristics_{stratify_on}{split_col}{add}{suffix}_percentages.csv',
                           index=False)
            return char_df

        for fold in range(folds):
            df_train = df_subset.loc[df_subset[str(fold)] == "train"]
            df_val = df_subset.loc[df_subset[str(fold)] == "val"]

            df_char_fold = get_combined_char_table_train_val_per_risk_group(df_train, df_val, use_percentages=True)
            df_char_fold.columns = [f'{col}_fold{fold}' if col != 'Patient Characteristics' else col for col in
                                    df_char_fold.columns]
            char_dfs.append(df_char_fold)

        merged_df = pd.concat(char_dfs, axis=1).fillna("-")
        merged_df.to_csv(
            f'{base_path}{partition_name}_characteristics_{stratify_on}{split_col}{add}{suffix}_percentages.csv',
            index=False)
        return merged_df

    for subset_name, condition in subsets.items():
        subset_df = df.loc[condition] if condition is not None else df
        suffix = f"_{subset_name}" if subset_name != "all" else ""
        aggregated_results[subset_name] = process_subset(subset_df, suffix)

    return aggregated_results


def call_folds_analysis(
    folds_df: DataFrame, data_df: DataFrame, stratify_on: str, group_on: str, kfolds: int, split_col
) -> None:
    merged_folds_df = pd.merge(folds_df, data_df, on=group_on, how="left")
    analyse_data_splits(merged_folds_df, folds=kfolds, stratify_on=stratify_on, split_col=split_col)
