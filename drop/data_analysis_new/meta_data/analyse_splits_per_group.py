from typing import TypeVar
DataFrame = TypeVar("pandas.core.frame.DataFrame")
from drop.data_analysis_new.meta_data.utils.utils_characteristics import *
from drop.data_analysis_new.meta_data.utils.metadata_analysis import get_median_observation_time


def get_stats_overall(df, group):
    # Get abs and perc stats
    total = len(df)
    print(total)
    results =[]
    results.append(calculate_chars_for_group(df, group))
    results.append(calculate_chars_for_group(df, group, use_percentages=True))
    merged_df = pd.merge(results[0], results[1], on="var", how="outer")
    merged_df.columns = ['var', 'Abs', 'perc']

    # Format the column to include the second value in brackets
    merged_df[group] = merged_df.apply(lambda row: f"{row['Abs']} ({row['perc']})", axis=1)
    merged_df = merged_df[['var', group]]
    merged_df = merged_df.rename(columns={"var": "Patient Characteristics"})
    merged_df.fillna(0, inplace=True)
    print(merged_df)
    return merged_df


def get_stats_by_risk_group(df):
    # Analyse relevant portion of the data iIBC and controls.
    median_obs, iqr_obs = get_median_observation_time(df)
    print(median_obs, iqr_obs, "Median Observation Time")
    results = []
    for group in ["Low risk", "High risk"]:
        risk_val = 0 if group=="Low risk" else 1
        risk_df = df[df["outcome"] == risk_val]
        print(len(risk_df))
        risk_values = calculate_chars_for_group(risk_df, group)
        results.append(risk_values)


    merged_df = pd.merge(results[0], results[1], on="var", how="outer")
    merged_df = merged_df.rename(columns={"var": "Patient Characteristics"})
    merged_df.fillna(0, inplace=True)
    print(merged_df)
    print(median_obs, iqr_obs, "Median Observation Time")
    breakpoint()
    return merged_df




def test_overlap_groups(df_train, df_val, group_on):
    train_tissue_block_ids = df_train[group_on].tolist()
    overlap_values = df_val[df_val[group_on].isin(train_tissue_block_ids)]
    if not overlap_values.empty:
        print("Some values in df_val are present in df_train.")
        print("Overlapping values:")
        print(overlap_values)
    else:
        print("No overlapping values between df_val and df_train.")



def get_combined_char_table_train_val(df_train, df_val, use_percentages=False):
    # Analyse relevant portion of the data
    results_df = []
    for (group, df) in [('train', df_train), ('val', df_val)]:
        char_values = calculate_chars_for_group(df, group, use_percentages)
        results_df.append(char_values)

    df = pd.merge(results_df[0], results_df[1], on="var", how="outer")
    df = df.rename(columns={"var": "Patient Characteristics"})
    df.fillna(0, inplace=True)

    return df


def _process_subsets_and_save(
        df: pd.DataFrame,
        stratify_on: str,
        group_on: str,
        base_path: str,
        subsets: dict,
        partition_name: str,
        clinical_features_name: str,
        use_percentages: bool,
        split_type: str,
        folds: Optional[int] = None,
):

    subsets = subsets or {"all": None}
    aggregated_results = {}


    for subset_name, condition in subsets.items():
        subset_df = df.loc[condition] if condition is not None else df

        if folds is not None:
            char_dfs = []
            for fold in range(folds):
                df_train = subset_df[subset_df[str(fold)] == "train"]
                df_val = subset_df[subset_df[str(fold)] == "val"]

                #  test_overlap_groups(df_train, df_val, group_on)

                df_char = get_combined_char_table_train_val(
                    df_train, df_val, use_percentages=use_percentages
                )

                fold_label = f'{split_type}_fold{fold}'
                df_char.columns = [
                    f'{col}_{fold_label}' if col != 'Patient Characteristics' else col
                    for col in df_char.columns
                ]
                char_dfs.append(df_char)

            # Merge all dataframes on "Patient Characteristics"
            from functools import reduce
            merged_df = reduce(lambda left, right: pd.merge(left, right, on='Patient Characteristics', how='outer'),
                               char_dfs)
            merged_df = merged_df.fillna("-")
        else:
            df_train = subset_df[subset_df['split'] == "train"]
            df_val = subset_df[subset_df['split'] == "test"]
            merged_df = get_combined_char_table_train_val(
                df_train, df_val, use_percentages=use_percentages
            )
        out_fn = get_fold_char_table_path(base_path, split_type, stratify_on, partition_name, subset_name, clinical_features_name,
                             use_percentages)
        print(out_fn)
        merged_df.to_csv(out_fn, index=False)
        aggregated_results[subset_name] = merged_df

    return aggregated_results


def analyse_single_split(
        df: pd.DataFrame,
        stratify_on: str,
        group_on: str = '',
        base_path: str = "/projects/drop/",
        subsets: dict = None,
        partition_name: str = "split",
        clinical_features_name: str = '',
        use_percentages: bool = False,
):
    return _process_subsets_and_save(
        df=df,
        stratify_on=stratify_on,
        group_on=group_on,
        base_path=base_path,
        subsets=subsets,
        partition_name=partition_name,
        clinical_features_name=clinical_features_name,
        use_percentages=use_percentages,
        split_type="single_split",
        folds=None
    )


def analyse_inner_folds(
        df: pd.DataFrame,
        folds: int,
        stratify_on: str,
        group_on: str = '',
        base_path: str = "/projects/drop/",
        subsets: dict = None,
        partition_name: str = "split",
        clinical_features_name: str = '',
        use_percentages: bool = False,
):
    return _process_subsets_and_save(
        df=df,
        folds=folds,
        stratify_on=stratify_on,
        group_on=group_on,
        base_path=base_path,
        subsets=subsets,
        partition_name=partition_name,
        clinical_features_name=clinical_features_name,
        use_percentages=use_percentages,
        split_type="inner_folds"
    )


def analyse_outer_folds(
        df: pd.DataFrame,
        folds: int,
        stratify_on: str,
        group_on: str = '',
        base_path: str = "/projects/drop/",
        subsets: dict = None,
        partition_name: str = "split",
        clinical_features_name: str = '',
        use_percentages: bool = False,
):
    return _process_subsets_and_save(
        df=df,
        folds=folds,
        stratify_on=stratify_on,
        group_on=group_on,
        base_path=base_path,
        subsets=subsets,
        partition_name=partition_name,
        clinical_features_name=clinical_features_name,
        use_percentages=use_percentages,
        split_type="outer_folds"
    )


def combine_single_split_dfs(base_path, stratify_on, partition_name, subset_name, k_folds, clinical_features_name,
                             use_percentages):
    dfs = []
    for split in range(k_folds):
        out_fn = get_fold_char_table_path(base_path, 'single_split', stratify_on,
                                          partition_name=f'{partition_name}{split}', subset_name=subset_name,
                                          clinical_features_name=clinical_features_name,
                                          use_percentages=use_percentages)
        df = pd.read_csv(out_fn)
        dfs.append(df)

    # Create a list of DataFrames with suffixed column names
    suffixed_dfs = [
        df.rename(
            columns={col: f"outer_fold{i}_{col}" for col in df.columns if col != "Patient Characteristics"}
        )
        for i, df in enumerate(dfs)
    ]
    # Merge all DataFrames on a common column (e.g., "Patient Characteristics")
    merged_df = suffixed_dfs[0]
    for df in suffixed_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="Patient Characteristics", how="outer")  # Adjust `how` if needed
    return merged_df


def get_averaged_outer_splits_characteristics(base_path, stratify_on, partition_name, subset_name, k_folds, clinical_features_name,
                     use_percentages):
    merged_df = combine_single_split_dfs(base_path, stratify_on, partition_name, subset_name, k_folds,
                                         clinical_features_name, use_percentages)
    averaged_df = create_average_df(merged_df)
    out_fn = get_averaged_fold_char_table(base_path, 'outer_folds', stratify_on, partition_name, subset_name,
                                          clinical_features_name,
                                          use_percentages)
    averaged_df.to_csv(out_fn, index=False)



