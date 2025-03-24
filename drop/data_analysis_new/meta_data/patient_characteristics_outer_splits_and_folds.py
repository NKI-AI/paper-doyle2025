import pandas as pd
from functools import reduce
from drop.data_analysis_new.meta_data.utils.utils_outer_splits_analysis import *
'''
Relies on analyse_data_splits being called on outer splits to create tables for outer splits.
Relies on analyse_data_splits being called on inner folds to create tables for folds.     
Could also create the data as in patient_characteristics_tables_overall.                     
'''

def get_patient_char_table_fn(base_path, stratify_on, split_col, addition=""):
    df = pd.read_csv(f'{base_path}split_characteristics_{stratify_on}{split_col}{addition}.csv')
    return df

def get_patient_char_table_folds_fn(base_path, stratify_on, split_col, addition=""):
    df = pd.read_csv(f'{base_path}fold_characteristics_{stratify_on}{split_col}{addition}_percentages.csv')
    return df

def average_fold_dfs(base_path, stratify_on, split_col, splits, addition):
    '''
    Relies on analyse_data_splits being called on inner folds to create tables for folds.
    '''
    dfs = []
    for split in range(splits):
        df = get_patient_char_table_folds_fn(base_path, stratify_on, f'{split_col}{split}', addition)
        dfs.append(create_average_df(df))
    # Create a list of DataFrames with suffixed column names
    suffixed_dfs = [
        df.rename(
            columns={col: f"{col}_split{i}" for col in df.columns if col != "Patient Characteristics"}
        )
        for i, df in enumerate(dfs)
    ]
    # Merge all DataFrames on a common column (e.g., "Patient Characteristics")
    merged_df = suffixed_dfs[0]
    for df in suffixed_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="Patient Characteristics", how="outer")  # Adjust `how` if needed

    out_path = f'{base_path}inner_folds_patient_characteristics{addition}_averaged.csv'
    merged_df.to_csv(out_path, index=False)

def combine_patient_char_table(base_path, stratify_on, split_col, splits, addition):
    '''Relies on analyse_data_splits being called on outer splits to create tables for outer splits. '''
    dfs = []
    for outer_split in range(splits):
        df = get_patient_char_table_fn(base_path, stratify_on, f'{split_col}{outer_split}', addition)
        unnamed_columns = [col for col in df.columns if 'Unnamed' in col]
        df.drop(columns=unnamed_columns, inplace=True)
        dfs.append(df)

    # Merge all DataFrames in the list
    merged_df = reduce(merge_dfs, dfs)
    out_path = f'{base_path}outer_splits_patient_characteristics{addition}.csv'
    merged_df.to_csv(out_path, index=False)

    # Average outer_splits
    out_path_averaged = out_path.replace(".csv", "_averaged.csv")
    averaged_df = create_average_df(merged_df)
    averaged_df.to_csv(out_path_averaged, index=False)
    print(out_path_averaged)


def create_all_tables_per_dataset(base_path, stratify_on, split_col, splits, addition):
    ''' Creates the the absolute and percentage patient characteristics tables for a certain dataset
     (specified by addition for the variable selection). It also creates the patient characteristics for the inner folds
        which are based on percentages.
      '''
    combine_patient_char_table(base_path, stratify_on, split_col, splits, addition)
    combine_patient_char_table(base_path, stratify_on, split_col, splits, addition + "_percentages")
    average_fold_dfs(base_path, stratify_on, split_col, splits, addition)

def combine_dutch_datasets_outer_splits(base_path, use_percentages):
    ''' Makes an average across the outer splits for each variable selection in the Dutch dataset.'''
    add = "_percentages" if use_percentages else ""
    all_fn = f'{base_path}outer_splits_patient_characteristics{add}_averaged.csv'
    clinical_basic_fn = f'{base_path}outer_splits_patient_characteristics_clinical_basic{add}_averaged.csv'
    clinical_extended_fn = f'{base_path}outer_splits_patient_characteristics_clinical_extended{add}_averaged.csv'
    df_all = pd.read_csv(all_fn)
    df_cb = pd.read_csv(clinical_basic_fn)
    df_ce = pd.read_csv(clinical_extended_fn)
    df_all = create_combined_mean_95ci_columns_for_metrics(df_all)
    df_cb = create_combined_mean_95ci_columns_for_metrics(df_cb)
    df_ce = create_combined_mean_95ci_columns_for_metrics(df_ce)

    # merge them together
    merged_df = pd.merge(
        df_all, df_cb,
        on="Patient Characteristics",
        suffixes=("_all", "_basic")
    )
    merged_df = pd.merge(
        merged_df, df_ce,
        on="Patient Characteristics",
        suffixes=("", "_extended")
    )
    merged_path = f'{base_path}outer_splits_patient_characteristics{add}_all_datasets_merged.csv'
    merged_df.to_csv(merged_path, index=False)


if __name__ == "__main__":
    base_path =  "/projects/drop/data_splits_drop/"
    stratify_on = 'outcome'
    split_col = 'split_non_rt_only'

    create_all_tables_per_dataset(base_path, stratify_on, split_col, 5, '')
    create_all_tables_per_dataset(base_path, stratify_on, split_col, 5, "_clinical_basic")
    create_all_tables_per_dataset(base_path, stratify_on, split_col, 5, "_clinical_extended")
    combine_dutch_datasets_outer_splits(base_path, use_percentages=True)

