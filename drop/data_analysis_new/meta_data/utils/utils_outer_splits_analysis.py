import numpy as np
import scipy.stats as stats
import pandas as pd

def create_combined_mean_95ci_columns_for_metrics(df):
    # Combine mean and 95% CI columns

    metrics_cols = [col for col in df.columns if col.endswith('mean')]
    # Loop through each metric column
    for metric_col in metrics_cols:
        # Extract the base name of the metric by removing " mean"
        metric_name = metric_col.replace('_mean', '')
        ci_col = metric_name + '_95%CI'  # The expected corresponding CI column

        # Check if the corresponding CI column exists
        if ci_col in df.columns:
            # Combine mean and CI into one column
            df[metric_name] = df[metric_col].astype(str) + ' ' + df[ci_col]
            # Drop the original mean and CI columns if needed
            df = df.drop(columns=[metric_col, ci_col])
    df.columns = [col.replace('test_', '').replace('_HR', '') for col in df.columns]
    return df

def create_average_df(merged_df):
    # Select the train and val columns
    train_columns = [col for col in merged_df.columns if "train" in col]
    val_columns = [col for col in merged_df.columns if "val" in col]

    # Apply the function to each row for train and val columns
    merged_df[['train_mean', 'train_95%CI']] = merged_df.apply(
        lambda row: pd.Series(calculate_mean_and_ci(row, train_columns)), axis=1
    )

    merged_df[['val_mean', 'val_95%CI']] = merged_df.apply(
        lambda row: pd.Series(calculate_mean_and_ci(row, val_columns)), axis=1
    )

    return merged_df[["Patient Characteristics", 'train_mean', 'train_95%CI', 'val_mean', 'val_95%CI']]

def merge_dfs(left, right):
    # Merge DataFrames
    merged_df = pd.merge(left, right, on='Patient Characteristics', how='outer')
    # Maintain suffixes
    if len(left.columns) > 1:  # Check if not the first merge
        suffixes = [col.split('_')[1] if '_' in col else '' for col in merged_df.columns]
        highest_int = 0
        new_suffixes = ['_' + str(highest_int+i//2) for i in range(len(suffixes)-1)]
        new_suffixes.insert(0, '')
        new_columns = [col.split('_')[0] + suff for col, suff in zip(merged_df.columns, new_suffixes)]
        merged_df.columns = new_columns
        print(new_columns)
    return merged_df

def calculate_mean_and_ci(row, columns):
    values = row[columns].values
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



