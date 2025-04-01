import numpy as np
import pandas as pd

from drop.data_analysis_new.predictions.utils import format_pvalue, get_out_cols, get_keep_cols
from drop.data_analysis_new.predictions.utils import calculate_surprise_value
from drop.data_analysis_new.predictions.utils.pool_outer_splits import *

# used by analyse_predictions
def post_process_results(df):
    """Compile results from multiple experiments into one dataframe. Expects indivudal results to be a series."""

    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df.set_index("years", inplace=True)
    keep_cols = ["Model"]
    if "split" in df.columns:
        keep_cols.append("split")
    metrics_names = ["npv", "spec", "f1_reverse", "aucpr", "auc", "hazards_ratio", "sens"]
    ci_names = ["auc_lower_CI", "auc_upper_CI","npv_lower_CI", "npv_upper_CI", "sens_lower_CI", "sens_upper_CI", "spec_lower_CI", "spec_upper_CI",
                "hr_lower_CI", "hr_upper_CI"]
    metrics_names.extend(ci_names)
    metrics_cols = keep_cols + [f"test_{metric}" for metric in metrics_names]
    metrics_cols.extend(["test_hr_pvalue", "test_log_rank_p", "test_hr_s_value"])
    res_metrics = df[metrics_cols]
    res_metrics.reset_index(inplace=True) # make years (index) a column
    rec_df = df[keep_cols + ["test_predicted_low", "test_predicted_high","test_misclassified_low", "test_misclassified_high",
                             'test_rec_rate_low', 'test_rec_rate_high', 'test_rec_rate_p_value',
                             "test_censored_low", "test_censored_high"]]
    rec_df.reset_index(inplace=True)

    phr_cols = [col for col in df.columns if col.endswith("_HR")]
    phr_df = df[keep_cols + phr_cols]

    return df, res_metrics, rec_df, phr_df


# used by compile results
def make_metrics_table(df):
    df = df.round(2)
    df.loc[:, "test_auc_ci"] = df.apply(lambda x: (x['test_auc_lower_CI'], x['test_auc_upper_CI']), axis=1)
    df.loc[:, "test_npv_ci"] = df.apply(lambda x: (x['test_npv_lower_CI'], x['test_npv_upper_CI']), axis=1)
    df.loc[:, "test_sens_ci"] = df.apply(lambda x: (x['test_sens_lower_CI'], x['test_sens_upper_CI']), axis=1)
    df.loc[:, "test_spec_ci"] = df.apply(lambda x: (x['test_spec_lower_CI'], x['test_spec_upper_CI']), axis=1)

    keep_cols = ["Model", "years"]
    out_cols = ['Model', 'Follow-up time']
    if "split" in df.columns:
        keep_cols.append("split")
        out_cols.append("Split")
    metrics_names = ["npv", "spec", "f1_reverse", "aucpr", "auc", "sens"]
    ci_names = ["auc_ci", "npv_ci", "sens_ci", "spec_ci"]
    metrics_names.extend(ci_names)
    keep_cols.extend([f"test_{metric}" for metric in metrics_names])
    df = df[keep_cols]
    out_cols += [ 'NPV', 'Specificity', 'F1', 'PR-AUC', 'ROC-AUC', "Sensitivity",
                  "ROC-AUC (95% CI)", "NPV (95% CI)",   "Sensitivity (95% CI)",  "Specificity (95% CI)"]
    df.columns = out_cols
    return df


def make_hazard_table(df):
    df["test_hazards_ratio"] = df["test_hazards_ratio"].round(2)
    df["test_hr_lower_CI"] = df["test_hr_lower_CI"].round(2)
    df["test_hr_upper_CI"] = df["test_hr_upper_CI"].round(2)

    df.loc[:, "test_hr_ci"] = df.apply(lambda x: (x['test_hr_lower_CI'], x['test_hr_upper_CI']), axis=1)
    df["HR p-value"] = df["test_hr_pvalue"].apply(lambda x: format_pvalue(x))
    df["KM p-value"] = df["test_log_rank_p"].apply(lambda x: format_pvalue(x))
    keep_cols = get_keep_cols(df, ["test_hazards_ratio", "test_hr_ci", "HR p-value", "test_hr_s_value", "KM p-value", "test_hr_pvalue", "test_log_rank_p"])
    out_cols = get_out_cols(df, [ 'Hazard Ratio', "Hazard Ratio (95% CI)", 'Hazard Ratio p-value', 'Hazard Ratio s-value', 'KM p-value', 'Hazard Ratio p-value raw', "KM p-value raw"])
    df = df[keep_cols]
    df.columns = out_cols
    return df

def make_rec_table(df):
    df["p-value"] = df["test_rec_rate_p_value"].apply(lambda x: format_pvalue(x))
    keep_cols = get_keep_cols(df, ["test_predicted_low", "test_predicted_high", "test_misclassified_low", "test_misclassified_high",
                     'test_rec_rate_low', 'test_rec_rate_high', "p-value", "test_rec_rate_p_value",
                     "test_censored_low", "test_censored_high"
                     ])
    out_cols = get_out_cols(df,
                            ['# Predicted LR', "# Predicted HR", "# Misclassified LR", "# Misclassified HR", '% LR',
                             '% HR', 'p-value', 'p-value raw', "% Censored LR", "% Censored HR"])
    df = df[keep_cols]
    df.columns = out_cols
    return df

def update_table_legend(df):
    # rename False in years to no limit
    df["Follow-up time"] = df["Follow-up time"].replace("False", "No limit")
    # replace basic with Clinical-basic and extended with Clinical-extended in Model column
    df["Model"] = df["Model"].replace("basic", "Clinical-basic")
    df["Model"] = df["Model"].replace("extended", "Clinical-extended")
    df["Model"] = df["Model"].replace("image-only", "Image-only")
    return df


def add_multiple_testing_results(df , metric, p_value_col="p-value corrected"):
    if metric != "":
        p_value_col = f"{metric} {p_value_col}"
    p_values_multipletests = apply_multipletests_correction(df[p_value_col].to_list(),
                                                            "hommel")
    s_values_multipletests = [calculate_surprise_value(pvalue) for pvalue in p_values_multipletests]
    df[f"{metric} p-value multiple-testing corrected"] = [format_pvalue(pvalue) for pvalue in
                                                                  p_values_multipletests]
    df[f"{metric} s-value multiple-testing corrected"] = [round(svalue, 2) for svalue in s_values_multipletests]
    return  df


def process_split_wise_pvalues(df, metric="% Recurrence"):
    grouped = df.groupby(['Model', 'Follow-up time'])
    results = []
    # Calculate mean and 95% CI for each group
    for (model, follow_up_time), group in grouped:
        result = {'Model': model, 'Follow-up time': follow_up_time}
        p_values  = group[f"{metric} p-value"].values
        p_value_adjusted = fishers_combined_test(p_values)
        result[f"{metric} p-value corrected"] = p_value_adjusted
        result[f"{metric} s-value corrected"] = calculate_surprise_value(p_value_adjusted)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = add_multiple_testing_results(results_df, metric)

    return results_df

def process_split_wise_hrs_external(df, metric="Hazard Ratio"):
    grouped = df.groupby(['Model', 'Follow-up time'])
    results = []
    # Calculate mean and 95% CI for each group
    for (model, follow_up_time), group in grouped:
        result = {'Model': model, 'Follow-up time': follow_up_time}
        values = group[metric].values
        ci_strings = group[f"{metric} (95% CI)"].values
        lower_cis = []
        upper_cis = []
        for ci in ci_strings:
            lower, upper = ci.strip('()').split(', ')
        lower_cis.append(float(lower))
        upper_cis.append(float(upper))

        mean_val, (ci_lower, ci_upper), pvalue = combine_hazard_ratios_asymmetric_invw(values, lower_cis, upper_cis)
        result[f"{metric} p-value corrected"] = pvalue
        result[f"{metric} s-value corrected"] = calculate_surprise_value(pvalue)
        result[f"{metric} pooled"] = round(mean_val, 2)
        result[f"{metric} pooled (95% CI)"] = f"({round(ci_lower, 2)}, {round(ci_upper, 2)})"
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = add_multiple_testing_results(results_df, metric)
    return results_df


def process_split_wise_hrs(df, metric="Hazard Ratio"):
    grouped = df.groupby(['Model', 'Follow-up time'])
    results =[]
    # Calculate mean and 95% CI for each group
    for (model, follow_up_time), group in grouped:
        result = {'Model': model, 'Follow-up time': follow_up_time}
        values = group[metric].values
        ci_strings = group[f"{metric} (95% CI)"].values
        lower_cis = []
        upper_cis = []
        variances = []
        ses = []
        for ci in ci_strings:
            lower, upper = ci.strip('()').split(', ')
            lower_cis.append(float(lower))
            upper_cis.append(float(upper))
            variances.append(get_variance(float(lower), float(upper)))
            ses.append(calculate_se_log_hr(float(lower), float(upper)))
        mean_val, (ci_lower, ci_upper), pvalue = random_effects_meta_analysis(values, ses)
        result[f"{metric} p-value corrected"] = pvalue
        result[f"{metric} s-value corrected"] =  calculate_surprise_value(pvalue)

        result[f"{metric} pooled"] = round(mean_val,2)
        result[f"{metric} pooled (95% CI)"] = f"({round(ci_lower, 2)}, {round(ci_upper, 2)})"
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = add_multiple_testing_results(results_df, metric)
    return results_df


def calculate_mean_and_ci_expanded(df):
    """
    Calculate the mean and 95% CI for all metrics that are not a confidence interval column,
    while keeping 'Model' and 'Follow-up time', and separating results into mean and 95% CI columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing the results across multiple splits.

    Returns:
    pd.DataFrame: DataFrame containing the mean and 95% CI for each metric, grouped by Model and Follow-up time.
    """
    # Identify columns that are not 95% CI columns
    metric_columns = [col for col in df.columns if "(95% CI)" not in col and col not in ['Model', 'Follow-up time', 'Split', "split"]]  # add "HR"
    # Initialize a list to store results
    results = []

    # Group by 'Model' and 'Follow-up time'
    grouped = df.groupby(['Model', 'Follow-up time'])

    # Calculate mean and 95% CI for each group
    for (model, follow_up_time), group in grouped:
        result = {'Model': model, 'Follow-up time': follow_up_time}

        for metric in metric_columns:
            if group[metric].dtype in [np.float64, np.float32, np.int64, np.int32] and "Hazard Ratio" not in metric:  # Ensure column is numeric
                values = group[metric].values
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                ci_lower = mean_val - 1.96 * (std_val / np.sqrt(len(values)))
                ci_upper = mean_val + 1.96 * (std_val / np.sqrt(len(values)))

                # Store the mean and CI in the result dictionary
                if "p-value" in metric:
                    if "raw" in metric:
                        metric = metric.replace(" raw", "")
                    result[f"{metric} mean"] = format_pvalue(mean_val)
                    result[f"{metric} (95% CI)"] = f"({format_pvalue(ci_lower)}, {format_pvalue(ci_upper)})"
                else:
                    result[f"{metric} mean"] = round(mean_val, 2)
                    result[f"{metric} (95% CI)"] = f"({round(ci_lower, 2)}, {round(ci_upper, 2)})"

        results.append(result)
    results_df = pd.DataFrame(results)
    return results_df


def set_recurrence_columns(df):
    columns_list = [
        'Model',
        'Follow-up time',
        '# Predicted LR mean', '# Predicted LR (95% CI)',
        '# Predicted HR mean', '# Predicted HR (95% CI)',
        '# Misclassified LR mean', '# Misclassified LR (95% CI)',
        '# Misclassified HR mean', '# Misclassified HR (95% CI)',
        '% LR mean', '% LR (95% CI)',
        '% HR mean', '% HR (95% CI)',
        '% Recurrence p-value',
        '% Recurrence s-value',
        '% Censored LR mean', '% Censored LR (95% CI)',
       '% Censored HR mean', '% Censored HR (95% CI)',
    # could also calculate a p-value for censoring
    ]
    return df[columns_list]

def set_metrics_columns(df):
    columns_list = [
        "Model",
        "Follow-up time",
        "ROC-AUC mean",
        "ROC-AUC (95% CI)",
        "NPV mean",
        "NPV (95% CI)",
        "Specificity mean",
        "Specificity (95% CI)",
        "Sensitivity mean",
        "Sensitivity (95% CI)",
        "Hazard Ratio pooled",
        "Hazard Ratio pooled (95% CI)",
        "Hazard Ratio p-value",
        "Hazard Ratio s-value",
        "Hazard Ratio p-value multiple-testing",
        "Hazard Ratio s-value multiple-testing",
    ]

    return df[columns_list]

def rename_model_names(df):
    # Mapping for renaming models
    rename_map = {
        'clinical_basic': 'Cox-Clinical',
        'clinical_extended': 'Cox-Clinical-Extended',
        'outer_cross_val_img': 'DL-Image-only',
        'outer_cross_val_int': 'DL-Integrative',
        'integrative_catboost': 'CatBoost-Integrative',
        'integrative_cox': 'Cox-Integrative'
    }
    df['Model'] = df['Model'].replace(rename_map)
    # Define the new order for the models
    new_order = [
        'Cox-Clinical',
        'Cox-Clinical-Extended',
        'DL-Image-only',
        'DL-Integrative',
        'CatBoost-Integrative',
        'Cox-Integrative'
    ]

    # Reorder the DataFrame
    df['Model'] = pd.Categorical(df['Model'], categories=new_order, ordered=True)
    df = df.sort_values('Model').reset_index(drop=True)  # necessary
    return df


def format_outer_cross_results(df, metrics:bool):
    if 'Hazard Ratio p-value multiple-testing corrected' in df.columns:
        df.rename(columns={'Hazard Ratio p-value corrected': 'Hazard Ratio p-value'}, inplace=True)
        df.rename(columns={'Hazard Ratio s-value corrected': 'Hazard Ratio s-value'}, inplace=True)
        df.rename(columns={'Hazard Ratio p-value multiple-testing corrected': 'Hazard Ratio p-value multiple-testing'}, inplace=True)
        df.rename(columns={'Hazard Ratio s-value multiple-testing corrected': 'Hazard Ratio s-value multiple-testing'}, inplace=True)
    if "% Recurrence p-value multiple-testing corrected" in df.columns:
        df.rename(columns={'% Recurrence p-value multiple-testing corrected': '% Recurrence p-value'}, inplace=True)
        df.rename(columns={'% Recurrence s-value multiple-testing corrected': '% Recurrence s-value'}, inplace=True)
    if metrics:
        df = set_metrics_columns(df)
    else:
        df = set_recurrence_columns(df)
    df = rename_model_names(df)
    return df


def create_combined_mean_95ci_columns_for_metrics(df):
    # Combine mean and 95% CI columns
    metrics_cols = [(col.replace(" (95% CI)", ""), col) for col in df.columns if col.endswith('(95% CI)')]
    # Loop through each metric column
    for metric_col, ci_col in metrics_cols:
        # Extract the base name of the metric by removing " mean"
        if f"{metric_col} mean" in df.columns:
            metric_name = metric_col
            metric_col = f"{metric_col} mean"
        else:
            metric_name = metric_col.split(" ")[0]
        # Check if the corresponding CI column exists
        if ci_col in df.columns:
            # Combine mean and CI into one column
            df[metric_name] = df[metric_col].astype(str) + ' ' + df[ci_col].astype(str)
            # Drop the original mean and CI columns if needed
            df = df.drop(columns=[ci_col])
            if metric_col != metric_name:
                df = df.drop(columns=[metric_col])

    df.columns = [col.replace('test_', '').replace('_HR', '') for col in df.columns]
    return df


