import pandas as pd
import logging
from drop.data_analysis_new.predictions.utils.post_process_results import *
from drop.data_analysis_new.predictions.utils.formatting_utils import *
save_path = f"/path/to/save/"

def get_results_path_all_splits(res_path, model_type, model_name, data_name, outer_splits, years):
    return f"{res_path}{model_type}models_{model_name}_{data_name}_{outer_splits}outersplits_cutoff_years{years}.csv"

def get_models(data_name):
    models = [
        {"type": "COXPH" if not no_PR else "COXPH_no_PR", "name": "clinical_basic"},
        {"type": "dl", "name": "outer_cross_val_img"},
        {"type": "dl", "name": "outer_cross_val_int"},
    ]
    if data_name != "Sloane":
        models.append({"type": "COXPH" if not no_PR else "COXPH_no_PR", "name": "clinical_extended"}
                      )
    if data_name == "NKI_ER":
        models = [
            {"type": "COXPH", "name": "clinical_basic_er0"},
            {"type": "COXPH", "name": "clinical_basic_er1"},
        ]
        data_name = "NKI05_Block1_Aperio"  # little bit sneaky trick
    return models, data_name


def read_acc_results(save_path, csv_file, data_name, years):
    models, data_name = get_models(data_name)
    # Read and concatenate results for each model
    results_combined = pd.DataFrame()
    outer_splits= 5
    for model in models:
        run_path = f'{save_path}{data_name}{model["type"]}_{model["name"]}_years{years}_outersplitsmerged{outer_splits}/'
        results_path = f"{run_path}{csv_file}"
        model_results = pd.read_csv(results_path)
        if ("clinical_basic_er" in model['name'] or csv_file == "merged_slide_df.csv"):
            model_results['Model'] = model['name']
        results_combined = pd.concat([results_combined, model_results], ignore_index=True)
    return results_combined

def read_results(res_path, data_name, years):
    """
    Reads and combines results from multiple models based on the specified parameters.

    Parameters:
    - res_path (str): The path to the results directory.
    - data_name (str): The name of the dataset.
    - years (int): The number of years for the results.

    Returns:
    - pd.DataFrame: A combined DataFrame containing results from all specified models.
    """

    models, data_name = get_models(data_name)

    # Read and concatenate results for each model
    results_combined = pd.DataFrame()
    outer_splits = 5
    for model in models:
        results_path = get_results_path_all_splits(res_path, model["type"], model["name"], data_name, outer_splits, years)
        model_results = pd.read_csv(results_path)
        if "clinical_basic_er" in model['name']:
            model_results['Model'] = model['name']
        results_combined = pd.concat([results_combined, model_results], ignore_index=True)
    return results_combined

def generate_result_paths(save_path):
    """
    Generates a list of result paths and their corresponding metric types.
    Returns a list of tuples containing the result path and the metric types.
    """
    paths_and_metrics = []

    # List of base result paths
    result_paths = [f"{save_path}metrics_", f"{save_path}rec_results_"]

    # Assign metric types based on the result path
    for res_path in result_paths:
        if "metrics_" in res_path:
            metric_types = ["overall", "hazard"]
        else:
            metric_types = ["rec"]

        paths_and_metrics.append((res_path, metric_types))

    return paths_and_metrics

def generate_result_paths_acc(save_path):
    """
    Generates a list of result paths and their corresponding metric types.
    Returns a list of tuples containing the result path and the metric types.
    """
    paths_and_metrics = []

    # List of base result paths
    result_paths = [f"res_metrics.csv", f"rec_results.csv"]
    # Assign metric types based on the result path
    for res_path in result_paths:
        if "metrics" in res_path:
            metric_types = ["overall", "hazard"]
        else:
            metric_types = ["rec"]

        paths_and_metrics.append((res_path, metric_types))

    return paths_and_metrics


def iterate_paths_and_metrics(save_path, accumulated=False):
    """
    Iterator function to yield result paths and their corresponding metric types.

    Parameters:
    - save_path (str): Path to the directory where results are saved.

    Yields:
    - Tuple of (res_path, metric_type): Each result path and its corresponding metric type.
    """
    # Generate the list of result paths and metric types
    if accumulated:
        paths_and_metrics = generate_result_paths_acc(save_path)
    else:
        paths_and_metrics = generate_result_paths(save_path)
    # Iterate through each result path and its corresponding metric types
    for res_path, metric_types in paths_and_metrics:
        # Yield each metric type for the given result path
        for metric_type in metric_types:
            yield res_path, metric_type


def process_single_run_results(df, metric_type):
    if metric_type == "overall":
        df = make_metrics_table(df)
    elif metric_type == "hazard":
        df = make_hazard_table(df)
    elif metric_type == "rec":
        df = make_rec_table(df)
    df = update_table_legend(df)
    return  df


def process_kfold_results(df, metrics: bool):
    df_processed = calculate_mean_and_ci_expanded(df)
    if metrics:
        hr_df = process_split_wise_hrs(df)
        df_processed = pd.merge(df_processed, hr_df, on=["Model", "Follow-up time"])
    else:
        df.loc[:, "% Recurrence p-value"] = df["p-value raw"]
        p_value_df = process_split_wise_pvalues(df)
        df_processed = pd.merge(df_processed, p_value_df, on=["Model", "Follow-up time"])
    df_processed = format_outer_cross_results(df_processed, metrics)
    df_processed = merge_mean_95ci_columns_for_metrics(df_processed)
    return df_processed

def process_kfold_results_external(df, metrics: bool):
    df_processed = calculate_mean_and_ci_expanded(df)
    if metrics:
        hr_df = process_split_wise_hrs_external(df)
        df_processed = pd.merge(df_processed, hr_df, on=["Model", "Follow-up time"])
    else:
        df.loc[:, "% Recurrence p-value"] = df["p-value raw"]
        p_value_df = process_split_wise_pvalues(df)
        df_processed = pd.merge(df_processed, p_value_df, on=["Model", "Follow-up time"])

    df_processed = format_outer_cross_results(df_processed, metrics)
    df_processed = merge_mean_95ci_columns_for_metrics(df_processed)
    return df_processed


# Example usage
def process_results(save_path, data_name, years):
    """
    Processes results based on the generated result paths and conditions.
    """
    result_dfs = {}
    for res_path, metric_type in iterate_paths_and_metrics(save_path):
        logging.info(f"Processing metric: {metric_type}")
        df = read_results(res_path, data_name, years)
        # Save the results as they are - for the k-fold outer cross validation
        df_single_runs = process_single_run_results(df, metric_type)
        df_single_runs_path = f"{res_path}{metric_type}{data_name}_cutoff_years{years}.csv"
        df_single_runs.astype(str).to_csv(df_single_runs_path, index=False)
        result_dfs[metric_type] = df_single_runs

    # make the final table
    metrics_table = result_dfs["overall"]
    hr_table = result_dfs["hazard"]
    common_columns = metrics_table.columns.intersection(hr_table.columns)
    final_k_fold_results = pd.merge(metrics_table, hr_table, on=list(common_columns))
    final_k_folds_path = f"{save_path}{data_name}_final_kfolds_cutoff_years{years}.csv"
    final_k_fold_results.astype(str).to_csv(final_k_folds_path, index=False)

    # Process K-fold results to get an overall performance estimate
    final_k_fold_results = pd.read_csv(final_k_folds_path)
    if data_name == "NKI05_Block1_Aperio":
     metrics_df_kfold_averaged = process_kfold_results(final_k_fold_results, metrics=True)
    else:
        metrics_df_kfold_averaged = process_kfold_results_external(final_k_fold_results, metrics=True)

    metrics_df_kfold_averaged_path = f"{save_path}{data_name}_final_kfolds_averaged_cutoff_years{years}.csv"
    metrics_df_kfold_averaged.astype(str).to_csv(metrics_df_kfold_averaged_path, index=False)

    rec_df_single_runs_path = f"{res_path}rec{data_name}_cutoff_years{years}.csv"
    rec_df_single_runs = pd.read_csv(rec_df_single_runs_path)
    if data_name == "internal_data":
        rec_df_kfold_averaged = process_kfold_results(rec_df_single_runs, metrics=False)
    else:
        rec_df_kfold_averaged = process_kfold_results_external(rec_df_single_runs, metrics=False)

    rec_df_kfold_averaged_path = f"{save_path}{data_name}_final_kfolds_averaged_cutoff_years{years}_rec.csv"
    rec_df_kfold_averaged.astype(str).to_csv(rec_df_kfold_averaged_path, index=False)
    print(rec_df_kfold_averaged)
    breakpoint()


def process_acc_results(save_path, data_name, years ):

    # combined_slide_df = read_acc_results(save_path, "merged_slide_df.csv", data_name, years)
    result_dfs = {}
    for csv_file, metric_type in iterate_paths_and_metrics(save_path, accumulated=True):
        # Your processing logic here
        print(f"Processing metric: {metric_type}")
        df = read_acc_results(save_path, csv_file, data_name, years)
        # Save the results as they are - for the k-fold outer cross validation
        df_single_runs = process_single_run_results(df, metric_type)
        df_single_runs_path = f"{save_path}{metric_type}{data_name}_cutoff_years{years}.csv"
        print(df_single_runs_path)
        df_single_runs.astype(str).to_csv(df_single_runs_path, index=False)
        result_dfs[metric_type] = df_single_runs


    metrics_table = result_dfs["overall"]
    hr_table = result_dfs["hazard"]
    common_columns = metrics_table.columns.intersection(hr_table.columns)
    df = pd.merge(metrics_table, hr_table, on=list(common_columns))
    df = add_multiple_testing_results(df, metric="Hazard Ratio", p_value_col="p-value raw")
    df_processed_metrics = merge_mean_95ci_columns_for_metrics(df)
    proc_metrics_path = f"{save_path}{data_name}pooled_model_preds_cutoff_years{years}.csv"
    df_processed_metrics.astype(str).to_csv(proc_metrics_path, index=False)

    rec_df = result_dfs["rec"]
    processed_rec_df = add_multiple_testing_results(rec_df, metric="", p_value_col="p-value raw")
    processed_rec_df = processed_rec_df.round(2)
    proc_rec_metrics_path = f"{save_path}{data_name}pooled_model_preds_cutoff_years{years}_rec.csv"
    processed_rec_df.astype(str).to_csv(proc_rec_metrics_path, index=False)


def get_phrs(save_path, data_name, model_name):
    outer_splits= 5
    results_path = get_results_path_all_splits(save_path, "phrs_COXPH", model_name, data_name, outer_splits, False)
    df = pd.read_csv(results_path)
    if "Follow-up time" not in df.columns:
        df["Follow-up time"] = years
    df = calculate_mean_and_ci_expanded(df)
    df = merge_mean_95ci_columns_for_metrics(df)
    df.astype(str).to_csv(f"{save_path}phrs_{model_name}_{data_name}_final_kfolds_averaged_cutoff_yearsFalse.csv", index=False)

if __name__ == "__main__":
    data_name = "internal_data"
    years = 5
    process_acc_results(save_path, data_name, years)
    model_name = "clinical_basic"
    get_phrs(save_path, data_name, model_name)
