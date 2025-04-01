import pandas as pd
import os
import logging
import numpy as np
from drop.data_analysis_new.predictions.utils.metrics import *
import drop.data_analysis_new.predictions.utils.survival_analysis as sa
import drop.data_proc.metadata_selection as ms
from drop.data_analysis_new.predictions.utils.analysis_utils import get_threshold_for_max_metric, plot_roc_auc


def calculate_metrics(merged_df, y_true_col, y_pred_col, stage):
    auc, npv, sens, spec, aucpr, f1_reverse = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
    c_index = np.NaN
    auc_CI, npv_CI, sens_CI, spec_CI = (np.NaN, np.NaN), (np.NaN, np.NaN), (np.NaN, np.NaN), (np.NaN, np.NaN)
    tp, tn, fp, fn = np.NaN, np.NaN, np.NaN, np.NaN
    binary_threshold= np.NaN
    try:
        auc = get_auc(merged_df, y_true_col=y_true_col, y_pred_col=y_pred_col)
        auc, auc_CI = bootstrap_auc(merged_df, y_true_col=y_true_col, y_pred_col=y_pred_col)
        binary_threshold = merged_df["binary_threshold"].unique()
        tp, tn, fp, fn = get_tp_tn_fp_fn(merged_df, threshold_col="binary_threshold", y_true_col=y_true_col,
                                         y_pred_col=y_pred_col, verbose=True)
        npv, npv_CI = get_npv(tp, tn, fp, fn, return_CI=True)
        sens, sens_CI = get_sensitivity(tp, fn, return_CI=True)
        spec, spec_CI = get_specificity(tn, fp, return_CI=True)
        f1_reverse = get_f1(tp, tn, fp, fn, pos_label=0, verbose=True)
        c_index = sa.calculate_cindex_risk_score(merged_df, y_pred_col,
                                                 y_true_col=y_true_col)
    except:
        print("Only one true class present")
    finally:
        logged_metrics = {f"{stage}_ci": c_index, f"{stage}_auc": auc, f"{stage}_npv": npv, f"{stage}_spec": spec, f"{stage}_sens": sens,
                          f"{stage}_f1_reverse": f1_reverse, "thrs": binary_threshold,
                          f"{stage}_auc_lower_CI": auc_CI[0], f"{stage}_auc_upper_CI": auc_CI[1],
                          f"{stage}_npv_lower_CI": npv_CI[0], f"{stage}_npv_upper_CI": npv_CI[1],
                          f"{stage}_sens_lower_CI": sens_CI[0], f"{stage}_sens_upper_CI": sens_CI[1],
                          f"{stage}_spec_lower_CI": spec_CI[0], f"{stage}_spec_upper_CI": spec_CI[1], }

        return logged_metrics



def do_survivalpd_analysis(stage, survival_pd, outcome_col, run_name, plot_path, ensemble, train_without_val):
    survivorship_metrics = sa.get_survivorship(survival_pd, outcome_col)
    survivorship_metrics = {
        'test_' + k: v
        for k, v in survivorship_metrics.items()
    }
    logging.info(f"Survivorship: {survivorship_metrics}")
    km_name = f"{run_name}_ensemble{ensemble}_train_without_val{train_without_val}"
    log_rank_pvalue = sa.log_rank_test(survival_pd, outcome_col)
    if stage == "test" and survival_pd['group'].nunique() > 1:
        hazards_ratio, hr_pvalue, hr_lower, hr_upper = sa.calculate_hazards_ratio(survival_pd, y_true_col=outcome_col)
        hr_svalue = sa.calculate_surprise_value(hr_pvalue)

    else:
        hazards_ratio, hr_pvalue, hr_lower, hr_upper = np.NaN, np.NaN, np.NaN, np.NaN
        hr_svalue = np.NaN

    if survival_pd['group'].nunique() > 1:
        high_risk_hr = (hazards_ratio, hr_lower, hr_upper)
        sa.make_kaplan_meier_plot(survival_pd, km_name, plot_path, high_risk_hr=high_risk_hr,
                                  log_rank_pvalue=log_rank_pvalue)
    survivorship_metrics.update({ f"{stage}_log_rank_p": log_rank_pvalue ,
                           f"{stage}_hazards_ratio": hazards_ratio, f"{stage}_hr_pvalue": hr_pvalue,
                           f"{stage}_hr_s_value": hr_svalue,
                           f"{stage}_hr_lower_CI": hr_lower, f"{stage}_hr_upper_CI": hr_upper,
                           })
    return survivorship_metrics

def analyse_slide_level_results(slide_df, merged_meta_df, stage, run_path, input, outer_split=None, fold=None, ensemble=False,
                                train_without_val=False, survival=False):
    merged_df = pd.merge(slide_df, merged_meta_df, on="imageName", how="left")
    outcome_col = "y_true"
    plot_path = os.path.join(run_path, "plots")
    subdirs = merged_meta_df['subdir'].unique().tolist()
    subdir_desc = subdirs[0].split('/')[0]

    if survival:
        cutoff_years = survival["cutoff_years"]
        y_pred_col = "y_pred_mean"
        analyse_cox_model_pred_at_time_point = False
        if analyse_cox_model_pred_at_time_point:
            # this is when you evaluate the coxph predictions on a specific  time period. as it outputs differnet predictions for diffent time periods
            if survival.get("model_type") in ["COXPH", "COXPH_noPR"]:
                y_pred_col = f"{y_pred_col}_limit{cutoff_years}" if type(cutoff_years) == int else y_pred_col

        outcome_col = "outcome"
        merged_df = ms.add_followup_cols(merged_df)
        merged_df = ms.make_outcome_years(merged_df, years=cutoff_years)
        # we are not following the same exact process as in the task_data_selection.py, as there there is column remapping
        merged_df[outcome_col] = merged_df["outcome_in_cutoff"]

    if stage == "val":
        binary_threshold, acc = get_threshold_for_max_metric(merged_df, y_true_col=outcome_col, y_pred_col=y_pred_col)
    else:
        binary_threshold = survival.get("binary_threshold")
        if binary_threshold is None:
            try:
                val_res = pd.read_csv(f"{run_path}/metrics/valfold_metrics.txt", sep='\t', index_col=0)
                binary_threshold = val_res["val/binary_threshold"]["Average"]
            except FileNotFoundError:
                binary_threshold = slide_df["binary_threshold"][0]
        logging.info(f"Using binary threshold {binary_threshold} for {stage} stage")

    run_name = sa.get_run_name("MIL model", input, cutoff_years, subdir_desc, stage, outer_split, fold)
    logged_metrics = calculate_metrics(merged_df, outcome_col, y_pred_col, stage)
    plot_roc_auc(merged_df, path=plot_path, y_true_col=outcome_col, y_pred_col=y_pred_col, name=f"roc_auc_{run_name}", reverse=False)


    if survival:
        keep_cols = [y_pred_col, "vital_status"] if "vital_status" in merged_df.columns else [y_pred_col]
        survival_pd = sa.preprocess_survival_columns(merged_df, y_true_col=outcome_col,  keep_cols=keep_cols)
        survival_pd = sa.get_risk_groups(survival_pd, binary_threshold, risk_col=y_pred_col)
        logged_metrics_survival = do_survivalpd_analysis(stage, survival_pd, outcome_col, run_name, plot_path, ensemble, train_without_val)
        logged_metrics.update(logged_metrics_survival)

        if survival['model_type'] == "COXPH" and type(outer_split) == int: # we aggregate these later in compile_results.py
            test_res = pd.read_csv(f"{run_path}/metrics/NKI05_Block1_Aperio_testfold_metrics.txt", sep='\t', index_col=0)
            phr_cols = [col for col in test_res.columns if col.endswith("_HR")]
            phr_entries = {f"{stage}_{col}": test_res[col][0] for col in phr_cols}
            logged_metrics.update(phr_entries)


    return logged_metrics

