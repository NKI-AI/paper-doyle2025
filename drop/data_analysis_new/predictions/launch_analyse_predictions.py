import os
import pandas as pd
import hydra
from omegaconf import DictConfig
from pathlib import Path
import subprocess

from drop.data_analysis_new.predictions.analyse_predictions import analyse_predictions, prepare_analyse_predictions_test
from drop.data_analysis_new.predictions.utils import post_process_results


@hydra.main(version_base="1.2", config_path="../../../configs/", config_name="config")
def main(cfg: DictConfig ) -> None:
    save_path = f"/home/s.doyle/tmp/DROPPAPER_Results_Nov_Acc/" #was DROPPAPER_Results
    Path(save_path).mkdir(parents=True, exist_ok=True)
    figures_path = f"{save_path}figures/"
    Path(figures_path).mkdir(parents=True, exist_ok=True)
    # don't usually use these
    args_dict= {"do_single_model": False, "do_cv": False,
               "do_test": True, "do_cv_test": False}

    # Chose dataset
    subdirs = ['dataset']
    data_name = 'data'
    years = [20, 5]


    model_type = "dl"
    model = "outer_cross_val_img"
    # model = "outer_cross_val_int"
    # model_type = "COXPH"
    # model = "clinical_basic"
    # model = "clinical_extended"
    # model_type = "ml"
    outer_splits = 5


    for cutoff_years in years:
        results = []
        slide_dfs = []
        for split in range(outer_splits):
            val_metrics, test_metrics, run_path, slide_df = analyse_predictions(model, subdirs, split, cutoff_years, **args_dict)
            test_metrics['split'] = split
            results.append(test_metrics)
            slide_dfs.append(slide_df)
            subprocess.run(f"cp -r {run_path}/plots/*png {figures_path}", shell=True)
        results = pd.concat(results, ignore_index=True, axis=1).T
        df, res_metrics, rec_results, phr_df = post_process_results(results)
        res_metrics_path = f"{save_path}metrics_{model_type}models_{model}_{data_name}_{outer_splits}outersplits_cutoff_years{cutoff_years}.csv"
        rec_results_path = f"{save_path}rec_results_{model_type}models_{model}_{data_name}_{outer_splits}outersplits_cutoff_years{cutoff_years}.csv"
        res_metrics.to_csv(res_metrics_path, index=False)
        rec_results.to_csv(rec_results_path, index=False)
        print(res_metrics.iloc[0])
        print(rec_results)
        print(res_metrics["test_hazards_ratio"])

        if model_type == "COXPH" and data_name == "NKI05_Block1_Aperio":
            phr_df.to_csv(
                f"{save_path}phrs_{model_type}models_{model}_{data_name}_{outer_splits}outersplits_cutoff_yearsFalse.csv",  # we only trained with unlimited FU
                index=False)

        print(res_metrics)

        merged_slide_df = pd.concat(slide_dfs)
        run_path = f"{save_path}{data_name}{model_type}_{model}_years{cutoff_years}_outersplitsmerged{outer_splits}/"
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        merged_slide_df.to_csv(f"{run_path}merged_slide_df.csv", index=False)

        meta_df, experiment, model_name, survival, run_path, ensemble, meta_df, train_without_val = prepare_analyse_predictions_test(
            model, 0, subdirs, cutoff_years)

        run_path = f"{save_path}{data_name}{model_type}_{model}_years{cutoff_years}_outersplitsmerged{outer_splits}/"
        merged_slide_df = pd.read_csv(f"{run_path}merged_slide_df.csv")
        from drop.data_analysis_new.predictions.analyse_predictions import get_test_metrics, analyse_slide_level_results
        logged_metrics = analyse_slide_level_results(merged_slide_df, meta_df, "test", run_path, model, ensemble=ensemble,
                                                     train_without_val=train_without_val, outer_split=range(outer_splits),
                                                     fold=None, survival=survival,
                                                     analyse_her2=False, analyse_grade=False, analyse_age=False)
        test_metrics = get_test_metrics(logged_metrics, model, cutoff_years)

        df, res_metrics, rec_results, phr_df = post_process_results(test_metrics)
        res_metrics_path = f"{run_path}res_metrics.csv"
        rec_results_path = f"{run_path}rec_results.csv"

        res_metrics.to_csv(res_metrics_path, index=False)
        rec_results.to_csv(rec_results_path, index=False)



if __name__ == "__main__":

    """" 
    Run with:
    python launch_analyse_predictions.py task=cls_mil/tile_dataset
    """
    main()