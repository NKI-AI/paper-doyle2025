import pandas as pd
from hydra.utils import instantiate
from drop.utils.env_setup import setup_environment
setup_environment()
from drop.data_analysis_new.predictions.slide_prediction_analysis import analyse_slide_level_results
from drop.data_analysis_new.predictions.utils.model_settings import get_model_details
from drop.data_analysis_new.predictions.utils.loading_utils import *


def setup_experiment(model, outer_split, subdirs, years):
    model_name, survival, multirun = get_model_details(model, outer_split)
    meta_df = get_meta_df(meta_path='/path/to/meta')
    if subdirs == ['test_dataset/']:
        meta_df["split"] = "test"
        meta_df["subdir"] = subdirs[0]
    survival['cutoff_years'] = years
    log_dir = "/path/to/logdir/"
    cfg_run = get_cfg_run(f"{log_dir}{model_name}")
    experiment = instantiate(cfg_run.experiment)
    run_path = experiment.out_dir
    experiment.setup(cfg_run)

    return meta_df, experiment, model_name, survival, run_path

def setup_test(meta_df, survival):
    ensemble = True if not survival.get("model_type") in ["COXPH",
                                                          "ml"] else False  # todo adjust to make ml also ensemble
    train_without_val = False if not survival.get("model_type") in ["COXPH", "ml"] else True

    return ensemble, meta_df, train_without_val

def prepare_analyse_predictions_test(model, outer_split, subdirs, years):
    meta_df, experiment, model_name, survival, run_path = setup_experiment(model, outer_split, subdirs, years)
    ensemble, meta_df, train_without_val = setup_test(meta_df, survival)
    return meta_df, experiment, model_name, survival, run_path, ensemble, meta_df, train_without_val

def get_test_metrics(logged_metrics, model, years):
    test_metrics = pd.Series(logged_metrics)
    test_metrics["years"] = str(years)
    test_metrics["Model"] = model
    return test_metrics

def analyse_predictions(model, subdirs, outer_split, years=None, **kwargs) -> None:
    val_metrics, test_metrics = [], []

    meta_df, experiment, model_name, survival, run_path, ensemble, meta_df, train_without_val = prepare_analyse_predictions_test(model, outer_split, subdirs, years)
    stage = "test"
    fold = None
    slide_df = read_slide_predictions(run_path, epoch=0, fold=fold, stage=stage, ensemble=ensemble,
                                      subdirs=subdirs,
                                      train_without_val=train_without_val)
    logged_metrics = analyse_slide_level_results(slide_df, meta_df, stage, run_path, model, ensemble=ensemble,
                                                 train_without_val=train_without_val, outer_split=outer_split,
                                                 fold=fold, survival=survival,
                                                 )


    test_metrics = get_test_metrics(logged_metrics, model, years)
    return val_metrics, test_metrics, run_path, slide_df





