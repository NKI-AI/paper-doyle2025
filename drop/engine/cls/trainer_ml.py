from trainer import *
from omegaconf import DictConfig, OmegaConf
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from drop.utils.logging import add_to_hparams
from drop.utils.checkpoint_paths import get_lightning_ckpt_path, get_metrics_path
import drop.data_analysis_new.predictions.utils.survival_analysis as sa
from drop.tools.json_saver import JsonSaver
from drop.data_analysis_new.predictions.slide_prediction_analysis import do_survivalpd_analysis, calculate_metrics
from drop.ml_utils.utils import *


def get_test_df(sel_data_df):
    test_df =  sel_data_df.loc[sel_data_df["split"] == "test"]
    return test_df

def get_train_val_df(sel_data_df, folds_df, fold, sample_col):
    fold = str(fold)
    names_train_fold = (
                        folds_df.loc[folds_df[fold] == "train"][sample_col].unique().tolist()
                    )
    names_val_fold = (
        folds_df.loc[folds_df[fold] == "val"][sample_col].unique().tolist()
    )
    train_df = sel_data_df.loc[sel_data_df[sample_col].isin(names_train_fold)]
    val_df = sel_data_df.loc[sel_data_df[sample_col].isin(names_val_fold)]
    return train_df, val_df


def preprocess_clinical_columns(df, clinical_vars_type):
    """ We modify the encoding of clinical variables for easier interpretation and better performance.
    """
    df["her2"] = df["her2"].astype(float)
    # more interpretable with grade low and high
    df.loc[df["cox2"].isin([1, 2]), "grade"] = 0.0
    df.loc[df["cox2"].isin([3]), "grade"] = 1.0
    df["age_diagnose"] = df['age_diagnose']

    if clinical_vars_type == "extended":
        df.loc[df["p53"] == 0, "p53"] = 0.0
        df["p53"] = np.where(df["p53"] > 70, 1, 0).astype(float)
        df.loc[df["cox2"].isin([1]), "cox2"] = 0.0
        df.loc[df["cox2"].isin([2, 3]), "cox2"] = 1.0
        df["p16"] = df["p16"].astype(float)

    return df


def fit_sklearn_model(model, train_df, keep_cols):
    y_train = train_df["outcome"]
    X_train = preprocess(train_df, keep_cols)
    model.fit(X_train, y_train)

    return model, X_train

def evaluate_sklearn_model(model, df, keep_cols, stage):
    y_val = df["outcome"]
    X_val = preprocess(df, keep_cols)
    y_pred_val = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred_val)
    survival_pd = df[["time_to_event", "outcome"]]
    survival_pd = survival_pd.assign(pred_score=y_pred_val)
    c_index = sa.calculate_cindex_risk_score(survival_pd, "pred_score") # todo potentially remove?
    logged_metrics = {f"{stage}_auc": auc, f"{stage}_ci": c_index}
    return logged_metrics, survival_pd

def fit_coxph_model(cph, survival_pd_train, keep_cols):
    cph.fit(survival_pd_train, 'time_to_event', 'outcome')
    test_cox_assumptions = False
    if test_cox_assumptions:
        test_proportional_hazards(cph, survival_pd_train)
        for covariate in keep_cols:
            plot_martingale_residuals(cph, survival_pd_train, covariate)

    return cph

def predict_survival(cph, survival_pd):
    cox_scores = get_coxscores(cph, survival_pd)
    c_index_coxph = cindex_from_coxscores(cph, survival_pd)
    survival_pd_with_cox_scores = pd.merge(survival_pd, cox_scores, left_index=True, right_index=True)

    return survival_pd_with_cox_scores, c_index_coxph

def evaluate_coxph_model(model, survival_pd_val, stage):
    survival_pd_val_with_cox_scores, c_index = predict_survival(model, survival_pd_val)
    hrs_metrics = {f"{k}_HR": np.exp(v) for k, v in model.params_.items()}
    cis_lower = {f"{k}_ci_lower_HR": np.exp(v) for k, v in model.confidence_intervals_["95% lower-bound"].items()}
    cis_upper = {f"{k}_ci_upper_HR": np.exp(v) for k, v in model.confidence_intervals_["95% upper-bound"].items()}
    p_value = {f"{k}_pvalue_HR": v for k, v in model.summary['p'].items()}
    auc_coxph = calculate_auc_coxph(survival_pd_val_with_cox_scores)
    coxph_metrics = {f"{stage}_auc": auc_coxph, f"{stage}_ci": c_index}
    coxph_metrics.update(hrs_metrics)
    coxph_metrics.update(cis_lower)
    coxph_metrics.update(cis_upper)
    coxph_metrics.update(p_value)
    return coxph_metrics, survival_pd_val_with_cox_scores


@hydra.main(version_base="1.2", config_path="../../../configs/", config_name="config_clinical")
def main(cfg: DictConfig):
    set_random_seed(cfg.random_seed)
    if cfg.enforce_deterministic:
        torch.use_deterministic_algorithms(mode=True)
    if cfg.experiment.train:
        logging.info(OmegaConf.to_yaml(cfg))
        if cfg.data.dataset_name not in ("Precision_Maartje", "Precision_NKI_89_05"):
            raise ValueError("Dataset must be either Precision_Maartje or Precision_NKI_89_05 for training!")


    data_prep = instantiate(cfg.data_prep)
    sel_data_df, folds_df = data_prep.prepare_data()
    sel_data_df['time_to_event'].value_counts()
    cutoff_years = cfg.data.data_sel_params.cutoff_years
    meta_cols = cfg.data.data_cols.meta
    model_name = cfg.model._target_.split(".")[-1]
    experiment = instantiate(cfg.experiment)
    experiment.fold_metrics_paths = experiment.get_fold_wise_list(
        [experiment.out_dir, cfg.paths.metrics_dir], experiment.fold_dirs, get_metrics_path
        )
    experiment.eval_ckpt_fn = f"{model_name}.pkl"
    experiment.fold_ckpt_paths = experiment.get_fold_wise_list([experiment.out_dir, cfg.paths.checkpoints_dir,
                                                                experiment.eval_ckpt_fn], experiment.fold_dirs,
                                                               get_lightning_ckpt_path)
    keep_cols = cfg.data.data_sel_params.drop_nas_in_cols
    sel_data_df = preprocess_clinical_columns(sel_data_df, cfg.clinical_vars_type)
    outcome_col = "outcome"
    pred_col = "pred_score"
    threshold_method = cfg.threshold_method
    subdirs = sel_data_df['subdir'].unique().tolist()
    subdir_desc = subdirs[0].split('/')[-2]
    val_kfold_metrics_collector = instantiate(
        cfg.kfold_metrics_collector.val,
        used_folds=experiment.use_folds,
        metrics_paths=experiment.fold_metrics_paths,
    )

    if not cfg.data.cv_splitter:
        threshold = None
        model_path = experiment.out_dir + experiment.eval_ckpt_fn
        pl_loggers: List = instantiate_loggers(cfg.get("logger"), experiment.out_dir)
        for logger in pl_loggers:
            logger.log_hyperparams({"vars": keep_cols})
        json_saver = JsonSaver("predictions", json_path=f"{experiment.out_dir}metrics/predictions.json")
        if experiment.train:
            train_df = sel_data_df.loc[sel_data_df["split"] == "train"]
            if cfg.load_img_model_preds:
                # load out-of-fold predictions for each model
                subdirs = ["Precision_NKI_89_05/Block1/datasets/Aperio/"]  # cfg.data.slide_mapping.subdirs
                train_df = pd.merge(train_df, oof_val_slide_df, how="inner", on="imageName")

            model = instantiate(cfg.model)
            if model_name in ["RandomForestRegressor", "CatBoostRegressor"]:
                model, X_train = fit_sklearn_model(model, train_df, keep_cols)
                train_df['pred_score'] = model.predict(X_train)
            elif model_name == "CoxPHFitter":
                survival_pd_train = sa.preprocess_survival_columns(train_df, keep_cols=keep_cols)
                model = fit_coxph_model(model, survival_pd_train, keep_cols)
                coxph_preds = get_coxscores(model, survival_pd_train)
                train_df = train_df.assign(pred_score=coxph_preds["pred_score"])
            threshold = get_threshold_using_method(train_df, method=threshold_method)
            with open(model_path, 'wb') as model_file:
                pickle.dump(model, model_file)
        if experiment.test:
            stage = "test"
            test_df = get_test_df(sel_data_df)
            if threshold is None:  # for external val
                precision_test_metrics = pd.read_csv(experiment.out_dir + "test_metrics.csv")
                threshold = precision_test_metrics["threshold"].values[0]

            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
            if model_name in ["RandomForestRegressor", "CatBoostRegressor"]:
                logged_metrics, survival_pd_with_preds = evaluate_sklearn_model(model, test_df, keep_cols, stage)
            elif model_name == "CoxPHFitter":
                survival_pd = sa.preprocess_survival_columns(test_df, keep_cols=keep_cols)
                logged_metrics, survival_pd_with_preds = evaluate_coxph_model(model, survival_pd, stage)
            logged_metrics.update({"threshold": threshold})
            run_name = sa.get_run_name(model_name, cfg.clinical_vars_type, cutoff_years, subdir_desc, stage)


            pred_scores_diff_eval_periods = [f'pred_score_limit{year}' for year in [5,8,10,20]]
            slide_level_df = survival_pd_with_preds[["outcome", "pred_score"] + pred_scores_diff_eval_periods ]
            slide_level_df = slide_level_df.join(test_df["imageName"])
            slide_level_df[f"binary_threshold"] = threshold

            # y_pred_bin is 1 if pred_score is greater than threshold, 0 otherwise
            slide_level_df["y_pred_bin"] = (slide_level_df["pred_score"] > threshold).astype(int)
            # Rename columns to match the format used in survival_pd for DL models
            slide_level_df.columns = [col.replace("pred_score", "y_pred_mean") for col in slide_level_df.columns]
            slide_level_df =  slide_level_df.rename(columns={ "outcome": "y_true" })
            slide_level_df.set_index('imageName', inplace=True)

            slide_level_res_dict = slide_level_df.to_dict()
            json_saver.save_selected_data({"stage": "test", "epoch": 0, "subdirs": subdirs,  "ensemble": False,
                                           "train_without_val": True},
                                          "slide_level_results",
                                          slide_level_res_dict
                                          )

            more_metrics = calculate_metrics(survival_pd_with_preds, outcome_col, pred_col, stage)
            da.plot_roc_auc(survival_pd_with_preds, path=experiment.out_dir, y_true_col=outcome_col,
                            y_pred_col=pred_col,
                            name=f"roc_auc_{run_name}", reverse=False)
            logged_metrics.update(more_metrics)

            survival_pd_with_preds = sa.get_risk_groups(survival_pd_with_preds, threshold, pred_col)
            survivorship = do_survivalpd_analysis(stage, survival_pd_with_preds, outcome_col, run_name,
                                                  plot_path=experiment.out_dir, ensemble=False,
                                                  train_without_val=False)
            logged_metrics.update(survivorship)
            for logger in pl_loggers:
                logger.log_metrics(logged_metrics)
            test_kfold_metrics_collector = instantiate(
                cfg.kfold_metrics_collector.test,
                metrics_paths=[experiment.out_dir],
            )
            test_kfold_metrics_collector.collect_metrics_fold(logged_metrics)
    else:
        for fold in experiment.use_folds:
            logging.info(f"Fold {fold}")
            pl_loggers: List = instantiate_loggers(cfg.get("logger"), experiment.fold_dirs[fold])
            for logger in pl_loggers:
                logger.log_hyperparams({"vars": keep_cols})
                logger.log_hyperparams({"model": cfg.model})

            if experiment.train:
                Path(experiment.fold_ckpt_paths[fold]).parent.mkdir(parents=True, exist_ok=True)
                Path(experiment.fold_metrics_paths[fold]).parent.mkdir(parents=True, exist_ok=True)
                train_df, val_df = get_train_val_df(sel_data_df, folds_df, fold, meta_cols.tissue_number_blockid)

                model = instantiate(cfg.model)
                if model_name in ["RandomForestRegressor", "CatBoostRegressor"]:
                    model, _ = fit_sklearn_model(model, train_df, keep_cols)
                    logged_metrics, survival_pd_val_with_preds = evaluate_sklearn_model(model, val_df, keep_cols, "val")
                elif model_name == "CoxPHFitter":
                    survival_pd_train = sa.preprocess_survival_columns(train_df, keep_cols=keep_cols)
                    model = fit_coxph_model(model, survival_pd_train, keep_cols)
                    survival_pd_val = sa.preprocess_survival_columns(val_df, keep_cols=keep_cols)
                    logged_metrics, survival_pd_val_with_preds = evaluate_coxph_model(model, survival_pd_val, "val")
                threshold = get_threshold_using_method(survival_pd_val_with_preds, method=threshold_method)

                with open(experiment.eval_ckpt_fn[fold], 'wb') as model_file:
                    pickle.dump(model, model_file)

                stage = "val"
                more_metrics = calculate_metrics(survival_pd_val_with_preds, outcome_col, pred_col, stage)
                logged_metrics.update(more_metrics)
                run_name = sa.get_run_name(model_name, cfg.clinical_vars_type, cutoff_years, "val", fold)
                survival_pd_val_with_preds = sa.get_risk_groups(survival_pd_val_with_preds, threshold, pred_col)
                survivorship = do_survivalpd_analysis(stage, survival_pd_val_with_preds, outcome_col, run_name, plot_path=experiment.fold_metrics_paths[fold], ensemble=False,
                                       train_without_val=False)
                logged_metrics.update(survivorship)
                for logger in pl_loggers:
                    logger.log_metrics(logged_metrics)
                val_kfold_metrics_collector.collect_metrics_fold(logged_metrics, fold)

            if experiment.test:
                stage = "test"
                test_kfold_metrics_collector = instantiate(
                    cfg.kfold_metrics_collector.test,
                    used_folds=experiment.use_folds,
                    metrics_paths=experiment.fold_metrics_paths,
                )
                test_df = get_test_df(sel_data_df)
                with open(experiment.eval_ckpt_fn[fold], 'rb') as model_file:
                    model = pickle.load(model_file)
                if model_name in ["RandomForestRegressor", "CatBoostRegressor"]:
                    logged_metrics, survival_pd_with_preds = evaluate_sklearn_model(model, test_df, keep_cols, stage)
                elif model_name == "CoxPHFitter":
                    survival_pd = sa.preprocess_survival_columns(test_df, keep_cols=keep_cols)
                    logged_metrics, survival_pd_with_preds = evaluate_coxph_model(model, survival_pd, keep_cols, stage)


                more_metrics = calculate_metrics(survival_pd_with_preds, outcome_col, pred_col, stage)
                run_name = sa.get_run_name(model_name, cfg.clinical_vars_type, cutoff_years, stage, fold)
                logged_metrics.update(more_metrics)
                survivorship = do_survivalpd_analysis(stage, survival_pd_with_preds, outcome_col, run_name,
                                                      plot_path=experiment.fold_metrics_paths[fold], ensemble=False,
                                                      train_without_val=False)
                logged_metrics.update(survivorship)
                for logger in pl_loggers:
                    logger.log_metrics(logged_metrics)
                test_kfold_metrics_collector.collect_metrics_fold(logged_metrics, fold)
            for logger in pl_loggers:
                logger.log_hyperparams(add_to_hparams({}, cfg.model))

    if experiment.train:
        if cfg.data.cv_splitter:
            fit_metrics = val_kfold_metrics_collector.compute_metrics()
            for logger in pl_loggers:
                logger.log_metrics(fit_metrics)
    if experiment.test:
        test_metrics = test_kfold_metrics_collector.compute_metrics()
        for logger in pl_loggers:
            logger.log_metrics(test_metrics)

if __name__ == "__main__":
    main()

