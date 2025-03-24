# hydra
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
# reproducibility
from drop.utils.misc import set_random_seed
import torch  # for enforce deterministic
# logging
import logging
# logging to loggers (mlflow, tensorboard, ...)
import mlflow
from drop.utils.logging import log_hyperparameters_loggers, set_tags_mlflow_logger
# typing
DataFrame = TypeVar("pandas.core.frame.DataFrame")
from typing import List, Any, Optional, TypeVar, Dict
# setup
from drop.utils.iterative_instantiation import (
    instantiate_list_of_cfgs,
    instantiate_metricstracker,
    instantiate_loggers,
)
# lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
# load environment paths
from drop.utils.env_setup import setup_environment
setup_environment()
# testing
from drop.lit_modules.cls.metrics_tracker import MetricsTrackerContainer
from drop.models.ensemble_model import EnsembleModel

def log_parameters_mlflow(log_dict):
    mlflow.pytorch.autolog()
    log_hyperparameters_loggers(log_dict)
    set_tags_mlflow_logger(log_dict)


def get_ensemble_model(experiment: Any, model_cfg: DictConfig) -> EnsembleModel:
    test_ckpt_paths = experiment.fold_test_ckpts
    model = EnsembleModel(
        folds_used=experiment.use_folds,
        model_paths=test_ckpt_paths,
        model_config=model_cfg,
    )
    return model

def setup(
    cfg: DictConfig, experiment: Any, folds_df: DataFrame, sel_data_df: DataFrame, fold: Optional[int] = None
) -> None:
    ds = instantiate(cfg.dataset, data_df=sel_data_df)
    datamodule: LightningDataModule = instantiate(cfg.datamodule, dataset=ds, folds_df=folds_df, fold=fold)
    logging.info(f"Built datamodule")
    transforms = instantiate(cfg.transforms)

    if experiment.ensemble:
        model = get_ensemble_model(experiment, cfg.model)
    else:
        model = instantiate(cfg.model)
        if not type(model.encoder) == DictConfig:
            model.encoder.add_pretrain_weights(pretrain_weights_path=experiment.pretrain_weights_path)

    if type(fold) == int:
        metrics_dir = experiment.fold_metrics_paths[fold]
        fold_dir = experiment.fold_dirs[fold]
    else:
        metrics_dir = experiment.metrics_dir
        fold_dir = None

    metrics_trackers: Dict[str, MetricsTrackerContainer] = instantiate_metricstracker(
        cfg.metrics_trackers,
        "trackers",
        metrics_dir=metrics_dir,
        subdirs=sel_data_df["subdir"].unique().tolist(),
    )
    lit_model: LightningModule = instantiate(
        cfg.lit_module,
        augmentations=transforms,
        model=model,
        metrics_trackers=metrics_trackers,
    )
    callbacks: List[Callback] = instantiate_list_of_cfgs(cfg.get("callbacks"), fold_dir)
    pl_loggers: List = instantiate_loggers(cfg.get("logger"), fold_dir)
    trainer: Trainer = instantiate(
        cfg.trainer,
        logger=pl_loggers,
        callbacks=callbacks,
        default_root_dir=cfg.hydra_runtime_output_dir,
        log_every_n_steps=20,
        deterministic=cfg.enforce_deterministic,
        profiler=None,
    )
    log_parameters_mlflow(log_dict={"cfg": cfg, "model": lit_model.model, "trainer": trainer})

    return datamodule, lit_model, trainer, pl_loggers


def test(datamodule, lit_model, trainer, ckpt_path=None, ensemble=False):
    if ensemble:
        trainer.test(lit_model, datamodule)
    else:
        trainer.test(lit_model, datamodule, ckpt_path=ckpt_path)

@hydra.main(version_base="1.2", config_path="../../../configs/", config_name="config")
def main(cfg: DictConfig):
    set_random_seed(cfg.random_seed)
    if cfg.enforce_deterministic:
        torch.use_deterministic_algorithms(mode=True)

    data_prep = instantiate(cfg.data_prep)
    sel_data_df, folds_df = data_prep.prepare_data()
    data_name = "".join([name for name in data_prep.datasets.keys()])
    experiment = instantiate(cfg.experiment)
    experiment.setup(cfg)
    add = "_ens" if experiment.ensemble else ""
    val_kfold_metrics_collector = instantiate(
        cfg.kfold_metrics_collector.val,
        used_folds=experiment.use_folds,
        metrics_paths=experiment.fold_metrics_paths,
    )
    if not cfg.data.cv_splitter:
        datamodule, lit_model, trainer, pl_loggers = setup(cfg, experiment, folds_df, sel_data_df, fold=None)
        if experiment.train:
            assert not experiment.ensemble, "Ensemble is not a valid option for train_without_val training"
            datamodule.setup("train_without_val")
            trainer.fit(lit_model, datamodule.train_dataloader())
        if experiment.test:
            # this is the only case an ensemble model may be used.
            lit_model.tracker_container.test.bin_prob_cutoff = val_kfold_metrics_collector.get_group_cutoff()
            test(datamodule, lit_model, trainer, ensemble=experiment.ensemble)
            if experiment.ensemble:
                ckpt_path = cfg.callbacks['modelcheckpoint_all'].dirpath
                trainer.save_checkpoint(f"{ckpt_path}/ensemble{cfg.data.name}{add}.ckpt")
            test_kfold_metrics_collector = instantiate(
                cfg.kfold_metrics_collector.test,
                eval_ckpt=experiment.eval_ckpt,
                metrics_paths=[experiment.metrics_dir],
            )
            test_kfold_metrics_collector.collect_metrics_fold(trainer.logged_metrics)
    else:
        for fold in experiment.use_folds:
            datamodule, lit_model, trainer, pl_loggers = setup(cfg, experiment, folds_df, sel_data_df, fold=fold)
            if experiment.train:
                trainer.fit(lit_model, datamodule, ckpt_path=experiment.fold_resume_ckpts[fold])
                logging.info(f"Training fold {fold}.")
                val_kfold_metrics_collector.collect_metrics_fold(trainer.logged_metrics, fold)
            if experiment.val_only:
                datamodule.setup("fit")
                trainer.validate(lit_model, datamodule.val_dataloader(), ckpt_path=experiment.fold_test_ckpts[fold])
                val_kfold_metrics_collector.collect_metrics_fold(trainer.logged_metrics, fold)
            if experiment.test:
                logging.info(f"Testing fold {fold}. {experiment.fold_test_ckpts[fold]}")
                test(datamodule, lit_model, trainer, ckpt_path=experiment.fold_test_ckpts[fold])
                test_kfold_metrics_collector = instantiate(
                    cfg.kfold_metrics_collector.test,
                    used_folds=experiment.use_folds,
                    eval_ckpt=experiment.eval_ckpt,
                    metrics_paths=experiment.fold_metrics_paths,
                )
                test_kfold_metrics_collector.collect_metrics_fold(trainer.logged_metrics, fold)
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
