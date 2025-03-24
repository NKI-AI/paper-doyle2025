import logging
from pathlib import Path
import sys
from typing import (
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
    Dict,
)

from drop.tools.json_saver import convert_DictConfig_to_dict_without_instantiation_args
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger, MLFlowLogger


def setup_logging(
    log_dir: Optional[str] = None,
    log_name: Optional[str] = "",
    log_level: str = "DEBUG",
):
    """
    Setup logging for the project.
    For root logger, only log_dir is needed,
    as root logger is referred to using empty string and log_level should be debug.
    To create a specific logger
    child_logger = setup_logging(log_dir, log_name, "INFO")
    log_name of class = type(self).__name__

    Parameters
    ------
    log_name : str
    Name of the logger file to be produced. If empty string, the root logger is referred to.
    log_dir: Optional[str]
        Path where to store the logs.
    log_level : str
        Logging level (i.e. "DEBUG", "INFO", "WARNING", "ERROR", or "EXCEPTION")

    Returns
    ------
    None
    """

    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION"]:
        raise ValueError(f"Unexpected log level got {log_level}.")

    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    # Capture warning from 'py.warnings'
    logging.captureWarnings(True)

    # set formatting for each log message
    formatter_str = "[%(asctime)s | %(name)s | %(levelname)s] %(message)s"
    formatter = logging.Formatter(formatter_str)

    # not needed with hydra
    # # handler for console Stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # handler for root out
    # log_dir = log_dir / Path(datetime.now().strftime(f"{log_name}_%Y-%m-%d_%H-%M-%S.log"))
    if log_dir is not None:
        log_file = Path(log_dir) / Path(f"root.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.debug(f"Created logger{log_name} with level {logger.getEffectiveLevel()}.")

    return logger


def set_tags_mlflow_logger(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """
    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]
    print(trainer)
    if len(trainer._loggers) == 0:
        logging.warning("Logger not found! Skipping hyperparameter logging...")
        return

    tags = {}
    ## Datamodule params
    tags["num_workers"] = cfg.datamodule.num_workers

    ## Data params
    tags["dataset_name"] = cfg.data.dataset_name
    tags["data_name"] = cfg.data.name

    tags["subdirs"] = cfg.data.get("subdirs")

    # hparams["data_sel_params"] = cfg.data.data_sel_params
    tags["event"] = cfg.data.data_sel_params.event  # also stored as y_col in dataset - but then just called outcome
    tags["data_sel_strategy"] = cfg.data.data_sel_strategy
    tags["data_sel_params"] = cfg.data.data_sel_params
    tags["regions_params"] = cfg.get("data.regions_prep.post_processor")  # might need more details
    # hparams["cv_params"] = cfg.data.cv_splitter.cv_params

    # # dataset params
    tags["target"] = cfg.get("dataset").y_col
    tags["target_mpp"] = cfg.get("tiling").region_mpp
    # hparams["tile_configs"] = cfg.get("dataset.tile_configs")
    tags["input_type"] = cfg.get("dataset").input_type

    ## Model params (litmodel params are stored with save_hyperparameters)
    tags["model/params/total"] = sum(p.numel() for p in model.parameters())
    tags["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tags["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    ## Norm params
    tags["normalisation"] = cfg.get("norms").name

    ## Pretraining
    if cfg.experiment.load_pretrain_weights:
        tags["pretrain_mode"] = cfg.get("pretrain").pretrain_mode
    else:
        tags["pretrain_mode"] = "None"


    ## Task params
    tags["eval_ckpt"] = cfg.get("experiment").eval_ckpt
    tags["eval_fold"] = cfg.get("experiment").eval_fold
    tags["kfolds"] = cfg.get("experiment").kfolds
    # store output dir, useful during multirun hyperparameter optimisation
    tags["cwd"] = cfg.get("experiment").out_dir
    tags["ensemble"] = cfg.get("experiment").ensemble
    tags["train_without_val"] = cfg.get("experiment").train_without_val

    ## Trainer params
    if trainer:
        # Important: This is what actually logs the hyperparameter dict to the logger - so that it can be seen in mlfow
        for trainer_logger in trainer._loggers:
            if isinstance(trainer_logger, MLFlowLogger):
                for key, value in tags.items():
                    trainer_logger.experiment.set_tag(trainer_logger.run_id, key, str(value))


def log_hyperparameters_loggers(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """
    cfg = object_dict["cfg"]
    hparams = {}

    ## Task params
    hparams["seed"] = cfg.get("random_seed")

    ## Datamodule params
    hparams["batch_size"] = cfg.datamodule.batch_size
    sampler_cfg = {f"sampler/{k}": v for k, v in cfg.sampler.items() if k not in ["train", "no_train"]}
    hparams = add_to_hparams(hparams, sampler_cfg)

    ## Regions - not sure yet if this is hyperparameter or not
    if cfg.data_prep.input_regions:
        # post_processor
        hparams = add_to_hparams(hparams, cfg.get("regions_prep").post_processor)

    ## Model params (litmodel params are stored with save_hyperparameters)
    model_encoder_cfg = {f"encoder_{k}": v for k, v in cfg.model.encoder.items() if k not in ["_target_", "_partial_"]}
    hparams = add_to_hparams(hparams, model_encoder_cfg)

    model_decoder_cfg = {f"decoder_{k}": v for k, v in cfg.model.decoder.items() if k not in ["_target_", "_partial_"]}
    hparams = add_to_hparams(hparams, model_decoder_cfg)

    model = object_dict["model"]
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    ## Trainer params
    # #not needed, there is only max_epochs in there and that is recorded by litmodule implicitly
    # send hparams to all loggers

    lr_scheduler_cfg = {
        f"scheduler/{k}": v for k, v in cfg.lr_scheduler.scheduler.items() if k not in ["_target_", "_partial_"]
    }
    lr_scheduler_cfg["scheduler/name"] = cfg.lr_scheduler.name
    hparams = add_to_hparams(hparams, lr_scheduler_cfg)

    loss_fn_cfg = {f"loss_fn/{k}": v for k, v in cfg.lit_module.loss_fn.items() if k not in ["_target_", "_partial_"]}
    hparams = add_to_hparams(
        hparams, loss_fn_cfg
    )  # name of loss_fn is added in lit_module directly with save_hyperparameters

    ## Transformation params
    hparams["transforms"] = cfg.get("transforms").name
    hparams["pre_transforms"] = cfg.get("pre_transforms").name

    trainer = object_dict["trainer"]
    if len(trainer._loggers) == 0:
        logging.warning("Logger not found! Skipping hyperparameter logging...")
        return
    ## Important: This is what actually logs the hyperparameter dict to the logger - so that it can be seen in mlfow
    for trainer_logger in trainer._loggers:
        trainer_logger.log_hyperparams(hparams)


def add_to_hparams(hparams: dict, dict_config: DictConfig) -> None:
    """Add contents of dict_config to the hparams dict.
    If the key already exists, it will be overwritten.

    Args:
        hparams (dict): The hparams dict.
        dict_config (DictConfig): The dict config to add to the hparams dict.
    """
    dict_object = convert_DictConfig_to_dict_without_instantiation_args(dict_config)
    hparams.update(dict_object)
    return hparams


