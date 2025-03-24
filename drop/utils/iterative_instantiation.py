from omegaconf import DictConfig
from typing import List, Dict, Any, Optional, Union
import logging
from hydra.utils import instantiate
from pathlib import Path
from drop.lit_modules.cls.metrics_tracker import MetricsTrackerContainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, MLFlowLogger


def instantiate_list_of_cfgs(list_cfg: DictConfig, fold_dir: Optional[Union[str, None]] = None) -> List:
    """Instantiates callbacks from config."""
    list_of_objects: List = []
    if not list_cfg:
        logging.warning("No List of configs found! Skipping..")
        return list_of_objects

    if not isinstance(list_cfg, DictConfig):
        raise TypeError("Config must be a DictConfig!")

    for cb_name, cb_conf in list_cfg.items():
        if isinstance(cb_conf, DictConfig):
            logging.info(f"Instantiating config <{cb_name}>")
            cb = instantiate(cb_conf)
            if fold_dir:
                cb.dirpath = Path(cb.dirpath) / Path(fold_dir)
            list_of_objects.append(cb)

    return list_of_objects


def instantiate_metricstracker(
    dict_cfg: DictConfig, key: str, metrics_dir: str, subdirs: List[str]
) -> Dict[str, MetricsTrackerContainer]:
    """Instantiates metrics trackers from config."""
    dict_of_cfgs: Dict[str, Any] = {}
    if not dict_cfg:
        logging.warning("No metrics tracker configs found! Skipping..")
        return dict_of_cfgs
    if not isinstance(dict_cfg, DictConfig):
        raise TypeError("metrics tracker config must be a DictConfig!")
    for name in dict_cfg.keys():
        if isinstance(dict_cfg[name], DictConfig):
            logging.info(f"Instantiating config <{name}>")
            obj = instantiate(dict_cfg[name], dirpath=metrics_dir, subdirs=subdirs)
            dict_of_cfgs[name] = obj

    return dict_of_cfgs


def instantiate_loggers(logger_cfg: DictConfig, fold_dir: Optional[Union[str, None]] = None) -> List:
    """Instantiates loggers from config."""
    loggers: List = []

    if not logger_cfg:
        logging.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig to be instantiated!")

    for logname, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logging.info(f"Instantiating logger <{lg_conf._target_}>")
            logger = instantiate(lg_conf)
            logger_spec = fold_dir if fold_dir else "no_folds"
            if isinstance(logger, TensorBoardLogger) or isinstance(logger, CSVLogger):
                logger._version = logger_spec
            elif isinstance(logger, MLFlowLogger):
                # logger._prefix = fold_dir.split("/")[0]
                logger._run_name = logger_spec.split("/")[0]

            loggers.append(logger)

    return loggers
