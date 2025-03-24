import logging
from typing import Any
from pathlib import Path
from typing import Optional



def get_lightning_ckpt_path(exp_dir, ckpt_dir, ckpt_fn, fold_dir) -> str:
    checkpoints_path = f"{exp_dir}{ckpt_dir}"
    return f"{checkpoints_path}{fold_dir}{ckpt_fn}"

def get_metrics_path(exp_dir: str, metrics_dir: str, fold_dir: str):
    return f"{exp_dir}{metrics_dir}{fold_dir}"

def get_weights_path(project_dir, resnet_arch, cfg_pretrain, d2: Optional[bool] = False) -> Any:
    """
    Returns
    ---
    Model weigths in torch format
    """
    d2 = "_d2" if d2 else ""
    if cfg_pretrain.pretrain_mode == "hissl":
        ssl_method = f"{resnet_arch}_{cfg_pretrain.method}_{cfg_pretrain.init}"
        ssl_model = f"{ssl_method}_ep{cfg_pretrain.epoch:03d}"
        model_weights_path = f"{project_dir}{cfg_pretrain.ssl_torch_d2_ckpt_dir}{ssl_method}/{ssl_model}{d2}.torch"

    elif "detectron2" in cfg_pretrain.pretrain_mode:
        ssl_method = cfg_pretrain.ssl_method
        ssl_model = cfg_pretrain.ssl_model
        model_weights_path = f"{project_dir}{cfg_pretrain.ssl_torch_d2_ckpt_dir}{ssl_method}/{ssl_model}"
    elif cfg_pretrain.pretrain_mode == "ibotvit":
        model_weights_path = f"{project_dir}{cfg_pretrain.ssl_ckpt_dir}{cfg_pretrain.model_name}"
    else:
        ssl_method = cfg_pretrain.ssl_method
        ssl_model = cfg_pretrain.ssl_model
        model_weights_path = f"{project_dir}{cfg_pretrain.ssl_torch_d2_ckpt_dir}{ssl_method}/{ssl_model}{d2}.torch"  # check this for detectron

    logging.info(f"Setting pretrain model weights path to: {model_weights_path}")

    return model_weights_path
