from omegaconf import OmegaConf, DictConfig
from typing import Union, Tuple, TypeVar, List, Optional
DataFrame = TypeVar("pandas.core.frame.DataFrame")
from drop.utils.checkpoint_paths import get_weights_path, get_lightning_ckpt_path, get_metrics_path


class Experiment:
    """Setting up an experiment, defining the paths and the mode of the experiment. Specifically checkpoint paths,
    metrics paths, and weight paths.
    """
    def __init__(
        self,
        # model
        out_dir: str,
        kfolds: Union[int, None],
        paths_cfg: DictConfig = None,
        load_pretrain_weights=False,
        resume_from_ckpt=False,
        resume_ckpt="last",
        eval_ckpt="best_loss",
        eval_fold=0,
        use_specific_folds: Optional[List[int]] = None,
        ensemble: bool = False,
        train_without_val: bool = False,
        # mode
        train: bool = False,
        test: bool = False,
        inference: bool = False,
        val_only: bool = False,
        visualise: bool = False,
    ):
        self.out_dir = out_dir
        if paths_cfg:
            self.metrics_dir = f"{out_dir}/{paths_cfg.metrics_dir}"
        else:
            self.metrics_dir = f"{out_dir}/metrics/"
        assert not (ensemble and train_without_val)
        self.ensemble = ensemble
        self.train_without_val = train_without_val

        self.load_pretrain_weights = load_pretrain_weights
        self.resume_from_ckpt = resume_from_ckpt
        self.resume_ckpt_fn = f"{resume_ckpt}.ckpt"
        self.eval_ckpt_fn = f"{eval_ckpt}.ckpt"
        self.eval_fold = eval_fold
        self.train = train
        self.test = test
        self.inference = inference
        self.val_only = val_only
        self.visualise = visualise


        self.fold_dirs = [f"fold{fold}/" for fold in range(kfolds)] if kfolds else [None]
        self.use_folds = use_specific_folds if use_specific_folds else list(range(kfolds))

        # paths to be set up:
        self.pretrain_weights_path = None
        self.fold_metrics_paths = None

        self.fold_resume_ckpts = None
        self.eval_ckpt = None
        self.fold_test_ckpts = None

    def setup(self, cfg: DictConfig):

        if self.load_pretrain_weights:
            if cfg.model.encoder.get("resnet_arch", None) is not None:
                model_name = cfg.model.encoder.resnet_arch
            else:
                model_name = None
            self.pretrain_weights_path = get_weights_path(cfg.paths.project, model_name, cfg.pretrain)

        self.fold_metrics_paths = self.get_fold_wise_list(
        [self.out_dir, cfg.paths.metrics_dir], self.fold_dirs, get_metrics_path
        )

        self.fold_resume_ckpts = self.get_fold_wise_list(
            [self.out_dir, cfg.paths.checkpoints_dir, self.resume_ckpt_fn],
            self.fold_dirs,
            get_lightning_ckpt_path,
            cond=[self.resume_from_ckpt],
        )

        if self.test or self.val_only:
            self.fold_test_ckpts = self.get_fold_wise_list(
                [self.out_dir, cfg.paths.checkpoints_dir, self.eval_ckpt_fn], self.fold_dirs, get_lightning_ckpt_path
            )

        if self.inference:
            self.eval_ckpt = get_lightning_ckpt_path(
                self.out_dir, cfg.paths.checkpoints_dir, self.eval_ckpt_fn, self.fold_dirs[self.eval_fold]
            )

    def get_fold_wise_list(
        self, arguments: List, fold_dirs: List[str], func, cond: Optional[List[bool]] = [True]
    ) -> List[str]:
        """Get for each fold the correct path in a list"""
        if any(cond):
            return [func(*arguments, fold_dir) for fold_dir in fold_dirs]
        else:
            return [None for fold_dir in fold_dirs]
