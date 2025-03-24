import logging
from typing import List, Optional, Tuple, Dict, Any, TypeVar
NDArray = TypeVar("np.array")
T = TypeVar("torch.Tensor")
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import torch


class RegionMILModule(pl.LightningModule):
    def __init__(
        self,
        augmentations: Dict[str, Any],
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: str,
        scheduler: str,
        monitor: str,
        lr: float,
        weight_decay: float,
        metrics_trackers: Dict[str, Any],
    ):
        """
        Initializes the RegionMILModule.

        Parameters:
            augmentations (Dict[str, Any]): Augmentations to be applied to the data.
            model (nn.Module): The model architecture.
            loss_fn (nn.Module): Loss function to be used.
            optimizer (str): The optimizer type, e.g., "Adam".
            scheduler (str): Learning rate scheduler type.
            monitor (str): Metric to monitor for early stopping or lr scheduling.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            metrics_trackers (Dict[str, Any]): Dictionary for tracking metrics for training, validation, and testing.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["model", "metrics_trackers", "loss_fn", "augmentations"], logger=True
        )

        self._augmentations = augmentations.get("augmentations", None)
        self.loss_fn = loss_fn
        self.optimizer_type = optimizer
        self.scheduler_type = scheduler
        self.monitor = monitor
        self.model = model
        self.tracker_container = metrics_trackers

        self.train_output = []
        self.val_output = []
        self.test_output = []

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            logging.error(f"Optimizer {self.optimizer_type} is not supported.")
            raise ValueError(f"Optimizer {self.optimizer_type} is not supported.")

        scheduler = self.scheduler.scheduler(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.monitor}

    def on_train_epoch_start(self):
        """Log the learning rate at the start of each epoch."""
        lr_scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        lr = lr_scheduler.get_last_lr()[0]
        self.log("lr", lr, prog_bar=False)

    def training_step(self, batch, batch_idx: int):
        """Perform a single step of training."""
        stage = "train"
        res = self.step(batch=batch, stage=stage)
        self.log_step_metric(stage, "loss", res["loss"])
        self.tracker_container[stage].loss.update(res["loss"])
        self.train_output.append(res)
        return res

    def on_train_epoch_end(self):
        """Log metrics at the end of the training epoch."""
        stage = "train"
        outputs = self.train_output
        stage_metrics_trackers = self.tracker_container[stage].compute_epoch(outputs, self.current_epoch)
        self.log_epoch_end_tracker(stage, stage_metrics_trackers, self.tracker_container[stage].loss)
        self.tracker_container[stage].reset_metrics()
        self.tracker_container[stage].loss.reset()
        self.train_output.clear()

    def validation_step(self, batch, batch_idx: int):
        """Perform a single step of validation."""
        stage = "val"
        res = self.step(batch=batch, stage=stage)
        self.log_step_metric(stage, "loss", res["loss"])
        self.tracker_container[stage].loss.update(res["loss"])
        self.val_output.append(res)
        return res

    def on_validation_epoch_end(self):
        """Log metrics at the end of the validation epoch."""
        stage = "val"
        outputs = self.val_output
        if not self.trainer.sanity_checking:
            stage_metrics_trackers = self.tracker_container[stage].compute_epoch(outputs, self.current_epoch)
            self.log_epoch_end_tracker(stage, stage_metrics_trackers, self.tracker_container[stage].loss)
        self.tracker_container[stage].reset_metrics()  # to empty memory
        self.tracker_container[stage].loss.reset()
        self.val_output.clear()

    def test_step(self, batch, batch_idx: int):
        """Perform a single step of testing."""
        stage = "test"
        res = self.step(batch=batch, stage=stage)
        self.log_step_metric(stage, "loss", res["loss"])
        self.tracker_container[stage].loss.update(res["loss"])
        self.test_output.append(res)

        return res

    def on_test_epoch_end(self):
        """Log metrics at the end of the test epoch."""
        stage = "test"
        outputs = self.test_output
        stage_metrics_trackers = self.tracker_container[stage].compute_epoch(outputs, self.current_epoch)
        self.log_epoch_end_tracker(stage, stage_metrics_trackers, self.tracker_container[stage].loss)
        self.test_output.clear()


    def step(self, batch: Dict, stage: str) -> Dict[str, Any]:
        """
        Shared step in which model predictions are generated and loss is computed.
        Parameters:
            batch: Dict containing  x:Tensor, y:Tensor, imageName:List[str], region_index:Tensor, subdir:List[str], sample_index:Tensor
        """
        imgs = batch.pop("x")
        stage = "no_train" if stage != "train" else "train"
        if self._augmentations and stage in self._augmentations:
            for aug in self._augmentations[stage]:
                imgs = aug(imgs)

        extra_features = batch.get("extra_features", None)
        if extra_features:
            extra_features = batch.pop("extra_features")
            extra_features = torch.stack(extra_features, dim=-1).float()    # [n_features, B]
        else:
            extra_features = None

        logits, _ = self.model(imgs, extra_features)

        if logits.shape[1] == 1 and isinstance(self.loss_fn, nn.CrossEntropyLoss):
            logits = logits.squeeze(dim=1)  # [1]
            batch["y"] = batch["y"].float()
        elif logits.shape[1] == 2 and isinstance(self.loss_fn, nn.CrossEntropyLoss):
            logits = logits.float()  # [B, C]
            batch["y"] = batch["y"].long()

        loss = self.loss_fn(logits, batch["y"])

        # Apply activation function and detach
        y_pred = torch.sigmoid(logits).detach() if logits.shape[1] == 1 else F.softmax(logits, dim=1).detach()[:, 1]

        res = {
            key: batch[key] for key in ["imageName", "y", "region_index", "subdir", "sample_index"]
        }
        res.update({"loss": loss, "y_pred": y_pred})
        return res

    def log_epoch_end_tracker(self, stage, stage_metrics_trackers, stage_loss):
        """Logs metrics at the end of each epoch."""
        for subdir, metrics_tracker in stage_metrics_trackers.items():
            for metric_name, metric_value in metrics_tracker.metrics.items():
                self.log_epoch_metric(f"{stage}{metrics_tracker.subdir}", metric_name, metric_value)
        self.log_epoch_metric(stage, "loss", stage_loss)

    def log_step_metric(self, stage, metric_name, metric):
        """Logs metrics at each step without averaging."""
        self.log(f"{stage}/{metric_name}", float(metric), on_step=True, on_epoch=False, prog_bar=False, logger=False)

    def log_epoch_metric(self, stage, metric_name, metric):
        """Logs metrics with running average after each epoch."""
        metric_value = float(metric.compute())
        self.log(f"{stage}/{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        print(f"{stage}/{metric_name}", metric_value)
