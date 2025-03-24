from typing import List, Optional, Dict
import pandas as pd
import torch
import numpy as np
import logging
from pathlib import Path
from torchmetrics import AUROC, F1Score, MaxMetric, MeanMetric, Specificity, Recall, ConfusionMatrix
from omegaconf import DictConfig
from drop.tools.json_saver import JsonSaver
from data_analysis_new.utils.data_analysis_utils import get_threshold_for_max_metric

# Typing for Pandas and Numpy
DataFrame = pd.DataFrame
NDArray = np.ndarray

# Collection of metrics to be used for training/evaluation
class MetricsCollection:
    def __init__(self):
        """ Initializes metrics used for tracking during model evaluation (per subdir). """
        self.metrics = {
            "epoch": MaxMetric(),
            "best_f1": MaxMetric(),
            "best_auc": MaxMetric(),
            "specificity": Specificity(task="binary"),
            "sensitivity": Recall(task="binary"),
            "f1": F1Score(task="binary"),
            "auc": AUROC(pos_label=1, task="binary"),
            "npr": MaxMetric(),
            "binary_threshold": MaxMetric(),  # Resets every epoch
        }

# Tracker class to handle metric updates
class MetricsTracker:
    def __init__(self, stage: str, metrics: DictConfig, subdir: Optional[str] = "") -> None:
        """
        Metrics tracker to store and compute metrics for different stages (train, val, test).
               # we want to store the regions and their predictions in a json for analysis and for visualisations
        # per epoch then store y_pred and y_true and slide_id and region_index

        """
        self.stage = stage
        self.metrics = metrics
        self.subdir = Path(subdir).stem

    def reset_metrics_tracker(self):
        """ Resets metric values after validation sanity check """
        for metric in self.metrics.values():
            metric.reset()
        logging.info(f"Reset metrics: {list(self.metrics.keys())}.")

    def reset_max_metrics_tracker(self):
        for metric in ["best_f1", "best_auc"]:
            self.metrics[metric].reset()
        print("Reset max metrics for after sanity check")

    def update_metrics_tracker(self, epoch: int, y_true: torch.Tensor, y_pred_mean: torch.Tensor,
                               y_pred_bin: torch.Tensor, binary_prob_threshold: float):
        """ Update metrics with the current epoch's predictions and true labels """
        logging.info(f"Updating metrics for stage: {self.stage}, epoch: {epoch}.")

        # Compute torchmetrics for binary classification
        self.metrics["specificity"](y_pred_bin, y_true)
        self.metrics["sensitivity"](y_pred_bin, y_true)
        self.metrics["f1"](y_pred_bin, y_true)
        self.metrics["auc"](y_pred_mean, y_true)

        confmat = ConfusionMatrix(task="binary", num_classes=2)
        confmat(y_pred_bin, y_true)
        confmat = confmat.compute()  # [[TN, FP], [FN, TP]]

        # Compute Negative Predictive Rate (NPR)
        npr = confmat[0, 0] / (confmat[0, 0] + confmat[1, 0])  # TN / (TN + FN)
        self.metrics["npr"](npr)

        # Update MaxMetrics for epoch and best metrics
        self.metrics["epoch"](epoch)
        self.metrics["binary_threshold"](binary_prob_threshold)
        self.metrics["best_f1"](self.metrics["f1"].compute())
        self.metrics["best_auc"](self.metrics["auc"].compute())

        return self.metrics


# Helper function to collect tensor data from batch outputs
def collect_tensor(step_outputs: List[Dict[str, torch.Tensor]], column_name: str) -> NDArray:
    """Concatenates batch values for the variable of interest and returns as a numpy array."""
    return np.concatenate([i[column_name].cpu().numpy() for i in step_outputs], axis=0)

# Helper function to collect numpy data
def collect_np(step_outputs: List[Dict[str, NDArray]], column_name: str) -> NDArray:
    """Concatenates numpy batch values into a single numpy array."""
    return np.concatenate([i[column_name] for i in step_outputs], axis=0)

# Helper function to collect string data
def collect_str(step_outputs: List[Dict[str, List[str]]], column_name: str) -> List[str]:
    """Concatenates batch string values for the specified column."""
    return sum([i[column_name] for i in step_outputs], [])


def get_region_df_from_step_outputs(step_outputs: List[Dict], server_cols) -> DataFrame:
    """ For Tile-supervision model, where batches are mixed"""
    y_pred = collect_np(step_outputs, "y_pred")
    logging.info(f"Mean y_pred per instance is {np.mean(y_pred)}")
    y_true = collect_np(step_outputs, "y")
    subdir = collect_str(step_outputs, "subdir")
    slide_ids = collect_str(step_outputs, server_cols.name)
    region_idc = collect_np(step_outputs, "region_index")  # now same name for tiles and regions
    keep_columns = ["y_pred", "y", server_cols.name, "region_index", server_cols.subdir]
    try:
        assert len(y_pred) == len(y_true) == len(slide_ids) == len(region_idc) == len(subdir)
    except AssertionError:
        print("Lengths of y_pred, y_true, slide_ids, region_idc, subdir are not equal. "
              "Should be the case of Tile-supervision model.")
    df = pd.DataFrame(list(zip(y_pred, y_true, slide_ids, region_idc, subdir)), columns=keep_columns, )
    return df

def get_slide_df_from_step_outputs(step_outputs: List[Dict], server_cols) -> DataFrame:
    """ For Slide-supervision model, where each batch is a slide."""
    y_pred = collect_tensor(step_outputs, "y_pred")
    logging.info(f"Mean y_pred per instance is {np.mean(y_pred)}")
    y_true = collect_tensor(step_outputs, "y")
    subdir = collect_str(step_outputs, "subdir")
    slide_ids = collect_str(step_outputs, server_cols.name)
    region_idc = collect_tensor(step_outputs, "region_index")  # now same name for tiles and regions
    keep_columns = ["y_pred", "y", server_cols.name, "region_index", server_cols.subdir]
    df = pd.DataFrame(list(zip(y_pred, y_true, slide_ids, region_idc, subdir)),columns=keep_columns,)
    return df


# Main class that collects and tracks metrics for multiple directories
class MetricsTrackerContainer:
    def __init__(
        self,
        subdirs: List[str],
        stage: str,
        server_cols: DictConfig,
        ensemble: Optional[bool] = False,
        train_without_val: Optional[bool] = False,
        separate_metrics_per_ds: Optional[bool] = False,
        dirpath: Optional[str] = None,
        store_region_level_results: Optional[bool] = False,
        store_slide_level_results: Optional[bool] = False,
        bin_prob_cutoff: Optional[float] = None,
    ) -> None:
        """
        Handles collection and tracking of metrics for different subdirectories and stages of training/evaluation.
        """
        self.stage = stage
        self.server_cols = server_cols
        self.loss = MeanMetric().to(torch.device("cuda", 0))
        self.bin_prob_cutoff = bin_prob_cutoff
        self.ensemble = ensemble
        self.train_without_val = train_without_val
        self.store_region_level_results = store_region_level_results
        self.store_slide_level_results = store_slide_level_results
        self.predictions_fn = f"{dirpath}predictions.json" if dirpath else None
        self.json_saver = JsonSaver("predictions", self.predictions_fn) if self.predictions_fn else None
        self.subdirs = subdirs
        self.separate_metrics_per_ds = separate_metrics_per_ds

        # Initialize separate metric trackers for each subdirectory
        self.metrics_subdir_trackers = {
            subdir: MetricsTracker(stage, MetricsCollection().metrics, subdir)
            for subdir in self.subdirs
        } if self.separate_metrics_per_ds else {
            "": MetricsTracker(stage, MetricsCollection().metrics, "")
        }

    def reset_metrics(self):
        """Reset all metrics trackers."""
        for tracker in self.metrics_subdir_trackers.values():
            tracker.reset_metrics_tracker()

    def reset_max_metrics(self):
        """Reset max metrics trackers."""
        for tracker in self.metrics_subdir_trackers.values():
            tracker.reset_max_metrics_tracker()

    def get_step_output_df(self, step_outputs: List[Dict], epoch: int) -> DataFrame:
        """Process and convert batch-level outputs into a DataFrame."""
        # Assuming this function needs to collect and process batch data
        # Here's the code from your previous version
        if len(step_outputs[0]["region_index"]) == len(step_outputs[0][self.server_cols.name]):
            df = get_region_df_from_step_outputs(step_outputs, self.server_cols)
        elif len(step_outputs[0]["region_index"]) > len(step_outputs[0][self.server_cols.name]):
            df = get_slide_df_from_step_outputs(step_outputs, self.server_cols)
        else:
            raise ValueError("Unknown data format.")
        return df

    def agg_ypred_by_prob_thresh(
        self, y_pred_auc_mean: PandasSeries, bin_prob_cutoff: float
    ) -> Tuple[PandasSeries, str]:
        """
        Calculate wsi level predictions. The cutoff refers to the mean prediction value.
        """
        y_pred_auc_th = y_pred_auc_mean.apply(lambda x: np.where(x >= bin_prob_cutoff, 1, 0))
        identifier = f"y_pred_by_threshold{bin_prob_cutoff}"
        return y_pred_auc_th, identifier


    def get_slide_results(self, df: DataFrame, epoch: int) -> Tuple[torch.Tensor]:
        """Returns predicted and target tensors for slide level predictions.
        Returns
        ------
        y_pred: IntTensor
            Tensor of predictions
        y_true: FloatTensor
            targets.
        """

        # Collect slide level predictions and targets - by grouping by id. - as a np.array
        slide_level_res_dict = {}
        y_true_series = df.groupby(self.server_cols.name)["y"].apply(lambda x: max(x))
        y_pred_series = df.groupby(self.server_cols.name)["y_pred"].apply(lambda x: np.array(x))

        # Aggregate the slide_level y_preds

        slide_level_res_dict["y_true"] = y_true_series.to_dict()
        y_pred_mean = y_pred_series.apply(lambda x: x.mean())
        slide_level_res_dict["y_pred_mean"] = y_pred_mean.to_dict()

        if  isinstance(self.bin_prob_cutoff, float):
            binary_threshold = self.bin_prob_cutoff
        else:
            binary_threshold, _ = get_threshold_for_max_metric(pd.DataFrame(slide_level_res_dict), y_true_col="y_true", y_pred_col="y_pred_mean")

        y_pred_bin, y_pred_identifier = self.agg_ypred_by_prob_thresh(y_pred_mean,  binary_threshold)
        slide_level_res_dict["binary_threshold"] = binary_threshold
        slide_level_res_dict["y_pred_bin"] = y_pred_bin.to_dict()
        subdir_desc = df[self.server_cols.subdir].unique().tolist()
        if self.store_slide_level_results is True:
            self.json_saver.save_selected_data(
                {"stage": self.stage, "epoch": epoch, "subdirs": subdir_desc,
                "ensemble": self.ensemble, "train_without_val": self.train_without_val},
                "slide_level_results",
                slide_level_res_dict,
            )
        # Convert back to tensor for metrics calculation.
        y_true = torch.FloatTensor(y_true_series.values.astype(np.float32))  # is it better to use an IntTensor here?
        y_pred_prob = torch.FloatTensor(y_pred_mean.values.astype(np.float32))
        y_pred_bin = torch.FloatTensor(y_pred_bin.values.astype(np.float32))
        binary_threshold = torch.FloatTensor([binary_threshold])
        print(y_pred_bin, "aggregated slide level predictions")

        return y_true, y_pred_prob, y_pred_bin, binary_threshold


    def compute_epoch(self, step_outputs: List[Dict], epoch: int):
        """Compute and update metrics for the current epoch."""
        df = self.get_step_output_df(step_outputs, epoch)
        for subdir in self.metrics_subdir_trackers.keys():
            if self.separate_metrics_per_ds:
                subdir_df = df[df["subdir"] == subdir].copy().reset_index(drop=True)
                if len(subdir_df) == 0:
                    logging.warning(f"No data for subdir {subdir} in epoch {epoch}.")
                    continue
                y_true, y_pred_mean, y_pred_bin, binary_prob_thrsh = self.get_slide_results(subdir_df, epoch)
            else:
                y_true, y_pred_mean, y_pred_bin, binary_prob_thrsh = self.get_slide_results(df, epoch)
            self.metrics_subdir_trackers[subdir].update_metrics_tracker(epoch, y_true, y_pred_mean, y_pred_bin, binary_prob_thrsh)

        return self.metrics_subdir_trackers

    def save_region_level_results(self, df: DataFrame, epoch: int):
        if self.store_region_level_results:  # can be tiles or detections
            if self.stage != "test":
                logging.warning("storing region level results only for the last epoch")
                epoch = "last"
            self.json_saver.save_selected_data({"stage": self.stage, "epoch": epoch, "subdirs": self.subdirs,
                                                "ensemble": self.ensemble,
                                                "train_without_val": self.train_without_val},
                                               "regions_results", df.to_dict())
            # res = self.json_saver.read_selected_data(epoch)[0]['results']




