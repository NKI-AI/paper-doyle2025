import os
import pandas as pd
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict, TypeVar, Any
import logging
DataFrame = TypeVar("pandas.core.frame.DataFrame")
Series = TypeVar("pandas.core.series.Series")
NDArray = TypeVar("np.array")
from drop.utils.misc import write_dict_to_csv


class MetricsCollectorFolds:
    """
    Collects and aggregates metrics from CSV logs across multiple folds.

    This class is responsible for reading fold-specific metric logs,
    computing aggregate statistics (mean, std), and writing results to a summary file.

    Attributes:
        stage (str): Stage of evaluation (e.g., "val" or "test").
        out_dir (str): Output directory where results are saved.
        metrics_paths (List[str]): Paths to individual fold metric files.
        data_name (Optional[str]): Dataset name used in naming convention.
        used_folds (Optional[List[int]]): List of folds to include in aggregation.
        eval_ckpt (Optional[str]): Checkpoint identifier for evaluation.
        fold_metrics_fn (Optional[str]): Filename for fold-level metrics (default: "metrics.csv") (Output of teh csv_logger).
        ensemble (Optional[bool]): Whether the metrics are for an ensemble model.
        train_without_val (Optional[bool]): Whether the model was trained without validation set.
    """

    def __init__(
            self,
            stage: str,
            out_dir: str,
            metrics_paths: List[str],
            data_name: Optional[str] = None,
            used_folds: Optional[List[int]] = None,
            eval_ckpt: Optional[str] = None,
            fold_metrics_fn: Optional[str] = "metrics.csv",
            ensemble: Optional[bool] = False,
            train_without_val: Optional[bool] = False,
    ) -> None:
        self.metrics = None
        self.used_folds = used_folds if used_folds is not None else [0]
        self.metrics_paths = metrics_paths

        # Construct overall metrics filename
        overall_fn = f"{stage}_fold_metrics.txt"
        if ensemble:
            overall_fn = f"ensemble_{overall_fn}"
        if train_without_val:
            overall_fn = f"train_without_val_{overall_fn}"
        if eval_ckpt:
            overall_fn = f"{eval_ckpt.split('.ckpt')[0]}_{overall_fn}"
        if data_name:
            overall_fn = f"{data_name}_{overall_fn}"

        self.kfolds_metrics_fn = os.path.join(out_dir, overall_fn)
        self.run_metrics_fn = f"{stage}_{fold_metrics_fn}"

        # Ensure metric directories exist
        for path in self.metrics_paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def collect_metrics_fold(self, logged_metrics: Dict, fold: int = 0) -> None:
        """
        Write test results of the selected evaluation checkpoint to CSV.

        Args:
            logged_metrics (Dict): Dictionary of metric values.
            fold (int, optional): Fold index. Defaults to 0.
        """
        self.metrics = list(logged_metrics.keys())
        logged_metrics = {
            k: round(float(v), 5) if isinstance(v, torch.Tensor) else v
            for k, v in logged_metrics.items()
        }
        metrics_df = pd.DataFrame([logged_metrics])

        fold = fold or 0  # Ensure fold is 0 if None
        metrics_df.to_csv(os.path.join(self.metrics_paths[fold], self.run_metrics_fn), index=False)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Computes and writes aggregated metrics across folds.

        Returns:
            Dict[str, float]: Dictionary containing average and standard deviation of metrics.
        """
        write_dict_to_csv(self.kfolds_metrics_fn, "w", self.metrics, write_header=True)

        # Collect fold results
        for fold in self.used_folds:
            fold_metrics = self.read_from_metrics_file(fold)
            write_dict_to_csv(self.kfolds_metrics_fn, "a", fold_metrics, write_header=False)

        # Compute mean and std
        res_df = pd.read_csv(self.kfolds_metrics_fn, sep="\t")
        res_df.loc["Average"] = res_df.mean()
        res_df.loc["Std"] = res_df.std()
        res_df = res_df.round(3)
        res_df.to_csv(self.kfolds_metrics_fn, sep="\t")

        self.print_results_to_console()

        # Initialize the return_dict to collect the results
        return_dict = {}

        # If only one fold is used, add the raw average values without '_avg' and '_std'
        if len(self.used_folds) == 1:
            for metric in self.metrics:
                return_dict[f"{metric}"] = res_df.loc["Average"][metric]
        else:
            # If multiple folds are used, add average and standard deviation with suffixes
            for metric in self.metrics:
                return_dict[f"{metric}_avg"] = res_df.loc["Average"][metric]
                return_dict[f"{metric}_std"] = res_df.loc["Std"][metric]

    def read_from_metrics_file(self, fold: int) -> Dict[str, float]:
        """
        Reads the best metric values from a fold-specific file.

        Args:
            fold (int): Fold index.

        Returns:
            Dict[str, float]: Dictionary of max values per metric.
        """
        fold_file = os.path.join(self.metrics_paths[fold], self.run_metrics_fn)
        fold_df = pd.read_csv(fold_file)

        self.metrics = [col for col in fold_df.columns if not col.startswith("Unnamed")]
        return fold_df[self.metrics].max().to_dict()

    def print_results_to_console(self) -> None:
        """
        Prints the computed metrics to the console.
        """
        res_df = pd.read_csv(self.kfolds_metrics_fn, sep="\t")
        logging.info(f"Results for {self.used_folds} folds from {self.kfolds_metrics_fn}")
        logging.info("--------------------------------")
        logging.info(res_df.to_string())

    def get_group_cutoff(self) -> float:
        """
        Retrieves the binary classification threshold for validation.

        If multiple validation datasets exist, the average threshold is returned.

        Returns:
            float: Computed binary classification threshold.
        """
        res_df = pd.read_csv(self.kfolds_metrics_fn, sep="\t", index_col=0)

        try:
            return res_df.at["Average", "val/binary_threshold"]
        except KeyError:
            val_cols = [col for col in res_df.columns if col.startswith("val") and col.endswith("binary_threshold")]
            return round(res_df[val_cols].loc["Average"].mean(), 3)

