import numpy as np
import logging
from drop.data_analysis_new.predictions.utils import get_tp_tn_fp_fn, get_accuracy, get_f1
from typing import TypeVar
DataFrame = TypeVar("pandas.core.frame.DataFrame")
from torchmetrics.classification import BinaryROC, AUROC
import torch
from matplotlib import pyplot as plt

def get_threshold_for_max_metric(df, y_true_col="outcome", y_pred_col="y_pred_mean", metric="accuracy"):
    """ """
    lower = df[y_pred_col].describe()['25%']
    upper = df[y_pred_col].describe()['75%']
    thresholds = np.linspace(lower, upper, num=20)
    res_values = []
    for threshold in thresholds:
        tp, tn, fp, fn = get_tp_tn_fp_fn(df, y_true_col=y_true_col, y_pred_col=y_pred_col, threshold=threshold, verbose=False)
        if metric == "accuracy":
            res = get_accuracy(tp, tn, fp, fn, verbose=False)
        elif metric == "f1_reverse":
            res = get_f1(tp, tn, fp, fn, pos_label=0, verbose=False)
        elif metric == "f1":
            res = get_f1(tp, tn, fp, fn, verbose=False)
        res_values.append(res)

    # Find the threshold with the highest NPR
    best_threshold = thresholds[np.argmax(res_values)]
    best_metric = np.max(res_values)

    print("Best Threshold:", best_threshold)
    print("Highest Metric:", best_metric)
    return best_threshold, best_metric


def plot_roc_auc(df, path, y_true_col, y_pred_col= "y_pred_mean", name="roc_auc", reverse=False, thresholds=None):
    broc = BinaryROC(thresholds=thresholds)
    y_pred = torch.tensor(df[y_pred_col].tolist())
    y_true = torch.tensor(df[y_true_col].tolist()).long()
    pos_label = 1
    if reverse:
        y_pred = abs(y_pred - 1)
        y_true = abs(y_true - 1)
        pos_label = 0

    auroc = AUROC(pos_label=pos_label, task="binary")
    auc = auroc(y_pred, y_true)
    logging.info(f"AUC: {auc:.3f}." )
    fpr, tpr, thresholds_roc = broc(y_pred, y_true)
    fpr = fpr.numpy()
    tpr = tpr.numpy()
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--")  # 45-degree diagonal for reference
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve, AUC: {:.3f}".format(auc))
    plt.legend()
    if thresholds:
        thresholds_roc = thresholds_roc.numpy()
        # Plot thresholds on the ROC curve
        for i, threshold in enumerate(thresholds_roc):
            plt.annotate(
                f"Threshold {threshold:.2f}", (fpr[i], tpr[i]), textcoords="offset points", xytext=(0, 10), ha="center"
            )
    # Save the plot as an image file (e.g., PNG)
    fig_path = f"{path}/{name}_thresholds{thresholds}_reverse_label{reverse}.png"
    plt.savefig(fig_path, dpi=300)
    logging.info(f"saved ROC plot to {fig_path}")
    plt.close()
