import logging
import pandas as pd
import numpy as np
import math

def zero_div(x, y):
    try:
        res = x / y
        return res
    except ZeroDivisionError:
        return 0

def proportion_CI(successes, total, confidence=0.95):
    # Proportion (e.g., sensitivity, specificity, etc.)
    if total == 0:
        return 0.0, 0.0  # Avoid division by zero, handle case with no trials
    proportion = zero_div(successes,total)

    # Z-score for the given confidence interval (default is 95%)
    z = 1.96  # for 95% confidence

    # Standard error of the proportion
    se = math.sqrt(zero_div((proportion * (1 - proportion)), total))

    # Confidence interval
    lower_bound = max(0, proportion - z * se)  # Ensure bounds are >= 0
    upper_bound = min(1, proportion + z * se)  # Ensure bounds are <= 1

    return lower_bound, upper_bound

def calculate_surprise_value(pvalue):
    """
    Calculate the surprise value (Shannon information) from a given p-value.

    Parameters:
    pvalue (float): The probability (p-value) (between 0 and 1).

    Returns:
    float: The surprise value corresponding to the p-value.
    """
    if pvalue <= 0 or pvalue > 1:
        raise ValueError("P-value must be between 0 (exclusive) and 1 (inclusive).")

    # Calculate the surprise value using Shannon's formula
    surprise_value = -np.log2(pvalue)

    return surprise_value


def get_tpr_fpr(tp, tn, fp, fn):
    TP, TN, FP, FN = len(tp), len(tn), len(fp), len(fn)
    tpr = TP/(TP+FN)
    fpr = FP/(FP+TN)
    tnr = TN/(TN+FP)
    fnr = FN/(FN+TP)
    logging.info(f"TPR: {tpr:.3f}, FPR: {fpr:.3f}, TNR: {tnr:.3f}, FNR: {fnr:.3f}")
    return tpr, fpr

def get_npv(tp, tn, fp, fn, return_CI=False):
    tn = len(tn)
    fn = len(fn)
    denominator = (tn+fn)
    npv = tn / denominator if  denominator > 0 else np.NaN
    logging.info(f"NPR: {npv:.3f}")
    if return_CI:
        CI = proportion_CI(tn, tn + fn)
        return npv, CI
    return npv

def get_accuracy(tp, tn, fp, fn, verbose=False):
    TP, TN, FP, FN = len(tp), len(tn), len(fp), len(fn)
    denominator = (TP + TN + FP + FN)
    accuracy = (TP + TN) / denominator if denominator > 0 else np.NaN
    if verbose:
        logging.info(f"Accuracy: {accuracy:.3f}")
    return accuracy

def get_f1(tp, tn, fp, fn, pos_label=1, verbose=False):
    TP, TN, FP, FN = len(tp), len(tn), len(fp), len(fn)
    if pos_label == 0:
        TP, FP, TN, FN = TN, FN, TP, FP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    denominator = precision + recall
    f1 = 2 * (precision * recall) / denominator if denominator > 0 else np.NaN
    if verbose:
        logging.info(f"F1 pos_label={pos_label}: {f1:.3f}")
    return f1

def get_auc(df, y_true_col="y_true", y_pred_col="y_pred_mean"):
    from sklearn.metrics import roc_auc_score
    y_pred = df[y_pred_col]
    y_true = df[y_true_col]
    if y_true.nunique() < 2:
        return np.NaN
    else:
        return roc_auc_score(y_true, y_pred)


def bootstrap_auc(df, y_true_col="y_true", y_pred_col="y_pred_mean", n_iterations=1000, ci=95, random_state=None):
    """
    Perform bootstrapping to calculate AUC and its confidence intervals.

    Parameters:
    - df: pandas DataFrame
        DataFrame containing true labels and predicted probabilities.
    - y_true_col: str, default="y_true"
        Column name for true binary labels (0 or 1).
    - y_pred_col: str, default="y_pred_mean"
        Column name for predicted probabilities for the positive class.
    - n_iterations: int, default=1000
        Number of bootstrap iterations.
    - ci: float, default=95
        Confidence interval percentage.
    - random_state: int or None, default=None
        Random seed for reproducibility.

    Returns:
    - mean_auc: float
        Mean AUC across bootstrap samples.
    - ci_lower: float
        Lower bound of the confidence interval.
    - ci_upper: float
        Upper bound of the confidence interval.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import resample
    # Extract columns from the DataFrame
    y_pred = df[y_pred_col].values
    y_true = df[y_true_col].values

    # Seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Check if there's enough class balance to compute AUC
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    auc_scores = []

    # Perform bootstrapping
    for _ in range(n_iterations):
        # Resample indices
        indices = resample(range(len(y_true)), replace=True)
        boot_y_true = y_true[indices]
        boot_y_pred = y_pred[indices]

        # Calculate AUC for the bootstrap sample
        auc_scores.append(roc_auc_score(boot_y_true, boot_y_pred))

    # Calculate mean AUC and confidence intervals
    mean_auc = np.mean(auc_scores)
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    ci_lower = np.percentile(auc_scores, lower_percentile)
    ci_upper = np.percentile(auc_scores, upper_percentile)

    return mean_auc, (ci_lower, ci_upper)


def get_auc_torch(df, y_true_col="y_true", y_pred_col="y_pred_mean"):
    from torchmetrics.classification import AUROC
    import torch
    y_pred = df[y_pred_col]
    y_true = df[y_true_col]
    y_pred = torch.tensor(y_pred.tolist())
    y_true = torch.tensor(y_true.tolist())
    auroc = AUROC(pos_label=1, task="binary")
    auc_res = auroc(y_pred, y_true)
    return auc_res

def get_confusion_matrix(df, y_true_col="y_true", y_pred_col="y_pred", verbose=False):

    df_confusion = pd.crosstab(
        df[y_true_col], df[y_pred_col], rownames=["Actual"], colnames=["Predicted"], margins=True
    )
    df_conf_perc = df_confusion / len(df)
    df_conf_norm = df_confusion.copy()
    df_conf_norm.iloc[0] = df_confusion.iloc[0] / df_confusion["All"][0]
    df_conf_norm.iloc[1] = df_confusion.iloc[1] / df_confusion["All"][1]
    df_conf_norm.iloc[2] = df_confusion.iloc[2] / df_confusion["All"]["All"]
    if verbose:
        print("Confusion matrix:")
        print(df_confusion)

        print("Confusion matrix in percentages:")
        print(df_conf_perc.round(2))
        print("Confusion matrix relative percentages for each class:")
        print(df_conf_norm.round(2))

    return df_confusion


def get_tp_tn_fp_fn(df, y_true_col="outcome", y_pred_col="y_pred_mean", threshold = None, threshold_col="binary_threshold", verbose=False):
    if threshold:
        y_pred = df.apply(lambda x: 1 if x[y_pred_col] >= threshold else 0, axis=1)
    else:
        y_pred = df.apply(lambda x: 1 if x[y_pred_col] >= x[threshold_col] else 0, axis=1)
    tp = df[(df[y_true_col] == 1) & (y_pred == 1)]
    tn = df[(df[y_true_col] == 0) & (y_pred == 0)]
    fp = df[(df[y_true_col] == 0) & (y_pred == 1)]
    fn = df[(df[y_true_col] == 1) & (y_pred == 0)]
    if verbose:
        logging.info(f"TP: {len(tp)}, TN: {len(tn)}, FP: {len(fp)}, FN: {len(fn)}")
        get_confusion_matrix(df, y_true_col=y_true_col, y_pred_col=y_pred_col, verbose=verbose)

    return tp, tn, fp, fn

def get_sensitivity(tp, fn, verbose=False, return_CI=False):
    tp = len(tp)
    fn = len(fn)
    sensitivity = tp/(tp+fn)
    if verbose:
        logging.info(f"Sensitivity: {sensitivity:.3f}")
    if return_CI:
        CI = proportion_CI(tp, tp + fn)
        return sensitivity, CI
    return sensitivity

def get_specificity(tn, fp, verbose=False, return_CI=False):
    tn = len(tn)
    fp = len(fp)
    specificity = tn/(tn+fp)
    if verbose:
        logging.info(f"Specificity: {specificity:.3f}")
    if return_CI:
        CI = proportion_CI(tn, tn + fp)
        return specificity, CI
    return specificity