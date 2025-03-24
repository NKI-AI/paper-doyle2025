from lifelines.statistics import proportional_hazard_test
from sklearn.base import BaseEstimator, TransformerMixin
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
import drop.data_analysis_new.predictions.utils.data_analysis_utils as da

def test_proportional_hazards(cph_model, df, time_transform='rank'):
    """
    Performs the proportional hazards assumption test and prints the summary.

    Parameters:
    - cph_model: Fitted CoxPHFitter model from the lifelines package.
    - df: DataFrame containing the survival data used to fit the model.
    - time_transform: The type of time transformation for the test (default is 'rank').

    Returns:
    - results: The results of the proportional hazards test.
    """
    # Perform the proportional hazards test
    results = proportional_hazard_test(cph_model, df, time_transform=time_transform)
    # Print the test summary
    results.print_summary()
    return results


def plot_martingale_residuals(cph_model, df, covariate):
    """
    Plots the martingale residuals against a specified covariate.

    Parameters:
    - cph_model: Fitted CoxPHFitter model from the lifelines package.
    - df: DataFrame containing the covariate and survival data.
    - covariate: The column name of the covariate to plot against the residuals.
    """
    # Compute martingale residuals
    residuals = cph_model.compute_residuals(kind='martingale', training_dataframe=df)
    import matplotlib.pyplot as plt  # Importing the required module

    # Plot residuals against the covariate
    plt.scatter(df[covariate], residuals['martingale'])
    plt.xlabel(covariate)
    plt.ylabel('Martingale Residuals')
    plt.title(f'Martingale Residuals vs {covariate}')
    plt.show()
    plt.savefig(f"/home/s.doyle/tmp/martingale_plot_drop_{covariate}.png")
    plt.close()


def get_threshold_using_method(df, method):
    if method == "median":
        return np.median(df["pred_score"])
    elif method == "mean":
        return np.mean(df["pred_score"])
    elif method in ["accuracy", "f1", "f1_reverse"]:
        threshold, _ = da.get_threshold_for_max_metric(df, y_true_col="outcome", y_pred_col="pred_score",
                                                       metric=method)
    elif method == "accuracy_quantiles":
        da.get_best_quantile_thresholds(df, y_true_col="outcome", y_pred_col="pred_score",
                                                       metric="accuracy")
    return threshold

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attributes_names):
        self.attributes_names = attributes_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attributes_names].values

def preprocess(X, cols):
    from sklearn.pipeline import Pipeline
    num_pipeline = Pipeline([
        ('select_data', DataFrameSelector(cols)),
        # ('Std_Scaler', StandardScaler())
    ])
    X_transformed = num_pipeline.fit_transform(X)
    return X_transformed


def get_coxscores(cph, survival_pd):
    """" Predict partial hazard scores / cumulative hazard scores from a cph model and a survival dataframe.  """
    # leads to the dame results as using the last time point of the cumulative hazard
    result = {"partial_hazard": cph.predict_partial_hazard(survival_pd),
                "pred_score": cph.predict_cumulative_hazard(survival_pd).iloc[-1]} # cumulative hazard scores for full time range
    for years in [5,8,10,20]:
        month_index = (np.abs(cph.baseline_survival_.index - years*12)).argmin()
        pred_scores = cph.predict_cumulative_hazard(survival_pd).iloc[month_index]
        result[f"pred_score_limit{years}"] = pred_scores
    return pd.DataFrame(result)

def cindex_from_coxscores(cph, survival_pd):
    """
    Concordance index is computed based on the partial hazard scores. The method is equivalent to:
    cph.score(survival_pd, scoring_method="concordance_index")
    """
    cox_scores = cph.predict_partial_hazard(survival_pd)
    c_index_coxph = concordance_index(survival_pd["time_to_event"], -cox_scores, survival_pd["outcome"])
    return c_index_coxph

def calculate_auc_coxph(survival_pd_val):
    auc_coxph = roc_auc_score(survival_pd_val["outcome"], survival_pd_val["pred_score"])
    return auc_coxph