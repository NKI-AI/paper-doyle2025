import numpy as np
import pandas as pd
from scipy.stats import norm

def get_variance(hr_lower, hr_upper):
    # Calculate standard error (SE)
    z_value = 1.96  # For 95% CI
    se = (np.log(hr_upper) - np.log(hr_lower)) / (2 * z_value)

    # Calculate variance
    variance = se ** 2
    return variance

def calculate_se_log_hr(ci_lower, ci_upper, confidence_level=0.95):
    """
    Calculate the standard error of the log hazard ratio from its confidence interval.

    Parameters:
    ci_lower (float): Lower bound of the confidence interval for the hazard ratio.
    ci_upper (float): Upper bound of the confidence interval for the hazard ratio.
    confidence_level (float): The confidence level for the CI (default is 95%).

    Returns:
    float: Standard error of the log hazard ratio.
    """
    # Z value for the desired confidence level (default is 1.96 for 95% CI)
    z_value = 1.96 if confidence_level == 0.95 else norm.ppf(1 - (1 - confidence_level) / 2)

    # Log-transform the CI bounds
    log_lower = np.log(ci_lower)
    log_upper = np.log(ci_upper)

    # Calculate the standard error of the log HR
    se_log_hr = (log_upper - log_lower) / (2 * z_value)

    return se_log_hr


def random_effects_meta_analysis(values, se_log_hr):
    """
    Perform random-effects meta-analysis to combine hazard ratios from multiple splits using combine_effects.

    Parameters:
    log_hr (array-like): Log-transformed hazard ratios for each split.
    se_log_hr (array-like): Standard errors of the log-transformed hazard ratios.

    Returns:
    dict: Pooled hazard ratio (HR), its standard error (SE), and 95% confidence interval (CI).
    """
    # Perform the random-effects meta-analysis using combine_effects
    from statsmodels.stats.meta_analysis import combine_effects

    log_hr = np.log(values)  # Log-transformed hazard ratios
    se_log_hr = np.array(se_log_hr)
    result = combine_effects(log_hr, se_log_hr, method_re="iterated", use_t=False)  # method=random
    pooled_log_hr = result.summary_frame()['eff']['random effect']
    pooled_log_se = result.summary_frame()['sd_eff']['random effect']

    # Calculate the z-statistic for the pooled log HR
    z_stat = pooled_log_hr / pooled_log_se

    # Calculate the p-value using the z-statistic
    pvalue = 2 * (1 - norm.cdf(np.abs(z_stat)))  # Two-tailed test
    res_exp = np.exp(result.summary_frame())

    pooled_hr = res_exp['eff']['random effect']
    ci_lower = res_exp['ci_low']['random effect']
    ci_upper = res_exp['ci_upp']['random effect']

    return pooled_hr, (ci_lower, ci_upper ), pvalue



def fishers_combined_test(p_values):
    from scipy.stats import chi2

    p_values = [max(p_value, 0.000001) for p_value in p_values]  # todo fix so that more exact p-value is saved.

    # Fisher's combined test statistic
    chi2_stat = -2 * np.sum(np.log(p_values))

    # Degrees of freedom: 2 times the number of splits (because each test has 2 degrees of freedom)
    df = 2 * len(p_values)

    # p-value from the chi-squared distribution
    p_value_combined = chi2.sf(chi2_stat, df)
    return p_value_combined



def apply_multipletests_correction(p_values, method="hommel"):
    """
    Apply multiple testing correction while preserving the original size and positions of NaN values.

    Parameters:
    - p_values: list or array of p-values (can include NaN values).
    - method: correction method to use (default is "hommel").

    Returns:
    - Array of corrected p-values, with NaN values in their original positions.
    """
    from statsmodels.stats.multitest import multipletests

    p_values = np.array(p_values)  # Ensure input is a NumPy array

    # Identify valid (non-NaN) indices
    valid_indices = ~np.isnan(p_values)

    # Extract valid p-values for correction
    valid_p_values = p_values[valid_indices]

    # Apply multiple testing correction
    _, corrected_pvals, _, _ = multipletests(valid_p_values, alpha=0.05, method=method)

    # Initialize the output array with NaN
    corrected_pvals_full = np.full_like(p_values, np.nan, dtype=float)

    # Insert the corrected p-values into their original positions
    corrected_pvals_full[valid_indices] = corrected_pvals

    return corrected_pvals_full


def combine_hazard_ratios_asymmetric_invw(hr, ci_lower, ci_upper):
    """
    Combine hazard ratios (HRs) and their asymmetric confidence intervals (CIs)
    across independent splits using inverse-variance weighting, and compute the p-value.

    Parameters:
        hr (array-like): Hazard ratios from each split.
        ci_lower (array-like): Lower bounds of the 95% confidence intervals.
        ci_upper (array-like): Upper bounds of the 95% confidence intervals.

    Returns:
        tuple: Combined hazard ratio, lower CI, upper CI, p-value
    """

    # Convert to log space
    log_hr = np.log(hr)
    se_upper = (np.log(ci_upper) - log_hr) / 1.96
    se_lower = (log_hr - np.log(ci_lower)) / 1.96

    # Average the standard errors to handle asymmetry
    se = (se_upper + se_lower) / 2

    # Compute combined log-HR and SE using inverse-variance weighting
    weights = 1 / se**2
    log_hr_combined = np.sum(log_hr * weights) / np.sum(weights)
    se_combined = np.sqrt(1 / np.sum(weights))

    # Transform back to HR and compute combined CIs
    hr_combined = np.exp(log_hr_combined)
    ci_lower_combined = np.exp(log_hr_combined - 1.96 * se_combined)
    ci_upper_combined = np.exp(log_hr_combined + 1.96 * se_combined)

    # Compute the z-score and p-value
    z_score = log_hr_combined / se_combined
    p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test

    return hr_combined, (ci_lower_combined, ci_upper_combined), p_value
