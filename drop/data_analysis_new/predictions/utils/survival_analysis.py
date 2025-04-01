import numpy as np
import pandas as pd
import logging
from pathlib import Path
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import itertools
from .custom_at_risk_group import add_at_risk_counts_custom
from .formatting_utils import format_pvalue
from matplotlib import pyplot as plt


def preprocess_survival_columns(survival_pd, y_true_col="outcome", keep_cols=None):
    survival_pd[[y_true_col]] = survival_pd[[y_true_col]].astype(float)
    if keep_cols:
        survival_pd = survival_pd[keep_cols + ["time_to_event", y_true_col]]
    else:
        survival_pd = survival_pd[["time_to_event", y_true_col]]
    return survival_pd

def get_risk_groups(data, threshold, risk_col):
    data = data.assign(group =data[risk_col].apply(lambda x: 'low' if x < threshold else 'high'))
    if "pred_score" not in data.columns:
        data = data.assign(pred_score=lambda d: d[risk_col])
        data = data.drop(columns=[risk_col])
    return data

def perform_statistical_test(group_outcomes_list, normality, var_equality):
    if normality and var_equality:
        logging.info("use t test")
        statistic, p_value = stats.ttest_ind(*group_outcomes_list, equal_var=True)
    if not var_equality and normality:
        logging.info("use welch t test")
        statistic, p_value = stats.ttest_ind(*group_outcomes_list, equal_var=False)
    if not normality or not var_equality:
        logging.info("don tuse t test, using Mann Whitney U test instead")
        statistic, p_value = stats.mannwhitneyu(*group_outcomes_list)
    return p_value

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

def get_survivorship(df, y_true_col="outcome"):
    survivorship = {}
    normality = True
    var_equality = True
    group_outcomes_list = []
    #works only for 2 risk groups, not doing it dynamically df["group"].unique() in case one group is not present
    for group_name in ["low", "high"]:
        group = df.loc[df["group"] == group_name]
        outcomes_group = group[y_true_col]
        survivorship[f"rec_rate_{group_name}"] = outcomes_group.sum() / len(outcomes_group)
        correct_class = 0.0 if group_name == 'low' else 1.0
        survivorship[f"misclassified_{group_name}"] = (outcomes_group != correct_class).sum()
        # insert censorship here
        survivorship[f"censored_{group_name}"] = (outcomes_group == 0).sum() /  len(outcomes_group)
        survivorship[f"predicted_{group_name}"] =  len(outcomes_group)
        if outcomes_group.nunique() >1:
            group_outcomes_list.append(outcomes_group.to_list())
            if stats.shapiro(outcomes_group.to_list()).pvalue < 0.05:
                normality = False
    if len(group_outcomes_list) == 2:
        _, p_value_levene = stats.levene(*group_outcomes_list)
        if p_value_levene > 0.05:
            var_equality = False
        survivorship[f"rec_rate_p_value"] = perform_statistical_test(group_outcomes_list, normality, var_equality)
    else:
        survivorship[f"rec_rate_p_value"] = np.nan

    return survivorship

def get_run_name(model_name, clinical_vars_type, limit_years, dataset, stage, outer_split=None, fold=None):
    var_name = f"({model_name}) - {dataset} {clinical_vars_type} variables - Limit years {limit_years} - {stage}"
    if type(outer_split) == int:
        var_name = f"{var_name} Split {outer_split} "
    if type(fold) == int:
        var_name = f"{var_name} fold {fold} "
    return var_name


def make_kaplan_meier_plot(data, var_name, output_dir, y_true_col="outcome", y_pred_col="pred_score",
                           high_risk_hr=None, log_rank_pvalue=None):
    """
    Plot Kaplan-Meier survival curves for each risk group based on the Cox PH model.
    """
    risk_groups = ["low", "high"] # to set the order correctly
    data = data.drop(columns=[y_pred_col])
    vital_status = True if "vital_status" in data.columns else False

    fig = plt.figure(figsize=(10, 8)) # use (10, 8) with risk counts and (10, 6) wihout
    ax = plt.subplot(111)
    fitters = []
    for group in risk_groups:  # predicted risk groups
        group_data = data[data['group'] == group]
        # predicted_survival = cph_vars.predict_survival_function(group_data)
        # plt.plot(predicted_survival.index, predicted_survival.values, label=group)

        if vital_status:

            # in old matched_data, vital status has not yet changed the signs = 1 means dead
            group_data['vital_status_at_risk'] = group_data.apply(lambda row: row['vital_status'] if row[y_true_col] == 0 else 0, axis=1)
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(group_data['time_to_event'], event_observed=group_data[y_true_col], label=group)
            kmf.plot(ax=ax, color=None)  # Let the color be automatically assigned
            # Set the x-axis limit to a maximum of 240
            # Set custom ticks (for example, every 50 units)
            max_time_to_event = group_data['time_to_event'].max()
            if max_time_to_event <= 60:
                x_lim = 60
                custom_ticks = [i for i in range(0, x_lim+1, 10)]
            elif max_time_to_event <= 96: #8 years for Sloane
                x_lim = 96
                custom_ticks = [i for i in range(0, x_lim + 5, 20)]
            elif max_time_to_event <= 120:
                x_lim = 120
                custom_ticks = [ i for i in range(0, x_lim+1, 20)]
            else:
                x_lim = 240
                custom_ticks = [i for i in range(0, x_lim+1, 40)]

            ax.set_xticks(custom_ticks)
            ax.set_xlim([0, x_lim])
            ax.set_ylim([0.0, 1.0])

            if vital_status:
                kmf_deceased =  KaplanMeierFitter()
                kmf_deceased.fit(group_data['time_to_event'], event_observed=group_data['vital_status'],
                                 label=f"{group} Deceased")

                kmf_deceased_at_risk = KaplanMeierFitter()
                kmf_deceased_at_risk.fit(group_data['time_to_event'], event_observed=group_data['vital_status_at_risk'],
                                 label=f"{group} Deceased")

                fitters.append((kmf,(kmf_deceased, kmf_deceased_at_risk)))
            else:
                fitters.append(kmf)
        except:
            logging.info('no data in group')
            pass

    if vital_status:
        add_at_risk_counts_custom(fitters, labels=risk_groups, xticks=custom_ticks)
    else:
        add_at_risk_counts(fitters[0], fitters[1], color='black', labels=risk_groups)  #rows_to_show=['At risk', 'Events']

    plt.tight_layout()
    plt.xlabel("Months")
    plt.ylabel("Survival Probability")

    if high_risk_hr and log_rank_pvalue:
        p_value = format_pvalue(log_rank_pvalue, for_plot=True)
        hr = high_risk_hr[0]
        ci_low, ci_high = high_risk_hr[1], high_risk_hr[2]

        # Add custom text to the existing plot
        ax.text(0.05, 0.15,
                f'Low-risk: Reference\nHigh-risk: HR: {hr:.2f} (95% CI {ci_low:.2f}-{ci_high:.2f})\nLog-rank test p{p_value}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # Adjust the layout to make sure nothing is cut off
    plt.tight_layout()
    ax.set_ylabel("Recurrence-Free Probability")
    fig.savefig(Path(output_dir) / f"survival_curves{var_name}.png", pad_inches=0.1, bbox_inches='tight')

    plt.close()
    logging.info(f"Saved survival curves to {Path(output_dir) / f'survival_curves{var_name}.png'}")

def log_rank_test(data, y_true_col="outcome"):
    """ Perform log-rank test for the survival curves of each pair of groups. """
    # Create a list of group names
    group_names = data['group'].unique()
    # Initialize an empty dictionary to store test results
    test_results = {}
    # Perform log-rank test for each pair of groups
    combinations = itertools.combinations(group_names, 2)
    for group1, group2 in combinations:
        subset_data1 = data[data['group'] == group1]
        subset_data2 = data[data['group'] == group2]

        # Perform log-rank test
        results = logrank_test(durations_A=subset_data1['time_to_event'],
                               durations_B=subset_data2['time_to_event'],
                               event_observed_A=subset_data1[y_true_col],
                               event_observed_B=subset_data2[y_true_col])
        # Store the results
        test_results[f"{group1}_vs_{group2}"] = results
    # Print the results
    verbose = False
    if verbose:
        for pair_name, result in test_results.items():
            print(f"Test for {pair_name}:")
            print(result.summary)
    if len(test_results) == 1:
        return test_results[f"{group1}_vs_{group2}"].p_value
    if test_results == {}:
        return 1.0
    else:
        return test_results

def calculate_cindex_risk_score(survival_pd, risk_col, y_true_col="outcome"):
    """ For a model that already has the risk score computed, compute the c-index. """
    y_pred = survival_pd[risk_col].to_numpy()
    c_index = concordance_index(survival_pd["time_to_event"], -y_pred, survival_pd[y_true_col])
    return c_index

def calculate_hazards_ratio(data, y_true_col="outcome"):
    """ Calculate the hazards ratio for the risk group.
      hazard ratio quantifies the relative risk of experiencing an event (such as death or failure)
       between the groups over time. It provides a direct measure of how the hazard rates (instantaneous event rates)
        in the two groups compare."""
    cph = CoxPHFitter()
    data['group'] = data['group'].astype('category')
    data_low_as_ref = data.copy()
    try:
        data_low_as_ref['group'] = data_low_as_ref['group'].cat.reorder_categories(['low', 'high'])
        cph.fit(data_low_as_ref, duration_col='time_to_event', event_col=y_true_col, formula='group', show_progress=False)
        # Extract the hazard ratios (odds ratios) for Group B compared to Group A
        hazards_ratio = cph.hazard_ratios_[0]
        p_value = cph.summary["p"][0]
        # 95% Ci intervals
        lower= np.exp(cph.confidence_intervals_['95% lower-bound'].item())
        upper = np.exp(cph.confidence_intervals_['95% upper-bound'].item())
        return hazards_ratio, p_value, lower, upper
    except ValueError:
        return np.NaN, np.NaN, np.NaN, np.NaN




