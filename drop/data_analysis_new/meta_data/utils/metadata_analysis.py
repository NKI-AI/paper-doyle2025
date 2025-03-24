import pandas as pd
from typing import List
import numpy as np

def format_sig_digits(number, sig_digits):
    return np.format_float_positional(number, precision=sig_digits, unique=False, fractional=False, trim='k')

def get_iqr(series):
    df = series.describe()
    # iqr = df.loc[df["index"] == '75%'][var].item() - df.loc[df["index"] == '25%'][var].item()
    iqr = df['75%'] - df['25%']
    return iqr

def safe_round(series, decimals=2):
    # Check if there is a series or if it is a np.NAN
    if type(series) == pd.Series:
        series = series.round(decimals)
    return series

def check_dfs(dfs: List, n: int, group):
    '''
    Check if the sum of values in a specific column (or group of columns) of each DataFrame in a list of DataFrames equals a given number n.
    '''
    for df in dfs:
        try:
            print(df.sum()[group])
            assert df.sum()[group] == n
        except:
            pass

def get_median_observation_time(df):
    obs_median = df['months_followup'].median()
    obs_iqr = get_iqr(df['months_followup'])
    return  obs_median, obs_iqr

def get_deceased_stats(df, group: str):
    var = 'vital_status'
    # Count the number of occurrences of each value in the 'var' column (e.g., 1 for alive, 0 for dead)
    df_stats = df[var].value_counts().reset_index()

    # Sort the values for consistent ordering
    df_stats = df_stats.sort_values(by='index').reset_index(drop=True)

    # Apply a lambda function to label the counts
    df_stats['index'] = df_stats['index'].apply(lambda x:
                                                'Vital' if x == 1 else
                                                'Deceased' if x == 0 else
                                                'Vital Status Missing'
                                                )

    # Rename the columns: 'var' for status and 'group' for counts
    df_stats.columns = ['var', group]


    return df_stats

def get_deceased_at_risk(df, group):
    # we need to check vital status and
    var = 'vital_status_at_risk'
    df[var] = df.apply(
        lambda row: row['vital_status'] if row["outcome"] == 0 else 0, axis=1)

    input = [("Deceased (No event)", df[var].sum())
             ]
    return pd.DataFrame(input, columns=["var", group])


def get_follow_up_stats(df, group: str):
    var = "time_to_event"
    median_fu = df[var].median().round(2)
    iqr = get_iqr(df[var]).round(2)
    input = [("Total", len(df)),
             ("Median Time-to-event/no-event", median_fu),
             ("IQR Time-to-event/no-event", iqr)]

    # If we are computing this separated by low and high risk group then some of the series wil be empty.
    # It also is not necessary to give the time to event and time to no event separately, as this is already done throught the group
    time_no_event = df[df['outcome'].isin(['no event', 0])][var]
    time_event = df[df['outcome'].isin(['iIBC', 1])][var]
    if len(time_event) > 0 and len(time_no_event) > 0:
        median_fu_no_event = safe_round(time_no_event.median())
        median_fu_event = safe_round(time_event.median())
        iqr_time_no_event = safe_round(get_iqr(time_no_event))
        iqr_time_event = safe_round(get_iqr(time_event))
        input += [
             ("Median Time-to-event", median_fu_event),
             ("IQR Time-to-event", iqr_time_event),
             ("Median Time-to-no-event", median_fu_no_event),
             ("IQR Time-to-no-event", iqr_time_no_event),
             ]
    return pd.DataFrame(input, columns=["var", group])

def get_outcome_stats(df, group: str):
    try:
        var = 'outcome'
        df = df[var].value_counts().reset_index()

    except:
        var = 'first_subseq_event'
        df = df[var].value_counts().reset_index()
    df.columns = ['var', group]
    df['var'] = df['var'].apply(lambda x: f"{var.upper()} {x if not x == 999 else 'Missing'}")
    df['var'] = df['var'].apply(lambda x: x.lower())
    return df

def get_age_stats(df, group: str):

    df = df["age_diagnose"].describe().reset_index()
    age_cols = ["mean", "std"]
    df = df[df["index"].isin(age_cols)]
    for var in age_cols:
        df.loc[df["index"]==var, "index"] = "Age Patient " + var
    df.columns = ['var', group]
    df = df.round(2).astype(str)

    return df

def get_age_slide_stats(df, group: str):
    # age dist for low risk
    df = df["age_slide"].describe().reset_index()
    # only keep rows with name called mean or std
    age_cols = ["mean", "std"]
    df = df[df["index"].isin(age_cols)]
    for var in age_cols:
        df.loc[df["index"]==var, "index"] = "Age Slide " + var
    df.columns = ['var', group]
    df = df.round(2).astype(str)

    return df

def get_grade_stats(df, var, group: str):
    df = df[var].value_counts().reset_index().astype(int)
    df = df.sort_values(by='index').reset_index(drop=True)
    df.columns = ['var', group]
    df['var'] = df['var'].apply(lambda x: f"{var.upper()} {x if not int(x)==999 else 'Missing'}")
    return df

def get_her2_stats(df, group: str):
    df = df["her2"].value_counts().reset_index().astype(int)
    df = df.sort_values(by='index').reset_index(drop=True)
    df.columns = ['var', group]
    # rename var column with Positive for 1 and Negative for 0, and Borderline for 2
    df['var'] = df['var'].apply(lambda x: "Positive" if x==1 else "Negative" if x==0 else "Borderline" if x==2 else "Missing")
    df['var'] = df['var'].apply(lambda x: "HER2 " + str(x))

    return df

def get_rec_stats(df, var, group: str):
    df = df[var].value_counts().reset_index().astype(int)
    df = df.sort_values(by='index').reset_index(drop=True)
    df.columns = ['var', group]
    # rename var column with Positive for 1 and Negative for 0, and Borderline for 2
    df['var'] = df['var'].apply(lambda x: "Positive" if x==1 else "Negative" if x==0 else "Borderline" if x==2 else "Missing")
    df['var'] = df['var'].apply(lambda x: f"{var.upper()} " + str(x))
    return df

def get_p53_stats(df, group: str):
    # p53 is a percentage
    try:
        df = df["p53"].value_counts().reset_index().astype(int)
    except:
        df = df["p53_percentage"].value_counts().reset_index().astype(int)
        print('using percentage for p53')
    df = df.sort_values(by='index').reset_index(drop=True)

    df.columns = ['var', group]
    # all values which are 0 or 100 are considered abnormal
    df['var'] = df['var'].apply(lambda x: "Abnormal" if x==0 or x==100 else "Missing" if x==999 else "Normal")
    df['var'] = df['var'].apply(lambda x: "P53 " + str(x))
    df = df.groupby("var").sum().reset_index()
    return df