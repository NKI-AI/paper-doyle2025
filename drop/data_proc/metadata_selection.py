from typing import Generator, List, Optional, Tuple, Dict, TypeVar, Union
import logging
import numpy as np
import pandas as pd
DataFrame = TypeVar("pandas.core.frame.DataFrame")
from pyats.datastructures import AttrDict
import math
from drop.data_proc.data_utils import transform_categorical_columns


def do_column_mapping_precision_nki(df):
    """ " Replace Nan with 0 or unknown (999), and encode categorical variables as numbers"""
    # code hidden


def add_followup_cols(df):
    """ Add follow-up columns if needed"""
    # code hidden

def make_outcome_years(df, years, outcome_in_cutoff_col="outcome_in_cutoff"):
    """
     If there is a cutoff for time to event:
    - we set the time to event to the cutoff if the followup time is longer, otherwise we set it to event_months.
    - We set recurrrences that happen after the followup time to no-event.
    If there is no cutoff:
    - we set the time to event to the months_followup, if there is no event, else we set it to event_months
    - we set the outcome_in_cutoff to 0 if there is no event in event_months, else we set it to 1. This would be
    the same as first_subseq_event.

     """
    df["event_months"] = df["event_months"].replace("", 0.0)
    df['event_months'] = df['event_months'].fillna(0.0).astype(float)
    if years:
        month_cutoff = years * 12
        df['time_to_event'] = df.apply(
            lambda x: x['months_followup'] if x['event_months'] == 0
            else x['event_months'] if 0 < x['event_months'] <= month_cutoff
            else month_cutoff, axis=1
        ).astype(float)
        # for those without recurrence(events_months=0) we set the time to event to the months_cutoff, if their followup time is longer than the cutoff.
        df['time_to_event'] = df['time_to_event'].apply(lambda x: month_cutoff if x > month_cutoff else x)
        df = df.assign(
            **{outcome_in_cutoff_col: df["event_months"].apply(lambda x: 1 if (0 < x <= month_cutoff) else 0)}
        )

    else:
        df = df.assign(time_to_event=df.apply(lambda x: x["event_months"] if (0 < x["event_months"]) else x["months_followup"], axis=1).astype(float))
        # Assign the new column based on whether 'event_months' occurred within the cutoff
        df = df.assign(
            **{outcome_in_cutoff_col: df["event_months"].apply(
                lambda x: 1.0 if x != 0.0 else 0.0)}
        )

    return df


def prepare_precision_nki_data(
    precision_data: DataFrame,
    target: Union[str, List[str]],
    cutoff_years: Union[int, bool] = False
) -> DataFrame:
    """
    First selects relevant samples based on target variable, then excludes samples based on exclusion criteria.
    Vital status 0 = alive, 1=dead. Changing that around, because it is not intuitive.
    """
    df = do_column_mapping_precision_nki(precision_data)
    df = add_followup_cols(df)
    df["vital_status"] = df["vital_status"].apply(lambda x: 0 if x == 1 else 1)
    if target in ["outcome", 'first_subseq_event']:
        df = make_outcome_years(df, cutoff_years)
    return df

def add_followup_cols_sloane(df):
    """
    We set the time to recurence (event_months) to age-recurrence minus age-diagnosis if there is a recurrence (and age recurrence is known),
    otherwise we set it to "" .
    """
    df = df.assign(months_followup=df["fu_months"])
    age = df["age_recurrence"].copy()
    df['age_recurrence'].replace("", 999, inplace=True)
    df['age_recurrence'] = df['age_recurrence'].astype(float)
    df['event_months'] = df.apply(lambda row: (row['age_recurrence'] - row['age_diagnose']) * 12 if row['age_recurrence'] != 999.0 else "", axis=1)
    # set age_recurrence back to ""
    df['age_recurrence'] = age
    return df



def prepare_sloane_data(sloane_meta_data, target, cutoff_years,
                        exclude_scottish: Optional[bool] = False,
        exclude_endocrine_therapy: Optional[bool] = True):

    """Read in Sloane labels file and filters for:
    - Endocrine therapy
    - Age (Scottish or not)
    All cases are breast conserverving surgery, not mastectomy.
    Encodes no recurrence as 0 and ipsilateral recurrence as 1.
    """
    sloane_meta_data["split"] = "test"
    if exclude_scottish == True:
        sloane_meta_data = sloane_meta_data[sloane_meta_data["age_diagnose"] != 999]
    if exclude_endocrine_therapy == True:
        sloane_meta_data = sloane_meta_data[sloane_meta_data["endocrinetherapy"] != 1]
    # equivalent to column mapping in precision
    sloane_meta_data["first_subseq_event"] = sloane_meta_data["first_subseq_event"].replace("", 0)
    sloane_meta_data["first_subseq_event"] = sloane_meta_data["first_subseq_event"].replace("ipsilateral IBC", 1)
    df = add_followup_cols_sloane(sloane_meta_data)
    if target == "outcome":
        # drop rows that do not have event_months but have a recurrence
        df = df.drop(df[(df["event_months"] == "") & (df["first_subseq_event"] != 0)].index)
        df = make_outcome_years(df, cutoff_years)

    return df

def prepare_data(df: DataFrame, dataset_name: str, criteria: Dict):
    """
    Returns a Dataframe where values of  relevant columns have been encoded as integers.
    Also outcome has been updated depending on followup time. Could also be updated based on which events to classify
    as an event, atm only iIBC."""

    criteria = AttrDict(criteria)
    if dataset_name == "Sloane":
        sel_data_df = prepare_sloane_data(df,
            target=criteria.target,
             cutoff_years=criteria.cutoff_years,
             exclude_scottish=criteria.exclude_scottish,
            exclude_endocrine_therapy=criteria.exclude_endocrine_therapy,
        )
    elif dataset_name == "Precision_NKI_89_05":
        sel_data_df = prepare_precision_nki_data(df,
            target=criteria.target,
            cutoff_years=criteria.cutoff_years,
        )
    else:
        raise ValueError

    return sel_data_df


def select_target_data(df, target):
    if target in ['first_subseq_event', "outcome"] :
        col = "outcome"
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # force into a here the column is renamed to outcome
        df.loc[:, "outcome"] = df[col]
        target_df = df[df["outcome"].isin([0, 1])]

    elif target == "grade":
        target_df = df.loc[~df["grade"].isin([999])]
        # for biomarker project
        target_df.loc[target_df["grade"] == 3, "grade"] = 1
        target_df.loc[target_df["grade"].isin([1, 2]), "grade"] = 0
    elif target == "her2":
        target_df = df.loc[~df["her2"].isin([ 999])]
        target_df.loc[target_df["her2"].isin([0, 1, 2]), "her2"] = 0
        target_df.loc[target_df["her2"] == 3, "her2"] = 1
    elif target == "er":
        target_df = df.loc[~df["er"].isin([999])]
    elif target == "pr":
        target_df = df.loc[~df["er"].isin([999])]
    elif target == "p16":
        target_df = df.loc[df["p16"].isin([0, 1])]
        target_df["p16"] = target_df["p16"].astype(float)
    elif target == "cox2":
        target_df = df.loc[df["cox2"].isin([0, 1, 2])]
        target_df.loc[target_df["cox2"] == 1, "cox2"] = 0
        target_df.loc[target_df["cox2"].isin([2, 3]), "cox2"] = 1
    elif target== "":
        target_df = df
    return target_df

def select_data(
        df: DataFrame,
        target: Union[str, List[str]],
        cutoff_years: Optional[int] = False,
        exclude_radiotherapy: bool = True,
        exclude_deceased: bool = False,
        exclude_philips: bool = False,
        drop_nas_in_cols: Optional[List[str]] = None,

) -> DataFrame:
    """ Columns have already been renamed to target name. """
    if isinstance(target, List):
        target_df = df
        for t in target:
            target_df = select_target_data(target_df, t)
    else:
        target_df = select_target_data(df, target)
    if target in ['first_subseq_event', "outcome"] and cutoff_years:
        target_df["outcome"] = target_df["outcome_in_cutoff"]
    if exclude_deceased:
        # vital status 1 means alive, 0 means dead (changed from original)
        target_df = target_df.loc[target_df["vital_status"] == 1]
    if exclude_radiotherapy:
        target_df = target_df.loc[target_df["radiotherapy"] == 0]
    if drop_nas_in_cols:
        # drop her2 ==2 for integrative paper
        target_df = df.loc[~df["her2"].isin([2, 999])]  # 10+ 29 = 39 cases for Sloane  # this is for integrative model
        target_df = target_df[target_df['her2'] != 2]
        # for integrative project - when using binary
        target_df.loc[target_df["grade"] == 1, "grade"] = 0
        target_df.loc[target_df["grade"].isin([2, 3]), "grade"] = 1
        target_df["age_diagnose"] = target_df['age_diagnose']  # /100
        for col in drop_nas_in_cols:
            target_df[col] = target_df[col].astype(float)
            print(col, len(target_df.loc[target_df[col] == 999]), "no missing values")
            target_df = target_df.loc[target_df[col] != 999]

    return target_df

