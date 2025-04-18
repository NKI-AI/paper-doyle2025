
def divide_by_1000(df, cols):
    for col in cols:
        mask = df[col].notna()
        df.loc[mask, col] = (df.loc[mask, col] / 1000)
    return df

def format_pvalue(p_value, for_plot=False, for_table=True):
    """
    Format a p-value according to specified rules:
    - Display to two significant figures unless p < 0.0001.

    Parameters:
    p_value (float): The p-value to format.

    Returns:
    str: The formatted p-value as a string.
    """

    if p_value < 0.0001:
        if for_table:
            return f"<0.0001"  # For very small p-values
        else:
            return f"{p_value:.2g}"
    else:
        if for_plot:
            return f"={p_value:.2g}"
        # Format to two significant figures
        return f"{p_value:.2g}"

def format_colums(df, p_value_cols, new_p_value_cols):
    # round values unless they are NA values
    df = divide_by_1000(df, p_value_cols)
    # Create a mask for columns to round
    columns_to_round = [col for col in df.columns if col not in p_value_cols]
    df[columns_to_round] = df[columns_to_round].round(2)
    for i, col in enumerate(p_value_cols):
        df.loc[:, new_p_value_cols[i]] = df[col].apply(lambda x: format_pvalue(x))

    df = df.drop(columns=p_value_cols)

    return df


def get_keep_cols(df, extra_cols):
    keep_cols = ["Model", "years"]
    if "split" in df.columns:
        keep_cols.append("split")

    keep_cols += extra_cols
    return keep_cols


def get_out_cols(df, extra_cols):
    out_cols = ['Model', 'Follow-up time']
    if "split" in df.columns:
        out_cols.append("Split")

    out_cols += extra_cols
    return out_cols


def update_table_legend(df):
    # rename False in years to no limit
    df["Follow-up time"] = df["Follow-up time"].replace("False", "No limit")
    # replace basic with Clinical-basic and extended with Clinical-extended in Model column
    df["Model"] = df["Model"].replace("basic", "Clinical-basic")
    df["Model"] = df["Model"].replace("extended", "Clinical-extended")
    df["Model"] = df["Model"].replace("image-only", "Image-only")
    return df


def set_recurrence_columns(df):
    columns_list = [
        'Model',
        'Follow-up time',
        '# Predicted LR mean', '# Predicted LR (95% CI)',
        '# Predicted HR mean', '# Predicted HR (95% CI)',
        '# Misclassified LR mean', '# Misclassified LR (95% CI)',
        '# Misclassified HR mean', '# Misclassified HR (95% CI)',
        '% LR mean', '% LR (95% CI)',
        '% HR mean', '% HR (95% CI)',
        '% Recurrence p-value',
        '% Recurrence s-value',
        '% Censored LR mean', '% Censored LR (95% CI)',
       '% Censored HR mean', '% Censored HR (95% CI)',
    # could also calculate a p-value for censoring
    ]
    return df[columns_list]

def set_metrics_columns(df):
    columns_list = [
        "Model",
        "Follow-up time",
        "ROC-AUC mean",
        "ROC-AUC (95% CI)",
        "NPV mean",
        "NPV (95% CI)",
        "Specificity mean",
        "Specificity (95% CI)",
        "Sensitivity mean",
        "Sensitivity (95% CI)",
        "Hazard Ratio pooled",
        "Hazard Ratio pooled (95% CI)",
        "Hazard Ratio p-value",
        "Hazard Ratio s-value",
        "Hazard Ratio p-value multiple-testing",
        "Hazard Ratio s-value multiple-testing",
    ]

    return df[columns_list]

def rename_model_names(df):
    # Mapping for renaming models
    rename_map = {
        'clinical_basic': 'Cox-Clinical',
        'clinical_extended': 'Cox-Clinical-Extended',
        'outer_cross_val_img': 'DL-Image-only',
        'outer_cross_val_int': 'DL-Integrative',
        'integrative_catboost': 'CatBoost-Integrative',
        'integrative_cox': 'Cox-Integrative'
    }
    df['Model'] = df['Model'].replace(rename_map)
    # Define the new order for the models
    new_order = [
        'Cox-Clinical',
        'Cox-Clinical-Extended',
        'DL-Image-only',
        'DL-Integrative',
        'CatBoost-Integrative',
        'Cox-Integrative'
    ]

    # Reorder the DataFrame
    df['Model'] = pd.Categorical(df['Model'], categories=new_order, ordered=True)
    df = df.sort_values('Model').reset_index(drop=True)  # necessary
    return df


def format_outer_cross_results(df, metrics:bool):
    if 'Hazard Ratio p-value multiple-testing corrected' in df.columns:
        df.rename(columns={'Hazard Ratio p-value corrected': 'Hazard Ratio p-value'}, inplace=True)
        df.rename(columns={'Hazard Ratio s-value corrected': 'Hazard Ratio s-value'}, inplace=True)
        df.rename(columns={'Hazard Ratio p-value multiple-testing corrected': 'Hazard Ratio p-value multiple-testing'}, inplace=True)
        df.rename(columns={'Hazard Ratio s-value multiple-testing corrected': 'Hazard Ratio s-value multiple-testing'}, inplace=True)
    if "% Recurrence p-value multiple-testing corrected" in df.columns:
        df.rename(columns={'% Recurrence p-value multiple-testing corrected': '% Recurrence p-value'}, inplace=True)
        df.rename(columns={'% Recurrence s-value multiple-testing corrected': '% Recurrence s-value'}, inplace=True)
    if metrics:
        df = set_metrics_columns(df)
    else:
        df = set_recurrence_columns(df)
    df = rename_model_names(df)
    return df



def merge_mean_95ci_columns_for_metrics(df):
    # should be able to use function from data analysis too
    # Combine mean and 95% CI columns
    metrics_cols = [(col.replace(" (95% CI)", ""), col) for col in df.columns if col.endswith('(95% CI)')]
    # Loop through each metric column
    for metric_col, ci_col in metrics_cols:
        # Extract the base name of the metric by removing " mean"
        if f"{metric_col} mean" in df.columns:
            metric_name = metric_col
            metric_col = f"{metric_col} mean"
        else:
            metric_name = metric_col.split(" ")[0]
        # Check if the corresponding CI column exists
        if ci_col in df.columns:
            # Combine mean and CI into one column
            df[metric_name] = df[metric_col].astype(str) + ' ' + df[ci_col].astype(str)
            # Drop the original mean and CI columns if needed
            df = df.drop(columns=[ci_col])
            if metric_col != metric_name:
                df = df.drop(columns=[metric_col])

    df.columns = [col.replace('test_', '').replace('_HR', '') for col in df.columns]
    return df

