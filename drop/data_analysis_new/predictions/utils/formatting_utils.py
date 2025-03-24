
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