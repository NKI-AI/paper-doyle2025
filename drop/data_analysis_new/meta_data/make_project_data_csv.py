import logging
import numpy as np
import pandas as pd
from drop.data_proc.metadata_selection import prepare_precision_nki_data
from drop.data_proc.cv_split_creation import MIL_CVSplitter
from omegaconf import DictConfig, OmegaConf

def create_outer_split(df, target_var, partition_name, base_path):

    cv_params = DictConfig({
        'strategy': 'GroupKFold',
        'kfolds': 5,
        'stratify_on': target_var,
        'group_on': 'tissue_number_blockid'
    })

    cv_splitter = MIL_CVSplitter('/home/s.doyle/tmp/', 'outer_split.json', cv_params)
    splits_dict = cv_splitter(df)
    splits_df = pd.DataFrame(splits_dict)
    df= df.merge(splits_df, on=cv_params.group_on)
    # save outer fold characteristics
    from drop.data_analysis_new.meta_data.analyse_splits_per_group import analyse_outer_folds
    analyse_outer_folds(df,
                        folds=cv_params.kfolds,
                        stratify_on= cv_params.stratify_on,
                        group_on=cv_params.group_on,
                        partition_name = partition_name,
                        use_percentages=False,
                        base_path=base_path
                        )
    new_columns = {str(i): f'{partition_name}{i}' for i in range(cv_params.kfolds)}
    df = df.rename(columns=new_columns)
    df = df.replace('val', 'test', regex=True)
    return df



def make_project_data_csv(target_var, base_path):
    meta_fn = '/path/to/meta.csv'
    meta_file = pd.read_csv(meta_fn)

    df = prepare_precision_nki_data(meta_file,  target_var)
    logging.info("split for df with non-RT only")
    target_var_name = ''
    outname = f"{base_path}Precision_Split"
    if not (target_var_name == '' or  target_var_name == None):
        outname = f"{outname}_{target_var_name}"

    print("Split for df with and without RT")
    df_all = df[df["radiotherapy"].isin([0, 1])]
    partition_name = "split_rt_and_non_rt"
    df_all_outer = create_outer_split(df_all, target_var, partition_name)

    print("Split for df without  RT ")
    partition_name = "split_non_rt_only"
    df_non_rt = df[df["radiotherapy"].isin([0])]
    df_non_rt_outer = create_outer_split(df_non_rt, target_var, partition_name)

    print("Split for df with  RT only")
    partition_name = 'split_rt_only'
    df_rt = df[df["radiotherapy"].isin([1])]
    df_rt_outer = create_outer_split(df_rt, target_var, partition_name)

    # merge df_rt df_non_rt and df_all, on all columns that they have in common
    common_cols = list(set(df_rt_outer.columns) & set(df_non_rt_outer.columns) & set(df_all_outer.columns))
    merged_df = pd.merge(pd.merge(df_rt_outer, df_non_rt_outer, on=common_cols, how='outer'), df_all_outer, on=common_cols, how='outer')
    merged_df.to_csv(outname, index=False)



if __name__ == "__main__":
    make_project_data_csv(target_var='outcome' , base_path='/base/path')
