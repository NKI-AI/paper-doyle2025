#!/bin/bash

YRS=False
TARGET="outcome"
DATASET="Precision_NKI_89_05"
DS_NAME="Block1_Aperio_ExtraSlides"
EXPERIMENT_NAME="patientstable_new"

#### To analyse the whole Dutch dataset, and the individual outer splits
RUN_CONFIG_SPLITS="task=cls_mil/tile_dataset \
  ++task.experiment_name=$EXPERIMENT_NAME \
  data="${DATASET}/${DS_NAME}.yaml" \
  data_prep.construct_dataset=true \
  data.data_sel_params.target=$TARGET \
  data.data_sel_params.cutoff_years=$YRS \
  data.data_sel_params.exclude_endocrine_therapy=true \
  data.data_sel_params.exclude_radiotherapy=true \
  data.data_sel_params.drop_nas_in_cols=[] \
"

#### First get overall characteristics for whole dataset, and first split
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS "data.meta_data_cols_orig.split=split_non_rt_only0"
# Then get it for clinical basic dataset
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data.data_sel_params.drop_nas_in_cols=[er,her2,grade,pr,age_diagnose]" \
  "data.meta_data_cols_orig.split=split_non_rt_only0"
# Then get it for clinical extended dataset
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data.data_sel_params.drop_nas_in_cols=[er,her2,grade,pr,age_diagnose,cox2,p16]" \
  "data.meta_data_cols_orig.split=split_non_rt_only0"

### Repeat this for each split, to get the characteristics in each split
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS "data.meta_data_cols_orig.split=split_non_rt_only1"
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data.data_sel_params.drop_nas_in_cols=[er,her2,grade,pr,age_diagnose]" \
  "data.meta_data_cols_orig.split=split_non_rt_only1"
# Then get it for clinical extended dataset
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data.data_sel_params.drop_nas_in_cols=[er,her2,grade,pr,age_diagnose,cox2,p16]" \
  "data.meta_data_cols_orig.split=split_non_rt_only1"

python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS "data.meta_data_cols_orig.split=split_non_rt_only2"
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data.data_sel_params.drop_nas_in_cols=[er,her2,grade,pr,age_diagnose]" \
  "data.meta_data_cols_orig.split=split_non_rt_only2"
# Then get it for clinical extended dataset
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data.data_sel_params.drop_nas_in_cols=[er,her2,grade,pr,age_diagnose,cox2,p16]" \
  "data.meta_data_cols_orig.split=split_non_rt_only2"

  ### Repeat this for each split, to get the characteristics in each split
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS "data.meta_data_cols_orig.split=split_non_rt_only3"
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data.data_sel_params.drop_nas_in_cols=[er,her2,grade,pr,age_diagnose]" \
  "data.meta_data_cols_orig.split=split_non_rt_only3"
# Then get it for clinical extended dataset
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data.data_sel_params.drop_nas_in_cols=[er,her2,grade,pr,age_diagnose,cox2,p16]" \
  "data.meta_data_cols_orig.split=split_non_rt_only3"

### Repeat this for each split, to get the characteristics in each split
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS "data.meta_data_cols_orig.split=split_non_rt_only4"
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data.data_sel_params.drop_nas_in_cols=[er,her2,grade,pr,age_diagnose]" \
  "data.meta_data_cols_orig.split=split_non_rt_only4"
# Then get it for clinical extended dataset
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data.data_sel_params.drop_nas_in_cols=[er,her2,grade,pr,age_diagnose,cox2,p16]" \
  "data.meta_data_cols_orig.split=split_non_rt_only4"

#### Then get the data for Sloane
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data=Sloane/default" \
  "data.cv_splitter=null"
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
  "data=Sloane/default" \
  "data.cv_splitter=null" \
  "data.data_sel_params.drop_nas_in_cols=[er,her2,grade,pr,age_diagnose]" \

