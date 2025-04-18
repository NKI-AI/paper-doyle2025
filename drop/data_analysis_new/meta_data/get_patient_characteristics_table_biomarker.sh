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
  data.data_sel_params.exclude_endocrine_therapy=false \
  data.data_sel_params.exclude_radiotherapy=false \
  data.data_sel_params.drop_nas_in_cols=[er,her2,grade] \
"
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS "data.meta_data_cols_orig.split=split_rt_and_non_rt0"
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS "data.meta_data_cols_orig.split=split_rt_and_non_rt1"
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS "data.meta_data_cols_orig.split=split_rt_and_non_rt2"
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS "data.meta_data_cols_orig.split=split_rt_and_non_rt3"
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS "data.meta_data_cols_orig.split=split_rt_and_non_rt4"

#
###### To analyse inner folds, set the relevant target variable (for instance HER2)
#TARGET="her2"
#python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS \
#    "data.meta_data_cols_orig.split=split_rt_and_non_rt0" \
#    "++data.data_sel_params.target=${TARGET}"
## ..etc for each outer fold.
#
#
#
#### To analyse the whole Sloane dataset
TARGET=""
python patient_characteristics_tables_overall.py $RUN_CONFIG_SPLITS  \
 "data=Sloane/default" \
  "data.slide_mapping.subdirs=[Sloane_rescanned/]" \
  "data.data_sel_params.exclude_scottish=false" \
  "data.cv_splitter=null" \
   "++data.data_sel_params.target=${TARGET}"
