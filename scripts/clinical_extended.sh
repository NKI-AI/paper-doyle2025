#!/bin/bash
# Mlflow configs
DATASET="Precision_NKI_89_05"
DS_NAME="Block1_Aperio_ExtraSlides"
TARGET="outcome"
CLINICAL_VARS="extended"
YRS=False
OUTER_SPLIT=4
EXPERIMENT_NAME="Clinical_ext_only_outersplit_bin${OUTER_SPLIT}COXPH_months_acc_noNAs_withouther2border${DS_NAME}_${CLINICAL_VARS}_years${YRS}_final"

echo $EXPERIMENT_NAME
MLFLOW_EXPERIMENT_NAME=$EXPERIMENT_NAME
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST")  # there is only one node at this point -- the master node
MLFLOW_PORT=43283
MLFLOW_ADDRESS="http://$MASTER_NODE:$MLFLOW_PORT"
MLFLOW_BACKEND_STORE_DIR="/home/s.doyle/to_mlflow_outputs_new_sept24/"
MLFLOW_ARTIFACT_DIR=/home/s.doyle/mlflow_artifacts_rhpc

CLINICAL_FEATURES="['er','her2','grade','age_diagnose','pr','p16','cox2']"

HYPERPARAMETERS="
model.alpha=0.05 \
model.l1_ratio=0.0 \
model.penalizer=0.1 \
"

RUN_CONFIG="task=clinical_vars/clinical_vars \
  ++task.experiment_name=$EXPERIMENT_NAME \
  clinical_vars_type=$CLINICAL_VARS \
  load_img_model_preds=false \
  data.data_sel_params.cutoff_years=$YRS \
  data="${DATASET}/${DS_NAME}.yaml" \
  data.meta_data_cols_orig.split=split_non_rt_only$OUTER_SPLIT \
  +outer_cross_val_split=$OUTER_SPLIT \
  +threshold_method=accuracy \
  ++data_prep.construct_dataset=true \
  data.data_sel_params.target=$TARGET \
  model=ml_model/cox_ph_model \
  experiment.train=true \
  data.cv_splitter=false \
  data.data_sel_params.drop_nas_in_cols=$CLINICAL_FEATURES \
  experiment.test=true \
  experiment.visualise=false \
  experiment.kfolds=5 \
  logger.mlflow_dir=file://$MLFLOW_BACKEND_STORE_DIR \
"

MLFLOW_PARAMETERS="logger.mlflow.experiment_name=$MLFLOW_EXPERIMENT_NAME \
logger.mlflow.artifact_location=$MLFLOW_ARTIFACT_DIR logger.mlflow.save_dir=$MLFLOW_BACKEND_STORE_DIR"

python ../../drop/engine/cls/trainer_ml.py $RUN_CONFIG $MLFLOW_PARAMETERS $HYPERPARAMETERS