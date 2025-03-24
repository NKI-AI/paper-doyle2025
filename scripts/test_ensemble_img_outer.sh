#!/bin/bash
## Machine parameters per hyperparam configuration
NUM_GPUS_PER_NODE=1
NUM_WORKERS=4
echo $EXPERIMENT_NAME
MLFLOW_EXPERIMENT_NAME=$EXPERIMENT_NAME
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST")  # there is only one node at this point -- the master node
MLFLOW_PORT=43283
MLFLOW_ADDRESS="http://$MASTER_NODE:$MLFLOW_PORT"
MLFLOW_BACKEND_STORE_DIR="/home/s.doyle/to_mlflow_ouputs_new_sept24/"
MLFLOW_ARTIFACT_DIR=/home/s.doyle/mlflow_artifacts_rhpc
# Hyperparameter config -- multirun example:  # Use the non-optimize bandwidth experiment
DATASET="Precision_NKI_89_05"
DS_NAME="Block1_Aperio_ExtraSlides"
TILE_SIZE=512 #works with batches up to 1000 images at 1mpp
BATCH_SIZE=200
TARGET="outcome"
OUTER_SPLIT=4
EXPERIMENT_NAME="Folds_correct_outer_cross${OUTER_SPLIT}_val_img${DS_NAME}_${TILE_SIZE}_${BATCH_SIZE}"  # change this every time you run the script outer_cross

echo $EXPERIMENT_NAME
MLFLOW_EXPERIMENT_NAME=$EXPERIMENT_NAME
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST")  # there is only one node at this point -- the master node
MLFLOW_PORT=43283
MLFLOW_ADDRESS="http://$MASTER_NODE:$MLFLOW_PORT"
MLFLOW_BACKEND_STORE_DIR="/home/s.doyle/to_mlflow_outputs_new_sept24/"
MLFLOW_ARTIFACT_DIR=/home/s.doyle/mlflow_artifacts_rhpc


HYPERPARAMETERS="\
lit_module.lr=0.00003 \
lr_scheduler=step_lr \
lr_scheduler.scheduler.gamma=0.5 \
datamodule.batch_size=$BATCH_SIZE \
model.encoder.conv_blocks=3  \
model.encoder.dropout_rate=0.0 \
model.encoder.freeze_layer_names=[] \
pretrain=hissl \
norms=imagenet \
tiling.region_mpp=1.0 \
\
"

EVAL_PER_EPOCH=5  # how often to run the validation set
EVAL='best_auc'  # or best or specific epoch with file extension .ckpt
RUN_CONFIG="task=cls_mil/tile_dataset \
  ++task.experiment_name=$EXPERIMENT_NAME \
  data="${DATASET}/${DS_NAME}.yaml" \
  ++data_prep.construct_dataset=true \
  data.data_sel_params.cutoff_years=false \
  data.meta_data_cols_orig.split=split_non_rt_only$OUTER_SPLIT \
  data.data_sel_params.target=$TARGET \
  ++dataset.debug=false \
  model.decoder.num_classes=2 \
  experiment.kfolds=5 \
  experiment.train=true \
  experiment.test=true \
  experiment.visualise=false \
  lit_module=cls_model/mil_module_focal_loss_bce \
  lit_module.loss_fn.alpha=-1 \
  logger.mlflow_dir=file://$MLFLOW_BACKEND_STORE_DIR \
  lr_scheduler.scheduler.step_size=$(($EVAL_PER_EPOCH )) \
  metrics_trackers=default \
  sampler=dfsubset_weighted_batchsampler \
  sampler.no_eval_per_epoch=$EVAL_PER_EPOCH \
  pre_transforms=default \
  transforms=shape_augs_gpu \
  ++tiling.region_size=[[$TILE_SIZE,$TILE_SIZE]] \
  tiling.outsize=$TILE_SIZE \
  ++trainer.max_epochs=$(( $EVAL_PER_EPOCH*4 )) \
"

# Overrride run config with ensemble config
ENSEMBLE_CONFIG="\
experiment.train=false \
experiment.test=true \
experiment.ensemble=true \
experiment.train_without_val=false \
data.cv_splitter=false \
++data_prep.construct_dataset=true \
"

# Data setup
DATA_PARAMETERS="datamodule.num_workers=$NUM_WORKERS"

#MLFlow setup
MLFLOW_PARAMETERS="logger.mlflow.experiment_name=$MLFLOW_EXPERIMENT_NAME \
logger.mlflow.artifact_location=$MLFLOW_ARTIFACT_DIR logger.mlflow.save_dir=$MLFLOW_BACKEND_STORE_DIR"
MACHINE_PARAMETERS="trainer.devices=$NUM_GPUS_PER_NODE" #trainer.num_nodes=$NUM_NODES


python ../../drop/engine/cls/trainer.py $RUN_CONFIG $MACHINE_PARAMETERS $DATA_PARAMETERS $MLFLOW_PARAMETERS $HYPERPARAMETERS $ENSEMBLE_CONFIG

SLOANE_PARAMS="data=Sloane/default.yaml experiment.train=false  data.data_sel_params.cutoff_years=False"
python ../../drop/engine/cls/trainer.py \
$RUN_CONFIG $MACHINE_PARAMETERS $DATA_PARAMETERS $MLFLOW_PARAMETERS $HYPERPARAMETERS $ENSEMBLE_CONFIG $SLOANE_PARAMS