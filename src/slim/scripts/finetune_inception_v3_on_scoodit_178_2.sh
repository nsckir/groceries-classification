#!/bin/bash
#
# This script performs the following operations:
# 2. Fine-tunes an InceptionV3 model on the scoodit_178 training set.
# 3. Evaluates the model on the scoodit_178 validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inceptionv3_on_scoodit_178_2.sh 32 1 4 4 1000 500
INCEPTION_V3_LINK=http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
INCEPTION_V4_LINK=http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
INCEPTION_RESNET_LINK=http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz

# Where the pre-trained InceptionV3 checkpoint is saved to.


# Where the dataset is saved to.
DATASET_DIR=${HOME}/PycharmProjects/scoodit_image_classification/data/processed/scoodit_178/

BATCH=$1
CLONES=$2
READERS=$3
THREADS=$4

STEPS1=$5
STEPS2=$6
LR1=$7
LR2=$8
DATASET=scoodit_178
MODEL=inception_v3
SUFFIX=bt_${BATCH}_cl_${CLONES}_r_${READERS}_thr_${THREADS}_lr1_${LR1}_lr2_${LR2}

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=${HOME}/PycharmProjects/scoodit_image_classification/models/${MODEL}/${DATASET}/${SUFFIX}

PRETRAINED_CHECKPOINT_DIR=${HOME}/PycharmProjects/scoodit_image_classification/models/inception_v3/scoodit_178/bt_32_cl_16_r_16_thr_64_lr1_0.1_lr2_0.001/all

# Fine-tune all the new layers.
python train_image_classifier.py \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=${DATASET} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL} \
  --max_number_of_steps=${STEPS2} \
  --learning_rate=${LR2} \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=30 \
  --save_summaries_secs=30 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --batch_size=${BATCH} \
  --num_clones=${CLONES} \
  --num_readers=${READERS} \
  --num_preprocessing_threads=${THREADS}

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=${DATASET} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL}
