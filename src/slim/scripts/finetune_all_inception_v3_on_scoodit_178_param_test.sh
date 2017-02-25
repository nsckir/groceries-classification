#!/bin/bash
#
# This script performs the following operations:
# 2. Fine-tunes an InceptionV3 model on the scoodit_178 training set.
# 3. Evaluates the model on the scoodit_178 validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inceptionv3_on_scoodit_178.sh 32 1 4 4 1000

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=${HOME}/PycharmProjects/scoodit_image_classification/models/downloaded_tf_models

# Where the dataset is saved to.
DATASET_DIR=${HOME}/PycharmProjects/scoodit_image_classification/data/processed/scoodit_178/

BATCH=$1
CLONES=$2
READERS=$3
THREADS=$4
STEPS1=$5
LR=$6
DATASET=scoodit_178
MODEL=inception_v3
SUFFIX=bt_${BATCH}_cl_${CLONES}_r_${READERS}_thr_${THREADS}_lr_${LR}

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=${HOME}/PycharmProjects/scoodit_image_classification/models/${MODEL}/${DATASET}/${SUFFIX}

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
  tar -xvf inception_v3_2016_08_28.tar.gz
  mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt
  rm inception_v3_2016_08_28.tar.gz
fi

# Fine-tune only the new layers.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=${STEPS1} \
  --learning_rate=${LR} \
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
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL}

