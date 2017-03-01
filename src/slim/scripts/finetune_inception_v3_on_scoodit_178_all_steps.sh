#!/bin/bash
#
# This script performs the following operations:
# 2. Fine-tunes an InceptionV3 model on the scoodit_178 training set.
# 3. Evaluates the model on the scoodit_178 validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inceptionv3_on_scoodit_178.sh

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=${HOME}/PycharmProjects/scoodit_image_classification/models/downloaded_tf_models

# Where the dataset is saved to.
DATASET_DIR=${HOME}/PycharmProjects/scoodit_image_classification/data/processed/scoodit_178/

BATCH=32
CLONES=16
READERS=16
THREADS=64
STEPS1=1000
STEPS2=3000
LR1=0.01
LR2=0.001
LR3=0.0005
LR4=0.0001
DATASET=scoodit_178
MODEL=inception_v3

LR=${LR1}
SUFFIX=lr_${LR}
# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=${HOME}/PycharmProjects/scoodit_image_classification/models/${MODEL}/${DATASET}/second_run/${SUFFIX}

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

LR=${LR2}
SUFFIX=lr_${LR}
# Where the training (fine-tuned) checkpoint and logs will be saved to.
PRETRAINED_CHECKPOINT_DIR=${TRAIN_DIR}
TRAIN_DIR=${HOME}/PycharmProjects/scoodit_image_classification/models/${MODEL}/${DATASET}/second_run/${SUFFIX}

# Fine-tune all the new layers.
python train_image_classifier.py \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL} \
  --max_number_of_steps=${STEPS2} \
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


LR=${LR3}
SUFFIX=lr_${LR}
# Where the training (fine-tuned) checkpoint and logs will be saved to.
PRETRAINED_CHECKPOINT_DIR=${TRAIN_DIR}
TRAIN_DIR=${HOME}/PycharmProjects/scoodit_image_classification/models/${MODEL}/${DATASET}/second_run/${SUFFIX}

# Fine-tune all the new layers.
python train_image_classifier.py \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL} \
  --max_number_of_steps=${STEPS2} \
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


LR=${LR4}
SUFFIX=lr_${LR}
# Where the training (fine-tuned) checkpoint and logs will be saved to.
PRETRAINED_CHECKPOINT_DIR=${TRAIN_DIR}
TRAIN_DIR=${HOME}/PycharmProjects/scoodit_image_classification/models/${MODEL}/${DATASET}/second_run/${SUFFIX}

# Fine-tune all the new layers.
python train_image_classifier.py \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL} \
  --max_number_of_steps=${STEPS2} \
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
