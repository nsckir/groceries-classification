#!/bin/bash

# Generate a list of  labels

LABELS_FILE="${HOME}/PycharmProjects/scoodit_image_classification/data/processed/scoodit_178/labels.txt"
SLIM_LABELS_FILE="${HOME}/PycharmProjects/scoodit_image_classification/data/processed/scoodit_178/slim_labels.txt"
#touch "${LABELS_FILE}"
TRAIN_DIRECTORY="${HOME}/PycharmProjects/scoodit_image_classification/data/raw/scoodit_178/train/"
ls -1 ${TRAIN_DIRECTORY} |  sed 's/\///' | sort | awk '{print NR":"$0}' > ${SLIM_LABELS_FILE}
ls -1 ${TRAIN_DIRECTORY} |  sed 's/\///' | sort  > ${LABELS_FILE}
