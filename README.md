Scoodit Image Classification
==============================

Classification of Cooking Ingredients Using Deep Learning

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train` (not used yet)
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources. (not used yet)
    │   ├── interim        <- Intermediate data that has been transformed. (not used yet)
    │   ├── processed      <- The final, canonical data sets for modeling which can be ingested by tensorflow.
    │   └── raw            <- The original, immutable data dump (raw images).
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details (not used yet)
    │
    ├── models             <- Downloaded pretrained tensorflow models and models trained on our dataset
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials. (not used yet)
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc. (not used yet)
    │   └── figures        <- Generated graphics and figures to be used in reporting (not used yet)
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   
    │   ├── features       <- Scripts to turn raw data into features for modeling (not used yet)
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make (not used yet)
    │   │                     predictions
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations (not used yet)
    │   │  
    │   │    
    │   └── slim           <- Clone of https://github.com/tensorflow/models/tree/master/slim
    │       │                Slim is a high level API to Tensorflow which make the training more convenient
    │       │
    │       ├── eval_image_classifier.py  <- Evaluate a model on a dataset           
    │       ├── train_image_classifier.py <- Train a model on a dataset
    │       └── datasets   <- Scrips to convert the raw data to dataset which can be used with slim 
    │            │
    │            ├── convert_scoodit_178.py
    │            ├── convert_scoodit_test_snaps.py
    │            ├── dataset_factory.py
    │            ├── scoodit_178.py
    │            ├── scoodit_178_test_snaps.py
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------
# Preparing the datasets
<a id='Data'></a>

Two datasets have been used in in this project. The first one is called  "scoodit_178".
It contains ~180K images  from ImageNet across 178 classes.
The classes correspond to the ingredients listed in `Ingredients_lists_to_be_updated.xls` on google drive.
The dataset can be downloaded from [s3://scoodit.image.classification.data/raw_data](https://console.aws.amazon.com/s3/buckets/scoodit.image.classification.data/raw_data?region=us-east-1)

After downloading the archives in each subdirectory must be unpacked, so that the data has the following structure (NO BLANKS IN FOLDER NAMES):
```shell
  ${PROJECT_FOLDER}/data/raw/scoodit_178/acorn_squash_n07717410/image.jpeg
  ${PROJECT_FOLDER}/data/raw/scoodit_178/acorn_squash_n07717410/another_image.jpeg
  ...
  ${PROJECT_FOLDER}/data/raw/scoodit_178/almond_n07750586/image.jpeg
  ${PROJECT_FOLDER}/data/raw/scoodit_178/almond_n07750586/another_image.jpeg
  ...
  ${PROJECT_FOLDER}/data/raw/scoodit_178/apple_n07739125/image.jpeg
  ${PROJECT_FOLDER}/data/raw/scoodit_178/apple_n07739125/another_image.jpeg
  ...
```
Run `src/notebooks/01_nsckir_temp_extract_archives.ipynb` to unpack all the archives at once
Run `src/data/create_scoodit178_train_test_split.sh` to put 95% of the images in the directory for training and the remaining 5% in the directory for validation.

After that you should have following data structure (NO BLANKS IN FOLDER NAMES):
```shell
  ${PROJECT_FOLDER}/data/raw/scoodit_178/train/acorn_squash_n07717410/143.jpeg
  ${PROJECT_FOLDER}/data/raw/scoodit_178/train/acorn_squash_n07717410/51234.jpeg
  ...
  ${PROJECT_FOLDER}/data/raw/scoodit_178/train/acorn_squash_n07717410/almond_n07750586/1467.jpeg
  ${PROJECT_FOLDER}/data/raw/scoodit_178/train/acorn_squash_n07717410/almond_n07750586/8765.jpeg
  ...
  ${PROJECT_FOLDER}/data/raw/scoodit_178/train/acorn_squash_n07717410/apple_n07739125/4356.jpeg
  ${PROJECT_FOLDER}/data/raw/scoodit_178/train/acorn_squash_n07717410/apple_n07739125/543.jpeg
  ...
  
  ${PROJECT_FOLDER}/data/raw/scoodit_178/test/acorn_squash_n07717410/134.jpeg
  ${PROJECT_FOLDER}/data/raw/scoodit_178/test/acorn_squash_n07717410/341.jpeg
  ...
  ${PROJECT_FOLDER}/data/raw/scoodit_178/test/acorn_squash_n07717410/almond_n07750586/43245.jpeg
  ${PROJECT_FOLDER}/data/raw/scoodit_178/test/acorn_squash_n07717410/almond_n07750586/3245.jpeg
  ...
  ${PROJECT_FOLDER}/data/raw/scoodit_178/test/acorn_squash_n07717410/apple_n07739125/1456.jpeg
  ${PROJECT_FOLDER}/data/raw/scoodit_178/test/acorn_squash_n07717410/apple_n07739125/654.jpeg
  ...
```

Now you can run `src/slim/datasets/convert_scoodit_178.py` to convert the raw data to TFRecord format
(You have to check the folder paths in the scripts. Might be that there are some absolute paths you have to adjust).

When the script finishes you will find several TFRecord files created:
```shell
  ${PROJECT_FOLDER}/data/processed/scoodit_178/scoodit_178_train_00000-of-00172.tfrecord
  ...
  ${PROJECT_FOLDER}/data/processed/scoodit_178/scoodit_178_train_00171-of-00172.tfrecord
  ...
  ${PROJECT_FOLDER}/data/processed/scoodit_178/scoodit_178_validation_00000-of-00016.tfrecord
  ...
  ${PROJECT_FOLDER}/data/processed/scoodit_178/scoodit_178_validation_00015-of-00016.tfrecord
  ${PROJECT_FOLDER}/data/processed/scoodit_178/labels.txt
```

Download and extract the pretrained checkpoint of Inception V3 from [this link](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
and save it to `${PROJECT_FOLDER}/models/downloaded_tf_models/`

Now you are ready to train the model.

Go to `${PROJECT_FOLDER}/src/slim/`

Fine tune only the last layer of Inception V3

```shell
$ DATASET_DIR=${PROJECT_FOLDER}/data/processed/scoodit_178
$ TRAIN_DIR=${PROJECT_FOLDER}/models/inception_v3/scoodit_178
$ CHECKPOINT_PATH=${PROJECT_FOLDER}/models/downloaded_tf_models/inception_v3.ckpt
$ python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=scoodit_178 \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits/Logits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits/Logits
```

Evaluate the results

```shell
$ python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=scoodit_178 \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3
```

The script `src/slim/scripts/finetune_inception_v3_on_scoodit_178_all_steps.sh` includes all steps and parameters
which retrain the inception_v3 model on the scoodit_178 data set