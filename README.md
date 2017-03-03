# Scoodit Image Classification
Classification of Cooking Ingredients Using Deep Learning

This directory contains code for training and evaluating of Inception v3 model 
on cooking ingredients. It contains scripts that will allow you to finetune the 
model trained on the 1000 classes of the ImageNet to 178 classes containing 
fruits, vegetables and other cooking ingredients. It also contains code for
converting the images to TensorFlow's native TFRecord format.
## Table of contents

<a href="#organization">Project Organization</a><br>
<a href='#data'>Preparing the Datasets</a><br>
<a href='#finetune'>Fine-tuning of Inception V3</a><br>
<a href='#validate'>Evaluating performance</a><br>

## Project Organization
<a id='organization'></a>

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
    │   ├─  downloaded_tf_models
    │   └── inception_v3
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
    │            └── scoodit_178_test_snaps.py
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------
## Preparing the datasets
<a id='data'></a>

Two datasets have been used in this project: `scoodit_178` and `scoodit_178_test_snaps`. 


`scoodit_178` contains ~180K images  from ImageNet across 178 classes.
The classes correspond to the ingredients listed in `Ingredients_lists_to_be_updated.xls` on google drive.

`scoodit_178_test_snaps` contains ~4K images of fruits, vegetables and
other groceries acquired for the validation of the model.

#### Download and convert `scoodit_178`

The dataset can be downloaded from [s3://scoodit.image.classification.data/raw_data](https://console.aws.amazon.com/s3/buckets/scoodit.image.classification.data/raw_data?region=us-east-1)

The archives in each subdirectory must be unpacked. You can use `src/notebooks/01_nsckir_temp_extract_archives.ipynb` 
to unpack all the archives at once. After extraction the data must have the following structure (NO BLANKS IN FOLDER NAMES):
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

Run `src/data/create_scoodit178_train_test_split.sh` to put 95% of the
images in the directory for training `data/raw/scoodit_178/train/ `
and the remaining 5% in the directory for validation `data/raw/scoodit_178/test/`.

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

Go to `src/slim/datasets` and run `convert_scoodit_178.py` to convert the raw data to TFRecord format
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

#### Download and convert `scoodit_178_test_snaps`

The original images are stored in [s3://grocerysnaps](https://console.aws.amazon.com/s3/buckets/grocerysnaps?region=us-east-1&tab=overview).
However, the folder structure must be EXACTLY the same as in
 `data/raw/scoodit_178/`.
Therefore I have renamed the folders and uploaded the images as
 a [single archive](https://s3.amazonaws.com/scoodit.image.classification.data/test_snaps).

Download the archive and extract in to `data/raw/test_snaps/`. 

Go to `src/slim/data/` and run `convert_scoodit_test_snaps.py`. Again, check if all tha paths
in the script are correct. 

When the script finishes you will find several TFRecord files created:
```shell
${PROJECT_FOLDER}/data/processed/scoodit_178_test_snaps/scoodit_178_validation_00000-of-00004.tfrecord
...
${PROJECT_FOLDER}/data/processed/scoodit_178_test_snaps/scoodit_178_validation_00003-of-00004.tfrecord
${PROJECT_FOLDER}/data/processed/scoodit_178_test_snaps/labels.txt
```

Now you are ready to train and evaluate the model.

## Train (fine tune) the Inception V3 Model on the scoodit_178 dataset
<a id='finetune'></a>

Download and extract the pretrained checkpoint of Inception V3 from [this link](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
and save it to `models/downloaded_tf_models/`

Go to `src/slim/`


The script `src/slim/scripts/finetune_inception_v3_on_scoodit_178_all_steps.sh` includes all steps and parameters
which retrain the inception_v3 model on the scoodit_178 data set. The final model will be saved
in `model/inception_v3/scoodit_178/`. The total run time on AWS p2.16xlarge is about 10 hours
using the parameter in the script. There might be potential to reduce this time by changing
the learning rate and the number of steps. The final model achieves 92% top5 accuracy
on the validation set. 

I have saved the final model in [s3://scoodit.models/](https://console.aws.amazon.com/s3/buckets/scoodit.models?region=us-east-1&tab=overview)
From there it can be deployed with Tensorflow Serving or AWS lambda. I have posted 
the details and examples in the slack channel.
 
## Validate the model on the grocery snaps
<a id='validate'></a>

Go to `src/slim/` and run:
```shell
python eval_image_classifier.py \
  --checkpoint_path=${MODEL_DIR} \
  --eval_dir=${MODEL_DIR}/test_snaps \
  --dataset_name=scoodit_178_test_snaps \
  --dataset_split_name=validation \
  --dataset_dir=${PROJECT_FOLDER}/data/processed/scoodit_178_test_snaps \
  --model_name=inception_v3
 ```
 
 `${MODEL_DIR}` is the folder where your fine tuned model is saved