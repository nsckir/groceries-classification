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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
