# opioid_time_series_analysis

The Office of Enterprise Data and Analytics investigated applications of multivariate time-series classification (MTSC) to predict opioid-related emergency department visits, and further understand factors such as: timing of opioid type/dose amount/fill dates, timing of medication for opioid use disorder (MOUD), as well potential non-time-series contributors, such as beneficiary demographics, presence of chronic conditions, and other important confounding factors.

Confluence Page: https://confluenceent.cms.gov/pages/viewpage.action?spaceKey=APP&title=OEDA+AI+Pilot+Team+Page

<h3> Environment Dependence </h2>

Some parts of the code will be broken outside the VRDC development environment, such as references to absolute filepaths or queries to the spark environment. This is expected, as the code here is meant to serve as a log of our work as well as inspiration for future work.

MLFlow, an open source machine learning tracking library that Databricks comes equipped with, was the backbone of our metric tracking during training. Databricks has MLFlow servers automatically setup to connect to within each instance that is provisioned for you. If you run this code outside of the Databricks environment, you will need to either edit or remove the MLFlow interaction in the training loop, or instantiate an MLFlow server yourself.

For access to production code in the VRDC, contact James DelAguila @ james.delaguila@cms.hhs.gov or Keri Apostle @ keri.apostle@cms.hhs.gov.

## Data
The underlying source of data are CCW tables, including the Enhanced Longitudinal Database (ELDB), Geographic Variation Database (GVDB), Chronic Conditions tables, and Geographically-Based Indices of Health (GBIH).

### Table Preprocessing

Preprocessing consists of several steps:
- Convert claim-level data into a time-series panel format, which awaits further preprocessing (data/make_spark_data_table.py).
- Coding of continuous time-series values, using the TSFRESH approach or an abridged set of custom time-series features. 
- One-hot encoding categorical variables (see the one_hot_encoder notebook) -- these tables are also saved for use as non-resampled inputs
- (optionally) SMOTE oversampling
  - (optionally) Undersampling using either the SMOTEENN (edited nearest neighbor) or Tomek methods.
  - Writing the resampled data frames to tables for reuse.

Each of these steps is described in more detail below.

#### Time-series panel format
Most time-series algorithms require specific table formats. Panel data consists of a table with 1 row per beneficiary and 1 column for each variable which themselves consist arrays with each element corresponding to a value during a particular time period (in our case, daily). This project includes a custom set of functions to build the panel format from line-level claims data.

#### TSFRESH encoding

TSFRESH is a python package that automatically calculates a large number of time series characteristics, the so called features. TSFRESH transformations are contained in the baseline models folder. See https://tsfresh.readthedocs.io/en/latest/ for full documentation

#### One-hot encoding

In the data/tables_preprocess_updated notebook, the first cells run (import) the one_hot_encoder notebook to use a
consistent one-hot encoding method and ordering of fields. The details
of that encoder are described within. In this notebook we use a modified
version of the `CMSPyTorchDataset` class used by the experimental model;
its main variation is that it produces a Pandas DataFrame instead of
a PyTorch vector dataset. This simplifies the remaining preprocessing
steps in this notebook.

After defining the encoder class, we apply it to our two input datasets
(the "TSFRESH" and "Abbridged TS Features" datasets) to produce one-hot
encoded DataFrames, which are then saved as Databricks tables. 

#### SMOTE Oversampling / ENN and Tomek Undersampling

To correct for class imbalance, we have created additional SMOTE datasets. Because our dataset includes many categorical variables, we can't apply plain SMOTE (or Approx-SMOTE, in its current implementation). We use SMOTENC from the `imbalanced-learn` library, which is designed to handle a mix of categorical and numeric variables.

We also optionally apply the Tomek's Links or ENN (Edited Nearest Neighbor) 
undersampling methods to remove samples from the majority class. They have 
slightly different strategies for removing samples, which can result in 
slightly different boundary tradeoffs in models trained from them.

More details of the SMOTE/SMOTE-NC, Tomek's Links, and ENN methods can
be found in the project technical report.

<h3 > Overview of Notebooks </h3>

#### Baseline Model
 - TSFRESH_Spark.py
    > A notebook which creates TSFRESH time-series characteristics, performs feature selection from amongst that set, and saves a Spark table.
 - abbreviated_feature_extraction.py
    > A notebook which creates an abbreviated set of feature characteristics (mean and variance over entire study period and last 30 days of study period), performs feature selection from amongst that set, and saves a Spark table.
 - logisitic_regression/Logistic_Regression_abbr_time_series_classweights.py
    > A notebook which weights observations according to class (ED visit vs. no ED visit), trains a regularized logistic regression model using the abbreviated set of feature characteristics, performs hyperparameter tuning, tests the resulting model on a validation 'hold-out' sample, and logs the result via MLFLOW
 - logisitic_regression/Logistic_Regression_abbr_time_series_smote.py
    > A notebook which uses the SMOTE version of the dataset, trains a regularized logistic regression model using the abbreviated set of feature characteristics, performs hyperparameter tuning, tests the resulting model on a validation 'hold-out' sample, and logs the result via MLFLOW
 - logisitic_regression/Logistic_Regression_tsfresh_classweights.py
    > A notebook which weights observations according to class (ED visit vs. no ED visit), trains a regularized logistic regression model using the TSFRESH set of feature characteristics, performs hyperparameter tuning, tests the resulting model on a validation 'hold-out' sample, and logs the result via MLFLOW
 - logisitic_regression/Logistic_Regression_tsfresh_smote.py
    > A notebook which uses the SMOTE version of the dataset, trains a regularized logistic regression model using the TSFRESH set of feature characteristics, performs hyperparameter tuning, tests the resulting model on a validation 'hold-out' sample, and logs the result via MLFLOW
 
 - xgboost_models/MLFLOW_Py_DistXgboost_max_delta.py
    > A notebook which uses weighted observations according to class (ED visit vs. no ED visit), trains an xgboost using the TSFRESH set of feature characteristics, performs hyperparameter tuning, tests the resulting model on a validation 'hold-out' sample, and logs the result via MLFLOW
 - xgboost_models/MLFLOW_Py_DistXgboost_max_delta_abbr_ts.py
    > A notebook which uses weighted observations according to class (ED visit vs. no ED visit), trains an xgboost using the abbreviated set of feature characteristics, performs hyperparameter tuning, tests the resulting model on a validation 'hold-out' sample, and logs the result via MLFLOW



#### Experimental Model
 - training_main.py
    > The training notebook that launches the training loop, imports code, and instantiates hyperparameters.
 - training_with_pytorch_dataloader_main_jw.py
    > A secondary training notebook to persist alternative training runs or to schedule jobs with on the Databricks environment. Some hyperparameters or even small bits of code could be different, but the core training loop and goal of this notebook is the same as training_with_pytorch_dataloader_main.ipynb
 - feature_representation_exploration.py
    > This notebook contains code for visualizing three dimensional representations of data that the model learns during training. 
 - utils/prep_data_from_spark.py
    > This notebook contains functions for querying spark and converting it into a format we can use in a training, validation, and test set.
 - utils/model_setup.py
    > This notebook contains python code pertaining to the instantiation of the model. It is mostly executed in the training notebooks where it is imported.
 - utils/dataloader_setup.py
    > This notebook contains python code that's needed to setup dataset and dataloader for the training. It is mostly executed in the training notebooks where it is imported. 
 - utils/evaluation_pipeline.py
    > This notebook contains the evaluation pipeline that we used to test the model. It derives and saves metrics relevant to performance. It also contains our subgroup analysis. 

<!-- PROJECT FOOTER -->
<h2 id="footer"> Project Details </h2>

<h3 id="Contact"> Contact </h3>

[![KUNGFU.AI][kungfu-shield]][kungfu-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-url]: ./LICENSE.md
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/kungfuai/
[python-url]: https://www.python.org
[docker-url]: https://www.docker.com
[docker-compose-url]: https://docs.docker.com/compose/install/
[nvidia-url]: https://github.com/NVIDIA/nvidia-container-runtime
[kungfu-shield]: https://img.shields.io/badge/KUNGFU.AI-2022-red
[kungfu-url]: https://www.kungfu.ai
