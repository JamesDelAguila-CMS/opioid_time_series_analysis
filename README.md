# opioid_time_series_analysis
https://confluenceent.cms.gov/pages/viewpage.action?spaceKey=APP&title=OEDA+AI+Pilot+Team+Page

# Table Preprocessing

This README file describes the data preprocessing performed
as part of the OEDA AI Explorers pilot project, and is intended to
accompany the tables_preprocess_updated notebook 
originally developed in the CCW VRDC environment. It depends on 
the data tables and resources deployed there.

## Preprocessing Overview

Preprocessing consists of several steps:
- One-hot encoding categorical variables (see the one_hot_encoder notebook) -- these tables are also saved for use as non-resampled inputs
- SMOTE oversampling
- (optionally) Undersampling using either the SMOTEENN (edited nearest neighbor) or Tomek methods.
- Writing the resampled data frames to tables for reuse.

Each of these steps is described in more detail below.

## One-hot encoding

The first cells run (import) the one_hot_encoder notebook to use a
consistent one-hot encoding method and ordering of fields. The details
of that encoder are described with it. In this notebook we use a modified
version of the `CMSPyTorchDataset` class used by the experimental model;
its main variation is that it produces a Pandas DataFrame instead of
a PyTorch vector dataset. This simplifies the remaining preprocessing
steps in this notebook.

After defining the encoder class, we apply it to our two input datasets
(the "TSFRESH" and "Abbridged TS Features" datasets) to produce one-hot
encoded DataFrames, which are then saved as Databricks tables. 

## SMOTE Oversampling / ENN and Tomek Undersampling

To correct for class imbalance, we apply SMOTE. Because our dataset includes many categorical variables, we can't apply plain SMOTE (or Approx-SMOTE, in its current implementation). We use SMOTENC from the `imbalanced-learn` library, which is designed to handle a mix of categorical and numeric variables.

We also optionally apply the Tomek's Links or ENN (Edited Nearest Neighbor) 
undersampling methods to remove samples from the majority class. They have 
slightly different strategies for removing samples, which can result in 
slightly different boundary tradeoffs in models trained from them.

More details of the SMOTE/SMOTE-NC, Tomek's Links, and ENN methods can
be found in the project technical report.

Note that `imblearn` is not Spark-aware, so the actual fit/resample step will run on the driver node, not the cluster worker nodes. This is extremely
compute-intensive and will take many hours to complete.

The bulk of the work in this notebook is done in this final cell, which
applies SMOTE-NC, and optionally one of ENN or Tomek's Links, to the training split of both of the one-hot encoded input tables. It then saves the resulting
resampled data to a new table, appending descriptive suffix to identify which
preprocessing was applied. To produce the resampled tables used in the model
development, this cell was run three times (each possible combination of
SMOTE-NC and undersampling) to produce six tables (TSFRESH and Abridged 
features for each combination).  The table names with their configurations
are summarized in the project technical report, as well as in a table
in the DistXGBoost notebook (which itself was not used in the final 
model runs).

One variation that was not tested due to time constraints is undersampling
 _without_ oversampling -- that is, only applying ENN or Tomek's Links
 editing. This could be done by modifying the processing code notionally
 as follows:
 ```
 from imblearn.under_sampling import EditedNearestNeighbors
 from imblearn.under_sampling import TomekLinks
   .
   .
   . 
   # omit SMOTE
   # smote = SMOTENC(random_state=2143, categorical_features=cat_cols)
   sme = TomekLinks() # or EditedNearestNeighbors()
   print('Starting resample...')
   start = time.time()
   X_res, y_res = sme.fit_resample(X, y)
   .
   .
   .
```


<!-- 
  TODOS: missing logo pic - can just kill the img if we want
  Missing license
  Needs an editting pass
-->
<br />
<p align="center">
  <img src="./data/kungfuai_logo.png" alt="KUNGFU.AI" width="200">
  <h3 align="center">KUNGFU.AI Project Template</h3>
  <p align="center">
    CMS Opioid Recidivism
    <br/>
  </p>
</p>

<!-- ABOUT THE PROJECT -->
<h2 id="about"> About The Project </h2>
We developed a neural network to predict the likelihood a beneficiary will be readmitted to the hospital given the previous ten months worth of prescription data, some basic demographic information, and their chronic conditions.

<h2 id="getting-started"> Getting Started </h2>

These instructions will give you an overall understanding of the codebase we developed and how the different notebooks fit together.

<h3> Broken Code </h2>

Some parts of the code will be broken outside the VRDC development environment, such as references to absolute filepaths or queries to the spark environment. This is expected, as the code here is meant to serve as a log of our work as well as inspiration for future work.

MLFlow, an open source machine learning tracking library that Databricks comes equipped with, was the backbone of our metric tracking during training. Databricks has MLFlow servers automatically setup to connect to within each instance that is provisioned for you. If you run this code outside of the Databricks environment, you will need to either edit or remove the MLFlow interaction in the training loop, or instantiate an MLFlow server yourself.

<h3 > Overview of Notebooks </h3>

 - training_with_pytorch_dataloader_main.ipynb
    > The training notebook that launches the training loop, imports code, and instantiates hyperparameters.
 - training_with_pytorch_dataloader_main_jw.ipynb
    > A secondary training notebook to persist alternative training runs or to schedule jobs with on the Databricks environment. Some hyperparameters or even small bits of code could be different, but the core training loop and goal of this notebook is the same as training_with_pytorch_dataloader_main.ipynb
 - feature_representation_exploration.ipynb
    > This notebook contains code for visualizing three dimensional representations of data that the model learns during training. 
 - utils/prep_data_from_spark.ipynb
    > This notebook contains functions for querying spark and converting it into a format we can use in a training, validation, and test set.
 - utils/model_setup.ipynb
    > This notebook contains python code pertaining to the instantiation of the model. It is mostly executed in the training notebooks where it is imported.
 - utils/dataloader_setup.ipynb
    > This notebook contains python code that's needed to setup dataset and dataloader for the training. It is mostly executed in the training notebooks where it is imported. 
 - utils/evaluation_pipeline.ipynb
    > This notebook contains the evaluation pipeline that we used to test the model. It derives and saves metrics relevant to performance. It also contains our subgroup analysis. 



<!-- PROJECT FOOTER -->
<h2 id="footer"> Project Details </h2>
<h3 id="license"> License </h3>

Please read our [license document][license-url] for more information.

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
