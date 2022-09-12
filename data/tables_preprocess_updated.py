# Databricks notebook source
# MAGIC %run "./one_hot_encoder"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table preprocessing
# MAGIC 
# MAGIC This notebook includes routines for building the preprocessed tables from the tsfresh and timeseries inputs.
# MAGIC 
# MAGIC Preprocessing consists of several steps:
# MAGIC - One-hot encoding categorical variables (see the one_hot_encoder notebook) -- these tables are also saved for use as non-resampled inputs
# MAGIC - SMOTE oversampling
# MAGIC - (optionally) Undersampling using either the SMOTEENN (edited nearest neighbor) or Tomek methods.
# MAGIC - Writing the resampled data frames to tables for reuse.

# COMMAND ----------

# Enable Arrow for faster pd/spark interop
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# use our project-standard splits
input_table_afe = 'eldb.opioid_SA_LA_hosp_final_abbr_ftr_extrctn_and_demos'
input_table_tsf = 'eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos'
train_split_sql = 'MOD(bene_id, 20) < 14'
val_split_sql = 'MOD(bene_id, 20) >= 14 AND MOD(bene_id, 20) < 18'
test_split_sql = 'MOD(bene_id, 20) >= 18'

# COMMAND ----------

# MAGIC %md
# MAGIC ## One-hot encoding
# MAGIC This is a modified version of the CMSPytorchDataset class that (also) produces a one-hot encoded pandas DataFrame.

# COMMAND ----------

import random
from torch.utils.data import Dataset, DataLoader

# pass in a pandas DataFrame (either via PySpark's toPandas() or via pd.read_csv)

class CMSPandasPytorchDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.bad_cateogries = ["STATIC_DEMO_bene_age"]
        self.prep_cols()
        self.run_one_hot_encoding()
        
    def prep_cols(self):
        for column in self.bad_cateogries:
            if column in self.df.columns:
                self.df.drop(column, axis=1, inplace=True)
        categorical_cols = [column for column in self.df.columns if "STATIC" in column]
        self.df.dropna(inplace=True, axis=0)
        self.categorical_df = self.df[categorical_cols].astype(str)
        self.df.drop(columns=categorical_cols, inplace=True)
        self.labels = self.df["target"].to_numpy().astype(float)
        self.df.drop(columns=["target"], axis=1, inplace=True)

    def run_one_hot_encoding(self):
        self.encoder = CMSOneHotEncoder(self.categorical_df.to_numpy())
        categorical_df_as_numpy = self.encoder.get_one_hots(self.categorical_df.to_numpy())
        self.pdf = pd.concat([self.df.astype(float), self.encoder.get_one_hots_df(self.categorical_df.to_numpy()), pd.DataFrame(self.labels, columns=["labels"])], axis=1)
        self.data = np.concatenate((self.df.to_numpy(), categorical_df_as_numpy), axis=1).astype(float)
        
    def __getitem__(self, index):
        features = self.data[index]
        label = self.labels[index]
        return features, label

    def __len__(self):
        return len(self.df)

# COMMAND ----------

## Create and catalog one-hot-encoded tables
for input_table in [input_table_afe, input_table_tsf]:
  pdf = spark.table(input_table).toPandas()
  cms_ds = CMSPandasPytorchDataset(df=pdf)
  ohe_df = spark.createDataFrame(cms_ds.pdf)
  ohe_df.write.mode("overwrite").saveAsTable(input_table + '_ohe')


# COMMAND ----------

# MAGIC %md
# MAGIC ## SMOTE Resampling
# MAGIC To correct class imbalance we apply SMOTE. Becasue our dataset includes many categorical variables, we can't apply plain SMOTE (or Approx-SMOTE, in its current implementation). We use SMOTENC from the `imbalanced-learn` library, which is designed to handle a mix of categorical and numeric variables.
# MAGIC 
# MAGIC We also optionally apply the Tomek or ENN (Edited Nearest Neighbor) undersampling methods to remove samples from the majority class. They have slightly different strategies for removing samples, which can result in slightly different boundary tradeoffs in models trained from them.
# MAGIC 
# MAGIC Note that `imblearn` is not Spark-aware, so the actual fit/resample step will run on the driver node, not the cluster worker nodes.

# COMMAND ----------

from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import pandas as pd
import time

for input_table in [input_table_afe, input_table_tsf]:
  print(f'Processing {input_table}')
  pdf = spark.table(input_table+'_ohe').where(train_split_sql).toPandas()
  X = pdf.drop(['bene_id','labels'], axis=1)
  y = pdf['labels']
  print(f'Original dataset samples per class {Counter(y)}')
  # categorical vars are:
  # - all CC_
  # - all STATIC_
  cat_cols = [X.columns.get_loc(col) for col in X.columns if col.startswith("CC_") or col.startswith("STATIC_")] 
  smote = SMOTENC(random_state=2143, categorical_features=cat_cols)
  sme = SMOTETomek(smote=smote) #SMOTEENN(smote=smote)
  print('Starting resample...')
  start = time.time()
  X_res, y_res = sme.fit_resample(X, y)  # to oversample only, use smote.fit_resample(X, y) instead
  end = time.time()
  print(f'Resampling took {end - start} seconds')
  print(f'Resampled dataset samples per class {Counter(y_res)}')
  
  spark.createDataFrame(pd.concat([X_res, y_res], axis=1)) \
    .write.mode("overwrite") \
    .saveAsTable(input_table+'_train_smotenctomek')

# COMMAND ----------


