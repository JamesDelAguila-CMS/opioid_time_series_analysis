# Databricks notebook source
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import re, string, timeit

def remove_punctuation(s):
  """
    Helper function to remove punctuation from category names so that they play nice with being columns 
  """
  s = s.replace(" ", "_").lower()
  s = s.replace("-", "_")
  return re.sub(r'\W+', '', s)

class CMSOneHotEncoder:
    """
      A class dedicated to reliably and repeatably one hot encoding the categorical features in the eldb.opioid_SA_LA_hosp_sktime_table.
      DO NOT EDIT THE ORDER OF THIS LIST OR DICTIONARY WITHOUT CONSULTING OTHERS ON THE PROJECT
      DO NOT EDIT THE ORDER OF THIS LIST OR DICTINOARY UNLESS YOU ARE PREPARED TO THROW AWAY ALL PAST MODELS
        Specifically, self.category_list and the list on line 50 are used to order the output vectors on line 40.
        More context: The order of input features to the model must be consistent before and after training. The model will interpret 
        your data in the same paradigm you trained it in, so if you change the feature order after training and feed it in, it will
        essentially be nonesense to the trained model. Hence, keeping the ordering consistent here is important. If you change this order
        and retrain, you are in the clear, but any previously trainied model would not work with the new code. 
        This ordering does not occur anywhere else in the code - ie you don't need to worry about this ordering problem anywhere 
        outside of this class. If you change the ordering here and reload your dataset, that is sufficient. 
        
      Uses sklearn's OneHotEncoder class
    """
    def __init__(self, data):
        self.category_info = {
          'STATIC_DEMO_crec_label': ['Old age and survivor’s insurance (OASI)', 'End-stage renal disease (ESRD)', 'Disability insurance benefits (DIB)'],
          'STATIC_DEMO_race_label': ['Black or African-American', 'Asian / Pacific Islander','Unknown','Other','Hispanic','American Indian / Alaska Native','Non-Hispanic White'],
          'STATIC_DEMO_sex_label': ['Female', 'Male'],
          'STATIC_DEMO_state_cd': ['SC','AZ','LA','MN','NJ','DC','OR','VA','RI','KY','WY','NH','MI','NV','WI','ID','CA','CT','NE','MT','NC','VT','MD', 'DE', 'MO',         'VI','IL','ME','ND','WA','MS','AL','IN','OH','TN','IA','NM','PA','SD','NY','TX','WV','GA','MA','KS','FL','CO','AK','AR','OK','PR','UT','HI','Unassigned'],
          'STATIC_DEMO_RUCA1': ['1','4','2','7','3','9','10','8','6','5'],
          'STATIC_DEMO_ADI_NATRANK_binned': ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','Unassigned']
                             }
        self.category_list = [
          self.category_info["STATIC_DEMO_sex_label"], 
          self.category_info["STATIC_DEMO_state_cd"], 
          self.category_info["STATIC_DEMO_race_label"], 
          self.category_info["STATIC_DEMO_crec_label"], 
          self.category_info["STATIC_DEMO_RUCA1"],
          self.category_info["STATIC_DEMO_ADI_NATRANK_binned"]
        ]
        self.encoder = OneHotEncoder(categories=self.category_list)
        self.encoder.fit(data)
        
    def get_one_hots(self, input_data):
        return self.encoder.transform(input_data).toarray()
    
    def get_one_hots_df(self, input_data):
        # This order is specific for the list above
        col_names = []
        # This for loop constructs the proper column names for each OHE'd column
        for dict_key in ["STATIC_DEMO_sex_label", "STATIC_DEMO_state_cd", "STATIC_DEMO_race_label", "STATIC_DEMO_crec_label","STATIC_DEMO_RUCA1","STATIC_DEMO_ADI_NATRANK_binned"]:
          cur_category_list = self.category_info[dict_key]
          cur_category_list = map(remove_punctuation, cur_category_list)
          categories_to_add = [f"{dict_key}_{sub_cat}"for sub_cat in cur_category_list]
          col_names.extend(categories_to_add)

        transformed_data = self.encoder.transform(input_data).toarray()
        return pd.DataFrame(data=transformed_data, columns=col_names)

# COMMAND ----------

import random
from torch.utils.data import Dataset, DataLoader

class CMSPytorchDataset(Dataset):
    """
      This class inherits from torch's Dataset class, and performs final step data prep, like normalization and OHEing
    """
    def __init__(self, csv_path = "csv_path.csv", one_hot_encoding = True, drop_ts=False):
        self.df = pd.read_csv(csv_path)
        if "target" in self.df.columns:
          self.target_col_name = "target"
        else:
          self.target_col_name = "labels"
        self.cat_normalization = {"STATIC_DEMO_bene_age": lambda x: x/100}
        
        self.normalize_cols()
        self.remove_cols()
        self.drop_ts = drop_ts
        self.one_hot_encoding = one_hot_encoding
        if self.drop_ts:
          self.drop_ts()
        if self.one_hot_encoding:
          self.run_one_hot_encoding()
        else:
          self.set_data()
          
        self.input_shape = self.data.shape[1]
          
    def normalize_cols(self):
        # Perform normalization. If additional features were added that needed normalization, 
        # one could add the corresponding normalization function to the dictionary above.
        for column, norm_func in self.cat_normalization.items():
          if column in self.df.columns:
            self.df[column] = self.df[column].apply(norm_func)

    def remove_cols(self):
        # Drop rows with nan values, if any were to ever appear
        self.df.dropna(inplace=True, axis=0)

        # Save but drop the labels and bene_ids. They both will be in the same order as the feature
        # ie self.labels[i] for some valid i corresponds to self.bene_ids[i] and self.df.iloc[i]

        self.labels = self.df[self.target_col_name].to_numpy().astype(float)
        self.df.drop(columns=[self.target_col_name], axis=1, inplace=True)
        
        if "bene_id" in self.df.columns:
          self.bene_ids = self.df["bene_id"].to_numpy().astype(float)
          self.df.drop(columns=["bene_id"], axis=1, inplace=True)
         
    def drop_ts(self):
      print('Removing columns...')
      for column in self.df.columns:
        if "TS_" in column:
          print(f"Removing {column}")
          self.df.drop(columns=[column], axis=1, inplace=True)
        
    def run_one_hot_encoding(self):
        """
          Run the One Hot Encoder and assign self.data the full dataset
        """
        categorical_cols = [column for column in self.df.columns if "STATIC" in column]
        self.categorical_df = self.df[categorical_cols].astype(str)
        self.df.drop(columns=categorical_cols, inplace=True)
        
        self.encoder = CMSOneHotEncoder(self.categorical_df.to_numpy())
        categorical_df_as_numpy = self.encoder.get_one_hots(self.categorical_df.to_numpy())
        self.data = np.concatenate((self.df.to_numpy(), categorical_df_as_numpy), axis=1).astype(float)
        
    def set_data(self):
      self.data = self.df.to_numpy()
        
    def get_full_dataset_as_df(self):
        """
          Return the dataset as a dataframe with One Hot Encoded categorical features included.
          The column names will contain both the cateogrical group and specific label separated by an underscore.
          Dataframes are slightly slower to access than numpy arrays, hence using numpy in self.data
        """
        print(self.one_hot_encoding)
        if self.one_hot_encoding:
          ohe_df = self.encoder.get_one_hots_df(self.categorical_df.to_numpy())
          labels = pd.DataFrame(data=self.labels,columns=[self.target_col_name])
          return pd.concat([self.df, ohe_df, labels], axis=1)
        else:
          labels = pd.DataFrame(data=self.labels,columns=[self.target_col_name])
          return pd.concat([self.df, labels], axis=1)

    def __getitem__(self, index):
        """
          A pythonic function for using the dataloader as both
          an iterator and to index it like one would a list (ie dataset[0])
        """
        features = self.data[index]
        label = self.labels[index]
        return features, label

    def __len__(self):
        """
          A pythonic function to allow len(dataset)
        """
        return len(self.df)
