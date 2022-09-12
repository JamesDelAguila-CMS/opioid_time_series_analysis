# Databricks notebook source
from sklearn.preprocessing import OneHotEncoder
import re, string, timeit

def remove_punctuation(s):
  s = s.replace(" ", "_").lower()
  s = s.replace("-", "_")
  return re.sub(r'\W+', '', s)

class CMSOneHotEncoder:
    def __init__(self, data):
        self.category_info = {
          'STATIC_DEMO_crec_label': ['Old age and survivor’s insurance (OASI)', 'End-stage renal disease (ESRD)', 'Disability insurance benefits (DIB)'],
          'STATIC_DEMO_race_label': ['Black or African-American', 'Asian / Pacific Islander','Unknown','Other','Hispanic','American Indian / Alaska Native','Non-Hispanic White'],
          'STATIC_DEMO_sex_label': ['Female', 'Male'],
          'STATIC_DEMO_state_cd': ['SC','AZ','LA','MN','NJ','DC','OR','VA','RI','KY','WY','NH','MI','NV','WI','ID','CA','CT','NE','MT','NC','VT','MD', 'DE', 'MO',         'VI','IL','ME','ND','WA','MS','AL','IN','OH','TN','IA','NM','PA','SD','NY','TX','WV','GA','MA','KS','FL','CO','AK','AR','OK','PR','UT','HI','Unassigned'],
          'STATIC_DEMO_RUCA1': ['1','4','2','7','3','9','10','8','6','5'],
          'STATIC_DEMO_ADI_NATRANK_binned': ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','Unassigned']
                             }
        self.category_list = [self.category_info["STATIC_DEMO_sex_label"], self.category_info["STATIC_DEMO_state_cd"], self.category_info["STATIC_DEMO_race_label"], self.category_info["STATIC_DEMO_crec_label"], self.category_info["STATIC_DEMO_RUCA1"],self.category_info["STATIC_DEMO_ADI_NATRANK_binned"]]
        
        self.encoder = OneHotEncoder(categories=self.category_list)
        self.encoder.fit(data)
        
    def get_one_hots(self, input_data):
        return self.encoder.transform(input_data).toarray()
    
    def get_one_hots_df(self, input_data):
        # This order is specific for the list above
        col_names = []
        for dict_key in ["STATIC_DEMO_sex_label", "STATIC_DEMO_state_cd", "STATIC_DEMO_race_label", "STATIC_DEMO_crec_label","STATIC_DEMO_RUCA1","STATIC_DEMO_ADI_NATRANK_binned"]:
          cur_category_list = self.category_info[dict_key]
          cur_category_list = map(remove_punctuation, cur_category_list)
          categories_to_add = [f"{dict_key}_{sub_cat}"for sub_cat in cur_category_list]
          col_names.extend(categories_to_add)

        transformed_data = self.encoder.transform(input_data).toarray()
        return pd.DataFrame(data=transformed_data, columns=col_names)

# COMMAND ----------

import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class CMSPytorchDataset(Dataset):
    def __init__(self, csv_path = "csv_path.csv"):
        self.df = pd.read_csv(csv_path)
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
        self.data = np.concatenate((self.df.to_numpy(), categorical_df_as_numpy), axis=1).astype(float)
        
    def __getitem__(self, index):
        features = self.data[index]
        label = self.labels[index]
        return features, label

    def __len__(self):
        return len(self.df)

# COMMAND ----------


