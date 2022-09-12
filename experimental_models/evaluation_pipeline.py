# Databricks notebook source
from sklearn import metrics
from tqdm import tqdm
import pandas as pd
import json
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve,PrecisionRecallDisplay
from matplotlib import pyplot as plt

class EvalPipeline:
  """
    This class serves to gather predictions from a model on a datasets, then evaluates 
  """
  def __init__(self, model, dataset=None, dataset_collate_fn=None):
    self.model = model
    self.subgroup_cols = ['STATIC_DEMO_bene_age', 'STATIC_DEMO_sex_label', 'STATIC_DEMO_state_cd', 'STATIC_DEMO_race_label', 'STATIC_DEMO_crec_label', 'STATIC_DEMO_RUCA1']
    self.brier_scores = {subgroup: {} for subgroup in self.subgroup_cols}
    self.preds = []
    self.dataset = dataset
    self.convert_row_to_model_input = dataset_collate_fn
    self.ground_truth_column_name = "target" if "target" in self.dataset.columns else "labels"

  def run_model(self):
    print(f"Total len of dataset: {len(self.dataset)}")
    for idx, data in tqdm(self.dataset.iterrows()):
      x, y = self.convert_row_to_model_input(data)
      preds = self.model.forward(x)
      self.preds.append(float(preds))
    
  def amend_predictions_to_data(self):
    self.dataset["preds"] = self.preds
    self.dataset.to_csv("eval_outputs.csv", index=False)
      
  def evaluate_bias(self):
    brier_scores = {}
    for col in self.dataset.columns:
      if "STATIC" in col:
        cur_df = self.dataset[self.dataset[col] == 1]
        if len(cur_df) > 0:
          preds = cur_df["preds"].to_numpy()
          labels = cur_df[self.ground_truth_column_name].to_numpy()
          brier_score = metrics.brier_score_loss(labels, preds)
    #       f1 = metrics.f1_score(labels,np.rint(preds))
    #       precision = metrics.precision_score(labels, np.rint(preds))
    #       recall = metrics.precision_score(labels, np.rint(preds))
          brier_scores[col] = brier_score
        else:
          print(f"no positive examples for {col}")
      
    preds = self.dataset["preds"].to_numpy()
    labels = self.dataset[self.ground_truth_column_name].to_numpy()
    full_brier_score = metrics.brier_score_loss(labels, preds)
    normalized_dict = {}
    for key in brier_scores:
      normalized_dict[f"{key}_normalized"] = brier_scores[key]/full_brier_score
    brier_scores["all"] = full_brier_score
    brier_scores.update(normalized_dict)
    self.brier_scores = brier_scores

  def save_report(self):
    self.dataset.to_csv("dataset_with_predictions.csv")
    with open('brier_scores.json', 'w') as fp:
      json.dump(self.brier_scores, fp)
    print("yo we saved the report we rule!!")
  
  def plot_pr_curve(self):
    y_true = self.dataset[self.ground_truth_column_name]
    preds = self.dataset['preds']

    precision, recall, thresholds = precision_recall_curve(y_true,preds )
    
    print("Precision, Recall, Threshold")
    for p,r,t in zip(precision, recall, thresholds):
      print(f"{p},{r},{t}")
    plt.plot( recall,precision)
    plt.title("Precision Recall Curve")
    plt.ylabel("precision")
    plt.xlabel("recall")
    
  def confusion_matrix(self,threshold = None):
    y_true = self.dataset[self.ground_truth_column_name]
    preds = self.dataset['preds']
    confusion_matrix = metrics.confusion_matrix(y_true, preds>np.percentile(preds,0.99))
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix)
    disp.plot()
    plt.show()
    
  def error_analysis(self):
    df = self.dataset
    df_pos = df[df["target"] == 1]
    self.correlations = df_pos.corr()['preds'].sort_values().dropna()
    print(self.correlations)
    
    for idx, c in self.correlations.items():
      print(idx, c)
      
  def run(self):
#     self.init_data()
    self.run_model()
    self.amend_predictions_to_data()
    self.evaluate_bias()
    self.save_report()

# COMMAND ----------

try:
    import pytorch_lightning as pl
except:
    !pip config --user set global.index-url https://pypi.ccwdata.org/simple
    !pip install pytorch-lightning
    import pytorch_lightning as pl

# COMMAND ----------

# MAGIC %run ./utils/model_setup

# COMMAND ----------

# MAGIC %run ./utils/dataloader_setup

# COMMAND ----------

# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/ee87f388c3244c44a7361a86f52168a3/epoch=6-step=24976.ckpt"
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/38509948b29348a88d6daf84c07929d4/epoch=1-step=3568.ckpt"
#resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/c198121ba8ad4d57b3c5e1ca39ad9618/epoch=26-step=48168.ckpt"
#resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/b4426b6e19ce46029e14fc8f5d0347d2/epoch=6-step=12488.ckpt"



#no predictions
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/1ff1d49022724dde98d8007cee634afc/epoch=20-step=18795.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([45])), layer_sizes = [128, 64, 32, 16, 8, 4], input_size = 149)

# F1-score 0.011
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/0454a985a5b44844a175d9eff544d531/epoch=19-step=17900.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([45])), layer_sizes = [128,128,128, 64, 32, 16, 8, 4], input_size = 349)

# F1-score 0.002
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/ef0fec0e56374ebaa69c2a16ce6b2f9a/epoch=9-step=8950.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([45])), layer_sizes = [128,128,128, 64, 32, 16, 8, 4], input_size = 349)

# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/ef0fec0e56374ebaa69c2a16ce6b2f9a/epoch=9-step=8950.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([45])), layer_sizes = [128,128,128, 64, 32, 16, 8, 4], input_size = 349)

# F1-score 0.01
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/29a863e8d3c94abcb2f7a7f39c91a931/epoch=19-step=17900.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([45])), layer_sizes = [128, 64, 32, 16, 8, 4], input_size = 349)

#no predictions
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/4bc34e93b15944f18b071db6b50a993d/epoch=29-step=26850.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, criterion = nn.BCELoss(), layer_sizes = [128,128,128, 64, 32, 16, 8, 4], input_size = 349)

#no predictions
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/1ff1d49022724dde98d8007cee634afc/epoch=20-step=18795.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([45])), layer_sizes = [128, 64, 32, 16, 8, 4], input_size = 149)

# max - idk what this one resulted in
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/84fa99cf5ed147d48ff28939ce0fab12/epoch=6-step=6265.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([45])), layer_sizes = [256, 256, 128, 128, 128, 64, 32, 16, 8, 4], input_size = 349)

# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/644d70cabaa04776948bfc75225a0166/epoch=13-step=12530.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([45])), layer_sizes = [128, 128, 128, 64, 32, 16, 8, 4], input_size = 349)

# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/09fa125bfe884d9e8e88ffe98367d56d/epoch=49-step=89250.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, input_size=150, layer_sizes=[128,64,4])

#Average precision = 0.0034
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/93ea04d95ea645228f5fd4ffd29aff59/epoch=29-step=9600.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, input_size=204, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([450])),layer_sizes=[128, 128, 128, 64, 32, 16, 3])

# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/ac50cfc2abf54fc3883b5c499d856046/epoch=36-step=11840.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, input_size=204, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([450])),layer_sizes=[128, 128, 128, 64, 32, 16, 3])

# strong model from expt 1
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/167b9bcfd82e47c38712a4ceb6792b1b/epoch=49-step=8000.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, input_size=204, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([13.338265419006348])),layer_sizes=[192, 192, 128, 128, 64, 32, 16, 3])

# PR_AUC = 0.0037
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/d8822bdb41cc41739c92034a701b3b7c/epoch=28-step=4640.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, input_size=204, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([450])),layer_sizes=[128, 128, 128, 128, 128, 64, 32, 16, 3])

#best thus far 3:22pm 7/26
# PR = 0.0052
resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/167b9bcfd82e47c38712a4ceb6792b1b/epoch=49-step=8000.ckpt"
model = BasicMLP.load_from_checkpoint(resume_path, input_size=204, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([450])),layer_sizes=	[192, 192, 128, 128, 64, 32, 16, 3])

# PR = 0.0033
# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/460ef6fe6d184796aab4b33e52c103ce/epoch=49-step=8000-v1.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, input_size=204, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([450])),layer_sizes=	[128, 128, 128, 128, 64, 32, 16, 3])

# resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/460ef6fe6d184796aab4b33e52c103ce/epoch=20-step=3360-v1.ckpt"
# model = BasicMLP.load_from_checkpoint(resume_path, input_size=204, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([450])),layer_sizes=	[128, 128, 128, 128, 64, 32, 16, 3])

def dataset_collate_fn(row):
  if "labels" in row.index:
    target_col = "labels"
  else:
    target_col = "target"
  y = row[target_col]
  row.drop(labels=[target_col], inplace=True)
  x = torch.Tensor(row.to_numpy())
  return x, y

# COMMAND ----------

# from torch.utils.data import DataLoader
# preds = []
# labels = []
# dataset = CMSPytorchDataset(csv_path="/dbfs/mnt/eldb_mnt/MMA394/data/test_ts_data.csv")
# dataloader = DataLoader(dataset, batch_size = 32,num_workers = 4)
# for input, label in dataloader:
# #  print(input.shape)
#   preds.extend(model.forward(input.float()).detach().numpy())
#   labels.extend(label.numpy())
# #  print(pred.shape)

# preds = np.array(preds).ravel()
# labels = np.array(labels).ravel()

# COMMAND ----------


# preds_disc = preds
# thresh = 0.97
# print(f"Mean prediction is {np.mean(preds)}")
# preds_disc[preds_disc>=thresh] = 1
# preds_disc[preds_disc<thresh] = 0

# print(metrics.classification_report(labels, preds_disc))
# print(metrics.f1_score(labels, preds_disc, average = "binary"))

# plt.hist(preds, bins = 100)

# confusion_matrix = metrics.confusion_matrix(labels, preds>=thresh)
# disp = metrics.ConfusionMatrixDisplay(confusion_matrix)
# disp.plot()
# plt.show()

# precision, recall, thresholds = precision_recall_curve(labels,preds )
# plt.plot( recall,precision)
# plt.title("PR Curve")
# plt.ylabel("precision")
# plt.xlabel("recall")

# COMMAND ----------

dataset = CMSPytorchDataset(csv_path="/dbfs/mnt/eldb_mnt/MMA394/data/eval_ts_abr.csv", one_hot_encoding=True)
full_df = dataset.get_full_dataset_as_df()
# full_df = full_df.drop(["bene_id"],axis=1)
#full_df = full_df.head(100000)
model.eval()
eval_pipeline = EvalPipeline(model, dataset=full_df, dataset_collate_fn = dataset_collate_fn)
eval_pipeline.run()

# COMMAND ----------

eval_pipeline.plot_pr_curve()
eval_pipeline.confusion_matrix()
eval_pipeline.dataset["preds"].hist(bins=100)


# COMMAND ----------

df = eval_pipeline.dataset
trues = eval_pipeline.dataset['target'].to_numpy()
preds = eval_pipeline.dataset['preds'].to_numpy()

preds_disc = preds.copy()
thresh = np.percentile(preds,0.99)
#thresh = 0.4980
preds_disc[preds_disc>thresh] = 1
preds_disc[preds_disc<=thresh] = 0

print(metrics.classification_report(trues, preds_disc))
print(metrics.f1_score(trues, preds_disc, average = "binary"))
print(metrics.average_precision_score(trues, preds, average = None))

# COMMAND ----------

eval_pipeline.brier_scores

# COMMAND ----------

df = eval_pipeline.dataset
df_pos = df[df["target"] == 1]
correlations = df_pos.corr()['preds'].sort_values().dropna()

for idx, c in correlations.items():
  print(idx, c)

# COMMAND ----------

def plot_preds_correlation(df = df_pos,input_var = "TS_MME_ALL_SA__count_above__t_90"):
  df.plot.scatter('preds',input_var, title = f"Effect of {input_var} on model predictions")
  
plot_preds_correlation()

# COMMAND ----------

plot_preds_correlation(df,'TS_MME_ALL_SA__count_above__t_90')

# COMMAND ----------

plot_preds_correlation(df_pos,'TS_MME_ALL_SA__standard_deviation') 
plot_preds_correlation(df_pos,'TS_MME_ALL_LA__standard_deviation')

# COMMAND ----------

plot_preds_correlation(df_pos,'TS_MME_ALL_SA__longest_strike_above_mean') 
plot_preds_correlation(df_pos,'TS_MME_ALL_SA__linear_trend__attr_slope') 

# COMMAND ----------

y_true = eval_pipeline.dataset[eval_pipeline.ground_truth_column_name]
preds = eval_pipeline.dataset['preds']
thresh = 0.05
print(thresh)
confusion_matrix = metrics.confusion_matrix(y_true, preds>thresh)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix)
disp.plot()
plt.show()
print(confusion_matrix)
