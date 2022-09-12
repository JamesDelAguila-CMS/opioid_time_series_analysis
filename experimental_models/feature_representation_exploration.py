# Databricks notebook source
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

class CMSFeatureExtractor(BasicMLP):
  """
    Returns the representation of a trained BaseMLP from before the sigmoid layer
  """
  def __init__(self, *args, **kwargs):
    # Keep everything the same as BasicMLP
    super().__init__(*args, **kwargs)

  def forward(self, input):
    e = input
    for fc in self.fc_layers:
        e = self.hidden_drop(self.hidden_act(fc(e)))
    # Do not use final layer, return intermediate result.
    return e 

# COMMAND ----------

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class FeatureAggregator:
  def __init__(self, feature_extractor, dataset):
    self.feature_extractor = feature_extractor
    # Put pytorch model in eval mode 
    if self.feature_extractor.training:
      self.feature_extractor.eval()
    self.dataset = dataset
    self.negs = []
    self.poses = []
    self.pos_colors = []
    self.neg_colors = []
                       
  def get_representations(self):
    # disable gradients which take up a lot of extra RAM
    with torch.no_grad():
      for input_data, label in tqdm(self.dataset):
        extracted_features = self.feature_extractor.forward(torch.Tensor(input_data))
        # The feature extractor should automatically exclude the layer of size (x, 1)
        # In all our final experiments, x=3, so the final layer of (3,1) has been removed
        df_row = extracted_features.numpy().tolist()
        df_row.append(label)
#         df_row.extend(input_data)
#         print(label)
        if label == 1.0:
          self.pos_colors.append('red')
          self.poses.append(df_row)
        else:
          self.neg_colors.append('blue')
          self.negs.append(df_row)
        
  def construct_df(self):
    col_names = ["out1", "out2", "out3", "label"]
    # This code is commented. At first, we wanted to be able to use any dataset, including ones w/ and w/o TS, Smote, etc
    # Unfortunately there is a little bit of uplift to properly include all the input data
    # and have the columns correctly labeled as well. Given that this visualization is 
#     col_names.extend(list(self.dataset.df.columns))
    self.neg_df = pd.DataFrame(self.negs, columns=col_names)
    self.pos_df = pd.DataFrame(self.poses, columns=col_names)
    
  def generate_visualization(self, pos_or_neg="pos"):
    # TODO: if output_shape < 3, run umap (update: umap is not available on VRDC. TSNE from sklearn should be fine
    # though some writeups imply umap does a better job of preserving high dimensional structure than 
    # t-sne https://pair-code.github.io/understanding-umap/)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if pos_or_neg == "pos":
      df = self.pos_df
      colors = self.pos_colors
    else:
      df = self.neg_df
      colors = self.neg_colors
    ax.scatter3D(df["out1"], df["out2"], df["out3"], color=colors, alpha = 0.1) # cmap='viridis',
    
  def run(self):
    self.get_representations()
    self.construct_df()
    self.generate_visualization('pos')
    self.generate_visualization('neg')

# COMMAND ----------

import plotly.express as p
pd.set_option("plotting.backend", "matplotlib")
fig = px.scatter_3d(aggregator.df, x='out1', y='out2', z='out3',
              color='label')
fig.show()

# COMMAND ----------

resume_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/167b9bcfd82e47c38712a4ceb6792b1b/epoch=49-step=8000.ckpt"
model = CMSFeatureExtractor.load_from_checkpoint(resume_path, input_size=204, criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([450])),layer_sizes=	[192, 192, 128, 128, 64, 32, 16, 3])
dataset = CMSPytorchDataset(csv_path="/dbfs/mnt/eldb_mnt/MMA394/data/test_ts_abr_data.csv", one_hot_encoding=True)

# COMMAND ----------

aggregator = FeatureAggregator(model, dataset)
aggregator.run()

# COMMAND ----------


