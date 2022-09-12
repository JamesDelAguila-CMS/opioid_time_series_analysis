# Databricks notebook source
# MAGIC %md
# MAGIC ## Install MLFlow
# MAGIC Install and then import the necessary packages.

# COMMAND ----------

from tqdm import tqdm
import mlflow
from mlflow import log_metric

import tempfile
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import numpy as np 
import pandas as pd

import os
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch

from sklearn import metrics

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer, seed_everything

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Code
# MAGIC ### Model setup
# MAGIC The MLP Model is instantiated with a number of variables.
# MAGIC 
# MAGIC learning_rate=.0001  - the learning rate is the manitude at which the gradient is applied to each parameter. 1e-3 is a standard default value. A LR Optimizer can change this. We implemented one that lowers the LR after a plateau in performance during training.
# MAGIC 
# MAGIC dropout=.2 - the percentage of random parameters to ignore during training updates. This helps prevent overfitting, as well as improves generalization. As any parameter could be ignored during at specific training step, dropout forces the model to identify important patterns throughout the parameters, as opposed to routing all the info into certain paths.
# MAGIC 
# MAGIC layer_sizes=[128,128,128,64,32,16,8,4] - A list that the model uses to instantiate the dense layers. 
# MAGIC 
# MAGIC criterion=None - This is the loss function - for us, generally BCELoss or BCELossWithLogits, as this is a binary classification problem. BCELWL expects no sigmoid applied to the last layer, so there is some code checking for that internally, scouting for the BCELoss class specifically. Both of those loss functions are the same underlying formula - Binary Cross Entropy - but BCELossWithLogits makes it a bit easier to upweight positive samples programatically. 
# MAGIC 
# MAGIC device="cpu" - "cpu" or "gpu" to direct the model to what hardware to use. GPUs speed up training by orders of magnitude. 
# MAGIC 
# MAGIC weight_decay=0 - the coefficient for weight decay, which is implemented programatically in the Adam Optimizer
# MAGIC 
# MAGIC input_size=149 - the size of the feature vector input into the model. 

# COMMAND ----------

class BasicMLP(pl.LightningModule):
    def __init__(self, learning_rate=.0001, dropout=.2, layer_sizes=[128,128,128,64,32,16,8,4], criterion=None, device="cpu",weight_decay=0, input_size = 149, pred_threshold = .5):
        super(BasicMLP, self).__init__()
        self.dvc = device if device == "cpu" else "cuda"
        self.input_size = input_size
        self.big_preds = []
        self.hidden_act = nn.ReLU()
        self.hidden_drop = nn.Dropout(p=dropout)    
        self.fc_layers = self.init_fc_layers(layer_sizes)
        self.out = nn.Linear(layer_sizes[-1], 1, self.dvc)        
        self.out_activation = torch.nn.Sigmoid()
        self.criterion = criterion
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.pred_threshold = pred_threshold
  
    def init_fc_layers(self, layer_sizes):   
        fc_layers = [nn.Linear(self.input_size, layer_sizes[0], device=self.dvc)]
        for i in range(len(layer_sizes)):
          if i == 0:
            continue
          print(f"Layer {i} created with dimensions {layer_sizes[i-1]},{layer_sizes[i]}")
          fc_layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i], device=self.dvc))
        return nn.ModuleList(fc_layers)
      
    def needs_sigmoid(self):
        # BCELoss need sigmoid, but BCELossWithLogits does not
        return isinstance(self.criterion, nn.BCELoss)
      
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # TODO : change these to be parameterizable
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=3, verbose=True)
        return {"optimizer":optimizer, 
                "lr_scheduler":
                    {
                        "scheduler": lr_scheduler,
                        "monitor": "train_loss",
                        "frequency": 1
                    }
               }
      
    def forward(self, input):
        e = input
        for fc in self.fc_layers[:-1]:
            e = self.hidden_drop(self.hidden_act(fc(e)))
        e = self.hidden_act(self.fc_layers[-1](e))
        if self.needs_sigmoid() or not self.training:
          result = self.out_activation(self.out(e).squeeze(-1)) 
        else:
          result = self.out(e).squeeze(-1)
        return result
      
    def training_step(self, batch):
        """
          The code for a training step. It's meant to be extremely similar to the validation step.
          Note: this function needs to return either the raw loss value, or a dictionary with a "loss" entry.
          This is an overwrite of a PTL function, and is required for PTL to work properly.
        """
        ts, labels = batch
        preds = self.forward(ts.float())
        labels = labels.float()
        metric_dict = self.calculate_metrics(preds, labels)
        self.log_values("train", metric_dict, labels, preds)
        return metric_dict

    def validation_step(self, batch, batch_idx):
        """
          The code for a validation step. It's meant to be extremely similar to the training step.
          Note: this function needs to return either the raw loss value, or a dictionary with a "loss" entry.
          This is an overwrite of a PTL function, and is required for PTL to work properly.
        """
        ts, labels = batch
        preds = self.forward(ts.float())
        labels = labels.float()
        metric_dict = self.calculate_metrics(preds, labels)
        self.log_values("val", metric_dict, labels, preds)
        return metric_dict
      
    def calculate_metrics(self, preds, labels):
        """
          A function to process and accumulate metrics from predictions and labels. 
        """
        metric_dict = {}
        # Take the loss - if sigmoid needed to be applied, it will have already happened in the forward pass
        metric_dict["loss"] = self.criterion(preds, labels)
        # Apply sigmoid if needed
        if self.needs_sigmoid():
          local_preds = [pred for pred in torch.sigmoid(preds).cpu().tolist()]
        else:
          local_preds = [pred for pred in preds.cpu().tolist()]
        
        
        # Set threshold so that the top self.pred_threshold proportion of predictions are predicted as true
        lp = local_preds
        lp.sort(reverse = True)
        threshold = lp[ round(len(lp)*self.pred_threshold)]
        
        non_tensor_preds = []
        # Round to turn predictions into labels
        for pred in local_preds:
          if pred < threshold:
            pred = 0.0
          else:
            pred = 1.0
          non_tensor_preds.append(pred)
#         non_tensor_preds = [min(round(max(pred,0)), 1) for pred in local_preds]
        # Put tensor on cpu and cast to numpy, as tensors are not processable by sklearn 
        non_tensor_labels = labels.cpu().tolist()
        self.big_preds.extend(non_tensor_preds)
        metric_dict["f1"] = metrics.f1_score(non_tensor_labels, non_tensor_preds, zero_division=0, pos_label=1.0, average="binary")
        metric_dict["precision"] = metrics.precision_score(non_tensor_labels, non_tensor_preds, zero_division=0, pos_label=1.0, average="binary")
        metric_dict["recall"] = metrics.recall_score(non_tensor_labels, non_tensor_preds, zero_division=0, pos_label=1.0, average="binary")
        metric_dict['pr_auc'] = metrics.average_precision_score(non_tensor_labels, local_preds, pos_label=1.0, average="micro")
        return metric_dict
  
    def log_values(self, train_or_val, metric_dict, labels, preds):
        # log all metrics in mlflow and ptl logger 
        for key, value in metric_dict.items(): 
            if key == "loss":
                self.log(f"{train_or_val}_{key}",float(value),prog_bar=True)
            else:
                self.log(f"{train_or_val}_{key}",float(value))
            log_metric(f"{train_or_val}_{key}",float(value))
        
    def gather_mean(self, dict_key, val_dicts):
        metric_list = np.stack([batch[dict_key] for batch in val_dicts if dict_key in batch])
        mean = metric_list.mean()
        log_metric(f"val_epoch_end_{dict_key}", mean)
    
    def validation_epoch_end(self, val_outputs):
        """
          Val outputs is a list of all validation_step outputs. Each Element will be the output of one validation batch
        """
        self.gather_mean("f1", val_outputs)
        self.gather_mean("precision", val_outputs)
        self.gather_mean("recall", val_outputs)
        self.gather_mean("pr_auc", val_outputs)

# COMMAND ----------

import shutil
class MLFlowTrainer(Trainer):
    """
      MLFlowTrainer - a class to imitate a PTL trainer, but also save a model to a persistent location when it logs a checkpoint artifact
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_checkpoint(self, filepath, weights_only):
        # we still want all the normal functionality of the ptl save_checkpoint function
        super().save_checkpoint(filepath, weights_only)
        run_id = mlflow.active_run().info.run_id
        basename = os.path.basename(filepath)
        # derive the path to save the checkpoint
        run_dir = f"/dbfs/mnt/eldb_mnt/MMA394/model_storage/{run_id}" 
        try:
          os.makedirs(run_dir, exist_ok=True)
          if os.path.isfile(filepath):
              # log the model checkpoint as an MLflow object, then copy it over to the new location
              mlflow.log_artifact(local_path=filepath) 
              shutil.copyfile(filepath, f"{run_dir}/{basename}")
          else:
              print(f"Could not find {filepath} when saving checkpoint")
        except Exception as e:
          # we were troubleshooting our training run failures and were considering
          # that saving directly to the filesytem could be messing with things. It
          # did not seem to be the case, tho we left the try/catch in as we've
          # encountered other errors when saving the model before.
          print(f"couldn't save model to: {filepath}")
          print(f"Error: {e}")

# COMMAND ----------

# !ls /dbfs/mnt/eldb_mnt/MMA394/model_storage/e370befd10604f92a4ab91a75b346cb9

# COMMAND ----------


