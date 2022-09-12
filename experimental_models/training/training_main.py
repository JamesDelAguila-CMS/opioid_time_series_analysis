# Databricks notebook source
try:
    import pytorch_lightning as pl
except:
    !pip config --user set global.index-url https://pypi.ccwdata.org/simple
    !pip install pytorch-lightning
    import pytorch_lightning as pl

# COMMAND ----------

# MAGIC %run ../utils/dataloader_setup

# COMMAND ----------

# MAGIC %run ../utils/model_setup

# COMMAND ----------

np.seterr(invalid='ignore')

# COMMAND ----------


dataset_params = dict(
    train_path = "/dbfs/mnt/eldb_mnt/MMA394/data/train_ts_abr_data.csv",
    val_path = "/dbfs/mnt/eldb_mnt/MMA394/data/val_ts_abr_data.csv",
    
    batch_size = 256, 
    num_workers = 2,
    shuffle = True,
    one_hot_encoding = True,
    drop_ts = False
)

train_dataset = CMSPytorchDataset(
    csv_path=dataset_params["train_path"],
    one_hot_encoding=dataset_params["one_hot_encoding"],
    drop_ts=dataset_params["drop_ts"]
)
val_dataset = CMSPytorchDataset(
    csv_path=dataset_params["val_path"],
    one_hot_encoding=dataset_params["one_hot_encoding"],
    drop_ts=dataset_params["drop_ts"]
)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=dataset_params["batch_size"], 
    num_workers=dataset_params["num_workers"], 
    shuffle=dataset_params["shuffle"],
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=dataset_params["batch_size"], 
    num_workers=dataset_params["num_workers"],
)

# COMMAND ----------

trainer_params = dict(
    gpus = 1,
    max_epochs = 50,
    limit_train_batches = .5
)

trainer = MLFlowTrainer(
    max_epochs=trainer_params['max_epochs'],
    gpus=trainer_params['gpus'],
    limit_train_batches=trainer_params['limit_train_batches'],
)

# COMMAND ----------

train_dataset.input_shape

# COMMAND ----------

model_params = dict(
    weight_decay = 1e-2,
    learning_rate = 3e-4,
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10])),
    dropout = 0.5, 
    layer_sizes=[128, 128, 128, 64, 32, 16, 3],
    input_size=train_dataset.input_shape,
    pred_threshold=.5
)
# Save the actual weight used as a parameter. If not, MLFlow will not 
# explicitly show your declared pos_weight
if isinstance(model_params["criterion"], nn.BCEWithLogitsLoss):
    model_params["pos_weight"] = float(model_params["criterion"].pos_weight)

# Set device based on specified number of gpus
if trainer_params["gpus"] >= 1:
    model_params["device"] = "gpu"
else: 
    model_params["device"] = "cpu"

model = BasicMLP(
    learning_rate = model_params['learning_rate'], 
    dropout=model_params["dropout"],
    layer_sizes=model_params["layer_sizes"],
    criterion=model_params['criterion'],
    weight_decay=model_params["weight_decay"],
    device=model_params["device"],
    input_size=model_params["input_size"],
    pred_threshold=model_params["pred_threshold"]
)
model.cuda()

# COMMAND ----------

"""
    This cell is for resuming training. You will to manually set the corresponding
    architecture parameters - this is usually just layer_sizes, criterion, and input_size.
    Check MLFlow for the correct parameters, they should be saved in the corresponding run.
    It is feasible to load directly from the parameter json that MLFlow saves, but MLFlow
    will save all params as strings, and we deferred writing that particular loading function.
"""
# this is where we're saving the models for persistence
#!ls /dbfs/mnt/eldb_mnt/MMA394/model_storage/
# resume_params = dict(
#     checkpoint_path = "/dbfs/mnt/eldb_mnt/MMA394/model_storage/0a3dfc9761df46a3a73b87c10de2a41f/epoch=49-step=63900.ckpt",
#     criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10])),

# #     "layer_sizes": [128,128,128, 64, 32, 16, 8, 4], 
# #     "criterion": nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([45])),
# #     "input_size": 204
# )

# # This will use model_params but override anything in model_params w/ w/e has been put in resume_params.
# # Extend the loading dictionary.
# resume_kwargs = dict(model_params, **resume_params)
# model = BasicMLP.load_from_checkpoint(**resume_kwargs)

# COMMAND ----------

note = "smote seems like a total failure, retrying w/ normal dataset, slightly weighted loss only."
try:
    param_list = [dataset_params, trainer_params, resume_kwargs]
except:
    param_list = [dataset_params, trainer_params, model_params]
    
experiment = mlflow.set_experiment("/eldb/02_sandbox/kungfu/training_with_pytorch_dataloader_main")
with mlflow.start_run() as run:
    mlflow.log_param("note", note)
    # if not resuming, replace resume_kwargs w/ model_params resume_kwargs
    for params in [dataset_params, trainer_params, model_params]:
        for key, value in params.items():
            mlflow.log_param(key, value)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# COMMAND ----------


