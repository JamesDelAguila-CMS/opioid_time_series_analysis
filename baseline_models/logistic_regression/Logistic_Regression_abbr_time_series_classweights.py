# Databricks notebook source
# MAGIC %md
# MAGIC ## Logistic Regression 
# MAGIC with Abbreviated Time Series
# MAGIC Using class weights (As opposed to SMOTE)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC TODO: basic regression w/elasticnet, class weights for ohe set, mlflow, hyperopt, crossvalidation, VIF 
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType, FloatType, IntegerType
from hyperopt import fmin, tpe, Trials
import numpy as np
import mlflow
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import matplotlib.pyplot as plt 
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from hyperopt import hp
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn import metrics

# COMMAND ----------

def balanceDataset(df):
    from pyspark.sql.types import DoubleType

    # Re-balancing (weighting) of records to be used in the logistic loss objective function
    numNegatives = df.filter("labels = 1").count()
    datasetSize = df.count()
    balancingRatio = (datasetSize - numNegatives) / datasetSize

    calculateWeights = udf(lambda d: 1 * balancingRatio if (d == 1.0) else (1 * (1.0 - balancingRatio)), DoubleType())

    weightedDataset = df.withColumn("classWeightCol", calculateWeights("labels"))
    return weightedDataset

def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    return udf(to_array_, ArrayType(DoubleType())).asNondeterministic()(col)
  
def Fbeta(beta, precision, recall):
    return (beta*beta + 1)*precision*recall / (beta*beta*precision + recall)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Class Weights

# COMMAND ----------

# MAGIC 
# MAGIC %md
# MAGIC Weighting, see:
# MAGIC https://stackoverflow.com/questions/33372838/dealing-with-unbalanced-datasets-in-spark-mllib

# COMMAND ----------

# Applying weighting to all sets, but not utilizing in val/test
train_data = balanceDataset(spark.sql("SELECT * FROM eldb.opioid_SA_LA_hosp_final_abbr_ftr_extrctn_and_demos_ohe where MOD(bene_id, 20) < 14 and round(TS_MME_ALL_SA__mean, 0) <= 277 and round(TS_MME_ALL_LA__mean, 0) <= 351").drop("CC_Depressive_dis"))
val_data = balanceDataset(spark.sql("SELECT * FROM eldb.opioid_SA_LA_hosp_final_abbr_ftr_extrctn_and_demos_ohe where MOD(bene_id, 20) >= 14").drop("CC_Depressive_dis"))
#test_data = balanceDataset(spark.sql ("SELECT * FROM eldb.opioid_SA_LA_hosp_final_abbr_ftr_extrctn_and_demos_ohe where MOD(bene_id, 20) >= 18"))

# COMMAND ----------

train_data = train_data.withColumn('static_demo_bene_age_tsfrm', F.col('static_demo_bene_age')-65).drop('static_demo_bene_age')
val_data = val_data.withColumn('static_demo_bene_age_tsfrm', F.col('static_demo_bene_age')-65).drop('static_demo_bene_age')

# COMMAND ----------

train_data.cache().count()

# COMMAND ----------

val_data.cache().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Model

# COMMAND ----------

# Assemble features vector
vecAssembler = VectorAssembler(inputCols=[col for col in train_data.columns if col not in ('bene_id', 'labels', 'classWeightCol')], outputCol="features")

# Define logistic regression model as GLM
lr = GeneralizedLinearRegression(labelCol="labels", family='binomial', link='logit', maxIter = 1000, weightCol = 'classWeightCol', tol=.0000000001) # note GLR uses iteratively reweighted least squares. When positive regParam it is RIDGE (L2)

# pipeline assembler/model
pipeline = Pipeline(stages=[vecAssembler, lr])

# define evaluation metric for optimal thresholds
binaryClassificationEvaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="labels", metricName='areaUnderPR')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Hyperopt

# COMMAND ----------

# Objective function and search space for HyperOpt

def objective_function(params):    
  # set the hyperparameters that we want to tune
  regParam = params["regParam"]
  tol = params["tol"]
  #elasticNetParam = params["elasticNetParam"]
  
  with mlflow.start_run():
    estimator = pipeline.copy({lr.regParam: regParam, lr.tol: tol})
    pipelineModel = estimator.fit(train_data)

    preds = pipelineModel.transform(val_data)
    prauc = binaryClassificationEvaluator.evaluate(preds)
    aic = pipelineModel.stages[-1].summary.aic

    prediction_pddf = preds.toPandas()
    fpr, tpr, thresholds = roc_curve(prediction_pddf['labels'], prediction_pddf['prediction'], pos_label=1);
    roc_auc = auc(fpr, tpr)

  return 1-roc_auc

'''
regParam: regularization parameter (>= 0). (default: 0.0) --- this is lambda, which controls the importance of the regularization term
elasticNetParam: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty. (default: 0.0)  
'''

search_space = {
  "regParam": hp.quniform("regParam", .01, .5, .05),
  "tol": hp.quniform("tol", .00000001, .000001, .0000005)
  #"elasticNetParam": hp.quniform("elasticNetParam", 0, .7, .05)
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Model/Get Metrics

# COMMAND ----------

# Manually log 

mlflow.pyspark.ml.autolog(log_models=False)


# Find best value of hyperparameters
num_evals = 20

trials = Trials()
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.RandomState(42))

#best_regParam = .02
#best_tol = 5.5e-7
with mlflow.start_run():
  best_regParam = best_hyperparam["regParam"]
  best_tol = best_hyperparam["tol"]
  estimator = pipeline.copy({lr.regParam: best_regParam, lr.tol: best_tol})#, lr.elasticNetParam: best_elasticNetParam}) 
  
  # Set up cross validation
  paramGrid = ParamGridBuilder() \
  .addGrid(lr.regParam, [best_regParam]) \
  .addGrid(lr.tol, [best_tol]) \
  .build()

  cv = CrossValidator(estimator=estimator, evaluator=binaryClassificationEvaluator, estimatorParamMaps=paramGrid, numFolds=5) 

  # Fit model, Get best
  pipelineModel = cv.fit(train_data)
  pipelineModel = pipelineModel.bestModel
  mlflow.spark.log_model(pipelineModel, "logistic_tsfresh_classweights")
  predDF = pipelineModel.transform(val_data)
  
  # Get precision-recall AUC
  prauc = binaryClassificationEvaluator.evaluate(predDF)
  
  # Find the optimal threshold using geometric mean (note: there is virtually no precision in the model, making it difficult to use this metric for deriving a decent threshold)
  prediction_pddf = predDF.toPandas()
  fpr, tpr, thresholds = roc_curve(prediction_pddf['labels'], prediction_pddf['prediction'], pos_label=1);
  roc_auc = auc(fpr, tpr) # get roc_auc while we're at it
  gmean = np.sqrt(tpr * (1-fpr))
  index = np.argmax(gmean)
  thresholdOpt = round(thresholds[index], ndigits = 8)
  mlflow.log_metric("optimal_gmean_threshold", thresholdOpt) 
  
  # Calculate the elements of the confusion matrix
  TN = predDF.filter(f'prediction < {thresholdOpt} AND labels = 0').count()
  TP = predDF.filter(f'prediction >= {thresholdOpt} AND labels = 1').count()
  FN = predDF.filter(f'prediction < {thresholdOpt} AND labels = 1').count()
  FP = predDF.filter(f'prediction >= {thresholdOpt} AND labels = 0').count()

  try:
    accuracy = (TN + TP) / (TN + TP + FN + FP) 
  except: accuracy = 0
  try:
    precision = TP / (TP + FP)
  except: precision = 0
  try:
    recall = TP / (TP + FN)
  except: recall =0
  try:
    F1 =  Fbeta(1, precision, recall)
  except: F1=0
  try:
    F2 =  Fbeta(2, precision, recall)
  except: F2=0
  
  # Calculate Brier Score
  predDF = predDF.withColumn('difference', F.col('labels') - F.col('prediction'))
  predDF = predDF.withColumn('squared_difference', F.pow(F.col('difference'), F.lit(2).astype(IntegerType())))
  mse = predDF.groupBy().agg(F.avg(F.col('squared_difference')).alias('mse'))
  brier_score = mse.collect()[0][0]

  # Log param and metrics for the final model
  mlflow.log_param("regParam", best_regParam)
  mlflow.log_param("tol", best_tol)
  mlflow.log_metric("pr-auc", prauc)
  mlflow.log_metric("roc-auc", roc_auc)
  mlflow.log_metric("TN", TN)
  mlflow.log_metric("TP", TP)
  mlflow.log_metric("FN", FN)
  mlflow.log_metric("FP", FP)
  mlflow.log_metric("accuracy", accuracy)
  mlflow.log_metric("precision", precision)
  mlflow.log_metric("recall", recall)
  mlflow.log_metric("F1", F1)
  mlflow.log_metric("F2", F2)
  mlflow.log_metric("brier_score", brier_score)
  mlflow.log_metric("intercept_log_odds", pipelineModel.stages[-1].intercept)
  mlflow.log_metric("intercept_probability", np.exp(pipelineModel.stages[-1].intercept)/(1+np.exp(pipelineModel.stages[-1].intercept)))
  mlflow.log_metric("aic", pipelineModel.stages[-1].summary.aic)

  # PR CURVE
  precision, recall, thresholds = precision_recall_curve(prediction_pddf['labels'], prediction_pddf['prediction'], pos_label=1)
  prc_auc = auc(recall, precision) 
  
  pr_curve_fig = plt.figure(figsize=(5,5))
  plt.plot(precision, recall, label='P-R curve (area = %0.2f)' % prc_auc)
  #plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Recall (TP/(TP+FN))')
  plt.ylabel('Precision (TP/(TP+FP))')
  plt.title('Precision-Recall Curve')
  plt.legend(loc="lower right")
  mlflow.log_figure(pr_curve_fig, 'pr_curve_fig.png')
  
  # Confusion Matrixfrom matplotlib.lines import Line2D
  array = [[TN, FP], [FN, TP]]

  df_cm = pd.DataFrame(array, )
  s = plt.figure(figsize=(10,7))

  sn.set(font_scale=1.4) # for label size
  s = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g', cbar=False, cmap="YlGnBu") # font size
  s.set(xlabel='Predicted', ylabel='Actual')

  for t in s.texts:
    t.set_text('{:,d}'.format(int(t.get_text())))
  
  mlflow.log_figure(s.get_figure(), 'confusion-matrix.png')
  
  # Get model stats
  weights = pd.DataFrame(list(zip(pipelineModel.stages[-2].getInputCols(), pipelineModel.stages[-1].coefficients, np.exp(pipelineModel.stages[-1].coefficients)/(1+np.exp(pipelineModel.stages[-1].coefficients)), pipelineModel.stages[-1].summary.pValues, pipelineModel.stages[-1].summary.coefficientStandardErrors)), columns=["feature", "log odds", "probabilities", 'p_values', 'standard_errors']).sort_values('p_values', ascending=True)
  weights.to_html('model_statistics.html')
  mlflow.log_artifact('model_statistics.html')
  
  # brier score for subgroups
  prediction_pddf = prediction_pddf.rename(columns = {'labels':'target', 'prediction':'preds'})
  dataset = prediction_pddf
  brier_scores = {}
  for col in dataset.columns:
    if "STATIC" in col:
      cur_df = dataset[dataset[col] == 1]
      if len(cur_df) > 0:
        preds = cur_df["preds"].to_numpy()
        labels = cur_df["target"].to_numpy()
        brier_score = metrics.brier_score_loss(labels, preds)
        brier_scores[col] = brier_score
      else:
        print(f"no positive examples for {col}")

  preds = dataset["preds"].to_numpy()
  labels = dataset["target"].to_numpy()
  full_brier_score = metrics.brier_score_loss(labels, preds)
  normalized_dict = {}
  for key in brier_scores:
    normalized_dict[f"{key}_normalized"] = full_brier_score/brier_scores[key]
  brier_scores["all"] = full_brier_score
  brier_scores.update(normalized_dict)
  brier_scores = brier_scores
  brierdf = pd.DataFrame.from_dict(brier_scores, orient='index', columns=['brier_score'])
  brierdf.to_html('brier_score_subgroups.html')
  mlflow.log_artifact('brier_score_subgroups.html')

# COMMAND ----------


