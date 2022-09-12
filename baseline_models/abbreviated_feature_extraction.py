# Databricks notebook source
# MAGIC %md
# MAGIC # Abridged Features
# MAGIC Author: James DelAguila (CMS)
# MAGIC 
# MAGIC This notebook builds a set of abridged time-series features from time-series, then:
# MAGIC 
# MAGIC 1. Eliminates features that fail F-test with p > 0.05
# MAGIC 2. When Pearson correlation is above threshold value (e.g., .8), the feature with the highest F-statistic is chosen

# COMMAND ----------

# Set these parameters

# The original input from the preprocess_timeseries process
original_data = 'eldb.opioid_SA_LA_hosp_sktime_table'

# results from ANOVA process
anova_table_loc = 'dua_000000_jde328.opioid_abbr_ftr_extrctn_anova_062922'

# final data table after time series feature selection and imputation steps
final_table_loc = 'eldb.opioid_SA_LA_hosp_final_abbr_ftr_extrctn_and_demos'

# threshold of correlation between tsfresh metrics to initiate selection process
correlation_cutoff = 0.8

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import array, col, explode, lit, struct, posexplode, slice
from pyspark.sql import DataFrame
from pyspark.sql.window import Window    
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

import re
from functools import reduce
from typing import Iterable

from pyspark.sql.types import FloatType
from tqdm import tqdm

# COMMAND ----------

# helper function to rearrange dataframe to long/skinny for tsfresh
  
def get_pct_null(df):
  null_pcts = []
  for c in df.columns:
    total_rows = df.count()
    null_rows = df.where(F.col(c).isNull()).count()
    null_pcts.append((c,null_rows/total_rows, null_rows))
  
  null_df = pd.DataFrame(null_pcts,columns=['ColumnName', 'PercentNull', 'NumberNull'])
  display(null_df.sort_values(by = ['NumberNull'], ascending = False))

def drop_zero_variance_or_null_columns(df):
    from pyspark.sql.functions import variance
    """
    This function drops all columns which contain null values.
    :param df: A PySpark DataFrame
    """
    var_calc = df.select([F.variance(F.col(c)).alias(c) for c in df.columns]).collect()[0].asDict()
    to_drop = [k for k, v in var_calc.items() if v == 0]
    df = df.drop(*to_drop)
    
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
    to_drop = [k for k, v in null_counts.items() if v > 0]
    df = df.drop(*to_drop)
    
    print(len(df.columns))
    
    return df
  
def melt(
        df: DataFrame, 
        id_vars: Iterable[str], value_vars: Iterable[str], 
        var_name: str="variable", value_name: str="value") -> DataFrame:
    """Convert :class:`DataFrame` from wide to long format."""

    # Create array<struct<variable: str, value: ...>>
    _vars_and_vals = array(*(
        struct(lit(c).alias(var_name), col(c).alias(value_name)) 
        for c in value_vars))

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", explode(_vars_and_vals))

    cols = id_vars + [
            col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)
  
array_mean = udf(lambda x: float(np.mean(x)), FloatType())
spark.udf.register("array_mean", array_mean)

array_variance = udf(lambda x: float(np.var(x)), FloatType())
spark.udf.register("array_variance", array_variance)


# COMMAND ----------

# MAGIC %md 
# MAGIC # Get Dataset with Time-series means

# COMMAND ----------

df = spark.sql("SELECT * FROM eldb.opioid_SA_LA_hosp_sktime_table")

# Select all time-series columns
ts_df = df.select(*[c for c in df.columns if (c.startswith('TS_'))] + ['bene_id'])

# COMMAND ----------

# Calculate features and pivot

ts_temp = ts_df
for ts_col in [c for c in ts_df.columns if c !='bene_id']:
    ts_temp = ts_temp.withColumn(f'{ts_col}__mean', array_mean(ts_col))
    ts_temp = ts_temp.withColumn(f'{ts_col}__mean_30_day', array_mean(slice(ts_col, -30, 30)))
    ts_temp = ts_temp.withColumn(f'{ts_col}_variance', array_variance(ts_col))
    ts_temp = ts_temp.withColumn(f'{ts_col}_variance_30_day', array_variance(slice(ts_col, -30, 30)))
    ts_temp = ts_temp.drop(ts_col)
   

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection

# COMMAND ----------

# MAGIC %md
# MAGIC ### TS Feature Selection
# MAGIC One-way ANOVA for time series-related values<p>
# MAGIC Note: in order to align with other parallel models, non-time series related variables will not be limited by this procedure

# COMMAND ----------

ts_features = ts_temp.drop(*[c for c in df.columns if (c.startswith('TS_'))])

# Bring in target
target = spark.sql(f"SELECT bene_id, target FROM {original_data}")

# Join
final_ts_features_df = target.join(ts_features, "bene_id", "left")

print((final_ts_features_df.count(), len(final_ts_features_df.columns)))

# COMMAND ----------

# Run F test on all continuous variables

variables = drop_zero_variance_or_null_columns(final_ts_features_df.drop("bene_id", "target")).columns 
keys = []
tables = []

# Loop over tsfresh variables
for variable in tqdm(variables):
    model = smf.ols('{} ~ target'.format(variable), data=final_ts_features_df.select(variable, 'target').toPandas()).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    keys.append(variable)
    tables.append(anova_table)

df_anova = pd.concat(tables, keys=keys, axis=0)
df_anova = df_anova.rename(columns={"PR(>F)": "p_value"}).reset_index()

# Save anova table
df_anova_spark = spark.createDataFrame(df_anova).withColumnRenamed("level_0", "feature").withColumnRenamed("level_1", "level")

df_anova_spark.write   \
  .format("delta")   \
  .mode("overwrite") \
  .option('overwriteSchema', True)\
  .saveAsTable(f"{anova_table_loc}")

# View results
display(df_anova_spark.filter("level = 'target'").orderBy("p_value"))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {anova_table_loc}"))

# COMMAND ----------

# Get tsfresh variables with ANOVA p < .05
df_anova_spark = spark.sql(f"SELECT * FROM {anova_table_loc}")
null_dropped = list(set(final_ts_features_df.drop("bene_id", "target").columns) - set(drop_zero_variance_or_null_columns(final_ts_features_df.drop("bene_id", "target")).columns))
cont_best = df_anova_spark.filter(f"level = 'target' and p_value <= .05 ").select('feature', 'p_value').toPandas().sort_values('p_value')
cont_best_list = list(cont_best['feature'].unique())
cont_best_list = [col for col in cont_best_list if col not in null_dropped]
cont_worst = df_anova_spark.filter(f"level = 'target' and (p_value > .05 or p_value is null)").select('feature', 'p_value').toPandas().sort_values('p_value')
cont_worst_list = list(cont_worst['feature'].unique())
cont_worst_list = set([col for col in cont_worst_list if col not in null_dropped]) - set(cont_best_list)


print("Total features: " + str(len(set(final_ts_features_df.drop("bene_id", "target").columns))))
print("Selected continuous features: " + str(len(set(cont_best_list))))
print("Total features dropped for null values = " + str(len(set(null_dropped))))

print("Total ANOVA droppable features = " + str(len(set(cont_worst_list))))
print("")
print("dropped list:")
cont_worst_list

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical Feature Selection
# MAGIC Not currently limiting on non-time series features.
# MAGIC See Descriptive Statistics notebook for more information on categorical variable hypothesis tests

# COMMAND ----------

# MAGIC %md
# MAGIC ## Restrict Correlation
# MAGIC When features are highly correlated, select the one with highest p_value against target and drop the other

# COMMAND ----------

final_ts_features_dropped_df = final_ts_features_df.drop(*cont_worst_list, *null_dropped, 'bene_id', 'target', 'STATIC_DEMO_bene_age')

ts_df = final_ts_features_dropped_df.repartition(200).to_pandas_on_spark()

corrMatrix = ts_df.corr().reset_index().to_spark()

corrlong = melt(corrMatrix, id_vars=['index'], value_vars=[col for col in corrMatrix.columns if col.startswith('TS_')])

corrlong = corrlong.filter("index != variable")

corrlong = corrlong.withColumn('mark_remove', F.when(F.col('value') >= correlation_cutoff, 1).otherwise(0))

anova1 = spark.sql(f"SELECT feature, p_value as p_value_index FROM {anova_table_loc} WHERE level='target'")
anova2 = spark.sql(f"SELECT feature, p_value as p_value_variable FROM {anova_table_loc} WHERE level='target'")

corrlong = (corrlong.join(anova1, corrlong.index==anova1.feature, 'left')
            .join(anova2, corrlong.variable==anova2.feature, 'left'))

display(corrlong.filter('mark_remove == 1').orderBy('index'))


# COMMAND ----------

corr_dropped = [row[0] for row in corrlong.filter("mark_remove == 1 and ((p_value_index > p_value_variable) or p_value_index is null)").select("index").distinct().collect()]
final_tsfresh_unimputed = final_ts_features_df.drop(*cont_worst_list, *null_dropped, *corr_dropped, 'target', 'STATIC_DEMO_bene_age')

print("removed columns by correlation reduction process = " + str(len(corr_dropped)))
print("final time-series columns = " + str(len(final_tsfresh_unimputed.columns)))

# Bring in target
orig = spark.sql(f"SELECT * FROM {original_data}")
orig = orig.drop(*[col for col in orig.columns if col.startswith('TS_')]).drop("STATIC_DEMO_ADI_NATRANK", "STATIC_DEMO_ADI_STATERNK_str")

# Join
final_combined_unimputed = orig.join(final_tsfresh_unimputed, "bene_id", "left").drop("CC_drug_use") # removing cc_drug_use based on correlation to 'cc_opioid_dx' (0.87)

print("Final dataset total columns = " + str(len(final_combined_unimputed.columns)))
display(final_combined_unimputed)

# COMMAND ----------

final_combined_unimputed.cache().count()

# COMMAND ----------

get_pct_null(final_combined_unimputed)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Final

# COMMAND ----------

final_combined_unimputed.write   \
  .format("delta")   \
  .mode("overwrite") \
  .option('overwriteSchema', True)\
  .saveAsTable(f"{final_table_loc}")


# COMMAND ----------


