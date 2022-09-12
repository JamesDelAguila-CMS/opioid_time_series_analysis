# Databricks notebook source
# MAGIC %md 
# MAGIC # Create TSFRESH-ized Dataset
# MAGIC Author: James DelAguila (CMS)
# MAGIC 
# MAGIC This notebook builds a set of TSFRESH features from time-series, then:
# MAGIC 
# MAGIC 1. Eliminates features that fail F-test with p > 0.05
# MAGIC 2. When Pearson correlation is above threshold value (e.g., .8), the feature with the highest F-statistic is chosen

# COMMAND ----------

# MAGIC %pip config --user set global.index-url https://pypi.ccwdata.org/simple
# MAGIC %pip install tsfresh

# COMMAND ----------

# Set storage tables

# The original input from the preprocess_timeseries process
original_data = 'eldb.opioid_SA_LA_hosp_sktime_table'

# A long/skinny version of the original datatable, consumable by tsfresh library
long_skinny_loc = 'eldb.opioid_SA_LA_hosp_sktime_long_table'

# complete set of selected tsfresh metrics
tsfresh_all_loc = 'eldb.opioid_SA_LA_hosp_all_desired_tsfresh_features'

# results from F-test process
anova_table_loc = 'dua_000000_jde328.opioid_tsfresh_anova_062422'

# final data table after time series feature selection and imputation steps
final_table_loc = 'eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos'

# threshold of correlation between tsfresh metrics to initiate selection process
correlation_cutoff = 0.8

# COMMAND ----------

from tsfresh.convenience.bindings import spark_feature_extraction_on_chunk

from pyspark.sql import functions as F
from pyspark.sql.functions import array, col, explode, lit, struct, posexplode
from pyspark.sql import DataFrame
from pyspark.sql.window import Window    

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

import re
from typing import Iterable

from tqdm import tqdm

# COMMAND ----------

# helper functions to rearrange dataframe to long/skinny for tsfresh

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

def get_pct_null(df):
  null_pcts = []
  for c in df.columns:
    total_rows = df.count()
    null_rows = df.where(F.col(c).isNull()).count()
    null_pcts.append((c,null_rows/total_rows, null_rows))
  
  null_df = pd.DataFrame(null_pcts,columns=['ColumnName', 'PercentNull', 'NumberNull'])
  display(null_df.sort_values(by = ['NumberNull'], ascending = False))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Long/Skinny Table
# MAGIC Required for tsfresh on Spark

# COMMAND ----------

# rearrange time series components to long/skinny and save

sdf = spark.sql(f"SELECT * FROM {original_data}")

# 'Melt' or convert the array columns into long format
ndf = melt(sdf, id_vars=['bene_id'], value_vars=[col for col in sdf.columns if col.startswith('TS_')])
ndf = ndf.select(F.col('bene_id').alias('id'), F.col('variable').alias('kind'), posexplode('value').alias('time', 'value')).repartition(100000)

# This is useful on its own. Save it off as an intermediary table
ndf.write   \
  .format("delta")   \
  .mode("overwrite") \
  .option('overwriteSchema', True)\
  .saveAsTable(f"{long_skinny_loc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build TSFRESH Metrics
# MAGIC Start here if not regenerateing long table, but want tsfresh metrics

# COMMAND ----------

# Calculate TSFRESH features and pivot
# Note: the decision of which metrics to select and how to adjust the parameters is highly dependent on compute time you can withstand, as well as the specific problem. There are a few pre-configured options given by tsfresh which were not a perfect fit for our problem.

discarded_ts_tf = [
"time_reversal_asymmetry_statistic",
"autocorrelation",
"c3",
"symmetry_looking",
"large_standard_deviation",
"partial_autocorrelation",
"binned_entropy",
"cwt_coefficients",
"ar_coefficient",
"value_count",         
"linear_trend - some levels",         
"agg_linear_trend",
"linear_trend_timewise",
"augmented_dickey_fuller",
"approximate_entropy",
"number_crossing_m",
"range_count",
"cid_ce - normalize false",                
"friedrich_coefficients",
"query_similarity_count",
"matrix_profile",
"benford_correlation",
"has_duplicate_max",
"percentage_of_reoccurring_datapoints_to_all_datapoints",
"sum_of_reoccurring_data_points",
"index_mass_quantile",
"sample_entropy",
"number_cwt_peaks",
"spkt_welch_density",
"fft_coefficient",
"fft_aggregated",
"max_langevin_fixed_point",
"lempel_ziv_complexity",
"fourier_entropy"
]

fc_parameters = {
"abs_energy": None,
"maximum": None,
"minimum": None,
"mean": None, 
"median": None,
"standard_deviation": None,
"sum_values": None,
"mean_abs_change": None,
"mean_change": None,
"mean_second_derivative_central": None,
"has_duplicate_max": None,
"has_duplicate_min": None,
"variation_coefficient": None,
"variance": None,
"skewness": None,
"kurtosis": None,
"root_mean_square": None,
"absolute_sum_of_changes": None,
"longest_strike_below_mean": None,
"longest_strike_above_mean": None,
"count_above_mean": None,
"count_below_mean": None,
"last_location_of_maximum": None,
"first_location_of_maximum": None,
"last_location_of_minimum": None,
"first_location_of_minimum": None,
"percentage_of_reoccurring_values_to_all_values": None,
"sum_of_reoccurring_values": None,
"ratio_value_number_to_time_series_length": None,
"cid_ce": [{"normalize": True}],
"quantile": [
  {"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
  ],
"number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
"change_quantiles": [
{"ql": ql, "qh": qh, "isabs": b, "f_agg": f}
for ql in [0.6, 0.8]
for qh in [0.8, 1.0]
for b in [True]
for f in ["mean"]
if ql < qh
],
"linear_trend": [
  {"attr": "slope"}],
"energy_ratio_by_chunks": [
  {"num_segments": 10, "segment_focus": i} for i in range(10)],
"ratio_beyond_r_sigma": [
  {"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]],
"count_above": [{"t": 90}], # above this level is typically considered a high MME
"count_below": [{"t": 89}], # above this level is typically considered a high MME
"permutation_entropy": [
  {"tau": 1, "dimension": x} for x in [7]],
"mean_n_absolute_max": [{"number_of_maxima": 10}]
}

# Build tsfresh features
ndf_grouped = spark.sql(f"SELECT * FROM {prod_schema}.opioid_SA_LA_hosp_sktime_long_table").repartition(500).groupby(["id", "kind"])
features = spark_feature_extraction_on_chunk(ndf_grouped, column_id="id", column_kind="kind", column_sort="time", column_value="value", default_fc_parameters=fc_parameters)

# Pivot
spark.conf.set("spark.sql.pivotMaxValues", "50000")
pivotDF = features.groupBy("id").pivot("variable").sum("value")

# Write to file. 
# This will name the variables with some ugly characters (punctuation). Deal with it later
pivotDF.write   \
  .format("delta")   \
  .mode("overwrite") \
  .option('overwriteSchema', True)\
  .saveAsTable(f"{tsfresh_all_loc}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection

# COMMAND ----------

# MAGIC %md
# MAGIC ### TS Feature Selection
# MAGIC Univariate tests for time series-related values<p>
# MAGIC Note: in order to align with other parallel models, non-time series related variables will not be limited by this procedure

# COMMAND ----------

# All tsfresh features (no demographics, etc)
pivdf2 = spark.sql(f"SELECT * FROM {tsfresh_all_loc}")

# Rename columns to eliminate punctuation
cols=[re.sub('\W+', "", i) for i in pivdf2.columns]
pivdf2 = pivdf2.toDF(*cols)
pivdf2 = pivdf2.withColumnRenamed("id", "bene_id")
del cols

# Bring in target
target = spark.sql(f"SELECT bene_id, target FROM {original_data}")

# Join
final_tsfresh_df = target.join(pivdf2, "bene_id", "left")

print((final_tsfresh_df.count(), len(final_tsfresh_df.columns)))

# COMMAND ----------

# Run univariate tests on all continuous variables

# Get all TSFRESH variables with positive variance
variables = drop_zero_variance_or_null_columns(final_tsfresh_df.drop("bene_id", "target")).columns 

# Loop over tsfresh variables
keys = []
tables = []

for variable in tqdm(variables):
    model = smf.ols('{} ~ target'.format(variable), data=final_tsfresh_df.select(variable, 'target').toPandas()).fit()
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

# Get tsfresh variables with p < .05
df_anova_spark = spark.sql(f"SELECT * FROM {anova_table_loc}")
null_dropped = list(set(final_tsfresh_df.drop("bene_id", "target").columns) - set(drop_zero_variance_or_null_columns(final_tsfresh_df.drop("bene_id", "target")).columns))
cont_best = df_anova_spark.filter(f"level = 'target' and p_value <= .05 ").select('feature', 'p_value').toPandas().sort_values('p_value')
cont_best_list = list(cont_best['feature'].unique())
cont_best_list = [col for col in cont_best_list if col not in null_dropped]
cont_worst = df_anova_spark.filter(f"level = 'target' and (p_value > .05 or p_value is null)").select('feature', 'p_value').toPandas().sort_values('p_value')
cont_worst_list = list(cont_worst['feature'].unique())
cont_worst_list = set([col for col in cont_worst_list if col not in null_dropped]) - set(cont_best_list)


print("Total features: " + str(len(set(final_tsfresh_df.drop("bene_id", "target").columns))))
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

# Drop those based on univariate tests above
final_tsfresh_dropped_df = final_tsfresh_df.drop(*cont_worst_list, *null_dropped, 'bene_id', 'target', 'STATIC_DEMO_bene_age')

# Make correlational matrix and melt it to rows
ts_df = final_tsfresh_dropped_df.repartition(200).to_pandas_on_spark()
corrMatrix = ts_df.corr().reset_index().to_spark()
corrlong = melt(corrMatrix, id_vars=['index'], value_vars=[col for col in corrMatrix.columns if col.startswith('TS_')])
corrlong = corrlong.filter("index != variable")

# Identify potential variables to drop
window = Window.partitionBy('index').orderBy(F.abs(F.col('value')).asc_nulls_last())
corrlong = corrlong.withColumn('rank', F.row_number().over(window))

corrlong = corrlong.withColumn('mark_remove', F.when(F.col('value') >= correlation_cutoff, 1).otherwise(0))

anova1 = spark.sql(f"SELECT feature, p_value as p_value_index FROM {anova_table_loc} WHERE level='target'")
anova2 = spark.sql(f"SELECT feature, p_value as p_value_variable FROM {anova_table_loc} WHERE level='target'")
corrlong = (corrlong.join(anova1, corrlong.index==anova1.feature, 'left')
            .join(anova2, corrlong.variable==anova2.feature, 'left'))


display(corrlong.filter('mark_remove == 1').orderBy('index'))

# COMMAND ----------

# Drop variables with highly-correlated better alternative

corr_dropped = [row[0] for row in corrlong.filter("mark_remove == 1 and ((p_value_index > p_value_variable) or p_value_index is null)").select("index").distinct().collect()]
final_tsfresh_unimputed = final_tsfresh_df.drop(*cont_worst_list, *null_dropped, *corr_dropped, 'target', 'STATIC_DEMO_bene_age')

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


