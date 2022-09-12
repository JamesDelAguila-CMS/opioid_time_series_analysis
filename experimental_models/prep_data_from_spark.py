# Databricks notebook source
from pyspark.sql.functions import col, explode, array, lit, rand

def load_data(train_test_val, upsample=False, include_ts = False, spark_table_name = "eldb.opioid_SA_LA_hosp_sktime_table", output_location = "/dbfs/mnt/eldb_mnt/MMA394/data/", filename = None, contains_bene_id = True, tsfresh_features_to_include = [''], return_df = False):
    """
      This function will choose one of three predefined SQL queries, send it to PySpark, 
      do some table manipulation, upsample the positive class if requested, then 
      save it to a csv in a persistent location.We found saving and loading a csv to be 
      quite fast, and this method saved considerable time while we iterated quickly during 
      development. Once run, you do not need to run this again to access the data now saved in the CSV.
    """
    
    # Choose sql statement based on train/val/test/eval
    if train_test_val == "train":
      if contains_bene_id:
        sql_statement = f"""select * from  {spark_table_name}
                               where MOD(bene_id,20)<14"""
      else:
        sql_statement = f"""select * from  {spark_table_name}"""
    elif train_test_val == "val":
        sql_statement = f"""select * from {spark_table_name}
                               where MOD(bene_id,20)>=14 AND MOD(bene_id,20)<18"""
    elif train_test_val == "test":
        sql_statement = f"""select * from {spark_table_name}
                               where MOD(bene_id,20)>=18"""
    elif train_test_val == "eval":
        sql_statement = f"""select * from {spark_table_name}
                               where MOD(bene_id,20)>=14"""
    else:
        raise ValueError(f"Expected train_test_val to be train, test, val, or eval. Instead got {train_test_val}")
    
    # Get the spark table
    spark_table = spark.sql(sql_statement)
    
    # Remove unwanted columns
    for column in spark_table.columns:
        # We keep columns with 'CC' (chronic conditions) or 'STATIC' (demographic info)
        if "CC" not in column and "STATIC" not in column:
            # We need to keep these two columns, so another branch
            if column != "target" and column != "bene_id":
              # Finally, some conditionally included Time Series Analysis Columns from tsfresh
              # In order for a TimeSeries column to be included:
              #   include_ts must be set to true for these not to be dropped
              #   'TS' must be in the column title
              #   You must indicate which columns you'd like by providing a subset of the column names as string in a list
              #      This is so you can choose the same stats from a number of different time series or only one column.
              if (include_ts and 'TS' in column and any([n in column for n in tsfresh_features_to_include])):
                pass
              else:
                # If we got here, the column should be dropped
                spark_table = spark_table.drop(column)
        if "Opioid" in column:
            # We dropped two columns with 'Opiod' in the name for fear of data leakage. 
            # This was a project wide decision. IIRC, those fields are no longer in the database.
            print(f"Dropping {column}")
            spark_table = spark_table.drop(column)
        
    if upsample and train_test_val == "train":
        # upsample the positive examples.
        major_df = spark_table.filter(col("target") == 0)
        minor_df = spark_table.filter(col("target") == 1)
        
        # Find how many more 0 datapoints there are than 1 datapoints
        ratio = int(major_df.count()/minor_df.count())
        print('Number of 0s: ', major_df.count())
        print('Number of 1s: ', minor_df.count())
        print('Ratio: ', ratio)
        a = range(ratio)
        # usample by that ratio, so that there are the same number of 0 and 1 datapoints.
        oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a])))
        spark_table = major_df.unionAll(oversampled_df.drop('dummy')).orderBy(rand()).coalesce(5)
      
      
    # As Spark only executes once you try to access the rows, 
    # this command takes a majority of the runtime of the whole function.
    pandas_df = spark_table.toPandas()
    
    #Some final conditions for saving or returning the dataframe 
    if include_ts:
      train_test_val+='_ts'
    if return_df:
      return pandas_df
    if not filename:
      filename = f"{train_test_val}_data.csv"
    output_fn = f"{output_location}{filename}"
    print(f"Saving df to {output_fn}")
    pandas_df.to_csv(output_fn, index=False)

# COMMAND ----------

def load_smote_data():
  """
    This function loads the smote data from the corresponding database. Becuase smote creates
    fake data on the same distribution, it's been uploaded directly to its own tables.
    This function is less parameterized, as our smote tests instantly failed (all predictions 
    were identical for all data points), and so we did not do additional development, though
    the data is still usable. Point is, if you need to rerun this function, you will have to
    manually change output locations, sql statements, etc.
  """
  
  sql_statement = "select * from eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos_train_smotenc"
  spark_table = spark.sql(sql_statement)
  for column in spark_table.columns:
    if "TS_" in column:
        spark_table = spark_table.drop(column)
  pandas_df = spark_table.toPandas()
  output_fn = "/dbfs/mnt/eldb_mnt/MMA394/data/train_smote_no_ts.csv"
  print("saving train")
  pandas_df.to_csv(output_fn, index=False)
  
  val_sql_statement = "select * from eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos_ohe where MOD(bene_id,20)>=14 AND MOD(bene_id,20)<18"
  val_table = spark.sql(val_sql_statement)
  for column in val_table.columns:
    if "TS_" in column:
        val_table = val_table.drop(column)
  pandas_df_val = val_table.toPandas()
  val_output_fn = "/dbfs/mnt/eldb_mnt/MMA394/data/val_smote_no_ts.csv"
  print("saving val")
  pandas_df_val.to_csv(val_output_fn, index=False)
  
  test_sql_statement = "select * from eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos_ohe where MOD(bene_id,20)>=18"
  test_table = spark.sql(test_sql_statement)
  for column in val_table.columns:
    if "TS_" in column:
        test_table = test_table.drop(column)
  pandas_df_test = test_table.toPandas()
  test_output_fn = "/dbfs/mnt/eldb_mnt/MMA394/data/test_smote_no_ts.csv"
  print("saving test")
  pandas_df_test.to_csv(test_output_fn, index=False)
  

# COMMAND ----------

load_smote_data()

# COMMAND ----------

load_data("train", upsample=False)
load_data("val", upsample=False)
load_data("test", upsample=False)

# COMMAND ----------

load_data("train", upsample=False,include_ts = True)
load_data("val", upsample=False,include_ts = True)
load_data("test", upsample=False,include_ts = True)

# COMMAND ----------

load_data("train", upsample=False,include_ts = True, spark_table_name = "eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos_train_smoteennc", filename = "train_ts_smoteenc.csv",contains_bene_id = False)
load_data("val", upsample=False,include_ts = True, spark_table_name = "eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos_full_smoteennc", filename = "val_ts_smoteenc.csv")
load_data("test", upsample=False,include_ts = True, spark_table_name = "eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos_full_smoteennc", filename = "test_ts_smoteenc.csv")

# COMMAND ----------

df = load_data("train", upsample=False,include_ts = True, spark_table_name = "eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos", filename = "train_ts_abr.csv",contains_bene_id = False,tsfresh_features_to_include = ['TS_MME_ALL_'], return_df = True)

# COMMAND ----------

load_data("eval", upsample=False,include_ts = True, spark_table_name = "eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos", filename = "eval_ts_abr.csv",contains_bene_id = False,tsfresh_features_to_include = ['TS_MME_ALL_'])

# COMMAND ----------

load_data("val", upsample=False,include_ts = True, spark_table_name = "eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos", filename = "val_ts_abr_data.csv",contains_bene_id = False,tsfresh_features_to_include = ['TS_MME_ALL_'])

load_data("test", upsample=False,include_ts = True, spark_table_name = "eldb.opioid_SA_LA_hosp_final_tsfresh_and_demos", filename = "test_ts_abr_data.csv",contains_bene_id = False,tsfresh_features_to_include = ['TS_MME_ALL_'])
