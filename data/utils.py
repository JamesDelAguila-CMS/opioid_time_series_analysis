# Databricks notebook source
from pyspark.sql.functions import broadcast, when, sequence, to_date, explode, col, expr, struct, collect_list, max, udf, sort_array, date_add, to_date, udf
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import TimestampType
import numpy as np

class preprocess_timeseries:
  '''
  
  Author: James DelAguila (CMS)
  
  Certain time-series analysis libraries like sktime require table formats such as panel, where time series are arranged as nested arrays - where each element of the array consists of 1 period (e.g., 1 day). This Python class processes a Pyspark dataframe composed of line-level claims data/Part D Events and returns a Pyspark dataframe with nested array 'panel' layout where each element of the array consists of 1 day. This Dataframe layout can be converted to a pandas dataframe via the to_pd() function, or used as a standalone. This class is purpose-built with specific functionality for processing part D Event time series.
  
  Note that the execution time and ability to convert to pandas dataframe depend heavily on Spark Session configuration. For pandas conversion, these settings worked for our use-case of ~600k individuals, 24 time series, with 305 days in each array: spark.driver.memory = 40g, maxresultsize = 20g, network.timeout = 10000000, at least 2 workers online.
  
  Parameters:
  ------------
  
  cohort_df: Spark dataframe, a dataframe consisting of relevant claims data from Medicare claims analytical tables
  id: string, the beneficiary identifier for this dataset
  ptd_sql: string, the sql statement which identifies Part D drug lines in cohort_df
  target_col_name: Name of target column from cohort_df
  date_col_name: str, the name of the column in the cohort dataframe that drives the time-series designation (eg., 'frst_srvc_dt' in ELDB)
  sample: float, Proportional random sample of beneficiaries to request full claims history. Full sample if unspecified
  randomseed: int, random seed to apply to sample. 42 if unspecified
  
  
  Example Usages:
  ______________
  
  # For a Spark table
  df = (preprocess_timeseries({cohort_df}, sample=1.0, random_seed = 42)
          .build_time_series(date_col_name = 'first_srvc_dt',
                      start_date = '2020-01-01', 
                      end_date = '2020-10-31',
                      flag_dict = flag_dict,
                      broadcast_dict = broadcast_dict)
          .convert_ts_2_array(flag_dict = flag_dict,
                        broadcast_dict = broadcast_dict)
          .output()
         )
   
   # For a Pandas table      
   df = (preprocess_timeseries({cohort_df}, sample=1.0, random_seed = 42)
          .build_time_series(date_col_name = 'first_srvc_dt',
                      start_date = '2020-01-01', 
                      end_date = '2020-10-31',
                      flag_dict = flag_dict,
                      broadcast_dict = broadcast_dict)
          .convert_ts_2_array(flag_dict = flag_dict,
                        broadcast_dict = broadcast_dict)
          .to_pd()
          .output()
         )
  '''
  
  def __init__(self, cohort_df, idx = 'bene_id', ptd_sql = "pymt_sys = 'PTD-DRUG'", target_col_name = 'target', sample=1.0, random_seed=42):
    self.sample = float(sample)
    self.idx = idx
    self.ptd_sql = ptd_sql
    self.random_seed = random_seed
    self.target_col_name = target_col_name
    self.cohort_df = cohort_df
    self.cohort_df = self.get_random_benes()
    
   
  def get_random_benes(self):
    '''
    Returns sampled dataframe to preprocess_timeseries class
    
    '''
    # Select a sample of distinct beneficiaries and audit
    cohort_df = self.cohort_df
    sample = self.sample 
    random_seed = self.random_seed
    distinct_benes = cohort_df.select(self.idx).distinct().orderBy(self.idx).sample(fraction=sample, seed=random_seed).cache()
    cohort_df = cohort_df.join(distinct_benes, self.idx, 'inner')
    print('Created bene sample and cached dataframe with distinct benes: ', distinct_benes.count())
        
    return cohort_df
  
  def build_time_series(self, date_col_name, start_date, end_date, flag_dict, flag_list_max, broadcast_dict, cohort_df=None):
  
    '''
    For start date to end date, builds a table that contains 1 day/1 bene indicator for each condition set out in the dictionaries.

    Parameters:
    ------------

    start_date: start date of time-series test period (YYYY-MM-DD)
    end_date: start date of time-series test period (YYYY-MM-DD) 
    flag_dict: Dictionary containing pairs indicating naming of condition for which a time-series is constructed and Spark SQL conditions. 
    broadast_dict: Dictionary containing pairs indicating naming of condition for which a time-series is constructed and Spark SQL conditions (e.g., {'TS_DQ_BENZOS' : "CASE WHEN OMS_BENZO_FLAG = 'Y' THEN (SRVC_VOL/CLM_DAYS) END"}. Broadcasts result to all days from Rx fill date to Rx fill date + days_supply
    
    '''
    cohort_df = self.cohort_df
    
    cohort_df = cohort_df.withColumnRenamed(date_col_name, 'date')
    date_df = spark.sql(f"SELECT sequence(to_date('{start_date}'), to_date('{end_date}'), interval 1 day) as date").withColumn("date", explode(col("date")))

    # Path for features that do not broadcast to day
    noa_cohort_df = (broadcast(date_df.select("date").distinct())
    .crossJoin(cohort_df.select(self.idx).distinct())
    .join(cohort_df, ['date', self.idx], "leftouter")
    ).orderBy(self.idx, 'date')

    # Path for features that broadcast to day
    opioid_cohort_df = (broadcast(date_df.select("date").distinct())
    .crossJoin(cohort_df.select(self.idx).distinct()).alias('a')
    .join(cohort_df.filter(self.ptd_sql).alias('b'), expr(f"a.{self.idx} = b.{self.idx} AND a.date between b.date and date_add(b.date, cast(b.clm_days as int))"), "leftouter")
    ).orderBy(f'a.{self.idx}', 'a.date')

    # Process non-opioid agonist conditions and add column   
    for key, value in flag_dict.items():
      noa_cohort_df = noa_cohort_df.withColumn(key, expr(value))

    # Process opioid agonist conditions and add columns
    for key, value in broadcast_dict.items():
      opioid_cohort_df = opioid_cohort_df.withColumn(key, expr(value)).filter(self.ptd_sql)

    # Get sum of rows for time series
    noa_list = list(flag_dict.keys())
    noa_list.extend([self.idx,'date'])

    opx_list = list(broadcast_dict.keys())
    opx_list.extend([f'a.{self.idx}', 'a.date'])

    agg_dict = {}
    sum_keys = [c for c in noa_list if c not in flag_list_max]
    sum_keys.remove(self.idx)
    sum_keys.remove('date')
    max_keys = flag_list_max

    for i in sum_keys:
        agg_dict[i] = 'sum'
    for i in max_keys:
        agg_dict[i] = 'max'
    
    noa_cohort_df = (noa_cohort_df.select(noa_list)
                     .groupBy(self.idx, 'date')
                     .agg(agg_dict)
                    )
    
    opioid_cohort_df = (opioid_cohort_df.select(opx_list)
                          .groupBy(self.idx, 'date')
                          .sum()
                       )

    # Rename fields back to assigned values
    for key in flag_dict.keys():
      noa_cohort_df = (noa_cohort_df.withColumnRenamed('sum(' + str(key) + ')', key)
                                    .withColumnRenamed('max(' + str(key) + ')', key)
                      )
    for key in broadcast_dict.keys():
      opioid_cohort_df = opioid_cohort_df.withColumnRenamed('sum(' + str(key) + ')', key)  

    # Join back broadcasting/non-broadcasting dataframes
    cohort_df = (noa_cohort_df.join(opioid_cohort_df, ['date', self.idx], "leftouter")
                 .na.fill(0).orderBy(self.idx, 'date').cache()
                )
    
    print('Built long time-series with total rows: ', cohort_df.count(), ' and ', cohort_df.select(self.idx).distinct().count(), ' benes') 
                 # If you don't the cache AND this, you get different random benes every iteration (lazy eval)
    self.cohort_df = cohort_df
    return self

  def convert_ts_2_array(self, flag_dict, broadcast_dict):
    '''
    Builds a Spark Dataframe with time series for each measure in dictionaries as array within cell for each beneficiary - panel layout within a Spark Dataframe
    
    Parameters:
    ------------
  
    flag_dict: Dictionary containing pairs indicating naming of condition for which a time-series is constructed and Spark SQL conditions. 
    broadast_dict: Dictionary containing pairs indicating naming of condition for which a time-series is constructed and Spark SQL conditions. Broadcasts result to all days from Rx fill date to Rx fill date + days_supply
    '''

    cohort_df = self.cohort_df
    # Build array timelines in cell
    ndf2_schema = 0
    merged_dictionaries = {**flag_dict, **broadcast_dict}
    for key in merged_dictionaries.keys():

      jump_df = (cohort_df
                 .groupBy(self.idx)
                 .agg(sort_array(collect_list(struct('date', key)))  # Ensures proper ordering of array list
                 .alias('collected_list'))
                 .withColumn(key, col(f"collected_list.{key}"))
                 #.withColumn(key, col(f"collected_list")) # for testing
                 .drop("collected_list")
                ).orderBy(self.idx)

      jump_df = jump_df.withColumnRenamed(jump_df.columns[-1], key)

      # For the first key
      if ndf2_schema < 1:
        ndf2=jump_df
      # All others
      else:
        ndf2 = ndf2.join(jump_df, self.idx, 'outer')

      ndf2_schema +=1
    
    cohort_df = ndf2
    self.cohort_df = cohort_df
    return self
  
  def to_pd(self):
    from sktime.datatypes._panel._convert import is_nested_dataframe
    import pandas as pd
    '''
    Converts Spark nested panel layout to Pandas dataframe compatible with sktime libary. Note that this process may be very memory intensive - if it fails, considering increasing spark settings: sparkdriver.memory, maxresultsize, network.timeout
    '''
    cohort_df = self.cohort_df
    # set arrow
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    
    partitions = int(round((cohort_df.count()/5) + 10000, -4))
    print('Converting to pandas...')
    # Crazy high partition size helps this not crash
    cohort_df = cohort_df.repartition(partitions).toPandas()

    # Turn each element of columns into nested pandas series, as opposed to np.ndarray
    cohort_df[[col for col in cohort_df.columns if isinstance(cohort_df[col], np.ndarray)]] = cohort_df[[col for col in cohort_df.columns if isinstance(cohort_df[col], np.ndarray)]].applymap(lambda x: pd.Series(x))
    
    self.cohort_df = cohort_df
    
    return self  
    
  def output(self):
    '''Output dataframe at any stage before convert to pandas'''
    cohort_df = self.cohort_df
    return cohort_df

# COMMAND ----------


