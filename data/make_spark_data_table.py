# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Create Spark Datatable
# MAGIC This notebook collects data from various inputs, including the underlying claims to build a panel format Spark table.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Run Install Requirements

# COMMAND ----------

# MAGIC %run "../Requirements"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Reference Lists
# MAGIC Note: OMS lookup table accessed via CCW Snowflake

# COMMAND ----------

# MAGIC %run "../data/Reference Tables/reference_lists"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Utilities

# COMMAND ----------

# MAGIC %run "../data/utils"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Reference Tables

# COMMAND ----------

# MAGIC %run "../data/Reference Tables/Import Opioid Reference"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import year, countDistinct
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, ShortType, FloatType, ByteType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Cohort Dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC ### Main Query

# COMMAND ----------

# Main Query
# Cohort: 65+ Continuously-enrolled FFS ABD benes (non-hospice, non-dual) in 2020
# Target: non-elective opioid-related adverse event ICD-10 code in hospital/ED between Nov-Dec 2020

cohort_df = spark.sql(f"""
SELECT 
a.bene_id,
a.clm_id,
a.line_num,
a.srvc_dlvrd, 
a.first_srvc_dt, 
a.prvdr_spclty,
a.prvdr_id,
a.dgns_cd1,
a.dgns_cd2,
a.dgns_cd3,
a.dgns_cd4,
a.dgns_cd5,
a.dgns_cd6,
a.dgns_cd7,
a.dgns_cd8,
a.dgns_cd9,
a.dgns_cd10,
a.dgns_cd11,
a.dgns_cd12,
a.pymt_sys,
a.srvc_vol,
a.clm_days,

CASE WHEN
  (DGNS_CD1 in ({FULL_Opx_Dx_list}) OR
  DGNS_CD2 in ({FULL_Opx_Dx_list}) OR
  DGNS_CD3 in ({FULL_Opx_Dx_list}) OR
  DGNS_CD4 in ({FULL_Opx_Dx_list}) OR
  DGNS_CD5 in ({FULL_Opx_Dx_list}) OR
  DGNS_CD6 in ({FULL_Opx_Dx_list}) OR
  DGNS_CD7 in ({FULL_Opx_Dx_list}) OR
  DGNS_CD8 in ({FULL_Opx_Dx_list}) OR
  DGNS_CD9 in ({FULL_Opx_Dx_list}) OR
  DGNS_CD10 in ({FULL_Opx_Dx_list}) OR
  DGNS_CD11 in ({FULL_Opx_Dx_list}) OR
  DGNS_CD12 in ({FULL_Opx_Dx_list}))
  AND ER_count > 0 
  AND first_srvc_dt BETWEEN '2020-11-01' and '2020-12-31' -- target period
  AND CLM_IP_ADMSN_TYPE_CD not in ('3') -- non-elective
THEN 1 
ELSE 0 
END AS target,
count(*) as rows_count

FROM eldb.rt_eldb_pt_abd_2017_2022_service_level_100pct_binned a

INNER JOIN (SELECT bene_id,  
CASE WHEN BUYIN_desc in('Part A and Part B','Part A state buy-in','Part B state buy-in','Part A and Part B state buy-in') AND OP_MDCD_monthly = 'NONDUAL' AND PTD_enrlmt = 'Enrolled in PTD for the month' AND HMOIND_desc = 'Not a member of an HMO'
THEN row_number() OVER (PARTITION BY bene_id ORDER BY month_begin) END AS ABD_NONDUAL_months

FROM eldb.longitudinal_eldb_bene 
WHERE year(month_begin) = 2020) b on a.bene_id = b.bene_id

LEFT JOIN (SELECT clm_id, 
COUNT(CASE WHEN REV_CNTR_CD in ('0450', '0451', '0452', '0456', '0459','0981') 
THEN 1 
ELSE 0 END) AS ER_count

FROM extracts.gvclm18_pta_revenue_2020
GROUP BY 1) c on a.clm_id = c.clm_id

LEFT JOIN (SELECT clm_id, 
CLM_IP_ADMSN_TYPE_CD

FROM extracts.gvclm18_pta_claim_2020
) d on a.clm_id = d.clm_id

ANTI JOIN (SELECT bene_id
FROM extracts.gvclm18_pta_revenue_2020
WHERE SRVC_2 = 'HOS' AND REV_CNTR_CD in ('0651', '0652', '0655','0656') 
) c2 on a.bene_id = c2.bene_id

WHERE year(first_srvc_dt) = 2020 and (ABD_NONDUAL_months=12 or (ABD_NONDUAL_months = 11 and bene_death_dt between '2020-12-01' and '2020-12-31') or (ABD_NONDUAL_months = 10 and bene_death_dt between '2020-11-01' and '2020-11-30')) and bene_age >=65 and (bene_death_dt is null or bene_death_dt >= '2020-11-01')
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 -- making sure there are no duplicates, especially in Part D events

ORDER BY bene_id, first_srvc_dt
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join reference table

# COMMAND ----------


# Join drug ref table from reference table
pre_joined_df = (cohort_df.join(drug_ref_df, [drug_ref_df.FDB_NDC == cohort_df.srvc_dlvrd, year(cohort_df.first_srvc_dt).between(drug_ref_df.BEGIN_YR, drug_ref_df.END_YR)], 'left'))

# Select distinct benes that have a non-MAT opioid during October
opioid_group = pre_joined_df.filter("OMS_OPIOID_FLAG = 'Y' AND OMS_NALOXONE_FLAG = 'N' AND OMS_MAT_FLAG = 'N' AND FDB_IND = 'Y' AND first_srvc_dt between '2020-10-01' and '2020-10-31' AND FDB_GNN != 'AMLODIPINE BESYLATE'").select('bene_id').distinct()
# Select only benes in cohort table that have a non-MAT opioid during October
joined_df = pre_joined_df.join(opioid_group, 'bene_id', 'inner')

# Get cum sum of distinct Opx prescribers (IDs)
window_spec = (Window.partitionBy('bene_id').orderBy('first_srvc_dt')
             .rowsBetween(Window.unboundedPreceding, 0))
opioid_group_rxers =  (pre_joined_df
                       .filter("OMS_OPIOID_FLAG = 'Y' AND OMS_NALOXONE_FLAG = 'N' AND OMS_MAT_FLAG = 'N' AND FDB_IND = 'Y' AND FDB_GNN != 'AMLODIPINE BESYLATE'")
                       .withColumn("cum_distinct_opx_prescribers", 
                                   F.size(F.collect_set('PRVDR_ID').over(window_spec)))
                                           )

# Add data from cumulative prescribers to main table            
joined_df = (joined_df.join(opioid_group_rxers
                   .select('bene_id', 'clm_id', 'first_srvc_dt', 'cum_distinct_opx_prescribers'), 
                           ['bene_id', 'clm_id', 'first_srvc_dt'], 
                           'left'))


# COMMAND ----------

# Cache table
joined_df.cache().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### View Details

# COMMAND ----------

# Total benes
display(joined_df.agg(countDistinct('bene_id')))
# Opioid hosp/ED visit benes
display(joined_df.filter("target = 1").agg(countDistinct('bene_id')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Other Elements

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time Series

# COMMAND ----------

# Note AMLODIPINE BESYLATE in ref table as opioid - Can only be removed explicitly by generic name or CDC_OPIOID_LA_SA_ACTING_FLAG in ('SA', 'LA')
# Buprenorphine and buprenorphine HCL not entered - no MME conversion
# https://www.cms.gov/files/document/otp-billing-and-payment-fact-sheet.pdf


global_Opx_conds = "OMS_OPIOID_FLAG = 'Y' and 2020 BETWEEN BEGIN_YR and END_YR AND OMS_NALOXONE_FLAG = 'N' AND OMS_MAT_FLAG = 'N' AND FDB_IND = 'Y' AND FDB_GNN != 'AMLODIPINE BESYLATE'"


flag_dict = {
    'TS_CLMLINES_OTP_NONDRUG': "CASE WHEN SRVC_DLVRD IN ('G2074', 'G2076', 'G2077') THEN 1 ELSE 0 END",
    'TS_CLMLINES_METHADONE_OTP_MAT': "CASE WHEN SRVC_DLVRD IN ('G2067', 'G2078') THEN 1 ELSE 0 END",
    'TS_CLMLINES_METHADONE_INJ_NONOTP': "CASE WHEN SRVC_DLVRD IN ('J1230') THEN 1 ELSE 0 END",
    'TS_CLMLINES_BUPRENORPHINE_ORAL_OTP_MAT': "CASE WHEN SRVC_DLVRD IN ('G2068', 'G2079') THEN 1 ELSE 0 END", 
    'TS_CLMLINES_BUPRENORPHINE_INJ_OTP_MAT': "CASE WHEN SRVC_DLVRD IN ('G2069') THEN 1 ELSE 0 END", 
    'TS_CLMLINES_BUPRENORPHINE_IMPL_OTP_MAT': "CASE WHEN SRVC_DLVRD IN ('G2070', 'G2072') THEN 1 ELSE 0 END""", # G2071 is an implant removal presumed no drug
    'TS_CLMLINES_BUPRENORPHINE_ORAL_NONOTP': "CASE WHEN SRVC_DLVRD IN ('J0571', 'J0572', 'J0573', 'J0574', 'J0575') OR (OMS_MAT_FLAG = 'Y' AND FDB_GNN LIKE '%BUPRENORPHINE%') THEN 1 ELSE 0 END",
    'TS_CLMLINES_BUPRENORPHINE_INJ_NONOTP': "CASE WHEN SRVC_DLVRD IN ('J0592', 'Q9991', 'Q9992') THEN 1 ELSE 0 END",
    
    'TS_CLMLINES_NALTREXONE_INJ_OTP_MAT' : "CASE WHEN SRVC_DLVRD = 'G0273' THEN 1 ELSE 0 END",
    'TS_CLMLINES_NALTREXONE_INJ_NONOTP_MAT' : "CASE WHEN SRVC_DLVRD in ('J1235') THEN 1 ELSE 0 END",
    
    'TS_CLMLINES_NALOXONE' : "CASE WHEN SRVC_DLVRD in ('G2215', 'G2216') OR OMS_NALOXONE_FLAG = 'Y' THEN 1 ELSE 0 END",
    'TS_CLMLINES_OUD_PSYCHOTHERAPY' : "CASE WHEN SRVC_DLVRD IN ('G2086', 'G2087', 'G2088') THEN 1 ELSE 0 END",
    'TS_CUM_OPX_RXERS' : "MAX(cum_distinct_opx_prescribers) OVER (PARTITION BY BENE_ID ORDER BY DATE ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS intermediate",
    'TS_FLAG_DENTIST_OPX_RXERS': f"CASE WHEN PRVDR_SPCLTY = '19' AND {global_Opx_conds} THEN 1 ELSE 0 END",
    'TS_FLAG_PYCHLGST_RXERS': f"CASE WHEN PRVDR_SPCLTY IN ('62', '68') THEN 1 ELSE 0 END",
    'TS_FLAG_NP_PA_OPX_RXERS': f"CASE WHEN PRVDR_SPCLTY IN ('50', '97') AND {global_Opx_conds} THEN 1 ELSE 0 END",
    'TS_FLAG_OPX_ED_VISIT': f"""CASE WHEN
                                      (DGNS_CD1 in ({FULL_Opx_Dx_list}) OR
                                      DGNS_CD2 in ({FULL_Opx_Dx_list}) OR
                                      DGNS_CD3 in ({FULL_Opx_Dx_list}) OR
                                      DGNS_CD4 in ({FULL_Opx_Dx_list}) OR
                                      DGNS_CD5 in ({FULL_Opx_Dx_list}) OR
                                      DGNS_CD6 in ({FULL_Opx_Dx_list}) OR
                                      DGNS_CD7 in ({FULL_Opx_Dx_list}) OR
                                      DGNS_CD8 in ({FULL_Opx_Dx_list}) OR
                                      DGNS_CD9 in ({FULL_Opx_Dx_list}) OR
                                      DGNS_CD10 in ({FULL_Opx_Dx_list}) OR
                                      DGNS_CD11 in ({FULL_Opx_Dx_list}) OR
                                      DGNS_CD12 in ({FULL_Opx_Dx_list}))
                                      THEN 1 
                                    ELSE 0 
                                    END"""
                                    
}

broadcast_dict = { # These results get broadcast to all days from Rx fill date to Rx fill date + days_supply
  
  'TS_MME_ALL_SA' : f"""CASE WHEN (CDC_OPIOID_LA_SA_ACTING_FLAG = 'SA' OR FDB_GNN in ('HYDROCODONE/ACETAMINOPHEN', 'HYDROCODONE/IBUPROFEN', 'OXYCODONE HCL/ACETAMINOPHEN', 'OXYCODONE HCL/ASPIRIN', 'ACETAMINOPHEN WITH CODEINE', 'BUTALBIT/ACETAMIN/CAFF/CODEINE', 'BUTALBITAL/ASPIRIN/CAFFEINE', 'CODEINE/BUTALBITAL/ASA/CAFFEIN')) AND {global_Opx_conds} THEN (SRVC_VOL/CLM_DAYS) * OMS_MME_CONVERSION_FACTOR * OMS_OPIOID_STR ELSE 0 END""",
  
  'TS_MME_ALL_LA' : f"CASE WHEN CDC_OPIOID_LA_SA_ACTING_FLAG = 'LA' AND {global_Opx_conds} THEN (SRVC_VOL/CLM_DAYS) * OMS_MME_CONVERSION_FACTOR * OMS_OPIOID_STR ELSE 0 END",
    
  'TS_DQ_BENZOS' : "CASE WHEN OMS_BENZO_FLAG = 'Y' THEN (SRVC_VOL/CLM_DAYS) END",
  
  'TS_DQ_ANTIPSYCH' : "CASE WHEN ANTIPSYCH_FLAG = 'Y' THEN (SRVC_VOL/CLM_DAYS) END",
  
  'TS_DQ_BUPRENORPHINE_PART_AGON': "CASE WHEN FDB_GNN LIKE '%BUPRENORPHINE%' AND OMS_MAT_FLAG = 'N' THEN (SRVC_VOL/CLM_DAYS) ELSE 0 END",
  'TS_DQ_NALTREXONE_PART_AGON': "CASE WHEN FDB_GNN LIKE '%NALTREXONE%' AND FDB_GNN NOT LIKE '%METHYLNALTREXONE%' THEN (SRVC_VOL/CLM_DAYS) ELSE 0 END"
  }

# output nested array Pandas dataframe
ts_df = (preprocess_timeseries(joined_df, sample=1.0, random_seed = 42)
          .build_time_series(date_col_name = 'first_srvc_dt',
                      start_date = '2020-01-01', 
                      end_date = '2020-10-31',
                      flag_dict = flag_dict,
                      flag_list_max = ['TS_CUM_OPX_RXERS', 'TS_FLAG_DENTIST_OPX_RXERS', 'TS_FLAG_OPX_ED_VISIT','TS_FLAG_PYCHLGST_RXERS', 'TS_FLAG_NP_PA_OPX_RXERS'],
                      broadcast_dict = broadcast_dict)
          .convert_ts_2_array(flag_dict = flag_dict,
                        broadcast_dict = broadcast_dict)
          .output()
         )
         
ts_df.cache().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add target

# COMMAND ----------

targets = joined_df.select('bene_id', 'target').groupBy('bene_id').agg(max('target').alias('target'))

df_wtargets = ts_df.join(targets, 'bene_id', 'inner')

df_wtargets.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Chronic Conditions

# COMMAND ----------

# Assign combined CC table to variable
condition = sqlContext.sql(f"""
SELECT * FROM extracts.mbsf_mbsf_cc_2020 a

FULL OUTER JOIN extracts.mbsf_mbsf_otcc_2020 b
ON a.BENE_ID = b.BENE_ID

"""
              )

# COMMAND ----------

# Recoding Conditions to be readable, dummy encoding

cc_dict = {
'ATRIAL_FIB':'Afib'
, 'ALZH_DEMEN':'Alz_Dem'
, 'ALZH': 'Alzheimers' 
, 'AMI':'Act_Myo_Inf'
, 'ANEMIA':'Anemia'
, 'ASTHMA':'Asthma'
, 'CANCER_BREAST':'Breast_Cncr'
, 'CATARACT':'Cataract'
, 'CHF':'CHF'
, 'CHRONICKIDNEY':'CKD'
, 'COPD':'COPD'
, 'CANCER_COLORECTAL':'Colorectal_Cncr'
, 'DEPRESSION':'Depression'
, 'DIABETES':'Diabetes'
, 'CANCER_ENDOMETRIAL':'Endometrial_Cncr'
, 'GLAUCOMA':'Glaucoma'
, 'HIP_FRACTURE':'Hip_Fracture'
, 'HYPERL':'Hyperlipidemia'
, 'HYPERP':'Ben_Prost_Hyprpl'
, 'HYPERT':'Hypertension'
, 'HYPOTH':'Hypothyroidism'
, 'ISCHEMICHEART':'Ischemic_HD'
, 'CANCER_LUNG':'Lung_cncr'
, 'OSTEOPOROSIS':'Osteoporosis'
, 'RA_OA': 'RA_Osteoarth'
, 'STROKE_TIA': 'Stroke'
, 'CANCER_PROSTATE': 'Prostate_Cncr'
, 'ACP_MEDICARE':'Conduct_dis'
, 'ALCO_MEDICARE':'Alcoholism'
, 'ANXI_MEDICARE':'Anxiety'
, 'AUTISM_MEDICARE':'Autism'
, 'BIPL_MEDICARE':'Bipolar'
, 'BRAINJ_MEDICARE':'Brain_injury'
, 'CERPAL_MEDICARE':'Cerebral_palsy'
, 'CYSFIB_MEDICARE':'Cystic_Fibrs'
, 'DEPSN_MEDICARE':'Depressive_dis'
, 'DRUG_MEDICARE':'Drug_use'
, 'EPILEP_MEDICARE':'Epilepsy'
, 'FIBRO_MEDICARE':'Fibromyalgia'
, 'HEARIM_MEDICARE':'Hearing_imp'
, 'HEPVIRAL_MEDICARE':'Hepatitis_viral'
, 'HIVAIDS_MEDICARE':'HIVAIDS'
, 'INTDIS_MEDICARE':'Intillectual_dsblty'
, 'LEADIS_MEDICARE':'Learning_dsblty'
, 'LEUKLYMPH_MEDICARE':'Leuk_lymph'
, 'LIVER_MEDICARE':'Liver_dis'
, 'MIGRAINE_MEDICARE':'Migraine'
, 'MOBIMP_MEDICARE':'Mobility_impairment'
, 'MULSCL_MEDICARE':'Multiple_sclrs'
, 'MUSDYS_MEDICARE':'Muscular_dys'
, 'OBESITY_MEDICARE':'Obesity'
, 'OTHDEL_MEDICARE':'Other_developmental_dis'
, 'OUD_ANY_MEDICARE': 'Opioid_any'
, 'OUD_DX_MEDICARE' : 'Opioid_dx'
, 'OUD_HOSP_MEDICARE' : 'Opioid_hospitalization'  
, 'OUD_MAT_MEDICARE': 'Opioid_medically_assisted'
, 'PSDS_MEDICARE':'Personality_dis'
, 'PTRA_MEDICARE':'PTSD'
, 'PVD_MEDICARE':'Periph_vasc_dis'
, 'SCD_MEDICARE': 'Sickle_cell_disease'
, 'SCHI_MEDICARE':'Schizophrenia'
, 'SCHIOT_MEDICARE':'Schizophrenia_and_other_psychotic'
, 'SPIBIF_MEDICARE':'Spina_bifida'
, 'SPIINJ_MEDICARE':'Spinal_injury'
, 'TOBA_MEDICARE':'Tobacco_use'
, 'ULCERS_MEDICARE':'Ulcers'
, 'VISUAL_MEDICARE':'Visual_imp' 
}

# Create lists of old naming and new naming of chronic conditions
before_list = list(cc_dict.keys())
after_list = list(cc_dict.values())

conditions_recode = condition

# Flag as 1 if claims condition met for CC (1 or 3)
for (x, y) in zip(before_list, after_list):
  conditions_recode=conditions_recode\
  .withColumn(y, F.when((F.col(x) == 1) | (F.col(x) == 3), 1).otherwise(0))

# Add the encode suffix to the columns
for column in after_list:
  conditions_recode=conditions_recode\
.withColumnRenamed(column, "CC_" + column)

after_list = ["{}{}".format('CC_', i) for i in after_list]

after_list.extend(['a.bene_id'])

# Select only human readable column names and add _encode suffix                       
conditions_recode=conditions_recode\
.select(after_list)

# Dropping any reference to Opioid hospitalization from the chronic conditions tables, which would self-reference target
conditions_recode = conditions_recode.drop('CC_Opioid_any', 'CC_Opioid_hospitalization', 'CC_Opioid_dx', 'CC_Opioid_medically_assisted', 'CC_Drug_use')

# COMMAND ----------

df_wtargets_cc = df_wtargets.join(conditions_recode, 'bene_id', 'inner')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Demographics

# COMMAND ----------

demos = spark.sql("""
SELECT 
a.bene_id,
cast(round(DATEDIFF('2020-12-31', birth_dt)/365.25) AS INT) as STATIC_DEMO_bene_age,
sex_desc as STATIC_DEMO_sex_label,
state_cd as STATIC_DEMO_state_cd,
race_cd_desc as STATIC_DEMO_race_label,
crec_desc as STATIC_DEMO_crec_label,
CASE WHEN ruca.RUCA1 BETWEEN 1 AND 10 THEN ruca.RUCA1 ELSE null END AS STATIC_DEMO_RUCA1,
CAST(first(ADI_NATRANK_BG_2019_v3_1) as INT) as STATIC_DEMO_ADI_NATRANK,
CAST(
CASE
	WHEN CAST(first(ADI_NATRANK_BG_2019_v3_1) AS DECIMAL) Between 0 AND 10.999 THEN '0-10'
	WHEN CAST(first(ADI_NATRANK_BG_2019_v3_1) AS DECIMAL) Between 11 AND 20.999 THEN '11-20'
	WHEN CAST(first(ADI_NATRANK_BG_2019_v3_1) AS DECIMAL) Between 21 AND 30.999 THEN '21-30'
	WHEN CAST(first(ADI_NATRANK_BG_2019_v3_1) AS DECIMAL) Between 31 AND 40.999 THEN '31-40'
	WHEN CAST(first(ADI_NATRANK_BG_2019_v3_1) AS DECIMAL) Between 41 AND 50.999 THEN '41-50'
	WHEN CAST(first(ADI_NATRANK_BG_2019_v3_1) AS DECIMAL) Between 51 AND 60.999 THEN '51-60'
	WHEN CAST(first(ADI_NATRANK_BG_2019_v3_1) AS DECIMAL) Between 61 AND 70.999 THEN '61-70'
	WHEN CAST(first(ADI_NATRANK_BG_2019_v3_1) AS DECIMAL) Between 71 AND 80.999 THEN '71-80'
	WHEN CAST(first(ADI_NATRANK_BG_2019_v3_1) AS DECIMAL) Between 81 AND 90.999 THEN '81-90'
	WHEN CAST(first(ADI_NATRANK_BG_2019_v3_1) AS DECIMAL) Between 91 AND 100 THEN '91-100'
		ELSE 
		CASE
			WHEN first(ADI_NATRANK_BG_2019_v3_1) in ('PH', 'GQ', 'PH-GQ', 'QDI') THEN first(ADI_NATRANK_BG_2019_v3_1)
			ELSE 'Unassigned'
			END 
END AS STRING) AS STATIC_DEMO_ADI_NATRANK_binned

FROM eldb.longitudinal_eldb_bene a

LEFT JOIN eldb.ruca_2010_zipcode ruca
           on a.zip5_ltst = ruca.ZIP_CODE
           and a.month_end = '2020-10-31'
           
LEFT JOIN eldb.adi_ccw_table adi 
          on a.bene_id = adi.bene_id
          and '2020-10-31' between drvd_adr_bgn_dt and drvd_adr_end_dt 

WHERE a.month_end = '2020-10-31'
GROUP BY 1,2,3,4,5,6,7

"""
                 )

# COMMAND ----------

df_wtargets_cc_demos = df_wtargets_cc.join(demos, 'bene_id', 'inner')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Nursing Home Data

# COMMAND ----------

# Create the array for NH daily status
date_df = spark.sql(f"SELECT sequence(to_date('2020-01-01'), to_date('2020-10-31'), interval 1 day) as date").withColumn("date", explode(col("date")))
NH_daily = (broadcast(date_df.select("date").distinct())
               .crossJoin(joined_df.select("bene_id").distinct())
               .join(spark.sql("SELECT * FROM eldb.mds_timeline_v0_1_0"), ['date', 'bene_id'], "leftouter")
               .select('bene_id', 'date', 'in_nh_today')
               .filter("date BETWEEN '2020-01-01' AND '2020-10-31'")
               .fillna(0)
           )

NH_daily = (NH_daily
                 .groupBy("bene_id")
                 .agg(sort_array(collect_list(struct('date', 'in_nh_today')))  # Ensures proper ordering of array list
                 .alias('collected_list'))
                 .withColumn('TS_IN_NH_TODAY', col(f"collected_list.in_nh_today"))
                 .drop("collected_list")
                ).orderBy('bene_id')

# COMMAND ----------

df_wtargets_cc_demos_nh = df_wtargets_cc_demos.join(NH_daily, 'bene_id', 'inner')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write table

# COMMAND ----------

# Change data types to compress file a bit
df_wtargets_cc_demos_nh = (df_wtargets_cc_demos_nh
 .select([col for col in df_wtargets_cc_demos_nh.columns if col.startswith('TS')==False]+
   [F.col(col).cast(ArrayType(ShortType())).alias(col) for col in df_wtargets_cc_demos_nh.columns if col.startswith('TS_CLMLINES')]+
   [F.col(col).cast(ArrayType(FloatType())).alias(col) for col in df_wtargets_cc_demos_nh.columns if col.startswith('TS_MME')]+
   [F.col(col).cast(ArrayType(ByteType())).alias(col) for col in df_wtargets_cc_demos_nh.columns if col.startswith('TS_FLAG')]+
   [F.col(col).cast(ArrayType(ShortType())).alias(col) for col in df_wtargets_cc_demos_nh.columns if col.startswith('TS_CUM')]+
   [F.col(col).cast(ArrayType(ShortType())).alias(col) for col in df_wtargets_cc_demos_nh.columns if col.startswith('TS_DQ')]
 )
           )

# COMMAND ----------

# Extra cleanup - drop unnecessary and imputation

median_ruca = df_wtargets_cc_demos_nh.groupBy().agg(F.percentile_approx('STATIC_DEMO_RUCA1', 0.5)).collect()[0][0]
df_wtargets_cc_demos_nh = df_wtargets_cc_demos_nh.drop('STATIC_DEMO_ADI_NATRANK','STATIC_DEMO_ADI_STATERNK_str')
df_wtargets_cc_demos_nh = df_wtargets_cc_demos_nh.fillna({"STATIC_DEMO_state_cd":"Unassigned", 'STATIC_DEMO_RUCA1': median_ruca}, subset=['STATIC_DEMO_state_cd', 'STATIC_DEMO_RUCA1'])

# COMMAND ----------

df_wtargets_cc_demos_nh.write \
  .format("delta")\
  .option("overwriteSchema", "true")\
  .mode("overwrite") \
  .saveAsTable(f"eldb.opioid_SA_LA_hosp_sktime_table")         
         
