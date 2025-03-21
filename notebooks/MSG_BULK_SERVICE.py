# Databricks notebook source
#%pip install vowpalwabbit
%pip install paramiko
%pip install python-gnupg
%pip install numba
#%pip install /dbfs/FileStore/absplatform/libs/connection-helpers/connection_helpers-0.0.7-py3-none-any.whl

# COMMAND ----------

import os
import time
import json
import re
from datetime import datetime
import uuid

# Third-Party Library Imports
import base64
import pytz
import numpy as np
import paramiko
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType, IntegerType,
                               TimestampType, BooleanType, DateType, DecimalType, ArrayType)
from pyspark.sql.functions import (lit, current_timestamp, regexp_extract, col,
                                   to_timestamp, date_format, to_json, struct,
                                   collect_list, array_join, expr, concat_ws, explode, from_json, row_number, desc)
from pyspark.sql.window import Window
#import vowpalwabbit

# Local Imports
from gcp_utils.gcp_util import *
from connection_helpers.bq_helper import *
#from absplatform import snowflake

from recipeRunner import recipes_extractor
from rt_functions import *


# COMMAND ----------

env = dbutils.widgets.get("ENV")
print(env)


# COMMAND ----------

gcp_env,domain=("NONPRD","wmpz") if env=='DEV' else ("PRD","dppz")


# COMMAND ----------

gcp_env

# COMMAND ----------

gcon = GCPConnection(env=gcp_env, domain=domain)
scope, credentials, parent_project, feature_project_id, bq_dataset, materialization_dataset, bqvw_pid, gcs_bucket = gcon.gcp_connection_setup()

# COMMAND ----------


scope = "SNF-DOPS-AUTH-APP-AA01MP-SCP"

if env == 'DEV':
    aas_url = dbutils.secrets.get(scope=scope, key="AzureSQLDevExpPlatformConnectionStr")
    aas_user = dbutils.secrets.get(scope=scope, key="AzureSQLDevExpPlatformUsername")
    aas_pass = dbutils.secrets.get(scope=scope, key="AzureSQLDevExpPlatformPassword")

    b2b_user = dbutils.secrets.get(scope=scope, key="AzureMSGDevB2bUser")
    b2b_pwd = dbutils.secrets.get(scope=scope, key="AzureMSGDevB2bKey")
    sfmc_pwd = dbutils.secrets.get(scope=scope, key="AzureMSGDevSfmcPublicKey")

    dbricks_env = 'DEV'

    sftp_host_name = 'mft-qa-int.albertsons.com'
    storage_account='aampmlpdevst01'


elif env == 'PRD' or env == 'PROD' or env == 'prd' or env == 'prod':
    aas_url = dbutils.secrets.get(scope=scope, key="AzureSQLProdExpPlatformConnectionStr")
    aas_user = dbutils.secrets.get(scope=scope, key="AzureSQLProdExpPlatformUsername")
    aas_pass = dbutils.secrets.get(scope=scope, key="AzureSQLProdExpPlatformPassword")

    b2b_user = dbutils.secrets.get(scope=scope, key="AzureMSGProdB2bUser")
    b2b_pwd = dbutils.secrets.get(scope=scope, key="AzureMSGProdB2bKey")
    sfmc_pwd = dbutils.secrets.get(scope=scope, key="AzureMSGProdSfmcPublicKey")

    dbricks_env = 'prod'
    sftp_host_name = 'mft-int.albertsons.com'
    storage_account='aampmlpprodst01'




# COMMAND ----------

query_tag_bq = {
    "project_version": "02",
    "app_code": "aamp",
    "project_name":"msg_bulk_service",
    "env":dbricks_env.lower(),
    "portfolio_name":"digital_personalization",
    "object_name":"aaml_pzn_msg_bulk_service"
}

# COMMAND ----------

InputContainer = dbutils.widgets.get("InputContainer")
InputFolder = dbutils.widgets.get("InputFolder")
InputFileName = dbutils.widgets.get("InputFile")


# COMMAND ----------

pattern = r'(.*)/[^/]+/?$'
# app_path = InputContainer + '/' + re.sub(pattern, r'\1',InputFolder)
app_path = re.sub(pattern, r'\1', InputFolder)
input_file = 'dbfs:/mnt/' + app_path + '/input/' + InputFileName
archive_file = 'dbfs:/mnt/' + app_path + '/archive/' + InputFileName
temp_path = app_path + '/temp/'
temp_path_model = 'dbfs:/mnt/' + app_path + '/temp/'
output_path = 'dbfs:/mnt/' + app_path + '/output/'
output_file_old = output_path + InputFileName[:-21] + '_Output' + InputFileName[-15:]
output_file_tokens = InputFileName.split('Input')
output_file = output_path + output_file_tokens[0] + 'Output' + output_file_tokens[1]
dest_file_name = output_file_tokens[0] + 'Output' + output_file_tokens[1]
input_path = 'dbfs:/mnt/' + app_path + '/input/'
archive_path = 'dbfs:/mnt/' + app_path + '/archive/'
local_tmp_path = '/tmp/msg/'
encrypted_file_name = dest_file_name + '.pgp'

print(input_file)
print(output_file_old)
print(output_file)
print(temp_path_model)
print(temp_path)
print(input_path)
print(archive_path)
print(dest_file_name)
print(local_tmp_path)
print(encrypted_file_name)

# COMMAND ----------

# Cleaning up temp directory
try:
    files = dbutils.fs.ls(local_tmp_path)
    if len(files) > 0:
        print(len(files))
        for file in files:
            print(file.path)
            dbutils.fs.rm(file.path, recurse=True)
except Exception as e:
    # print(e)
    if 'java.io.FileNotFoundException' in str(e):
        # create temp directory for MSG
        dbutils.fs.mkdirs(local_tmp_path)
    else:
        raise e

# COMMAND ----------

dbutils.fs.ls(local_tmp_path)

# COMMAND ----------

# Cleaning up temp model path directory
# example /mnt/msg/emcn/temp/
try:
    files = dbutils.fs.ls(temp_path_model)
    if len(files) > 0:
        print(len(files))
        for file in files:
            print(file.path)
            dbutils.fs.rm(file.path, recurse=True)
except Exception as e:
    # print(e)
    if 'java.io.FileNotFoundException' in str(e):
        # create temp directory for MSG
        dbutils.fs.mkdirs(temp_path_model)
    else:
        raise e

# COMMAND ----------

dbutils.fs.ls(temp_path_model)

# COMMAND ----------

folders = app_path.split(os.path.sep)
if folders[-1] == '':
    folders = folders[:-1]
app_id = folders.pop(-1)

# COMMAND ----------

url = aas_url
properties = {
    "user": aas_user,
    "password": aas_pass,
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# COMMAND ----------

spark = SparkSession.builder.appName("ReadCSV") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memoryOverhead", "2g") \
    .config("spark.driver.memoryOverhead", "2g") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "2") \
    .config("spark.dynamicAllocation.maxExecutors", "10") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
    .getOrCreate()



# COMMAND ----------

spark

# COMMAND ----------

query = "dbo.BulkService"
df_config = spark.read.jdbc(url=url, table=query, properties=properties)
display(df_config)

# COMMAND ----------

df_config = df_config[df_config.app_id == app_id]
display(df_config)

# COMMAND ----------

schema = StructType([
    StructField("HOUSEHOLD_ID", StringType(), True),
    StructField("MODEL_CONFIGURATION_ID", StringType(), True),
    StructField("CONTEXT", StringType(), True)
])

# COMMAND ----------

# access_token = dbutils.secrets.get(scope="aamp-dev-wu-kv-02-scp", key="aampmlpdevst01-access-token")
# spark.conf.set("fs.azure.account.key.aampmlpdevst01.blob.core.windows.net",access_token)

# COMMAND ----------

if dbricks_env == 'DEV':
    access_token = dbutils.secrets.get(scope="aamp-dev-wu-kv-02-scp", key="aampmlpdevst01-access-token")
    spark.conf.set("fs.azure.account.key.aampmlpdevst01.blob.core.windows.net",access_token)
else:
    access_token = dbutils.secrets.get(scope="aamp-prod-wu-kv-02-scp", key="aampmlpprodst01-access-token")
    spark.conf.set("fs.azure.account.key.aampmlpprodst01.blob.core.windows.net",access_token)

# COMMAND ----------

file_path = f"wasbs://msg@{storage_account}.blob.core.windows.net/emcn/input/{InputFileName}"
print(file_path)
df = spark.read.csv(file_path, header=True, escape='\"')
# df.write.mode("overwrite").csv(input_file)
df.coalesce(1).write.mode("overwrite").csv(input_file)
df.show(n=10, truncate=False)

# COMMAND ----------

# dbutils.fs.head(input_file)

# COMMAND ----------

# df.show()

# COMMAND ----------

random_hash = str(uuid.uuid4()).replace('-', '')

# COMMAND ----------

random_hash

# COMMAND ----------

table_name_bq =f"{feature_project_id}.{bq_dataset}.msg_bulk_hhids_{random_hash}"

# COMMAND ----------

table_name_bq

# COMMAND ----------

# Write to BigQuery
write_to_bq(df, table_name_bq, "overwrite", gcs_bucket,query_tag=query_tag_bq)

# COMMAND ----------

from pyspark.sql.functions import concat, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, LongType
from pyspark.sql.functions import col, from_json,concat_ws, lit
column_name = "model_configuration_id"
df_offergrid = df.filter(col(column_name) == '66bd2bbb78100d5bcb923701').filter(col("CONTEXT").isNotNull())
if df_offergrid.count() > 0:
    json_schema = StructType([
        StructField("page", StructType([
            StructField("offers", ArrayType(StringType()), True)
        ]), True)
    ])

    df_parsed = df_offergrid.withColumn("only_page", from_json(col("CONTEXT"), json_schema))
    df_parsed.show(10)

    # Extract the offers array and convert it to a comma-separated string
    df_offers = df_parsed.withColumn("offers_str", concat_ws(",", col("only_page.page.offers")))

    # Collect the comma-separated offers string
    df_offergrid = df_offers.drop('CONTEXT').drop('only_page')
    df_offergrid.show()

# COMMAND ----------

read_upc_bpn_query = f'''
                            SELECT DISTINCT
    DIGITAL_PRODUCT_UPC.UPC_NBR,
    DIGITAL_PRODUCT_UPC.BASE_PRODUCT_NBR
  FROM
    gcp-abs-udco-bqvw-prod-prj-01.udco_ds_spex.DIGITAL_PRODUCT_UPC
  WHERE DIGITAL_PRODUCT_UPC.BASE_PRODUCT_NBR IN(
    SELECT DISTINCT
        cast(BPN_ID as INT64)
      FROM
        {feature_project_id}.{bq_dataset}.SM_V2_DS_SMART_LIST_RANKING)
                         '''
if dbricks_env == 'DEV':
    offer_grid_hhids='gcp-abs-aamp-wmfs-prod-prj-01.aamp_ds_pz_wkg.offer_grid_bulk_hhids'
    upc_bpn_map='gcp-abs-aamp-wmfs-prod-prj-01.aamp_ds_pz_wkg.drop_upc_bpn_map'
    if df_offergrid.count() > 0:
        write_to_bq(df_offergrid, offer_grid_hhids, "overwrite", gcs_bucket,query_tag=query_tag_bq)
        df_offergrid.show()
        df_upc_bpn = read_from_bq(query=read_upc_bpn_query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)
        write_to_bq(df_upc_bpn, upc_bpn_map, "overwrite", gcs_bucket,query_tag=query_tag_bq)
else:
    offer_grid_hhids='gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.offer_grid_bulk_hhids'
    upc_bpn_map='gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.drop_upc_bpn_map'
    if df_offergrid.count() > 0:
        write_to_bq(df_offergrid, offer_grid_hhids, "overwrite", gcs_bucket,query_tag=query_tag_bq)
        df_offergrid.show()
        df_upc_bpn = read_from_bq(query=read_upc_bpn_query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)
        write_to_bq(df_upc_bpn, upc_bpn_map, "overwrite", gcs_bucket,query_tag=query_tag_bq)
   


# COMMAND ----------

from pyspark.sql.functions import concat,lit

# COMMAND ----------

def getProductThemes(df_write, model_configuration_id, model_id):
    if dbricks_env == 'DEV':
        inputTable = table_name_bq
    else:
        inputTable = table_name_bq

    print(inputTable)

    query = f'''SELECT
    x.household_id,
    CASE WHEN item IS NOT NULL THEN JSON_VALUE(item.bpn_id) else JSON_VALUE(RATING.bpn_id) end  AS RECOMMENDATION_ITEM_ID,
   CASE WHEN item IS NOT NULL THEN JSON_VALUE(item.rnk) else JSON_VALUE(RATING.rnk) end  AS RANK,
    x.theme_id AS META
  FROM
    (
      SELECT
          y.household_id,
          PARSE_JSON(rating).theme_id AS theme_id,
          PARSE_JSON(rating).theme_nm AS theme_name,
          PARSE_JSON(rating).ratings AS rating_data,
          PARSE_JSON(rating) AS rating
        FROM
          (
            SELECT
                aa.*
              FROM
                (
                  SELECT
                      household_id,
                      THEME_RANK,
                      rating,
                      ROW_NUMBER() OVER (PARTITION BY HOUSEHOLD_ID ORDER BY THEME_RANK ASC) AS row_num
                    FROM
                      (
                        SELECT
                            PRD_BATCH_THEMES_DIVERSIFICATION_THEMES_RANKED_RECS_PFY_ALL_FINAL.*
                          FROM
                            `{feature_project_id}.{bq_dataset}.PRD_BATCH_THEMES_DIVERSIFICATION_THEMES_RANKED_RECS_PFY_ALL_FINAL` AS PRD_BATCH_THEMES_DIVERSIFICATION_THEMES_RANKED_RECS_PFY_ALL_FINAL
                            INNER JOIN {inputTable} AS msg_bulk_hhids_test2 ON PRD_BATCH_THEMES_DIVERSIFICATION_THEMES_RANKED_RECS_PFY_ALL_FINAL.household_id = CAST(msg_bulk_hhids_test2.household_id AS INT64)
                          WHERE msg_bulk_hhids_test2.MODEL_CONFIGURATION_ID = '{model_configuration_id}'
                      )
                ) AS aa

              WHERE aa.row_num = 1
          ) AS y
           
    ) AS x
    left join  UNNEST(JSON_EXTRACT_ARRAY(rating.ratings)) AS item
  WHERE  CAST(CASE WHEN item IS NOT NULL THEN JSON_VALUE(item.rnk) else JSON_VALUE(rating.rnk) end AS INT64) < 7'''

    print(query)

    df_features = read_from_bq(query=query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)

    # df_features.show(10)
    json_str_metadata = concat(
        lit('"{""themeId"":""'),
        col("META"), lit('""}"')
    )

    df_banner_hhid = df_features.select("HOUSEHOLD_ID", "RECOMMENDATION_ITEM_ID", "RANK", "META")
    df_banner_hhid = df_banner_hhid.withColumn("METADATA", json_str_metadata
                                               )
    df_banner_hhid = df_banner_hhid.withColumn("timestamp", current_timestamp())
    df_banner_hhid = df_banner_hhid.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))

    # df_banner_hhid.show()

    # RANK = "1"
    RECOMMENDATION_TYPE = "PRODUCT"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id

    df_out = df_banner_hhid.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        "RANK",
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        "METADATA"
    )

    return df_out
# df_gcp=getProductThemes(df, '66d0e06efd6bf45887987c19', 'PRODUCT_THEMES_PHASE1')
# df_gcp.display()


# COMMAND ----------

def getSeasonalThemes(df_write, model_configuration_id, model_id):
    if dbricks_env == 'DEV':
        inputTable = table_name_bq
    else:
        inputTable = table_name_bq

    print(inputTable)

    query = f'''
    SELECT
        x.Household_id,
        CASE 
            WHEN item IS NOT NULL THEN JSON_VALUE(item.bpn_id) 
            ELSE JSON_VALUE(rating.bpn_id) 
        END AS Recommendation_item_id,
        CASE 
            WHEN item IS NOT NULL THEN JSON_VALUE(item.rnk) 
            ELSE JSON_VALUE(rating.rnk) 
        END AS Rank,
        x.theme_id AS Meta,
        ROW_NUMBER() OVER (
            PARTITION BY x.Household_id, 
            CASE 
                WHEN item IS NOT NULL THEN JSON_VALUE(item.bpn_id) 
                ELSE JSON_VALUE(rating.bpn_id) 
            END 
            ORDER BY 
            CASE 
                WHEN item IS NOT NULL THEN JSON_VALUE(item.rnk) 
                ELSE JSON_VALUE(rating.rnk) 
            END
        ) AS row_num
    FROM (
        SELECT
            y.household_id,
            PARSE_JSON(rating).theme_id AS theme_id,
            PARSE_JSON(rating).theme_nm AS theme_name,
            PARSE_JSON(rating).ratings AS rating_data,
            PARSE_JSON(rating) AS rating
        FROM (
            SELECT
                aa.*
            FROM (
                SELECT
                    household_id,
                    THEME_RANK,
                    rating,
                    ROW_NUMBER() OVER (PARTITION BY HOUSEHOLD_ID ORDER BY THEME_RANK ASC) AS row_num
                FROM (
                    SELECT
                        PRD_BATCH_THEMES_DIVERSIFICATION_THEMES_RANKED_RECS_PFY_ALL_FINAL.*
                    FROM
                        `{feature_project_id}.{bq_dataset}.PRD_BATCH_THEMES_DIVERSIFICATION_THEMES_RANKED_RECS_PFY_ALL_FINAL` AS PRD_BATCH_THEMES_DIVERSIFICATION_THEMES_RANKED_RECS_PFY_ALL_FINAL
                    INNER JOIN 
                        {inputTable} AS msg_bulk_hhids_test2 
                    ON 
                        PRD_BATCH_THEMES_DIVERSIFICATION_THEMES_RANKED_RECS_PFY_ALL_FINAL.household_id = CAST(msg_bulk_hhids_test2.household_id AS INT64)
                    WHERE 
                        msg_bulk_hhids_test2.MODEL_CONFIGURATION_ID = '{model_configuration_id}'
                )
            ) AS aa
            WHERE aa.row_num = 1
        ) AS y
    ) AS x
    LEFT JOIN 
        UNNEST(JSON_EXTRACT_ARRAY(rating.ratings)) AS item
    LEFT JOIN 
        gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.TI_EMBEDDING_SEASONAL_PRODUCT_BY_HOLIDAYS AS holiday_data
    ON 
        CAST(
          CASE 
            WHEN item IS NOT NULL THEN JSON_VALUE(item.bpn_id) 
            ELSE JSON_VALUE(rating.bpn_id) 
        END AS INT64) =  holiday_data.bpn_id
    WHERE 
        CAST(CASE 
                WHEN item IS NOT NULL THEN JSON_VALUE(item.rnk) 
                ELSE JSON_VALUE(rating.rnk) 
            END AS INT64) < 7
        AND holiday_data.year in ('2023','2024')
    '''

    print(query)

    df_features = read_from_bq(query=query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)

    # df_features.show(10)
    json_str_metadata = concat(
        lit('"{""themeId"":""'),
        col("META"), lit('""}"')
    )

    df_banner_hhid = df_features.select("HOUSEHOLD_ID", "RECOMMENDATION_ITEM_ID", "RANK", "META")
    df_banner_hhid = df_banner_hhid.withColumn("METADATA", json_str_metadata
                                               )
    df_banner_hhid = df_banner_hhid.withColumn("timestamp", current_timestamp())
    df_banner_hhid = df_banner_hhid.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))

    # df_banner_hhid.show()

    # RANK = "1"
    RECOMMENDATION_TYPE = "PRODUCT"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id

    df_out = df_banner_hhid.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        "RANK",
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        "METADATA"
    )

    return df_out
# df_gcp=getSeasonalThemes(df_distinct_households, '6750a8a1a731e50c9442a1dc', 'SEASONAL_PRODUCT_THEMES')
# display(df_gcp)

# COMMAND ----------

from pyspark.sql.functions import col, to_json, struct, lit, explode, from_json, broadcast, collect_list, udf
from pyspark.sql.types import ArrayType, StringType, IntegerType
from concurrent.futures import ThreadPoolExecutor
import json
from functools import reduce
from datetime import datetime
import pandas as pd


def get_features_data_batch(inputTable, model_config_id):

    recommendations_query = f"""
        SELECT r.HHID, r.RECOMMENDATION
        FROM `gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.PROD_manual_curated_recommendations` r
        JOIN {inputTable} i ON r.HHID = i.household_id
        WHERE i.MODEL_CONFIGURATION_ID = '{model_config_id}'
    """
    
    df_recommendations = read_from_bq(
        query=recommendations_query,
        materialization_dataset=materialization_dataset,
        query_tag=query_tag_bq
    )
    if df_recommendations.rdd.isEmpty():
        return None

    
    df_household_ids = df_recommendations.select("HHID").distinct()

    
    household_features_query = f"""
        SELECT f.*
        FROM `gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.HH_FEATURES_FOR_DL` f
        JOIN {inputTable} i ON f.household_id = i.household_id
        WHERE i.MODEL_CONFIGURATION_ID = '{model_config_id}'
    """
    
    df_household_features = read_from_bq(
        query=household_features_query,
        materialization_dataset=materialization_dataset,
        query_tag=query_tag_bq
    )

    
    df_household_features = df_household_features.join(
        df_household_ids,
        df_household_features.household_id == df_household_ids.HHID,
        how="inner"
    ).drop(df_household_ids.HHID)

    if df_household_features.rdd.isEmpty():
        return None

    df_household_out = df_household_features.select(
        col("household_id").alias("HOUSEHOLD_ID"),
        to_json(struct(*[c for c in df_household_features.columns if c != "household_id"])).alias("FEATURES")
    )

    
    @udf(returnType=ArrayType(StringType()))
    def extract_bpn_ids(recommendation_str):
        try:
            data = json.loads(recommendation_str)
            return [str(item.get('bpnId')) for item in data.get("recommendation", [])]
        except:
            return []

    df_bpn_ids = (
        df_recommendations
        .withColumn("BPN_IDS", extract_bpn_ids(col("RECOMMENDATION")))
        .select("HHID", explode("BPN_IDS").alias("BPN_ID"))
        .distinct()
    )
    if df_bpn_ids.rdd.isEmpty():
        return None

    
    df_household_bpn_map = (
        df_bpn_ids
        .groupBy("HHID")
        .agg(collect_list("BPN_ID").alias("BPN_LIST"))
    )

    
    all_bpn_ids = [row.BPN_ID for row in df_bpn_ids.select("BPN_ID").distinct().collect()]
    all_bpn_ids = [bpn_id for bpn_id in all_bpn_ids if bpn_id is not None and bpn_id != "None"]
    if not all_bpn_ids:
        return None
    
    bpn_ids_str = ', '.join(map(str, all_bpn_ids))

    
    product_features_query = f"""
        SELECT *
        FROM `gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.IT_FEATURES_FOR_DL`
        WHERE BPN_ID IN ({bpn_ids_str})
    """
    
    df_product_features = read_from_bq(
        query=product_features_query,
        materialization_dataset=materialization_dataset,
        query_tag=query_tag_bq
    )
    df_product_out = df_product_features.select(
        col("BPN_ID"),
        to_json(struct(*[c for c in df_product_features.columns if c != "BPN_ID"])).alias("FEATURES")
    )

    
    catalog_features_query = """
        SELECT *
        FROM `gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.CATALOG_FEATURES_FOR_DL`
    """
    df_catalog_features = read_from_bq(
        query=catalog_features_query,
        materialization_dataset=materialization_dataset,
        query_tag=query_tag_bq
    )
    df_catalog_out = df_catalog_features.select(
        col("shelf_feature_index").alias("CATALOG_ID"),
        to_json(struct(*[c for c in df_catalog_features.columns if c != "shelf_feature_index"])).alias("FEATURES")
    )

    return {
        "household_features": df_household_out,
        "item_features": df_product_out,
        "catalog_features": df_catalog_out,
        "household_bpn_map": df_household_bpn_map
    }

# COMMAND ----------

from pyspark.sql.functions import col, to_json, struct, lit, explode, from_json, broadcast, collect_list, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, TimestampType
from rt_re_ranking import e2e_ranking
from pyspark import SparkContext
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json
from datetime import datetime
from functools import reduce
import numpy as np


def getRtReRankProducts(df_write, model_configuration_id, model_id):

    if dbricks_env == 'DEV':
        inputTable = table_name_bq
    else:
        inputTable = table_name_bq
    
    sc = df_write.sql_ctx._sc
    
    
    household_ids_df = df_write.select("household_id").distinct().cache()
    household_ids_list = [row.household_id for row in household_ids_df.collect()]
    
    if not household_ids_list:
        return df_write.sql_ctx.createDataFrame([], schema=df_write.schema)

    
    features = get_features_data_batch(inputTable, model_config_id=model_configuration_id)
    if not features:
        household_ids_df.unpersist()
        return df_write.sql_ctx.createDataFrame([], schema=df_write.schema)

    
    df_household_features = broadcast(features["household_features"])
    df_household_bpn_map = broadcast(features["household_bpn_map"])
    
    item_features_dict = sc.broadcast(
        features["item_features"].rdd
        .map(lambda row: (str(row["BPN_ID"]), row["FEATURES"]))  
        .collectAsMap()
    )

    
    catalog_features_pd = features["catalog_features"].toPandas()
    df_catalog_extracted = pd.json_normalize(catalog_features_pd["FEATURES"].apply(json.loads))
    df_catalog_features_pd_final = pd.concat(
        [catalog_features_pd[["CATALOG_ID"]].reset_index(drop=True), df_catalog_extracted],
        axis=1
    )

    output_schema = StructType([
        StructField("HOUSEHOLD_ID", StringType()),
        StructField("RECOMMENDATION_ITEM_ID", StringType()),
        StructField("RECOMMENDATION_CREATE_TS", StringType()),
        StructField("RANK", IntegerType()),
        StructField("RECOMMENDATION_TYPE", StringType()),
        StructField("EXPERIMENT_ID", StringType()),
        StructField("EXPERIMENT_VARIANT", StringType()),
        StructField("MODEL_CONFIGURATION_ID", StringType()),
        StructField("MODEL_ID", StringType()),
        StructField("METADATA", StringType())
    ])

    def rank_udf(pdf: pd.DataFrame) -> pd.DataFrame:
        if pdf.empty:
            return pd.DataFrame(columns=output_schema.fieldNames())

        household_id = str(pdf["household_id"].iloc[0])
        hh_features_json = pdf["FEATURES"].iloc[0]
        hh_features_dict = json.loads(hh_features_json) if hh_features_json else {}
        hh_df = pd.DataFrame([{**hh_features_dict, "HOUSEHOLD_ID": household_id}])
        
        bpn_ids = set()
        for ids in pdf["bpn_ids"].dropna():
            if isinstance(ids, (list, np.ndarray)):
                bpn_ids.update(str(id_) for id_ in ids)
            else:
                bpn_ids.add(str(ids))
        
        if not bpn_ids:
            return pd.DataFrame(columns=output_schema.fieldNames())

        local_item_dicts = []
        for bpn in bpn_ids:
            if bpn in item_features_dict.value:
                try:
                    features = json.loads(item_features_dict.value[bpn])
                    local_item_dicts.append({"BPN_ID": bpn, **features})
                except (json.JSONDecodeError, TypeError):
                    continue
        
        if not local_item_dicts:
            return pd.DataFrame(columns=output_schema.fieldNames())
            
        it_features_pd = pd.DataFrame(local_item_dicts)
        print(it_features_pd)
        ranked_df = e2e_ranking(
            hh_df=hh_df,
            it_df=it_features_pd,
            digital_catalog_embedding_df=df_catalog_features_pd_final,
            version="simple",
            parameters=None,
            do_diversify=False
        )

        if ranked_df is None or ranked_df.empty:
            return pd.DataFrame(columns=output_schema.fieldNames())

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ranked_df = ranked_df.assign(
            HOUSEHOLD_ID=household_id,
            RECOMMENDATION_CREATE_TS=current_time,
            RECOMMENDATION_TYPE="PRODUCT",
            EXPERIMENT_ID="",
            EXPERIMENT_VARIANT="",
            MODEL_CONFIGURATION_ID=model_configuration_id,
            MODEL_ID=model_id,
            METADATA=""
        )

        ranked_df = (ranked_df
            .rename(columns={
                "BPN_ID": "RECOMMENDATION_ITEM_ID",
                "RANKING": "RANK"
            })
            .reindex(columns=output_schema.fieldNames()))
        
        if "RANK" not in ranked_df.columns:
            ranked_df["RANK"] = 1

        return ranked_df

    df_prepared = (
        df_write
        .join(df_household_features, 
              df_write.household_id == df_household_features.HOUSEHOLD_ID, "inner")
        .drop(df_household_features.HOUSEHOLD_ID)
        .join(df_household_bpn_map,
              df_write.household_id == df_household_bpn_map.HHID, "left")
        .drop(df_household_bpn_map.HHID)
        .withColumnRenamed("BPN_LIST", "bpn_ids")
        .cache()
    )

    df_ranked = df_prepared.groupBy("household_id").applyInPandas(
        rank_udf, 
        schema=output_schema
    )

    
    df_prepared.unpersist()
    household_ids_df.unpersist()
    item_features_dict.unpersist()

    return df_ranked



# df_gcp = getRtReRankProducts(df, '677da25d672be50fc84a1499', 'RT_RE_RANKING_V1')
# display(df_gcp)
# display(df_gcp.count())

# COMMAND ----------

def getCategoryTiles(df_write, model_configuration_id, model_id):
    
    if dbricks_env == 'DEV':
        inputTable = table_name_bq
    else:
        inputTable = table_name_bq

    print(inputTable)
    
    query = f'''WITH RankedTransactions AS (
    SELECT 
        a.HOUSEHOLD_ID, 
        a.BPN_ID, 
        a.LAST_UPDATE_TS, 
        a.DEPT_RANK_FINAL, 
        a.DEPT_ID,  
        ROW_NUMBER() OVER (PARTITION BY a.HOUSEHOLD_ID, a.DEPT_ID ORDER BY a.LAST_UPDATE_TS DESC) AS rank
    FROM 
        `gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.SM_V2_DS_SMART_LIST_RANKING` a
    INNER JOIN 
        (
            SELECT DISTINCT HOUSEHOLD_ID 
            FROM `{inputTable}`
            WHERE MODEL_CONFIGURATION_ID = '{model_configuration_id}'
        ) b
    ON 
        CAST(a.HOUSEHOLD_ID AS STRING) = CAST(b.HOUSEHOLD_ID AS STRING)
    WHERE 
        a.LAST_UPDATE_TS IS NOT NULL
),
TopTransactions AS (
    -- Select only the top 4 unique DEPT_IDs per household
    SELECT 
        HOUSEHOLD_ID, 
        BPN_ID, 
        LAST_UPDATE_TS, 
        DEPT_RANK_FINAL, 
        DEPT_ID,  
        ROW_NUMBER() OVER (PARTITION BY HOUSEHOLD_ID ORDER BY LAST_UPDATE_TS DESC) AS overall_rank
    FROM 
        RankedTransactions
    WHERE 
        rank = 1
),
FilteredTopTransactions AS (
    SELECT 
        HOUSEHOLD_ID, 
        BPN_ID, 
        LAST_UPDATE_TS, 
        DEPT_RANK_FINAL, 
        DEPT_ID,  
        overall_rank
    FROM 
        TopTransactions
    WHERE 
        overall_rank <= 4
),
PrimaryStore AS (
    SELECT 
        HOUSEHOLD_ID, 
        PRIM_STORE_ID 
    FROM 
        `gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_PROFILE`
),
DepartmentDetails AS (
    SELECT 
        t.HOUSEHOLD_ID, 
        t.BPN_ID, 
        p.PRIM_STORE_ID, 
        t.DEPT_ID,  
        t.overall_rank,  
        COALESCE(d1.DEPARTMENT_DIGITAL_NM, d2.DEPARTMENT_DIGITAL_NM) AS DEPARTMENT_DIGITAL_NM
    FROM 
        FilteredTopTransactions t
    INNER JOIN 
        PrimaryStore p
    ON 
        t.HOUSEHOLD_ID = p.HOUSEHOLD_ID
    LEFT JOIN 
        `gcp-abs-udco-bqvw-prod-prj-01.udco_ds_mrch.D1_UPC_ROG_STORE` d1
    ON 
        CAST(p.PRIM_STORE_ID AS STRING) = d1.FACILITY_NBR 
        AND CAST(t.BPN_ID AS INT64) = d1.BASE_PRODUCT_NBR
        AND d1.DW_LOGICAL_DELETE_IND = false  
    LEFT JOIN 
        `gcp-abs-udco-bqvw-prod-prj-01.udco_ds_spex.DIGITAL_PRODUCT_LST` d2  
    ON 
        CAST(t.BPN_ID AS INT64) = d2.BASE_PRODUCT_NBR
)
SELECT DISTINCT
    HOUSEHOLD_ID,
    PRIM_STORE_ID, 
    BPN_ID, 
    DEPT_ID,  
    DEPARTMENT_DIGITAL_NM,
    overall_rank  
FROM 
    DepartmentDetails
WHERE 
    DEPARTMENT_DIGITAL_NM IS NOT NULL
ORDER BY 
    HOUSEHOLD_ID, overall_rank
    '''
    print(query)
    model_data = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)

    model_data = model_data.withColumn("timestamp", current_timestamp())
    model_data = model_data.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))

    RECOMMENDATION_TYPE = "CATEGORY"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    METADATA = ''

    df_out = model_data.select(
        "HOUSEHOLD_ID",
        col("DEPT_ID").alias("RECOMMENDATION_ITEM_ID"),
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        col("overall_rank").alias("RANK"),
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(model_configuration_id).alias("MODEL_CONFIGURATION_ID"),
        lit(model_id).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        lit(METADATA).alias("METADATA")
    )

    return df_out

# df_gcp=getCategoryTiles(df1, '67a122f51f812809d3b012cd', 'CATEGORY_TILES')
# df_gcp.display()
# df_gcp.count()

# COMMAND ----------

def getThemes(df_write, model_configuration_id, model_id):
    if dbricks_env == 'DEV':
        inputTable = table_name_bq
    else:
        inputTable = table_name_bq

    print(inputTable)

    query = f'''WITH inp_json AS (SELECT PARSE_JSON(RATING) AS VALUE,* FROM
{feature_project_id}.{bq_dataset}.PRD_BATCH_THEMES_DIVERSIFICATION_THEMES_RANKED_RECS_PFY_ALL_FINAL)
 
SELECT
a.HOUSEHOLD_ID,
JSON_VALUE(VALUE.theme_id) AS RECOMMENDATION_ITEM_ID,
a.THEME_RANK AS RANK,
JSON_VALUE(VALUE.theme_nm) AS META
FROM
  inp_json a
JOIN (
  SELECT
    HOUSEHOLD_ID
  FROM
    {inputTable}
  WHERE
    MODEL_CONFIGURATION_ID='{model_configuration_id}')b
ON
  a.HOUSEHOLD_ID = CAST(b.HOUSEHOLD_ID AS INT64)
ORDER BY
  1,
  3 ASC
    '''

    print(query)
    df_features = read_from_bq(query=query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)
    

    # df_features.show(10)
    json_str_metadata = concat(
        lit('"{""themeName"":""'),
        col("META"), lit('""}"')
    )
    df_banner_hhid = df_features.select("HOUSEHOLD_ID", "RECOMMENDATION_ITEM_ID", "RANK", "META")
    df_banner_hhid = df_banner_hhid.withColumn("METADATA", json_str_metadata)
    df_banner_hhid = df_banner_hhid.withColumn("timestamp", current_timestamp())
    df_banner_hhid = df_banner_hhid.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))

    # df_banner_hhid.show()

    # RANK = "1"
    RECOMMENDATION_TYPE = "THEME"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id

    df_out = df_banner_hhid.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        "RANK",
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        "METADATA"
    )

    return df_out
# df_gcp=getThemes(df, '66d0e076202a0e74e38aecf9', 'THEMES_PHASE1')
# df_gcp.display()


# COMMAND ----------

def getOfferTopDeals(df_write, model_configuration_id, model_id):
    if dbricks_env == 'DEV':
        inputTable = table_name_bq
    else:
        inputTable = table_name_bq

    print(inputTable)

    query = f'''
    WITH vt_offers AS (
     SELECT a.oms_offer_Id, a.external_offer_id
    FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_mrch.OMS_OFFER a        
    WHERE a.PROGRAM_CD IN ('PD', 'SPD', 'BPD')
      AND a.PROGRAM_TYPE_CD <> 'CONTINUITY'
      AND a.OFFER_PROTOTYPE_CD <> 'CONTINUITY'
	  AND a.offer_status_cd='PU'
	  AND a.POD_REFERENCE_OFFER_ID='NA'
      AND a.display_effective_end_dt > CURRENT_DATE 
      AND a.DW_LAST_EFFECTIVE_DT > CURRENT_DATE
	  AND a.DISPLAY_EFFECTIVE_START_DT > CURRENT_DATE - 30
      AND a.dw_current_version_ind = TRUE
      AND a.dw_logical_delete_ind = FALSE
),
 
offer_hhids AS (
    SELECT DISTINCT a.household_id, a.req_id, a.rnk
    FROM (SELECT household_id,req_id,rnk,week_id FROM gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.JAA_CONSOL_ALLOC WHERE PROGRAM_TYPE = 'DivWkly'
          AND STRATEGY LIKE 'PC-JIT%' AND week_id >= (SELECT (max_week_id-1) as week_id FROM (
  SELECT max(week_id) as max_week_id FROM gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.JAA_CONSOL_ALLOC
  WHERE PROGRAM_TYPE = 'DivWkly' AND STRATEGY LIKE 'PC-JIT%' AND LAST_UPDATE > CURRENT_DATE - 60
)) AND LAST_UPDATE > CURRENT_DATE - 60
    )a 
  JOIN (SELECT HOUSEHOLD_ID,CLIENT_OFFER_ID,WEEK_ID FROM gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.J4U_OFFER_ALLOCATION_HIST WHERE LAST_UPDATE_TS > (CURRENT_DATE - 30)
       )b ON a.household_id = b.HOUSEHOLD_ID AND a.week_id = b.WEEK_ID AND a.req_id = b.CLIENT_OFFER_ID
  JOIN (SELECT HOUSEHOLD_ID
    FROM {inputTable}
    WHERE MODEL_CONFIGURATION_ID = '{model_configuration_id}')d ON CAST(d.household_id AS INT64)= a.household_id
    JOIN vt_offers c
      ON a.req_id = c.external_offer_id  
),
not_clipped AS (
    SELECT b.household_id, b.req_id, b.rnk
    FROM offer_hhids b
    LEFT JOIN  gcp-abs-udco-bqvw-prod-prj-01.udco_ds_loyl.COUPON_CLIP a
        ON a.external_offer_id = b.req_id
          AND a.household_id = b.household_id 
          AND a.dw_current_version_ind
          AND NOT a.dw_logical_delete_ind 
          AND a.clip_type_cd = 'C' 
          AND a.CLIP_DT > CURRENT_DATE - 90
		  WHERE external_offer_id is null

)

SELECT cast (a.HOUSEHOLD_ID as string) AS HOUSEHOLD_ID, a.req_id AS RECOMMENDATION_ITEM_ID, a.row_num AS RANK
FROM ( SELECT HOUSEHOLD_ID, req_id, 
           ROW_NUMBER() OVER (PARTITION BY HOUSEHOLD_ID ORDER BY rnk DESC) AS row_num
    FROM not_clipped
) a WHERE a.row_num < 7
'''

    print(query)

    df_features =  read_from_bq(query=query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)

    # df_features.show(10)

    df_banner_hhid = df_features.select("HOUSEHOLD_ID", "RECOMMENDATION_ITEM_ID", "RANK")
    df_banner_hhid = df_banner_hhid.withColumn("timestamp", current_timestamp())
    df_banner_hhid = df_banner_hhid.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))

    # df_banner_hhid.show()

    #RANK = "1"
    RECOMMENDATION_TYPE = "OFFER"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id
    METADATA = ''

    df_out = df_banner_hhid.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        "RANK",
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        lit(METADATA).alias("METADATA")
    )
    #df_out.show()

  
    return df_out
# df_gcp=getOfferTopDeals(df, '668dad8ef90739630e279165', 'OFFER_TD_PHASE1')
# df_gcp.display()


# COMMAND ----------

def getOfferBonusPath(df_write, model_configuration_id, model_id):
    if dbricks_env == 'DEV':
        inputTable = table_name_bq
    else:
        inputTable = table_name_bq

    query = f'''WITH vt_offers as (
SELECT a.oms_offer_Id
       , a.external_offer_id
       , a.offer_status_cd
       , CASE WHEN a.offer_status_cd = 'PU' THEN 'EARN' 
              WHEN a.offer_status_cd = 'DE' THEN 'BURN' 
              ELSE 'NEED QC' 
        END AS offer_comp
      , a.offer_prototype_cd
      , a.programsubtype
      , a.display_effective_start_dt
      , a.display_effective_end_dt
      , a.DW_FIRST_EFFECTIVE_DT
      , a.DW_LAST_EFFECTIVE_DT 
      , a.pod_reference_offer_id
      , a.is_primary_pod_offer_ind
      , a.effective_start_dt
      , a.effective_end_dt 
      , b.points_program_nm
      FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_mrch.OMS_OFFER a  
      join gcp-abs-udco-bqvw-prod-prj-01.udco_ds_mrch.OMS_OFFER_BENEFIT_POINTS b
      on a.oms_offer_id =  b.oms_offer_id      
      where 1=1 
      AND a.offer_prototype_cd = 'CONTINUITY'
      AND a.DW_LAST_EFFECTIVE_DT > CURRENT_DATE 
      AND a.EFFECTIVE_END_DT > CURRENT_DATE
      AND a.offer_status_cd in ('DE','PU') 
      AND a.programsubtype in ('PRODUCE', 'MEAT SEAFOOD','DAIRY','FROZEN','CATEGORY FS MEAT SEAFOOD')
      AND a.dw_current_version_ind
      AND NOT a.dw_logical_delete_ind
      AND b.dw_current_version_ind
      AND NOT b.dw_logical_delete_ind
) 
, allocated_offers as (SELECT b.household_id
                       ,a.oms_offer_Id, a.external_offer_id, a.offer_comp,a.points_program_nm,a.pod_reference_offer_id,
                       a.effective_start_dt,a.effective_end_dt 
FROM vt_offers  a
join (SELECT * FROM gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.J4U_OFFER_ALLOCATION_HIST WHERE last_update_ts >= current_date - 90
     )b on b.client_offer_id  = a.external_offer_id
join (SELECT HOUSEHOLD_ID FROM {inputTable} where MODEL_CONFIGURATION_ID='{model_configuration_id}') c 
                       on b.household_id = CAST(c.household_id AS INT64)  
) 
, not_clipped as (
    select b.household_id,b.oms_offer_Id, b.external_offer_id,'READY_TO_ACTIVATE' as meta_data 
                from allocated_offers b left join  gcp-abs-udco-bqvw-prod-prj-01.udco_ds_loyl.COUPON_CLIP a
                                                on a.external_offer_id = b.external_offer_id and a.household_id=b.household_id 
                                                AND a.dw_current_version_ind
                                                and NOT a.dw_logical_delete_ind   
                                                and a.clip_type_cd='C' 
                                                AND a.CLIP_DT > current_date - 90   where a.external_offer_id is null
) 
, clipped as (
    select b.household_id,b.oms_offer_Id, b.external_offer_id,'IN_PROGRESS' as meta_data,b.effective_start_dt
                from allocated_offers b 
                join (SELECT * FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_loyl.COUPON_CLIP aa WHERE  
                          aa.dw_current_version_ind 
                          and NOT aa.dw_logical_delete_ind     
                          and aa.clip_type_cd='C' 
                          AND aa.CLIP_DT > current_date - 90
                        )a 
                ON a.external_offer_id = b.external_offer_id and a.household_id=b.household_id 
) ,burned_offers as (
    SELECT b.household_id, b.external_offer_id, 'COMPLETED' as meta_data
    FROM allocated_offers b
    JOIN (
        SELECT aggs.household_id, aggs.offer_id AS earn_offer_id, aggs.sum_points, 
               w.oms_offer_id AS burn_offer_id, w.points_group_value_amt AS challenge_goal
        FROM (
            SELECT a.household_id, sp.offer_id, Sum(sp.points_earned_nbr) AS sum_points
            FROM (
                SELECT * 
                FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_loyl.EPE_TRANSACTION_HEADER 
                WHERE status_cd = 'COMPLETED' 
                AND dw_current_version_ind
                and NOT dw_logical_delete_ind
                AND TRANSACTION_TS > CURRENT_DATE - 90
            ) a
            JOIN (
                SELECT transaction_integration_id, offer_id, points_earned_nbr
                FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_loyl.EPE_TRANSACTION_HEADER_SAVING_POINTS
                WHERE dw_current_version_ind and NOT dw_logical_delete_ind
            ) sp ON a.transaction_integration_id = sp.transaction_integration_id
            WHERE a.TRANSACTION_TS > current_date - 90
            GROUP BY a.household_id, sp.offer_id
        ) aggs
        JOIN (
            SELECT * 
            FROM vt_offers 
            WHERE NOT is_primary_pod_offer_ind
            AND programsubtype in ('PRODUCE', 'MEAT SEAFOOD','DAIRY','FROZEN','CATEGORY FS MEAT SEAFOOD')
        ) v ON concat(aggs.offer_id, '-D') = v.pod_reference_offer_id
        JOIN (
            SELECT * 
            FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_mrch.OMS_OFFER_QUALIFICATION_POINTS_GROUP
            WHERE dw_current_version_ind
        ) w ON v.oms_offer_id = w.oms_offer_id
    ) aa ON b.household_id = aa.household_id AND b.oms_offer_id = aa.burn_offer_id
    WHERE aa.sum_points >= aa.challenge_goal
)
SELECT * FROM (
    SELECT cast (household_id as STRING) as HOUSEHOLD_ID, external_offer_id as RECOMMENDATION_ITEM_ID, meta_data as META from not_clipped
    UNION DISTINCT
    SELECT cast (household_id as STRING) as HOUSEHOLD_ID, external_offer_id as RECOMMENDATION_ITEM_ID, meta_data as META from clipped
    UNION DISTINCT
    SELECT cast (household_id as STRING) as HOUSEHOLD_ID, external_offer_id as RECOMMENDATION_ITEM_ID, meta_data as META from burned_offers
) aa

'''

    print(query)

    df_features =  read_from_bq(query=query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)

    # df_features.show(10)

    # df_banner_hhid = df_features.select("HOUSEHOLD_ID",'RECOMMENDATION_ITEM_ID','META')
    # df_banner_hhid = df_features.select("HOUSEHOLD_ID",'RECOMMENDATION_ITEM_ID','METADATA')
    # df_banner_hhid = df_banner_hhid.withColumn("timestamp", current_timestamp())
    # df_banner_hhid = df_banner_hhid.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))

    json_str_metadata = concat(
        lit('"{""bonusPathStatus"":""'),
        col("META"), lit('""}"')
    )

    df_banner_hhid = df_features.select("HOUSEHOLD_ID", 'RECOMMENDATION_ITEM_ID', 'META')
    df_banner_hhid = df_banner_hhid.withColumn("METADATA", json_str_metadata
                                               )
    df_banner_hhid = df_banner_hhid.withColumn("timestamp", current_timestamp())
    df_banner_hhid = df_banner_hhid.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))
    df_banner_hhid = df_banner_hhid.drop("META")


    RANK = "1"
    RECOMMENDATION_TYPE = "OFFER"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id

    df_out = df_banner_hhid.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        lit(RANK).alias("RANK"),
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        "METADATA",
    )
    #df_out.show()
    return df_out
# df_gcp=getOfferBonusPath(df, '668dad8ef90739630e279153', 'OFFER_BP_PHASE1')
# df_gcp.display()

# COMMAND ----------

def getBuyItAgainProducts(df_write, model_configuration_id, model_id):

    if dbricks_env == 'DEV':
        biaTable = "gcp-abs-aamp-wmfs-prod-prj-01.aamp_ds_pz_wkg.SM_V2_DS_SMART_LIST_RANKING"
        inputTable = table_name_bq
    else:
        biaTable = "gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.SM_V2_DS_SMART_LIST_RANKING"
        inputTable = table_name_bq


    query = f'''SELECT b.HOUSEHOLD_ID as HOUSEHOLD_ID, BPN_ID as RECOMMENDATION_ITEM_ID, bpn_rank_final*-1 as RANK, date_of_rec
            from {biaTable} a INNER 
            JOIN (SELECT HOUSEHOLD_ID FROM {inputTable} WHERE MODEL_CONFIGURATION_ID='{model_configuration_id}') b on a.household_id  = CAST(b.household_id AS INT64) WHERE SMART_BASKET_FLAG = 1 order by b.HOUSEHOLD_ID, bpn_rank_final desc
            '''
    print(query)

    df_edm =  read_from_bq(query=query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)
    df_edm = df_edm.withColumn("timestamp", to_timestamp("date_of_rec"))
    df_edm = df_edm.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))

    RECOMMENDATION_TYPE = "PRODUCT"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id
    METADATA = ''

    df_out = df_edm.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        "RANK",
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        lit(METADATA).alias("METADATA")
    )

    return df_out

# df_gcp=getBuyItAgainProducts(df, '65cba460d8e1cd1dc8057369', 'BUY_IT_AGAIN_V2_HOME_PAGE')
# df_gcp.display()



# COMMAND ----------

def getEglBanners(df_write, model_configuration_id, model_id):

    if dbricks_env == 'DEV':
        inputTable = table_name_bq
    else:
        inputTable = table_name_bq

    print(inputTable)

    query = f'''WITH latest_segment as (
        SELECT HOUSEHOLD_ID,EXPORT_TS,AIQ_SEGMENT_NM 
        FROM (
          SELECT HOUSEHOLD_ID,EXPORT_TS, AIQ_SEGMENT_NM,
           ROW_NUMBER() OVER (PARTITION BY HOUSEHOLD_ID ORDER BY EXPORT_TS DESC) AS rn 
          FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.RETAIL_CUSTOMER_BACKFEED_ACTIVATION 
          WHERE dw_current_version_ind = TRUE AND aiq_segment_nm IN ('P13N Engagement Ladder DUG or Delivery Purchase',
                                                                       'P13N Engagement Ladder Default Audience',
                                                                       'P13N Engagement Ladder FreshPass Customer Sign-Up',
                                                                       'P13N Engagement Ladder Schedule and Save Sign-Up'
                                                                      ) 
        )ss WHERE rn=1
        )
            select c.HOUSEHOLD_ID,
            COALESCE(
            CASE 
                WHEN a.aiq_segment_nm='P13N Engagement Ladder DUG or Delivery Purchase' THEN 'egl-dpurchase-092622'
                WHEN a.aiq_segment_nm='P13N Engagement Ladder FreshPass Customer Sign-Up' THEN 'egl-fpass-092622'
                WHEN a.aiq_segment_nm='P13N Engagement Ladder Schedule and Save Sign-Up' THEN 'egl-sns-0926022'
                WHEN a.aiq_segment_nm='P13N Engagement Ladder Default Audience' THEN 'egl-gl-092622'
                ELSE 'egl-gl-092622'
            END,
            'egl-gl-092622'
            )AS RECOMMENDATION_ITEM_ID 
            from (SELECT HOUSEHOLD_ID FROM {inputTable} WHERE MODEL_CONFIGURATION_ID='{model_configuration_id}'
        ) c LEFT JOIN latest_segment a on CAST(c.household_id AS INT64) = a.HOUSEHOLD_ID
            '''

    print(query)

    df_features =  read_from_bq(query=query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)

    df_banner_hhid = df_features.select("HOUSEHOLD_ID", "RECOMMENDATION_ITEM_ID")
    df_banner_hhid = df_banner_hhid.withColumn("timestamp", current_timestamp())
    df_banner_hhid = df_banner_hhid.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))

    # df_banner_hhid.show()

    RANK = "1"
    RECOMMENDATION_TYPE = "BANNER"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id
    METADATA = ''

    df_out = df_banner_hhid.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        lit(RANK).alias("RANK"),
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        lit(METADATA).alias("METADATA")
    )
    return df_out
# df_gcp=getEglBanners(df, '65cba65fdf50bf67a317c7de', 'EGL_FPASS_BANNER_PHASE_1')
# df_gcp.display()

# COMMAND ----------

def get_features_data(ranking_date, feature_tab, inputTable, model_configuration_id, seasonal_products_tab=None):
    query = f"""
      select a.*, a.BPN_ID as UPC_ID, -- a hack, not using UPC_ID anywhere
            (-1)*(RANK() OVER (PARTITION BY a.HOUSEHOLD_ID 
                ORDER BY CAST(a.BPN_RANK_FINAL AS INT), a.BPN_ID)) AS BATCH_BPN_RANK
      from {feature_tab} a
      join (select  CAST(household_id AS INT64) as household_id from {inputTable} WHERE MODEL_CONFIGURATION_ID='{model_configuration_id}') b on a.household_id=b.household_id
    """
    sdf = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq).cache()

    sdf = sdf[BIA_FEATURE_COLUMNS]
    int_cols = [
        'FREQUENT_SHOPPER_FLAG', 'PERSONAL_CYCLE_BPN', 'PERSONAL_CYCLE_SHELF', 'PERSONAL_CYCLE_AISLE',
        'TXN_COUNT_BPN', 'TXN_COUNT_SHELF', 'TXN_COUNT_AISLE', 'TXN_COUNT_ALL',
        'MED_AISLE_BPN_COUNT', 'MED_AISLE_SHELF_COUNT', 'MED_HHID_AISLE_COUNT', 'MED_SHELF_BPN_COUNT',
        'DEPT_RANK', 'AISLE_RANK', 'SHELF_RANK', 'SHELF_BPN_RANK', 'AISLE_BPN_RANK', 'AISLE_BPN_RANK_SUB',
        'BATCH_BPN_RANK',
        'PAST_M1_PURCHASE_COUNT_BPN',
        'PAST_M2_PURCHASE_COUNT_BPN',
        'PAST_M3_PURCHASE_COUNT_BPN',
        'PAST_M4_PURCHASE_COUNT_BPN',
        'PAST_M5_PURCHASE_COUNT_BPN',
        'PAST_M6_PURCHASE_COUNT_BPN',
        'IS_SEASONAL'
    ]
    long_cols = ['HOUSEHOLD_ID', 'BPN_ID', 'UPC_ID']
    for column in int_cols:
        if column in sdf.columns:
            sdf = sdf.withColumn(column, sdf[column].cast('integer'))
    for column in long_cols:
        if column in sdf.columns:
            sdf = sdf.withColumn(column, sdf[column].cast('long'))

    bpn_list_pdf =pd.DataFrame()
    if seasonal_products_tab:
        query = f"""SELECT PREDICTION_DATE, BPN_ID, SCORE, NDAYS_TILL_HOLIDAY,NDAYS_PASSED_HOLIDAY FROM {seasonal_products_tab}
    WHERE PREDICTION_DATE = '{ranking_date}'"""
        sdf2 = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq).cache()
        print(sdf2.schema)
        int_cols2 = ["SCORE", "NDAYS_TILL_HOLIDAY", "NDAYS_PASSED_HOLIDAY"]
        for column in int_cols2:
            sdf2 = sdf2.withColumn(column, sdf2[column].cast('string'))
            sdf2 = sdf2.withColumn(column, sdf2[column].cast('integer'))

            #sdf2 = sdf2.withColumn(column, regexp_replace(sdf2[column], '[^0-9]','').cast('integer'))

        for column in long_cols:
            if column in sdf2.columns:
                sdf2 = sdf2.withColumn(column, sdf2[column].cast('string'))
                sdf2 = sdf2.withColumn(column, sdf2[column].cast('long'))
        #hh_pdf_org = sdf[BIA_FEATURE_COLUMNS].toPandas()
        bpn_list_pdf = sdf2.toPandas()

    return sdf, bpn_list_pdf

# COMMAND ----------

from pyspark.sql.functions import col, to_date, current_date, datediff, lit, array_join, collect_list, flatten, \
        concat_ws

# COMMAND ----------

def getBuyItAgainProductsRT(df_write, model_configuration_id, model_id):
  
    if model_id == "BIA_SEASONAL_HOMEPAGE" or model_id == "BIA_SEASONAL_SMARTBASKET":
        do_seasonal = True
        if dbricks_env == 'DEV':
            seasoanal_products_tab="gcp-abs-aamp-wmfs-prod-prj-01.aamp_ds_pz_wkg.DEV_TI_EMBEDDING_ITEM_COMBINED_SCORES_FOR_BIA"
        else:
            seasoanal_products_tab="gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.TI_EMBEDDING_ITEM_COMBINED_SCORES_FOR_BIA"
    else:
        do_seasonal = False
        seasoanal_products_tab=None

    if dbricks_env == 'DEV':
        biaTable = "gcp-abs-aamp-wmfs-prod-prj-01.aamp_ds_pz_wkg.SM_V2_ITEMS_FREQ2_CUM"
        inputTable = table_name_bq  # table_name_sf
    else:
        biaTable = "gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.SM_V2_ITEMS_FREQ2_CUM"
        inputTable = table_name_bq  # table_name_sf

    # Get the current date
    print(do_seasonal)
    current_day = datetime.today().strftime("%Y-%m-%d")
    hhids = df_write.agg(concat_ws(",", collect_list("HOUSEHOLD_ID")).alias("hstr")).collect()[0]["hstr"]
    print(f'Before get_features_data:{datetime.today()}')

    # Fetch BIA features
    features_sdf, bpn_list_df = get_features_data(current_day,
                                                  biaTable,
                                                  inputTable=inputTable,
                                                  seasonal_products_tab=seasoanal_products_tab,
                                                  model_configuration_id=model_configuration_id
                                                  )
    print(bpn_list_df.columns)
    print(f'After get_features_data:{datetime.today()}')
    features_sdf.cache()
    features_sdf = features_sdf.repartition(200,"HOUSEHOLD_ID")
    # Define the schema for the output
    schema = """
        HOUSEHOLD_ID long, BPN_ID long, UPC_ID long,
        DEPT_RANK short, BPN_RANK_FINAL integer,
        SMART_BASKET_FLAG short, HOMEPAGE_FLAG short, SM_VER short,
        RECOMMENDATION_CREATE_TS string
    """

    def _apply_bia_algo(pdf):
        # pdf.drop("GROUP_ID", axis=1, inplace=True)
        hhs = pdf["HOUSEHOLD_ID"].unique()
        return_lst = []
        for hh in hhs:
            pdf_hh = pdf[pdf["HOUSEHOLD_ID"] == hh]
            return_df = bia_algo_for_pyspark(
                pdf_hh,
                current_day,
                bpn_list_df,
                do_seasonal=do_seasonal,
                do_lp_ranking=False
            )
            return_df["RECOMMENDATION_CREATE_TS"] = datetime.today().strftime("%m/%d/%Y %H:%M")
            return_lst.append(return_df)

        return_final_df = pd.concat(return_lst, axis=0)
        return return_final_df

    # Apply the BIA algorithm
    print(f'Before df_res:{datetime.today()}')
    features_sdf = features_sdf.distinct()
    df_result = features_sdf.groupby("HOUSEHOLD_ID").applyInPandas(_apply_bia_algo, schema=schema)
    print(f'After df_res:{datetime.today()}')
    
    if model_id == 'BIA_REALTIME_SMARTBASKET' or model_id == 'BIA_SEASONAL_SMARTBASKET':
        df_result = df_result[df_result.SMART_BASKET_FLAG == 1]
    if model_id == 'BIA_REALTIME_HOMEPAGE' or model_id == 'BIA_SEASONAL_HOMEPAGE':
        df_result = df_result[df_result.HOMEPAGE_FLAG == 1]
        
    RECOMMENDATION_TYPE = "PRODUCT"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id
    METADATA = ''

    df_out = df_result.select(
        "HOUSEHOLD_ID",
        col("BPN_ID").alias("RECOMMENDATION_ITEM_ID"),
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        col("BPN_RANK_FINAL").alias("RANK"),
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        lit(METADATA).alias("METADATA")
    )

    return df_out
# df_gcp=getBuyItAgainProductsRT(df, '6719574a7d34f40de6026c5b', 'BIA_SEASONAL_HOMEPAGE')
# df_gcp.display()

# COMMAND ----------

def getManualCuratedProducts(df_write, model_configuration_id, model_id):
    if dbricks_env == 'DEV':
        curatedTable = "gcp-abs-aamp-wmfs-prod-prj-01.aamp_ds_pz_wkg.DEV_manual_curated_recommendations"
        inputTable = table_name_bq
    else:
        curatedTable = "gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.PROD_manual_curated_recommendations"
        inputTable = table_name_bq

    query = f'''
    WITH input_households AS (
    SELECT HOUSEHOLD_ID 
    FROM {inputTable} 
    WHERE MODEL_CONFIGURATION_ID = '{model_configuration_id}'
),
filtered_recommendations AS (
    SELECT 
        HHID,
        RECOMMENDATION,
        MODIFIED_DATETIME
    FROM {curatedTable}
    WHERE 
        MODEL_ID = '{model_id}'
        AND TYPE = 'PRODUCT'
),
ranked_recommendation as(
SELECT 
    i.HOUSEHOLD_ID,
    CAST(JSON_EXTRACT_SCALAR(rec, '$.bpnId') AS INT64) AS RECOMMENDATION_ITEM_ID,
    CAST(JSON_EXTRACT_SCALAR(rec, '$.rank') AS INT64) AS RANK,
    r.MODIFIED_DATETIME AS date_of_rec,
    ROW_NUMBER() OVER (PARTITION BY i.HOUSEHOLD_ID ORDER BY CAST(JSON_EXTRACT_SCALAR(rec, '$.rank') AS INT64) ) AS rec_rank
FROM input_households i
INNER JOIN filtered_recommendations r
    ON cast(i.HOUSEHOLD_ID as INT64) = cast(r.HHID as INT64)
CROSS JOIN UNNEST(JSON_EXTRACT_ARRAY(r.RECOMMENDATION, '$.recommendation')) AS rec
)
SELECT HOUSEHOLD_ID,
       RECOMMENDATION_ITEM_ID,
       RANK,
       date_of_rec
       from ranked_recommendation where rec_rank<7
ORDER BY 
    HOUSEHOLD_ID, 
    RANK
    '''
    
    print(query)

    df_edm = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)

    #df_edm = df_edm.withColumn("timestamp", to_timestamp("date_of_rec"))
    df_edm = df_edm.withColumn("timestamp", current_timestamp())
    df_edm = df_edm.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))

    RECOMMENDATION_TYPE = "PRODUCT"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id
    METADATA = ''

    df_out = df_edm.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        "RANK",
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        lit(METADATA).alias("METADATA")
    )

    return df_out
# df_gcp=getManualCuratedProducts(df, '6719574a7d34f40de6026c5b', 'THANKSGIVING_2024_CRM_FAV')
# df_gcp.display()

# COMMAND ----------

def getManualCuratedRecipes(df_write, model_configuration_id, model_id):
    if dbricks_env == 'DEV':
        curatedTable = "gcp-abs-aamp-wmfs-prod-prj-01.aamp_ds_pz_wkg.PERF_manual_curated_recommendations"
        inputTable = table_name_bq
    else:
        curatedTable = "gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.PROD_manual_curated_recommendations"
        inputTable = table_name_bq

    query = f'''
     WITH input_households AS (
    SELECT HOUSEHOLD_ID 
    FROM {inputTable} 
    WHERE MODEL_CONFIGURATION_ID = '{model_configuration_id}'
),
filtered_recommendations AS (
    SELECT 
        HHID,
        RECOMMENDATION,
        MODIFIED_DATETIME
    FROM {curatedTable}
    WHERE 
        MODEL_ID = '{model_id}'
        AND TYPE = 'RECIPE'
)
SELECT 
    i.HOUSEHOLD_ID,
    CAST(JSON_EXTRACT_SCALAR(rec, '$.recipeId') AS INT64) AS RECOMMENDATION_ITEM_ID,
    CAST(JSON_EXTRACT_SCALAR(rec, '$.rank') AS INT64) AS RANK,
    r.MODIFIED_DATETIME AS date_of_rec
FROM input_households i
INNER JOIN filtered_recommendations r
    ON i.HOUSEHOLD_ID = r.HHID
CROSS JOIN UNNEST(JSON_EXTRACT_ARRAY(r.RECOMMENDATION, '$.recommendation')) AS rec
ORDER BY 
    i.HOUSEHOLD_ID, 
    CAST(JSON_EXTRACT_SCALAR(RECOMMENDATION, '$.rank') AS INT64)
    '''
    print(query)

    df_edm = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)

    #df_edm = df_edm.withColumn("timestamp", to_timestamp("date_of_rec"))
    df_edm = df_edm.withColumn("timestamp", current_timestamp())
    df_edm = df_edm.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))

    RECOMMENDATION_TYPE = "RECIPE"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id
    METADATA = ''

    df_out = df_edm.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        "RANK",
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        lit(METADATA).alias("METADATA")
    )

    return df_out

# df_gcp=getManualCuratedRecipes(df, '671956ee7d34f40de6026c55', 'FOODRECIPE_MODEL_TEST')
# df_gcp.display()

# COMMAND ----------

def process_recipe_recommendations(df, model_configuration_id, model_id) -> DataFrame:
  
    hits_schema = ArrayType(StructType([
        StructField("id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("imageUrl", StringType(), True)
    ]))

    df = df.withColumn("hits", expr("CAST(hits AS STRING)"))
    df_exploded = df.withColumn("hits", from_json(concat_ws(",", col("hits")), hits_schema)) \
        .select("HHID", "MODEL_ID", explode("hits").alias("hit")) \
        .select(col("HHID").alias("HOUSEHOLD_ID"), "MODEL_ID",
                col("hit.id").alias("RECOMMENDATION_ITEM_ID"),
                col("hit.title").alias("RECIPE_TITLE"),
                col("hit.imageUrl").alias("IMAGE_URL")) \
        .withColumn("RANK", row_number().over(Window.partitionBy("HOUSEHOLD_ID", "MODEL_ID").orderBy(desc("RECOMMENDATION_ITEM_ID"))))


    df_result = df_exploded.withColumn("RECOMMENDATION_TYPE", lit("RECIPE")) \
        .withColumn("EXPERIMENT_ID", lit("")) \
        .withColumn("EXPERIMENT_VARIANT", lit("")) \
        .withColumn("MODEL_CONFIGURATION_ID", lit(model_configuration_id)) \
        .withColumn("RECOMMENDATION_CREATE_TS", date_format(current_timestamp(), "MM/dd/yyyy HH:mm")) \
        .withColumn("METADATA", to_json(struct("RECIPE_TITLE", "IMAGE_URL"))) \
        .filter(col("RANK") <= 3) \
        .drop("RECIPE_TITLE", "IMAGE_URL")
    #print(df_result.columns,df_result.schema)

    return df_result.select(output_schema.fieldNames())

# COMMAND ----------

dbricks_env

# COMMAND ----------

def getBiaboostedRecipes(df_write, model_configuration_id, model_id):
    input_table = table_name_bq if dbricks_env == 'DEV' else table_name_bq
    print(input_table)

    query=f"""
    WITH households AS (
  SELECT DISTINCT HOUSEHOLD_ID
  FROM {input_table}
  WHERE MODEL_CONFIGURATION_ID = '{model_configuration_id}'
),
ranked_items AS (
  SELECT 
    CAST(s.household_id AS INT64) AS HOUSEHOLD_ID,
    s.bpn_id,
    s.bpn_rank_final,
    ROW_NUMBER() OVER (PARTITION BY s.household_id ORDER BY s.bpn_rank_final DESC) AS rank
  FROM 
    {feature_project_id}.{bq_dataset}.SM_V2_DS_SMART_LIST_RANKING s
  JOIN
    households t
  ON
    CAST(s.household_id AS INT64) = CAST(t.HOUSEHOLD_ID AS INT64)
  WHERE 
    s.sm_ver = 2
)
SELECT 
  HOUSEHOLD_ID,
  BPN_ID
FROM 
  ranked_items
WHERE 
  rank <= 20;
    """
    print(query)

    df_features = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)
    df_grouped = df_features.groupBy("household_id").agg(
        array_join(collect_list("BPN_ID"), ",").alias("boost_bpns")
    )
    df_final = df_grouped.withColumn("model_id", lit(model_id))
    #display(df_final)

    temp_table_name = f"{feature_project_id}.{bq_dataset}.temp_hhid_bpns_{model_id.lower().replace('_', '')}"
    write_to_bq(df_final, temp_table_name, "overwrite", gcs_bucket,query_tag=query_tag_bq)
    bq_table_name = recipes_extractor(spark, temp_table_name, dbricks_env,query_tag_bq)

    query = f"""
      SELECT 
            bt.HHID ,
            bt.MODEL_ID,
            JSON_EXTRACT_ARRAY(bt.RECOMMENDATION, '$.hits') as hits
        FROM (SELECT HOUSEHOLD_ID FROM {input_table}  where model_configuration_id='{model_configuration_id}') tmp
        JOIN {bq_table_name} bt
        ON CAST(tmp.HOUSEHOLD_ID AS STRING) = bt.HHID
        WHERE
            MODEL_ID = '{model_id}'
    """
    print(query)
    df = read_from_bq(query=query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)

    return process_recipe_recommendations(df, model_configuration_id, model_id)



def getLatestOrderboostedRecipes(df_write, model_configuration_id, model_id):
    input_table = table_name_bq if dbricks_env == 'DEV' else table_name_bq
    print(input_table)

    query = f'''
        WITH LatestTransactions AS (
  SELECT 
    a.HOUSEHOLD_ID,
    h.TXN_DTE,
    h.TXN_ID,
    ROW_NUMBER() OVER (PARTITION BY a.HOUSEHOLD_ID ORDER BY h.TXN_DTE DESC, h.TXN_ID DESC) AS rn
  FROM (
    SELECT HOUSEHOLD_ID 
    FROM {input_table}
    WHERE MODEL_CONFIGURATION_ID='{model_configuration_id}'
  ) h_id
  JOIN `{feature_project_id}.{bq_dataset}.SM_V2_LU_CARD_ACCOUNT` a
    ON a.HOUSEHOLD_ID  = h_id.HOUSEHOLD_ID
  JOIN `gcp-abs-udco-bqvw-prod-prj-01.udco_ds_retl.TXN_HDR` h
    ON a.CARD_NBR = CAST(h.CARD_NBR AS STRING)
  WHERE h.TXN_DTE > CURRENT_DATE - 180
)
SELECT DISTINCT 
  lt.HOUSEHOLD_ID,
  lt.TXN_DTE AS LATEST_ORDER_DATE,
  p.BASE_PRODUCT_NBR AS BPN_ID
FROM LatestTransactions lt
JOIN `gcp-abs-udco-bqvw-prod-prj-01.udco_ds_retl.TXN_ITEM` d
  ON lt.TXN_ID = d.TXN_ID
JOIN `gcp-abs-udco-bqvw-prod-prj-01.udco_ds_spex.DIGITAL_PRODUCT_UPC_MASTER` p
  ON d.UPC_ID = p.UPC_NBR
WHERE lt.rn = 1
ORDER BY lt.HOUSEHOLD_ID, lt.TXN_DTE DESC;
    '''
    print(query)

    df_features = read_from_bq(query=query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)
    df_grouped = df_features.groupBy("household_id").agg(
        array_join(collect_list("BPN_ID"), ",").alias("boost_bpns")
    )
    df_final = df_grouped.withColumn("model_id", lit(model_id))
    #display(df_final)

    temp_table_name = f"{feature_project_id}.{bq_dataset}.temp_hhid_bpns_{model_id.lower().replace('_', '')}"
    write_to_bq(df_final, temp_table_name, "overwrite", gcs_bucket,query_tag=query_tag_bq)
    bq_table_name = recipes_extractor(spark, temp_table_name, dbricks_env,query_tag_bq)

    query = f"""
            SELECT 
            bt.HHID,
            bt.MODEL_ID,
            JSON_EXTRACT_ARRAY(bt.RECOMMENDATION, '$.hits') as hits
        FROM (SELECT HOUSEHOLD_ID FROM {input_table}  where model_configuration_id='{model_configuration_id}') tmp
        JOIN {bq_table_name} bt
        ON CAST(tmp.HOUSEHOLD_ID AS STRING) = bt.HHID
        WHERE
            MODEL_ID = '{model_id}'
    """
    print(query)
    df = read_from_bq(query=query, materialization_dataset=materialization_dataset,query_tag=query_tag_bq)

    # Call the helper function for common logic
    return process_recipe_recommendations(df, model_configuration_id, model_id)

# COMMAND ----------

def getReciperecommendation(df_write, model_configuration_id, model_id):
    if model_id == "BIA_BOOSTED_RECIPES":
        return getBiaboostedRecipes(df_write, model_configuration_id, model_id)
    elif model_id == "LATEST_ORDER_BOOSTED_RECIPES":
        return getLatestOrderboostedRecipes(df_write, model_configuration_id, model_id)
    else:
        raise ValueError(f"Unsupported model_id: {model_id}. Please provide a valid model_id.")

# df_gcp=getReciperecommendation(df, '66bd2bbb78100d5bcb92378a', 'BIA_BOOSTED_RECIPES')
# df_gcp.display()

# COMMAND ----------

# This function is used to remove the type from the json object and collect the property to build context
def remove_type_and_collect_property(json_obj, context_list):
        if "property" in json_obj:
            context_list.append(json_obj["property"])
        if "type" in json_obj:
            del json_obj["type"]
        if "matchers" in json_obj and len(json_obj["matchers"]) > 0:
            for i in range(0, len(json_obj["matchers"])):
                remove_type_and_collect_property(json_obj["matchers"][i], context_list)

# This function is used to get the audience object from the model
def correct_audience(json_obj, context_list):
    for obj in json_obj["data"]["dataList"]:
        if "audience" in obj and "condition" in obj["audience"]:
            remove_type_and_collect_property(obj["audience"]["condition"], context_list)

# COMMAND ----------

# This function is used to get the model json based on model id from the model table
def get_model_json(model_id):
    query = "dbo.Model"
    df_models = spark.read.jdbc(url=url, table=query, properties=properties)
    final_model = df_models.filter(df_models.model_id == model_id)
    model_json = json.loads(final_model.select("json_data").first()[0])
    return model_json

# COMMAND ----------

query = "dbo.BulkSegmentsData"
df_segments = spark.read.jdbc(url=url, table=query, properties=properties)
display(df_segments)

# COMMAND ----------

facts_segments_list=df_segments.filter(df_segments.segment_name=="facts_segments_list").select("segment_list").collect()[0][0]
persona_list=df_segments.filter(df_segments.segment_name=="persona_list").select("segment_list").collect()[0][0]
buyer_segment_list=df_segments.filter(df_segments.segment_name=="buyer_segment_list").select("segment_list").collect()[0][0]
my_needs_segment_list=df_segments.filter(df_segments.segment_name=="my_needs_segment_list").select("segment_list").collect()[0][0]
egl_segments_list=df_segments.filter(df_segments.segment_name=="egl_segments_list").select("segment_list").collect()[0][0]
customer_attributes_list=df_segments.filter(df_segments.segment_name=="customer_attributes_list").select("segment_list").collect()[0][0]

# COMMAND ----------


def get_segments_query(table_name_bq, model_configuration_id, experiment_id):
    persona_str = persona_list
    buyer_segment_str = buyer_segment_list
    my_needs_segment_str= my_needs_segment_list

    query = f"""
        select
            CAST(hhids.household_id AS INT64) as household_id,
            seg.persona,
            seg.buyerSegment,
            seg.variant,
            hhids.model_configuration_id
        from 
            {table_name_bq} hhids
        left join
                (WITH segment_data AS (
                    SELECT 
                        household_id,
                        ARRAY_AGG(aiq_segment_nm ORDER BY EXPORT_TS DESC) AS segments,
                        ARRAY_AGG(THEME ORDER BY EXPORT_TS DESC) AS variants,
                    FROM 
                        gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.RETAIL_CUSTOMER_BACKFEED_ACTIVATION
                    WHERE 
                        DW_CURRENT_VERSION_IND = TRUE 
                        AND EXPERIMENT_ID = '{experiment_id}'
                    GROUP BY 
                        household_id
                )
                SELECT
                    household_id,
                    (SELECT segment FROM UNNEST(segments) AS segment 
                        WHERE segment IN ({persona_str})
                        LIMIT 1) AS persona,
                    (SELECT segment FROM UNNEST(segments) AS segment 
                        WHERE segment IN ({buyer_segment_str})
                        LIMIT 1) AS buyerSegment,
                    (SELECT variant FROM UNNEST(variants) AS variant 
                        LIMIT 1) AS variant
                FROM 
                    segment_data) seg
                on CAST(hhids.household_id AS INT64) = seg.household_id
        WHERE hhids.model_configuration_id = '{model_configuration_id}';
    """
    return query



def get_myneeds_segments_query(table_name_bq, model_configuration_id, experiment_id):
    persona_str = persona_list
    buyer_segment_str = buyer_segment_list
    my_needs_segment_str= my_needs_segment_list

    query = f"""
        SELECT CAST(hhids.household_id AS INT64) as household_id, myneeds.myNeedsSegment, hhids.model_configuration_id
        FROM 
            {table_name_bq} hhids
        LEFT JOIN
            (SELECT 
                HOUSEHOLD_ID,
                MYNEED_SEGMENT_DSC AS myNeedsSegment,
                WEEK_ID
             FROM
                gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_SEGMENTS
             WHERE
                MYNEED_SEGMENT_DSC IN ({my_needs_segment_str})
                AND  week_id = (select max(week_id) from gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_SEGMENTS
             where MYNEED_SEGMENT_DSC is not null )
             group by all
            ) myneeds
        ON CAST(hhids.household_id AS INT64) = CAST(myneeds.HOUSEHOLD_ID AS INT64)
        WHERE hhids.model_configuration_id = '{model_configuration_id}';
    """
    return query

def get_egl_segments_query(table_name_bq, model_configuration_id, experiment_id):
    persona_str =  persona_list
    buyer_segment_str = buyer_segment_list
    my_needs_segment_str= my_needs_segment_list
    egl_segment_str= egl_segments_list
    
    query=f'''
    WITH filtered_segments AS (
        SELECT
            household_id,
            aiq_segment_nm,
            EXPORT_TS
        FROM
            gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.RETAIL_CUSTOMER_BACKFEED_ACTIVATION
        WHERE
            DW_CURRENT_VERSION_IND = TRUE
            AND aiq_segment_nm IN ({egl_segment_str})
    ),
    segment_data AS (
        SELECT
            household_id,
            ARRAY_AGG(aiq_segment_nm ORDER BY EXPORT_TS DESC) AS segments
        FROM
            filtered_segments
        GROUP BY
            household_id
    )
    SELECT
        CAST(hhids.household_id AS INT64) AS household_id,
        seg.eglSegment,
        hhids.model_configuration_id
    FROM
        {table_name_bq} hhids
    LEFT JOIN
        (
            SELECT
                household_id,
                (
                    SELECT segment
                    FROM UNNEST(segments) AS segment
                    LIMIT 1
                ) AS eglSegment
            FROM
                segment_data
        ) seg
    ON
        CAST(hhids.household_id AS INT64) = seg.household_id
    WHERE
        hhids.model_configuration_id = '{model_configuration_id}';
    '''

    
    return query

def get_leap_and_segments_query(table_name_bq, model_configuration_id, experiment_id):
    persona_str =  persona_list
    buyer_segment_str = buyer_segment_list
    my_needs_segment_str= my_needs_segment_list

    query = f"""
        select
            CAST(hhids.household_id AS INT64) as household_id,
            CAST(L.STORE_ID AS INT64) as storeId,
            EM.BANNER as banner,
            CAST(L.DIVISION_ID AS INT64) as divisionNumber,
            seg.persona,
            seg.buyerSegment,
            seg.variant,
            hhids.model_configuration_id
        from 
            {table_name_bq} hhids
        left join
                (select DISTINCT
                    household_id,
                    STORE_ID,
                    DIVISION_ID
                from 
                    gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LEAP
                where
                    household_id is not Null) as L
                on CAST(hhids.household_id AS INT64) = CAST(L.household_id AS INT64)
        left join
            gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LT_EMAIL_VERSION EM 
            on l.store_id = em.store_id
        left join
                (WITH segment_data AS (
                    SELECT 
                        household_id,
                        ARRAY_AGG(aiq_segment_nm ORDER BY EXPORT_TS DESC) AS segments,
                        ARRAY_AGG(THEME ORDER BY EXPORT_TS DESC) AS variants,
                    FROM 
                        gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.RETAIL_CUSTOMER_BACKFEED_ACTIVATION
                    WHERE 
                        DW_CURRENT_VERSION_IND = TRUE 
                        AND EXPERIMENT_ID = '{experiment_id}'
                    GROUP BY 
                        household_id
                )
                SELECT
                    household_id,
                    (SELECT segment FROM UNNEST(segments) AS segment 
                        WHERE segment IN ({persona_str})
                        LIMIT 1) AS persona,
                    (SELECT segment FROM UNNEST(segments) AS segment 
                        WHERE segment IN ({buyer_segment_str})
                        LIMIT 1) AS buyerSegment,
                    (SELECT variant FROM UNNEST(variants) AS variant 
                        LIMIT 1) AS variant
                FROM 
                    segment_data) seg
                on CAST(hhids.household_id AS INT64) = seg.household_id
        WHERE hhids.model_configuration_id = '{model_configuration_id}';
    """
    return query


def get_leap_and_myneeds_segments_query(table_name_bq, model_configuration_id, experiment_id):
    persona_str = persona_list
    buyer_segment_str = buyer_segment_list
    my_needs_segment_str= my_needs_segment_list

    query = f"""
        select
            CAST(hhids.household_id AS INT64) as household_id,
            CAST(L.STORE_ID AS INT64) as storeId,
            EM.BANNER as banner,
            CAST(L.DIVISION_ID AS INT64) as divisionNumber,
            seg.myNeedsSegment,
            seg.variant,
            hhids.model_configuration_id
        from 
            {table_name_bq} hhids
        left join
                (select DISTINCT
                    household_id,
                    STORE_ID,
                    DIVISION_ID
                from 
                    gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LEAP
                where
                    household_id is not Null) as L
                on CAST(hhids.household_id AS INT64) = CAST(L.household_id AS INT64)
        left join
            gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LT_EMAIL_VERSION EM 
            on l.store_id = em.store_id
        left join (SELECT 
                HOUSEHOLD_ID,
                MYNEED_SEGMENT_DSC AS myNeedsSegment,
                WEEK_ID
             FROM
                gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_SEGMENTS
             WHERE
                MYNEED_SEGMENT_DSC IN ({my_needs_segment_str})
                AND  week_id = (select max(week_id) from gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_SEGMENTS
             where MYNEED_SEGMENT_DSC is not null )
             group by all
            ) myneeds
                on CAST(hhids.household_id AS INT64) = myneeds.household_id
        WHERE hhids.model_configuration_id = '{model_configuration_id}';
    """
    return query

def get_leap_and_egl_segments_query(table_name_bq, model_configuration_id, experiment_id):
    persona_str = persona_list
    buyer_segment_str = buyer_segment_list
    my_needs_segment_str= my_needs_segment_list
    egl_segment_str= egl_segments_list

    query = f"""
        SELECT
        CAST(hhids.household_id AS INT64) AS household_id,
        CAST(L.STORE_ID AS INT64) AS storeId,
        EM.BANNER AS banner,
        CAST(L.DIVISION_ID AS INT64) AS divisionNumber,
        seg.eglSegment,
        hhids.model_configuration_id
    FROM
        {table_name_bq} hhids
    LEFT JOIN
        (
            SELECT DISTINCT
                household_id,
                STORE_ID,
                DIVISION_ID
            FROM
                gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LEAP
            WHERE
                household_id IS NOT NULL
        ) AS L
        ON CAST(hhids.household_id AS INT64) = CAST(L.household_id AS INT64)
    LEFT JOIN
        gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LT_EMAIL_VERSION EM
        ON L.store_id = EM.store_id
    LEFT JOIN
        (
            WITH filtered_segments AS (
                SELECT
                    household_id,
                    aiq_segment_nm,
                    EXPORT_TS
                FROM
                    gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.RETAIL_CUSTOMER_BACKFEED_ACTIVATION
                WHERE
                    DW_CURRENT_VERSION_IND = TRUE
                    AND aiq_segment_nm IN ({egl_segment_str})
            ),
            segment_data AS (
                SELECT
                    household_id,
                    ARRAY_AGG(aiq_segment_nm ORDER BY EXPORT_TS DESC) AS segments
                FROM
                    filtered_segments
                GROUP BY
                    household_id
            )
            SELECT
                household_id,
                (
                    SELECT segment
                    FROM UNNEST(segments) AS segment
                    LIMIT 1
                ) AS eglSegment
            FROM
                segment_data
        ) seg
        ON CAST(hhids.household_id AS INT64) = seg.household_id
    WHERE
        hhids.model_configuration_id = '{model_configuration_id}';
    """
    return query

def get_leap_query(table_name_bq, model_configuration_id):
    query = f'''
            select
                CAST(hhids.household_id AS INT64) as household_id,
                CAST(L.STORE_ID AS INT64) as storeId,
                EM.BANNER as banner,
                CAST(L.DIVISION_ID AS INT64) as divisionNumber,
                hhids.model_configuration_id
            from 
                {table_name_bq} hhids
            left join
                (select DISTINCT
                    household_id,
                    STORE_ID,
                    DIVISION_ID
                from 
                    gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LEAP
                where
                    household_id is not Null) as L
                on CAST(hhids.household_id AS INT64) = CAST(L.household_id AS INT64)
            left join
                gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LT_EMAIL_VERSION EM 
                on l.store_id = em.store_id
            WHERE hhids.model_configuration_id = '{model_configuration_id}';
        '''
    return query

def get_leap_egl_and_segments_query(table_name_bq, model_configuration_id, experiment_id):
    persona_str = persona_list
    buyer_segment_str = buyer_segment_list
    my_needs_segment_str = my_needs_segment_list
    egl_segment_str = egl_segments_list
 
    query = f"""
        SELECT
            CAST(hhids.household_id AS INT64) AS household_id,
            CAST(L.STORE_ID AS INT64) AS storeId,
            EM.BANNER AS banner,
            CAST(L.DIVISION_ID AS INT64) AS divisionNumber,
            seg.persona,
            seg.buyerSegment,
            seg.variant,
            egl.eglSegment,
            hhids.model_configuration_id
        FROM
            {table_name_bq} hhids
        LEFT JOIN
            (SELECT DISTINCT
                household_id,
                STORE_ID,
                DIVISION_ID
            FROM
                gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LEAP
            WHERE
                household_id IS NOT NULL) AS L
            ON CAST(hhids.household_id AS INT64) = CAST(L.household_id AS INT64)
        LEFT JOIN
            gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LT_EMAIL_VERSION EM
            ON L.store_id = EM.store_id
        LEFT JOIN
            (WITH segment_data AS (
                SELECT
                    household_id,
                    ARRAY_AGG(aiq_segment_nm ORDER BY EXPORT_TS DESC) AS segments,
                    ARRAY_AGG(THEME ORDER BY EXPORT_TS DESC) AS variants
                FROM
                    gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.RETAIL_CUSTOMER_BACKFEED_ACTIVATION
                WHERE
                    DW_CURRENT_VERSION_IND = TRUE
                    AND EXPERIMENT_ID = '{experiment_id}'
                GROUP BY
                    household_id
            )
            SELECT
                household_id,
                (SELECT segment FROM UNNEST(segments) AS segment
                    WHERE segment IN ({persona_str})
                    LIMIT 1) AS persona,
                (SELECT segment FROM UNNEST(segments) AS segment
                    WHERE segment IN ({buyer_segment_str})
                    LIMIT 1) AS buyerSegment,
                (SELECT variant FROM UNNEST(variants) AS variant
                    LIMIT 1) AS variant
            FROM
                segment_data) seg
            ON CAST(hhids.household_id AS INT64) = seg.household_id
        LEFT JOIN
            (WITH filtered_segments AS (
                SELECT
                    household_id,
                    aiq_segment_nm,
                    EXPORT_TS
                FROM
                    gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.RETAIL_CUSTOMER_BACKFEED_ACTIVATION
                WHERE
                    DW_CURRENT_VERSION_IND = TRUE
                    AND aiq_segment_nm IN ({egl_segment_str})
            ),
            segment_data AS (
                SELECT
                    household_id,
                    ARRAY_AGG(aiq_segment_nm ORDER BY EXPORT_TS DESC) AS segments
                FROM
                    filtered_segments
                GROUP BY
                    household_id
            )
            SELECT
                household_id,
                (
                    SELECT segment
                    FROM UNNEST(segments) AS segment
                    LIMIT 1
                ) AS eglSegment
            FROM
                segment_data
            ) egl
            ON CAST(hhids.household_id AS INT64) = egl.household_id
        WHERE
            hhids.model_configuration_id = '{model_configuration_id}';
    """
    return query




# COMMAND ----------

# We are precomputing the banners for the unique set of rules(we don't have to run matchers on each and every household)
# Later this df will be joined to the input df coming from martec
def get_matched_banners_df(df_user_details, model_json, rule_fields):
    obj_li = []

    matcher = spark._jvm.com.albertsons.aapn.experiment.sdk.core.eligibility.matchers.MatcherMapper()
    exp_service_class = spark._jvm.com.albertsons.aapn.experiment.sdk.core.services.DefaultExperimentCoreService(None, matcher)
    obj_mapper = spark._jvm.com.fasterxml.jackson.databind.ObjectMapper()
    eligibility_matcher_class = spark._jvm.com.albertsons.aapn.experiment.sdk.core.dto.eligibility.EligibilityMatcher()
    default_banner = None
    for obj in model_json["data"]["dataList"]:
        matcher_obj = None
        if "audience" in obj and "condition" in obj["audience"]:
            matcher_json = json.dumps(obj["audience"]["condition"])
            matcher_obj = obj_mapper.readValue(matcher_json, eligibility_matcher_class.getClass())
        else:
            default_banner = obj["staticRecommendation"]["bannerRecommendation"]["recommendations"][0]["bannerId"]
        banner_dict = {}
        banner_dict["banner_id"] = obj["staticRecommendation"]["bannerRecommendation"]["recommendations"][0]["bannerId"]
        banner_dict["matcher"] = matcher_obj
        obj_li.append(banner_dict)

    def check_eligibility(dataList, context):
        for data in dataList:
            if exp_service_class.checkCondition(data["matcher"], context).getResult():
                return data["banner_id"]
        return None
            

    precomputed_data = []
    count = 0

    print(f'Unique rules: {len(df_user_details.select(rule_fields).distinct().collect())}')

    for distinct_df in df_user_details.select(rule_fields).distinct().collect():
        count = count + 1
        context_map = {}
        for rule in rule_fields:
            context_map[rule] = distinct_df[rule]
        banner_id = check_eligibility(obj_li, context_map)
        context_map["RECOMMENDATION_ITEM_ID"] = banner_id
        precomputed_data.append(context_map)

    print(f'BAU banner: {default_banner}')
    print(f'No of unique rules: {count}')

    precomputed_schema = StructType([
        StructField('RECOMMENDATION_ITEM_ID', StringType(), True),
        StructField('banner', StringType(), True),
        StructField('buyerSegment', StringType(), True),
        StructField('divisionNumber', LongType(), True),
        StructField('persona', StringType(), True),
        StructField('storeId', LongType(), True),
        StructField('variant', StringType(), True),
        StructField('myNeedsSegment', StringType(), True),
        StructField('eglSegment', StringType(), True),
        StructField('factsSegment', StringType(), True),
        StructField('ECOM_PURCHASE_IND', StringType(), True),
        StructField('ENG_MODE_LIFETIME', StringType(), True)
        
    ])

    precomputed_df = spark.createDataFrame(precomputed_data, precomputed_schema)
    return precomputed_df, default_banner

# COMMAND ----------

import json
from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, to_json, when, col, coalesce, broadcast
from pyspark.sql.types import StringType, LongType

def getRuleBasedBanners(df_write, model_configuration_id, model_id, experiment_id):
    model_json = get_model_json(model_id)

    context_properties_list = []

    correct_audience(model_json, context_properties_list)
    context_properties_set = set(context_properties_list)

    print(context_properties_set)
    final_rule_fields_list = list(context_properties_set)
    print(f'Rule fields list: {final_rule_fields_list}')

    context_to_table_map = {
        "leap": ["storeId", "banner", "divisionNumber"],
        "c360": ["buyerSegment", "persona","variant"],
        "my_needs": ["myNeedsSegment"],
        "egl": ["eglSegment"]
    }

    result_set = set()

    for key, values in context_to_table_map.items():
        if any(value in context_properties_set for value in values):
            result_set.add(key)

    print(result_set)

    query = ""

    if "leap" in result_set:
        if "c360" in result_set and "egl" in result_set:
            query = get_leap_egl_and_segments_query(table_name_bq, model_configuration_id, experiment_id)
        elif "c360" in result_set:
            query = get_leap_and_segments_query(table_name_bq, model_configuration_id, experiment_id)
        elif "my_needs" in result_set:
            query = get_leap_and_myneeds_segments_query(table_name_bq, model_configuration_id, experiment_id)
        elif "egl" in result_set:
            query = get_leap_and_egl_segments_query(table_name_bq, model_configuration_id, experiment_id)
        else:
            query = get_leap_query(table_name_bq, model_configuration_id)
    else:
        if "c360" in result_set:
            query = get_segments_query(table_name_bq, model_configuration_id, experiment_id)
        elif "my_needs" in result_set:
            query = get_myneeds_segments_query(table_name_bq, model_configuration_id, experiment_id)
        elif "egl" in result_set:
            query = get_egl_segments_query(table_name_bq, model_configuration_id, experiment_id)

    print(query)
    #print(rule_fields)

    df_user_details = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)

    print(f'Input hhids count: {df_user_details.count()}')

    precomputed_df, default_banner = get_matched_banners_df(df_user_details, model_json, final_rule_fields_list)
    
    fill_values = {
        StringType(): 'default',
        LongType(): 0
    }

    for dtype, value in fill_values.items():
        df_user_details = df_user_details.fillna(value, subset=[col for col in df_user_details.columns if df_user_details.schema[col].dataType == dtype])
        precomputed_df = precomputed_df.fillna(value, subset=[col for col in precomputed_df.columns if precomputed_df.schema[col].dataType == dtype])

    df_updated = df_user_details.join(
        broadcast(precomputed_df),
        on=final_rule_fields_list,
        how="left")

    df_updated = df_updated.withColumn("timestamp", current_timestamp())
    df_updated = df_updated.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))
    
    RANK = "1"
    RECOMMENDATION_TYPE = "BANNER"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id
    METADATA = ''

    df_updated = df_updated.withColumn('RECOMMENDATION_ITEM_ID', when(col('RECOMMENDATION_ITEM_ID').isNull(), default_banner).otherwise(col('RECOMMENDATION_ITEM_ID')))

    df_out = df_updated.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        lit(RANK).alias("RANK"),
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        lit(METADATA).alias("METADATA")
    )
    print(f'Final hhids count: {df_out.count()}')
    return df_out

# df_gcp=getRuleBasedBanners(df, '67195486c18d389101c93c2d', 'MY_NEEDS_BANNER_EMAIL_PROMO_CARD','664cfa21c281dd57b2b6ec6e')
# df_gcp.display()

# COMMAND ----------

def get_leap_join():
    return """
        LEFT JOIN (
            SELECT household_id, STORE_ID, DIVISION_ID
            FROM (
                SELECT DISTINCT household_id, STORE_ID, DIVISION_ID
                FROM gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LEAP
                WHERE household_id is not Null
            )
        ) as L ON CAST(hhids.household_id AS INT64) = CAST(L.household_id AS INT64)
        LEFT JOIN gcp-abs-udco-bsvw-prod-prj-01.udco_ds_bizops.LT_EMAIL_VERSION EM
            ON L.store_id = EM.store_id
    """
 
def get_c360_join(experiment_id, persona_str, buyer_segment_str):
    return f"""
        LEFT JOIN (
            WITH segment_data AS (
                SELECT
                    household_id,
                    aiq_segment_nm,
                    THEME,
                    EXPORT_TS,
                    ROW_NUMBER() OVER (
                        PARTITION BY household_id,
                        CASE
                            WHEN aiq_segment_nm IN ({persona_str}) THEN 'persona'
                            WHEN aiq_segment_nm IN ({buyer_segment_str}) THEN 'buyer'
                        END
                        ORDER BY EXPORT_TS DESC
                    ) as rn
                FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.RETAIL_CUSTOMER_BACKFEED_ACTIVATION
                WHERE DW_CURRENT_VERSION_IND = TRUE
                    AND EXPERIMENT_ID = '{experiment_id}'
                    AND aiq_segment_nm IN ({persona_str}, {buyer_segment_str})
            )
            SELECT
                household_id,
                MAX(CASE WHEN aiq_segment_nm IN ({persona_str}) THEN aiq_segment_nm END) as persona,
                MAX(CASE WHEN aiq_segment_nm IN ({buyer_segment_str}) THEN aiq_segment_nm END) as buyerSegment,
                MAX(THEME) as variant
            FROM segment_data
            WHERE rn = 1
            GROUP BY household_id
        ) seg ON CAST(hhids.household_id AS INT64) = seg.household_id
    """
 
def get_myneeds_join(experiment_id, my_needs_segment_str):
    return f"""
        LEFT JOIN (
            WITH latest_week AS (
                SELECT max(week_id) as max_week_id
                FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_SEGMENTS
                WHERE MYNEED_SEGMENT_DSC is not null
            )
            SELECT DISTINCT
                HOUSEHOLD_ID,
                MYNEED_SEGMENT_DSC AS myNeedsSegment
            FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_SEGMENTS
            WHERE MYNEED_SEGMENT_DSC IN ({my_needs_segment_str})
                AND week_id = (SELECT max_week_id FROM latest_week)
        ) myneeds ON CAST(hhids.household_id AS INT64) = CAST(myneeds.HOUSEHOLD_ID AS INT64)
    """
 
def get_egl_join(experiment_id, egl_segment_str):
    return f"""
        LEFT JOIN (
            WITH latest_records AS (
                SELECT
                    household_id,
                    aiq_segment_nm,
                    ROW_NUMBER() OVER (PARTITION BY household_id ORDER BY EXPORT_TS DESC) as rn
                FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.RETAIL_CUSTOMER_BACKFEED_ACTIVATION
                WHERE DW_CURRENT_VERSION_IND = TRUE
                    AND aiq_segment_nm IN ({egl_segment_str})
            )
            SELECT DISTINCT
                household_id,
                aiq_segment_nm as eglSegment
            FROM latest_records
            WHERE rn = 1
        ) egl ON CAST(hhids.household_id AS INT64) = egl.household_id
    """
 
def get_facts_join(facts_segment_str):
    return f"""
        LEFT JOIN (
            WITH latest_week AS (
                SELECT max(week_id) as max_week_id
                FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_SEGMENTS
                WHERE FACT_LEVEL2_SEGMENT_DSC is not null
            )
            SELECT DISTINCT
                HOUSEHOLD_ID,
                FACT_LEVEL2_SEGMENT_DSC as factsSegment
            FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_SEGMENTS
            WHERE FACT_LEVEL2_SEGMENT_DSC IN ({facts_segment_str})
                AND week_id = (SELECT max_week_id FROM latest_week)
        ) facts ON CAST(hhids.household_id AS INT64) = CAST(facts.HOUSEHOLD_ID AS INT64)
    """
 
def get_customer_attributes_join(customer_attr_str):
    print(customer_attr_str)
    return f"""
        LEFT JOIN
        (select household_id,
           {customer_attr_str},
           ROW_NUMBER() OVER (PARTITION BY HOUSEHOLD_ID) as rn
             FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_PROFILE
            ) cust
        ON CAST(hhids.household_id AS INT64) = CAST(cust.HOUSEHOLD_ID AS INT64)
        AND cust.rn = 1
    """
def get_unified_query(table_name_bq, model_configuration_id, experiment_id, result_set, required_attributes):
 
    print(f'Required customer attributes: {required_attributes}')
 
    # Build all segment strings at once
    facts_segment_str = facts_segments_list
    persona_str = persona_list
    buyer_segment_str = buyer_segment_list
    my_needs_segment_str = my_needs_segment_list
    egl_segment_str = egl_segments_list
    customer_attr_str = 'ECOM_PURCHASE_IND, DUG_IND, PRIM_STORE_ID, PRIM_DIVISION_ID, EMPLOYEE_IND, PHARMACY_PURCHASE_IND, EXPLICIT_DIET_PREFERENCE_CD, FRESHPASS_SUBSCRIPTION_STATUS_CD, PET_PRODUCT_PURCHASE_IND, MARKETABLE_EMAIL_IND, DIGITAL_LOGIN_P12M,ENG_MODE_LIFETIME, DELIVERY_IND, MEAT_PURCHASE_IND'
 
    # Dynamically build column selection based on result_set
    select_columns = ["CAST(hhids.household_id AS INT64) as household_id"]
   
    if "leap" in result_set:
        select_columns.extend([
            "CAST(L.STORE_ID AS INT64) as storeId",
            "EM.BANNER as banner",
            "CAST(L.DIVISION_ID AS INT64) as divisionNumber"
        ])
   
    if "c360" in result_set:
        select_columns.extend([
            "seg.persona",
            "seg.buyerSegment",
            "seg.variant"
        ])
   
    if "my_needs" in result_set:
        select_columns.append("myneeds.myNeedsSegment")
   
    if "egl" in result_set:
        select_columns.append("egl.eglSegment")
   
    if "facts" in result_set:
        select_columns.append("facts.factsSegment")
   
    if "customer_attributes" in result_set and required_attributes:
        select_columns.extend([f"cust.{attr}" for attr in required_attributes])
   
    select_columns.append("hhids.model_configuration_id")
   
    query = f"""
        SELECT {', '.join(select_columns)}
        FROM
            {table_name_bq} hhids
        {get_leap_join() if "leap" in result_set else ""}
        {get_c360_join(experiment_id, persona_str, buyer_segment_str) if "c360" in result_set else ""}
        {get_myneeds_join(experiment_id, my_needs_segment_str) if "my_needs" in result_set else ""}
        {get_egl_join(experiment_id, egl_segment_str) if "egl" in result_set else ""}
        {get_facts_join(facts_segment_str) if "facts" in result_set else ""}
        {get_customer_attributes_join(customer_attr_str) if "customer_attributes" in result_set else ""}
        WHERE hhids.model_configuration_id = '{model_configuration_id}';
    """
    return query
 
def get_required_customer_attributes(context_properties_set):
    customer_attributes_list=["DUG_IND", "PRIM_STORE_ID", "PRIM_DIVISION_ID", "EMPLOYEE_IND", "PHARMACY_PURCHASE_IND", "EXPLICIT_DIET_PREFERENCE_CD", "FRESHPASS_SUBSCRIPTION_STATUS_CD", "PET_PRODUCT_PURCHASE_IND", "ECOM_PURCHASE_IND", "MARKETABLE_EMAIL_IND", "DIGITAL_LOGIN_P12M", "DELIVERY_IND", "MEAT_PURCHASE_IND","ENG_MODE_LIFETIME"]
    all_customer_attributes = set(customer_attributes_list)
    return list(all_customer_attributes.intersection(context_properties_set))

# COMMAND ----------

import json
from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, to_json, when, col, coalesce, broadcast
from pyspark.sql.types import StringType, LongType


def getRuleBasedBannersnew(df_write, model_configuration_id, model_id, experiment_id):
    model_json = get_model_json(model_id)

    context_properties_list = []

    correct_audience(model_json, context_properties_list)
    context_properties_set = set(context_properties_list)    
    print(context_properties_set)

    final_rule_fields_list = list(context_properties_set)
    print(f'Rule fields list: {final_rule_fields_list}')

    required_customer_attributes = get_required_customer_attributes(context_properties_set)
    
    context_to_table_map = {
        "leap": ["storeId", "banner", "divisionNumber"],
        "c360": ["buyerSegment", "persona", "variant"],
        "my_needs": ["myNeedsSegment"],
        "egl": ["eglSegment"],
        "facts": ["factsSegment"],
        "customer_attributes": required_customer_attributes
    }
    
    result_set = set()
    for key, values in context_to_table_map.items():
        if any(value in context_properties_set for value in values):
            result_set.add(key)
            
    print(f"Required data sources: {result_set}")
    
    # Get query using unified function
    query = get_unified_query(table_name_bq, model_configuration_id, experiment_id, 
                            result_set, required_customer_attributes)
    
    print(query)
    df_user_details = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)
    display(df_user_details)

    print(f'Input hhids count: {df_user_details.count()}')

    precomputed_df, default_banner = get_matched_banners_df(df_user_details, model_json, final_rule_fields_list)
    display(precomputed_df)
    display(default_banner)
    
    fill_values = {
        StringType(): 'default',
        LongType(): 0
    }

    for dtype, value in fill_values.items():
        df_user_details = df_user_details.fillna(value, subset=[col for col in df_user_details.columns if df_user_details.schema[col].dataType == dtype])
        precomputed_df = precomputed_df.fillna(value, subset=[col for col in precomputed_df.columns if precomputed_df.schema[col].dataType == dtype])

    df_updated = df_user_details.join(
        broadcast(precomputed_df),
        on=final_rule_fields_list,
        how="left")

    df_updated = df_updated.withColumn("timestamp", current_timestamp())
    df_updated = df_updated.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))
    
    RANK = "1"
    RECOMMENDATION_TYPE = "BANNER"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id
    METADATA = ''

    df_updated = df_updated.withColumn('RECOMMENDATION_ITEM_ID', when(col('RECOMMENDATION_ITEM_ID').isNull(), default_banner).otherwise(col('RECOMMENDATION_ITEM_ID')))

    df_out = df_updated.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        lit(RANK).alias("RANK"),
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        lit(METADATA).alias("METADATA")
    )
    print(f'Final hhids count: {df_out.count()}')
    return df_out

# COMMAND ----------

def get_matched_recipe_df(df_user_details, model_json, rule_fields):
    obj_li = []

    matcher = spark._jvm.com.albertsons.aapn.experiment.sdk.core.eligibility.matchers.MatcherMapper()
    exp_service_class = spark._jvm.com.albertsons.aapn.experiment.sdk.core.services.DefaultExperimentCoreService(None, matcher)
    obj_mapper = spark._jvm.com.fasterxml.jackson.databind.ObjectMapper()
    eligibility_matcher_class = spark._jvm.com.albertsons.aapn.experiment.sdk.core.dto.eligibility.EligibilityMatcher()
    default_recipe = None

    for obj in model_json["data"]["dataList"]:
        matcher_obj = None
        if "audience" in obj and "condition" in obj["audience"]:
            matcher_json = json.dumps(obj["audience"]["condition"])
            matcher_obj = obj_mapper.readValue(matcher_json, eligibility_matcher_class.getClass())
        else:
                default_recipe = obj["modelId"]
        recipe_dict = {}
        recipe_dict["model_Id"] = obj["modelId"]
        recipe_dict["matcher"] = matcher_obj
        obj_li.append(recipe_dict)

    def check_eligibility(dataList, context):
        matched_models = []
        for data in dataList:
            if data["matcher"] and exp_service_class.checkCondition(data["matcher"], context).getResult():
                matched_models.append(data["model_Id"])  
        return matched_models if matched_models else [default_recipe]
            
    precomputed_data = []
    count = 0

    print(f'Unique rules: {len(df_user_details.select(rule_fields).distinct().collect())}')

    for distinct_df in df_user_details.select(rule_fields).distinct().collect():
        count = count + 1
        context_map = {}
        for rule in rule_fields:
            context_map[rule] = distinct_df[rule]
        recipe_ids = check_eligibility(obj_li, context_map)
        # Create a row for each matching recipe_id
        for recipe_id in recipe_ids:
            row = context_map.copy()  # Copy context_map to avoid overwriting
            row["RECOMMENDATION_ITEM_ID"] =recipe_id
            precomputed_data.append(row)


    print(f'BAU recipe: {default_recipe}')
    print(f'No of unique rules: {count}')

    precomputed_schema = StructType([
        StructField('RECOMMENDATION_ITEM_ID', StringType(), True),
        StructField('banner', StringType(), True),
        StructField('buyerSegment', StringType(), True),
        StructField('divisionNumber', LongType(), True),
        StructField('persona', StringType(), True),
        StructField('storeId', LongType(), True),
        StructField('variant', StringType(), True),
        StructField('myNeedsSegment', StringType(), True),
        StructField('eglSegment', StringType(), True)
        
    ])

    precomputed_df = spark.createDataFrame(precomputed_data, precomputed_schema)
    return precomputed_df,default_recipe

# COMMAND ----------

import requests

def Getcookbookdata(unique_cookbook_ids, batch_size=50):
    
    if dbricks_env == 'DEV':
        url = 'https://esap-share-nonprod-apim-01-west-az.albertsons.com/abs/acceptancepub/dirm/menuservice/v2/recipe-discovery/by-cookbook'
        headers = {'ocp-apim-subscription-key': 'cbcdfd64139a4800bd0a1562ce4eb4b0'}

    else:
        url = 'https://esap-apim-prod-01.albertsons.com/abs/pub/dirm/menuservice/v2/recipe-discovery/by-cookbook'
        headers = {'ocp-apim-subscription-key': 'ed0dadf2d6fe47a08c1e88e197f971a7'}
    
    all_data = {}
    for i in range(0, len(unique_cookbook_ids), batch_size):
        batch_ids = unique_cookbook_ids[i:i + batch_size]
        for cookbook_id in unique_cookbook_ids:
            params = {'id': cookbook_id}
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                all_data[cookbook_id] = data
            else:
                print("Failed to retrieve data. Status Code:", response.status_code)
    return json.dumps(all_data)

# COMMAND ----------

import json
from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, to_json, when, col, coalesce, broadcast
from pyspark.sql.types import StringType, LongType

def getRuleBasedReceipe(df_write, model_configuration_id, model_id_src, experiment_id):
    model_json = get_model_json(model_id_src)

    context_properties_list = []
    correct_audience(model_json, context_properties_list)
    context_properties_set = set(context_properties_list)

    print(context_properties_set)
    final_rule_fields_list = list(context_properties_set)
    print(f'Rule fields list: {final_rule_fields_list}')

    context_to_table_map = {
        "leap": ["storeId", "banner", "divisionNumber"],
        "c360": ["buyerSegment", "persona","variant"],
        "my_needs": ["myNeedsSegment"],
        "egl": ["eglSegment"]
    }

    result_set = set()

    for key, values in context_to_table_map.items():
        if any(value in context_properties_set for value in values):
            result_set.add(key)

    print(result_set)

    query = ""
    function_name=""

    if "leap" in result_set:
        if "c360" in result_set:
            query = get_leap_and_segments_query(table_name_bq, model_configuration_id, experiment_id)
        elif "my_needs" in result_set:
            query = get_leap_and_myneeds_segments_query(table_name_bq, model_configuration_id, experiment_id)
        elif "egl" in result_set:
            query = get_leap_and_egl_segments_query(table_name_bq, model_configuration_id, experiment_id)
        else:
            query = get_leap_query(table_name_bq, model_configuration_id)
    else :
        if "c360" in result_set:
            query = get_segments_query(table_name_bq, model_configuration_id, experiment_id)
        elif "my_needs" in result_set:
            query = get_myneeds_segments_query(table_name_bq, model_configuration_id, experiment_id)
        elif "egl" in result_set:
            query = get_egl_segments_query(table_name_bq, model_configuration_id, experiment_id)

    df_user_details = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)

    print(f'Input hhids count: {df_user_details.count()}')

    precomputed_df, default_recipe = get_matched_recipe_df(df_user_details, model_json, final_rule_fields_list)
    precomputed_df.display()
    df_user_details.display()
    
    fill_values = {
        StringType(): 'default',
        LongType(): 0
    }

    for dtype, value in fill_values.items():
        df_user_details = df_user_details.fillna(value, subset=[col for col in df_user_details.columns if df_user_details.schema[col].dataType == dtype])
        precomputed_df = precomputed_df.fillna(value, subset=[col for col in precomputed_df.columns if precomputed_df.schema[col].dataType == dtype])

    distinct_modelid_df = precomputed_df.select("RECOMMENDATION_ITEM_ID").distinct()
    cookbook_results = []
    
    for row in distinct_modelid_df.collect():
            model_id = row["RECOMMENDATION_ITEM_ID"]  # Get the model_id from the DataFrame row
            try:
                model_json = get_model_json(model_id)  # Fetch the model JSON for the given model_id
                cookbook_id = model_json.get("recipeProperties", {}).get("cookbookId", None)  # Extract cookbookId
                # Append the result as a dictionary
                cookbook_results.append({"model_id": model_id, "cookbookId": cookbook_id})
            except Exception as e:
                # Handle errors (e.g., model_id not found or JSON parsing issues)
                print(f"Error fetching or processing model_id {model_id}: {e}")
    cookbook_df = spark.createDataFrame(cookbook_results)
    
    unique_model_cookbook_ids = cookbook_df.select("model_id", "cookbookId").distinct().collect() # Fetch unique model IDs and their corresponding cookbook IDs
    model_to_cookbook_map = {row["cookbookId"]: row["model_id"] for row in unique_model_cookbook_ids} # Map modelId to cookbookId for API usage
    unique_cookbook_ids = list(model_to_cookbook_map.keys()) # Get unique cookbook IDs
    # Call the API to get recipe data
    cookbook_data = json.loads(Getcookbookdata(unique_cookbook_ids))
    # Parse the response to extract modelId, cookbookId, and recipeId
    recipe_results = []

    for cookbook_id, data in cookbook_data.items():
            model_id = model_to_cookbook_map.get(cookbook_id, None)  # Get the associated modelId
            if "hits" in data:
                rank = 1
                for recipe in data["hits"]:
                    recipe_results.append({
                    "RECOMMENDATION_ITEM_ID": model_id,
                    "cookbookId": cookbook_id,
                    "recipeId": recipe.get("id"),
                    "rank": rank,
                })
                    rank += 1
    # Convert the results to a new Spark DataFrame
    if recipe_results:
        recipe_df = spark.createDataFrame(recipe_results)
    else:
        print("No recipes found")

    precomputed_df = precomputed_df.alias("pre")
    recipe_df = recipe_df.alias("rec")
    recipe_precomputed_joined_df = broadcast(precomputed_df).join(
    broadcast(recipe_df),
    on="RECOMMENDATION_ITEM_ID",
    how="left"
    )

    df_selected = recipe_precomputed_joined_df.select(
    col("pre.RECOMMENDATION_ITEM_ID"),"pre.banner","pre.buyerSegment","pre.divisionNumber","pre.persona","pre.storeId","pre.variant","pre.myNeedsSegment","pre.eglSegment","rec.cookbookId", "rec.recipeId","rec.rank")

    df_updated = df_user_details.join(
        broadcast(df_selected),
        on=final_rule_fields_list,
        how="left")

    df_updated = df_updated.withColumn("timestamp", current_timestamp())
    df_updated = df_updated.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))
    
    # RANK = "1"
    RECOMMENDATION_TYPE = "RECIPE"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id_src
    METADATA = ''

    df_updated = df_updated.withColumn('RECOMMENDATION_ITEM_ID', when(col('pre.RECOMMENDATION_ITEM_ID').isNull(), default_recipe).otherwise(col('pre.RECOMMENDATION_ITEM_ID')))

    df_out = df_updated.select(
        "HOUSEHOLD_ID",
        col("recipeId").alias("RECOMMENDATION_ITEM_ID"),
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        col("rank").alias("RANK"),
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        lit(METADATA).alias("METADATA")
    )
    print(f'Final hhids count: {df_out.count()}')
    return df_out
# df_gcp=getRuleBasedReceipe(df, '67195486c18d389101c93c2d', 'MY_NEEDS_BANNER_EMAIL_PROMO_CARD','e7a4f420-c861-4774-a0da-9c59581e8d70')
# display(df_gcp)

# COMMAND ----------

from typing import Dict, List, Any, Optional
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, broadcast
import json
from ModelMapping import ModelMapping

def apply_category_filter(df_input, filter_config,env, feature_project_id, bq_dataset) :
    
    category_paths = filter_config.get("l2l3l4", [])
    print(category_paths)
    
    query = f"""
   select distinct BASE_PRODUCT_NBR,DEPARTMENT_DIGITAL_NM,AISLE_DIGITAL_NM,SHELF_DIGITAL_NM,SHELF_DIGITAL_NBR from gcp-abs-udco-bqvw-prod-prj-01.udco_ds_mrch.D1_UPC_ROG_STORE where BASE_PRODUCT_NBR is not null and AISLE_DIGITAL_NM is not null
    """
    
    product_attributes = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)
    
    # Join recommendations with product attributes
    df_filtered = df_input.join(
        broadcast(product_attributes),
        df_input.RECOMMENDATION_ITEM_ID == product_attributes.BASE_PRODUCT_NBR,
        "inner"
    )
    
    # Filter based on category paths
    if category_paths:
        df_filtered = df_filtered.filter(col("SHELF_DIGITAL_NBR").isin(category_paths))
    display(df_filtered)
    
    return df_filtered



def apply_filter(df_input, filter_config, env, feature_project_id, bq_dataset):
    filter_id = filter_config.get("filterId")
    filter_functions = {
        "CATEGORY_FILTER": apply_category_filter
    }
    
    filter_function = filter_functions.get(filter_id)
    if not filter_function:
        raise ValueError(f"No filter function found for filter_id: {filter_id}")
    df_input=df_input.distinct()
    return filter_function(df_input, filter_config, env, feature_project_id, bq_dataset)

def extract_model_details(model_json) :
 
    filter_configs = []
    model_list = []
    
    def recursive_extract(json_obj):
        if isinstance(json_obj, dict):
            if "filterConfig" in json_obj:
                filter_configs.append(json_obj["filterConfig"])
            if "modelId" in json_obj:
                model_list.append(json_obj["modelId"])
            for value in json_obj.values():
                if isinstance(value, dict):
                    recursive_extract(value)
    
    recursive_extract(model_json)
    return filter_configs, model_list

def apply_filters_sequence(df_input, filter_configs, env, feature_project_id, bq_dataset):
  
    df_filtered = df_input
    
    for filter_config in filter_configs:
        df_filtered = apply_filter(
            df_filtered, 
            filter_config, 
            env, 
            feature_project_id, 
            bq_dataset
        )
        
    return df_filtered

def get_filtered_products(df_write,
                         model_configuration_id,
                         model_id,
                         model_json,
                         env,
                         feature_project_id,
                         bq_dataset,
                         experiment_id=None) :
  
    try:
        # Extract filter and model details
        filter_configs, model_list = extract_model_details(model_json)
        
        # Get base model that will provide recommendations
        base_model_id = next((m for m in model_list if m != model_id), None)
        if not base_model_id:
            raise ValueError(f"No base model found in configuration for {model_id}")
        
        print(f"Base model ID: {base_model_id}")
        print(f"Filter configurations: {filter_configs}")

        # Get base recommendations
        base_function_name = ModelMapping.get_function_name(base_model_id)
        if not base_function_name:
            raise ValueError(f"No function mapping found for model {base_model_id}")

        try:
            base_function = globals()[base_function_name]
            if 'experiment_id' in base_function.__code__.co_varnames:
                df_base_recommendations = base_function(
                    df_write, 
                    model_configuration_id, 
                    base_model_id, 
                    experiment_id
                )
            else:
                df_base_recommendations = base_function(
                    df_write, 
                    model_configuration_id, 
                    base_model_id
                )
        except Exception as e:
            raise Exception(f"Error executing base function {base_function_name}: {str(e)}")

        # Apply filters
        df_filtered = apply_filters_sequence(
            df_base_recommendations, 
            filter_configs,
            env,
            feature_project_id,
            bq_dataset
        )

        # Return filtered results
        return df_filtered.select(
            "HOUSEHOLD_ID",
            "RECOMMENDATION_ITEM_ID",
            "RECOMMENDATION_TYPE",
            "RANK",
            "EXPERIMENT_ID",
            "EXPERIMENT_VARIANT",
            lit(model_configuration_id).alias("MODEL_CONFIGURATION_ID"),
            lit(model_id).alias("MODEL_ID"),
            "RECOMMENDATION_CREATE_TS",
            "METADATA"
        )

    except Exception as e:
        print(f"Error in get_filtered_products: {str(e)}")
        raise

def getProductfilters(df_write,
                     model_configuration_id,
                     model_id,
                     experiment_id = None):
  
    # Get model configuration
    model_json = get_model_json(model_id) 
    
    # Process and return filtered recommendations
    return get_filtered_products(
        df_write=df_write,
        model_configuration_id=model_configuration_id,
        model_id=model_id,
        model_json=model_json,
        env=dbricks_env,
        feature_project_id=feature_project_id,
        bq_dataset=bq_dataset,
        experiment_id=experiment_id
    )

# COMMAND ----------

def getOfferGridService(df_write, model_configuration_id, model_id):
    read_upc_bpn_query = f'''
                            SELECT DISTINCT
    DIGITAL_PRODUCT_UPC.UPC_NBR,
    DIGITAL_PRODUCT_UPC.BASE_PRODUCT_NBR
  FROM
    gcp-abs-udco-bqvw-prod-prj-01.udco_ds_spex.DIGITAL_PRODUCT_UPC
  WHERE DIGITAL_PRODUCT_UPC.BASE_PRODUCT_NBR IN(
    SELECT DISTINCT
        cast(BPN_ID as INT64)
      FROM
        {feature_project_id}.{bq_dataset}.SM_V2_DS_SMART_LIST_RANKING
  ) '''
    if env == 'DEV':
        inputTable = 'gcp-abs-aamp-wmfs-prod-prj-01.aamp_ds_pz_wkg.offer_grid_bulk_hhids'
        upcbpnTable = 'gcp-abs-aamp-wmfs-prod-prj-01.aamp_ds_pz_wkg.drop_upc_bpn_map'
    else:
        inputTable = 'gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.offer_grid_bulk_hhids'
        upcbpnTable = 'gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.drop_upc_bpn_map'

    query = f'''
WITH FLATTENED_OFFERS AS (
  SELECT
      CAST(MBH.HOUSEHOLD_ID AS INT64) as HOUSEHOLD_ID,
      trim(F) AS EXTERNAL_OFFER_ID
  FROM {inputTable} AS MBH
  CROSS JOIN UNNEST(split(MBH.OFFERS_STR, ',')) AS F
  WHERE trim(F) != ''
),

VALID_OFFERS AS (
  SELECT DISTINCT 
    O.EXTERNAL_OFFER_ID,
    O.OMS_OFFER_ID
  FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_mrch.oms_offer O 
  INNER JOIN FLATTENED_OFFERS FO ON O.EXTERNAL_OFFER_ID = FO.EXTERNAL_OFFER_ID
  WHERE O.DW_CURRENT_VERSION_IND = TRUE 
    AND O.DW_LOGICAL_DELETE_IND = FALSE 
    AND O.DISPLAY_EFFECTIVE_END_DT > CURRENT_DATE() 
    AND O.DW_LAST_EFFECTIVE_DT > CURRENT_DATE() 
    AND O.DISPLAY_EFFECTIVE_START_DT > DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
),

FILTERED_SMART_LIST AS (
  SELECT DISTINCT
    A.HOMEPAGE_RANK_FINAL,
    CAST(A.HOUSEHOLD_ID AS INT64) as DSLR_HOUSEHOLD_ID,
    CAST(A.BPN_ID AS INT64) as BPN_ID
  FROM {feature_project_id}.{bq_dataset}.SM_V2_DS_SMART_LIST_RANKING A
  INNER JOIN FLATTENED_OFFERS FO ON CAST(A.HOUSEHOLD_ID AS INT64) = FO.HOUSEHOLD_ID
),

VALID_BPN_UPC_MAP AS (
  SELECT DISTINCT
    CAST(DPU.BASE_PRODUCT_NBR AS INT64) as BPN_ID,
    CAST(DPU.UPC_NBR AS INT64) as UPC_NBR
  FROM {upcbpnTable} DPU
  INNER JOIN FILTERED_SMART_LIST FSL ON CAST(DPU.BASE_PRODUCT_NBR AS INT64) = FSL.BPN_ID
  INNER JOIN gcp-abs-udco-bqvw-prod-prj-01.udco_ds_mrch.retail_order_group_upc_extended RO 
    ON CAST(DPU.UPC_NBR AS INT64) = CAST(RO.UPC_NBR AS INT64)
  WHERE SAFE_CAST(RO.CORPORATION_ID AS INTEGER) = 1 
    AND RO.ITEM_STATUS_CD <> 'D'
),

DISTINCT_OFFER_PRODUCTS AS (
  SELECT *
  FROM (
    SELECT 
      FSL.DSLR_HOUSEHOLD_ID,
      FO.EXTERNAL_OFFER_ID,
      VO.OMS_OFFER_ID,
      FSL.BPN_ID,
      FSL.HOMEPAGE_RANK_FINAL,
      ROW_NUMBER() OVER (
        PARTITION BY FSL.BPN_ID, FSL.HOMEPAGE_RANK_FINAL 
        ORDER BY FSL.DSLR_HOUSEHOLD_ID
      ) AS ROW_NUM
    FROM FILTERED_SMART_LIST FSL
    INNER JOIN FLATTENED_OFFERS FO 
      ON FSL.DSLR_HOUSEHOLD_ID = FO.HOUSEHOLD_ID
    INNER JOIN VALID_OFFERS VO 
      ON FO.EXTERNAL_OFFER_ID = VO.EXTERNAL_OFFER_ID
    INNER JOIN VALID_BPN_UPC_MAP VBUM
      ON FSL.BPN_ID = VBUM.BPN_ID
  ) ranked
  WHERE ROW_NUM = 1
),

OFFER_COUNT AS (
  SELECT
    DSLR_HOUSEHOLD_ID,
    CAST(COUNT(DISTINCT EXTERNAL_OFFER_ID) as BIGNUMERIC) AS OFFER_COUNT
  FROM DISTINCT_OFFER_PRODUCTS
  GROUP BY DSLR_HOUSEHOLD_ID
),

RANKED_PRODUCTS AS (
  SELECT
    DOP.DSLR_HOUSEHOLD_ID,
    DOP.EXTERNAL_OFFER_ID,
    DOP.BPN_ID,
    DOP.HOMEPAGE_RANK_FINAL,
    RANK() OVER (
      PARTITION BY DOP.DSLR_HOUSEHOLD_ID, DOP.EXTERNAL_OFFER_ID 
      ORDER BY COALESCE(DOP.HOMEPAGE_RANK_FINAL, 999999)
    ) AS BPN_RANK,
    DENSE_RANK() OVER (
      PARTITION BY DOP.DSLR_HOUSEHOLD_ID 
      ORDER BY DOP.EXTERNAL_OFFER_ID
    ) AS OFFER_RANK,
    OC.OFFER_COUNT
  FROM DISTINCT_OFFER_PRODUCTS DOP
  INNER JOIN OFFER_COUNT OC 
    ON DOP.DSLR_HOUSEHOLD_ID = OC.DSLR_HOUSEHOLD_ID
),

FINAL_SELECTION AS (
  SELECT
    RP.DSLR_HOUSEHOLD_ID,
    RP.EXTERNAL_OFFER_ID,
    RP.BPN_ID,
    RP.HOMEPAGE_RANK_FINAL,
    RP.BPN_RANK,
    RANK() OVER (
      PARTITION BY RP.DSLR_HOUSEHOLD_ID, RP.EXTERNAL_OFFER_ID, RP.BPN_RANK 
      ORDER BY RP.BPN_ID
    ) AS DUP_BPN_RANK,
    RP.OFFER_RANK,
    RP.OFFER_COUNT
  FROM RANKED_PRODUCTS RP
)

SELECT DISTINCT
  FS.DSLR_HOUSEHOLD_ID AS HOUSEHOLD_ID,
  FS.EXTERNAL_OFFER_ID,
  FS.BPN_ID,
  FS.HOMEPAGE_RANK_FINAL,
  FS.OFFER_RANK
FROM FINAL_SELECTION FS
WHERE (FS.OFFER_COUNT = NUMERIC '1' AND FS.BPN_RANK <= NUMERIC '6' AND FS.DUP_BPN_RANK = NUMERIC '1')
   OR (FS.OFFER_COUNT = NUMERIC '2' AND FS.BPN_RANK <= NUMERIC '3' AND FS.DUP_BPN_RANK = NUMERIC '1')
   OR (FS.OFFER_COUNT = NUMERIC '3' AND FS.BPN_RANK <= NUMERIC '2' AND FS.DUP_BPN_RANK = NUMERIC '1')
   OR (FS.OFFER_COUNT = NUMERIC '4' AND (
        (FS.OFFER_RANK IN(NUMERIC '1', NUMERIC '2') AND FS.BPN_RANK <= NUMERIC '2' AND FS.DUP_BPN_RANK = NUMERIC '1')
        OR (FS.OFFER_RANK IN(NUMERIC '3', NUMERIC '4') AND FS.BPN_RANK = NUMERIC '1' AND FS.DUP_BPN_RANK = NUMERIC '1')
      ))
   OR (FS.OFFER_COUNT = NUMERIC '5' AND (
        (FS.OFFER_RANK = NUMERIC '1' AND FS.BPN_RANK <= NUMERIC '2' AND FS.DUP_BPN_RANK = NUMERIC '1')
        OR (FS.OFFER_RANK IN(NUMERIC '2', NUMERIC '3', NUMERIC '4', NUMERIC '5') AND FS.BPN_RANK = NUMERIC '1' AND FS.DUP_BPN_RANK = NUMERIC '1')
      ))
   OR (FS.OFFER_COUNT = NUMERIC '6' AND FS.BPN_RANK = NUMERIC '1' AND FS.DUP_BPN_RANK = NUMERIC '1')
ORDER BY
  HOUSEHOLD_ID NULLS LAST,
  OFFER_RANK NULLS LAST
    '''
    print(query)

    RECOMMENDATION_TYPE = "PRODUCT"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    METADATA = ''

    df_edm = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)
    df_edm = df_edm.withColumn("HOUSEHOLD_ID", col("HOUSEHOLD_ID"))
    df_edm = df_edm.withColumn("RECOMMENDATION_ITEM_ID", col("BPN_ID"))
    df_edm = df_edm.withColumn("RECOMMENDATION_TYPE", lit("PRODUCT"))
    df_edm = df_edm.withColumn("RANK", col("homepage_rank_final"))
    df_edm = df_edm.withColumn("EXPERIMENT_ID", lit(EXPERIMENT_ID))
    df_edm = df_edm.withColumn("EXPERIMENT_VARIANT", lit(EXPERIMENT_VARIANT))
    df_edm = df_edm.withColumn("MODEL_CONFIGURATION_ID", lit(model_configuration_id))
    df_edm = df_edm.withColumn("MODEL_ID", lit(model_id))
    df_edm = df_edm.withColumn("RECOMMENDATION_CREATE_TS", date_format(current_timestamp(), "MM/dd/yyyy HH:mm"))
    df_edm = df_edm.withColumn("METADATA", lit(METADATA))

    df_out = df_edm.select("HOUSEHOLD_ID",
                           "RECOMMENDATION_ITEM_ID",
                           "RECOMMENDATION_TYPE",
                           "RANK",
                           "EXPERIMENT_ID",
                           "EXPERIMENT_VARIANT",
                           "MODEL_CONFIGURATION_ID",
                           "MODEL_ID",
                           "RECOMMENDATION_CREATE_TS",
                           "METADATA"
    )
    # display(df_out)
    return df_out
# df_gcp=getOfferGridService(df, '66bd2bbb78100d5bcb923701', 'OFFER_GRID_SERVICE')
# df_gcp.display()


# COMMAND ----------


import json
import traceback
import time
from functools import lru_cache
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, BooleanType
from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, to_json, when, col, coalesce, broadcast
from pyspark.sql.types import StringType, LongType



def parse_customer_attributes_list(attributes_str):
    if not attributes_str:
        return []
    
    try:
        json_str = f"[{attributes_str}]"
        print(json_str)
        try:
            attributes = json.loads(json_str)
            return [attr.strip() for attr in attributes if attr]
        except json.JSONDecodeError:
            import re
            matches = re.findall(r'"([^"]*)"', attributes_str)
            if matches:
                return [match.strip() for match in matches if match.strip()]
    
    except Exception as e:
        print(f"Error parsing customer attributes: {e}")
        print(f"Raw string: {attributes_str}")
    
    return [attr.strip().strip('"') for attr in attributes_str.split(',') if attr.strip()]

def get_required_customer_attributes_dynamic(context_properties_set):
    try:
        query = "dbo.BulkSegmentsData"
        df_segments = spark.read.jdbc(url=url, table=query, properties=properties)
        
        customer_attributes_row = df_segments.filter(
            df_segments.segment_name == "customer_attributes_list"
        ).select("segment_list").collect()
        
        if customer_attributes_row and customer_attributes_row[0][0]:
            raw_attributes_str = customer_attributes_row[0][0]
            parsed_attributes = parse_customer_attributes_list(raw_attributes_str)
            
            if parsed_attributes:
                all_customer_attributes = set(parsed_attributes)
                required_attrs = list(all_customer_attributes.intersection(context_properties_set))
                print(f"Found {len(required_attrs)} required attributes out of {len(all_customer_attributes)} available attributes")
                print(f'Dynamic cust attrs: {required_attrs}')
                return required_attrs
            else:
                print(f"Warning: Parsed empty attributes list from: {raw_attributes_str}")
        
    except Exception as e:
        print(f"Error processing customer attributes: {e}")
    
    default_attributes = [
        "DUG_IND", "PRIM_STORE_ID", "PRIM_DIVISION_ID", "EMPLOYEE_IND",
        "PHARMACY_PURCHASE_IND", "EXPLICIT_DIET_PREFERENCE_CD", 
        "FRESHPASS_SUBSCRIPTION_STATUS_CD", "PET_PRODUCT_PURCHASE_IND",
        "ECOM_PURCHASE_IND", "MARKETABLE_EMAIL_IND", "DIGITAL_LOGIN_P12M",
        "DELIVERY_IND", "MEAT_PURCHASE_IND", "ENG_MODE_LIFETIME"
    ]
    all_customer_attributes = set(default_attributes)
    print(f"Using {len(all_customer_attributes)} default attributes")
    return list(all_customer_attributes.intersection(context_properties_set))

def get_matched_banners_df_dynamic(df_user_details, model_json, rule_fields):
   
    obj_li = []

    matcher = spark._jvm.com.albertsons.aapn.experiment.sdk.core.eligibility.matchers.MatcherMapper()
    exp_service_class = spark._jvm.com.albertsons.aapn.experiment.sdk.core.services.DefaultExperimentCoreService(None, matcher)
    obj_mapper = spark._jvm.com.fasterxml.jackson.databind.ObjectMapper()
    eligibility_matcher_class = spark._jvm.com.albertsons.aapn.experiment.sdk.core.dto.eligibility.EligibilityMatcher()
    default_banner = None
    
    for obj in model_json["data"]["dataList"]:
        matcher_obj = None
        if "audience" in obj and "condition" in obj["audience"]:
            matcher_json = json.dumps(obj["audience"]["condition"])
            matcher_obj = obj_mapper.readValue(matcher_json, eligibility_matcher_class.getClass())
        else:
            try:
                default_banner = obj["staticRecommendation"]["bannerRecommendation"]["recommendations"][0]["bannerId"]
            except (KeyError, IndexError) as e:
                print(f"Warning: Unable to extract default banner: {e}")
                default_banner = "default-banner"
                
        banner_dict = {}
        try:
            banner_dict["banner_id"] = obj["staticRecommendation"]["bannerRecommendation"]["recommendations"][0]["bannerId"]
        except (KeyError, IndexError) as e:
            print(f"Warning: Unable to extract banner ID: {e}")
            continue
            
        banner_dict["matcher"] = matcher_obj
        obj_li.append(banner_dict)

    def check_eligibility(dataList, context):
        for data in dataList:
                if exp_service_class.checkCondition(data["matcher"], context).getResult():
                    return data["banner_id"]
                
        return None
    
    precomputed_data = []
    count = 0

    print(f'Unique rules: {len(df_user_details.select(rule_fields).distinct().collect())}')

    for distinct_df in df_user_details.select(rule_fields).distinct().collect():
        count = count + 1
        context_map = {}
        for rule in rule_fields:
            context_map[rule] = distinct_df[rule] if rule in distinct_df else None
        banner_id = check_eligibility(obj_li, context_map)
        context_map["RECOMMENDATION_ITEM_ID"] = banner_id
        precomputed_data.append(context_map)

    print(f'BAU banner: {default_banner}')
    print(f'No of unique rules: {count}')

    schema_fields = [StructField('RECOMMENDATION_ITEM_ID', StringType(), True)]
    
    field_types = {}
    for field in df_user_details.schema.fields:
        field_types[field.name] = field.dataType
    
    for field in rule_fields:
        if field in field_types:
            schema_fields.append(StructField(field, field_types[field], True))
        else:
            schema_fields.append(StructField(field, StringType(), True))
    
    dynamic_schema = StructType(schema_fields)
    
    try:
        precomputed_df = spark.createDataFrame(precomputed_data, schema=dynamic_schema)
        return precomputed_df, default_banner
    except Exception as e:
        print(f"Error creating precomputed DataFrame: {e}")
        print(f"Schema fields: {[field.name for field in schema_fields]}")
        print(f"First data item: {json.dumps(precomputed_data[0] if precomputed_data else {})}")
        
        simple_schema = StructType([
            StructField('RECOMMENDATION_ITEM_ID', StringType(), True),
            *[StructField(field, StringType(), True) for field in rule_fields]
        ])
        
        safe_data = []
        for item in precomputed_data:
            safe_item = {k: str(v) if v is not None else None for k, v in item.items()}
            safe_data.append(safe_item)
            
        return spark.createDataFrame(safe_data, schema=simple_schema), default_banner

def get_customer_attributes_join_dynamic(required_attributes):
    if not required_attributes:
        return ""
    
    eng_mode_required = "ENG_MODE_LIFETIME" in required_attributes
    freshpass_required = "FRESHPASS_SUBSCRIPTION_STATUS_CD" in required_attributes
    
    regular_attributes = [attr for attr in required_attributes 
                         if attr not in ["ENG_MODE_LIFETIME", "FRESHPASS_SUBSCRIPTION_STATUS_CD"]]
    
    regular_attrs_select = ", ".join([f"attrs.{attr}" for attr in regular_attributes]) if regular_attributes else ""
    
    eng_mode_select = '''
        CASE
            -- "IN-STORE" only if ALL UUIDs are either 'IN-STORE' or NULL
            WHEN LOGICAL_AND(
                IFNULL(eng_modes.eng_mode = 'IN-STORE' OR eng_modes.eng_mode IS NULL, TRUE)
            ) THEN 'IN-STORE'
            
            -- "ECOM" only if ALL UUIDs are either 'ECOM' or NULL
            WHEN LOGICAL_AND(
                IFNULL(eng_modes.eng_mode = 'ONLINE' OR eng_modes.eng_mode IS NULL, TRUE)
            ) THEN 'ONLINE'
            
            -- "OMNI" if ANY UUID is 'OMNI'
            WHEN LOGICAL_OR(IFNULL(eng_modes.eng_mode = 'OMNI', FALSE)) THEN 'OMNI'
            
            -- "OMNI" if BOTH 'IN-STORE' and 'ONLINE' exist across UUIDs
            WHEN LOGICAL_OR(IFNULL(eng_modes.eng_mode = 'IN-STORE', FALSE)) 
                 AND LOGICAL_OR(IFNULL(eng_modes.eng_mode = 'ONLINE', FALSE)) THEN 'OMNI'
            
            ELSE 'UNKNOWN'
        END AS ENG_MODE_LIFETIME''' if eng_mode_required else ""
    
    freshpass_select = '''
        CASE
            -- "ACTIVE" if ANY UUID has values in ('ACTIVE','TRIAL','PENDING_CANCELLATION')
            WHEN LOGICAL_OR(
                IFNULL(freshpass.status IN ('ACTIVE','TRIAL','PENDING_CANCELLATION'), FALSE)
            ) THEN 'ACTIVE'
            
            ELSE NULL
        END AS FRESHPASS_SUBSCRIPTION_STATUS_CD''' if freshpass_required else ""
    
    selections = []
    if regular_attrs_select:
        selections.append(regular_attrs_select)
    if eng_mode_select:
        selections.append(eng_mode_select)
    if freshpass_select:
        selections.append(freshpass_select)
    
    all_selections = ",\n            ".join(selections)
    
    if (eng_mode_required or freshpass_required) and regular_attributes:
        return f"""
        LEFT JOIN (
            SELECT 
                base.household_id,
                {all_selections},
                ROW_NUMBER() OVER (PARTITION BY base.household_id) as rn
            FROM (select * from gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_PROFILE where household_id in (select distinct CAST(household_id AS INT64) from {table_name_bq})) base
            LEFT JOIN (
                SELECT 
                    household_id,
                    ENG_MODE_LIFETIME as eng_mode
                FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_PROFILE
                WHERE ENG_MODE_LIFETIME IS NOT NULL
            ) eng_modes
            ON base.household_id = eng_modes.household_id
            LEFT JOIN (
                SELECT 
                    household_id,
                    FRESHPASS_SUBSCRIPTION_STATUS_CD as status
                FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_PROFILE
                WHERE FRESHPASS_SUBSCRIPTION_STATUS_CD IS NOT NULL
            ) freshpass
            ON base.household_id = freshpass.household_id
            LEFT JOIN (
                SELECT 
                    profile.household_id,
                    {", ".join([f"profile.{attr} as {attr}" for attr in regular_attributes])}
                FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_PROFILE profile
                QUALIFY ROW_NUMBER() OVER (PARTITION BY profile.household_id) = 1
            ) attrs
            ON base.household_id = attrs.household_id
            GROUP BY 
                base.household_id,
                {", ".join([f"attrs.{attr}" for attr in regular_attributes]) if regular_attributes else "1"}
        ) cust
        ON CAST(hhids.household_id AS INT64) = CAST(cust.household_id AS INT64)
        AND cust.rn = 1
        """
    elif eng_mode_required or freshpass_required:
        return f"""
        LEFT JOIN (
            SELECT 
                base.household_id,
                {all_selections},
                ROW_NUMBER() OVER (PARTITION BY base.household_id) as rn
            FROM (select * from gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_PROFILE where household_id in (select distinct CAST(household_id AS INT64) from {table_name_bq})) base
            LEFT JOIN (
                SELECT 
                    household_id,
                    ENG_MODE_LIFETIME as eng_mode
                FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_PROFILE
                WHERE ENG_MODE_LIFETIME IS NOT NULL
            ) eng_modes
            ON base.household_id = eng_modes.household_id
            LEFT JOIN (
                SELECT 
                    household_id,
                    FRESHPASS_SUBSCRIPTION_STATUS_CD as status
                FROM gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_PROFILE
                WHERE FRESHPASS_SUBSCRIPTION_STATUS_CD IS NOT NULL
            ) freshpass
            ON base.household_id = freshpass.household_id
            GROUP BY 
                base.household_id
        ) cust
        ON CAST(hhids.household_id AS INT64) = CAST(cust.household_id AS INT64)
        AND cust.rn = 1
        """
    else:
        if not regular_attributes:
            return ""  
            
        return f"""
        LEFT JOIN (
            SELECT 
                profile.household_id,
                {regular_attrs_select},
                ROW_NUMBER() OVER (PARTITION BY profile.household_id) as rn
            FROM (select * from gcp-abs-udco-bqvw-prod-prj-01.udco_ds_cust.C360_CUSTOMER_PROFILE where household_id in (select distinct CAST(household_id AS INT64) from {table_name_bq})) profile
            WHERE ROW_NUMBER() OVER (PARTITION BY profile.household_id) = 1
        ) cust
        ON CAST(hhids.household_id AS INT64) = CAST(cust.household_id AS INT64)
        AND cust.rn = 1
        """

def get_unified_query_dynamic(table_name_bq, model_configuration_id, experiment_id, result_set, required_attributes):

    print(f'Required customer attributes: {required_attributes}')
  
    # Build all segment strings at once
    facts_segment_str = facts_segments_list
    persona_str = persona_list
    buyer_segment_str = buyer_segment_list
    my_needs_segment_str = my_needs_segment_list
    egl_segment_str = egl_segments_list
    customer_attr_str = customer_attributes_list 

    # Dynamically build column selection based on result_set
    select_columns = ["CAST(hhids.household_id AS INT64) as household_id"]
    
    if "leap" in result_set:
        select_columns.extend([
            "CAST(L.STORE_ID AS INT64) as storeId",
            "EM.BANNER as banner",
            "CAST(L.DIVISION_ID AS INT64) as divisionNumber"
        ])
    
    if "c360" in result_set:
        select_columns.extend([
            "seg.persona",
            "seg.buyerSegment",
            "seg.variant"
        ])
    
    if "my_needs" in result_set:
        select_columns.append("myneeds.myNeedsSegment")
    
    if "egl" in result_set:
        select_columns.append("egl.eglSegment")
    
    if "facts" in result_set:
        select_columns.append("facts.factsSegment")
    
    if "customer_attributes" in result_set and required_attributes:
        select_columns.extend([f"cust.{attr}" for attr in required_attributes])
    
    select_columns.append("hhids.model_configuration_id")
    
    query = f"""
        SELECT {', '.join(select_columns)}
        FROM 
            {table_name_bq} hhids
        {get_leap_join() if "leap" in result_set else ""}
        {get_c360_join(experiment_id, persona_str, buyer_segment_str) if "c360" in result_set else ""}
        {get_myneeds_join(experiment_id, my_needs_segment_str) if "my_needs" in result_set else ""}
        {get_egl_join(experiment_id, egl_segment_str) if "egl" in result_set else ""}
        {get_facts_join(facts_segment_str) if "facts" in result_set else ""}
        {get_customer_attributes_join_dynamic(required_attributes) if "customer_attributes" in result_set else ""}
        WHERE hhids.model_configuration_id = '{model_configuration_id}';
    """
    return query




def getRuleBasedBannersdynamic(df_write, model_configuration_id, model_id, experiment_id):
    
    model_json = get_model_json(model_id)

    context_properties_list = []

    correct_audience(model_json, context_properties_list)
    context_properties_set = set(context_properties_list)    
    print(f'Context properties set: {context_properties_set}')

    final_rule_fields_list = list(context_properties_set)
    print(f'Rule fields list: {final_rule_fields_list}')

    required_customer_attributes = get_required_customer_attributes_dynamic(context_properties_set)
    print(f'Required customer attributes: {required_customer_attributes}')
    
    context_to_table_map = {
        "leap": ["storeId", "banner", "divisionNumber"],
        "c360": ["buyerSegment", "persona", "variant"],
        "my_needs": ["myNeedsSegment"],
        "egl": ["eglSegment"],
        "facts": ["factsSegment"],
        "customer_attributes": required_customer_attributes
    }
    
    result_set = set()
    for key, values in context_to_table_map.items():
        if any(value in context_properties_set for value in values) and values:  # Only add if values exist
            result_set.add(key)
            
    print(f"Required data sources: {result_set}")
    
    # Get query using unified function
    query = get_unified_query_dynamic(table_name_bq, model_configuration_id, experiment_id, 
                            result_set, required_customer_attributes)
    
    print(query)
    df_user_details = read_from_bq(query=query, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)

    print(f'Input hhids count: {df_user_details.count()}')

    # Now using our dynamic schema approach
    precomputed_df, default_banner = get_matched_banners_df_dynamic(df_user_details, model_json, final_rule_fields_list)
    display(precomputed_df)
    print(default_banner)
    
    fill_values = {
        StringType(): 'default',
        LongType(): 0
    }

    for dtype, value in fill_values.items():
        df_user_details = df_user_details.fillna(value, subset=[col for col in df_user_details.columns if df_user_details.schema[col].dataType == dtype])
        precomputed_df = precomputed_df.fillna(value, subset=[col for col in precomputed_df.columns if precomputed_df.schema[col].dataType == dtype])

    df_updated = df_user_details.join(
        broadcast(precomputed_df),
        on=final_rule_fields_list,
        how="left")

    df_updated = df_updated.withColumn("timestamp", current_timestamp())
    df_updated = df_updated.withColumn("RECOMMENDATION_CREATE_TS", date_format("timestamp", "MM/dd/yyyy HH:mm"))
    
    RANK = "1"
    RECOMMENDATION_TYPE = "BANNER"
    EXPERIMENT_ID = ''
    EXPERIMENT_VARIANT = ''
    MODEL_CONFIGURATION_ID = model_configuration_id
    MODEL_ID = model_id
    METADATA = ''

    df_updated = df_updated.withColumn('RECOMMENDATION_ITEM_ID', when(col('RECOMMENDATION_ITEM_ID').isNull(), default_banner).otherwise(col('RECOMMENDATION_ITEM_ID')))

    df_out = df_updated.select(
        "HOUSEHOLD_ID",
        "RECOMMENDATION_ITEM_ID",
        lit(RECOMMENDATION_TYPE).alias("RECOMMENDATION_TYPE"),
        lit(RANK).alias("RANK"),
        lit(EXPERIMENT_ID).alias("EXPERIMENT_ID"),
        lit(EXPERIMENT_VARIANT).alias("EXPERIMENT_VARIANT"),
        lit(MODEL_CONFIGURATION_ID).alias("MODEL_CONFIGURATION_ID"),
        lit(MODEL_ID).alias("MODEL_ID"),
        "RECOMMENDATION_CREATE_TS",
        lit(METADATA).alias("METADATA")
    )
    print(f'Final hhids count: {df_out.count()}')
    return df_out

# COMMAND ----------

output_schema = StructType([
    StructField("HOUSEHOLD_ID", StringType(), True),
    StructField("RECOMMENDATION_ITEM_ID", StringType(), True),
    StructField("RECOMMENDATION_TYPE", StringType(), True),
    StructField("RANK", StringType(), True),
    StructField("EXPERIMENT_ID", StringType(), True),
    StructField("EXPERIMENT_VARIANT", StringType(), True),
    StructField("MODEL_CONFIGURATION_ID", StringType(), True),
    StructField("MODEL_ID", StringType(), True),
    StructField("RECOMMENDATION_CREATE_TS", StringType(), True),
    StructField("METADATA", StringType(), False)
])

# COMMAND ----------

def getRecommendations(df_write, model_configuration_id, model_id, function_name, experiment_id=None):
    # display(df)
    # print(model_configuration_id, model_id, function_name)
    # df_write = df[df.model_configuration_id == model_configuration_id]
    # display(df_write)
    msgFunction = globals()[function_name]
    if 'experiment_id' in msgFunction.__code__.co_varnames:
        df_recom = msgFunction(df_write, model_configuration_id, model_id, experiment_id)
    else:
        df_recom = msgFunction(df_write, model_configuration_id, model_id)
    # display(df_recom)
    return df_recom


# COMMAND ----------

from pyspark.sql.functions import col, countDistinct, lit, current_timestamp, date_format
from email_notifier import EmailNotification

def process_config(df, df_config):
    config_rows = df_config.collect()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in config_rows:
        app_id, model_configuration_id, model_id, function_name, experiment_id = row["app_id"], row["model_configuration_id"], row[
            "model_id"], row["function_name"], row["experiment_id"]
        print(app_id, model_configuration_id, model_id, function_name, experiment_id)
        df_normalized = df.toDF(*[c.lower() for c in df.columns])
        column_name = "MODEL_CONFIGURATION_ID".lower()
        #df_write = df[df.model_configuration_id == model_configuration_id]
        df_write = df_normalized[df_normalized[column_name] == model_configuration_id]
        #df_write.show(10)
        if df_write.head(1):
            # Step 4: Count Unique HOUSEHOLD_IDs per MODEL_CONFIGURATION_ID
            df_household_counts = df_write.groupBy("MODEL_CONFIGURATION_ID").agg(countDistinct("HOUSEHOLD_ID").alias("unique_households"))
            df_household_counts.display()
            print(f"Processing: {app_id}, {model_configuration_id}, {model_id}, {function_name}, {experiment_id}")
            # Step 5: Execute Function Dynamically
            try:
                df_recom = getRecommendations(df_write, model_configuration_id, model_id, function_name, experiment_id)
                temp_model_path = temp_path_model + model_id
                df_result = spark.createDataFrame(df_recom.rdd, output_schema)

                print(f"Saving results to: {temp_model_path}")

                df_result.write.format("csv").mode("overwrite").option("quote", "") \
                .option("header", "true").save(temp_model_path)

                # Step 6: Compute Processed vs. Unprocessed Household IDs
                df_processed = df_result.select("HOUSEHOLD_ID").distinct()
                processed_count = df_processed.count()

                initial_household_count = df_household_counts.agg({"unique_households": "sum"}).collect()[0][0]
                unprocessed_count = initial_household_count - processed_count

                end_time = datetime.now()
                processing_time = (end_time - datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")).total_seconds()
                
                # success_msg = f"Function {function_name} executed successfully for Model ID {model_id} at {current_time}.\nConfiguration inputs passed : {app_id}, {model_configuration_id}, {model_id}, {function_name}, {experiment_id} \nNumber of input HOUSEHOLD_IDs = {initial_household_count} : processed_count {processed_count}, unprocessed_count = {unprocessed_count} ,Processed Output File {temp_model_path} \nProcessing time: {processing_time} seconds."
                success_msg = (
                    f"Function {function_name} executed successfully for Model ID {model_id} at {current_time}.\n"
                    f"Configuration inputs passed:\n"
                    f"  - App ID: {app_id}\n"
                    f"  - Model Configuration ID: {model_configuration_id}\n"
                    f"  - Model ID: {model_id}\n"
                    f"  - Function Name: {function_name}\n"
                    
                    f"Number of input HOUSEHOLD_IDs: {initial_household_count}\n"
                    f"Processed count: {processed_count}\n"
                    f"Unprocessed count: {unprocessed_count}\n"
                    f"Processed Output File: {temp_model_path}\n"
                    f"Processing time: {processing_time} seconds."
                )
                print(success_msg)
                step = f"MSG_BULK_SERVICE - Success Notification for {env}:{InputFileName} - {model_id}"
                mailer = EmailNotification()
                mailer.send_success_notification(step, success_msg)
                

            except Exception as e:
                # error_msg = f"Configuration inputs passed : {app_id}, {model_configuration_id}, {model_id}, {function_name}, {experiment_id} \nError executing function {function_name} for Model ID {model_id}. \nError message: {str(e)}
                failed_msg = (
                    f"Configuration inputs passed:\n"
                    f"  - App ID: {app_id}\n"
                    f"  - Model Configuration ID: {model_configuration_id}\n"
                    f"  - Model ID: {model_id}\n"
                    f"  - Function Name: {function_name}\n"
                    f"Error executing function {function_name} for Model ID {model_id}.\n"
                    f"Error message: {str(e)}"
                )
                step = f"MSG_BULK_SERVICE - Error Alert! for {env}:{InputFileName} - {model_id}"
                mailer = EmailNotification()
                mailer.send_error_notification(step, failed_msg)
            yield temp_model_path

# COMMAND ----------

try:
    files_list = []
    print(files_list)
    
    for file_path in process_config(df, df_config):
        files_list.append(file_path)
    
    print(files_list)
    # Cleanup BigQuery table
    bq_cleanup_query = f"DROP TABLE IF EXISTS `{table_name_bq}`"
    execute_with_client(parent_project, bq_cleanup_query, credentials=credentials, query_tag=query_tag_bq)
    print(f"BigQuery table {table_name_bq} has been dropped.")
except Exception as e:
    print(f'error:{e}')
    # Cleanup BigQuery table
    bq_cleanup_query = f"DROP TABLE IF EXISTS `{table_name_bq}`"
    execute_with_client(parent_project, bq_cleanup_query, credentials=credentials, query_tag=query_tag_bq)
    print(f"BigQuery table {table_name_bq} has been dropped.")



# COMMAND ----------


df_result = spark.read.format("csv").option("header", "true").schema(output_schema).load(files_list)

# COMMAND ----------

dbutils.fs.ls(temp_path_model)

# COMMAND ----------


temp_path

# COMMAND ----------

df_result.coalesce(1).write \
    .format("csv") \
    .mode("overwrite") \
    .option("quote", "") \
    .option("header", "true") \
    .save(temp_path)

# COMMAND ----------

dbutils.fs.ls(temp_path)

# COMMAND ----------

files = dbutils.fs.ls(temp_path)
int_file_info = next(file for file in files if ".csv" in file.name)
int_file = int_file_info.path
print(int_file)

# COMMAND ----------

local_full_path = local_tmp_path + dest_file_name

# COMMAND ----------

output_pgp_file_path = local_tmp_path + encrypted_file_name

# COMMAND ----------

dbutils.fs.ls(local_tmp_path)


# COMMAND ----------

# encrypt the file contents:

def encrypt_csv_file(input_file_name, out_file_name):
    import gnupg
    from io import BytesIO
    gpg = gnupg.GPG()
    # read from Azure Vault service
    public_key_data = sfmc_pwd
    import_result = gpg.import_keys(public_key_data)
    print(import_result)

    if import_result.count == 0:
        raise ValueError("Public Key Failed...")

    csv_file_path = "/dbfs" + input_file_name
    print(csv_file_path)

    output_pgp_file_path_enc = "/dbfs" + out_file_name
    print(output_pgp_file_path_enc)

    with open(csv_file_path, 'rb') as csv_file:
        encrypted_data = gpg.encrypt_file(csv_file, recipients=['pankaj.verma@albertsons.com'], always_trust=True,
                                          output=output_pgp_file_path_enc)

    if not encrypted_data.ok:
        raise ValueError("Encryption Failed:" + encrypted_data.status)

    print('File encrypted successfully')

    return output_pgp_file_path_enc


# COMMAND ----------

def transfer_msg_b2b(local_file_name, dest_file_name):
    import paramiko
    from paramiko.ssh_exception import SSHException

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)

    try:

        ssh_client.connect(
            hostname=sftp_host_name,
            username=b2b_user,
            password=b2b_pwd,
            port=22
        )

        sftp = ssh_client.open_sftp()

        local_file_path = local_file_name
        remote_file_path = dest_file_name
        print(f'stp transfer: {local_file_path} & {remote_file_path}')

        sftp.put(local_file_path, remote_file_path)

        print('File pushed from MSG to B2B Successfully')
        sftp.close()

    except paramiko.ssh_exception.SSHException as ex:
        if "Error reading SSH protocol banner" in str(ex):
            print("Failed to read SSH protocol banner.")
        else:
            print(f"SSH connection failed:{str(ex)}")
    finally:
        ssh_client.close()


# COMMAND ----------

print(encrypted_file_name)

# COMMAND ----------

def load_output_table():
    if dbricks_env == 'DEV':
        output_table_bq='gcp-abs-aamp-wmfs-prod-prj-01.aamp_ds_pz_wkg.msg_bulk_output_recommendations'
    else:
        output_table_bq='gcp-abs-aamp-dpfs-prod-prj-01.aamp_ds_pz_prod.msg_bulk_output_recommendations'
    df1 = spark.read.csv(local_full_path, header=True, schema=output_schema)
    df1 = df1.withColumn("InputFileName", lit(InputFileName))

    df1 = df1.select(
        col("InputFileName"),
        col("HOUSEHOLD_ID"),
        col("RECOMMENDATION_ITEM_ID"),
        col("RECOMMENDATION_TYPE"),
        col("RANK"),
        col("EXPERIMENT_ID"),
        col("EXPERIMENT_VARIANT"),
        col("MODEL_CONFIGURATION_ID"),
        col("MODEL_ID"),
        to_timestamp(current_timestamp(), "MM/dd/yyyy HH:mm").alias("RECOMMENDATION_CREATE_TS"),
        col("METADATA")
    )

    record_count = df1.count()
    print(f"The DataFrame contains {record_count} records.")
    display(df1)

    write_to_bq(df1, output_table_bq, "append", gcs_bucket,query_tag=query_tag_bq)

# COMMAND ----------

if dbutils.fs.cp(int_file, local_full_path):
    print('file is copied to local file system')
    print(int_file)
    print(local_full_path)
    print(local_tmp_path)
    output_pgp_file_path_result = encrypt_csv_file(local_full_path, output_pgp_file_path)
    print(output_pgp_file_path_result)
    transfer_msg_b2b(output_pgp_file_path_result, encrypted_file_name)
    load_output_table()
    # dbutils.fs.mv(input_file, archive_path)
    # dbutils.fs.rm(local_full_path)
    # dbutils.fs.rm(int_file)
    dir_name =InputFileName
    archive_dir = os.path.join(archive_path, dir_name)
    print(archive_dir)
    def dir_exists(path):   
        try:
            dbutils.fs.ls(path)
            return True
        except:
            return False
    # Create the directory if it doesn't exist
    if dir_exists(archive_dir):
        dbutils.fs.rm(archive_dir, recurse=True)
    else:
        dbutils.fs.mkdirs(archive_dir)
    
    dbutils.fs.mv(input_file, archive_dir, recurse=True)
    dbutils.fs.rm(local_full_path)
    dbutils.fs.rm(int_file)

# COMMAND ----------

if dbricks_env == 'DEV':
    access_token = dbutils.secrets.get(scope="aamp-dev-wu-kv-02-scp", key="aampmlpdevst01-access-token")
    spark.conf.set("fs.azure.account.key.aampmlpdevst01.blob.core.windows.net",access_token)
else:
    access_token = dbutils.secrets.get(scope="aamp-prod-wu-kv-02-scp", key="aampmlpprodst01-access-token")
    spark.conf.set("fs.azure.account.key.aampmlpprodst01.blob.core.windows.net",access_token)	

input_file= f"wasbs://msg@{storage_account}.blob.core.windows.net/emcn/input/{InputFileName}"
outfile= f"wasbs://msg@{storage_account}.blob.core.windows.net/emcn/archive/"

dbutils.fs.mv(input_file, outfile)
